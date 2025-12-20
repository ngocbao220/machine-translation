import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import sacrebleu
import math
import os

# --- TỐI ƯU 1: Bật TF32 ---
torch.set_float32_matmul_precision('high')

class Trainer:
    def __init__(self, model, vocab, optimizer, criterion, scheduler, device, config):
        self.model = model
        self.vocab = vocab 
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.model_type = config.get('model_type', 'base') # Lấy từ config để phân nhánh logic
        
        # Setup Multi-GPU
        if self.config['gpu_mode'] and torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        # --- TỐI ƯU 2: Autocast ---
        self.mixed_precision_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.use_amp = config['use_amp']
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.mixed_precision_dtype == torch.float16 and self.use_amp))
        
        print(f"⚡ AMP Enabled: {self.use_amp} | Type: {self.mixed_precision_dtype} | Model: {self.model_type}")

    def _get_model_core(self):
        """Hàm hỗ trợ lấy model gốc để truy cập các thuộc tính đặc thù"""
        model_core = self.model
        if isinstance(model_core, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_core = model_core.module
        if hasattr(model_core, "_orig_mod"): # Nếu dùng torch.compile
             model_core = model_core._orig_mod
        return model_core

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        model_core = self._get_model_core()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        
        for batch in pbar:
            src, tgt = batch
            src, tgt = src.to(self.device, non_blocking=True), tgt.to(self.device, non_blocking=True)
            
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:] 

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=self.mixed_precision_dtype, enabled=self.use_amp):
                # forward() của cả 2 wrapper phải trả về logit dự đoán
                output = self.model.module(src, tgt_input)
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            
            if self.use_amp and self.mixed_precision_dtype == torch.float16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6e}"})
            
            if self.config['use_wandb']:
                wandb.log({"train_loss_step": loss.item(), "lr": current_lr})
                
        return total_loss / len(dataloader)

    def greedy_decode(self, src, max_len=50):
        """Hàm giải mã linh hoạt cho cả Base và Re-Transformer"""
        model_core = self._get_model_core()
        model_core.eval()
        
        bos_idx = self.config['bos_idx']
        eos_idx = self.config['eos_idx']
        src_tensor = src.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=self.mixed_precision_dtype, enabled=self.use_amp):
                # 1. ENCODING PHASE
                if self.model_type == 're':
                    src_mask = (src_tensor != model_core.pad_idx).unsqueeze(-2)
                    # Re-Transformer tích hợp PE bên trong encoder layer
                    memory = model_core.encoder(model_core.src_embed(src_tensor) * math.sqrt(model_core.d_model), src_mask)
                else:
                    # Logic cho Base Transformer sử dụng nn.Transformer
                    src_mask = (src_tensor != model_core.pad_idx).unsqueeze(1).unsqueeze(2)
                    memory = model_core.transformer.encoder(
                        model_core.positional_encoding(model_core.src_embedding(src_tensor) * math.sqrt(model_core.d_model)), 
                        mask=src_mask
                    )
                
                # 2. DECODING PHASE
                ys = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(self.device)
                
                for i in range(max_len - 1):
                    if self.model_type == 're':
                        _, tgt_mask, tgt_pad_mask = model_core.make_masks(src_tensor, ys)
                        tgt_emb = model_core.tgt_embed(ys) * math.sqrt(model_core.d_model)
                        out = model_core.decoder(tgt=tgt_emb, memory=memory, 
                                               tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
                    else:
                        # Logic cho Base Transformer
                        tgt_mask = model_core.generate_square_subsequent_mask(ys.size(1), self.device)
                        out = model_core.transformer.decoder(
                            model_core.positional_encoding(model_core.tgt_embedding(ys) * math.sqrt(model_core.d_model)), 
                            memory, 
                            tgt_mask=tgt_mask
                        )
                    
                    prob = model_core.fc_out(out[:, -1])
                    _, next_word = torch.max(prob, dim=1)
                    next_word = next_word.item()
                    
                    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                    if next_word == eos_idx:
                        break
        return ys[0]

    def val_epoch(self, dataloader, epoch, run_bleu=False):
        """Đánh giá mô hình trên tập Validation"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}", leave=False):
                src, tgt = batch
                src, tgt = src.to(self.device, non_blocking=True), tgt.to(self.device, non_blocking=True)
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                
                with torch.amp.autocast(device_type='cuda', dtype=self.mixed_precision_dtype, enabled=self.use_amp):
                    output = self.model.module(src, tgt_input)
                    loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        bleu = 0.0
        if run_bleu:
            print(f"📊 Calculating BLEU...")
            bleu = self.compute_bleu_sacrebleu(dataloader)
        return avg_loss, bleu

    def compute_bleu_sacrebleu(self, dataloader, max_samples=200):
        self.model.eval()
        preds, refs = [], []
        collected = 0
        with torch.no_grad():
            for src, tgt in tqdm(dataloader, desc="BLEU", leave=False):
                src = src.to(self.device)
                for i in range(src.size(0)):
                    if collected >= max_samples: break
                    pred_ids = self.greedy_decode(src[i], max_len=tgt.size(1) + 10)
                    pred_str = self.vocab.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
                    tgt_str = self.vocab.tokenizer.decode(tgt[i].tolist(), skip_special_tokens=True)
                    preds.append(pred_str)
                    refs.append(tgt_str)
                    collected += 1
                if collected >= max_samples: break
        if not preds: return 0.0
        return sacrebleu.corpus_bleu(preds, [refs]).score
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        save_dir = self.config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = self.model
        if isinstance(model_to_save, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_to_save = model_to_save.module
        state = {
            'epoch': epoch,
            'state_dict': model_to_save.state_dict(),
            'val_loss': val_loss
        }
        torch.save(state, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))