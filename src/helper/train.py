import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import sacrebleu
from torch.utils.data import DataLoader
from datetime import datetime
import math
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.base_transformer import TransformerTranslation
from data.loader import BPEDataManager


# --- TỐI ƯU 1: Bật TF32 cho RTX 3000/4000 series ---
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
        
        # Setup Multi-GPU (Nếu dùng 1 GPU thì bỏ qua cho nhanh)
        if self.config['gpu_mode'] and torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        # --- TỐI ƯU 2: Autocast dtype (Dùng bfloat16 cho 4090 để nhanh & ổn định) ---
        # RTX 4090 hỗ trợ tốt bfloat16, không cần GradScaler phức tạp
        self.mixed_precision_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.use_amp = config['use_amp']
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.mixed_precision_dtype == torch.float16 and self.use_amp))
        
        print(f"⚡ AMP Enabled: {self.use_amp} | Type: {self.mixed_precision_dtype}")

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        
        for batch in pbar:
            src, tgt = batch
            # Non-blocking transfer giúp tăng tốc load data
            src, tgt = src.to(self.device, non_blocking=True), tgt.to(self.device, non_blocking=True)
            
            tgt_input = tgt[:, :-1] 
            tgt_output = tgt[:, 1:] 

            self.optimizer.zero_grad(set_to_none=True) # set_to_none nhanh hơn zero_grad

            # --- TỐI ƯU 3: Mixed Precision Context ---
            with torch.amp.autocast(device_type='cuda', dtype=self.mixed_precision_dtype, enabled=self.use_amp):
                output = self.model(src, tgt_input) 
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            
            # Backward
            if self.use_amp and self.mixed_precision_dtype == torch.float16:
                # Dùng Scaler nếu là float16 (cũ)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # bfloat16 (RTX 30/40) không cần scaler, stable hơn
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if self.config['use_wandb']:
                wandb.log({"train_loss_step": loss.item()})
                
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def greedy_decode(self, src, max_len=50):
        # Lấy model gốc nếu đang dùng compile/dataparallel
        # torch.compile bọc model vào OptimizedModule, cần cẩn thận khi truy cập attributes
        model_core = self.model
        if isinstance(model_core, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_core = model_core.module
        # Nếu model bị compile, nó có thể có prefix '_orig_mod'
        if hasattr(model_core, "_orig_mod"):
             model_core = model_core._orig_mod
             
        model_core.eval()
        
        bos_idx = self.config['bos_idx']
        eos_idx = self.config['eos_idx']
        
        src_tensor = src.unsqueeze(0).to(self.device)
        
        # Inference cũng nên dùng AMP
        with torch.amp.autocast(device_type='cuda', dtype=self.mixed_precision_dtype, enabled=self.use_amp):
            src_mask = torch.zeros((src_tensor.shape[1], src_tensor.shape[1]), device=self.device).type(torch.bool)
            memory = model_core.transformer.encoder(
                model_core.positional_encoding(model_core.src_embedding(src_tensor) * math.sqrt(model_core.d_model)), 
                mask=src_mask
            )
            
            ys = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(self.device)
            
            for i in range(max_len - 1):
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
    
    def compute_bleu_sacrebleu(self, dataloader, max_samples=200, max_len_extra=5):
        self.model.eval()
        preds = []
        refs = []
        collected = 0

        with torch.no_grad():
            for src, tgt in tqdm(dataloader, desc="Calculating BLEU", leave=False):
                src, tgt = src.to(self.device), tgt.to(self.device)
                bsz = src.size(0)

                for i in range(bsz):
                    if collected >= max_samples: break

                    pred_ids = self.greedy_decode(src[i], max_len=tgt.size(1) + max_len_extra)
                    pred_str = self.vocab.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
                    tgt_str = self.vocab.tokenizer.decode(tgt[i].tolist(), skip_special_tokens=True)

                    preds.append(pred_str)
                    refs.append(tgt_str)
                    collected += 1
                if collected >= max_samples: break

        if not preds: return 0.0
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        return bleu.score

    def val_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"Val Epoch {epoch}")):
                src, tgt = batch
                src, tgt = src.to(self.device, non_blocking=True), tgt.to(self.device, non_blocking=True)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Val cũng dùng AMP để tiết kiệm VRAM và nhanh hơn
                with torch.amp.autocast(device_type='cuda', dtype=self.mixed_precision_dtype, enabled=self.use_amp):
                    output = self.model(src, tgt_input)
                    loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        bleu = self.compute_bleu_sacrebleu(dataloader=dataloader, max_samples=200)
        return avg_loss, bleu

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        save_dir = self.config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # Xử lý save khi dùng compile/dataparallel
        model_to_save = self.model
        if isinstance(model_to_save, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_to_save = model_to_save.module
        if hasattr(model_to_save, "_orig_mod"): # Nếu dùng torch.compile
             model_to_save = model_to_save._orig_mod

        state = {
            'epoch': epoch,
            'state_dict': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        filename = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(state, filename)
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"🌟 Saved new best model with loss: {val_loss:.4f}")

def build_save_dir(base_dir, model_name, dataset_name, src_lang, tgt_lang, add_timestamp=False):
    path = os.path.join(base_dir, model_name, dataset_name, f"{src_lang}_{tgt_lang}")
    if add_timestamp:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(path, time_str)
    return path

def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer Translation Model")
    # Path & Meta
    parser.add_argument('--data_dir', type=str, default='./data/processed/ALT', help='Đường dẫn data')
    parser.add_argument('--base_checkpoint_dir', type=str, default='./checkpoints', help='Thư mục gốc lưu model')
    parser.add_argument('--model_name', type=str, default='base_transformer', help='Tên mô hình')
    parser.add_argument('--dataset_name', type=str, default='ALT', help='Tên bộ dữ liệu')
    parser.add_argument('--src_lang', type=str, default='en', help='Ngôn ngữ nguồn')
    parser.add_argument('--tgt_lang', type=str, default='vi', help='Ngôn ngữ đích')
    parser.add_argument('--add_timestamp', action='store_true', help='Thêm timestamp vào folder save')
    
    # Training Config Optimized for RTX 4090
    parser.add_argument('--batch_size', type=int, default=128, help='Tăng lên 128/256 cho 4090') 
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Tăng số epoch lên 30')
    parser.add_argument('--no_gpu', action='store_true', help='Tắt GPU (ép dùng CPU)')
    parser.add_argument('--use_wandb', action='store_true', help='Sử dụng WandB để log')
    parser.add_argument('--wandb_api_key', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default="transformer_4090")
    parser.add_argument('--no_amp', action='store_true', help='Tắt Mixed Precision (không khuyến khích trên 4090)')

    # Model/Vocab
    parser.add_argument('--vocab_size', type=int, default=10000, help='Kích thước bộ từ điển')
    parser.add_argument('--bos_idx', type=int, default=2)
    parser.add_argument('--eos_idx', type=int, default=3)
    return parser.parse_args()

def run_training(config):
    print(f"🚀 Starting training with config: {config}")

    use_cuda = config['gpu_mode'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on: {device} | {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")
    
    # 1. Load Data
    dm = BPEDataManager(data_dir=config['data_dir'], src_lang=config['src_lang'],
                        tgt_lang=config['tgt_lang'], vocab_size=config['vocab_size'])
    
    actual_vocab_size = dm.tokenizer.get_vocab_size()
    print(f"⚠️ Actual Tokenizer Vocab Size: {actual_vocab_size}")
    config['vocab_size'] = actual_vocab_size
    config['pad_idx'] = dm.pad_id

    train_ds, val_ds = dm.get_datasets(val_ratio=0.1)

    # --- TỐI ƯU 4: Data Loader Workers & Pin Memory ---
    # Num workers = 4 là con số an toàn, có thể tăng lên 8 nếu CPU mạnh
    loader_kwargs = {
        "batch_size": config['batch_size'],
        "pin_memory": True,       # Bắt buộc cho GPU training
        "num_workers": 4,         # Load data song song
        "persistent_workers": True, # Giữ worker sống giữa các epoch
        "collate_fn": dm._collate_fn
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # --- TỐI ƯU 5: Config Model BASE (Phù hợp 4090) ---
    model = TransformerTranslation(
        src_vocab_size=config['vocab_size'], 
        tgt_vocab_size=config['vocab_size'], 
        d_model=512,          # Base: 512 (Trước là 256)
        nhead=8,              # Base: 8 (Trước là 4)
        num_encoder_layers=6, # Base: 6 (Trước là 4)
        num_decoder_layers=6, # Base: 6 (Trước là 4)
        dim_feedforward=2048, # Base: 2048 (Trước là 1024)
        dropout=0.1,          # Dùng 0.1 cho Base model, nếu data ít thì tăng 0.3
        pad_idx=dm.pad_id
    )
    
    model.to(device)

    # --- TỐI ƯU 6: Torch Compile (PyTorch 2.0+) ---
    # Tăng tốc độ training khoảng 20-30% trên 4090
    try:
        print("🔥 Compiling model with torch.compile...")
        model = torch.compile(model)
    except Exception as e:
        print(f"Warning: Could not compile model. Error: {e}")

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=dm.pad_id, label_smoothing=0.1) 

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['lr'], 
        steps_per_epoch=len(train_loader), 
        epochs=config['epochs'],
        pct_start=0.15 
    )

    if config['use_wandb']:
        if config.get('wandb_api_key'):
            wandb.login(key=config['wandb_api_key'])
        wandb.init(project=config['wandb_project'], config=config)

    trainer = Trainer(model, dm, optimizer, criterion, scheduler, device, config)
    
    best_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        # Validation mỗi epoch hoặc mỗi 5 epoch để tiết kiệm thời gian
        val_loss, val_bleu = trainer.val_epoch(val_loader, epoch)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {val_bleu:.2f}")
        
        if config['use_wandb']:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_bleu": val_bleu
            })
        
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            
        trainer.save_checkpoint(epoch, val_loss, is_best)

if __name__ == "__main__":
    args = get_args()
    config = vars(args)
    config['gpu_mode'] = not args.no_gpu 
    config['use_amp'] = not args.no_amp # Mặc định bật AMP

    config['save_dir'] = build_save_dir(
        base_dir=args.base_checkpoint_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        add_timestamp=args.add_timestamp
    )
    
    os.makedirs(config['save_dir'], exist_ok=True)
    print(f"Checkpoints will be saved to: {config['save_dir']}")

    run_training(config)