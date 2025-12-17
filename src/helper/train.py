import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from torchmetrics.text import BLEUScore
from torch.utils.data import DataLoader
from datetime import datetime
import math
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.base_transformer import TransformerTranslation
from data.loader import BPEDataManager

class Trainer:
    def __init__(self, model, vocab, optimizer, criterion, scheduler, device, config):
        self.model = model
        self.vocab = vocab # Object chứa tokenizer (để decode tính BLEU)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.bleu_metric = BLEUScore()
        
        # Setup Multi-GPU
        if self.config['gpu_mode'] and torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        
        for batch in pbar:
            src, tgt = batch
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # --- INPUT PREPARATION FOR TRANSFORMER ---
            # Transformer Decoder cần:
            # 1. Input: Từ đầu đến áp chót (<bos> ... token_cuối)
            # 2. Target (Label): Từ thứ 2 đến hết (token_1 ... <eos>)
            tgt_input = tgt[:, :-1] 
            tgt_output = tgt[:, 1:] 

            # Forward
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input) 
            # Output Shape: (Batch, Seq_Len, Vocab_Size)
            
            # Reshape để tính Loss
            # output: (Batch * Seq_Len, Vocab_Size)
            # tgt_output: (Batch * Seq_Len)
            loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            
            # Backward
            loss.backward()
            
            # Clip gradient để tránh bùng nổ gradient (quan trọng cho Transformer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log step-level
            if self.config['use_wandb']:
                wandb.log({"train_loss_step": loss.item()})
                
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def greedy_decode(self, src, max_len=50):
        """Hàm dịch cơ bản (Greedy) để dùng cho việc Validation"""
        # Lưu ý: Khi dùng DataParallel, truy cập model gốc qua model.module
        model_core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_core.eval()
        
        bos_idx = self.config['bos_idx']
        eos_idx = self.config['eos_idx']
        
        # src: (1, Seq_Len)
        src_tensor = src.unsqueeze(0).to(self.device)
        
        # Encoder output (chạy 1 lần)
        src_mask = torch.zeros((src_tensor.shape[1], src_tensor.shape[1]), device=self.device).type(torch.bool)
        memory = model_core.transformer.encoder(
            model_core.positional_encoding(model_core.src_embedding(src_tensor) * math.sqrt(model_core.d_model)), 
            mask=src_mask
        )
        
        # Bắt đầu decoder với <bos>
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

    def val_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        preds_text = []
        targets_text = []
        
        # Chỉ lấy 1 subset để tính BLEU cho nhanh nếu data quá lớn
        # Ở đây tôi tính loss trên toàn bộ, nhưng BLEU có thể sample
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"Val Epoch {epoch}")):
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = self.model(src, tgt_input)
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                total_loss += loss.item()

                # --- TÍNH BLEU (SAMPLE) ---
                # Tính BLEU tốn thời gian vì phải chạy greedy decode từng câu
                # Chỉ tính trên 5 batch đầu tiên mỗi epoch để theo dõi tiến độ
                if i < 5: 
                    # Decode câu đầu tiên trong batch
                    pred_indices = self.greedy_decode(src[0], max_len=tgt.shape[1] + 5)
                    
                    # Convert IDs -> Text
                    # self.vocab.tokenizer là instance của Tokenizer library
                    pred_str = self.vocab.tokenizer.decode(pred_indices.tolist(), skip_special_tokens=True)
                    target_str = self.vocab.tokenizer.decode(tgt[0].tolist(), skip_special_tokens=True)
                    
                    print(f"\n--- DEBUG STEP {i} ---")
                    print(f"Src IDs : {src[0][:10].tolist()}...") # Xem input vào có đúng ko
                    print(f"Pred IDs: {pred_indices.tolist()}")    # Xem model nhả ra ID gì
                    print(f"Pred Str: '{pred_str}'")               # Xem text
                    print(f"Real Str: '{target_str}'")             # Xem nhãn

                    preds_text.append(pred_str)
                    targets_text.append([target_str]) # BLEU cần list of references

        avg_loss = total_loss / len(dataloader)
        
        # Compute BLEU
        bleu = 0
        if preds_text:
            bleu = self.bleu_metric(preds_text, targets_text).item()
            
            # Print sample visualization
            print(f"\nExample Val Pred: {preds_text[0]}")
            print(f"Example Val Real: {targets_text[0][0]}")
            
        return avg_loss, bleu

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        save_dir = self.config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # Save per epoch
        filename = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(state, filename)
        
        # Save best
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"🌟 Saved new best model with loss: {val_loss:.4f}")

def build_save_dir(base_dir, model_name, dataset_name, src_lang, tgt_lang, add_timestamp=False):
    """Tạo đường dẫn lưu checkpoint"""
    path = os.path.join(
        base_dir,
        model_name,
        dataset_name,
        f"{src_lang}_{tgt_lang}"
    )

    if add_timestamp:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(path, time_str)

    return path

def get_args():
    """Định nghĩa các tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description="Train Transformer Translation Model")

    # --- Path Arguments ---
    parser.add_argument('--data_dir', type=str, default='./data/processed/ALT', help='Đường dẫn data')
    parser.add_argument('--base_checkpoint_dir', type=str, default='./checkpoints', help='Thư mục gốc lưu model')

    # --- Meta Data ---
    parser.add_argument('--model_name', type=str, default='base_transformer', help='Tên mô hình')
    parser.add_argument('--dataset_name', type=str, default='ALT', help='Tên bộ dữ liệu')
    parser.add_argument('--src_lang', type=str, default='en', help='Ngôn ngữ nguồn')
    parser.add_argument('--tgt_lang', type=str, default='vi', help='Ngôn ngữ đích')
    parser.add_argument('--add_timestamp', action='store_true', help='Thêm timestamp vào folder save')

    # --- Training Hyperparams ---
    parser.add_argument('--batch_size', type=int, default=64, help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Số epoch')
    
    # --- Flags (Boolean) ---
    # Mặc định là False, nếu thêm flag vào câu lệnh thì thành True
    parser.add_argument('--no_gpu', action='store_true', help='Tắt GPU (ép dùng CPU)')
    parser.add_argument('--use_wandb', action='store_true', help='Sử dụng WandB để log')
    # --- MỚI THÊM: WandB API Key ---
    parser.add_argument('--wandb_api_key', type=str, default=None, help='WandB API Key để login tự động')
    parser.add_argument('--wandb_project', type=str, default="transformer_base", help='WandB API Key để login tự động')

    # --- Model/Vocab ---
    parser.add_argument('--vocab_size', type=int, default=10000, help='Kích thước bộ từ điển')
    parser.add_argument('--bos_idx', type=int, default=2)
    parser.add_argument('--eos_idx', type=int, default=3)

    return parser.parse_args()

def run_training(config):
    """Hàm train nhận config từ bên ngoài"""
    print(f"🚀 Starting training with config: {config}")

    # 1. Setup Device
    # Logic: Nếu user gõ --no_gpu thì config['gpu_mode'] sẽ là False
    use_cuda = config['gpu_mode'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on: {device}")
    
    # 2. Load Data (Phần code thật của bạn)
    dm = BPEDataManager(data_dir=config['data_dir'], src_lang=config['src_lang'],
                        tgt_lang=config['tgt_lang'], vocab_size=config['vocab_size'])
    
    print(f"BOS ID: {dm.tokenizer.token_to_id('<bos>')}")
    print(f"EOS ID: {dm.tokenizer.token_to_id('<eos>')}")
    print(f"PAD ID: {dm.tokenizer.token_to_id('<pad>')}")

    # FIX BUGG
    actual_vocab_size = dm.tokenizer.get_vocab_size()
    print(f"⚠️ Actual Tokenizer Vocab Size: {actual_vocab_size} (Config requested: {config['vocab_size']})")
    
    # Cập nhật lại config và dùng số thực tế này để init model
    config['vocab_size'] = actual_vocab_size

    train_ds, val_ds = dm.get_datasets(val_ratio=0.1)

    train_loader = DataLoader(
    train_ds,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=dm._collate_fn
    )

    val_loader = DataLoader(
    val_ds,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=dm._collate_fn
    )

    # --- MOCK DATA LOADER ---
    # from unittest.mock import MagicMock
    # train_loader = [ (torch.randint(0,100,(4,10)), torch.randint(0,100,(4,10))) for _ in range(10) ]
    # val_loader = [ (torch.randint(0,100,(4,10)), torch.randint(0,100,(4,10))) for _ in range(2) ]
    
    model = TransformerTranslation(config['vocab_size'], config['vocab_size'], pad_idx=1)
    # vocab_mock = MagicMock()
    # vocab_mock.tokenizer.decode.return_value = "xin chào việt nam"
    # ------------------------

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1) 

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.0005, 
        steps_per_epoch=len(train_loader), 
        epochs=config['epochs'],
        pct_start=0.1  # 10% thời gian đầu dùng để warmup
    )

    # 4. Init WandB
    if config['use_wandb']:
        # --- MỚI THÊM: Logic Login ---
        if config.get('wandb_api_key'):
            print("🔑 Found WandB API Key. Logging in...")
            wandb.login(key=config['wandb_api_key'])
        
        wandb.init(project=config['wandb_project'], config=config)

    # 5. Init Trainer
    trainer = Trainer(model, dm, optimizer, criterion, scheduler, device, config)
    
    best_loss = float('inf')
    
    # 6. Training Loop
    for epoch in range(1, config['epochs'] + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
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
    # 1. Parse Arguments từ CLI
    args = get_args()
    
    # 2. Convert args (Namespace) sang Dict để tương thích code cũ
    config = vars(args)
    
    # 3. Xử lý logic custom (ví dụ đảo ngược logic no_gpu thành gpu_mode)
    config['gpu_mode'] = not args.no_gpu 

    # 4. Tạo đường dẫn save_dir tự động
    config['save_dir'] = build_save_dir(
        base_dir=args.base_checkpoint_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        add_timestamp=args.add_timestamp
    )
    
    # Đảm bảo folder tồn tại
    os.makedirs(config['save_dir'], exist_ok=True)
    print(f"Checkpoints will be saved to: {config['save_dir']}")

    # 5. Chạy training
    run_training(config)