import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import os

# Import từ các module đã tách
from src.data.loader import BPEDataManager
from src.models import build_model
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"🚀 Loading Config: {args.config}")

    # 2. Setup Data
    # Giả sử trong class BPEDataManager bạn đã sửa để nhận dataset_name
    # data_root/raw/{dataset_name}
    dm = BPEDataManager(
        data_dir=os.path.join("./data/processed", config['data']['dataset_name']), 
        src_lang=config['data']['src_lang'],
        tgt_lang=config['data']['tgt_lang'],
        vocab_size=config['data']['vocab_size']
    )
    
    real_vocab = dm.tokenizer.get_vocab_size()
    print(f"⚠️ Actual Tokenizer Vocab Size: {real_vocab}")
    
    # Update config runtime
    config['model']['vocab_size'] = real_vocab # Update để truyền vào model
    config['train']['bos_idx'] = dm.tokenizer.token_to_id('<bos>')
    config['train']['eos_idx'] = dm.tokenizer.token_to_id('<eos>')
    config['train']['pad_idx'] = dm.pad_id

    train_ds, val_ds = dm.get_datasets(val_ratio=0.1)

    loader_kwargs = {
        "batch_size": config['train']['batch_size'],
        "pin_memory": True,
        "num_workers": 4,
        "persistent_workers": True,
        "collate_fn": dm._collate_fn
    }
    
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # 3. Build Model (Dùng Factory Function)
    # config['model'] chứa các tham số: type, d_model, nhead...
    model = build_model(
        config=config['model'], 
        vocab_size=real_vocab, 
        pad_idx=dm.pad_id
    )
    
    # 4. Setup Device & Optimizer
    use_cuda = config['train']['gpu_mode'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'], betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=dm.pad_id, label_smoothing=0.1) 

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['train']['lr'], 
        steps_per_epoch=len(train_loader), 
        epochs=config['train']['epochs'],
        pct_start=0.15 
    )

    # WandB
    if config['logging']['use_wandb']:
        wandb.login(key=config['logging']['wandb_api_key'])
        wandb.init(project=config['logging']['project_name'], config=config)

    # 5. Start Training
    # config['train'] chứa: use_amp, epochs, save_dir...
    trainer = Trainer(
        model, 
        dm, 
        optimizer, 
        criterion, 
        scheduler, 
        device, 
        config['train'] # Truyền đúng phần config train
    )
    
    # Custom training loop ở ngoài hoặc gọi hàm train() nếu bạn viết hàm đó trong Trainer
    # Ở code trainer trên tôi viết train_epoch/val_epoch rời, nên ta loop ở đây
    
    print("🚀 Start Training Loop...")
    best_loss = float('inf')
    
    for epoch in range(1, config['train']['epochs'] + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Logic tính BLEU mỗi 10 epoch
        run_bleu = (epoch % 10 == 0) or (epoch == config['train']['epochs'])
        val_loss, val_bleu = trainer.val_epoch(val_loader, epoch, run_bleu=run_bleu)
        
        print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | BLEU: {val_bleu}")
        
        if config['logging']['use_wandb']:
             wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_bleu": val_bleu})
             
        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, is_best=True)

if __name__ == '__main__':
    main()