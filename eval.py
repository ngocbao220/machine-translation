import argparse
import yaml
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import chính xác các module bạn đã sử dụng trong train.py
from src.data.loader import BPEDataManager
from src.models import build_model
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of sample translations to show')
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"🔎 Evaluating Model Type: {config['model']['model_type']}")

    # 2. Setup Data (Tái sử dụng logic từ train.py)
    # Tokenizer sẽ được load lại từ folder đã train thay vì train mới
    dm = BPEDataManager(
        data_dir=os.path.join("./data/raw", config['data']['dataset_name']), 
        src_lang=config['data']['src_lang'],
        tgt_lang=config['data']['tgt_lang'],
        vocab_size=config['data']['vocab_size']
    )
    
    real_vocab = dm.tokenizer.get_vocab_size()
    
    # Cập nhật thông số kỹ thuật cho model/trainer
    config['train']['bos_idx'] = dm.tokenizer.token_to_id('<bos>')
    config['train']['eos_idx'] = dm.tokenizer.token_to_id('<eos>')
    config['train']['pad_idx'] = dm.pad_id
    config['model']['model_type'] = config['model'].get('model_type', 'base')

    # Lấy tập Validation (hoặc Test nếu bạn đã tách riêng) để đánh giá
    _, val_ds = dm.get_datasets(val_ratio=0.1)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], 
                            shuffle=False, collate_fn=dm._collate_fn)

    # 3. Build Model & Load Weights
    model = build_model(
        config=config['model'], 
        vocab_size=real_vocab, 
        pad_idx=dm.pad_id
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config['train']['gpu_mode'] else "cpu")
    
    # Tải trọng số
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Xử lý trường hợp checkpoint lưu cả state_dict hoặc chỉ có trọng số
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    # 4. Khởi tạo Trainer (Chỉ dùng để lấy hàm greedy_decode và compute_bleu)
    # Các tham số optimizer/scheduler để None vì không dùng khi Eval
    trainer = Trainer(
        model, dm, None, None, None, device, config['train']
    )

    # 5. Đánh giá định lượng (BLEU Score)
    print(f"\n📊 Calculating BLEU score on {len(val_ds)} samples...")
    # Tăng max_samples nếu bạn muốn đo trên toàn bộ bộ test
    final_bleu = trainer.compute_bleu_sacrebleu(val_loader, max_samples=1000)
    print(f"🔥 FINAL TEST BLEU: {final_bleu:.2f}")

    # 6. Đánh giá định tính (Dịch thử mẫu thực tế)
    print(f"\n📝 Qualitative Analysis (Top {args.num_samples} samples):")
    print("-" * 60)
    
    for i in range(min(args.num_samples, len(val_ds))):
        src_ids, tgt_ids = val_ds[i]
        
        # Chuyển src_ids sang tensor
        src_tensor = torch.LongTensor(src_ids).to(device)
        
        # Dịch bằng greedy_decode (Tự động nhận diện Re-Transformer/Base)
        pred_ids = trainer.greedy_decode(src_tensor, max_len=config['data'].get('max_seq_len', 50))
        
        # Giải mã ID sang Text
        src_text = dm.tokenizer.decode(src_ids, skip_special_tokens=True)
        ref_text = dm.tokenizer.decode(tgt_ids, skip_special_tokens=True)
        pred_text = dm.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
        
        print(f"Source: {src_text}")
        print(f"Reference: {ref_text}")
        print(f"Predicted: {pred_text}")
        print("-" * 40)

if __name__ == '__main__':
    main()