import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
from tqdm import tqdm
import pandas as pd
import argparse
import os
import sys

# --- Import các class từ file cũ (giả sử bạn lưu code model ở file model.py và data ở data_loader.py) ---
# Nếu bạn để chung 1 file notebook thì không cần import, chỉ cần copy paste class vào đay
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from model.base_transformer import TransformerTranslation
    from data.loader import BPEDataManager
except ImportError:
    print("⚠️ Vui lòng đảm bảo class TransformerTranslation và BPEDataManager có thể import được.")
    pass

def greedy_decode(model, src, max_len, device, bos_idx, eos_idx):
    """
    Hàm giải mã Greedy (Lấy token có xác suất cao nhất tại mỗi bước)
    Input: src shape (1, Seq_Len)
    """
    model.eval()
    
    # 1. Encoder
    # Tạo mask cho src (padding mask)
    # src_padding_mask = (src == model.pad_idx) # Nếu model có attribute pad_idx
    # Ở đây ta làm đơn giản, giả sử src đã clean
    
    # Forward qua Encoder để lấy memory
    src_emb = model.positional_encoding(model.src_embedding(src) * torch.sqrt(torch.tensor(model.d_model)))
    memory = model.transformer.encoder(src_emb)
    
    # 2. Decoder loop
    # Khởi tạo input cho decoder là <bos>
    ys = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        # Tạo mask tam giác (causal mask)
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)
        
        # Embed decoder input
        tgt_emb = model.positional_encoding(model.tgt_embedding(ys) * torch.sqrt(torch.tensor(model.d_model)))
        
        # Forward Decoder
        out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        # Projection layer
        prob = model.fc_out(out[:, -1]) # Lấy token cuối cùng
        
        # Chọn từ có xác suất cao nhất
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # Nối vào chuỗi kết quả
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        # Nếu gặp <eos> thì dừng
        if next_word == eos_idx:
            break
            
    return ys

def evaluate(model, dataloader, tokenizer, device, max_len=50):
    model.eval()
    
    sources = []
    targets = []
    predictions = []
    
    bleu_metric = BLEUScore()
    
    bos_idx = tokenizer.token_to_id("<bos>")
    eos_idx = tokenizer.token_to_id("<eos>")
    
    print("🚀 Bắt đầu quá trình dịch (Inference)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src_batch, tgt_batch = batch
            src_batch = src_batch.to(device)
            
            # Vì Greedy Decode chạy vòng lặp, ta xử lý từng câu một (Batch size = 1 logic)
            # Hoặc lặp qua batch hiện tại
            for i in range(src_batch.shape[0]):
                src_sent = src_batch[i].unsqueeze(0) # (1, Seq_Len)
                
                # --- PREDICT ---
                # Chạy Greedy Decode
                pred_indices = greedy_decode(model, src_sent, max_len, device, bos_idx, eos_idx)
                
                # --- DECODE TO TEXT ---
                # Loại bỏ token đặc biệt để output nhìn sạch sẽ
                pred_str = tokenizer.decode(pred_indices.squeeze().tolist(), skip_special_tokens=True)
                
                # Target thực tế
                tgt_str = tokenizer.decode(tgt_batch[i].tolist(), skip_special_tokens=True)
                
                # Source gốc (để đối chiếu)
                src_str = tokenizer.decode(src_batch[i].tolist(), skip_special_tokens=True)
                
                sources.append(src_str)
                targets.append(tgt_str)
                predictions.append(pred_str)

    # --- TÍNH BLEU ---
    # Targets cho BLEU cần format là list of list of refs: [[ref1], [ref2], ...]
    bleu_targets = [[t] for t in targets]
    score = bleu_metric(predictions, bleu_targets)
    
    return score.item(), sources, targets, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn folder data')
    parser.add_argument('--checkpoint', type=str, required=True, help='File .pth model đã train')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='vi')
    parser.add_argument('--output_file', type=str, default='test_results.csv', help='File lưu kết quả dịch')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data Manager (để lấy Tokenizer)
    # Lưu ý: vocab_size phải khớp với lúc train, ở đây giả định tokenizer đã lưu file json
    print("Loading Data Manager...")
    dm = BPEDataManager(args.data_dir, args.src_lang, args.tgt_lang)
    vocab_size = dm.tokenizer.get_vocab_size()
    
    # 2. Tạo DataLoader cho tập Test
    # Nếu BPEDataManager chưa có hàm get_test_dataloader, ta dùng tạm logic get_dataloader
    # Bạn cần đảm bảo trỏ đúng file test (ví dụ: test.en.txt)
    # Ở đây hack nhẹ: thay file src/tgt của dm bằng file test
    # (Cách chuẩn là viết thêm hàm load_test trong class BPEDataManager)
    print("Loading Test Data...")
    
    # Logic tìm file test (tùy chỉnh theo tên file thực tế của bạn)
    test_src = os.path.join(args.data_dir, f"test.{args.src_lang}.txt") # Hoặc data_test_en.txt
    test_tgt = os.path.join(args.data_dir, f"test.{args.tgt_lang}.txt")
    
    # Kiểm tra file tồn tại không, nếu không dùng file val hoặc train demo
    if not os.path.exists(test_src):
        print(f"Không thấy file test {test_src}, dùng tạm file train để demo code...")
        test_src = dm.src_file
        test_tgt = dm.tgt_file
    
    # Override file path và tạo loader
    dm.src_file = test_src
    dm.tgt_file = test_tgt
    test_loader = dm.get_dataloader(batch_size=32, shuffle=False) # Shuffle False để so sánh

    # 3. Load Model Structure
    print("Initializing Model...")
    # CÁC THAM SỐ NÀY PHẢI KHỚP 100% VỚI FILE TRAIN
    # Tốt nhất là lưu config vào checkpoint, nhưng ở đây ta hardcode cho Transformer Base
    model = TransformerTranslation(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        pad_idx=dm.pad_id
    )
    
    # 4. Load Weights
    print(f"Loading Checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Xử lý trường hợp lưu model với DataParallel (key có prefix 'module.')
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(device)
    
    # 5. Run Evaluation
    bleu, srcs, tgts, preds = evaluate(model, test_loader, dm.tokenizer, device)
    
    print(f"\n============================")
    print(f"🏅 TEST BLEU SCORE: {bleu:.4f}")
    print(f"============================")
    
    # 6. Save Results
    df = pd.DataFrame({
        'Source': srcs,
        'Target': tgts,
        'Prediction': preds
    })
    df.to_csv(args.output_file, index=False, encoding='utf-8')
    print(f"Results saved to {args.output_file}")
    
    # In thử vài câu
    print("\n--- Examples ---")
    for i in range(3):
        print(f"Src : {srcs[i]}")
        print(f"Ref : {tgts[i]}")
        print(f"Pred: {preds[i]}")
        print("-" * 20)

if __name__ == "__main__":
    main()