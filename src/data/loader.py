import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

class BPEDataManager:
    """
    Quản lý dữ liệu Dịch máy sử dụng BPE Tokenizer.
    """
    def __init__(self, data_dir, src_lang, tgt_lang, vocab_size=30000, model_prefix="bpe_model"):
        self.data_dir = data_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # Lưu tokenizer vào folder data để tái sử dụng
        self.tokenizer_path = os.path.join(data_dir, f"{model_prefix}_{vocab_size}.json")
        
        # 1. Tìm file dữ liệu
        self.src_file = self._find_file(src_lang)
        self.tgt_file = self._find_file(tgt_lang)
        
        print(f"✅ Found Source: {self.src_file}")
        print(f"✅ Found Target: {self.tgt_file}")

        # 2. Load hoặc Train Tokenizer
        if os.path.exists(self.tokenizer_path):
            print(f"🔄 Loading existing tokenizer from {self.tokenizer_path}...")
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            print("⚠️ Tokenizer not found. Training new BPE Tokenizer...")
            self.tokenizer = self._train_bpe(vocab_size)
            self.tokenizer.save(self.tokenizer_path)
            print(f"💾 Tokenizer saved to {self.tokenizer_path}")
            
        print(f"📊 Vocab Size: {self.tokenizer.get_vocab_size()}")
        
        # Cache special token IDs
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

    def _find_file(self, lang):
        """Tìm file linh hoạt: hỗ trợ các định dạng tên file phổ biến"""
        # Ưu tiên các định dạng tên file thường gặp
        candidates = [
            f"data_{lang}.txt",     # ALT format
            f"train.{lang}.txt",    # IWSLT format
            f"{lang}.txt",          # Simple format
            f"train.{lang}",        # Raw format
            f"test.{lang}.txt"      # Test files
        ]
        
        for fname in candidates:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                return path
                
        # Liệt kê các file trong thư mục để báo lỗi rõ hơn
        files_in_dir = os.listdir(self.data_dir) if os.path.exists(self.data_dir) else "Directory not found"
        raise FileNotFoundError(
            f"Could not find data file for language '{lang}' in {self.data_dir}.\n"
            f"Tried: {candidates}\n"
            f"Available files: {files_in_dir}"
        )

    def _train_bpe(self, vocab_size):
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            show_progress=True
        )
        
        files = [self.src_file, self.tgt_file]
        tokenizer.train(files, trainer)
        return tokenizer

    def _collate_fn(self, batch):
        """Gom batch và padding"""
        src_batch, tgt_batch = [], []
        
        for src_item, tgt_item in batch:
            src_batch.append(src_item)
            tgt_batch.append(tgt_item)
            
        # Padding value phải lấy từ self.pad_id
        src_batch = pad_sequence(src_batch, padding_value=self.pad_id, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_id, batch_first=True)
        
        return src_batch, tgt_batch
    
    def get_datasets(self, val_ratio=0.1, seed=42):
        """Chia train/val set"""
        full_dataset = MTDataset(self.src_file, self.tgt_file, self.tokenizer)

        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size

        return random_split(
            full_dataset, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(seed)
        )

    # Thêm hàm này để tương thích với code cũ nếu cần
    def get_dataloader(self, batch_size=32, shuffle=True):
        dataset = MTDataset(self.src_file, self.tgt_file, self.tokenizer)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=self._collate_fn,
            num_workers=2, # Tăng tốc load data
            pin_memory=True # Tăng tốc GPU transfer
        )

class MTDataset(Dataset):
    """
    Dataset class tối ưu:
    1. Cache token IDs trong __init__ để không phải lookup lại mỗi lần __getitem__.
    2. Encode on-the-fly.
    """
    def __init__(self, src_path, tgt_path, tokenizer):
        self.tokenizer = tokenizer
        
        # Cache IDs (QUAN TRỌNG: Tăng tốc đáng kể khi train)
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.unk_id = tokenizer.token_to_id("<unk>")
        
        print(f"Reading data from:\n  src: {src_path}\n  tgt: {tgt_path}")
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f if line.strip()] # Bỏ dòng trống
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f if line.strip()] # Bỏ dòng trống
            
        # Kiểm tra lệch dòng
        if len(self.src_lines) != len(self.tgt_lines):
            print(f"⚠️ Warning: Source ({len(self.src_lines)}) and Target ({len(self.tgt_lines)}) length mismatch!")
            min_len = min(len(self.src_lines), len(self.tgt_lines))
            self.src_lines = self.src_lines[:min_len]
            self.tgt_lines = self.tgt_lines[:min_len]
            print(f"   -> Truncated both to {min_len} lines.")

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]
        
        # Encode
        # Sử dụng tokenizer đã cache
        src_ids = self.tokenizer.encode(src_text).ids
        tgt_ids = self.tokenizer.encode(tgt_text).ids
        
        # Thêm BOS & EOS thủ công (nhanh hơn dùng processor)
        # Sử dụng ID đã cache trong self (nhanh hơn lookup dictionary)
        src_out = [self.bos_id] + src_ids + [self.eos_id]
        tgt_out = [self.bos_id] + tgt_ids + [self.eos_id]
        
        return torch.tensor(src_out, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long)