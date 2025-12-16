import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from torch.utils.data import random_split

class BPEDataManager:
    """
    Quản lý dữ liệu Dịch máy sử dụng BPE Tokenizer.
    Tự động train BPE trên dữ liệu nguồn và đích để tạo Shared Vocabulary.
    """
    def __init__(self, data_dir, src_lang, tgt_lang, vocab_size=30000, model_prefix="bpe_model"):
        self.data_dir = data_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_path = os.path.join(data_dir, f"{model_prefix}.json")
        
        # Định dạng tên file dựa trên hình ảnh bạn cung cấp
        # Ví dụ: data_en.txt, data_vi.txt hoặc train.en.txt
        # Bạn có thể sửa logic tìm file ở đây
        self.src_file = self._find_file(src_lang)
        self.tgt_file = self._find_file(tgt_lang)
        
        print(f"Found Source: {self.src_file}")
        print(f"Found Target: {self.tgt_file}")

        # 1. Load hoặc Train Tokenizer
        if os.path.exists(self.tokenizer_path):
            print(f"Loading existing tokenizer from {self.tokenizer_path}...")
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            print("Training BPE Tokenizer...")
            self.tokenizer = self._train_bpe(vocab_size)
            self.tokenizer.save(self.tokenizer_path)
            
        print(f"Vocab Size: {self.tokenizer.get_vocab_size()}")
        
        # Lấy ID của các token đặc biệt để dùng cho padding sau này
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

    def _find_file(self, lang):
        """Logic tìm file linh hoạt: hỗ trợ cả data_en.txt (ALT) và train.en.txt (IWSLT)"""
        candidates = [f"data_{lang}.txt", f"train.{lang}.txt", f"{lang}.txt"]
        for fname in candidates:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find data file for language '{lang}' in {self.data_dir}")

    def _train_bpe(self, vocab_size):
        """Train BPE tokenizer trên cả 2 file src và tgt (Shared Vocab)"""
        # Khởi tạo mô hình BPE
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # Pre-normalization & Pre-tokenization (tách dấu câu, khoảng trắng)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        # Thiết lập Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            show_progress=True
        )
        
        # Train trên danh sách các file (kết hợp cả src và tgt để học tốt hơn)
        files = [self.src_file, self.tgt_file]
        tokenizer.train(files, trainer)
        
        return tokenizer

    def _collate_fn(self, batch):
        """Hàm gom batch và padding"""
        src_batch, tgt_batch = [], []
        
        for src_item, tgt_item in batch:
            src_batch.append(src_item)
            tgt_batch.append(tgt_item)
            
        # Padding: batch_first=True -> (Batch, Seq_Len)
        src_batch = pad_sequence(src_batch, padding_value=self.pad_id, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_id, batch_first=True)
        
        return src_batch, tgt_batch
    
    def get_datasets(self, val_ratio=0.1, seed=42):
        dataset = MTDataset(self.src_file, self.tgt_file, self.tokenizer)

        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size

        torch.manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        return train_ds, val_ds

    def get_dataloader(self, batch_size=32, shuffle=True):
        dataset = MTDataset(self.src_file, self.tgt_file, self.tokenizer)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=self._collate_fn
        )

class MTDataset(Dataset):
    """Dataset đọc file và encode on-the-fly để tiết kiệm RAM"""
    def __init__(self, src_path, tgt_path, tokenizer):
        self.tokenizer = tokenizer
        # Đọc toàn bộ lines vào memory (nếu file < vài GB thì vẫn OK)
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f]
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f]
            
        assert len(self.src_lines) == len(self.tgt_lines), "Source and Target files must have same length"

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]
        
        # Encode thêm <bos> và <eos> thủ công hoặc dùng post-processor của tokenizer
        # Ở đây làm thủ công cho trực quan:
        # Encode text -> lấy ids
        src_ids = self.tokenizer.encode(src_text).ids
        tgt_ids = self.tokenizer.encode(tgt_text).ids
        
        # Thêm BOS (bắt đầu) và EOS (kết thúc)
        bos_id = self.tokenizer.token_to_id("<bos>")
        eos_id = self.tokenizer.token_to_id("<eos>")
        
        src_out = [bos_id] + src_ids + [eos_id]
        tgt_out = [bos_id] + tgt_ids + [eos_id]
        
        return torch.tensor(src_out, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long)

# --- HƯỚNG DẪN SỬ DỤNG VỚI DATA CỦA BẠN ---
if __name__ == "__main__":
    # Đường dẫn tới thư mục ALT-Parallel-Corpus trong hình của bạn
    # Ví dụ: đường dẫn tuyệt đối hoặc tương đối
    data_path = "../../data/processed/ALT-Parallel-Corpus-20191206"
    
    # 1. Khởi tạo Manager
    # Nó sẽ tự tìm file 'data_en.txt' và 'data_vi.txt' trong folder trên
    # Lần đầu chạy sẽ hơi lâu vì phải Train BPE
    dm = BPEDataManager(data_dir=data_path, src_lang="en", tgt_lang="vi", vocab_size=10000)

    # 2. Lấy DataLoader
    train_loader = dm.get_dataloader(batch_size=4)

    # 3. Test thử
    for src, tgt in train_loader:
        print("\n--- Batch Info ---")
        print("Source Shape:", src.shape) 
        print("Target Shape:", tgt.shape)
        
        # Decode thử dòng đầu tiên để xem BPE hoạt động thế nào
        print("\nOriginal Decoded (Src):")
        print(dm.tokenizer.decode(src[0].tolist()))
        print("\nOriginal Decoded (Tgt):")
        print(dm.tokenizer.decode(tgt[0].tolist()))
        break