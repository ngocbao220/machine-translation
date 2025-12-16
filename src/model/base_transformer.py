import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tạo matrix vị trí (position)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Áp dụng sin cho vị trí chẵn, cos cho vị trí lẻ
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Thêm dimension cho batch: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Đăng ký buffer để không bị coi là parameter (không train cái này)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch_Size, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTranslation(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 nhead=8, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 pad_idx=1): # pad_idx mặc định là 1 trong code trước
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 1. Embedding Layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 3. Transformer Core (PyTorch implementation)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Quan trọng: Input shape là (Batch, Seq)
        )
        
        # 4. Output Layer (Projection)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_square_subsequent_mask(self, sz, device):
        """Tạo mask tam giác trên để che các từ tương lai trong Decoder"""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, device):
        """Tạo tất cả các mask cần thiết cho 1 batch"""
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # Target Mask (Causal Mask): Che tương lai
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        
        # Source Mask: Thường là False hết (để encoder nhìn thấy toàn bộ câu nguồn)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

        # Padding Mask: Che vị trí <pad> (True là bị che, False là được nhìn)
        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        """
        src: (Batch, Src_Seq_Len)
        tgt: (Batch, Tgt_Seq_Len) - Lưu ý: tgt ở đây là input cho decoder (đã bỏ token cuối)
        """
        device = src.device
        
        # Tạo masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt, device)
        
        # Embed + Position
        # Theo paper gốc: nhân embedding với sqrt(d_model)
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer Pass
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask, # Optional trong encoder nhưng cứ để
            tgt_mask=tgt_mask, # BẮT BUỘC cho decoder
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask # Mask memory theo padding của src
        )
        
        # Projection -> Vocab Size
        return self.fc_out(outs)

# --- RUN TEST ---
if __name__ == "__main__":
    # 1. Giả sử đã có tokenizer từ bước trước
    # vocab_size = dm.tokenizer.get_vocab_size()
    # pad_id = dm.pad_id
    
    # Giả lập tham số
    VOCAB_SIZE = 10000 
    PAD_ID = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Khởi tạo Model (Transformer Base config)
    model = TransformerTranslation(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=512,          # Base
        nhead=8,              # Base
        num_encoder_layers=6, # Base
        num_decoder_layers=6, # Base
        dim_feedforward=2048, # Base
        pad_idx=PAD_ID
    ).to(device)

    # 3. Tạo dữ liệu giả để test flow
    # Batch = 2, Seq_Len = 10
    src_sample = torch.randint(2, VOCAB_SIZE, (2, 10)).to(device)
    tgt_sample = torch.randint(2, VOCAB_SIZE, (2, 9)).to(device) # Tgt input ngắn hơn 1 chút cũng đc

    # 4. Forward Pass
    print("Input Src Shape:", src_sample.shape)
    print("Input Tgt Shape:", tgt_sample.shape)
    
    output = model(src_sample, tgt_sample)
    
    print("Output Shape:", output.shape) 
    # Output Shape kỳ vọng: (Batch, Seq_Len_Tgt, Vocab_Size) -> (2, 9, 10000)
    
    # Đếm số lượng tham số
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng tham số: {trainable_params:,}")