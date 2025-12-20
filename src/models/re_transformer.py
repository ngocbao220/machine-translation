import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable

def attention(q, k, v, mask=None, dropout=None):
    """Tính toán Attention score"""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output, scores

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False)
        if x.is_cuda: pe.cuda()
        x = x + pe
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores, _ = attention(q, k, v, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(concat)

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        return self.linear_2(x)

class ReEncoderLayer(nn.Module):
    """Lớp Encoder đặc biệt thực hiện 2 lần Attention"""
    def __init__(self, d_model, heads, dropout=0.1, has_ff=True):
        super().__init__()
        self.attn1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, dropout=dropout) if has_ff else None
        if has_ff:
            self.norm3 = Norm(d_model)
            self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn1(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attn2(x2, x2, x2, mask))
        if self.ff is not None:
            x2 = self.norm3(x)
            x = x + self.dropout3(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        return x + self.dropout_3(self.ff(x2))

class ReEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N_enc=6, heads=8, dropout=0.1):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        # Cấu trúc chu kỳ [s-s-f]
        layers = [
            ReEncoderLayer(d_model, heads, dropout, has_ff=False),
            ReEncoderLayer(d_model, heads, dropout, has_ff=False),
            ReEncoderLayer(d_model, heads, dropout, has_ff=True),
        ] * (N_enc // 3)
        self.layers = nn.ModuleList(layers)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.pe(self.embed(src))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N_dec, heads, dropout):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N_dec)])
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.pe(self.embed(trg))
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
    
class ReTransformerTranslation(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, pad_idx=0, N_enc=6, N_dec=2, heads=8, dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = ReEncoder(src_vocab_size, d_model, N_enc, heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, N_dec, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)

    def make_src_mask(self, src):
        """Tạo mask cho Encoder để bỏ qua các token PAD"""
        # Shape: (batch_size, 1, 1, seq_len)
        src_mask = (src != self.pad_idx).unsqueeze(-2)
        return src_mask

    def make_trg_mask(self, trg):
        """Tạo mask cho Decoder (kết hợp PAD mask và Look-ahead mask)"""
        # 1. Mask cho token PAD
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(-2)
        
        # 2. Look-ahead mask (No-peak mask)
        size = trg.size(1)
        nopeak_mask = torch.triu(torch.ones((1, size, size), device=trg.device), diagonal=1).bool()
        nopeak_mask = ~nopeak_mask # Đảo ngược để phần tam giác dưới bằng True
        
        # Kết hợp cả hai
        trg_mask = trg_pad_mask & nopeak_mask
        return trg_mask

    def forward(self, src, trg):
        # Tự động tạo mask bên trong hàm forward
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output