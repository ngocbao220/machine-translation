#!/usr/bin/env python
# coding: utf-8

# # Sử dụng mô hình Transformer để giải quyết bài toán dịch máy

# ## Dependency

# In[ ]:


import os
import random
import pandas as pd
import sentencepiece as spm
import sacrebleu
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import zipfile
from tqdm import tqdm
import math
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# ## Configuration and Hyperparameters

# In[ ]:


# Data path
TRAIN_EN_PATH = "../data/IWSLT15/train.en.txt"
TRAIN_VI_PATH = "../data/IWSLT15/train.vi.txt"

TEST_EN_12_PATH = "../data/IWSLT15/tst2012.en.txt"
TEST_VI_12_PATH = "../data/IWSLT15/tst2012.vi.txt"

TEST_EN_13_PATH = "../data/IWSLT15/tst2013.en.txt"
TEST_VI_13_PATH = "../data/IWSLT15/tst2013.vi.txt"

SAVE_DIR = "./checkpoints"
SPM_EN_PREFIX = os.path.join(SAVE_DIR, "spm_en")
SPM_VI_PREFIX = os.path.join(SAVE_DIR, "spm_vi")

# Model hyperparameters
VOCAB_SIZE = 10000
SEED = 42
MAX_LEN=100
BATCH_SIZE=64

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device use: {DEVICE}")

# Set random seeds
os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ## Data loading and Preprocessing
# 
# ### Data is stored in a list

# In[ ]:


def readlines(path):
    """Read lines from a text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

#Loading training and test data
train_src = readlines(TRAIN_EN_PATH)
train_tgt = readlines(TRAIN_VI_PATH)

test_src = readlines(TEST_EN_12_PATH)
test_tgt = readlines(TEST_VI_12_PATH)

# test_src = readlines(TEST_EN_13_PATH)
# test_tgt = readlines(TEST_VI_13_PATH)

print(f"Training samples: {len(train_src)}")
print(f"Test sample: {len(test_src)}")
print(f"\nExample English sentence: {train_src[1]}")
print(f"Example Vietnamese sentence: {train_tgt[1]}")


# ## SentencePiece Tokenization
# Train BPE tokenizers for both English and Vietnamese

# In[ ]:


def train_spm(input_file, model_prefix, vocab_size=VOCAB_SIZE):
    """Train a SentencePiece BPE model"""
    args = (
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size}"
        "--model_type=bpe --character_coverage=1.0"
        "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    )

    spm.SentencePieceTrainer.Train(args)
    print(f"Trained SentencePiece model: {model_prefix}.model")

def load_sp(model_path):
    """Load a trained SentencePiece model"""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


# In[ ]:


# Train English tokenizer
tmp_en = os.path.join(SAVE_DIR, "tmp_en.txt")
if not os.path.exists(SPM_EN_PREFIX + ".model"):
    with open(tmp_en, 'w', encoding='utf-8') as f:
        for s in train_src:
            f.write(s + "\n")
    train_spm(tmp_en, SPM_EN_PREFIX)

# Train Vietnamese tokenizer
tmp_vi = os.path.join(SAVE_DIR, "tmp_vi.txt")
if not os.path.exists(SPM_VI_PREFIX + ".model"):
    with open(tmp_vi, 'w', encoding='utf-8') as f:
        for s in train_tgt:
            f.write(s + "\n")
    train_spm(tmp_vi, SPM_VI_PREFIX)

# Load tokenizers
sp_en = load_sp(SPM_EN_PREFIX + ".model")
sp_vi = load_sp(SPM_VI_PREFIX + ".model")

print(f"\nEnglish vocab size: {sp_en.GetPieceSize()}")
print(f"Vietnamese vocab size: {sp_vi.GetPieceSize()}")

# Test tokenization
test_sent = train_src[0]
tokens = sp_en.encode(test_sent)
sent = sp_en.decode(tokens)
print(f"\nExample tokenization:")
print(f"Original: {sent}")
print(f"Token IDs: {tokens[:20]}...")


# ## Dataset and Dataloader

# ### Dataset

# In[ ]:


class TranslationDataset(Dataset):
    """
    Dataset Parallel 
    """
    def __init__(self, src, tgt, sp_src, sp_tgt, max_len=MAX_LEN):
        self.src = src
        self.tgt = tgt
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):
        src_ids = [2] + self.sp_src.encode(self.src[index])[:self.max_len-2] + [3]
        tgt_ids = [2] + self.sp_tgt.encode(self.tgt[index])[:self.max_len-2] + [3]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


# ### Dataloader

# In[ ]:


def collate_fn(batch):
    """
    batch: list of (src_tensor, tgt_tensor)
    tgt_tensor: đã gồm <bos> ... <eos>
    """

    src_list, tgt_list = zip(*batch)

    # tính max length
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)

    batch_size = len(batch)

    # tạo tensor đã pad
    src_pad = torch.full((batch_size, max_src), 0, dtype=torch.long)
    tgt_in_pad = torch.full((batch_size, max_tgt - 1), 0, dtype=torch.long)
    tgt_out_pad = torch.full((batch_size, max_tgt - 1), 0, dtype=torch.long)

    for i, tgt in enumerate(tgt_list):
        src = src_list[i]

        # fill src padded
        src_pad[i, :len(src)] = src

        # tạo tgt_in = [bos, ... token[:-1]]
        tgt_in = tgt[:-1]
        tgt_in_pad[i, :len(tgt_in)] = tgt_in

        # tạo tgt_out = [... token[1:], eos]
        tgt_out = tgt[1:]
        tgt_out_pad[i, :len(tgt_out)] = tgt_out

    return src_pad, tgt_in_pad, tgt_out_pad


# In[ ]:


# Create training dataset and dataloader
# Split data: 90% train, 10% validation

total_samples = len(train_src)
train_size = int(0.9 * total_samples)

train_src_split = train_src[:train_size]
train_tgt_split = train_tgt[:train_size]
val_src = train_src[train_size:]
val_tgt = train_tgt[train_size:]

print(f"Total samples: {total_samples}")
print(f"Training size: {len(train_tgt_split)}")
print(f"Validation size: {len(val_src)}")

train_dataset = TranslationDataset(src=train_src_split, tgt=train_tgt_split, sp_src=sp_en, sp_tgt=sp_vi)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

val_dataset = TranslationDataset(src=val_src, tgt=val_tgt, sp_src=sp_en, sp_tgt=sp_vi)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# ### TransformerMT

# In[ ]:


class TransformerMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, pad_idx=0):
        super().__init__()
        self.model_type = "Transformer"
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.tgt_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.generator.bias.data.zero_()
        self.generator.weight.data.uniform_(-initrange, initrange)

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: input
        src_mask = ?
        src_key_padding_mask = ?
        """
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer.encoder(src_emb.transpose(0,1), src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.transformer.decoder(tgt_emb.transpose(0,1), memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return output

    def forward(self, src, tgt_in, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # src: (batch, src_len)
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        out = self.decode(tgt_in, memory,
                          tgt_mask=tgt_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=src_key_padding_mask)
        out = self.generator(out.transpose(0,1))
        return out  # (batch, tgt_len, vocab)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # compute positional encodings once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd dims: last column will be zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)
    
    
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz), device='cpu') * float('-inf'), diagonal=1)
    return mask


# In[ ]:


# ----------------------------
# Training/Validation loops
# ----------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    for src_batch, tgt_in_batch, tgt_out_batch in dataloader:
        src = src_batch.to(device)
        tgt_in = tgt_in_batch.to(device)
        tgt_out = tgt_out_batch.to(device)

        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt_in == 0)

        tgt_mask = generate_square_subsequent_mask(tgt_in.size(1)).to(device)

        optimizer.zero_grad()
        output = model(src, tgt_in, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask)
        # output: (batch, tgt_len, vocab)
        loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item() * src.size(0)
    return total_loss / len(dataloader.dataset)

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_in_batch, tgt_out_batch in dataloader:
            src = src_batch.to(device)
            tgt_in = tgt_in_batch.to(device)
            tgt_out = tgt_out_batch.to(device)
            src_key_padding_mask = (src == 0)
            tgt_key_padding_mask = (tgt_in == 0)
            tgt_mask = generate_square_subsequent_mask(tgt_in.size(1)).to(device)
            output = model(src, tgt_in, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
            total_loss += loss.item() * src.size(0)
    return total_loss / len(dataloader.dataset)

# ----------------------------
# Early Stopping
# ----------------------------
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def step(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return False
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return False

# ----------------------------
# Save / Load
# ----------------------------
def save_checkpoint(path, model, optimizer, scheduler, epoch, best=False):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(state, path)
    if best:
        best_path = os.path.splitext(path)[0] + ".best.pt"
        torch.save(state, best_path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and ckpt.get('optim_state_dict'):
        optimizer.load_state_dict(ckpt['optim_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0)

# ----------------------------
# Inference (greedy)
# ----------------------------
def translate_sentence(model, src_tokens, src_vocab, tgt_vocab, device, max_len=100):
    model.eval()
    with torch.no_grad():
        src_ids = torch.tensor([src_vocab.encode(src_tokens)], dtype=torch.long).to(device)
        src_key_padding_mask = (src_ids == 0)
        memory = model.encode(src_ids, src_key_padding_mask=src_key_padding_mask)
        ys = torch.tensor([[2]], dtype=torch.long).to(device)
        for i in range(max_len):
            tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.decode(ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
            out = model.generator(out.transpose(0,1))
            prob = out[:, -1, :]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
            if next_word == 3:
                break
        decoded = tgt_vocab.decode(ys.squeeze(0).tolist())
        # remove BOS and everything after EOS
        if decoded and decoded[0] == 2:
            decoded = decoded[1:]
        if 3 in decoded:
            eos_idx = decoded.index(3)
            decoded = decoded[:eos_idx]
        return " ".join(decoded)


# ### Train

# In[ ]:


# ============================
# Config
# ============================
config = {
    "lr": 0.0003,
    "epochs": 20,
    "patience": 5,
    "run_name": "transformer_mt_run",
    "save_dir": "./checkpoints",
    "use_wandb": True,
}

# ============================
# Model setup
# ============================
model = TransformerMT(VOCAB_SIZE, VOCAB_SIZE)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

# ============================
# Wandb (optional)
# ============================
use_wandb = config["use_wandb"]

if use_wandb:
    import wandb
    wandb.init(project="mt-project", name=config["run_name"], config=config)
    wandb.watch(model, log="gradients", log_freq=100)

# ============================
# Early stopping
# ============================
early_stopper = EarlyStopping(patience=config["patience"])
best_val_loss = float("inf")

# ============================
# Training Loop
# ============================
for epoch in range(1, config["epochs"] + 1):
    start = time.time()

    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss = eval_epoch(model, val_loader, criterion, DEVICE)
    epoch_time = time.time() - start

    # LR scheduler step
    scheduler.step(val_loss)

    # Logging
    msg = (
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Time: {epoch_time:.1f}s"
    )
    print(msg)

    if use_wandb:
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
            "time": epoch_time
        })

    # Save latest checkpoint
    latest_path = os.path.join(config["save_dir"], f"{config['run_name']}.epoch{epoch}.pt")
    save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best=False)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = os.path.join(config["save_dir"], f"{config['run_name']}.best.pt")
        save_checkpoint(best_path, model, optimizer, scheduler, epoch, best=True)
        print(f"New best model saved to {best_path}")

    # Early stopping check
    if early_stopper.step(val_loss):
        print(f"Early stopping triggered at epoch {epoch}.")
        break


# ============================
# Save final model
# ============================
final_path = os.path.join(config["save_dir"], f"{config['run_name']}.final.pt")
save_checkpoint(final_path, model, optimizer, scheduler, epoch, best=False)
print("Training finished. Final model saved to", final_path)

if use_wandb:
    wandb.finish()

# ============================
# Quick inference demo
# ============================
for _ in range(10):
    if len(val_src) > 0:
        i = random.randrange(len(val_src))
        src_example = val_src[i]
        tgt_example = val_tgt[i]

        print("Example src:", " ".join(src_example))
        print("Ref tgt   :", " ".join(tgt_example))

        pred = translate_sentence(model, src_example, src_vocab=sp_en, tgt_vocab=sp_vi, device=DEVICE)
        print("Pred tgt  :", pred)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script transformerMT.ipynb')


# In[ ]:




