%%writefile train_mt.py
"""
train_mt.py

Simple Machine Translation training with PyTorch Transformer.
Features:
 - EarlyStopping (monitor val loss)
 - Logging train/val loss to console and Weights & Biases (wandb)
 - LR scheduler (ReduceLROnPlateau) that reduces LR when val loss plateaus
 - Checkpoint saving (best and latest)
 - Example inference function
 - Usage instructions in docstring / argparse
"""

import os
import math
import argparse
from typing import List, Tuple
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optional: pip install sentencepiece if you want subword tokenization
# import sentencepiece as spm

# ----------------------------
# Utilities: dataset & tokenization
# ----------------------------
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

class SimpleVocab:
    """A tiny vocab utility. Replace with SentencePiece for real use."""
    def __init__(self, tokens=None, min_freq=1):
        self.freq = {}
        self.itos = []
        self.stoi = {}  
        if tokens:
            for t in tokens:
                self.add_token(t)
        # ensure special tokens
        for t in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            self.add_token(t)

    def add_token(self, token):
        if token in self.freq:
            self.freq[token] += 1
        else:
            self.freq[token] = 1
        # will rebuild later

    def build_from_corpus(self, corpus_tokens, max_size=None, min_freq=1):
        for toklist in corpus_tokens:
            for t in toklist:
                self.freq[t] = self.freq.get(t, 0) + 1
        items = [t for t, f in sorted(self.freq.items(), key=lambda x: (-x[1], x[0])) if f >= min_freq]
        # ensure special tokens at front
        specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        final = []
        for s in specials:
            if s in items:
                items.remove(s)
            final.append(s)
        if max_size:
            items = items[: max_size - len(final)]
        final.extend(items)
        self.itos = final
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi[UNK_TOKEN]) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if i < len(self.itos) else UNK_TOKEN for i in ids]

    def __len__(self):
        return len(self.itos)

class ParallelDataset(Dataset):
    """
    Expect simple tsv file with source \t target per line OR lists of token lists.
    For speed, use prepared tokenized data (list of tokens).
    """
    def __init__(self, src_sentences: List[List[str]], tgt_sentences: List[List[str]]):
        assert len(src_sentences) == len(tgt_sentences)
        self.src = src_sentences
        self.tgt = tgt_sentences

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def collate_fn(batch, src_vocab: SimpleVocab, tgt_vocab: SimpleVocab, max_len=256):
    src_batch, tgt_batch = zip(*batch)
    # add BOS/EOS for targets
    src_ids = []
    tgt_input = []
    tgt_output = []
    for s, t in zip(src_batch, tgt_batch):
        s_ids = src_vocab.encode(s)[:max_len]
        t_ids = tgt_vocab.encode(t)[:max_len-2]
        # target input begins with BOS, output ends with EOS
        tgt_in = [tgt_vocab.stoi[BOS_TOKEN]] + t_ids
        tgt_out = t_ids + [tgt_vocab.stoi[EOS_TOKEN]]
        src_ids.append(torch.tensor(s_ids, dtype=torch.long))
        tgt_input.append(torch.tensor(tgt_in, dtype=torch.long))
        tgt_output.append(torch.tensor(tgt_out, dtype=torch.long))
    # pad
    src_padded = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_vocab.stoi[PAD_TOKEN])
    tgt_in_padded = nn.utils.rnn.pad_sequence(tgt_input, batch_first=True, padding_value=tgt_vocab.stoi[PAD_TOKEN])
    tgt_out_padded = nn.utils.rnn.pad_sequence(tgt_output, batch_first=True, padding_value=tgt_vocab.stoi[PAD_TOKEN])
    return src_padded, tgt_in_padded, tgt_out_padded

# ----------------------------
# Model: Transformer wrapper
# ----------------------------
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

# ----------------------------
# Masks
# ----------------------------
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz), device='cpu') * float('-inf'), diagonal=1)
    return mask

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

        src_key_padding_mask = (src == src_vocab.stoi[PAD_TOKEN])
        tgt_key_padding_mask = (tgt_in == tgt_vocab.stoi[PAD_TOKEN])

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
            src_key_padding_mask = (src == src_vocab.stoi[PAD_TOKEN])
            tgt_key_padding_mask = (tgt_in == tgt_vocab.stoi[PAD_TOKEN])
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
def translate_sentence(model, src_tokens: List[str], src_vocab: SimpleVocab, tgt_vocab: SimpleVocab, device, max_len=100):
    model.eval()
    with torch.no_grad():
        src_ids = torch.tensor([src_vocab.encode(src_tokens)], dtype=torch.long).to(device)
        src_key_padding_mask = (src_ids == src_vocab.stoi[PAD_TOKEN])
        memory = model.encode(src_ids, src_key_padding_mask=src_key_padding_mask)
        ys = torch.tensor([[tgt_vocab.stoi[BOS_TOKEN]]], dtype=torch.long).to(device)
        for i in range(max_len):
            tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.decode(ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
            out = model.generator(out.transpose(0,1))
            prob = out[:, -1, :]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
            if next_word == tgt_vocab.stoi[EOS_TOKEN]:
                break
        decoded = tgt_vocab.decode(ys.squeeze(0).tolist())
        # remove BOS and everything after EOS
        if decoded and decoded[0] == BOS_TOKEN:
            decoded = decoded[1:]
        if EOS_TOKEN in decoded:
            eos_idx = decoded.index(EOS_TOKEN)
            decoded = decoded[:eos_idx]
        return " ".join(decoded)

# ----------------------------
# Main training harness / CLI
# ----------------------------
def read_parallel_file(path, max_lines=None, split_on="\t"):
    srcs = []
    tgts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(split_on)
            if len(parts) < 2:
                continue
            src, tgt = parts[0].split(), parts[1].split()
            srcs.append(src)
            tgts.append(tgt)
    return srcs, tgts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train simple Transformer MT with PyTorch")
    parser.add_argument("--train", type=str, required=True, help="train file tsv: src \\t tgt")
    parser.add_argument("--val", type=str, required=True, help="val file tsv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5, help="early stop patience on val loss")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default="mt_run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vocab", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading data...")
    train_src, train_tgt = read_parallel_file(args.train)
    val_src, val_tgt = read_parallel_file(args.val)

    # build vocabs (you will likely want SentencePiece for real models)
    print("Building vocabs...")
    src_vocab = SimpleVocab()
    tgt_vocab = SimpleVocab()
    src_vocab.build_from_corpus(train_src + val_src, max_size=args.max_vocab)
    tgt_vocab.build_from_corpus(train_tgt + val_tgt, max_size=args.max_vocab)

    # create datasets & dataloaders
    train_dataset = ParallelDataset(train_src, train_tgt)
    val_dataset = ParallelDataset(val_src, val_tgt)

    def collate_wrapper(batch):
        return collate_fn(batch, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper)

    device = torch.device(args.device)
    print("Device:", device)

    print("Building model...")
    model = TransformerMT(len(src_vocab), len(tgt_vocab), d_model=args.d_model, pad_idx=src_vocab.stoi[PAD_TOKEN])
    model = model.to(device)

    # loss: ignore pad index
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[PAD_TOKEN])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Scheduler: reduce lr on plateau (monitor val loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Optional: integrate wandb if available
    try:
        import wandb
        use_wandb = False
    except Exception:
        use_wandb = False

    if use_wandb:
        wandb.init(project="mt-project", name=args.run_name, config=vars(args))
        wandb.watch(model, log="gradients", log_freq=100)

    early_stopper = EarlyStopping(patience=args.patience)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - start

        # scheduler step (for ReduceLROnPlateau)
        scheduler.step(val_loss)

        # logging
        msg = f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s"
        print(msg)
        if use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch, "time": epoch_time})

        # save latest
        latest_path = os.path.join(args.save_dir, f"{args.run_name}.epoch{epoch}.pt")
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best=False)

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, f"{args.run_name}.best.pt")
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best=True)
            print(f"New best model saved to {best_path}")

        # early stopping check
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # final: save final model
    final_path = os.path.join(args.save_dir, f"{args.run_name}.final.pt")
    save_checkpoint(final_path, model, optimizer, scheduler, epoch, best=False)
    print("Training finished. Final model saved to", final_path)

    if use_wandb:
        wandb.finish()

    # quick inference demo (pick random val sample)
    for j in range(10):
      if len(val_src) > 0:
          i = random.randrange(len(val_src))
          src_example = val_src[i]
          tgt_example = val_tgt[i]
          print("Example src:", " ".join(src_example))
          print("Ref tgt   :", " ".join(tgt_example))
          pred = translate_sentence(model, src_example, src_vocab, tgt_vocab, device)
          print("Pred tgt  :", pred)