"""
Tiny LLM from scratch (single file, CPU/GPU) — character-level Transformer
--------------------------------------------------------------------------
Goals
- Minimal, readable implementation of a causal Transformer LM
- Byte-level tokenizer (0–255) to avoid external deps
- Train on any text file; works on CPU, faster on GPU if available
- Good base to extend with BPE, rotary pos-emb, FlashAttention, etc.

Usage (example)
- Put a text file next to this script, e.g. shakespeare.txt
- python tiny_llm.py --data shakespeare.txt --device cuda --steps 2000
- Generate: python tiny_llm.py --generate --ckpt ckpt.pt --device cuda --max-new-tokens 400 --prompt "To be or not to be"

Notes
- This is intentionally compact and pedagogical. No PyTorch Lightning, no HF.
- For clarity, we avoid clever micro-opts.
"""

import math
import os
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Tokenizer ------------------------------------
class ByteTokenizer:
    """Byte-level tokenizer: maps text <-> ints in [0,255]."""
    def __init__(self):
        self.vocab_size = 256
    def encode(self, s: str):
        return list(s.encode('utf-8', errors='ignore'))
    def decode(self, ids):
        return bytes(ids).decode('utf-8', errors='ignore')

# ----------------------------- Data -----------------------------------------
class CharDataset(Dataset):
    def __init__(self, data_bytes: torch.Tensor, block_size: int):
        assert data_bytes.dim() == 1
        self.data = data_bytes
        self.block_size = block_size
    def __len__(self):
        return self.data.numel() - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y

# ----------------------------- Model ----------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # Causal mask: [1, 1, T, T]
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):  # x: (B,T,C)
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)
        # reshape to heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,nh,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,nh,T,T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B,nh,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.fc(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T) of token ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def forward(self, idx):  # idx: (B,T)
        B, T = idx.size()
        tok = self.tok_emb(idx)  # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T,C)
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ----------------------------- Training -------------------------------------
@dataclass
class Config:
    data: str = ''
    steps: int = 1000
    batch_size: int = 64
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    lr: float = 3e-4
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt: str = 'ckpt.pt'
    seed: int = 1337
    eval_interval: int = 200
    eval_batches: int = 50
    generate: bool = False
    prompt: str = ''
    max_new_tokens: int = 300
    temperature: float = 1.0
    top_k: int | None = None


def load_text_bytes(path: str) -> torch.Tensor:
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            data = f.read()
    else:
        # tiny built-in corpus (Shakespeare-ish lines) so script runs out-of-box
        sample = (
            "To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune,\n"
            "Or to take arms against a sea of troubles\n"
            "And by opposing end them.\n"
        )
        data = sample.encode('utf-8')
    x = torch.tensor(list(data), dtype=torch.long)
    return x


def get_splits(data: torch.Tensor, split=0.9):
    n = data.numel()
    n_train = int(n * split)
    train = data[:n_train]
    val = data[n_train:]
    return train, val


def estimate_loss(model, loader, device, batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    tokenizer = ByteTokenizer()

    raw = load_text_bytes(cfg.data)
    train_bytes, val_bytes = get_splits(raw)

    train_ds = CharDataset(train_bytes, cfg.block_size)
    val_ds = CharDataset(val_bytes, cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    model = TinyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)

    best_val = float('inf')
    t0 = time.time()
    for step, (x, y) in enumerate(train_loader):
        if step >= cfg.steps:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # periodic eval
        if (step + 1) % cfg.eval_interval == 0 or step == 0:
            val_loss = estimate_loss(model, val_loader, device, cfg.eval_batches)
            dt = time.time() - t0
            print(f"step {step+1:5d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | elapsed {dt:.1f}s")
            t0 = time.time()
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, cfg.ckpt)
                print(f"Saved checkpoint to {cfg.ckpt}")

    # final save
    torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, cfg.ckpt)
    print(f"Training complete. Checkpoint saved to {cfg.ckpt}")


def do_generate(cfg: Config):
    device = torch.device(cfg.device)
    state = torch.load(cfg.ckpt, map_location=device)
    mcfg = state['cfg']
    model = TinyTransformerLM(
        vocab_size=256,
        block_size=mcfg['block_size'],
        n_layer=mcfg['n_layer'],
        n_head=mcfg['n_head'],
        n_embd=mcfg['n_embd'],
        dropout=mcfg['dropout'],
    ).to(device)
    model.load_state_dict(state['model'])
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(cfg.prompt)
    if not prompt_ids:
        prompt_ids = [ord('\n')]
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)
    text = tokenizer.decode(out[0].tolist())
    print(text)


# ----------------------------- CLI ------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='', help='Path to training text file (default: tiny builtin sample)')
    p.add_argument('--steps', type=int, default=1000, help='Training steps (batches)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--n-head', type=int, default=4)
    p.add_argument('--n-embd', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ckpt', type=str, default='ckpt.pt')
    p.add_argument('--eval-interval', type=int, default=200)
    p.add_argument('--eval-batches', type=int, default=50)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--generate', action='store_true', help='Generation mode')
    p.add_argument('--prompt', type=str, default='')
    p.add_argument('--max-new-tokens', type=int, default=300)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data=args.data,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        lr=args.lr,
        dropout=args.dropout,
        device=args.device,
        ckpt=args.ckpt,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        generate=args.generate,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    if cfg.generate:
        do_generate(cfg)
    else:
        train(cfg)


if __name__ == '__main__':
    main()
