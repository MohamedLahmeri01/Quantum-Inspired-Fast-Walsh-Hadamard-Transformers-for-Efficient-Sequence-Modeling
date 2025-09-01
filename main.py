# experiment_realbench_quantum_vs_vanilla.py
import os
import math
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# -------------------------
# Utilities
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# -------------------------
# Fast Walsh-Hadamard Transform (FWHT) utils
# -------------------------
def next_power_of_two(n: int) -> int:
    return 1 << ((n - 1).bit_length())

def fwht_real(x: torch.Tensor) -> torch.Tensor:
    """
    In-place-ish normalized FWHT implementation for last dim.
    x shape (..., n) where n is power of two.
    Returns normalized transform Hx / sqrt(n). FWHT is its own inverse with this normalization.
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    assert (n & (n - 1)) == 0, "FWHT requires power-of-two length"
    x = x.view(-1, n)
    h = 1
    while h < n:
        # pairwise butterfly
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        x = x.view(-1, n)
        h *= 2
    x = x.view(*orig_shape)
    return x / math.sqrt(n)

# -------------------------
# Quantum-inspired attention (resource-efficient)
# -------------------------
class QuantumAttention(nn.Module):
    """
    FWHT-based quantum-inspired attention:
    - projects to heads
    - applies FWHT on head-dim (O(d log d))
    - learns a small diag scaling in FWHT domain and gating scalar
    - real-valued (no complex dtype) and efficient
    """
    def __init__(self, embed_dim:int, num_heads:int, dropout:float=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # padded dimension for FWHT
        self.padded_dim = next_power_of_two(self.head_dim)
        # small learned diagonal in FWHT domain
        self.diag = nn.Parameter(torch.ones(self.padded_dim) * 0.05)
        # gating scalar
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # small bias
        self.bias = nn.Parameter(torch.zeros(self.head_dim))

        self.last_attn = None

    def _pad(self, t: torch.Tensor) -> torch.Tensor:
        # t: [..., d]
        d = t.shape[-1]
        if d == self.padded_dim:
            return t
        pad = self.padded_dim - d
        return F.pad(t, (0, pad), "constant", 0.0)

    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, heads, seq_len, head_dim]
        returns: same shape with transformed features added (gated residual)
        """
        b,h,seq,d = x.shape
        # flatten
        x_flat = x.reshape(-1, d)  # [b*h*seq, d]
        # pad to padded_dim
        if d != self.padded_dim:
            x_flat = F.pad(x_flat, (0, self.padded_dim - d))
        # fwht
        x_f = fwht_real(x_flat)
        # diag scaling
        x_f = x_f * self.diag.view(1, -1)
        # inverse fwht (same op)
        x_inv = fwht_real(x_f)
        # crop
        x_inv = x_inv[:, :d].view(b,h,seq,d)
        # gated residual & bias
        return x + self.alpha * x_inv + self.bias.view(1,1,1,-1)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        x: [batch, seq_len, embed_dim]
        attention_mask: [batch, seq_len] (1 for keep, 0 for pad) or None
        """
        b, seq, _ = x.shape
        q,k,v = torch.chunk(self.qkv(x), 3, dim=-1)
        # to heads [b, heads, seq, head_dim]
        def to_heads(t):
            return t.view(b, seq, self.num_heads, self.head_dim).transpose(1,2)
        qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)

        # apply transform
        qh_sup = self.apply_transform(qh)
        kh_sup = self.apply_transform(kh)

        # compute scores
        scores = torch.einsum('bhqd,bhkd->bhqk', qh_sup, kh_sup) / math.sqrt(self.head_dim)

        # mask
        if attention_mask is not None:
            # attention_mask: [b, seq] where 1 means token present
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [b,1,1,seq]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        # save for viz
        try:
            self.last_attn = weights.detach().cpu()
        except Exception:
            self.last_attn = None

        out = torch.einsum('bhqk,bhkd->bhqd', weights, vh)
        out = out.transpose(1,2).contiguous().view(b, seq, -1)
        return self.out(out)

# -------------------------
# Vanilla attention block (MHA)
# -------------------------
class VanillaAttention(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout:float=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.last_attn = None

    def forward(self, x:torch.Tensor, attention_mask: torch.Tensor = None):
        # PyTorch MHA expects key_padding_mask: bool with True for positions that should be masked
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # [b, seq] True->mask
        out, attn_weights = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        # attn_weights shape: [b, heads, seq, seq] (if average_attn_weights False; else [b, seq, seq])
        if attn_weights is not None:
            self.last_attn = attn_weights.detach().cpu()
        else:
            self.last_attn = None
        return out

# -------------------------
# Transformer block + model wrappers
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, ff_dim:int, use_quantum:bool, dropout:float=0.1):
        super().__init__()
        self.use_quantum = use_quantum
        self.attn = QuantumAttention(embed_dim, num_heads, dropout) if use_quantum else VanillaAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x:torch.Tensor, attention_mask:torch.Tensor=None):
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, attention_mask)
        x = x + attn_out
        x = self.norm2(x)
        f = self.ff(x)
        return x + f

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, num_heads:int, num_layers:int, ff_dim:int, num_labels:int=None, use_quantum:bool=True, max_len:int=256, dropout:float=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim, use_quantum, dropout) for _ in range(num_layers)])
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_labels = num_labels
        if num_labels is not None:
            self.classifier = nn.Linear(embed_dim, num_labels)
            self.is_lm = False
        else:
            self.lm_head = nn.Linear(embed_dim, vocab_size)
            self.is_lm = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # input_ids: [b, seq]
        b, seq = input_ids.shape
        x = self.embed(input_ids) + self.pos[:, :seq, :].to(input_ids.device)
        for layer in self.layers:
            x = layer(x, attention_mask)
        if self.is_lm:
            return self.lm_head(x)
        else:
            # masked mean pooling
            if attention_mask is None:
                pooled = x.mean(dim=1)
            else:
                denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
            return self.classifier(pooled)

# -------------------------
# Data preparation helpers
# -------------------------
def prepare_classification_dataset(dataset_name:str, tokenizer, seq_len:int, split_subsample:int=None):
    """
    Returns train_dataset, val_dataset (HF Dataset objects tokenized with input_ids, attention_mask, label)
    """
    if dataset_name == "ag_news":
        ds = load_dataset("ag_news")
        train = ds["train"]
        val = ds["test"]
    elif dataset_name in ("sst2", "sst-2"):
        ds = load_dataset("glue", "sst2")
        train = ds["train"]
        val = ds["validation"]
    else:
        raise ValueError("Unsupported classification dataset")

    # optionally subsample
    if split_subsample and split_subsample > 0:
        train = train.shuffle(seed=42).select(range(min(len(train), split_subsample)))
        val = val.shuffle(seed=42).select(range(min(len(val), max(1000, split_subsample//10))))

    # tokenization function
    def tok_batch(batch):
        texts = batch["text"] if "text" in batch else batch.get("sentence", None)
        # datasets passes list of examples; we rely on HF tokenizer batching
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    # detect appropriate text key and map accordingly
    if "text" not in train.column_names and "sentence" in train.column_names:
        # rename sentence -> text for uniform handling
        train = train.rename_column("sentence", "text")
        val = val.rename_column("sentence", "text")

    train = train.map(tok_batch, batched=True, remove_columns=[c for c in train.column_names if c not in ("label",)])
    val = val.map(tok_batch, batched=True, remove_columns=[c for c in val.column_names if c not in ("label",)])
    return train, val

def prepare_lm_dataset(seq_len:int, tokenizer, split_subsample:int=None):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    # optional subsample
    train_texts = ds["train"]["text"]
    val_texts = ds["validation"]["text"]
    if split_subsample and split_subsample > 0:
        train_texts = train_texts[:split_subsample]
        val_texts = val_texts[:max(1000, split_subsample//10)]

    # tokenize then group into blocks
    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=False)
    tokenized = {}
    tokenized_train = tokenizer(train_texts, truncation=False)
    input_ids = tokenized_train["input_ids"]
    # concatenate and chunk
    all_ids = []
    for ids in input_ids:
        all_ids.extend(ids)
    # make blocks
    blocks = []
    for i in range(0, len(all_ids) - seq_len + 1, seq_len):
        blocks.append(all_ids[i:i+seq_len])
    # wrap into HF dataset like object? Simpler: return lists for loader
    train_inputs = torch.tensor(blocks, dtype=torch.long)
    # validation
    tokenized_val = tokenizer(val_texts, truncation=False)
    all_ids_val = []
    for ids in tokenized_val["input_ids"]:
        all_ids_val.extend(ids)
    blocks_val = []
    for i in range(0, len(all_ids_val) - seq_len + 1, seq_len):
        blocks_val.append(all_ids_val[i:i+seq_len])
    val_inputs = torch.tensor(blocks_val, dtype=torch.long)
    return train_inputs, val_inputs

# -------------------------
# Collate & DataLoader helpers
# -------------------------
def collate_classification(batch):
    # batch items are dict-like from datasets with keys input_ids, attention_mask, label
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return input_ids, attention_mask, labels

def collate_lm_blocks(batch):
    # batch is a tensor of shape [bs, seq_len]
    input_ids = torch.stack(batch)
    labels = input_ids.clone()
    return input_ids, labels

# -------------------------
# Training & evaluation loops
# -------------------------
def train_classification_one_epoch(model, dataloader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_classification(model, dataloader, device, loss_fn):
    model.eval()
    preds = []
    trues = []
    losses = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            losses += loss.item() * input_ids.size(0)
            pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy().tolist())
    avg_loss = losses / len(dataloader.dataset)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="weighted")
    cm = confusion_matrix(trues, preds)
    return avg_loss, acc, f1, cm

def train_lm_one_epoch(model, dataloader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    tokens = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)  # [b, seq, vocab]
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_lm(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# -------------------------
# Visualization helpers
# -------------------------
def plot_training_curves(history: Dict, outpath: Path, show_acc=True):
    ensure_dir(outpath.parent)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(8,4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    if show_acc and "val_acc" in history:
        plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.legend(); plt.grid(True)
    plt.title(outpath.stem)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_confusion_matrix(cm, labels, outpath: Path):
    ensure_dir(outpath.parent)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_attention_heatmap(attn_tensor, tokens: List[str], outpath: Path, head:int=0):
    # attn_tensor: [b, heads, q, k]
    ensure_dir(outpath.parent)
    if attn_tensor is None:
        return
    att = attn_tensor[0, head]
    plt.figure(figsize=(6,6))
    sns.heatmap(att.numpy(), xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title("Attention heatmap")
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_attention_entropy(attn_tensor, outpath: Path):
    ensure_dir(outpath.parent)
    if attn_tensor is None:
        return
    p = attn_tensor.numpy()
    p = np.clip(p, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=-1)  # [b, heads, q]
    mean_ent = ent.mean(axis=(0,2))
    plt.figure(figsize=(6,3))
    plt.bar(np.arange(len(mean_ent)), mean_ent)
    plt.xlabel("head"); plt.ylabel("entropy"); plt.title("Attention entropy per head")
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

# -------------------------
# High-level experiment runner
# -------------------------
def run_task_classification(task_name: str,
                            tokenizer_name: str = "distilbert-base-uncased",
                            model_config: Dict = None,
                            seeds: List[int] = [0,1,2],
                            max_samples_per_split: int = 2000,
                            device_name: str = "cuda"):
    """
    Runs classification on AG_NEWS or SST-2 for Quantum vs Vanilla, multiple seeds.
    Saves per-seed metrics and aggregated summary.
    """
    device = torch.device(device_name if torch.cuda.is_available() and device_name=="cuda" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    train_ds, val_ds = prepare_classification_dataset(task_name, tokenizer, model_config["seq_len"], split_subsample=max_samples_per_split)

    # Build HF-style lists to feed DataLoader collate
    def to_dataloader(split_ds, batch_size, shuffle=False):
        return DataLoader(split_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_classification)

    results = {}
    labels_map = None
    # determine number of labels
    sample_label = train_ds[0]["label"]
    num_labels = len(set(train_ds["label"])) if hasattr(train_ds, "__getitem__") else 2

    for model_type in ["quantum", "vanilla"]:
        print(f"\n=== Running {task_name} : {model_type} ===")
        per_seed_metrics = []
        for seed in seeds:
            set_seed(seed)
            outdir = Path("experiments") / task_name / model_type / f"seed_{seed}"
            if outdir.exists():
                # optional: clear
                pass
            ensure_dir(outdir)

            # create model
            model = SimpleTransformer(
                vocab_size=tokenizer.vocab_size,
                embed_dim=model_config["embed_dim"],
                num_heads=model_config["num_heads"],
                num_layers=model_config["num_layers"],
                ff_dim=model_config["ff_dim"],
                num_labels=num_labels,
                use_quantum=(model_type=="quantum"),
                max_len=model_config["seq_len"],
                dropout=model_config.get("dropout", 0.1)
            ).to(device)

            # dataloaders
            train_loader = to_dataloader(train_ds, batch_size=model_config["batch_size"], shuffle=True)
            val_loader = to_dataloader(val_ds, batch_size=model_config["batch_size"], shuffle=False)

            optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["lr"], weight_decay=1e-2)
            loss_fn = nn.CrossEntropyLoss()
            history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

            best_val_acc = -1.0
            patience = model_config.get("patience", 2)
            wait = 0

            for epoch in range(1, model_config["epochs"]+1):
                train_loss = train_classification_one_epoch(model, train_loader, optimizer, device, loss_fn)
                val_loss, val_acc, val_f1, cm = evaluate_classification(model, val_loader, device, loss_fn)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)

                print(f"[{model_type} seed{seed}] Epoch {epoch}/{model_config['epochs']} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

                # save checkpoint + history
                torch.save(model.state_dict(), outdir / f"model_epoch{epoch}.pt")
                save_json(history, outdir / "history.json")

                # early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    wait = 0
                    # save best
                    torch.save(model.state_dict(), outdir / "best.pt")
                    save_json(history, outdir / "best_history.json")
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping")
                        break

            # evaluate final best
            model.load_state_dict(torch.load(outdir / "best.pt", map_location=device))
            _, val_acc, val_f1, cm = evaluate_classification(model, val_loader, device, loss_fn)
            per_seed_metrics.append({"seed": seed, "val_acc": val_acc, "val_f1": val_f1, "confusion_matrix": cm.tolist()})

            # plots & viz
            plot_training_curves(history, outdir / "training.png", show_acc=True)
            save_confusion_matrix(cm, [str(i) for i in range(num_labels)], outdir / "confusion.png")
            # attention viz: pick first layer's module if available
            try:
                layer0 = model.layers[0].attn
                attn_tensor = getattr(layer0, "last_attn", None)
                # get sample tokens for display (use first val example)
                input_ids = torch.tensor([val_ds[0]["input_ids"]], dtype=torch.long)
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
                plot_attention_heatmap(attn_tensor, tokens, outdir / "attn_layer0_head0.png", head=0)
                plot_attention_entropy(attn_tensor, outdir / "attn_entropy.png")
            except Exception as e:
                # skip if not available
                pass

        # aggregate seeds
        accs = [m["val_acc"] for m in per_seed_metrics]
        f1s = [m["val_f1"] for m in per_seed_metrics]
        summary = {
            "per_seed": per_seed_metrics,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs, ddof=1)) if len(accs)>1 else 0.0,
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s, ddof=1)) if len(f1s)>1 else 0.0
        }
        results[model_type] = summary

    # save summary
    out_summary = Path("experiments") / task_name / "summary.json"
    ensure_dir(out_summary.parent)
    save_json(results, out_summary)
    print(f"Done task {task_name}; summary saved to {out_summary}")
    return results

def run_task_lm(tokenizer_name: str = "distilbert-base-uncased",
                model_config: Dict = None,
                seeds: List[int] = [0,1,2],
                max_samples_per_split: int = 100000,
                device_name: str = "cuda"):
    device = torch.device(device_name if torch.cuda.is_available() and device_name=="cuda" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # prepare LM blocks
    train_blocks, val_blocks = prepare_lm_dataset(model_config["seq_len"], tokenizer, split_subsample=max_samples_per_split)
    # wrap into dataloaders
    train_loader = DataLoader(train_blocks, batch_size=model_config["batch_size"], shuffle=True, collate_fn=lambda x: (torch.stack(x), torch.stack(x)))
    val_loader = DataLoader(val_blocks, batch_size=model_config["batch_size"], shuffle=False, collate_fn=lambda x: (torch.stack(x), torch.stack(x)))

    results = {}
    for model_type in ["quantum", "vanilla"]:
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            outdir = Path("experiments") / "wikitext2" / model_type / f"seed_{seed}"
            ensure_dir(outdir)

            model = SimpleTransformer(
                vocab_size=tokenizer.vocab_size,
                embed_dim=model_config["embed_dim"],
                num_heads=model_config["num_heads"],
                num_layers=model_config["num_layers"],
                ff_dim=model_config["ff_dim"],
                num_labels=None,
                use_quantum=(model_type=="quantum"),
                max_len=model_config["seq_len"],
                dropout=model_config.get("dropout",0.1)
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["lr"], weight_decay=1e-2)
            loss_fn = nn.CrossEntropyLoss()

            history = {"train_loss":[],"val_ppl":[]}
            best_ppl = float("inf")
            wait = 0
            patience = model_config.get("patience", 2)

            for epoch in range(1, model_config["epochs"]+1):
                train_loss = train_lm_one_epoch(model, train_loader, optimizer, device, loss_fn)
                _, ppl = evaluate_lm(model, val_loader, device, loss_fn)
                history["train_loss"].append(train_loss)
                history["val_ppl"].append(ppl)
                print(f"[{model_type} seed{seed}] Epoch {epoch}/{model_config['epochs']} train_loss={train_loss:.4f} val_ppl={ppl:.4f}")
                save_json(history, outdir / "history.json")

                if ppl < best_ppl:
                    best_ppl = ppl
                    wait = 0
                    torch.save(model.state_dict(), outdir / "best.pt")
                else:
                    wait += 1
                    if wait >= patience:
                        break

            # evaluate best
            model.load_state_dict(torch.load(outdir / "best.pt", map_location=device))
            _, best_ppl = evaluate_lm(model, val_loader, device, loss_fn)
            per_seed.append({"seed": seed, "ppl": float(best_ppl)})
            plot_training_curves({"train_loss": history["train_loss"], "val_loss": history["val_ppl"]}, outdir / "training.png", show_acc=False)

        arr = np.array([x["ppl"] for x in per_seed])
        results[model_type] = {
            "per_seed": per_seed,
            "mean_ppl": float(arr.mean()),
            "std_ppl": float(arr.std(ddof=1)) if len(arr)>1 else 0.0
        }
    save_json(results, Path("experiments") / "wikitext2" / "summary.json")
    return results

# -------------------------
# Entrypoint / Config & Examples
# -------------------------
if __name__ == "__main__":
    # Configuration: adjust as needed
    model_cfg = {
        "seq_len": 128,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "ff_dim": 1024,
        "batch_size": 32,
        "epochs": 6,
        "lr": 3e-4,
        "dropout": 0.1,
        "patience": 2
    }

    # Use smaller subsample for quick CPU testing.
    # Set to 0 or None to use full dataset (GPU recommended).
    max_samples = 0  # set to None or 0 to use full dataset

    # pick device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Run AG_NEWS
    res1 = run_task_classification("ag_news", tokenizer_name="distilbert-base-uncased",
                                   model_config=model_cfg, seeds=[42,43,44],
                                   max_samples_per_split=(None if not max_samples else max_samples),
                                   device_name=device)
    print("AG_NEWS results:", res1)

    # Run SST-2
    res2 = run_task_classification("sst2", tokenizer_name="distilbert-base-uncased",
                                   model_config=model_cfg, seeds=[42,43,44],
                                   max_samples_per_split=(None if not max_samples else max_samples),
                                   device_name=device)
    print("SST-2 results:", res2)

    # Run LM (wikitext-2)
    lm_cfg = model_cfg.copy()
    lm_cfg.update({"seq_len": 128, "batch_size": 32, "epochs": 6})
    res3 = run_task_lm(tokenizer_name="distilbert-base-uncased", model_config=lm_cfg,
                       seeds=[42,43,44],
                       max_samples_per_split=(None if not max_samples else max_samples),
                       device_name=device)
    print("WikiText-2 results:", res3)

    print("All experiments finished. Artifacts saved under experiments/")
