"""
Fine-tune all-MiniLM-L6-v2 with LoRA on 62 known BANKING77 intents.

Only LoRA adapter weights + classifier head are trained.
After training the fine-tuned encoder is saved for use in embed.py.

Usage (from project root):
  CUDA_VISIBLE_DEVICES=0 python BERT/finetune.py

Output:
  checkpoints/minilm_lora/   ← fine-tuned encoder weights
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "src")
from dataset import TRAIN_CSV, build_label_map

CKPT_DIR   = "checkpoints/minilm_lora"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS     = 30
BATCH      = 64
LR         = 3e-4
MAX_LEN    = 64
HIDDEN     = 384
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, label2id):
        df = pd.read_csv(csv_path)
        self.labels = [label2id[lt] for lt in df["label_text"].tolist()]
        self.enc = tokenizer(
            df["text"].tolist(),
            padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "label":          torch.tensor(self.labels[idx]),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    token_embs = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    return torch.sum(token_embs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class MiniLMLoRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = AutoModel.from_pretrained(MODEL_NAME)
        lora_cfg = LoraConfig(
            task_type=None,
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        self.encoder    = get_peft_model(base, lora_cfg)
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(HIDDEN, num_classes)

    def forward(self, input_ids, attention_mask):
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embs = mean_pooling(out, attention_mask)
        embs = F.normalize(embs, dim=-1)
        return self.classifier(self.dropout(embs)), embs

    def print_trainable(self):
        self.encoder.print_trainable_parameters()


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    label2id, _ = build_label_map(TRAIN_CSV)
    num_classes  = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = MiniLMLoRA(num_classes).to(DEVICE)
    model.print_trainable()

    loader = DataLoader(
        IntentDataset(TRAIN_CSV, tokenizer, label2id),
        batch_size=BATCH, shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01,
    )
    total_steps = len(loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if exists
    start_epoch = 1
    resume_path = os.path.join(CKPT_DIR, "trainer_state.pt")
    if os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = state["epoch"] + 1
        print(f"Resumed from epoch {state['epoch']} (acc={state['acc']:.4f})")

    if start_epoch > EPOCHS:
        print(f"Already trained {EPOCHS} epochs. Increase EPOCHS to continue.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(ids, mask)
            loss = criterion(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct    += (logits.argmax(-1) == lbls).sum().item()
            total      += lbls.size(0)

        acc = correct / total
        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(loader):.4f}  acc={acc:.4f}")

        torch.save({
            "epoch":           epoch,
            "acc":             acc,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, resume_path)

    # Save fine-tuned encoder (without classifier head)
    model.encoder.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)
    print(f"Saved → {CKPT_DIR}")


if __name__ == "__main__":
    train()
