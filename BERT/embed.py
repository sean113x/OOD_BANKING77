"""
Extract all-MiniLM-L6-v2 embeddings using raw transformers (no sentence-transformers).

Output (BERT/embeddings/):
  train_embs.npy       (7932, 384)
  train_labels.npy     (7932,)
  id_test_embs.npy     (2480, 384)
  id_test_labels.npy   (2480,)
  ood_test_embs.npy    (2671, 384)

Usage (from project root):
  python BERT/embed.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, "src")
from dataset import TRAIN_CSV, ID_TEST, OOD_TEST, build_label_map

OUT_DIR        = "BERT/embeddings"
MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED_DIR  = "BERT/models/minilm_lora"   # use fine-tuned if available
BATCH          = 64
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(model_output, attention_mask):
    token_embs = model_output[0]                          # (B, L, 384)
    mask = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    return torch.sum(token_embs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


@torch.no_grad()
def encode(tokenizer, model, texts: list[str]) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = model(**enc)
        embs = mean_pooling(out, enc["attention_mask"])
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().numpy())
        if (i // BATCH) % 10 == 0:
            print(f"  {i + len(batch)}/{len(texts)}", end="\r")
    print()
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def load_split(csv_path, label2id):
    df = pd.read_csv(csv_path)
    texts  = df["text"].tolist()
    labels = np.array([label2id.get(lt, -1) for lt in df["label_text"].tolist()])
    return texts, labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    label2id, _ = build_label_map(TRAIN_CSV)

    # Use fine-tuned LoRA model if available, otherwise fall back to base
    if os.path.isdir(FINETUNED_DIR):
        print(f"Loading fine-tuned model from {FINETUNED_DIR} ...")
        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
        model     = AutoModel.from_pretrained(FINETUNED_DIR).to(DEVICE)
    else:
        print(f"Loading base {MODEL_NAME} on {DEVICE} ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    for name, csv_path in [("train", TRAIN_CSV), ("id_test", ID_TEST), ("ood_test", OOD_TEST)]:
        print(f"Encoding {name} ...")
        texts, labels = load_split(csv_path, label2id)
        embs = encode(tokenizer, model, texts)
        np.save(f"{OUT_DIR}/{name}_embs.npy",   embs)
        np.save(f"{OUT_DIR}/{name}_labels.npy", labels)
        print(f"  {name}: {embs.shape}")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
