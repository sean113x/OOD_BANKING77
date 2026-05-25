"""
Dataset utilities for BANKING77 OOD project.

Split files (dataset/banking77/preprocessed/):
  OOD_train.csv           — 7932 rows, 62 known intents  → training
  classification_test.csv — 2480 rows, 62 known intents  → in-distribution test
  OOD_test.csv            — 2671 rows, 15 held-out intents → near-OOD test
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

TRAIN_CSV = "dataset/banking77/preprocessed/OOD_train.csv"
ID_TEST   = "dataset/banking77/preprocessed/classification_test.csv"
OOD_TEST  = "dataset/banking77/preprocessed/OOD_test.csv"


def build_label_map(train_csv: str = TRAIN_CSV) -> tuple[dict, dict]:
    """Build label_text ↔ int id mapping from the 62 known intents."""
    df = pd.read_csv(train_csv)
    unique = sorted(df["label_text"].unique())
    label2id = {name: i for i, name in enumerate(unique)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


class Banking77Dataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: PreTrainedTokenizerFast, label2id: dict, max_length: int = 64):
        df = pd.read_csv(csv_path)
        self.texts  = df["text"].tolist()
        # OOD_test has held-out labels not in label2id → -1
        self.labels = [label2id.get(lt, -1) for lt in df["label_text"].tolist()]
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }
