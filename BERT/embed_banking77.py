"""Generate BANKING77 sentence embeddings with the local sentence-transformer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_DIR = Path("BERT/all-MiniLM-L6-v2")
DEFAULT_DATA_DIR = Path("dataset/preprocessed")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "embedded"
SPLITS = {
    "OOD_train": "OOD_train.csv",
    "classification_test": "classification_test.csv",
    "OOD_test": "OOD_test.csv",
}


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_csv(path: Path) -> dict[str, np.ndarray | list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [row["text"] for row in rows]
    labels = np.array([row["label"] for row in rows])
    label_texts = np.array([row["label_text"] for row in rows])
    is_ood = np.array([row.get("is_ood", "0") for row in rows])
    ood_types = np.array([row.get("ood_type", "id") for row in rows])
    sources = np.array([row.get("source", "banking77") for row in rows])
    return {
        "texts": texts,
        "labels": labels,
        "label_texts": label_texts,
        "is_ood": is_ood,
        "ood_types": ood_types,
        "sources": sources,
    }


def write_embeddings(
    path: Path,
    embeddings: np.ndarray,
    data: dict[str, np.ndarray | list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings.astype(np.float32),
        texts=np.array(data["texts"]),
        labels=data["labels"],
        label_texts=data["label_texts"],
        is_ood=data["is_ood"],
        ood_types=data["ood_types"],
        sources=data["sources"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device() if args.device == "auto" else args.device
    print(f"Using device: {device}")

    model = SentenceTransformer(str(args.model_dir), device=device)

    for split_name, filename in SPLITS.items():
        input_path = args.data_dir / filename
        output_path = args.output_dir / f"{split_name}_embeddings.npz"
        data = read_csv(input_path)
        embeddings = model.encode(
            data["texts"],
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        write_embeddings(output_path, embeddings, data)
        print(f"Wrote {output_path} with shape {embeddings.shape}")


if __name__ == "__main__":
    main()
