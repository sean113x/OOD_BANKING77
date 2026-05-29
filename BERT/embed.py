"""Generate preprocessed dataset embeddings with the MiniLM LoRA adapter."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from BERT.embedding_utils import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL_NAME,
    encode_texts,
    load_embedding_model,
)


DEFAULT_DATA_DIR = Path("dataset/preprocessed")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "embedded"
SPLITS = {
    "OOD_train": "OOD_train.csv",
    "classification_test": "classification_test.csv",
    "OOD_validation": "OOD_validation.csv",
    "OOD_test_eval": "OOD_test_eval.csv",
    "OOD_test": "OOD_test.csv",
}


def read_csv(path: Path) -> dict[str, np.ndarray | list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [row["text"] for row in rows]
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
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
    model_dir: Path,
    base_model_name: str | None,
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
        model_dir=np.array(str(model_dir)),
        base_model_name=np.array(base_model_name or ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base-model-name", default=DEFAULT_BASE_MODEL_NAME)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model, device = load_embedding_model(args.model_dir, args.base_model_name, args.device)
    print(f"Using model: {args.model_dir}")
    print(f"Using base model: {args.base_model_name}")
    print(f"Using device: {device}")

    for split_name, filename in SPLITS.items():
        input_path = args.data_dir / filename
        output_path = args.output_dir / f"{split_name}_embeddings.npz"
        data = read_csv(input_path)
        embeddings = encode_texts(
            tokenizer,
            model,
            data["texts"],
            device,
            batch_size=args.batch_size,
            show_progress=True,
        )
        write_embeddings(output_path, embeddings, data, args.model_dir, args.base_model_name)
        print(f"Wrote {output_path} with shape {embeddings.shape}")


if __name__ == "__main__":
    main()
