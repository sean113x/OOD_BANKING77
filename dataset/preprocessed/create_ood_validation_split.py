"""Create a stratified validation split from the combined OOD test CSV.

The source ``OOD_test.csv`` contains ID, near-OOD, and far-OOD rows. This
script extracts a reproducible validation set for hyperparameter selection and
writes the remaining rows as a separate final-evaluation split.

Stratification keys:
  - ID and near-OOD: ``ood_type`` + ``label_text``
  - far-OOD: ``ood_type`` + ``source_label_text`` because ``label_text`` is the
    shared value ``far_ood_trec`` for all TREC rows.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path("dataset/preprocessed/OOD_test.csv")
DEFAULT_VALIDATION = Path("dataset/preprocessed/OOD_validation.csv")
DEFAULT_EVAL = Path("dataset/preprocessed/OOD_test_eval.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        return reader.fieldnames, rows


def stratification_key(row: dict[str, str]) -> tuple[str, str]:
    ood_type = row["ood_type"]
    if ood_type == "far":
        return ood_type, row["source_label_text"]
    return ood_type, row["label_text"]


def validation_size(group_size: int, ratio: float) -> int:
    if group_size <= 1:
        return 0
    target = int(math.floor(group_size * ratio + 0.5))
    return max(1, min(group_size - 1, target))


def split_rows(
    rows: list[dict[str, str]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[stratification_key(row)].append(idx)

    validation_indices: set[int] = set()
    for indices in groups.values():
        n_validation = validation_size(len(indices), validation_ratio)
        validation_indices.update(rng.sample(indices, n_validation))

    validation_rows = [
        row for idx, row in enumerate(rows)
        if idx in validation_indices
    ]
    eval_rows = [
        row for idx, row in enumerate(rows)
        if idx not in validation_indices
    ]
    return validation_rows, eval_rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_by_ood_type(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row["ood_type"] for row in rows))


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")

    fieldnames, rows = read_rows(args.source)
    validation_rows, eval_rows = split_rows(rows, args.validation_ratio, args.seed)

    write_rows(args.validation_output, fieldnames, validation_rows)
    write_rows(args.eval_output, fieldnames, eval_rows)

    print(f"source: {args.source} ({len(rows)} rows)")
    print(f"validation: {args.validation_output} ({len(validation_rows)} rows)")
    print(f"validation ood_type counts: {count_by_ood_type(validation_rows)}")
    print(f"eval: {args.eval_output} ({len(eval_rows)} rows)")
    print(f"eval ood_type counts: {count_by_ood_type(eval_rows)}")


if __name__ == "__main__":
    main()
