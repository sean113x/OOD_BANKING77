"""Experiment 3: near-OOD difficulty analysis.

Near-OOD intents are grouped by their maximum centroid similarity to known
training intents. Each model is calibrated on ``OOD_validation`` and evaluated
on ID examples plus easy/medium/hard near-OOD groups from ``OOD_test_eval``.
The output is one CSV file.

Usage:
  python experiments/experiment3_near_ood_difficulty.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common import (
    RESULT_DIR,
    all_model_specs,
    calibrate_threshold,
    known_class_centroids,
    load_or_fit_model,
    load_standard_splits,
    metric_row,
    normalized_rows,
    print_metric_table,
    score_model,
    write_csv,
)


DEFAULT_OUTPUT = RESULT_DIR / "experiment3_near_ood_difficulty.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def difficulty_groups(train, eval_split):
    known_labels, known_centroids = known_class_centroids(train)
    eval_embeddings = normalized_rows(eval_split["embeddings"])
    eval_labels = eval_split["label_texts"].astype(str)
    ood_types = eval_split["ood_types"].astype(str)
    near_labels = sorted(set(eval_labels[ood_types == "near"].tolist()))

    intent_rows = []
    for label in near_labels:
        mask = (ood_types == "near") & (eval_labels == label)
        centroid = normalized_rows(eval_embeddings[mask].mean(axis=0, keepdims=True))
        similarities = centroid @ known_centroids.T
        nearest_idx = int(np.argmax(similarities[0]))
        intent_rows.append(
            {
                "label_text": label,
                "max_known_similarity": float(similarities[0, nearest_idx]),
                "nearest_known_intent": str(known_labels[nearest_idx]),
                "n_near_samples": int(np.sum(mask)),
            }
        )

    intent_rows.sort(key=lambda row: row["max_known_similarity"])
    buckets = np.array_split(np.array(intent_rows, dtype=object), 3)
    names = ("easy", "medium", "hard")
    groups = {}
    for name, bucket in zip(names, buckets):
        rows = [dict(row) for row in bucket.tolist()]
        labels = {row["label_text"] for row in rows}
        similarities = np.array([row["max_known_similarity"] for row in rows], dtype=float)
        groups[name] = {
            "labels": labels,
            "intent_rows": rows,
            "similarity_min": float(np.min(similarities)),
            "similarity_mean": float(np.mean(similarities)),
            "similarity_max": float(np.max(similarities)),
        }
    return groups


def run(output_path: Path = DEFAULT_OUTPUT, force_train: bool = False) -> Path:
    train, _, validation, eval_split = load_standard_splits()
    groups = difficulty_groups(train, eval_split)
    eval_ood_types = eval_split["ood_types"].astype(str)
    eval_labels = eval_split["label_texts"].astype(str)
    rows = []

    for spec in all_model_specs():
        model, loaded, fit_seconds = load_or_fit_model(spec, train, force_train=force_train)
        threshold, _, _ = calibrate_threshold(spec, model, validation, strategy="validation_best_f1")
        eval_scores, _, score_seconds = score_model(spec, model, eval_split["embeddings"])

        for difficulty, info in groups.items():
            mask = (eval_ood_types == "id") | (
                (eval_ood_types == "near") & np.isin(eval_labels, list(info["labels"]))
            )
            y_true = (eval_ood_types[mask] != "id").astype(int)
            scores = eval_scores[mask]
            intent_names = sorted(info["labels"])
            extra = {
                "model_key": spec.key,
                "model_source": "loaded" if loaded else "trained",
                "difficulty": difficulty,
                "near_intent_count": len(intent_names),
                "near_intents": ";".join(intent_names),
                "similarity_min": info["similarity_min"],
                "similarity_mean": info["similarity_mean"],
                "similarity_max": info["similarity_max"],
            }
            rows.append(
                metric_row(
                    experiment="experiment3_near_ood_difficulty",
                    family=spec.family_name,
                    model=spec.display_name,
                    score=spec.score_name,
                    hyperparameters=spec.hyperparameters,
                    threshold_source="validation_best_f1",
                    data_split="eval",
                    metric_split=f"Near-OOD-{difficulty}",
                    y_true=y_true,
                    scores=scores,
                    threshold=threshold,
                    fit_seconds=fit_seconds,
                    score_seconds=score_seconds,
                    extra=extra,
                )
            )

    output = write_csv(output_path, rows)
    print_metric_table(
        rows,
        "Experiment 3 metrics (eval / near-OOD difficulty groups)",
        metric_split=None,
        max_rows=120,
    )
    return output


def main() -> None:
    args = parse_args()
    output = run(args.output, force_train=args.force_train)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
