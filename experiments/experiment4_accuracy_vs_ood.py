"""Experiment 4: classification accuracy vs OOD reliability.

Classifier-output OOD scores are calibrated on ``OOD_validation`` and reported
on both validation and ``OOD_test_eval``. Closed-set accuracy is measured on
the ID portion of ``OOD_test_eval``. The output is one CSV file.

Usage:
  python experiments/experiment4_accuracy_vs_ood.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from experiments.common import (
    RESULT_DIR,
    calibrate_threshold,
    classifier_specs,
    evaluate_split_rows,
    load_or_fit_model,
    load_standard_splits,
    print_metric_table,
    predict_probabilities,
    score_model,
    split_masks,
    write_csv,
)


DEFAULT_OUTPUT = RESULT_DIR / "experiment4_accuracy_vs_ood.csv"
CONFIDENCE_THRESHOLD = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def _classifier_key(model_key: str) -> str:
    return model_key.split("_", 1)[0]


def _id_classifier_metrics(model, eval_split) -> tuple[float, float]:
    ood_types = eval_split["ood_types"].astype(str)
    mask = ood_types == "id"
    pred = model.predict(eval_split["embeddings"][mask]).astype(str)
    labels = eval_split["labels"][mask].astype(str)
    return (
        float(accuracy_score(labels, pred)),
        float(f1_score(labels, pred, average="macro")),
    )


def _confidence_stats(model, split, metric_split: str) -> dict[str, float | int]:
    ood_types = split["ood_types"].astype(str)
    masks = split_masks(ood_types)
    mask = masks[metric_split]
    probs = predict_probabilities(model, split["embeddings"])
    confidence = probs.max(axis=1)
    id_values = confidence[mask & (ood_types == "id")]
    ood_values = confidence[mask & (ood_types != "id")]

    return {
        "msp_confidence_threshold": CONFIDENCE_THRESHOLD,
        "id_mean_confidence": float(np.mean(id_values)) if len(id_values) else float("nan"),
        "ood_mean_confidence": float(np.mean(ood_values)) if len(ood_values) else float("nan"),
        "ood_p90_confidence": float(np.percentile(ood_values, 90)) if len(ood_values) else float("nan"),
        "ood_high_conf_count": int(np.sum(ood_values >= CONFIDENCE_THRESHOLD)),
        "ood_high_conf_rate": float(np.mean(ood_values >= CONFIDENCE_THRESHOLD)) if len(ood_values) else float("nan"),
    }


def run(output_path: Path = DEFAULT_OUTPUT, force_train: bool = False) -> Path:
    train, _, validation, eval_split = load_standard_splits()
    rows = []
    id_metric_cache: dict[str, tuple[float, float]] = {}

    for spec in classifier_specs():
        model, loaded, fit_seconds = load_or_fit_model(spec, train, force_train=force_train)
        threshold, validation_scores, _ = calibrate_threshold(spec, model, validation, "validation_best_f1")
        eval_scores, _, score_seconds = score_model(spec, model, eval_split["embeddings"])

        clf_key = _classifier_key(spec.key)
        if clf_key not in id_metric_cache:
            id_metric_cache[clf_key] = _id_classifier_metrics(model, eval_split)
        id_accuracy, id_macro_f1 = id_metric_cache[clf_key]
        base_extra = {
            "model_key": spec.key,
            "classifier_key": clf_key,
            "model_source": "loaded" if loaded else "trained",
            "id_accuracy": id_accuracy,
            "id_macro_f1": id_macro_f1,
            "has_raw_decision_scores": hasattr(model, "decision_function"),
        }

        for data_split, split, scores, seconds in (
            ("validation", validation, validation_scores, None),
            ("eval", eval_split, eval_scores, score_seconds),
        ):
            split_rows = evaluate_split_rows(
                experiment="experiment4_accuracy_vs_ood",
                family=spec.family_name,
                model=spec.display_name,
                score=spec.score_name,
                hyperparameters=spec.hyperparameters,
                threshold_source="validation_best_f1",
                data_split=data_split,
                split=split,
                scores=scores,
                threshold=threshold,
                fit_seconds=fit_seconds,
                score_seconds=seconds,
                extra=base_extra,
            )
            for row in split_rows:
                row.update(_confidence_stats(model, split, row["metric_split"]))
            rows.extend(split_rows)

    output = write_csv(output_path, rows)
    print_metric_table(rows, "Experiment 4 metrics (eval / Overall)")
    return output


def main() -> None:
    args = parse_args()
    output = run(args.output, force_train=args.force_train)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
