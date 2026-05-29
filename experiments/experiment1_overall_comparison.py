"""Experiment 1: overall OOD method comparison.

Each model is trained on ``OOD_train``, calibrated on ``OOD_validation``, and
evaluated on ``OOD_test_eval``. The output is one CSV file.

Usage:
  python experiments/experiment1_overall_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import (
    RESULT_DIR,
    all_model_specs,
    calibrate_threshold,
    evaluate_split_rows,
    load_or_fit_model,
    load_standard_splits,
    print_metric_table,
    score_model,
    write_csv,
)


DEFAULT_OUTPUT = RESULT_DIR / "experiment1_overall_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def run(output_path=DEFAULT_OUTPUT, force_train: bool = False):
    train, _, validation, eval_split = load_standard_splits()
    rows = []

    for spec in all_model_specs():
        model, loaded, fit_seconds = load_or_fit_model(spec, train, force_train=force_train)
        threshold, validation_scores, _ = calibrate_threshold(
            spec,
            model,
            validation,
            strategy="validation_best_f1",
        )
        eval_scores, _, score_seconds = score_model(spec, model, eval_split["embeddings"])
        source = "loaded" if loaded else "trained"
        extra = {
            "model_source": source,
            "model_key": spec.key,
        }

        rows.extend(
            evaluate_split_rows(
                experiment="experiment1_overall_comparison",
                family=spec.family_name,
                model=spec.display_name,
                score=spec.score_name,
                hyperparameters=spec.hyperparameters,
                threshold_source="validation_best_f1",
                data_split="validation",
                split=validation,
                scores=validation_scores,
                threshold=threshold,
                fit_seconds=fit_seconds,
                extra=extra,
            )
        )
        rows.extend(
            evaluate_split_rows(
                experiment="experiment1_overall_comparison",
                family=spec.family_name,
                model=spec.display_name,
                score=spec.score_name,
                hyperparameters=spec.hyperparameters,
                threshold_source="validation_best_f1",
                data_split="eval",
                split=eval_split,
                scores=eval_scores,
                threshold=threshold,
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
                extra=extra,
            )
        )

    output = write_csv(output_path, rows)
    print_metric_table(rows, "Experiment 1 metrics (eval / Overall)")
    return output


def main() -> None:
    args = parse_args()
    output = run(args.output, force_train=args.force_train)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
