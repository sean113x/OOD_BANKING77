"""Experiment 3: Near-OOD ratio analysis.

Each model is calibrated on ``OOD_validation`` and evaluated on a fixed ID set
plus an OOD pool whose Near-OOD ratio is swept from 0% to 100%. The total OOD
sample count is held constant so the curve reflects semantic closeness rather
than a changing number of OOD examples. The output is one CSV file.

Usage:
  python experiments/experiment3_near_ood_difficulty.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    RESULT_DIR,
    all_model_specs,
    calibrate_threshold,
    load_or_fit_model,
    load_standard_splits,
    metric_row,
    print_metric_table,
    score_model,
    write_csv,
)


DEFAULT_OUTPUT = RESULT_DIR / "experiment3_near_ood_difficulty.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--ratio-step",
        type=int,
        default=10,
        help="Near-OOD percentage step size. Default: 10 for 0, 10, ..., 100.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=77,
        help="Seed used for deterministic Near/Far OOD subsampling.",
    )
    parser.add_argument(
        "--ood-total",
        type=int,
        default=None,
        help="Fixed OOD sample count per ratio. Defaults to min(n_near, n_far).",
    )
    return parser.parse_args()


def near_ood_ratio_percents(ratio_step: int) -> list[int]:
    if ratio_step <= 0 or ratio_step > 100:
        raise ValueError("--ratio-step must be between 1 and 100.")
    percents = list(range(0, 101, ratio_step))
    if percents[-1] != 100:
        percents.append(100)
    return percents


def near_ood_ratio_groups(
    eval_split,
    *,
    ratio_step: int = 10,
    random_seed: int = 77,
    ood_total: int | None = None,
) -> list[dict[str, object]]:
    ood_types = eval_split["ood_types"].astype(str)
    id_indices = np.flatnonzero(ood_types == "id")
    near_indices = np.flatnonzero(ood_types == "near")
    far_indices = np.flatnonzero(ood_types == "far")

    if len(id_indices) == 0:
        raise ValueError("Evaluation split has no ID examples.")
    if len(near_indices) == 0 or len(far_indices) == 0:
        raise ValueError("Evaluation split must contain both Near-OOD and Far-OOD examples.")

    max_balanced_ood_total = min(len(near_indices), len(far_indices))
    fixed_ood_total = max_balanced_ood_total if ood_total is None else int(ood_total)
    if fixed_ood_total <= 0:
        raise ValueError("--ood-total must be positive.")
    if fixed_ood_total > max_balanced_ood_total:
        raise ValueError(
            "--ood-total must be <= min(n_near, n_far) so both 0% and 100% ratios are feasible. "
            f"Received {fixed_ood_total}, maximum is {max_balanced_ood_total}."
        )

    rng = np.random.default_rng(random_seed)
    near_order = rng.permutation(near_indices)
    far_order = rng.permutation(far_indices)

    groups: list[dict[str, object]] = []
    for percent in near_ood_ratio_percents(ratio_step):
        near_count = int(round(fixed_ood_total * percent / 100.0))
        far_count = fixed_ood_total - near_count
        selected_ood_indices = np.concatenate(
            [near_order[:near_count], far_order[:far_count]]
        )
        mask = np.zeros(len(ood_types), dtype=bool)
        mask[id_indices] = True
        mask[selected_ood_indices] = True

        groups.append(
            {
                "near_ood_percent": percent,
                "near_ood_ratio": percent / 100.0,
                "near_ood_count": near_count,
                "far_ood_count": far_count,
                "ood_total": fixed_ood_total,
                "mask": mask,
            }
        )
    return groups


def run(
    output_path: Path = DEFAULT_OUTPUT,
    force_train: bool = False,
    ratio_step: int = 10,
    random_seed: int = 77,
    ood_total: int | None = None,
) -> Path:
    train, _, validation, eval_split = load_standard_splits()
    groups = near_ood_ratio_groups(
        eval_split,
        ratio_step=ratio_step,
        random_seed=random_seed,
        ood_total=ood_total,
    )
    eval_ood_types = eval_split["ood_types"].astype(str)
    rows = []

    for spec in all_model_specs():
        model, loaded, fit_seconds = load_or_fit_model(spec, train, force_train=force_train)
        threshold, _, _ = calibrate_threshold(spec, model, validation, strategy="validation_best_f1")
        eval_scores, _, score_seconds = score_model(spec, model, eval_split["embeddings"])

        for info in groups:
            mask = info["mask"]
            y_true = (eval_ood_types[mask] != "id").astype(int)
            scores = eval_scores[mask]
            extra = {
                "model_key": spec.key,
                "model_source": "loaded" if loaded else "trained",
                "near_ood_percent": info["near_ood_percent"],
                "near_ood_ratio": info["near_ood_ratio"],
                "near_ood_count": info["near_ood_count"],
                "far_ood_count": info["far_ood_count"],
                "ood_total": info["ood_total"],
                "sampling_seed": random_seed,
            }
            rows.append(
                metric_row(
                    experiment="experiment3_near_ood_ratio",
                    family=spec.family_name,
                    model=spec.display_name,
                    score=spec.score_name,
                    hyperparameters=spec.hyperparameters,
                    threshold_source="validation_best_f1",
                    data_split="eval",
                    metric_split=f"Near-OOD-ratio-{int(info['near_ood_percent']):03d}pct",
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
        "Experiment 3 metrics (eval / Near-OOD ratio sweep)",
        metric_split=None,
        max_rows=120,
    )
    return output


def main() -> None:
    args = parse_args()
    output = run(
        args.output,
        force_train=args.force_train,
        ratio_step=args.ratio_step,
        random_seed=args.random_seed,
        ood_total=args.ood_total,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
