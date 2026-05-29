"""Export threshold-comparison tables for classifier-output OOD scores."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedding_scoring.energy import energy_score  # noqa: E402
from embedding_scoring.entropy_score import entropy_score  # noqa: E402
from embedding_scoring.margin import margin_score  # noqa: E402
from embedding_scoring.max_logit import max_logit_score  # noqa: E402
from embedding_scoring.msp import msp_score  # noqa: E402
from embedding_scoring.utils import CLASSIFIER_NAMES, classifier_path, load_embeddings  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "embedding_scoring" / "results" / "threshold_grid.csv"
ID_PERCENTILES = (50, 60, 70, 80, 85, 90, 92.5, 95, 97.5, 99)
SCORE_FUNCTIONS = {
    "MSP": msp_score,
    "Entropy": entropy_score,
    "Margin": margin_score,
    "MaxLogit": max_logit_score,
    "Energy": energy_score,
}


def _evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_tpr": recall_score(y_true, y_pred),
        "fpr": fpr,
        "specificity": specificity,
        "youden_j": tpr - fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, id_embs, _, ood_embs = load_embeddings()
    embeddings = np.vstack([id_embs, ood_embs])
    y_true = np.concatenate([
        np.zeros(len(id_embs), dtype=int),
        np.ones(len(ood_embs), dtype=int),
    ])

    rows = []
    for model_name in CLASSIFIER_NAMES:
        with classifier_path(model_name).open("rb") as f:
            clf = pickle.load(f)

        for score_name, score_fn in SCORE_FUNCTIONS.items():
            id_scores = score_fn(clf, id_embs)
            all_scores = score_fn(clf, embeddings)
            for percentile in ID_PERCENTILES:
                threshold = float(np.percentile(id_scores, percentile))
                metrics = _evaluate(y_true, all_scores, threshold)
                rows.append({
                    "model": model_name.upper(),
                    "score": score_name,
                    "threshold_source": "id_percentile",
                    "id_percentile": percentile,
                    **metrics,
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "score",
        "threshold_source",
        "id_percentile",
        "threshold",
        "accuracy",
        "balanced_acc",
        "f1",
        "precision",
        "recall_tpr",
        "fpr",
        "specificity",
        "youden_j",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
