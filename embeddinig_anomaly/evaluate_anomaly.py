"""Train and evaluate embedding-based anomaly detection OOD baselines.

Methods:
  - Isolation Forest
  - Local Outlier Factor (novelty mode)

All scores are oriented so that higher = more OOD.
Threshold-dependent metrics use the threshold that maximizes F1 per split.

Usage:
  python embeddinig_anomaly/evaluate_anomaly.py
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedding_scoring.utils import load_embedding_split  # noqa: E402


MODEL_DIR = Path(__file__).resolve().parent / "models"
RESULT_DIR = Path(__file__).resolve().parent / "results"


def _fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    candidates = np.where(tpr >= 0.95)[0]
    return float(fpr[candidates[0]]) if len(candidates) else 1.0


def _metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _best_f1_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    best_idx = int(np.nanargmax(f1_values))
    return _metrics_at_threshold(y_true, scores, float(thresholds[best_idx]))


def _evaluate_split(
    method: str,
    split: str,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int | str]:
    best = _best_f1_metrics(y_true, scores)
    return {
        "method": method,
        "split": split,
        "auroc": roc_auc_score(y_true, scores),
        "aupr": average_precision_score(y_true, scores),
        "fpr95": _fpr95(y_true, scores),
        **best,
    }


def _save_model(name: str, model) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / f"{name}.pkl").open("wb") as f:
        pickle.dump(model, f)


def _write_csv(rows: list[dict[str, float | int | str]]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "anomaly_best_f1_by_split.csv"
    fieldnames = [
        "method",
        "split",
        "auroc",
        "aupr",
        "fpr95",
        "threshold",
        "f1",
        "precision",
        "recall",
        "tpr",
        "fpr",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _print_table(rows: list[dict[str, float | int | str]]) -> None:
    print(
        "method,split,AUROC,AUPR,FPR95,F1,Precision,Recall,TPR,FPR,Accuracy,threshold"
    )
    for row in rows:
        print(
            ",".join(
                [
                    str(row["method"]),
                    str(row["split"]),
                    f"{row['auroc']:.4f}",
                    f"{row['aupr']:.4f}",
                    f"{row['fpr95']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['tpr']:.4f}",
                    f"{row['fpr']:.4f}",
                    f"{row['accuracy']:.4f}",
                    f"{row['threshold']:.10g}",
                ]
            )
        )


def main() -> None:
    train = load_embedding_split("OOD_train")
    test = load_embedding_split("OOD_test")

    train_embeddings = train["embeddings"]
    test_embeddings = test["embeddings"]
    ood_types = test["ood_types"].astype(str)

    masks = {
        "Overall": np.ones(len(test_embeddings), dtype=bool),
        "Near-OOD": (ood_types == "id") | (ood_types == "near"),
        "Far-OOD": (ood_types == "id") | (ood_types == "far"),
    }

    models = {
        "isolation_forest": IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
        "lof": LocalOutlierFactor(
            n_neighbors=20,
            novelty=True,
            n_jobs=-1,
        ),
    }

    rows: list[dict[str, float | int | str]] = []
    for method, model in models.items():
        model.fit(train_embeddings)
        _save_model(method, model)

        if method == "isolation_forest":
            scores = -model.decision_function(test_embeddings)
        else:
            scores = -model.score_samples(test_embeddings)

        for split, mask in masks.items():
            y_true = (ood_types[mask] != "id").astype(int)
            rows.append(_evaluate_split(method, split, y_true, scores[mask]))

    output_path = _write_csv(rows)
    _print_table(rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
