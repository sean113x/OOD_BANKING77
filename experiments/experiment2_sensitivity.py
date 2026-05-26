"""Experiment 2: hyperparameter and threshold sensitivity.

This script evaluates the current four baselines:
  - Logistic Regression + Energy score
  - MLP + Entropy score
  - LOF
  - Isolation Forest

All OOD scores are oriented as higher = more OOD. Threshold-dependent metrics
use the threshold that maximizes F1 for the split unless otherwise stated.

Usage:
  python experiments/experiment2_sensitivity.py
"""

from __future__ import annotations

import csv
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.special import logsumexp
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedding_scoring.utils import load_embedding_split  # noqa: E402


RESULT_DIR = PROJECT_ROOT / "experiments" / "results"
ID_PERCENTILES = (50, 60, 70, 80, 90, 95, 99)


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = load_embedding_split("OOD_train")
    id_test = load_embedding_split("classification_test")
    ood_test = load_embedding_split("OOD_test")

    return (
        train["embeddings"],
        train["labels"],
        id_test["embeddings"],
        id_test["labels"],
        ood_test["embeddings"],
        ood_test["ood_types"].astype(str),
    )


def _fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    candidates = np.where(tpr >= 0.95)[0]
    return float(fpr[candidates[0]]) if len(candidates) else 1.0


def _metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
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
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "youden_j": tpr - fpr,
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
    return _metrics_at(y_true, scores, float(thresholds[best_idx]))


def _evaluate_scores(
    model: str,
    score_name: str,
    hyperparameters: str,
    split: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    id_accuracy: float | None = None,
    id_macro_f1: float | None = None,
) -> dict[str, float | int | str | None]:
    best = _best_f1_metrics(y_true, scores)
    return {
        "model": model,
        "score": score_name,
        "hyperparameters": hyperparameters,
        "split": split,
        "id_accuracy": id_accuracy,
        "id_macro_f1": id_macro_f1,
        "auroc": roc_auc_score(y_true, scores),
        "aupr": average_precision_score(y_true, scores),
        "fpr95": _fpr95(y_true, scores),
        **best,
    }


def _energy_from_decision_scores(clf, embeddings: np.ndarray) -> np.ndarray:
    logits = clf.decision_function(embeddings)
    if logits.ndim == 1:
        logits = logits[:, None]
    return -logsumexp(logits, axis=1)


def _entropy_from_probabilities(clf, embeddings: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(embeddings)
    return scipy_entropy(probs, axis=1)


def _split_masks(ood_types: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Overall": np.ones(len(ood_types), dtype=bool),
        "Near-OOD": (ood_types == "id") | (ood_types == "near"),
        "Far-OOD": (ood_types == "id") | (ood_types == "far"),
    }


def _classification_metrics(clf, id_embeddings: np.ndarray, id_labels: np.ndarray) -> tuple[float, float]:
    pred = clf.predict(id_embeddings)
    return (
        accuracy_score(id_labels, pred),
        f1_score(id_labels, pred, average="macro"),
    )


def run_hyperparameter_sweep() -> list[dict[str, float | int | str | None]]:
    train_embeddings, train_labels, id_embeddings, id_labels, test_embeddings, ood_types = _load_data()
    rows: list[dict[str, float | int | str | None]] = []
    masks = _split_masks(ood_types)

    lr_configs = [
        {"C": 0.01},
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
        {"C": 100.0},
    ]
    for cfg in lr_configs:
        clf = LogisticRegression(
            C=cfg["C"],
            max_iter=3000,
            solver="saga",
            random_state=42,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*convergence.*")
            clf.fit(train_embeddings, train_labels)
            id_acc, id_f1 = _classification_metrics(clf, id_embeddings, id_labels)
            scores = _energy_from_decision_scores(clf, test_embeddings)

        for split, mask in masks.items():
            y_true = (ood_types[mask] != "id").astype(int)
            rows.append(
                _evaluate_scores(
                    "LR",
                    "Energy",
                    f"C={cfg['C']}",
                    split,
                    y_true,
                    scores[mask],
                    id_acc,
                    id_f1,
                )
            )

    mlp_configs = [
        {"hidden_layer_sizes": (256,), "learning_rate_init": 0.001, "alpha": 0.001},
        {"hidden_layer_sizes": (512,), "learning_rate_init": 0.001, "alpha": 0.001},
        {"hidden_layer_sizes": (512, 256), "learning_rate_init": 0.0005, "alpha": 0.001},
        {"hidden_layer_sizes": (512, 256), "learning_rate_init": 0.001, "alpha": 0.001},
        {"hidden_layer_sizes": (512, 256), "learning_rate_init": 0.003, "alpha": 0.001},
    ]
    for cfg in mlp_configs:
        clf = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            learning_rate_init=cfg["learning_rate_init"],
            alpha=cfg["alpha"],
            max_iter=500,
            tol=1e-5,
            n_iter_no_change=20,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")
            clf.fit(train_embeddings, train_labels)
            id_acc, id_f1 = _classification_metrics(clf, id_embeddings, id_labels)
            scores = _entropy_from_probabilities(clf, test_embeddings)

        hidden = "x".join(str(v) for v in cfg["hidden_layer_sizes"])
        params = f"hidden={hidden}, lr={cfg['learning_rate_init']}, alpha={cfg['alpha']}"
        for split, mask in masks.items():
            y_true = (ood_types[mask] != "id").astype(int)
            rows.append(
                _evaluate_scores(
                    "MLP",
                    "Entropy",
                    params,
                    split,
                    y_true,
                    scores[mask],
                    id_acc,
                    id_f1,
                )
            )

    lof_configs = [
        {"n_neighbors": 5, "metric": "euclidean"},
        {"n_neighbors": 10, "metric": "euclidean"},
        {"n_neighbors": 20, "metric": "euclidean"},
        {"n_neighbors": 50, "metric": "euclidean"},
        {"n_neighbors": 100, "metric": "euclidean"},
        {"n_neighbors": 20, "metric": "cosine"},
    ]
    for cfg in lof_configs:
        clf = LocalOutlierFactor(
            n_neighbors=cfg["n_neighbors"],
            metric=cfg["metric"],
            novelty=True,
            n_jobs=1,
        )
        clf.fit(train_embeddings)
        scores = -clf.score_samples(test_embeddings)
        params = f"k={cfg['n_neighbors']}, metric={cfg['metric']}"

        for split, mask in masks.items():
            y_true = (ood_types[mask] != "id").astype(int)
            rows.append(
                _evaluate_scores("LOF", "negative_score_samples", params, split, y_true, scores[mask])
            )

    if_configs = [
        {"n_estimators": 100, "max_samples": "auto", "contamination": "auto"},
        {"n_estimators": 200, "max_samples": "auto", "contamination": "auto"},
        {"n_estimators": 500, "max_samples": "auto", "contamination": "auto"},
        {"n_estimators": 200, "max_samples": 0.5, "contamination": "auto"},
        {"n_estimators": 200, "max_samples": 0.75, "contamination": "auto"},
        {"n_estimators": 200, "max_samples": 1.0, "contamination": "auto"},
        {"n_estimators": 200, "max_samples": "auto", "contamination": 0.05},
        {"n_estimators": 200, "max_samples": "auto", "contamination": 0.1},
    ]
    for cfg in if_configs:
        clf = IsolationForest(
            n_estimators=cfg["n_estimators"],
            max_samples=cfg["max_samples"],
            contamination=cfg["contamination"],
            random_state=42,
            n_jobs=1,
        )
        clf.fit(train_embeddings)
        scores = -clf.decision_function(test_embeddings)
        params = (
            f"n_estimators={cfg['n_estimators']}, "
            f"max_samples={cfg['max_samples']}, "
            f"contamination={cfg['contamination']}"
        )

        for split, mask in masks.items():
            y_true = (ood_types[mask] != "id").astype(int)
            rows.append(
                _evaluate_scores("IsolationForest", "negative_decision_function", params, split, y_true, scores[mask])
            )

    return rows


def run_threshold_sweep(best_rows: list[dict[str, float | int | str | None]]) -> list[dict[str, float | int | str | None]]:
    train_embeddings, train_labels, _, _, test_embeddings, ood_types = _load_data()
    masks = _split_masks(ood_types)
    threshold_rows: list[dict[str, float | int | str | None]] = []

    configs = [
        (
            "LR",
            "Energy",
            "C=10.0",
            LogisticRegression(C=10.0, max_iter=3000, solver="saga", random_state=42, n_jobs=1),
            lambda model, values: _energy_from_decision_scores(model, values),
        ),
        (
            "MLP",
            "Entropy",
            "hidden=512x256, lr=0.001, alpha=0.001",
            MLPClassifier(
                hidden_layer_sizes=(512, 256),
                learning_rate_init=0.001,
                alpha=0.001,
                max_iter=500,
                tol=1e-5,
                n_iter_no_change=20,
                random_state=42,
            ),
            lambda model, values: _entropy_from_probabilities(model, values),
        ),
        (
            "LOF",
            "negative_score_samples",
            "k=20, metric=euclidean",
            LocalOutlierFactor(n_neighbors=20, metric="euclidean", novelty=True, n_jobs=1),
            lambda model, values: -model.score_samples(values),
        ),
        (
            "IsolationForest",
            "negative_decision_function",
            "n_estimators=200, max_samples=auto, contamination=auto",
            IsolationForest(n_estimators=200, max_samples="auto", contamination="auto", random_state=42, n_jobs=1),
            lambda model, values: -model.decision_function(values),
        ),
    ]

    for model_name, score_name, params, clf, scorer in configs:
        clf.fit(train_embeddings, train_labels) if model_name in {"LR", "MLP"} else clf.fit(train_embeddings)
        scores = scorer(clf, test_embeddings)
        id_scores = scores[ood_types == "id"]

        for split, mask in masks.items():
            y_true = (ood_types[mask] != "id").astype(int)
            split_scores = scores[mask]
            for percentile in ID_PERCENTILES:
                threshold = float(np.percentile(id_scores, percentile))
                threshold_rows.append(
                    {
                        "model": model_name,
                        "score": score_name,
                        "hyperparameters": params,
                        "split": split,
                        "threshold_source": "id_percentile",
                        "id_percentile": percentile,
                        **_metrics_at(y_true, split_scores, threshold),
                    }
                )

    return threshold_rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str | None]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, float | int | str | None]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _write_report(
    hyper_rows: list[dict[str, float | int | str | None]],
    threshold_rows: list[dict[str, float | int | str | None]],
) -> Path:
    report_path = RESULT_DIR / "experiment2_sensitivity_report.md"
    columns = [
        "model",
        "score",
        "hyperparameters",
        "auroc",
        "aupr",
        "fpr95",
        "f1",
        "precision",
        "recall",
        "tpr",
        "fpr",
        "accuracy",
    ]

    parts = ["# Experiment 2: Hyperparameter and Threshold Sensitivity", ""]
    for model in ("LR", "MLP", "LOF", "IsolationForest"):
        rows = [
            row for row in hyper_rows
            if row["model"] == model and row["split"] == "Overall"
        ]
        parts.append(f"## {model} Hyperparameter Sweep (Overall)")
        parts.append("")
        parts.append(_markdown_table(rows, columns))
        parts.append("")

    threshold_columns = [
        "model",
        "id_percentile",
        "f1",
        "precision",
        "recall",
        "tpr",
        "fpr",
        "accuracy",
        "balanced_accuracy",
        "youden_j",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    parts.append("## Threshold Sweep (Overall)")
    parts.append("")
    rows = [
        row for row in threshold_rows
        if row["split"] == "Overall" and row["id_percentile"] in {50, 70, 90, 95, 99}
    ]
    parts.append(_markdown_table(rows, threshold_columns))
    parts.append("")
    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path


def main() -> None:
    hyper_rows = run_hyperparameter_sweep()
    threshold_rows = run_threshold_sweep(hyper_rows)

    hyper_fields = [
        "model",
        "score",
        "hyperparameters",
        "split",
        "id_accuracy",
        "id_macro_f1",
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
        "balanced_accuracy",
        "youden_j",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    threshold_fields = [
        "model",
        "score",
        "hyperparameters",
        "split",
        "threshold_source",
        "id_percentile",
        "threshold",
        "f1",
        "precision",
        "recall",
        "tpr",
        "fpr",
        "accuracy",
        "balanced_accuracy",
        "youden_j",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    hyper_path = RESULT_DIR / "experiment2_hyperparameter_sweep.csv"
    threshold_path = RESULT_DIR / "experiment2_threshold_sweep.csv"
    _write_csv(hyper_path, hyper_rows, hyper_fields)
    _write_csv(threshold_path, threshold_rows, threshold_fields)
    report_path = _write_report(hyper_rows, threshold_rows)

    print(f"Wrote {hyper_path}")
    print(f"Wrote {threshold_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
