"""Experiment runner for class-wise boundary OOD methods."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from centroid_distance import CentroidDistanceModel
from mahalanobis_distance import MahalanobisDistanceModel
from radius_threshold import RadiusThresholdModel


DEFAULT_EMBEDDED_DIR = Path("dataset/preprocessed/embedded")
DEFAULT_OUTPUT_DIR = Path("embedding_class-wise methods/outputs")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(tpr >= 0.95)[0]
    return float(np.min(fpr[valid])) if len(valid) else float("nan")


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted_ood = scores > threshold
    true_ood = y_true == 1
    tp = int(np.sum(predicted_ood & true_ood))
    fp = int(np.sum(predicted_ood & ~true_ood))
    fn = int(np.sum(~predicted_ood & true_ood))
    tn = int(np.sum(~predicted_ood & ~true_ood))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    id_recall = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (recall + id_recall) / 2
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "ood_precision": float(precision),
        "ood_recall": float(recall),
        "ood_f1": float(f1),
        "id_recall": float(id_recall),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_accuracy_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find the test-label threshold that maximizes binary OOD accuracy."""
    y_true = y_true.astype(int)
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positives_by_score = np.zeros(len(unique_scores), dtype=np.int64)
    np.add.at(positives_by_score, inverse, y_true)
    negatives_by_score = counts - positives_by_score

    total_positive = int(np.sum(y_true))
    prefix_positive = np.concatenate([[0], np.cumsum(positives_by_score)])
    prefix_negative = np.concatenate([[0], np.cumsum(negatives_by_score)])

    tp = total_positive - prefix_positive
    tn = prefix_negative
    accuracy = (tp + tn) / len(y_true)
    best_idx = int(np.argmax(accuracy))

    if best_idx == 0:
        return float(unique_scores[0] - 1e-12)
    if best_idx == len(unique_scores):
        return float(unique_scores[-1] + 1e-12)
    return float((unique_scores[best_idx - 1] + unique_scores[best_idx]) / 2)


def evaluate(method: str, model, test_set: dict[str, np.ndarray], output_dir: Path) -> dict[str, float | int | str]:
    scores, nearest_classes = model.score_embeddings(test_set["embeddings"])
    y_true = test_set["is_ood"].astype(int)
    id_mask = y_true == 0
    ood_mask = y_true == 1
    metrics = {
        "method": method,
        "auroc": float(roc_auc_score(y_true, scores)),
        "aupr_ood": float(average_precision_score(y_true, scores)),
        "fpr95": fpr95(y_true, scores),
        "id_nearest_class_accuracy": float(
            np.mean(nearest_classes[id_mask] == test_set["label_texts"][id_mask])
        ),
        "id_score_mean": float(np.mean(scores[id_mask])),
        "ood_score_mean": float(np.mean(scores[ood_mask])),
    }
    metrics.update(threshold_metrics(y_true, scores, float(model.threshold)))
    optimal = threshold_metrics(y_true, scores, best_accuracy_threshold(y_true, scores))
    metrics.update(
        {
            "optimal_threshold": optimal["threshold"],
            "optimal_accuracy": optimal["accuracy"],
            "optimal_precision": optimal["precision"],
            "optimal_recall": optimal["recall"],
            "optimal_ood_f1": optimal["ood_f1"],
            "optimal_id_recall": optimal["id_recall"],
            "optimal_balanced_accuracy": optimal["balanced_accuracy"],
            "optimal_tp": optimal["tp"],
            "optimal_fp": optimal["fp"],
            "optimal_fn": optimal["fn"],
            "optimal_tn": optimal["tn"],
        }
    )
    write_scores(output_dir / f"{method}_scores.csv", test_set, scores, nearest_classes, float(model.threshold))
    return metrics


def write_scores(
    path: Path,
    test_set: dict[str, np.ndarray],
    scores: np.ndarray,
    nearest_classes: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "text",
        "label",
        "label_text",
        "is_ood",
        "ood_type",
        "source",
        "ood_score",
        "predicted_nearest_class",
        "predicted_is_ood",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, score in enumerate(scores):
            writer.writerow(
                {
                    "text": str(test_set["texts"][idx]),
                    "label": str(test_set["labels"][idx]),
                    "label_text": str(test_set["label_texts"][idx]),
                    "is_ood": int(test_set["is_ood"][idx]),
                    "ood_type": str(test_set.get("ood_types", np.array(["unknown"]))[idx]),
                    "source": str(test_set.get("sources", np.array(["unknown"]))[idx]),
                    "ood_score": float(score),
                    "predicted_nearest_class": str(nearest_classes[idx]),
                    "predicted_is_ood": int(score > threshold),
                }
            )


def write_metrics(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "method",
        "auroc",
        "aupr_ood",
        "fpr95",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "ood_precision",
        "ood_recall",
        "ood_f1",
        "id_recall",
        "balanced_accuracy",
        "optimal_threshold",
        "optimal_accuracy",
        "optimal_precision",
        "optimal_recall",
        "optimal_ood_f1",
        "optimal_id_recall",
        "optimal_balanced_accuracy",
        "id_nearest_class_accuracy",
        "id_score_mean",
        "ood_score_mean",
        "tp",
        "fp",
        "fn",
        "tn",
        "optimal_tp",
        "optimal_fp",
        "optimal_fn",
        "optimal_tn",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedded-dir", type=Path, default=DEFAULT_EMBEDDED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run_class_wise_experiment() -> None:
    args = parse_args()
    train = load_npz(args.embedded_dir / "OOD_train_embeddings.npz")
    test_set = load_npz(args.embedded_dir / "OOD_test_embeddings.npz")

    models = [
        ("centroid_distance", CentroidDistanceModel(model_path=args.output_dir / "centroid_distance_model.npz")),
        ("radius_threshold", RadiusThresholdModel(model_path=args.output_dir / "radius_threshold_model.npz")),
        (
            "mahalanobis_distance",
            MahalanobisDistanceModel(
                model_path=args.output_dir / "mahalanobis_distance_model.npz",
                regularization_lambda=1e-2,
            ),
        ),
    ]

    rows = []
    for method, model in models:
        model.fit(train["embeddings"], train["label_texts"], save=True)
        metrics = evaluate(method, model, test_set, args.output_dir)
        rows.append(metrics)
        print(
            f"{method}: AUROC={metrics['auroc']:.4f}, "
            f"FPR95={metrics['fpr95']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}, "
            f"OptAcc={metrics['optimal_accuracy']:.4f}@{metrics['optimal_threshold']:.6g}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"OOD-F1={metrics['ood_f1']:.4f}"
        )

    write_metrics(args.output_dir / "metrics.csv", rows)
