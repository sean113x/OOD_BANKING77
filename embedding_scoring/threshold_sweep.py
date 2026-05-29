"""Sweep OOD thresholds for classifier-output scores."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedding_scoring.energy import energy_score  # noqa: E402
from embedding_scoring.entropy_score import entropy_score  # noqa: E402
from embedding_scoring.margin import margin_score  # noqa: E402
from embedding_scoring.max_logit import max_logit_score  # noqa: E402
from embedding_scoring.msp import msp_score  # noqa: E402
from embedding_scoring.utils import CLASSIFIER_NAMES, classifier_path, load_embeddings  # noqa: E402


SCORE_FUNCTIONS = {
    "MSP": msp_score,
    "Entropy": entropy_score,
    "Margin": margin_score,
    "MaxLogit": max_logit_score,
    "Energy": energy_score,
}


def _metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_tpr": recall_score(y_true, y_pred),
        "fpr": fpr,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "youden_j": tpr - fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _best_f1(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return _metrics_at(y_true, scores, thresholds[int(np.nanargmax(f1))])


def _best_balanced_acc(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    balanced = (tpr + (1.0 - fpr)) / 2.0
    return _metrics_at(y_true, scores, thresholds[int(np.nanargmax(balanced))])


def _tpr95_min_fpr(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    candidates = np.where(tpr >= 0.95)[0]
    idx = candidates[np.argmin(fpr[candidates])] if len(candidates) else int(np.nanargmax(tpr - fpr))
    return _metrics_at(y_true, scores, thresholds[int(idx)])


def main() -> None:
    _, _, id_embs, _, ood_embs = load_embeddings()
    embeddings = np.vstack([id_embs, ood_embs])
    y_true = np.concatenate([
        np.zeros(len(id_embs), dtype=int),
        np.ones(len(ood_embs), dtype=int),
    ])

    print("model,score,criterion,threshold,f1,precision,recall_tpr,fpr,balanced_acc,youden_j,tn,fp,fn,tp")
    for model_name in CLASSIFIER_NAMES:
        with classifier_path(model_name).open("rb") as f:
            clf = pickle.load(f)

        for score_name, score_fn in SCORE_FUNCTIONS.items():
            scores = score_fn(clf, embeddings)
            for criterion, selector in (
                ("best_f1", _best_f1),
                ("best_bal_acc", _best_balanced_acc),
                ("tpr95_min_fpr", _tpr95_min_fpr),
            ):
                metrics = selector(y_true, scores)
                print(
                    ",".join(
                        [
                            model_name.upper(),
                            score_name,
                            criterion,
                            f"{metrics['threshold']:.10g}",
                            f"{metrics['f1']:.4f}",
                            f"{metrics['precision']:.4f}",
                            f"{metrics['recall_tpr']:.4f}",
                            f"{metrics['fpr']:.4f}",
                            f"{metrics['balanced_acc']:.4f}",
                            f"{metrics['youden_j']:.4f}",
                            str(metrics["tn"]),
                            str(metrics["fp"]),
                            str(metrics["fn"]),
                            str(metrics["tp"]),
                        ]
                    )
                )


if __name__ == "__main__":
    main()
