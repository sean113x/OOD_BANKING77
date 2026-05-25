"""Shared loaders and metrics for embedding-based OOD scoring."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_DIR = PROJECT_ROOT / "dataset" / "preprocessed" / "embedded"
CLASSIFIER_DIR = PROJECT_ROOT / "embedding_scoring" / "classifier"
CLASSIFIER_NAMES = ("lr", "mlp", "gnb", "lda", "qda")


def classifier_path(name: str) -> Path:
    return CLASSIFIER_DIR / f"clf_{name}.pkl"


def load_embedding_split(split_name: str, embed_dir: str | Path = DEFAULT_EMBED_DIR) -> dict[str, np.ndarray]:
    path = Path(embed_dir) / f"{split_name}_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing embedding file: {path}. Run `python BERT/embed.py` first."
        )

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_embeddings(embed_dir: str | Path = DEFAULT_EMBED_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = load_embedding_split("OOD_train", embed_dir)
    id_test = load_embedding_split("classification_test", embed_dir)
    ood_test = load_embedding_split("OOD_test", embed_dir)

    ood_mask = ood_test["is_ood"].astype(str) == "1"
    return (
        train["embeddings"],
        train["labels"],
        id_test["embeddings"],
        id_test["labels"],
        ood_test["embeddings"][ood_mask],
    )


def predict_probabilities(clf, embeddings: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            probs = clf.predict_proba(embeddings)
    elif hasattr(clf, "predict_log_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            probs = np.exp(clf.predict_log_proba(embeddings))
    else:
        raise TypeError(f"{type(clf).__name__} does not provide probability outputs.")

    if not np.isfinite(probs).all():
        raise FloatingPointError(f"{type(clf).__name__} produced non-finite probabilities.")
    return probs


def predict_log_scores(clf, embeddings: np.ndarray) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            scores = clf.decision_function(embeddings)
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{type(clf).__name__} produced non-finite scores.")
        return scores[:, None] if scores.ndim == 1 else scores

    probs = np.clip(predict_probabilities(clf, embeddings), 1e-12, 1.0)
    return np.log(probs)


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray, label: str) -> None:
    y_true = np.concatenate([
        np.zeros(len(id_scores), dtype=int),
        np.ones(len(ood_scores), dtype=int),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    candidates = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[candidates[0]]) if len(candidates) else 1.0

    print(f"{label}: AUROC={auroc:.4f}  AUPR={aupr:.4f}  FPR95={fpr95:.4f}")
