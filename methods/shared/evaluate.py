"""Shared evaluation utilities. Convention: higher score = more likely OOD."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

EMB_DIR = "BERT/embeddings"


def load_embeddings():
    train_embs   = np.load(f"{EMB_DIR}/train_embs.npy")
    train_labels = np.load(f"{EMB_DIR}/train_labels.npy")
    id_embs      = np.load(f"{EMB_DIR}/id_test_embs.npy")
    id_labels    = np.load(f"{EMB_DIR}/id_test_labels.npy")
    ood_embs     = np.load(f"{EMB_DIR}/ood_test_embs.npy")
    return train_embs, train_labels, id_embs, id_labels, ood_embs


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray, name: str):
    """Print AUROC / AUPR / FPR@95TPR. Higher score = more OOD."""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(labels, scores)
    aupr  = average_precision_score(labels, scores)
    thr   = np.percentile(id_scores, 95)   # threshold at 95th percentile of ID
    fpr95 = (ood_scores >= thr).mean()
    print(f"  [{name:<35}]  AUROC={auroc:.4f}  AUPR={aupr:.4f}  FPR95={fpr95:.4f}")
