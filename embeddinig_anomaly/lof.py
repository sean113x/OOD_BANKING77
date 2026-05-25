"""
Method: Local Outlier Factor (LOF)
Category: 3 - Embedding + Anomaly Detection

Fit on training embeddings (62 known classes).
OOD score = -score_samples(x)  (higher = more OOD)

Usage (from project root):
  python methods/3_anomaly/lof.py
"""

import sys
sys.path.insert(0, "methods")

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from shared.evaluate import load_embeddings, ood_metrics


def main():
    train_embs, _, id_embs, _, ood_embs = load_embeddings()

    # novelty=True enables predict/score_samples on unseen data
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    clf.fit(train_embs)

    # score_samples: higher = more normal → negate for OOD score
    id_scores  = -clf.score_samples(id_embs)
    ood_scores = -clf.score_samples(ood_embs)

    print("── Section 3: Anomaly Detection ──")
    ood_metrics(id_scores, ood_scores, "Local Outlier Factor (LOF)")


if __name__ == "__main__":
    main()
