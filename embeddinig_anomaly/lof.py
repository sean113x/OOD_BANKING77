"""
Method: Local Outlier Factor (LOF)
Category: 3 - Embedding + Anomaly Detection

Fit on training embeddings (62 known classes).
OOD score = -score_samples(x)  (higher = more OOD)

Usage (from project root):
  python embeddinig_anomaly/lof.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.neighbors import LocalOutlierFactor
from embedding_scoring.utils import load_embeddings, ood_metrics


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
