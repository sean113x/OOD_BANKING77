"""
Method: Isolation Forest
Category: 3 - Embedding + Anomaly Detection

Fit on training embeddings (62 known classes).
OOD score = -decision_function(x)  (higher = more OOD)

Usage (from project root):
  python embeddinig_anomaly/isolation_forest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import IsolationForest
from embedding_scoring.utils import load_embeddings, ood_metrics


def main():
    train_embs, _, id_embs, _, ood_embs = load_embeddings()

    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    clf.fit(train_embs)

    # decision_function: higher = more normal → negate for OOD score
    id_scores  = -clf.decision_function(id_embs)
    ood_scores = -clf.decision_function(ood_embs)

    print("── Section 3: Anomaly Detection ──")
    ood_metrics(id_scores, ood_scores, "Isolation Forest")


if __name__ == "__main__":
    main()
