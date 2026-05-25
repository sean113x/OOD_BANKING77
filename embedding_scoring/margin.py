"""
Method: Margin
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = 1 - (top1_prob - top2_prob)  (smaller margin = more OOD)

Usage (from project root):
  python embedding_scoring/margin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
from embedding_scoring.utils import (
    classifier_path,
    CLASSIFIER_NAMES,
    load_embeddings,
    ood_metrics,
    predict_probabilities,
)


def margin_score(clf, embs: np.ndarray) -> np.ndarray:
    probs     = predict_probabilities(clf, embs)
    sorted_p  = np.sort(probs, axis=1)[:, ::-1]
    margin    = sorted_p[:, 0] - sorted_p[:, 1]
    return 1.0 - margin                   # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Margin ──")
    for name in CLASSIFIER_NAMES:
        with classifier_path(name).open("rb") as f:
            clf = pickle.load(f)
        ood_metrics(margin_score(clf, id_embs), margin_score(clf, ood_embs), f"Margin ({name.upper()})")


if __name__ == "__main__":
    main()
