"""
Method: Entropy
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = -sum(p * log(p))  (higher entropy = more OOD)

Usage (from project root):
  python embedding_scoring/entropy_score.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
from scipy.stats     import entropy as scipy_entropy
from embedding_scoring.utils import (
    classifier_path,
    CLASSIFIER_NAMES,
    load_embeddings,
    ood_metrics,
    predict_probabilities,
)


def entropy_score(clf, embs: np.ndarray) -> np.ndarray:
    probs = predict_probabilities(clf, embs)
    return scipy_entropy(probs, axis=1)   # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Entropy ──")
    for name in CLASSIFIER_NAMES:
        with classifier_path(name).open("rb") as f:
            clf = pickle.load(f)
        ood_metrics(entropy_score(clf, id_embs), entropy_score(clf, ood_embs), f"Entropy ({name.upper()})")


if __name__ == "__main__":
    main()
