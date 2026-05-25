"""
Method: Maximum Softmax Probability (MSP)
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = 1 - max(softmax(logits))  (higher = more OOD)

Usage (from project root):
  python embedding_scoring/msp.py
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


def msp_score(clf, embs: np.ndarray) -> np.ndarray:
    probs = predict_probabilities(clf, embs)
    return 1.0 - probs.max(axis=1)       # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — MSP ──")
    for name in CLASSIFIER_NAMES:
        with classifier_path(name).open("rb") as f:
            clf = pickle.load(f)
        ood_metrics(msp_score(clf, id_embs), msp_score(clf, ood_embs), f"MSP ({name.upper()})")


if __name__ == "__main__":
    main()
