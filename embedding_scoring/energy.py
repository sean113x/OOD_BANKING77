"""
Method: Energy Score
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = -T * log(sum(exp(logits / T))) using raw, unnormalized logits.

When a classifier exposes raw decision scores through decision_function, those
scores are used. Probability-only classifiers fall back to log probabilities so
the same scoring API is available for every classifier. At T=1 this proxy can
collapse toward a constant because:

  -log(sum(exp(log p_i))) = -log(sum(p_i)) = -log(1) = 0

so compare that fallback with care.

Usage (from project root):
  python embedding_scoring/energy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
from scipy.special   import logsumexp
from embedding_scoring.utils import (
    classifier_path,
    CLASSIFIER_NAMES,
    load_embeddings,
    ood_metrics,
    predict_log_scores,
)

TEMPERATURE = 1.0


def energy_score(clf, embs: np.ndarray, T: float = TEMPERATURE) -> np.ndarray:
    logits = predict_log_scores(clf, embs)
    return -T * logsumexp(logits / T, axis=1)   # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Energy ──")
    for name in CLASSIFIER_NAMES:
        with classifier_path(name).open("rb") as f:
            clf = pickle.load(f)
        ood_metrics(energy_score(clf, id_embs), energy_score(clf, ood_embs), f"Energy (T={TEMPERATURE}, {name.upper()})")


if __name__ == "__main__":
    main()
