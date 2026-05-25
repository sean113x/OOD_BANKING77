"""
Method: Energy Score
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = -T * log(sum(exp(logits / T))) using raw, unnormalized logits.

Energy is only meaningful for classifiers that expose raw decision scores.
For sklearn MLPClassifier, only normalized probabilities are exposed. If log
probabilities are substituted for logits, then at T=1:

  -log(sum(exp(log p_i))) = -log(sum(p_i)) = -log(1) = 0

which collapses the score to an almost constant value and cannot rank OOD
samples reliably.

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
    if not hasattr(clf, "decision_function"):
        raise ValueError(
            f"Energy score requires raw decision scores/logits; {type(clf).__name__} "
            "only exposes probabilities in sklearn."
        )
    logits = predict_log_scores(clf, embs)
    return -T * logsumexp(logits / T, axis=1)   # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Energy ──")
    for name in CLASSIFIER_NAMES:
        with classifier_path(name).open("rb") as f:
            clf = pickle.load(f)
        if not hasattr(clf, "decision_function"):
            print(f"Energy ({name.upper()}): skipped; raw logits/decision scores are not available.")
            continue
        ood_metrics(energy_score(clf, id_embs), energy_score(clf, ood_embs), f"Energy (T={TEMPERATURE}, {name.upper()})")


if __name__ == "__main__":
    main()
