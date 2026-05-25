"""
Method: Maximum Logit
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = -max(logits)  (lower max logit = more OOD)

Usage (from project root):
  python embedding_scoring/max_logit.py
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
    predict_log_scores,
)


def max_logit_score(clf, embs: np.ndarray) -> np.ndarray:
    logits = predict_log_scores(clf, embs)
    return -logits.max(axis=1)


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Max Logit ──")
    for name in CLASSIFIER_NAMES:
        with classifier_path(name).open("rb") as f:
            clf = pickle.load(f)
        ood_metrics(max_logit_score(clf, id_embs), max_logit_score(clf, ood_embs), f"Max Logit ({name.upper()})")


if __name__ == "__main__":
    main()
