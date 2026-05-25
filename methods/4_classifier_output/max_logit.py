"""
Method: Maximum Logit
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = -max(logits)  (lower max logit = more OOD)

Usage (from project root):
  python methods/4_classifier_output/max_logit.py
"""

import sys
sys.path.insert(0, "methods")

import pickle
import numpy as np
from shared.evaluate import load_embeddings, ood_metrics


def max_logit_score(clf, embs: np.ndarray) -> np.ndarray:
    logits = clf.predict_log_proba(embs)
    return -logits.max(axis=1)


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Max Logit ──")
    for name in ["lr", "mlp"]:
        with open(f"checkpoints/clf_{name}.pkl", "rb") as f:
            clf = pickle.load(f)
        ood_metrics(max_logit_score(clf, id_embs), max_logit_score(clf, ood_embs), f"Max Logit ({name.upper()})")


if __name__ == "__main__":
    main()
