"""
Method: Entropy
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = -sum(p * log(p))  (higher entropy = more OOD)

Usage (from project root):
  python methods/4_classifier_output/entropy_score.py
"""

import sys
sys.path.insert(0, "methods")

import pickle
import numpy as np
from scipy.special   import softmax
from scipy.stats     import entropy as scipy_entropy
from shared.evaluate import load_embeddings, ood_metrics


def entropy_score(clf, embs: np.ndarray) -> np.ndarray:
    logits = clf.predict_log_proba(embs)
    probs  = softmax(logits, axis=1)
    return scipy_entropy(probs, axis=1)   # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Entropy ──")
    for name in ["lr", "mlp"]:
        with open(f"models/clf_{name}.pkl", "rb") as f:
            clf = pickle.load(f)
        ood_metrics(entropy_score(clf, id_embs), entropy_score(clf, ood_embs), f"Entropy ({name.upper()})")


if __name__ == "__main__":
    main()
