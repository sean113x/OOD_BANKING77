"""
Method: Maximum Softmax Probability (MSP)
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = 1 - max(softmax(logits))  (higher = more OOD)

Usage (from project root):
  python methods/4_classifier_output/msp.py
"""

import sys
sys.path.insert(0, "methods")

import pickle
import numpy as np
from scipy.special       import softmax
from shared.evaluate     import load_embeddings, ood_metrics


def msp_score(clf, embs: np.ndarray) -> np.ndarray:
    logits = clf.predict_log_proba(embs)  # (N, C) log-probs
    probs  = softmax(logits, axis=1)
    return 1.0 - probs.max(axis=1)       # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — MSP ──")
    for name in ["lr", "mlp"]:
        with open(f"checkpoints/clf_{name}.pkl", "rb") as f:
            clf = pickle.load(f)
        ood_metrics(msp_score(clf, id_embs), msp_score(clf, ood_embs), f"MSP ({name.upper()})")


if __name__ == "__main__":
    main()
