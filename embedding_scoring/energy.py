"""
Method: Energy Score
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = -T * log(sum(exp(logits / T)))  (lower energy = more OOD)

Usage (from project root):
  python methods/4_classifier_output/energy.py
"""

import sys
sys.path.insert(0, "methods")

import pickle
import numpy as np
from scipy.special   import logsumexp
from shared.evaluate import load_embeddings, ood_metrics

TEMPERATURE = 1.0


def energy_score(clf, embs: np.ndarray, T: float = TEMPERATURE) -> np.ndarray:
    logits = clf.predict_log_proba(embs)
    return -T * logsumexp(logits / T, axis=1)   # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Energy ──")
    for name in ["lr", "mlp"]:
        with open(f"models/clf_{name}.pkl", "rb") as f:
            clf = pickle.load(f)
        ood_metrics(energy_score(clf, id_embs), energy_score(clf, ood_embs), f"Energy (T={TEMPERATURE}, {name.upper()})")


if __name__ == "__main__":
    main()
