"""
Method: Margin
Category: 4 - Embedding + Classifier-output OOD Scoring

OOD score = 1 - (top1_prob - top2_prob)  (smaller margin = more OOD)

Usage (from project root):
  python methods/4_classifier_output/margin.py
"""

import sys
sys.path.insert(0, "methods")

import pickle
import numpy as np
from scipy.special   import softmax
from shared.evaluate import load_embeddings, ood_metrics


def margin_score(clf, embs: np.ndarray) -> np.ndarray:
    logits    = clf.predict_log_proba(embs)
    probs     = softmax(logits, axis=1)
    sorted_p  = np.sort(probs, axis=1)[:, ::-1]
    margin    = sorted_p[:, 0] - sorted_p[:, 1]
    return 1.0 - margin                   # higher = more OOD


def main():
    _, _, id_embs, _, ood_embs = load_embeddings()

    print("── Section 4: Classifier-output OOD Scoring — Margin ──")
    for name in ["lr", "mlp"]:
        with open(f"models/clf_{name}.pkl", "rb") as f:
            clf = pickle.load(f)
        ood_metrics(margin_score(clf, id_embs), margin_score(clf, ood_embs), f"Margin ({name.upper()})")


if __name__ == "__main__":
    main()
