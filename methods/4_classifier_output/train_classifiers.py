"""
Train Logistic Regression and MLP on sentence-transformer embeddings.
Saves classifiers to checkpoints/ for use by all Section 4 scoring scripts.

Usage (from project root):
  python methods/4_classifier_output/train_classifiers.py
"""

import sys
sys.path.insert(0, "methods")

import os
import pickle
import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics        import accuracy_score
from shared.evaluate        import load_embeddings

CKPT_DIR = "checkpoints"


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    train_embs, train_labels, id_embs, id_labels, _ = load_embeddings()

    classifiers = {
        "lr":  LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=200, random_state=42),
    }

    for name, clf in classifiers.items():
        print(f"Training {name.upper()}...")
        clf.fit(train_embs, train_labels)
        acc = accuracy_score(id_labels, clf.predict(id_embs))
        print(f"  ID test accuracy: {acc:.4f}")

        path = f"{CKPT_DIR}/clf_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        print(f"  Saved → {path}")


if __name__ == "__main__":
    main()
