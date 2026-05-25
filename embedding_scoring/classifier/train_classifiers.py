"""
Train Logistic Regression and MLP on sentence-transformer embeddings.
Saves classifiers beside this script for use by all embedding scoring scripts.

Usage (from project root):
  python embedding_scoring/classifier/train_classifiers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import pickle
import warnings
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics        import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from embedding_scoring.utils import classifier_path, load_embeddings

MODEL_DIR = Path(__file__).resolve().parent


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_embs, train_labels, id_embs, id_labels, _ = load_embeddings()

    classifiers = {
        "lr": LogisticRegression(max_iter=3000, C=10.0, solver="saga", random_state=42, n_jobs=1),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(512, 256),
            max_iter=500,
            alpha=0.001,
            tol=1e-5,
            n_iter_no_change=20,
            random_state=42,
        ),
        "gnb": Pipeline([
            ("pca", PCA(n_components=100, whiten=True, random_state=42)),
            ("gnb", GaussianNB(var_smoothing=0.01)),
        ]),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "qda": Pipeline([
            ("pca", PCA(n_components=75, whiten=True, random_state=42)),
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0.01)),
        ]),
    }

    for name, clf in classifiers.items():
        print(f"Training {name.upper()}...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*covariance matrix of class.*")
            clf.fit(train_embs, train_labels)
            pred = clf.predict(id_embs)
            acc = accuracy_score(id_labels, pred)
            macro_f1 = f1_score(id_labels, pred, average="macro")
        print(f"  ID test accuracy: {acc:.4f}  macro-F1: {macro_f1:.4f}")

        path = classifier_path(name)
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        print(f"  Saved → {path}")


if __name__ == "__main__":
    main()
