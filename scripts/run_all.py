"""
Run all OOD methods and save results to results/results.csv and results/results.txt

Usage (from project root):
  python scripts/run_all.py
"""

import sys
import os
import csv
import io
from contextlib import redirect_stdout

sys.path.insert(0, "methods")

import numpy as np
import pickle
from shared.evaluate import load_embeddings, ood_metrics as _ood_metrics
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS_DIR = "results"


def compute_metrics(id_scores, ood_scores):
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(labels, scores)
    aupr  = average_precision_score(labels, scores)
    thr   = np.percentile(id_scores, 95)
    fpr95 = (ood_scores >= thr).mean()
    return auroc, aupr, fpr95


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_embs, train_labels, id_embs, id_labels, ood_embs = load_embeddings()
    num_classes = int(train_labels.max()) + 1

    rows = []   # (section, method, auroc, aupr, fpr95)

    # ── Section 3: Anomaly Detection ─────────────────────────────────────────
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    print("Running Section 3: Anomaly Detection...")

    clf_if = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    clf_if.fit(train_embs)
    auroc, aupr, fpr95 = compute_metrics(-clf_if.decision_function(id_embs), -clf_if.decision_function(ood_embs))
    rows.append(("3", "Isolation Forest", auroc, aupr, fpr95))

    clf_lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    clf_lof.fit(train_embs)
    auroc, aupr, fpr95 = compute_metrics(-clf_lof.score_samples(id_embs), -clf_lof.score_samples(ood_embs))
    rows.append(("3", "LOF", auroc, aupr, fpr95))

    # ── Section 4: Classifier-output OOD Scoring ─────────────────────────────
    from scipy.special import softmax, logsumexp
    from scipy.stats   import entropy as scipy_entropy

    print("Running Section 4: Classifier-output OOD Scoring...")

    for clf_name in ["lr", "mlp"]:
        with open(f"checkpoints/clf_{clf_name}.pkl", "rb") as f:
            clf = pickle.load(f)

        id_logits  = clf.predict_log_proba(id_embs)
        ood_logits = clf.predict_log_proba(ood_embs)

        id_probs  = softmax(id_logits,  axis=1)
        ood_probs = softmax(ood_logits, axis=1)

        # MSP
        auroc, aupr, fpr95 = compute_metrics(1 - id_probs.max(1), 1 - ood_probs.max(1))
        rows.append(("4", f"MSP ({clf_name.upper()})", auroc, aupr, fpr95))

        # Entropy
        auroc, aupr, fpr95 = compute_metrics(scipy_entropy(id_probs, axis=1), scipy_entropy(ood_probs, axis=1))
        rows.append(("4", f"Entropy ({clf_name.upper()})", auroc, aupr, fpr95))

        # Margin
        def margin(probs):
            s = np.sort(probs, axis=1)[:, ::-1]
            return 1 - (s[:, 0] - s[:, 1])
        auroc, aupr, fpr95 = compute_metrics(margin(id_probs), margin(ood_probs))
        rows.append(("4", f"Margin ({clf_name.upper()})", auroc, aupr, fpr95))

        # Max Logit
        auroc, aupr, fpr95 = compute_metrics(-id_logits.max(1), -ood_logits.max(1))
        rows.append(("4", f"Max Logit ({clf_name.upper()})", auroc, aupr, fpr95))

        # Energy
        T = 1.0
        auroc, aupr, fpr95 = compute_metrics(-T * logsumexp(id_logits / T, axis=1), -T * logsumexp(ood_logits / T, axis=1))
        rows.append(("4", f"Energy ({clf_name.upper()})", auroc, aupr, fpr95))

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path = f"{RESULTS_DIR}/results.csv"
    txt_path = f"{RESULTS_DIR}/results.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "method", "AUROC", "AUPR", "FPR95"])
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])

    with open(txt_path, "w") as f:
        f.write(f"{'Section':<10}{'Method':<35}{'AUROC':>8}{'AUPR':>8}{'FPR95':>8}\n")
        f.write("-" * 70 + "\n")
        for sec, method, auroc, aupr, fpr95 in rows:
            f.write(f"{sec:<10}{method:<35}{auroc:>8.4f}{aupr:>8.4f}{fpr95:>8.4f}\n")

    # Print
    with open(txt_path) as f:
        print(f.read())

    print(f"Saved → {csv_path}")
    print(f"Saved → {txt_path}")


if __name__ == "__main__":
    main()
