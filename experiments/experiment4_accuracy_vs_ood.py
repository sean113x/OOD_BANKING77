"""Experiment 4: closed-set accuracy vs OOD reliability.

This analysis asks whether a classifier with higher known-intent accuracy also
produces more reliable OOD scores.

Outputs:
  - classifier/score OOD metrics by split
  - classifier summary ranked by best Overall AUROC
  - MSP overconfidence table
  - accuracy-vs-AUROC scatter plot
  - MSP confidence histogram plot

Usage:
  python experiments/experiment4_accuracy_vs_ood.py
"""

from __future__ import annotations

import csv
import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.special import logsumexp
from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedding_scoring.utils import CLASSIFIER_NAMES, classifier_path, load_embedding_split  # noqa: E402


RESULT_DIR = PROJECT_ROOT / "experiments" / "results"
CONFIDENCE_THRESHOLD = 0.9
MODEL_DISPLAY = {
    "lr": "LR",
    "mlp": "MLP",
    "gnb": "GNB",
    "lda": "LDA",
    "qda": "QDA",
}


def _load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train = load_embedding_split("OOD_train")
    id_test = load_embedding_split("classification_test")
    ood_test = load_embedding_split("OOD_test")
    return train, id_test, ood_test


def _predict_probabilities(clf, embeddings: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        probs = clf.predict_proba(embeddings)
    if not np.isfinite(probs).all():
        raise FloatingPointError(f"{type(clf).__name__} produced non-finite probabilities.")
    return probs


def _decision_scores(clf, embeddings: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        scores = clf.decision_function(embeddings)
    if scores.ndim == 1:
        scores = scores[:, None]
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"{type(clf).__name__} produced non-finite decision scores.")
    return scores


def _score_functions(clf, embeddings: np.ndarray) -> dict[str, np.ndarray]:
    probs = _predict_probabilities(clf, embeddings)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    scores = {
        "MSP": 1.0 - probs.max(axis=1),
        "Entropy": scipy_entropy(probs, axis=1),
        "Margin": 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1]),
    }

    if hasattr(clf, "decision_function"):
        logits = _decision_scores(clf, embeddings)
        scores["MaxLogit"] = -logits.max(axis=1)
        scores["Energy"] = -logsumexp(logits, axis=1)
    else:
        # sklearn MLP/GNB do not expose raw logits. This is a log-probability
        # proxy, so Energy is intentionally not computed for these models.
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        scores["MaxLogit_logprob_proxy"] = -log_probs.max(axis=1)

    return scores


def _fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    candidates = np.where(tpr >= 0.95)[0]
    return float(fpr[candidates[0]]) if len(candidates) else 1.0


def _metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _best_f1_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    best_idx = int(np.nanargmax(f1_values))
    return _metrics_at(y_true, scores, float(thresholds[best_idx]))


def _split_masks(ood_types: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Overall": np.ones(len(ood_types), dtype=bool),
        "Near-OOD": (ood_types == "id") | (ood_types == "near"),
        "Far-OOD": (ood_types == "id") | (ood_types == "far"),
    }


def _evaluate_score(
    model: str,
    score: str,
    split: str,
    id_accuracy: float,
    id_macro_f1: float,
    has_raw_decision_scores: bool,
    y_true: np.ndarray,
    values: np.ndarray,
) -> dict[str, float | int | str | bool]:
    best = _best_f1_metrics(y_true, values)
    return {
        "model": model,
        "score": score,
        "split": split,
        "id_accuracy": id_accuracy,
        "id_macro_f1": id_macro_f1,
        "has_raw_decision_scores": has_raw_decision_scores,
        "auroc": roc_auc_score(y_true, values),
        "aupr": average_precision_score(y_true, values),
        "fpr95": _fpr95(y_true, values),
        **best,
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _map_point(box: tuple[int, int, int, int], x: float, y: float, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    x_norm = (x - xlim[0]) / max(xlim[1] - xlim[0], 1e-12)
    y_norm = (y - ylim[0]) / max(ylim[1] - ylim[0], 1e-12)
    return x0 + x_norm * (x1 - x0), y1 - y_norm * (y1 - y0)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    x0, y0, x1, y1 = box
    title_font = _font(26, bold=True)
    label_font = _font(18)
    tick_font = _font(14)
    draw.text((x0, y0 - 50), title, fill=(20, 20, 20), font=title_font)
    for i in range(6):
        x_value = xlim[0] + i * (xlim[1] - xlim[0]) / 5
        y_value = ylim[0] + i * (ylim[1] - ylim[0]) / 5
        x, _ = _map_point(box, x_value, ylim[0], xlim, ylim)
        _, y = _map_point(box, xlim[0], y_value, xlim, ylim)
        draw.line((x, y0, x, y1), fill=(225, 225, 225), width=1)
        draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
        draw.text((x - 18, y1 + 10), f"{x_value:.2f}", fill=(70, 70, 70), font=tick_font)
        draw.text((x0 - 52, y - 8), f"{y_value:.2f}", fill=(70, 70, 70), font=tick_font)
    draw.rectangle(box, outline=(25, 25, 25), width=2)
    draw.text(((x0 + x1) // 2 - 80, y1 + 48), xlabel, fill=(20, 20, 20), font=label_font)
    draw.text((x0 - 98, (y0 + y1) // 2 - 8), ylabel, fill=(20, 20, 20), font=label_font)


def _plot_accuracy_vs_auroc(summary_rows: list[dict]) -> Path:
    output_path = RESULT_DIR / "experiment4_accuracy_vs_auroc.png"
    width, height = 1300, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    box = (140, 125, 1180, 745)
    classifier_rows = [row for row in summary_rows if row["id_accuracy"] is not None]
    baseline_rows = [row for row in summary_rows if row["id_accuracy"] is None]
    x_values = [float(row["id_accuracy"]) for row in classifier_rows]
    y_values = [float(row["best_auroc"]) for row in classifier_rows + baseline_rows]
    xlim = (max(0.0, min(x_values) - 0.02), min(1.0, max(x_values) + 0.02))
    ylim = (max(0.0, min(y_values) - 0.04), min(1.0, max(y_values) + 0.04))
    _draw_axes(
        draw,
        box,
        "Closed-set accuracy vs best OOD AUROC",
        "ID Accuracy",
        "OOD AUROC",
        xlim,
        ylim,
    )

    colors = {
        "LR": (37, 99, 235),
        "MLP": (220, 38, 38),
        "GNB": (217, 119, 6),
        "LDA": (5, 150, 105),
        "QDA": (124, 58, 237),
    }
    label_font = _font(17, bold=True)
    small_font = _font(14)
    for row in classifier_rows:
        model = str(row["model"])
        x, y = _map_point(box, float(row["id_accuracy"]), float(row["best_auroc"]), xlim, ylim)
        color = colors.get(model, (60, 60, 60))
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
        label = f"{model} ({row['best_score']})"
        draw.text((x + 12, y - 18), label, fill=color, font=label_font)
        draw.text((x + 12, y + 2), f"acc={float(row['id_accuracy']):.3f}, auroc={float(row['best_auroc']):.3f}", fill=(50, 50, 50), font=small_font)

    baseline_colors = {
        "LOF baseline": (14, 116, 144),
        "IsolationForest baseline": (107, 114, 128),
    }
    for idx, row in enumerate(baseline_rows):
        color = baseline_colors.get(str(row["model"]), (90, 90, 90))
        _, y = _map_point(box, xlim[0], float(row["best_auroc"]), xlim, ylim)
        draw.line((box[0], y, box[2], y), fill=(*color, 170), width=3)
        draw.text((box[0] + 12, y - 24 - idx * 2), f"{row['model']} AUROC={float(row['best_auroc']):.3f}", fill=color, font=_font(16, bold=True))

    x = np.array(x_values)
    y = np.array([float(row["best_auroc"]) for row in classifier_rows])
    corr = float(np.corrcoef(x, y)[0, 1]) if len(classifier_rows) > 1 else float("nan")
    note_font = _font(18)
    draw.rounded_rectangle((150, 770, 720, 845), radius=8, fill=(245, 245, 245), outline=(180, 180, 180), width=1)
    draw.text((170, 790), f"Pearson r = {corr:.3f}", fill=(25, 25, 25), font=note_font)
    draw.text((170, 818), "LOF/IF are OOD baselines; they have no closed-set classifier accuracy.", fill=(25, 25, 25), font=note_font)
    image.save(output_path)
    return output_path


def _draw_hist_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    histograms: dict[str, np.ndarray],
    colors: dict[str, tuple[int, int, int]],
) -> None:
    x0, y0, x1, y1 = box
    title_font = _font(18, bold=True)
    tick_font = _font(11)
    draw.text((x0, y0 - 28), title, fill=(20, 20, 20), font=title_font)
    max_density = max(float(values.max()) for values in histograms.values())
    max_density = max(max_density, 1e-12)
    draw.rectangle(box, outline=(30, 30, 30), width=1)
    for i in range(6):
        x = x0 + i * (x1 - x0) / 5
        draw.line((x, y0, x, y1), fill=(232, 232, 232), width=1)
        draw.text((x - 10, y1 + 5), f"{i / 5:.1f}", fill=(75, 75, 75), font=tick_font)
    for group, values in histograms.items():
        color = colors[group]
        bin_width = (x1 - x0) / len(values)
        for idx, value in enumerate(values):
            left = x0 + idx * bin_width
            right = left + bin_width * 0.86
            top = y1 - (float(value) / max_density) * (y1 - y0)
            draw.rectangle((left, top, right, y1), fill=(*color, 75), outline=(*color, 120))


def _plot_msp_histograms(confidence_by_model: dict[str, dict[str, np.ndarray]]) -> Path:
    output_path = RESULT_DIR / "experiment4_msp_confidence_histogram.png"
    width, height = 1700, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(28, bold=True)
    draw.text((50, 32), "MSP confidence distribution by classifier", fill=(15, 15, 15), font=title_font)
    draw.text((50, 70), "MSP confidence = max predicted class probability. High OOD confidence indicates overconfidence.", fill=(55, 55, 55), font=_font(17))

    colors = {
        "ID": (37, 99, 235),
        "Near-OOD": (220, 38, 38),
        "Far-OOD": (5, 150, 105),
    }
    bins = np.linspace(0.0, 1.0, 21)
    models = list(confidence_by_model)
    panel_w, panel_h = 490, 300
    positions = [
        (70, 170),
        (610, 170),
        (1150, 170),
        (70, 610),
        (610, 610),
    ]
    for model, (x, y) in zip(models, positions):
        hists = {}
        for group, values in confidence_by_model[model].items():
            hist, _ = np.histogram(values, bins=bins, density=True)
            hists[group] = hist
        _draw_hist_panel(draw, (x, y, x + panel_w, y + panel_h), model, hists, colors)

    legend_x, legend_y = 1150, 620
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 410, legend_y + 130), radius=8, fill=(248, 248, 248), outline=(190, 190, 190))
    for idx, (group, color) in enumerate(colors.items()):
        yy = legend_y + 22 + idx * 34
        draw.rectangle((legend_x + 22, yy, legend_x + 52, yy + 18), fill=(*color, 90), outline=color)
        draw.text((legend_x + 66, yy - 2), group, fill=(35, 35, 35), font=_font(16, bold=True))
    draw.text((50, height - 60), "If Near-OOD mass remains near 1.0, the classifier is overconfident on held-out banking intents.", fill=(55, 55, 55), font=_font(18))
    image.save(output_path)
    return output_path


def _write_report(
    metric_rows: list[dict],
    summary_rows: list[dict],
    overconfidence_rows: list[dict],
    scatter_path: Path,
    hist_path: Path,
) -> Path:
    report_path = RESULT_DIR / "experiment4_accuracy_vs_ood_report.md"
    summary_columns = [
        "model",
        "id_accuracy",
        "id_macro_f1",
        "best_score",
        "best_auroc",
        "best_aupr",
        "best_fpr95",
        "best_f1",
        "best_precision",
        "best_recall",
        "best_fpr",
        "best_accuracy",
    ]
    metric_columns = [
        "model",
        "score",
        "auroc",
        "aupr",
        "fpr95",
        "f1",
        "precision",
        "recall",
        "fpr",
        "accuracy",
    ]
    over_columns = [
        "model",
        "group",
        "total",
        "high_conf_count",
        "high_conf_rate",
        "mean_confidence",
        "median_confidence",
    ]

    overall_rows = [
        row for row in metric_rows
        if row["split"] == "Overall"
    ]
    over_rows = [
        row for row in overconfidence_rows
        if row["group"] in {"Near-OOD", "Far-OOD", "All-OOD"}
    ]
    parts = [
        "# Experiment 4: Classification Accuracy vs OOD Reliability",
        "",
        "## Classifier Summary",
        "",
        _markdown_table(summary_rows, summary_columns),
        "",
        "## OOD Metrics by Score (Overall)",
        "",
        _markdown_table(overall_rows, metric_columns),
        "",
        "## MSP Overconfidence (confidence >= 0.9)",
        "",
        _markdown_table(over_rows, over_columns),
        "",
        "## Plots",
        "",
        f"- Accuracy vs AUROC: `{scatter_path}`",
        f"- MSP confidence histogram: `{hist_path}`",
        "",
    ]
    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path


def main() -> None:
    train, id_test, ood_test = _load_data()
    test_embeddings = ood_test["embeddings"]
    ood_types = ood_test["ood_types"].astype(str)
    masks = _split_masks(ood_types)

    metric_rows: list[dict] = []
    overconfidence_rows: list[dict] = []
    confidence_by_model: dict[str, dict[str, np.ndarray]] = {}

    for model_name in CLASSIFIER_NAMES:
        display_name = MODEL_DISPLAY.get(model_name, model_name.upper())
        with classifier_path(model_name).open("rb") as f:
            clf = pickle.load(f)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            id_pred = clf.predict(id_test["embeddings"])
        id_accuracy = accuracy_score(id_test["labels"], id_pred)
        id_macro_f1 = f1_score(id_test["labels"], id_pred, average="macro")
        has_raw_scores = hasattr(clf, "decision_function")

        probs = _predict_probabilities(clf, test_embeddings)
        confidence = probs.max(axis=1)
        confidence_by_model[display_name] = {
            "ID": confidence[ood_types == "id"],
            "Near-OOD": confidence[ood_types == "near"],
            "Far-OOD": confidence[ood_types == "far"],
        }

        for group, group_mask in {
            "ID": ood_types == "id",
            "Near-OOD": ood_types == "near",
            "Far-OOD": ood_types == "far",
            "All-OOD": ood_types != "id",
        }.items():
            values = confidence[group_mask]
            high_count = int(np.sum(values >= CONFIDENCE_THRESHOLD))
            overconfidence_rows.append(
                {
                    "model": display_name,
                    "group": group,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "total": int(len(values)),
                    "high_conf_count": high_count,
                    "high_conf_rate": high_count / len(values) if len(values) else 0.0,
                    "mean_confidence": float(np.mean(values)) if len(values) else math.nan,
                    "median_confidence": float(np.median(values)) if len(values) else math.nan,
                    "p90_confidence": float(np.percentile(values, 90)) if len(values) else math.nan,
                }
            )

        score_values = _score_functions(clf, test_embeddings)
        for score_name, values in score_values.items():
            for split, mask in masks.items():
                y_true = (ood_types[mask] != "id").astype(int)
                metric_rows.append(
                    _evaluate_score(
                        display_name,
                        score_name,
                        split,
                        id_accuracy,
                        id_macro_f1,
                        has_raw_scores,
                        y_true,
                        values[mask],
                    )
                )

    baseline_models = {
        "LOF baseline": (
            "negative_score_samples",
            LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=1),
            lambda model, values: -model.score_samples(values),
        ),
        "IsolationForest baseline": (
            "negative_decision_function",
            IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=1),
            lambda model, values: -model.decision_function(values),
        ),
    }
    for display_name, (score_name, model, scorer) in baseline_models.items():
        model.fit(train["embeddings"])
        values = scorer(model, test_embeddings)
        for split, mask in masks.items():
            y_true = (ood_types[mask] != "id").astype(int)
            metric_rows.append(
                _evaluate_score(
                    display_name,
                    score_name,
                    split,
                    None,
                    None,
                    False,
                    y_true,
                    values[mask],
                )
            )

    summary_rows: list[dict] = []
    for model in sorted({row["model"] for row in metric_rows}):
        rows = [
            row for row in metric_rows
            if row["model"] == model and row["split"] == "Overall"
        ]
        best = max(rows, key=lambda row: float(row["auroc"]))
        near = next(
            row for row in metric_rows
            if row["model"] == model and row["score"] == best["score"] and row["split"] == "Near-OOD"
        )
        far = next(
            row for row in metric_rows
            if row["model"] == model and row["score"] == best["score"] and row["split"] == "Far-OOD"
        )
        summary_rows.append(
            {
                "model": model,
                "id_accuracy": best["id_accuracy"],
                "id_macro_f1": best["id_macro_f1"],
                "best_score": best["score"],
                "best_auroc": best["auroc"],
                "best_aupr": best["aupr"],
                "best_fpr95": best["fpr95"],
                "best_f1": best["f1"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "best_fpr": best["fpr"],
                "best_accuracy": best["accuracy"],
                "near_auroc_for_best_score": near["auroc"],
                "far_auroc_for_best_score": far["auroc"],
            }
        )
    summary_rows.sort(key=lambda row: float(row["best_auroc"]), reverse=True)

    metric_fields = [
        "model",
        "score",
        "split",
        "id_accuracy",
        "id_macro_f1",
        "has_raw_decision_scores",
        "auroc",
        "aupr",
        "fpr95",
        "threshold",
        "f1",
        "precision",
        "recall",
        "tpr",
        "fpr",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    summary_fields = [
        "model",
        "id_accuracy",
        "id_macro_f1",
        "best_score",
        "best_auroc",
        "best_aupr",
        "best_fpr95",
        "best_f1",
        "best_precision",
        "best_recall",
        "best_fpr",
        "best_accuracy",
        "near_auroc_for_best_score",
        "far_auroc_for_best_score",
    ]
    overconfidence_fields = [
        "model",
        "group",
        "confidence_threshold",
        "total",
        "high_conf_count",
        "high_conf_rate",
        "mean_confidence",
        "median_confidence",
        "p90_confidence",
    ]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metric_path = RESULT_DIR / "experiment4_classifier_ood_metrics.csv"
    summary_path = RESULT_DIR / "experiment4_classifier_summary.csv"
    overconfidence_path = RESULT_DIR / "experiment4_msp_overconfidence.csv"
    _write_csv(metric_path, metric_rows, metric_fields)
    _write_csv(summary_path, summary_rows, summary_fields)
    _write_csv(overconfidence_path, overconfidence_rows, overconfidence_fields)

    scatter_path = _plot_accuracy_vs_auroc(summary_rows)
    hist_path = _plot_msp_histograms(confidence_by_model)
    report_path = _write_report(metric_rows, summary_rows, overconfidence_rows, scatter_path, hist_path)

    print(f"Wrote {metric_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {overconfidence_path}")
    print(f"Wrote {scatter_path}")
    print(f"Wrote {hist_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
