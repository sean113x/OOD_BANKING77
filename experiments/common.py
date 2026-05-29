"""Shared experiment and interactive OOD utilities."""

from __future__ import annotations

import csv
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy as scipy_entropy
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_WISE_DIR = PROJECT_ROOT / "embedding_class-wise methods"
DISTRIBUTION_DIR = PROJECT_ROOT / "embedding_distribution analysis methods"
EMBEDDED_DIR = PROJECT_ROOT / "dataset" / "preprocessed" / "embedded"
PREPROCESSED_DIR = PROJECT_ROOT / "dataset" / "preprocessed"
RESULT_DIR = PROJECT_ROOT / "experiments" / "results"
EXPERIMENT_MODEL_DIR = PROJECT_ROOT / "experiments" / "models"
CLASSIFIER_DIR = PROJECT_ROOT / "embedding_scoring" / "classifier"
ANOMALY_MODEL_DIR = PROJECT_ROOT / "embeddinig_anomaly" / "models"

for path in (PROJECT_ROOT, CLASS_WISE_DIR, DISTRIBUTION_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from centroid_distance import CentroidDistanceModel  # noqa: E402
from gaussian_mixture import GaussianMixtureOODModel  # noqa: E402
from global_mahalanobis import GlobalMahalanobisModel  # noqa: E402
from knn_distance import KNNDistanceModel  # noqa: E402
from mahalanobis_distance import MahalanobisDistanceModel  # noqa: E402
from one_class_svm import OneClassSVMModel  # noqa: E402
from pca_reconstruction import PCAReconstructionModel  # noqa: E402
from radius_threshold import RadiusThresholdModel  # noqa: E402


Split = dict[str, np.ndarray]

OOD_CSV_COLUMNS = (
    "text",
    "label",
    "label_text",
    "is_ood",
    "ood_type",
    "source",
    "source_label",
    "source_label_text",
)

SPLIT_LABELS = ("Overall", "Near-OOD", "Far-OOD")


def classifier_path(name: str) -> Path:
    return CLASSIFIER_DIR / f"clf_{name}.pkl"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ood_row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in OOD_CSV_COLUMNS)


def materialize_derived_embedding_split(split_name: str) -> Path:
    """Create validation/eval embeddings from the original OOD_test embeddings."""

    if split_name not in {"OOD_validation", "OOD_test_eval"}:
        raise ValueError(f"Cannot materialize derived split {split_name!r}.")

    output_path = EMBEDDED_DIR / f"{split_name}_embeddings.npz"
    source_embedding_path = EMBEDDED_DIR / "OOD_test_embeddings.npz"
    source_csv_path = PREPROCESSED_DIR / "OOD_test.csv"
    target_csv_path = PREPROCESSED_DIR / f"{split_name}.csv"

    if not source_embedding_path.exists():
        raise FileNotFoundError(
            f"Missing {source_embedding_path}. Run `python BERT/embed.py` first."
        )
    if not target_csv_path.exists():
        raise FileNotFoundError(
            f"Missing {target_csv_path}. Run "
            "`uv run python dataset/preprocessed/create_ood_validation_split.py` first."
        )

    source_rows = read_csv_rows(source_csv_path)
    target_rows = read_csv_rows(target_csv_path)

    indices_by_key: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    for idx, row in enumerate(source_rows):
        indices_by_key[_ood_row_key(row)].append(idx)

    target_indices: list[int] = []
    for row in target_rows:
        key = _ood_row_key(row)
        if not indices_by_key[key]:
            raise ValueError(f"Could not match row in {target_csv_path}: {row}")
        target_indices.append(indices_by_key[key].popleft())

    indices = np.array(target_indices, dtype=np.int64)
    with np.load(source_embedding_path, allow_pickle=False) as data:
        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            value = data[key]
            if value.shape and len(value) == len(source_rows):
                payload[key] = value[indices]
            else:
                payload[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_embedding_split(split_name: str) -> Split:
    path = EMBEDDED_DIR / f"{split_name}_embeddings.npz"
    if not path.exists() and split_name in {"OOD_validation", "OOD_test_eval"}:
        materialize_derived_embedding_split(split_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_standard_splits() -> tuple[Split, Split, Split, Split]:
    return (
        load_embedding_split("OOD_train"),
        load_embedding_split("classification_test"),
        load_embedding_split("OOD_validation"),
        load_embedding_split("OOD_test_eval"),
    )


def y_true_from_split(split: Split) -> np.ndarray:
    if "is_ood" not in split:
        return np.zeros(len(split["embeddings"]), dtype=int)
    return split["is_ood"].astype(str).astype(int)


def ood_types_from_split(split: Split) -> np.ndarray:
    if "ood_types" not in split:
        return np.full(len(split["embeddings"]), "id")
    return split["ood_types"].astype(str)


def split_masks(ood_types: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Overall": np.ones(len(ood_types), dtype=bool),
        "Near-OOD": (ood_types == "id") | (ood_types == "near"),
        "Far-OOD": (ood_types == "id") | (ood_types == "far"),
    }


def _safe_metric(metric_fn: Callable[[], float]) -> float:
    try:
        return float(metric_fn())
    except ValueError:
        return float("nan")


def fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    candidates = np.where(tpr >= 0.95)[0]
    return float(fpr[candidates[0]]) if len(candidates) else 1.0


def metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return float(np.max(scores)) if len(scores) else 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def id_percentile_threshold(y_true: np.ndarray, scores: np.ndarray, percentile: float = 95.0) -> float:
    id_scores = scores[y_true == 0]
    if len(id_scores) == 0:
        return float(np.percentile(scores, percentile))
    return float(np.percentile(id_scores, percentile))


def metric_row(
    *,
    experiment: str,
    family: str,
    model: str,
    score: str,
    hyperparameters: str,
    threshold_source: str,
    data_split: str,
    metric_split: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    fit_seconds: float | None = None,
    score_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_metrics = metrics_at_threshold(y_true, scores, threshold)
    row: dict[str, Any] = {
        "experiment": experiment,
        "family": family,
        "model": model,
        "score": score,
        "hyperparameters": hyperparameters,
        "threshold_source": threshold_source,
        "data_split": data_split,
        "metric_split": metric_split,
        "n_samples": int(len(scores)),
        "n_id": int(np.sum(y_true == 0)),
        "n_ood": int(np.sum(y_true == 1)),
        "auroc": _safe_metric(lambda: roc_auc_score(y_true, scores)),
        "aupr": _safe_metric(lambda: average_precision_score(y_true, scores)),
        "fpr95": fpr95(y_true, scores),
        **threshold_metrics,
    }
    if fit_seconds is not None:
        row["fit_seconds"] = fit_seconds
    if score_seconds is not None:
        row["score_seconds"] = score_seconds
    if extra:
        row.update(extra)
    return row


def evaluate_split_rows(
    *,
    experiment: str,
    family: str,
    model: str,
    score: str,
    hyperparameters: str,
    threshold_source: str,
    data_split: str,
    split: Split,
    scores: np.ndarray,
    threshold: float,
    fit_seconds: float | None = None,
    score_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    y_true = y_true_from_split(split)
    ood_types = ood_types_from_split(split)
    rows: list[dict[str, Any]] = []
    for metric_split, mask in split_masks(ood_types).items():
        rows.append(
            metric_row(
                experiment=experiment,
                family=family,
                model=model,
                score=score,
                hyperparameters=hyperparameters,
                threshold_source=threshold_source,
                data_split=data_split,
                metric_split=metric_split,
                y_true=y_true[mask],
                scores=scores[mask],
                threshold=threshold,
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
                extra=extra,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _format_terminal_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)] + "..."


def print_metric_table(
    rows: list[dict[str, Any]],
    title: str,
    *,
    data_split: str | None = "eval",
    metric_split: str | None = "Overall",
    max_rows: int = 120,
) -> None:
    display_rows = rows
    if data_split is not None:
        display_rows = [row for row in display_rows if row.get("data_split") == data_split]
    if metric_split is not None:
        display_rows = [row for row in display_rows if row.get("metric_split") == metric_split]
    if not display_rows:
        display_rows = rows

    shown_rows = display_rows[:max_rows]
    omitted = max(0, len(display_rows) - len(shown_rows))
    columns = [
        "model",
        "score",
        "hyperparameters",
        "threshold_source",
        "data_split",
        "metric_split",
        "auroc",
        "aupr",
        "fpr95",
        "f1",
        "precision",
        "recall",
        "fpr",
        "accuracy",
    ]
    max_widths = {
        "model": 30,
        "score": 24,
        "hyperparameters": 34,
        "threshold_source": 20,
        "data_split": 10,
        "metric_split": 18,
        "auroc": 8,
        "aupr": 8,
        "fpr95": 8,
        "f1": 8,
        "precision": 9,
        "recall": 8,
        "fpr": 8,
        "accuracy": 8,
    }

    formatted_rows: list[dict[str, str]] = []
    for row in shown_rows:
        formatted = {}
        for column in columns:
            formatted[column] = _shorten(_format_terminal_value(row.get(column, "")), max_widths[column])
        formatted_rows.append(formatted)

    widths = {
        column: max(
            len(column),
            *(len(row[column]) for row in formatted_rows),
        )
        for column in columns
    }

    print(f"\n{title}")
    print(f"Showing {len(shown_rows)} of {len(display_rows)} rows in terminal. Full results are saved to CSV.")
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in formatted_rows:
        print(" | ".join(row[column].ljust(widths[column]) for column in columns))
    if omitted:
        print(f"... {omitted} more rows omitted from terminal output.")


def _pickle_load(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        with path.open("rb") as f:
            return pickle.load(f)


def _pickle_save(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def predict_probabilities(clf: Any, embeddings: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        probs = clf.predict_proba(embeddings)
    if not np.isfinite(probs).all():
        raise FloatingPointError(f"{type(clf).__name__} produced non-finite probabilities.")
    return probs


def predict_log_scores(clf: Any, embeddings: np.ndarray) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            scores = clf.decision_function(embeddings)
        if scores.ndim == 1:
            scores = scores[:, None]
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{type(clf).__name__} produced non-finite decision scores.")
        return scores

    probs = np.clip(predict_probabilities(clf, embeddings), 1e-12, 1.0)
    return np.log(probs)


def msp_score(clf: Any, embeddings: np.ndarray) -> np.ndarray:
    probs = predict_probabilities(clf, embeddings)
    return 1.0 - probs.max(axis=1)


def entropy_score(clf: Any, embeddings: np.ndarray) -> np.ndarray:
    return scipy_entropy(predict_probabilities(clf, embeddings), axis=1)


def margin_score(clf: Any, embeddings: np.ndarray) -> np.ndarray:
    probs = predict_probabilities(clf, embeddings)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    return 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])


def max_logit_score(clf: Any, embeddings: np.ndarray) -> np.ndarray:
    logits = predict_log_scores(clf, embeddings)
    return -logits.max(axis=1)


def energy_score(clf: Any, embeddings: np.ndarray) -> np.ndarray:
    logits = predict_log_scores(clf, embeddings)
    return -logsumexp(logits, axis=1)


def build_classifier(name: str) -> Any:
    if name == "lr":
        return LogisticRegression(max_iter=3000, C=10.0, solver="saga", random_state=42, n_jobs=1)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(512, 256),
            max_iter=500,
            alpha=0.001,
            tol=1e-5,
            n_iter_no_change=20,
            random_state=42,
        )
    if name == "gnb":
        return Pipeline([
            ("pca", PCA(n_components=100, whiten=True, random_state=42)),
            ("gnb", GaussianNB(var_smoothing=0.01)),
        ])
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "qda":
        return Pipeline([
            ("pca", PCA(n_components=75, whiten=True, random_state=42)),
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0.01)),
        ])
    raise ValueError(f"Unknown classifier: {name}")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family_id: int
    family_name: str
    score_name: str
    hyperparameters: str
    model_path: Path
    build: Callable[[], Any]
    fit: Callable[[Any, Split, bool], Any]
    score: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]]
    load: Callable[[Path], Any] | None = None
    save: Callable[[Any, Path], None] | None = None


def _fit_custom(model: Any, train: Split, save: bool) -> Any:
    return model.fit(train["embeddings"], train["label_texts"], save=save)


def _score_custom(model: Any, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return model.score_embeddings(embeddings)


def _fit_anomaly(model: Any, train: Split, save: bool) -> Any:
    model.fit(train["embeddings"])
    if save:
        _pickle_save(model, model.model_path if hasattr(model, "model_path") else EXPERIMENT_MODEL_DIR / "anomaly.pkl")
    return model


def _score_isolation_forest(model: Any, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = -model.decision_function(embeddings)
    return scores.astype(np.float32), np.full(len(scores), "isolation_forest")


def _score_lof(model: Any, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = -model.score_samples(embeddings)
    return scores.astype(np.float32), np.full(len(scores), "lof")


def _fit_classifier(model: Any, train: Split, save: bool) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*convergence.*")
        warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")
        warnings.filterwarnings("ignore", message=".*covariance matrix of class.*")
        model.fit(train["embeddings"], train["labels"])
    return model


def _classifier_score_fn(score_fn: Callable[[Any, np.ndarray], np.ndarray]) -> Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def score(model: Any, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = score_fn(model, embeddings).astype(np.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            labels = model.predict(embeddings).astype(str)
        return values, labels

    return score


def _custom_spec(
    *,
    key: str,
    display_name: str,
    family_id: int,
    family_name: str,
    score_name: str,
    hyperparameters: str,
    model_path: Path,
    cls: Any,
    kwargs: dict[str, Any] | None = None,
) -> ModelSpec:
    kwargs = kwargs or {}
    return ModelSpec(
        key=key,
        display_name=display_name,
        family_id=family_id,
        family_name=family_name,
        score_name=score_name,
        hyperparameters=hyperparameters,
        model_path=model_path,
        build=lambda: cls(model_path, **kwargs),
        fit=_fit_custom,
        score=_score_custom,
        load=cls.load,
    )


def class_wise_specs() -> list[ModelSpec]:
    output_dir = CLASS_WISE_DIR / "outputs"
    return [
        _custom_spec(
            key="centroid",
            display_name="Centroid distance",
            family_id=1,
            family_name="Class-wise boundary",
            score_name="nearest_centroid_distance",
            hyperparameters="distance=cosine, threshold_percentile=95",
            model_path=output_dir / "centroid_distance_model.npz",
            cls=CentroidDistanceModel,
        ),
        _custom_spec(
            key="radius",
            display_name="Class-wise radius",
            family_id=1,
            family_name="Class-wise boundary",
            score_name="distance_minus_radius",
            hyperparameters="distance=cosine, radius_percentile=90, minimum_radius=1e-6",
            model_path=output_dir / "radius_threshold_model.npz",
            cls=RadiusThresholdModel,
        ),
        _custom_spec(
            key="class_mahalanobis",
            display_name="Class-wise Mahalanobis",
            family_id=1,
            family_name="Class-wise boundary",
            score_name="min_class_mahalanobis",
            hyperparameters="regularization_lambda=1e-2, threshold_percentile=95",
            model_path=output_dir / "mahalanobis_distance_model.npz",
            cls=MahalanobisDistanceModel,
        ),
    ]


def distribution_specs() -> list[ModelSpec]:
    output_dir = DISTRIBUTION_DIR / "outputs"
    return [
        _custom_spec(
            key="knn",
            display_name="kNN distance",
            family_id=2,
            family_name="Distance/support/density",
            score_name="mean_knn_distance",
            hyperparameters="k=1, distance=cosine, threshold_percentile=95",
            model_path=output_dir / "knn_distance_model.npz",
            cls=KNNDistanceModel,
            kwargs={"n_neighbors": 1},
        ),
        _custom_spec(
            key="global_mahalanobis",
            display_name="Global Mahalanobis",
            family_id=2,
            family_name="Distance/support/density",
            score_name="global_mahalanobis",
            hyperparameters="LedoitWolf shrinkage, threshold_percentile=95",
            model_path=output_dir / "global_mahalanobis_model.npz",
            cls=GlobalMahalanobisModel,
        ),
        _custom_spec(
            key="gmm",
            display_name="Gaussian Mixture",
            family_id=2,
            family_name="Distance/support/density",
            score_name="negative_log_likelihood",
            hyperparameters="n_components=62, covariance=full, reg_covar=1e-5",
            model_path=output_dir / "gaussian_mixture_model.pkl",
            cls=GaussianMixtureOODModel,
            kwargs={"n_components": 62, "covariance_type": "full", "reg_covar": 1e-5, "max_iter": 200},
        ),
        _custom_spec(
            key="one_class_svm",
            display_name="One-Class SVM",
            family_id=2,
            family_name="Distance/support/density",
            score_name="negative_decision_function",
            hyperparameters="kernel=rbf, nu=0.3, gamma=30",
            model_path=output_dir / "one_class_svm_model.pkl",
            cls=OneClassSVMModel,
            kwargs={"nu": 0.3, "gamma": 30.0},
        ),
        _custom_spec(
            key="pca_reconstruction",
            display_name="PCA reconstruction",
            family_id=2,
            family_name="Distance/support/density",
            score_name="reconstruction_mse",
            hyperparameters="n_components=256, threshold_percentile=95",
            model_path=output_dir / "pca_reconstruction_model.pkl",
            cls=PCAReconstructionModel,
            kwargs={"n_components": 256},
        ),
    ]


def anomaly_specs() -> list[ModelSpec]:
    def if_builder() -> Any:
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=1)
        model.model_path = ANOMALY_MODEL_DIR / "isolation_forest.pkl"
        return model

    def lof_builder() -> Any:
        model = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=1)
        model.model_path = ANOMALY_MODEL_DIR / "lof.pkl"
        return model

    return [
        ModelSpec(
            key="isolation_forest",
            display_name="Isolation Forest",
            family_id=3,
            family_name="Anomaly detection",
            score_name="negative_decision_function",
            hyperparameters="n_estimators=200, contamination=auto",
            model_path=ANOMALY_MODEL_DIR / "isolation_forest.pkl",
            build=if_builder,
            fit=_fit_anomaly,
            score=_score_isolation_forest,
            load=_pickle_load,
            save=_pickle_save,
        ),
        ModelSpec(
            key="lof",
            display_name="Local Outlier Factor",
            family_id=3,
            family_name="Anomaly detection",
            score_name="negative_score_samples",
            hyperparameters="n_neighbors=20, novelty=True",
            model_path=ANOMALY_MODEL_DIR / "lof.pkl",
            build=lof_builder,
            fit=_fit_anomaly,
            score=_score_lof,
            load=_pickle_load,
            save=_pickle_save,
        ),
    ]


def classifier_specs() -> list[ModelSpec]:
    classifier_names = {
        "lr": "Logistic Regression",
        "mlp": "MLP",
        "gnb": "Gaussian Naive Bayes",
        "lda": "LDA",
        "qda": "QDA",
    }
    score_functions: list[tuple[str, str, Callable[[Any, np.ndarray], np.ndarray]]] = [
        ("msp", "MSP", msp_score),
        ("entropy", "Entropy", entropy_score),
        ("margin", "Margin", margin_score),
        ("max_logit", "MaxLogit", max_logit_score),
        ("energy", "Energy", energy_score),
    ]

    specs: list[ModelSpec] = []
    for clf_name, clf_display in classifier_names.items():
        for score_key, score_display, score_fn in score_functions:
            specs.append(
                ModelSpec(
                    key=f"{clf_name}_{score_key}",
                    display_name=f"{clf_display} + {score_display}",
                    family_id=4,
                    family_name="Classifier-output scoring",
                    score_name=score_display,
                    hyperparameters="default classifier settings",
                    model_path=classifier_path(clf_name),
                    build=lambda name=clf_name: build_classifier(name),
                    fit=_fit_classifier,
                    score=_classifier_score_fn(score_fn),
                    load=_pickle_load,
                    save=_pickle_save,
                )
            )
    return specs


def all_model_specs() -> list[ModelSpec]:
    return class_wise_specs() + distribution_specs() + anomaly_specs() + classifier_specs()


def specs_by_family() -> dict[int, list[ModelSpec]]:
    grouped: dict[int, list[ModelSpec]] = defaultdict(list)
    for spec in all_model_specs():
        grouped[spec.family_id].append(spec)
    return dict(grouped)


def fit_model(spec: ModelSpec, train: Split, save: bool = True) -> tuple[Any, float]:
    model = spec.build()
    start = time.perf_counter()
    model = spec.fit(model, train, save)
    fit_seconds = time.perf_counter() - start
    if save and spec.save is not None:
        spec.save(model, spec.model_path)
    return model, fit_seconds


def load_or_fit_model(spec: ModelSpec, train: Split, force_train: bool = False) -> tuple[Any, bool, float]:
    if spec.model_path.exists() and not force_train and spec.load is not None:
        start = time.perf_counter()
        model = spec.load(spec.model_path)
        return model, True, time.perf_counter() - start
    model, fit_seconds = fit_model(spec, train, save=True)
    return model, False, fit_seconds


def score_model(spec: ModelSpec, model: Any, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
        scores, labels = spec.score(model, embeddings)
    return scores, labels, time.perf_counter() - start


def calibrate_threshold(
    spec: ModelSpec,
    model: Any,
    validation: Split,
    strategy: str = "validation_best_f1",
) -> tuple[float, np.ndarray, np.ndarray]:
    validation_scores, validation_labels, _ = score_model(spec, model, validation["embeddings"])
    y_validation = y_true_from_split(validation)
    if strategy == "validation_best_f1":
        threshold = best_f1_threshold(y_validation, validation_scores)
    elif strategy == "validation_id95":
        threshold = id_percentile_threshold(y_validation, validation_scores, 95.0)
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")
    return threshold, validation_scores, validation_labels


def classifier_accuracy(model: Any, split: Split) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        pred = model.predict(split["embeddings"]).astype(str)
    labels = split["labels"].astype(str)
    return (
        float(accuracy_score(labels, pred)),
        float(f1_score(labels, pred, average="macro")),
    )


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def known_class_centroids(train: Split) -> tuple[np.ndarray, np.ndarray]:
    embeddings = normalized_rows(train["embeddings"])
    labels = train["label_texts"].astype(str)
    classes = np.array(sorted(set(labels.tolist())))
    centroids = np.vstack([
        embeddings[labels == class_name].mean(axis=0)
        for class_name in classes
    ])
    return classes, normalized_rows(centroids)
