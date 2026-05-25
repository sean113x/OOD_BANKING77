"""Quick evaluation entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).parent
CLASS_WISE_DIR = ROOT / "embedding_class-wise methods"
DISTRIBUTION_DIR = ROOT / "embedding_distribution analysis methods"
EMBEDDED_DIR = ROOT / "dataset/preprocessed/embedded"

sys.path.insert(0, str(CLASS_WISE_DIR))
sys.path.insert(0, str(DISTRIBUTION_DIR))

from centroid_distance import CentroidDistanceModel  # noqa: E402
from gaussian_mixture import GaussianMixtureOODModel  # noqa: E402
from global_mahalanobis import GlobalMahalanobisModel  # noqa: E402
from knn_distance import KNNDistanceModel  # noqa: E402
from mahalanobis_distance import MahalanobisDistanceModel  # noqa: E402
from one_class_svm import OneClassSVMModel  # noqa: E402
from pca_reconstruction import PCAReconstructionModel  # noqa: E402
from radius_threshold import RadiusThresholdModel  # noqa: E402


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def best_accuracy_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positives_by_score = np.zeros(len(unique_scores), dtype=np.int64)
    np.add.at(positives_by_score, inverse, y_true.astype(int))
    negatives_by_score = counts - positives_by_score

    total_positive = int(np.sum(y_true))
    prefix_positive = np.concatenate([[0], np.cumsum(positives_by_score)])
    prefix_negative = np.concatenate([[0], np.cumsum(negatives_by_score)])
    accuracy = (total_positive - prefix_positive + prefix_negative) / len(y_true)
    best_idx = int(np.argmax(accuracy))

    if best_idx == 0:
        return float(unique_scores[0] - 1e-12)
    if best_idx == len(unique_scores):
        return float(unique_scores[-1] + 1e-12)
    return float((unique_scores[best_idx - 1] + unique_scores[best_idx]) / 2)


def print_metrics(name: str, split: str, y_true: np.ndarray, scores: np.ndarray) -> None:
    threshold = best_accuracy_threshold(y_true, scores)
    y_pred = (scores > threshold).astype(int)
    print(
        f"{name} [{split}]: "
        f"AUROC={roc_auc_score(y_true, scores):.4f}, "
        f"Acc={accuracy_score(y_true, y_pred):.4f}, "
        f"Precision={precision_score(y_true, y_pred, zero_division=0):.4f}, "
        f"Recall={recall_score(y_true, y_pred, zero_division=0):.4f}, "
        f"OptThr={threshold:.6g}"
    )


def print_metrics_at_threshold(name: str, split: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    y_pred = (scores > threshold).astype(int)
    print(
        f"{name} [{split}]: "
        f"AUROC={roc_auc_score(y_true, scores):.4f}, "
        f"Acc={accuracy_score(y_true, y_pred):.4f}, "
        f"Precision={precision_score(y_true, y_pred, zero_division=0):.4f}, "
        f"Recall={recall_score(y_true, y_pred, zero_division=0):.4f}, "
        f"Thr={threshold:.6g}"
    )


def masks_by_ood_type(test: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ood_types = test["ood_types"]
    near_mask = (ood_types == "id") | (ood_types == "near")
    far_mask = (ood_types == "id") | (ood_types == "far")
    return near_mask, far_mask


def print_split_metrics(name: str, y_true: np.ndarray, scores: np.ndarray, near_mask: np.ndarray, far_mask: np.ndarray) -> None:
    print_metrics(name, "overall", y_true, scores)
    print_metrics(name, "near", y_true[near_mask], scores[near_mask])
    print_metrics(name, "far", y_true[far_mask], scores[far_mask])


def print_split_metrics_at_threshold(
    name: str,
    label: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    near_mask: np.ndarray,
    far_mask: np.ndarray,
    threshold: float,
) -> None:
    print_metrics_at_threshold(name, f"overall_{label}", y_true, scores, threshold)
    print_metrics_at_threshold(name, f"near_{label}", y_true[near_mask], scores[near_mask], threshold)
    print_metrics_at_threshold(name, f"far_{label}", y_true[far_mask], scores[far_mask], threshold)


def evaluate(name: str, model, train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> None:
    print(f"\n{name}")
    model.fit(train["embeddings"], train["label_texts"], save=True)
    scores, _ = model.score_embeddings(test["embeddings"])
    y_true = test["is_ood"].astype(int)
    near_mask, far_mask = masks_by_ood_type(test)
    print_split_metrics(name, y_true, scores, near_mask, far_mask)


def evaluate_gmm_sweep(train: dict[str, np.ndarray], test: dict[str, np.ndarray], output_dir: Path) -> None:
    y_true = test["is_ood"].astype(int)
    near_mask, far_mask = masks_by_ood_type(test)
    train_embeddings = train["embeddings"]
    covariance_types = ["full", "tied", "diag", "spherical"]
    threshold_percentiles = [90, 95, 97, 98, 99]

    for covariance_type in covariance_types:
        name = f"gmm_62_{covariance_type}"
        print(f"\n{name}")
        model = GaussianMixtureOODModel(
            output_dir / f"{name}_model.pkl",
            n_components=62,
            covariance_type=covariance_type,
            reg_covar=1e-5,
            max_iter=200,
        )
        model.fit(train_embeddings, train["label_texts"], save=True)
        scores, _ = model.score_embeddings(test["embeddings"])
        train_scores, _ = model.score_embeddings(train_embeddings)

        print_split_metrics(name, y_true, scores, near_mask, far_mask)
        for percentile in threshold_percentiles:
            threshold = float(np.percentile(train_scores, percentile))
            print_split_metrics_at_threshold(name, f"p{percentile}", y_true, scores, near_mask, far_mask, threshold)


def evaluate_one_class_svm_sweep(train: dict[str, np.ndarray], test: dict[str, np.ndarray], output_dir: Path) -> None:
    y_true = test["is_ood"].astype(int)
    near_mask, far_mask = masks_by_ood_type(test)
    nu = 0.3
    for gamma in [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]:
        gamma_tag = str(gamma).replace(".", "_")
        name = f"one_class_svm_nu0_3_gamma{gamma_tag}"
        print(f"\n{name}")
        model = OneClassSVMModel(output_dir / f"{name}_model.pkl", nu=nu, gamma=gamma)
        model.fit(train["embeddings"], train["label_texts"], save=True)
        scores, _ = model.score_embeddings(test["embeddings"])
        print_split_metrics(name, y_true, scores, near_mask, far_mask)
        print_split_metrics_at_threshold(name, "default", y_true, scores, near_mask, far_mask, model.threshold)


def main() -> None:
    class_output_dir = CLASS_WISE_DIR / "outputs"
    distribution_output_dir = DISTRIBUTION_DIR / "outputs"
    train = load_npz(EMBEDDED_DIR / "OOD_train_embeddings.npz")
    test = load_npz(EMBEDDED_DIR / "OOD_test_embeddings.npz")

    evaluate("centroid", CentroidDistanceModel(class_output_dir / "centroid_distance_model.npz"), train, test)
    evaluate("radius", RadiusThresholdModel(class_output_dir / "radius_threshold_model.npz"), train, test)
    evaluate("mahalanobis", MahalanobisDistanceModel(class_output_dir / "mahalanobis_distance_model.npz"), train, test)

    evaluate("knn_distance", KNNDistanceModel(distribution_output_dir / "knn_distance_model.npz"), train, test)
    evaluate("global_mahalanobis", GlobalMahalanobisModel(distribution_output_dir / "global_mahalanobis_model.npz"), train, test)
    evaluate("pca_reconstruction", PCAReconstructionModel(distribution_output_dir / "pca_reconstruction_model.pkl"), train, test)
    evaluate_gmm_sweep(train, test, distribution_output_dir)
    evaluate_one_class_svm_sweep(train, test, distribution_output_dir)


if __name__ == "__main__":
    main()
