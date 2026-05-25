"""Class-wise Mahalanobis distance OOD model."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_MODEL_PATH = Path("embedding_class-wise methods/outputs/mahalanobis_distance_model.npz")
DEFAULT_BERT_PATH = Path("BERT/minilm_lora")


def _embed_texts(texts: str | list[str], model_dir: Path = DEFAULT_BERT_PATH) -> np.ndarray:
    from BERT.embedding_utils import embed_texts

    return embed_texts(texts, model_dir=model_dir, batch_size=32, show_progress=False)


class MahalanobisDistanceModel:
    """Fit class Gaussian statistics and score by minimum Mahalanobis distance."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        regularization_lambda: float = 1e-2,
        threshold_percentile: float = 95.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.regularization_lambda = regularization_lambda
        self.threshold_percentile = threshold_percentile
        self.classes: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.inv_vars: np.ndarray | None = None
        self.threshold: float | None = None

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_label_texts: np.ndarray,
        save: bool = True,
    ) -> "MahalanobisDistanceModel":
        embeddings = np.asarray(train_embeddings, dtype=np.float32)
        labels = np.asarray(train_label_texts)
        self.classes = np.array(sorted(set(labels.tolist())))
        self.means = np.vstack(
            [embeddings[labels == class_name].mean(axis=0) for class_name in self.classes]
        ).astype(np.float32)

        inv_vars = []
        for class_name in self.classes:
            class_embeddings = embeddings[labels == class_name]
            variance = class_embeddings.var(axis=0) + self.regularization_lambda
            inv_vars.append(1.0 / np.maximum(variance, 1e-12))
        self.inv_vars = np.vstack(inv_vars).astype(np.float32)

        train_scores, _ = self.score_embeddings(embeddings)
        self.threshold = float(np.percentile(train_scores, self.threshold_percentile))
        if save:
            self.save()
        return self

    def score_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.classes is None or self.means is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")
        if self.inv_vars is None:
            raise ValueError("Missing diagonal covariance parameters.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        all_scores = []
        for class_idx in range(len(self.classes)):
            diff = embeddings - self.means[class_idx]
            squared = np.sum(diff * diff * self.inv_vars[class_idx], axis=1)
            all_scores.append(np.sqrt(np.maximum(squared, 0.0)))

        scores_by_class = np.vstack(all_scores).T
        nearest_idx = np.argmin(scores_by_class, axis=1)
        scores = scores_by_class[np.arange(len(embeddings)), nearest_idx]
        nearest_classes = self.classes[nearest_idx]
        return scores.astype(np.float32), nearest_classes

    def predict_embeddings(
        self, embeddings: np.ndarray, threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, nearest_classes = self.score_embeddings(embeddings)
        active_threshold = self.threshold if threshold is None else threshold
        if active_threshold is None:
            raise ValueError("No threshold is set.")
        is_ood = scores > active_threshold
        return is_ood, scores, nearest_classes

    def score_text(
        self, texts: str | list[str], model_dir: str | Path = DEFAULT_BERT_PATH
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings = _embed_texts(texts, Path(model_dir))
        return self.score_embeddings(embeddings)

    def save(self, path: str | Path | None = None) -> None:
        if self.classes is None or self.means is None or self.threshold is None:
            raise ValueError("Cannot save before fit().")

        output_path = Path(path) if path is not None else self.model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classes": self.classes,
            "means": self.means,
            "inv_vars": self.inv_vars,
            "threshold": np.array(self.threshold, dtype=np.float32),
            "regularization_lambda": np.array(self.regularization_lambda, dtype=np.float32),
            "threshold_percentile": np.array(self.threshold_percentile, dtype=np.float32),
        }
        np.savez_compressed(output_path, **payload)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "MahalanobisDistanceModel":
        data = np.load(path, allow_pickle=False)
        model = cls(
            model_path=path,
            regularization_lambda=float(data["regularization_lambda"]),
            threshold_percentile=float(data["threshold_percentile"]),
        )
        model.classes = data["classes"]
        model.means = data["means"]
        model.threshold = float(data["threshold"])
        model.inv_vars = data["inv_vars"]
        return model
