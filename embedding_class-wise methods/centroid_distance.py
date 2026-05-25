"""Class centroid distance OOD model."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_MODEL_PATH = Path("embedding_class-wise methods/outputs/centroid_distance_model.npz")
DEFAULT_BERT_PATH = Path("BERT/minilm_lora")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _embed_texts(texts: str | list[str], model_dir: Path = DEFAULT_BERT_PATH) -> np.ndarray:
    from BERT.embedding_utils import embed_texts

    return embed_texts(texts, model_dir=model_dir, batch_size=32, show_progress=False)


class CentroidDistanceModel:
    """Fit one centroid per known class and score by nearest-centroid distance."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        distance_metric: str = "cosine",
        threshold_percentile: float = 95.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.distance_metric = distance_metric
        self.threshold_percentile = threshold_percentile
        self.classes: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.threshold: float | None = None

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_label_texts: np.ndarray,
        save: bool = True,
    ) -> "CentroidDistanceModel":
        if self.distance_metric != "cosine":
            raise ValueError("CentroidDistanceModel currently supports cosine distance.")

        embeddings = _normalize_rows(np.asarray(train_embeddings, dtype=np.float32))
        labels = np.asarray(train_label_texts)
        self.classes = np.array(sorted(set(labels.tolist())))
        self.centroids = np.vstack(
            [embeddings[labels == class_name].mean(axis=0) for class_name in self.classes]
        ).astype(np.float32)
        self.centroids = _normalize_rows(self.centroids)

        train_scores, _ = self.score_embeddings(embeddings)
        self.threshold = float(np.percentile(train_scores, self.threshold_percentile))
        if save:
            self.save()
        return self

    def score_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.classes is None or self.centroids is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
        similarities = embeddings @ self.centroids.T
        nearest_idx = np.argmax(similarities, axis=1)
        scores = 1.0 - similarities[np.arange(len(embeddings)), nearest_idx]
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
        if self.classes is None or self.centroids is None or self.threshold is None:
            raise ValueError("Cannot save before fit().")

        output_path = Path(path) if path is not None else self.model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            classes=self.classes,
            centroids=self.centroids,
            threshold=np.array(self.threshold, dtype=np.float32),
            distance_metric=np.array(self.distance_metric),
            threshold_percentile=np.array(self.threshold_percentile, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "CentroidDistanceModel":
        data = np.load(path, allow_pickle=False)
        model = cls(
            model_path=path,
            distance_metric=str(data["distance_metric"]),
            threshold_percentile=float(data["threshold_percentile"]),
        )
        model.classes = data["classes"]
        model.centroids = data["centroids"]
        model.threshold = float(data["threshold"])
        return model
