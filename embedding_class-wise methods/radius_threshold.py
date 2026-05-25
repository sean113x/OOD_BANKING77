"""Class-wise radius threshold OOD model."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_MODEL_PATH = Path("embedding_class-wise methods/outputs/radius_threshold_model.npz")
DEFAULT_BERT_PATH = Path("BERT/all-MiniLM-L6-v2")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _choose_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _embed_texts(texts: str | list[str], model_dir: Path = DEFAULT_BERT_PATH) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    text_list = [texts] if isinstance(texts, str) else texts
    model = SentenceTransformer(str(model_dir), device=_choose_device())
    embeddings = model.encode(
        text_list,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


class RadiusThresholdModel:
    """Fit one centroid and one radius threshold per known class."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        distance_metric: str = "cosine",
        radius_percentile: float = 95.0,
        minimum_radius: float = 1e-6,
    ) -> None:
        self.model_path = Path(model_path)
        self.distance_metric = distance_metric
        self.radius_percentile = radius_percentile
        self.minimum_radius = minimum_radius
        self.classes: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.radii: np.ndarray | None = None
        self.threshold: float = 0.0

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_label_texts: np.ndarray,
        save: bool = True,
    ) -> "RadiusThresholdModel":
        if self.distance_metric != "cosine":
            raise ValueError("RadiusThresholdModel currently supports cosine distance.")

        embeddings = _normalize_rows(np.asarray(train_embeddings, dtype=np.float32))
        labels = np.asarray(train_label_texts)
        self.classes = np.array(sorted(set(labels.tolist())))
        self.centroids = np.vstack(
            [embeddings[labels == class_name].mean(axis=0) for class_name in self.classes]
        ).astype(np.float32)
        self.centroids = _normalize_rows(self.centroids)

        radii = []
        for idx, class_name in enumerate(self.classes):
            class_embeddings = embeddings[labels == class_name]
            distances = 1.0 - (class_embeddings @ self.centroids[idx])
            radius = float(np.percentile(distances, self.radius_percentile))
            radii.append(max(radius, self.minimum_radius))
        self.radii = np.array(radii, dtype=np.float32)

        if save:
            self.save()
        return self

    def score_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.classes is None or self.centroids is None or self.radii is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
        similarities = embeddings @ self.centroids.T
        nearest_idx = np.argmax(similarities, axis=1)
        nearest_distances = 1.0 - similarities[np.arange(len(embeddings)), nearest_idx]
        scores = nearest_distances - self.radii[nearest_idx]
        nearest_classes = self.classes[nearest_idx]
        return scores.astype(np.float32), nearest_classes

    def predict_embeddings(
        self, embeddings: np.ndarray, threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, nearest_classes = self.score_embeddings(embeddings)
        active_threshold = self.threshold if threshold is None else threshold
        is_ood = scores > active_threshold
        return is_ood, scores, nearest_classes

    def score_text(
        self, texts: str | list[str], model_dir: str | Path = DEFAULT_BERT_PATH
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings = _embed_texts(texts, Path(model_dir))
        return self.score_embeddings(embeddings)

    def save(self, path: str | Path | None = None) -> None:
        if self.classes is None or self.centroids is None or self.radii is None:
            raise ValueError("Cannot save before fit().")

        output_path = Path(path) if path is not None else self.model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            classes=self.classes,
            centroids=self.centroids,
            radii=self.radii,
            threshold=np.array(self.threshold, dtype=np.float32),
            distance_metric=np.array(self.distance_metric),
            radius_percentile=np.array(self.radius_percentile, dtype=np.float32),
            minimum_radius=np.array(self.minimum_radius, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "RadiusThresholdModel":
        data = np.load(path, allow_pickle=False)
        model = cls(
            model_path=path,
            distance_metric=str(data["distance_metric"]),
            radius_percentile=float(data["radius_percentile"]),
            minimum_radius=float(data["minimum_radius"]),
        )
        model.classes = data["classes"]
        model.centroids = data["centroids"]
        model.radii = data["radii"]
        model.threshold = float(data["threshold"])
        return model
