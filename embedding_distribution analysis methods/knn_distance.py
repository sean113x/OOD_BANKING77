"""Global kNN distance OOD model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


DEFAULT_MODEL_PATH = Path("embedding_distribution analysis methods/outputs/knn_distance_model.npz")
DEFAULT_BERT_PATH = Path("BERT/minilm_lora")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _embed_texts(texts: str | list[str], model_dir: Path = DEFAULT_BERT_PATH) -> np.ndarray:
    from BERT.embedding_utils import embed_texts

    return embed_texts(texts, model_dir=model_dir, batch_size=32, show_progress=False)


class KNNDistanceModel:
    """Score by average distance to the k nearest known-intent embeddings."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        n_neighbors: int = 1,
        distance_metric: str = "cosine",
        threshold_percentile: float = 95.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.threshold_percentile = threshold_percentile
        self.train_embeddings: np.ndarray | None = None
        self.threshold: float | None = None
        self._neighbors: NearestNeighbors | None = None

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_label_texts: np.ndarray | None = None,
        save: bool = True,
    ) -> "KNNDistanceModel":
        embeddings = np.asarray(train_embeddings, dtype=np.float32)
        if self.distance_metric == "cosine":
            embeddings = _normalize_rows(embeddings)
        self.train_embeddings = embeddings
        self._fit_neighbors()

        train_scores = self._score_train_embeddings()
        self.threshold = float(np.percentile(train_scores, self.threshold_percentile))
        if save:
            self.save()
        return self

    def _fit_neighbors(self) -> None:
        if self.train_embeddings is None:
            raise ValueError("Missing train embeddings.")
        self._neighbors = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.train_embeddings)),
            metric=self.distance_metric,
        )
        self._neighbors.fit(self.train_embeddings)

    def _score_train_embeddings(self) -> np.ndarray:
        if self.train_embeddings is None:
            raise ValueError("Missing train embeddings.")
        neighbors = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(self.train_embeddings)),
            metric=self.distance_metric,
        )
        neighbors.fit(self.train_embeddings)
        distances, _ = neighbors.kneighbors(self.train_embeddings)
        if distances.shape[1] > 1:
            distances = distances[:, 1:]
        return distances.mean(axis=1).astype(np.float32)

    def score_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._neighbors is None:
            if self.train_embeddings is None:
                raise ValueError("Model is not fitted. Call fit() or load() first.")
            self._fit_neighbors()

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.distance_metric == "cosine":
            embeddings = _normalize_rows(embeddings)
        distances, _ = self._neighbors.kneighbors(embeddings)
        scores = distances.mean(axis=1).astype(np.float32)
        return scores, np.full(len(scores), "global_knn")

    def predict_embeddings(
        self, embeddings: np.ndarray, threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score_embeddings(embeddings)
        active_threshold = self.threshold if threshold is None else threshold
        if active_threshold is None:
            raise ValueError("No threshold is set.")
        return scores > active_threshold, scores, labels

    def score_text(
        self, texts: str | list[str], model_dir: str | Path = DEFAULT_BERT_PATH
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings = _embed_texts(texts, Path(model_dir))
        return self.score_embeddings(embeddings)

    def save(self, path: str | Path | None = None) -> None:
        if self.train_embeddings is None or self.threshold is None:
            raise ValueError("Cannot save before fit().")

        output_path = Path(path) if path is not None else self.model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            train_embeddings=self.train_embeddings,
            threshold=np.array(self.threshold, dtype=np.float32),
            n_neighbors=np.array(self.n_neighbors, dtype=np.int32),
            distance_metric=np.array(self.distance_metric),
            threshold_percentile=np.array(self.threshold_percentile, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "KNNDistanceModel":
        data = np.load(path, allow_pickle=False)
        model = cls(
            model_path=path,
            n_neighbors=int(data["n_neighbors"]),
            distance_metric=str(data["distance_metric"]),
            threshold_percentile=float(data["threshold_percentile"]),
        )
        model.train_embeddings = data["train_embeddings"]
        model.threshold = float(data["threshold"])
        model._fit_neighbors()
        return model
