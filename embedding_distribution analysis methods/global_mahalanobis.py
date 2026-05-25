"""Global Mahalanobis OOD model with shrinkage covariance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf


DEFAULT_MODEL_PATH = Path("embedding_distribution analysis methods/outputs/global_mahalanobis_model.npz")
DEFAULT_BERT_PATH = Path("BERT/minilm_lora")


def _embed_texts(texts: str | list[str], model_dir: Path = DEFAULT_BERT_PATH) -> np.ndarray:
    from BERT.embedding_utils import embed_texts

    return embed_texts(texts, model_dir=model_dir, batch_size=32, show_progress=False)


class GlobalMahalanobisModel:
    """Fit one global ID Gaussian and score by Mahalanobis distance."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        threshold_percentile: float = 95.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold_percentile = threshold_percentile
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None
        self.shrinkage: float | None = None
        self.threshold: float | None = None

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_label_texts: np.ndarray | None = None,
        save: bool = True,
    ) -> "GlobalMahalanobisModel":
        embeddings = np.asarray(train_embeddings, dtype=np.float32)
        covariance = LedoitWolf().fit(embeddings)
        self.mean = covariance.location_.astype(np.float32)
        self.precision = covariance.precision_.astype(np.float32)
        self.shrinkage = float(covariance.shrinkage_)

        train_scores, _ = self.score_embeddings(embeddings)
        self.threshold = float(np.percentile(train_scores, self.threshold_percentile))
        if save:
            self.save()
        return self

    def score_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.precision is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        diff = embeddings - self.mean
        squared = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        scores = np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)
        return scores, np.full(len(scores), "global_gaussian")

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
        if self.mean is None or self.precision is None or self.threshold is None:
            raise ValueError("Cannot save before fit().")

        output_path = Path(path) if path is not None else self.model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            mean=self.mean,
            precision=self.precision,
            shrinkage=np.array(self.shrinkage, dtype=np.float32),
            threshold=np.array(self.threshold, dtype=np.float32),
            threshold_percentile=np.array(self.threshold_percentile, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "GlobalMahalanobisModel":
        data = np.load(path, allow_pickle=False)
        model = cls(
            model_path=path,
            threshold_percentile=float(data["threshold_percentile"]),
        )
        model.mean = data["mean"]
        model.precision = data["precision"]
        model.shrinkage = float(data["shrinkage"])
        model.threshold = float(data["threshold"])
        return model
