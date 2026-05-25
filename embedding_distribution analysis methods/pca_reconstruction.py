"""PCA reconstruction error OOD model."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


DEFAULT_MODEL_PATH = Path("embedding_distribution analysis methods/outputs/pca_reconstruction_model.pkl")
DEFAULT_BERT_PATH = Path("BERT/minilm_lora")


def _embed_texts(texts: str | list[str], model_dir: Path = DEFAULT_BERT_PATH) -> np.ndarray:
    from BERT.embedding_utils import embed_texts

    return embed_texts(texts, model_dir=model_dir, batch_size=32, show_progress=False)


class PCAReconstructionModel:
    """Fit a global PCA subspace and score by reconstruction error."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        n_components: int | float = 256,
        threshold_percentile: float = 95.0,
        random_state: int = 42,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.model: PCA | None = None
        self.threshold: float | None = None

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_label_texts: np.ndarray | None = None,
        save: bool = True,
    ) -> "PCAReconstructionModel":
        embeddings = np.asarray(train_embeddings, dtype=np.float32)
        self.model = PCA(n_components=self.n_components, random_state=self.random_state)
        self.model.fit(embeddings)

        train_scores, _ = self.score_embeddings(embeddings)
        self.threshold = float(np.percentile(train_scores, self.threshold_percentile))
        if save:
            self.save()
        return self

    def score_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        projected = self.model.transform(embeddings)
        reconstructed = self.model.inverse_transform(projected)
        scores = np.mean((embeddings - reconstructed) ** 2, axis=1).astype(np.float32)
        return scores, np.full(len(scores), "pca_reconstruction")

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
        if self.model is None or self.threshold is None:
            raise ValueError("Cannot save before fit().")

        output_path = Path(path) if path is not None else self.model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "PCAReconstructionModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)
