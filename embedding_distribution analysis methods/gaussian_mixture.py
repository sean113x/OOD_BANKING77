"""Gaussian Mixture Model OOD model."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


DEFAULT_MODEL_PATH = Path("embedding_distribution analysis methods/outputs/gaussian_mixture_model.pkl")
DEFAULT_BERT_PATH = Path("BERT/minilm_lora")


def _embed_texts(texts: str | list[str], model_dir: Path = DEFAULT_BERT_PATH) -> np.ndarray:
    from BERT.embedding_utils import embed_texts

    return embed_texts(texts, model_dir=model_dir, batch_size=32, show_progress=False)


class GaussianMixtureOODModel:
    """Fit a global ID density model and score by negative log likelihood."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        n_components: int = 8,
        covariance_type: str = "diag",
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        random_state: int = 42,
        threshold_percentile: float = 95.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.threshold_percentile = threshold_percentile
        self.model: GaussianMixture | None = None
        self.threshold: float | None = None

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_label_texts: np.ndarray | None = None,
        save: bool = True,
    ) -> "GaussianMixtureOODModel":
        embeddings = np.asarray(train_embeddings, dtype=np.float32)
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
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
        scores = (-self.model.score_samples(embeddings)).astype(np.float32)
        return scores, np.full(len(scores), "gmm_density")

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
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "GaussianMixtureOODModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)
