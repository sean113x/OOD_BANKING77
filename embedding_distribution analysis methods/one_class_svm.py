"""One-Class SVM OOD model."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import OneClassSVM


DEFAULT_MODEL_PATH = Path("embedding_distribution analysis methods/outputs/one_class_svm_model.pkl")
DEFAULT_BERT_PATH = Path("BERT/minilm_lora")


def _embed_texts(texts: str | list[str], model_dir: Path = DEFAULT_BERT_PATH) -> np.ndarray:
    from BERT.embedding_utils import embed_texts

    return embed_texts(texts, model_dir=model_dir, batch_size=32, show_progress=False)


class OneClassSVMModel:
    """Fit an ID support boundary and score by negative decision function."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str | float = "scale",
    ) -> None:
        self.model_path = Path(model_path)
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model: OneClassSVM | None = None
        self.threshold: float = 0.0

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_label_texts: np.ndarray | None = None,
        save: bool = True,
    ) -> "OneClassSVMModel":
        embeddings = np.asarray(train_embeddings, dtype=np.float32)
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.model.fit(embeddings)
        if save:
            self.save()
        return self

    def score_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        scores = (-self.model.decision_function(embeddings)).astype(np.float32)
        return scores, np.full(len(scores), "one_class_svm")

    def predict_embeddings(
        self, embeddings: np.ndarray, threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score_embeddings(embeddings)
        active_threshold = self.threshold if threshold is None else threshold
        return scores > active_threshold, scores, labels

    def score_text(
        self, texts: str | list[str], model_dir: str | Path = DEFAULT_BERT_PATH
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings = _embed_texts(texts, Path(model_dir))
        return self.score_embeddings(embeddings)

    def save(self, path: str | Path | None = None) -> None:
        if self.model is None:
            raise ValueError("Cannot save before fit().")

        output_path = Path(path) if path is not None else self.model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "OneClassSVMModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)
