"""Shared MiniLM LoRA embedding utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


DEFAULT_ADAPTER_DIR = Path("BERT/minilm_lora")
DEFAULT_BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_LENGTH = 64


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_base_model(model_dir: Path, base_model_name: str | None) -> str:
    if base_model_name:
        return base_model_name
    adapter_config_path = model_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open(encoding="utf-8") as f:
            adapter_config = json.load(f)
        return str(adapter_config["base_model_name_or_path"])

    return str(model_dir)


def load_embedding_model(
    model_dir: str | Path = DEFAULT_ADAPTER_DIR,
    base_model_name: str | None = DEFAULT_BASE_MODEL_NAME,
    device: str = "auto",
) -> tuple[AutoTokenizer, torch.nn.Module, str]:
    model_path = Path(model_dir)
    active_device = choose_device() if device == "auto" else device

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if (model_path / "adapter_config.json").exists():
        base_model = AutoModel.from_pretrained(_resolve_base_model(model_path, base_model_name))
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        model = AutoModel.from_pretrained(str(model_path))

    model.to(active_device)
    model.eval()
    return tokenizer, model, active_device


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


@torch.inference_mode()
def encode_texts(
    tokenizer,
    model: torch.nn.Module,
    texts: list[str],
    device: str,
    batch_size: int = 128,
    max_length: int = DEFAULT_MAX_LENGTH,
    show_progress: bool = False,
) -> np.ndarray:
    all_embeddings = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model_output = model(**encoded)
        embeddings = mean_pooling(model_output, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, dim=-1)
        all_embeddings.append(embeddings.cpu().numpy())

        if show_progress:
            done = min(start + len(batch), total)
            print(f"  {done}/{total}", end="\r")

    if show_progress:
        print()
    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def embed_texts(
    texts: str | list[str],
    model_dir: str | Path = DEFAULT_ADAPTER_DIR,
    base_model_name: str | None = DEFAULT_BASE_MODEL_NAME,
    device: str = "auto",
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    text_list = [texts] if isinstance(texts, str) else texts
    tokenizer, model, active_device = load_embedding_model(model_dir, base_model_name, device)
    return encode_texts(
        tokenizer,
        model,
        text_list,
        active_device,
        batch_size=batch_size,
        show_progress=show_progress,
    )
