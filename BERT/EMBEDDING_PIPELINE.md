# Embedding Pipeline

`BERT/` is responsible for converting BANKING77 text into sentence embeddings.
Downstream method folders should load these saved embeddings instead of running
the BERT model again.

## Model

```text
MiniLM LoRA adapter: BERT/minilm_lora/
Base MiniLM model: sentence-transformers/all-MiniLM-L6-v2
embedding dim: 384
```

## Input CSV Files

```text
dataset/preprocessed/OOD_train.csv
dataset/preprocessed/classification_test.csv
dataset/preprocessed/OOD_test.csv
```

Each file uses:

```text
text,label,label_text
```

## Output Embedding Files

Run:

```text
python BERT/embed.py
```

Output location:

```text
dataset/preprocessed/embedded/
```

Files:

```text
dataset/preprocessed/embedded/OOD_train_embeddings.npz
dataset/preprocessed/embedded/classification_test_embeddings.npz
dataset/preprocessed/embedded/OOD_test_embeddings.npz
```

Each `.npz` should contain:

```text
embeddings: float32 array, shape [num_samples, 384]
texts: original text array
labels: original BANKING77 label ids
label_texts: original BANKING77 label names
is_ood: 0 for ID, 1 for OOD
ood_types: id, near, or far
sources: source split / dataset name
model_dir: adapter path used for embedding
base_model_name: Hugging Face base model name
```

## Embedding Settings

Use the same settings for every split:

```text
normalize_embeddings: true
batch_size: 128, adjust if GPU memory is limited
device priority: cuda -> mps -> cpu
```

Normalization is recommended because most embedding-space methods will use
cosine distance or dot-product similarity.

## Responsibility Boundary

`BERT/` handles:

```text
CSV text -> sentence embeddings -> saved embedding files
```

Method folders handle:

```text
saved embeddings -> method-specific model -> OOD score
```

Evaluation folders handle:

```text
OOD scores -> threshold selection -> metrics
```
