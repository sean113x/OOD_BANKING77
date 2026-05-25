# OOD_BANKING77
cse363 machine learning final project

## Overall Pipeline

All methods follow the same high-level flow:

```text
Raw query
-> sentence embedding
-> OOD method
-> OOD score
-> threshold
-> ID or OOD
```

The shared embedding model is:

```text
MiniLM LoRA adapter: BERT/minilm_lora/
Base MiniLM model: sentence-transformers/all-MiniLM-L6-v2
```

Embedding extraction and saved embedding files are managed under `BERT/`.

## Method Overview

| Section | Category | Methods |
| --- | --- | --- |
| 1 | Embedding + class-wise boundary | Centroid distance, class-wise radius, class-wise Mahalanobis |
| 2 | Embedding + distribution / support estimation | LDA, Gaussian Naive Bayes, kNN, Linear One-Class SVM, RBF Kernel One-Class SVM |
| 3 | Embedding + anomaly detection | Isolation Forest, LOF |
| 4 | Embedding + classifier-output OOD scoring | Multinomial Logistic Regression, MLP Classifier; MSP, entropy, margin, maximum logit, energy |
| 5 | Advanced / SOTA-inspired | Temperature-scaled energy, temperature-scaled MSP/entropy, ensemble disagreement |

## Shared Convention

For every method, convert scores so that:

```text
higher score = more likely OOD
```
