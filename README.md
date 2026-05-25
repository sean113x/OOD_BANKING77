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
| 2 | Embedding + distance / support / density estimation | kNN distance, global Mahalanobis with shrinkage covariance, Gaussian Mixture Model, One-Class SVM, PCA reconstruction error |
| 3 | Embedding + anomaly detection | Isolation Forest, LOF |
| 4 | Embedding + classifier-output OOD scoring | Multinomial Logistic Regression, MLP Classifier; MSP, entropy, margin, maximum logit, energy |
| 5 | Advanced / SOTA-inspired | Temperature-scaled energy, temperature-scaled MSP/entropy, ensemble disagreement |

## Shared Convention

For every method, convert scores so that:

```text
higher score = more likely OOD
```

## Current Parameter Settings

These are the current settings used by `main.py` for evaluation.

### Class-wise Boundary Methods

| Method | Main parameters | OOD score |
| --- | --- | --- |
| Centroid distance | `distance_metric="cosine"`, `threshold_percentile=95.0` | Distance to the nearest class centroid |
| Class-wise radius | `distance_metric="cosine"`, `radius_percentile=90.0`, `minimum_radius=1e-6` | Distance beyond the nearest class radius |
| Class-wise Mahalanobis | Diagonal covariance, `regularization_lambda=1e-2`, `threshold_percentile=95.0` | Minimum class-wise Mahalanobis distance |

### Distance / Support / Density Estimation Methods

| Method | Main parameters | OOD score |
| --- | --- | --- |
| kNN distance | `n_neighbors=1`, `distance_metric="cosine"`, `threshold_percentile=95.0` | Mean distance to nearest known embedding |
| Global Mahalanobis | Ledoit-Wolf shrinkage covariance, `threshold_percentile=95.0` | Global Mahalanobis distance |
| Gaussian Mixture Model | `n_components=62`, `covariance_type="full"`, `reg_covar=1e-5`, `max_iter=200`, `random_state=42`, `threshold_percentile=95.0` | Negative log likelihood |
| One-Class SVM | `kernel="rbf"`, `nu=0.3`, `gamma=30.0` | Negative decision function |
| PCA reconstruction | `n_components=256`, `random_state=42`, `threshold_percentile=95.0` | Mean squared reconstruction error |

### Evaluation Threshold

For reporting in `main.py`, the decision threshold is selected per evaluation split by maximizing binary ID/OOD accuracy. AUROC and AUPR are threshold-free and are used as the main comparison metrics.
