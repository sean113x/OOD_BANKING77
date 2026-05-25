# Class-Wise Boundary Methods Implementation Note

This folder implements the method to create boundaries for known intent classes in the embedding space and determine that a new query is an OOD if it does not fall into any class boundary.

## Input

```text
dataset/preprocessed/embedded/OOD_train_embeddings.npz
dataset/preprocessed/embedded/OOD_test_embeddings.npz
```

필요한 array:

```text
embeddings
texts
labels
label_texts
is_ood
ood_types
sources
```

## Shared Flow

```text
1. Load saved embeddings
2. Fit class-wise model using OOD_train embeddings
3. Use OOD_test embeddings as the combined ID / near-OOD / far-OOD test set
4. Score OOD_test embeddings
5. Save model parameters, OOD scores, and summary metrics
6. Higher score = more likely OOD
```

`main.py` handles the quick class-wise test analysis for this folder. Broader
cross-method evaluation can still be handled outside this folder later.

## Methods To Implement

### 1. Class Centroid Distance

Fit:

```text
For each label_text:
  centroid = mean(class embeddings)
```

Score:

```text
score(x) = min distance from x to any class centroid
```

OOD rule:

```text
larger nearest-centroid distance -> more OOD
```

Threshold to expose:

```text
tau_centroid
ID if score <= tau_centroid
OOD if score > tau_centroid
```

Hyperparameters:

```text
distance_metric: cosine, default
threshold_percentile: candidate values 90, 95, 97.5, 99
```

### 2. Class-Wise Radius Threshold

Fit:

```text
For each label_text:
  centroid = mean(class embeddings)
  radius = chosen percentile of training distances to centroid
```

Score:

```text
nearest_class = class with smallest centroid distance
score(x) = distance(x, nearest_class_centroid) - radius(nearest_class)
```

OOD rule:

```text
score > 0 means outside the nearest class radius
larger score -> more OOD
```

Threshold to expose:

```text
radius_percentile_per_class
default: 95
```

The model can use `score > 0` as the direct OOD rule after each class radius is
chosen.

Hyperparameters:

```text
distance_metric: cosine, default
radius_percentile: candidate values 90, 95, 97.5, 99
minimum_radius: optional small floor value to avoid overly tight classes
```

### 3. Class-Wise Mahalanobis Distance

Fit:

```text
For each label_text:
  mean = mean(class embeddings)
  covariance = regularized covariance(class embeddings)
```

Score:

```text
score(x) = min Mahalanobis distance from x to any class distribution
```

OOD rule:

```text
larger minimum Mahalanobis distance -> more OOD
```

Threshold to expose:

```text
tau_mahalanobis
ID if score <= tau_mahalanobis
OOD if score > tau_mahalanobis
```

Hyperparameters:

```text
covariance_type: diagonal
regularization_lambda: candidate values 1e-4, 1e-3, 1e-2, 1e-1
threshold_percentile: candidate values 90, 95, 97.5, 99
```

Recommended first version:

```text
covariance_type: diagonal
regularization_lambda: 1e-2
```

## Suggested Code Structure

```text
embedding_class-wise methods/
  IMPLEMENTATION.md
  centroid_distance.py
  radius_threshold.py
  mahalanobis_distance.py
  class_wise_experiment.py
  outputs/

main.py
```

Suggested responsibilities:

```text
centroid_distance.py
  - define CentroidDistanceModel
  - fit class centroids
  - produce centroid OOD scores

radius_threshold.py
  - define RadiusThresholdModel
  - fit class centroids and class radii
  - produce radius-based OOD scores

mahalanobis_distance.py
  - define MahalanobisDistanceModel
  - fit class means and regularized covariance
  - produce Mahalanobis OOD scores

class_wise_experiment.py
  - train all three models on OOD_train_embeddings.npz
  - save model parameters immediately after fit
  - use OOD_test_embeddings.npz as the combined ID / near-OOD / far-OOD test set
  - write score CSV files and metrics.csv

main.py
  - root-level entry point
  - keep it minimal and call the experiment runner
```

Every model file should expose one main class with this interface:

```text
fit(train_embeddings, train_label_texts, save=True)
load(model_path)
score_embeddings(target_embeddings)
predict_embeddings(target_embeddings, threshold=None)
score_text(text_or_texts, model_dir="BERT/minilm_lora")
```

Class behavior:

```text
fit(..., save=True)
  -> trains model parameters
  -> immediately saves parameters under outputs/

score_embeddings(...)
  -> uses precomputed embeddings for batch evaluation

score_text(...)
  -> calls the BERT embedding model for one-off manual tests
```

## Output Files

Recommended output format:

```text
outputs/
  centroid_distance_model.npz
  radius_threshold_model.npz
  mahalanobis_distance_model.npz
  centroid_scores.csv
  radius_scores.csv
  mahalanobis_scores.csv
  metrics.csv
```

Score CSV columns:

```text
text,label,label_text,split,is_ood,ood_score,predicted_nearest_class
```

## Notes

- Use `label_text` as the class name.
- Keep score direction consistent: higher score means more likely OOD.
- Use the same BERT embeddings for all three methods.
- Start with cosine distance for centroid/radius methods.
- Use covariance regularization for Mahalanobis distance.
- Do not choose final thresholds from `OOD_test.csv`; expose scores and
  threshold candidates for the evaluation step.
