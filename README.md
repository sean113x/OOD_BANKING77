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
| 4 | Embedding + classifier-output OOD scoring | Multinomial Logistic Regression, MLP Classifier, Gaussian Naive Bayes, Linear Discriminant Analysis, Quadratic Discriminant Analysis; MSP, entropy, margin, maximum logit, energy |
| 5 | Advanced / SOTA-inspired | Temperature-scaled energy, temperature-scaled MSP/entropy, ensemble disagreement |

## Energy Score Caveat

Energy score should be reported only for classifiers that expose raw,
unnormalized logits or decision scores. Logistic Regression exposes multiclass
decision scores through `decision_function`, so LR Energy is valid in this
project.

`sklearn.neural_network.MLPClassifier` and `sklearn.naive_bayes.GaussianNB` do
not expose raw logits; they expose normalized class probabilities.
Substituting `log(probability)` for logits makes the energy score degenerate:

```text
E(x) = -log(sum(exp(log p_i))) = -log(sum(p_i)) = -log(1) = 0
```

At temperature `T=1`, every sample receives an almost constant energy value, so
thresholding cannot meaningfully separate ID from OOD. For this reason, we
exclude MLP/GNB Energy from the main comparison and use them with MSP, entropy,
or margin instead.

## Shared Convention

For every method, convert scores so that:

```text
higher score = more likely OOD
```
