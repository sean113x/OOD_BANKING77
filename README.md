# OOD_BANKING77

OOD_BANKING77 is an experimental framework for out-of-distribution detection on
the BANKING77 intent classification dataset. It uses a shared MiniLM LoRA
sentence encoder and compares embedding-distance, density/support, anomaly, and
classifier-output OOD scoring methods under the same ID/OOD split.

The project consists of the following main components:

1. BANKING77 known-intent training and ID/OOD validation/test splits
2. MiniLM LoRA embedding extraction for all text splits
3. OOD method implementations across four model families
4. Experiment runners that export one CSV per experiment
5. An interactive `main.py` entry point for testing user queries as ID or OOD

OOD_BANKING77 workflow:

```text
Raw query
-> MiniLM LoRA sentence embedding
-> selected OOD model
-> OOD score
-> validation-calibrated threshold
-> ID or OOD prediction
```

For every method, scores are oriented as:

```text
higher score = more likely OOD
```

## Prerequisites

* Python 3.10 or newer
* `pip` or `uv`
* Dependencies listed in `requirements.txt`
* Preprocessed dataset files under `dataset/preprocessed/`
* Embedding model files under `BERT/minilm_lora/`

The shared embedding model is:

```text
LoRA adapter: BERT/minilm_lora/
Base model:   sentence-transformers/all-MiniLM-L6-v2
```

## Orientation

Important files and directories:

```text
main.py                                      interactive tester and experiment entry point
BERT/embed.py                               embedding generation
BERT/embedding_utils.py                     MiniLM LoRA loading and text encoding
dataset/preprocessed/                       final text splits
dataset/preprocessed/embedded/              saved embedding splits
embedding_class-wise methods/               class-wise OOD models
embedding_distribution analysis methods/    distance/support/density OOD models
embeddinig_anomaly/                         anomaly detection baselines
embedding_scoring/                          classifier-output OOD scores
experiments/common.py                       shared experiment/model registry utilities
experiments/experiment1_overall_comparison.py
experiments/experiment2_sensitivity.py
experiments/experiment3_near_ood_difficulty.py
experiments/experiment4_accuracy_vs_ood.py
experiments/results/                        experiment CSV outputs
```

Experiment 5 is intentionally not implemented yet.

## Installation

### Using `venv` and `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Using `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Check that the entry point imports correctly:

```bash
python main.py --list-models
```

## Dataset Splits

The main preprocessed files are:

```text
OOD_train.csv
classification_test.csv
OOD_validation.csv
OOD_test_eval.csv
OOD_test.csv
```

Split summary:

| File | Purpose | Rows |
| --- | --- | ---: |
| `OOD_train.csv` | known BANKING77 intents used for fitting OOD models | 7,932 |
| `classification_test.csv` | known-intent closed-set classifier test split | 2,480 |
| `OOD_validation.csv` | validation split for hyperparameters and thresholds | 2,209 |
| `OOD_test_eval.csv` | final OOD evaluation split after validation extraction | 8,823 |
| `OOD_test.csv` | original combined ID/near/far OOD file | 11,032 |

`OOD_validation.csv` and `OOD_test_eval.csv` are derived from `OOD_test.csv`
with `validation_ratio=0.2` and `seed=42`.

Validation counts:

```text
id:   496
near: 533
far:  1180
```

Final evaluation counts:

```text
id:   1984
near: 2138
far:  4701
```

Regenerate the validation/evaluation text split:

```bash
python dataset/preprocessed/create_ood_validation_split.py
```

## Embeddings

Saved embeddings are stored in:

```text
dataset/preprocessed/embedded/
```

Generate or refresh embeddings:

```bash
python BERT/embed.py
```

The embedding script exports `.npz` files for:

```text
OOD_train
classification_test
OOD_validation
OOD_test_eval
OOD_test
```

If `OOD_validation_embeddings.npz` or `OOD_test_eval_embeddings.npz` is missing,
the experiment utilities can also derive them from `OOD_test_embeddings.npz`
using the corresponding CSV rows.

## Usage

### Interactive OOD Tester

Run:

```bash
python main.py
```

The program first asks for a model family:

```text
1. Class-wise boundary
2. Distance/support/density
3. Anomaly detection
4. Classifier-output scoring
```

After selecting a family, choose a model inside that family. The selected model
is loaded from disk when a saved model exists; otherwise it is trained on
`OOD_train`. The threshold is calibrated on `OOD_validation`, validation metrics
are printed, the embedding encoder is loaded once, and the program enters a
query loop. Query predictions reuse the already-loaded OOD model and encoder.

Example interaction:

```text
query> How do I activate my card?
ID | score=..., threshold=..., nearest/predicted=activate_my_card

query> Who invented the telephone?
OOD | score=..., threshold=..., nearest/predicted=...
```

Exit the loop with:

```text
q
quit
exit
```

List available models without opening the interactive loop:

```bash
python main.py --list-models
```

Force retraining instead of loading saved models:

```bash
python main.py --force-train
```

### Experiment Entry Point

Experiments are also launched through `main.py`:

```bash
python main.py --experiment
python main.py --experiment 1
python main.py --experiment 2
python main.py --experiment 3
python main.py --experiment 4
python main.py --experiment all
```

Running `python main.py --experiment` without a value opens an experiment
selection menu.

Use `--force-train` for experiments that support retraining saved default
models:

```bash
python main.py --experiment 1 --force-train
```

## Experiments

Each experiment writes one CSV file.
Each run also prints a terminal metric table so key values are visible without
opening the CSV.

| Experiment | Goal | Output |
| --- | --- | --- |
| 1 | Overall method comparison across implemented OOD families | `experiments/results/experiment1_overall_comparison.csv` |
| 2 | Hyperparameter and threshold sensitivity | `experiments/results/experiment2_hyperparameter_sensitivity.csv` |
| 3 | Near-OOD difficulty by centroid similarity to known intents | `experiments/results/experiment3_near_ood_difficulty.csv` |
| 4 | Closed-set classification accuracy vs OOD reliability | `experiments/results/experiment4_accuracy_vs_ood.csv` |

Experiment 2 can take longer than the others because it fits multiple
hyperparameter configurations.

## Models

### Class-wise Boundary

| Model | Main parameters | OOD score |
| --- | --- | --- |
| Centroid distance | cosine distance, threshold percentile 95 | distance to nearest class centroid |
| Class-wise radius | cosine distance, radius percentile 90 | distance beyond nearest class radius |
| Class-wise Mahalanobis | diagonal covariance, regularization `1e-2` | minimum class-wise Mahalanobis distance |

### Distance / Support / Density

| Model | Main parameters | OOD score |
| --- | --- | --- |
| kNN distance | `k=1`, cosine distance | mean nearest-neighbor distance |
| Global Mahalanobis | Ledoit-Wolf shrinkage covariance | global Mahalanobis distance |
| Gaussian Mixture Model | 62 components, full covariance | negative log likelihood |
| One-Class SVM | RBF kernel, `nu=0.3`, `gamma=30` | negative decision function |
| PCA reconstruction | 256 components | reconstruction mean squared error |

### Anomaly Detection

| Model | Main parameters | OOD score |
| --- | --- | --- |
| Isolation Forest | 200 estimators, auto contamination | negative decision function |
| Local Outlier Factor | 20 neighbors, novelty mode | negative score samples |

### Classifier-output Scoring

Supported classifiers:

```text
Logistic Regression
MLP
Gaussian Naive Bayes
Linear Discriminant Analysis
Quadratic Discriminant Analysis
```

Supported OOD scores:

```text
MSP
Entropy
Margin
MaxLogit
Energy
```

Every classifier can be paired with every supported score. `MaxLogit` and
`Energy` use raw `decision_function` scores when the classifier exposes them;
otherwise they fall back to log probabilities from `predict_proba`. The
probability fallback keeps the experiment grid complete, although Energy at
temperature 1 can become a weak proxy for probability-only classifiers because
normalized probabilities sum to 1.

## Metrics

The experiment CSV files include:

```text
AUROC
AUPR
FPR95
OOD precision
OOD recall
OOD F1
Accuracy
Balanced accuracy
Confusion matrix counts
Fit time
Score time
Threshold source
```

Thresholds are calibrated using validation scores, most commonly with
`validation_best_f1`.

## Inputs

Text CSV columns:

```text
text,label,label_text
```

OOD CSV columns:

```text
text,label,label_text,is_ood,ood_type,source,source_label,source_label_text
```

Embedding `.npz` files include arrays such as:

```text
embeddings
texts
labels
label_texts
is_ood
ood_types
sources
```

## Outputs

Model artifacts are saved under their method directories:

```text
embedding_class-wise methods/outputs/
embedding_distribution analysis methods/outputs/
embeddinig_anomaly/models/
embedding_scoring/classifier/
```

Experiment outputs are saved under:

```text
experiments/results/
```

## Common Issues

### Missing embedding file

If an experiment reports a missing embedding split, run:

```bash
python BERT/embed.py
```

If only validation/eval embeddings are missing, the common loader can derive
them from `OOD_test_embeddings.npz` as long as `OOD_validation.csv` and
`OOD_test_eval.csv` exist.

### Saved sklearn model version warning

Some saved `.pkl` files may have been created with an older scikit-learn
version. For strict reproducibility, retrain under the current environment:

```bash
python main.py --force-train
```

or for supported experiments:

```bash
python main.py --experiment 1 --force-train
```

### Experiment 2 is slow

Experiment 2 fits many configurations and can take noticeably longer than
Experiments 1, 3, and 4.

## Reference

Dataset:

```text
BANKING77 intent classification dataset
TREC question classification dataset for far-OOD examples
```

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2 with LoRA adapter
```
