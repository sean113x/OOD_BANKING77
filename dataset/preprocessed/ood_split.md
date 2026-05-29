# Preprocessed OOD Splits

This folder contains the final preprocessed files used by the project.

## Files

```text
OOD_train.csv
classification_test.csv
OOD_validation.csv
OOD_test_eval.csv
OOD_test.csv
embedded/
```

## Training Set

`OOD_train.csv` contains only BANKING77 known-intent examples. The 15 held-out
BANKING77 intents are excluded from training.

## Classification Test Set

`classification_test.csv` contains BANKING77 known-intent test examples. It is
kept as a separate file for closed-set intent classification evaluation.

## Validation Set

`OOD_validation.csv` is sampled from `OOD_test.csv` for hyperparameter and
threshold selection. The split is reproducible with:

```text
uv run python dataset/preprocessed/create_ood_validation_split.py
```

Validation rows are sampled with `validation_ratio=0.2` and `seed=42`.
Stratification uses:

```text
ID examples:       ood_type + label_text
Near-OOD examples: ood_type + label_text
Far-OOD examples:  ood_type + source_label_text
```

The far-OOD split uses `source_label_text` because all TREC rows share the
generic `label_text` value `far_ood_trec`.

Current validation counts:

```text
id:   496
near: 533
far:  1180
total: 2209
```

## Final Evaluation Set

`OOD_test_eval.csv` contains the rows left after extracting
`OOD_validation.csv`. Use this file for final OOD evaluation after model
selection.

Current final-evaluation counts:

```text
id:   1984
near: 2138
far:  4701
total: 8823
```

## OOD Test Set

`OOD_test.csv` is the original combined OOD file and remains unchanged. It
contains:

```text
ID examples:       BANKING77 classification_test examples
Near-OOD examples: held-out BANKING77 intent examples
Far-OOD examples:  TREC question classification examples
```

Rows are marked with:

```text
is_ood = 0 or 1
ood_type = id, near, or far
source = banking77_classification_test, banking77_near_ood, trec_train, or trec_test
```

Original counts:

```text
id:   2480
near: 2671
far:  5881
total: 11032
```

The downloaded TREC files contain 5952 rows in total. After exact duplicate
removal, 5881 TREC rows are used as far-OOD examples.

## Columns

`OOD_train.csv` and `classification_test.csv` use:

```text
text,label,label_text
```

`OOD_validation.csv`, `OOD_test_eval.csv`, and `OOD_test.csv` use:

```text
text,label,label_text,is_ood,ood_type,source,source_label,source_label_text
```
