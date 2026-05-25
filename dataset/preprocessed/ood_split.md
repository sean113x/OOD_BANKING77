# Preprocessed OOD Splits

This folder contains the final preprocessed files used by the project.

## Files

```text
OOD_train.csv
classification_test.csv
OOD_test.csv
embedded/
```

## Training Set

`OOD_train.csv` contains only BANKING77 known-intent examples. The 15 held-out
BANKING77 intents are excluded from training.

## Classification Test Set

`classification_test.csv` contains BANKING77 known-intent test examples. It is
kept as a separate file for closed-set intent classification evaluation.

## OOD Test Set

`OOD_test.csv` is the combined OOD evaluation file. It contains:

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

Current counts:

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

`OOD_test.csv` uses:

```text
text,label,label_text,is_ood,ood_type,source,source_label,source_label_text
```
