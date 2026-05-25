# BANKING77 Near-OOD Split

This folder contains the raw BANKING77 files and the near-OOD split files used
for this project.

## Raw Files

The original downloaded files are:

- `train.csv`
- `dev.csv`
- `test.csv`

Each raw CSV has the same schema:

- `text`: user query
- `label`: original BANKING77 intent id
- `label_text`: original BANKING77 intent name

Important note: in this downloaded copy, `dev.csv` and `test.csv` are identical.
Although `dev` usually means validation, using it as a separate validation set
would duplicate the test set. Therefore, this split removes duplicate overlap
when combining the files.

## Near-OOD Setup

We hold out 15 of the 77 BANKING77 intents as near-OOD intents. The classifier
is trained only on the remaining 62 known intents. The held-out intents simulate
unsupported banking requests that should be rejected instead of forced into a
known class.

The held-out near-OOD intents are:

```text
card_delivery_estimate
card_payment_not_recognised
cash_withdrawal_not_recognised
declined_cash_withdrawal
failed_transfer
transfer_not_received_by_recipient
top_up_failed
top_up_reverted
verify_top_up
pin_blocked
beneficiary_not_allowed
exchange_rate
Refund_not_showing_up
card_acceptance
supported_cards_and_currencies
```

These labels were chosen because they are semantically close to labels that
remain in the known-intent set, making this a near-OOD setting rather than an
easy far-OOD setting.

## Generated Split Files

Only three processed CSV files are used:

- `OOD_train.csv`: all known-intent examples from the original `train.csv` and
  `dev.csv`, excluding the 15 held-out labels. Any row that overlaps with
  `classification_test.csv` is removed. Since `dev.csv` is identical to
  `test.csv`, this file effectively contains the known-intent portion of
  `train.csv`.
- `OOD_test.csv`: all held-out-intent examples from the original `train.csv`,
  `dev.csv`, and `test.csv`, with duplicate rows removed. This is the near-OOD
  evaluation set.
- `classification_test.csv`: all known-intent examples from the original
  `test.csv`, excluding the 15 held-out labels. This is used for final
  known-intent classification evaluation.

All three generated files keep the original BANKING77 schema:

- `text`
- `label`
- `label_text`

No new label ids are introduced in these files.

## Expected Counts

With the 15 held-out intents above, the generated files have the following
counts:

```text
OOD_train.csv: 7932
OOD_test.csv: 2671
classification_test.csv: 2480
```

There is no row overlap between `OOD_train.csv` and `classification_test.csv`
under the exact key `(text, label, label_text)`.

## Evaluation Usage

Use `OOD_train.csv` to train the known-intent classifier or embedding-based OOD
model on the 62 supported intents.

Use `classification_test.csv` to report known-intent classification accuracy
and macro F1.

Use `OOD_test.csv` as the near-OOD set for OOD detection metrics such as AUROC,
AUPR, FPR95, OOD precision, OOD recall, and OOD F1.
