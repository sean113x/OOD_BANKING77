# Experiment 2: Hyperparameter and Threshold Sensitivity

## LR Hyperparameter Sweep (Overall)

| model | score | hyperparameters | auroc | aupr | fpr95 | f1 | precision | recall | tpr | fpr | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | Energy | C=0.01 | 0.7766 | 0.9310 | 0.7976 | 0.8773 | 0.7834 | 0.9967 | 0.9967 | 0.9504 | 0.7838 |
| LR | Energy | C=0.1 | 0.8785 | 0.9635 | 0.7056 | 0.8842 | 0.8698 | 0.8991 | 0.8991 | 0.4641 | 0.8174 |
| LR | Energy | C=1.0 | 0.9249 | 0.9786 | 0.4976 | 0.9161 | 0.9090 | 0.9234 | 0.9234 | 0.3190 | 0.8689 |
| LR | Energy | C=10.0 | 0.9340 | 0.9808 | 0.3794 | 0.9227 | 0.8896 | 0.9584 | 0.9584 | 0.4101 | 0.8755 |
| LR | Energy | C=100.0 | 0.9209 | 0.9771 | 0.4613 | 0.9155 | 0.8996 | 0.9321 | 0.9321 | 0.3589 | 0.8667 |

## MLP Hyperparameter Sweep (Overall)

| model | score | hyperparameters | auroc | aupr | fpr95 | f1 | precision | recall | tpr | fpr | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | Entropy | hidden=256, lr=0.001, alpha=0.001 | 0.8781 | 0.9592 | 0.4996 | 0.9075 | 0.8748 | 0.9428 | 0.9428 | 0.4653 | 0.8511 |
| MLP | Entropy | hidden=512, lr=0.001, alpha=0.001 | 0.8900 | 0.9628 | 0.4415 | 0.9153 | 0.8913 | 0.9407 | 0.9407 | 0.3956 | 0.8651 |
| MLP | Entropy | hidden=512x256, lr=0.0005, alpha=0.001 | 0.8594 | 0.9482 | 0.5645 | 0.9005 | 0.8672 | 0.9364 | 0.9364 | 0.4944 | 0.8396 |
| MLP | Entropy | hidden=512x256, lr=0.001, alpha=0.001 | 0.8712 | 0.9552 | 0.5423 | 0.9036 | 0.8754 | 0.9338 | 0.9338 | 0.4585 | 0.8456 |
| MLP | Entropy | hidden=512x256, lr=0.003, alpha=0.001 | 0.8771 | 0.9571 | 0.4738 | 0.9110 | 0.8714 | 0.9544 | 0.9544 | 0.4855 | 0.8555 |

## LOF Hyperparameter Sweep (Overall)

| model | score | hyperparameters | auroc | aupr | fpr95 | f1 | precision | recall | tpr | fpr | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOF | negative_score_samples | k=5, metric=euclidean | 0.9157 | 0.9740 | 0.5081 | 0.9075 | 0.8907 | 0.9250 | 0.9250 | 0.3915 | 0.8539 |
| LOF | negative_score_samples | k=10, metric=euclidean | 0.9259 | 0.9769 | 0.4399 | 0.9161 | 0.9004 | 0.9324 | 0.9324 | 0.3556 | 0.8677 |
| LOF | negative_score_samples | k=20, metric=euclidean | 0.9361 | 0.9795 | 0.3427 | 0.9278 | 0.8974 | 0.9604 | 0.9604 | 0.3786 | 0.8842 |
| LOF | negative_score_samples | k=50, metric=euclidean | 0.9313 | 0.9761 | 0.3000 | 0.9345 | 0.9028 | 0.9685 | 0.9685 | 0.3597 | 0.8948 |
| LOF | negative_score_samples | k=100, metric=euclidean | 0.9219 | 0.9717 | 0.3496 | 0.9273 | 0.9075 | 0.9480 | 0.9480 | 0.3331 | 0.8848 |
| LOF | negative_score_samples | k=20, metric=cosine | 0.9350 | 0.9791 | 0.3419 | 0.9278 | 0.9014 | 0.9558 | 0.9558 | 0.3605 | 0.8847 |

## IsolationForest Hyperparameter Sweep (Overall)

| model | score | hyperparameters | auroc | aupr | fpr95 | f1 | precision | recall | tpr | fpr | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IsolationForest | negative_decision_function | n_estimators=100, max_samples=auto, contamination=auto | 0.7258 | 0.8974 | 0.8052 | 0.8760 | 0.7885 | 0.9855 | 0.9855 | 0.9117 | 0.7838 |
| IsolationForest | negative_decision_function | n_estimators=200, max_samples=auto, contamination=auto | 0.7990 | 0.9271 | 0.6927 | 0.8846 | 0.8151 | 0.9671 | 0.9671 | 0.7565 | 0.8045 |
| IsolationForest | negative_decision_function | n_estimators=500, max_samples=auto, contamination=auto | 0.8354 | 0.9409 | 0.6984 | 0.8850 | 0.8116 | 0.9729 | 0.9729 | 0.7786 | 0.8039 |
| IsolationForest | negative_decision_function | n_estimators=200, max_samples=0.5, contamination=auto | 0.9166 | 0.9748 | 0.4738 | 0.9109 | 0.8805 | 0.9435 | 0.9435 | 0.4415 | 0.8570 |
| IsolationForest | negative_decision_function | n_estimators=200, max_samples=0.75, contamination=auto | 0.9223 | 0.9769 | 0.4452 | 0.9142 | 0.8862 | 0.9441 | 0.9441 | 0.4181 | 0.8627 |
| IsolationForest | negative_decision_function | n_estimators=200, max_samples=1.0, contamination=auto | 0.9167 | 0.9755 | 0.4992 | 0.9084 | 0.8801 | 0.9386 | 0.9386 | 0.4411 | 0.8532 |
| IsolationForest | negative_decision_function | n_estimators=200, max_samples=auto, contamination=0.05 | 0.7990 | 0.9271 | 0.6927 | 0.8846 | 0.8151 | 0.9671 | 0.9671 | 0.7565 | 0.8045 |
| IsolationForest | negative_decision_function | n_estimators=200, max_samples=auto, contamination=0.1 | 0.7990 | 0.9271 | 0.6927 | 0.8846 | 0.8151 | 0.9671 | 0.9671 | 0.7565 | 0.8045 |

## Threshold Sweep (Overall)

| model | id_percentile | f1 | precision | recall | tpr | fpr | accuracy | balanced_accuracy | youden_j | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | 50 | 0.9186 | 0.8703 | 0.9726 | 0.9726 | 0.5000 | 0.8664 | 0.7363 | 0.4726 | 1240 | 1240 | 234 | 8318 |
| LR | 70 | 0.9193 | 0.9140 | 0.9246 | 0.9246 | 0.3000 | 0.8741 | 0.8123 | 0.6246 | 1736 | 744 | 645 | 7907 |
| LR | 90 | 0.8765 | 0.9651 | 0.8027 | 0.8027 | 0.1000 | 0.8246 | 0.8514 | 0.7027 | 2232 | 248 | 1687 | 6865 |
| LR | 95 | 0.8593 | 0.9814 | 0.7643 | 0.7643 | 0.0500 | 0.8060 | 0.8571 | 0.7143 | 2356 | 124 | 2016 | 6536 |
| LR | 99 | 0.7759 | 0.9954 | 0.6356 | 0.6356 | 0.0101 | 0.7153 | 0.8128 | 0.6256 | 2455 | 25 | 3116 | 5436 |
| MLP | 50 | 0.9025 | 0.8666 | 0.9417 | 0.9417 | 0.5000 | 0.8424 | 0.7208 | 0.4417 | 1240 | 1240 | 499 | 8053 |
| MLP | 70 | 0.8922 | 0.9096 | 0.8755 | 0.8755 | 0.3000 | 0.8360 | 0.7877 | 0.5755 | 1736 | 744 | 1065 | 7487 |
| MLP | 90 | 0.7657 | 0.9565 | 0.6383 | 0.6383 | 0.1000 | 0.6972 | 0.7692 | 0.5383 | 2232 | 248 | 3093 | 5459 |
| MLP | 95 | 0.6311 | 0.9699 | 0.4677 | 0.4677 | 0.0500 | 0.5761 | 0.7089 | 0.4177 | 2356 | 124 | 4552 | 4000 |
| MLP | 99 | 0.3905 | 0.9881 | 0.2433 | 0.2433 | 0.0101 | 0.4112 | 0.6166 | 0.2333 | 2455 | 25 | 6471 | 2081 |
| LOF | 50 | 0.9219 | 0.8710 | 0.9791 | 0.9791 | 0.5000 | 0.8714 | 0.7395 | 0.4791 | 1240 | 1240 | 179 | 8373 |
| LOF | 70 | 0.9240 | 0.9147 | 0.9335 | 0.9335 | 0.3000 | 0.8810 | 0.8167 | 0.6335 | 1736 | 744 | 569 | 7983 |
| LOF | 90 | 0.8820 | 0.9655 | 0.8117 | 0.8117 | 0.1000 | 0.8316 | 0.8559 | 0.7117 | 2232 | 248 | 1610 | 6942 |
| LOF | 95 | 0.8483 | 0.9810 | 0.7473 | 0.7473 | 0.0500 | 0.7929 | 0.8487 | 0.6973 | 2356 | 124 | 2161 | 6391 |
| LOF | 99 | 0.7158 | 0.9948 | 0.5591 | 0.5591 | 0.0101 | 0.6559 | 0.7745 | 0.5490 | 2455 | 25 | 3771 | 4781 |
| IsolationForest | 50 | 0.8660 | 0.8578 | 0.8744 | 0.8744 | 0.5000 | 0.7902 | 0.6872 | 0.3744 | 1240 | 1240 | 1074 | 7478 |
| IsolationForest | 70 | 0.8089 | 0.8946 | 0.7382 | 0.7382 | 0.3000 | 0.7296 | 0.7191 | 0.4382 | 1736 | 744 | 2239 | 6313 |
| IsolationForest | 90 | 0.6377 | 0.9432 | 0.4816 | 0.4816 | 0.1000 | 0.5757 | 0.6908 | 0.3816 | 2232 | 248 | 4433 | 4119 |
| IsolationForest | 95 | 0.5148 | 0.9604 | 0.3516 | 0.3516 | 0.0500 | 0.4861 | 0.6508 | 0.3016 | 2356 | 124 | 5545 | 3007 |
| IsolationForest | 99 | 0.2366 | 0.9787 | 0.1346 | 0.1346 | 0.0101 | 0.3269 | 0.5623 | 0.1245 | 2455 | 25 | 7401 | 1151 |
