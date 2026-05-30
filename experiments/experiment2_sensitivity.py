"""Experiment 2: hyperparameter and threshold sensitivity.

Each configuration is trained on ``OOD_train``. Thresholds are selected from
train ID scores or ``OOD_validation`` and then reported on both validation and
``OOD_test_eval``. The output is one CSV file.

Usage:
  python experiments/experiment2_sensitivity.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from experiments.common import (
    EXPERIMENT_MODEL_DIR,
    RESULT_DIR,
    GaussianMixtureOODModel,
    KNNDistanceModel,
    OneClassSVMModel,
    PCAReconstructionModel,
    RadiusThresholdModel,
    ModelSpec,
    best_f1_threshold,
    calibrate_threshold,
    energy_score,
    entropy_score,
    evaluate_split_rows,
    fit_model,
    id_percentile_threshold,
    load_standard_splits,
    print_metric_table,
    score_model,
    write_csv,
    y_true_from_split,
)


DEFAULT_OUTPUT = RESULT_DIR / "experiment2_hyperparameter_sensitivity.csv"
SELECTION_METRIC = "f1"
OCSVM_NU_VALUES = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)
OCSVM_GAMMA_VALUES = ("scale", 0.01, 0.1, 1.0, 5.0, 10.0, 30.0)
GMM_CONFIGS = (
    (8, "diag"),
    (16, "diag"),
    (32, "diag"),
    (62, "diag"),
    (77, "diag"),
    (128, "diag"),
    (256, "diag"),
    (16, "full"),
    (32, "full"),
    (62, "full"),
    (128, "full"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _custom_spec(
    key: str,
    display_name: str,
    family_name: str,
    score_name: str,
    hyperparameters: str,
    cls: Any,
    kwargs: dict[str, Any],
) -> ModelSpec:
    path = EXPERIMENT_MODEL_DIR / "experiment2" / f"{key}.pkl"

    def build():
        return cls(path, **kwargs)

    def fit(model, train, save):
        return model.fit(train["embeddings"], train["label_texts"], save=False)

    def score(model, embeddings):
        return model.score_embeddings(embeddings)

    return ModelSpec(
        key=key,
        display_name=display_name,
        family_id=0,
        family_name=family_name,
        score_name=score_name,
        hyperparameters=hyperparameters,
        model_path=path,
        build=build,
        fit=fit,
        score=score,
    )


def _classifier_spec(
    key: str,
    display_name: str,
    score_name: str,
    hyperparameters: str,
    clf: Any,
    score_fn,
) -> ModelSpec:
    path = EXPERIMENT_MODEL_DIR / "experiment2" / f"{key}.pkl"

    def build():
        return clf

    def fit(model, train, save):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*convergence.*")
            warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")
            model.fit(train["embeddings"], train["labels"])
        return model

    def score(model, embeddings):
        values = score_fn(model, embeddings)
        labels = model.predict(embeddings).astype(str)
        return values, labels

    return ModelSpec(
        key=key,
        display_name=display_name,
        family_id=0,
        family_name="Classifier-output scoring",
        score_name=score_name,
        hyperparameters=hyperparameters,
        model_path=path,
        build=build,
        fit=fit,
        score=score,
    )


def model_configs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    for percentile in (90.0, 95.0, 97.5, 99.0):
        specs.append(
            _custom_spec(
                key=f"radius_p{str(percentile).replace('.', '_')}",
                display_name="Class-wise radius",
                family_name="Class-wise boundary",
                score_name="distance_minus_radius",
                hyperparameters=f"radius_percentile={percentile}",
                cls=RadiusThresholdModel,
                kwargs={"radius_percentile": percentile},
            )
        )

    for k in (1, 5, 10, 20, 50):
        specs.append(
            _custom_spec(
                key=f"knn_k{k}",
                display_name="kNN distance",
                family_name="Distance/support/density",
                score_name="mean_knn_distance",
                hyperparameters=f"k={k}, distance=cosine",
                cls=KNNDistanceModel,
                kwargs={"n_neighbors": k},
            )
        )

    for nu in OCSVM_NU_VALUES:
        for gamma in OCSVM_GAMMA_VALUES:
            specs.append(
                _custom_spec(
                    key=f"ocsvm_nu{str(nu).replace('.', '_')}_gamma{str(gamma).replace('.', '_')}",
                    display_name="One-Class SVM",
                    family_name="Distance/support/density",
                    score_name="negative_decision_function",
                    hyperparameters=f"kernel=rbf, nu={nu}, gamma={gamma}",
                    cls=OneClassSVMModel,
                    kwargs={"nu": nu, "gamma": gamma},
                )
            )

    for n_components, covariance_type in GMM_CONFIGS:
        specs.append(
            _custom_spec(
                key=f"gmm_c{n_components}_{covariance_type}",
                display_name="Gaussian Mixture",
                family_name="Distance/support/density",
                score_name="negative_log_likelihood",
                hyperparameters=f"n_components={n_components}, covariance={covariance_type}",
                cls=GaussianMixtureOODModel,
                kwargs={
                    "n_components": n_components,
                    "covariance_type": covariance_type,
                    "reg_covar": 1e-5,
                    "max_iter": 200,
                },
            )
        )

    for n_components in (64, 128, 256):
        specs.append(
            _custom_spec(
                key=f"pca_{n_components}",
                display_name="PCA reconstruction",
                family_name="Distance/support/density",
                score_name="reconstruction_mse",
                hyperparameters=f"n_components={n_components}",
                cls=PCAReconstructionModel,
                kwargs={"n_components": n_components},
            )
        )

    for c_value in (0.1, 1.0, 10.0, 100.0):
        specs.append(
            _classifier_spec(
                key=f"lr_c{str(c_value).replace('.', '_')}_energy",
                display_name="Logistic Regression + Energy",
                score_name="Energy",
                hyperparameters=f"C={c_value}",
                clf=LogisticRegression(C=c_value, max_iter=3000, solver="saga", random_state=42, n_jobs=1),
                score_fn=energy_score,
            )
        )

    mlp_configs = (
        ((256,), 0.001, 0.001),
        ((512,), 0.001, 0.001),
        ((512, 256), 0.0005, 0.001),
        ((512, 256), 0.001, 0.001),
    )
    for hidden, lr, alpha in mlp_configs:
        hidden_text = "x".join(str(value) for value in hidden)
        specs.append(
            _classifier_spec(
                key=f"mlp_{hidden_text}_lr{str(lr).replace('.', '_')}_entropy",
                display_name="MLP + Entropy",
                score_name="Entropy",
                hyperparameters=f"hidden={hidden_text}, lr={lr}, alpha={alpha}",
                clf=MLPClassifier(
                    hidden_layer_sizes=hidden,
                    learning_rate_init=lr,
                    alpha=alpha,
                    max_iter=500,
                    tol=1e-5,
                    n_iter_no_change=20,
                    random_state=42,
                ),
                score_fn=entropy_score,
            )
        )

    return specs


def _thresholds(spec: ModelSpec, model, train, validation, validation_scores):
    train_scores, _, _ = score_model(spec, model, train["embeddings"])
    y_train = y_true_from_split(train)
    y_validation = y_true_from_split(validation)
    return [
        ("train_id95", id_percentile_threshold(y_train, train_scores, 95.0)),
        ("validation_id95", id_percentile_threshold(y_validation, validation_scores, 95.0)),
        ("validation_best_f1", best_f1_threshold(y_validation, validation_scores)),
    ]


def _selection_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["family"]),
        str(row["model"]),
        str(row["score"]),
        str(row["threshold_source"]),
    )


def _annotate_validation_selection(rows: list[dict]) -> None:
    validation_overall = [
        row
        for row in rows
        if row["data_split"] == "validation" and row["metric_split"] == "Overall"
    ]
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in validation_overall:
        grouped.setdefault(_selection_key(row), []).append(row)

    selected_by_group = {}
    rank_by_group_and_model = {}
    for key, group_rows in grouped.items():
        ranked = sorted(
            group_rows,
            key=lambda row: (
                float(row.get(SELECTION_METRIC, float("-inf"))),
                float(row.get("balanced_accuracy", float("-inf"))),
                float(row.get("auroc", float("-inf"))),
            ),
            reverse=True,
        )
        selected_by_group[key] = ranked[0]
        for rank, row in enumerate(ranked, start=1):
            rank_by_group_and_model[(key, row["model_key"])] = rank

    for row in rows:
        key = _selection_key(row)
        selected = selected_by_group[key]
        row["validation_selection_metric"] = SELECTION_METRIC
        row["validation_selection_group"] = " | ".join(key)
        row["validation_selected_model_key"] = selected["model_key"]
        row["validation_selected_hyperparameters"] = selected["hyperparameters"]
        row["validation_selected_score"] = selected[SELECTION_METRIC]
        row["validation_selection_rank"] = rank_by_group_and_model[(key, row["model_key"])]
        row["is_validation_selected"] = row["model_key"] == selected["model_key"]


def run(output_path: Path = DEFAULT_OUTPUT) -> Path:
    train, _, validation, eval_split = load_standard_splits()
    rows = []

    for spec in model_configs():
        model, fit_seconds = fit_model(spec, train, save=False)
        _, validation_scores, _ = calibrate_threshold(spec, model, validation, "validation_best_f1")
        eval_scores, _, score_seconds = score_model(spec, model, eval_split["embeddings"])

        for threshold_source, threshold in _thresholds(spec, model, train, validation, validation_scores):
            extra = {"model_key": spec.key}
            rows.extend(
                evaluate_split_rows(
                    experiment="experiment2_hyperparameter_sensitivity",
                    family=spec.family_name,
                    model=spec.display_name,
                    score=spec.score_name,
                    hyperparameters=spec.hyperparameters,
                    threshold_source=threshold_source,
                    data_split="validation",
                    split=validation,
                    scores=validation_scores,
                    threshold=threshold,
                    fit_seconds=fit_seconds,
                    extra=extra,
                )
            )
            rows.extend(
                evaluate_split_rows(
                    experiment="experiment2_hyperparameter_sensitivity",
                    family=spec.family_name,
                    model=spec.display_name,
                    score=spec.score_name,
                    hyperparameters=spec.hyperparameters,
                    threshold_source=threshold_source,
                    data_split="eval",
                    split=eval_split,
                    scores=eval_scores,
                    threshold=threshold,
                    fit_seconds=fit_seconds,
                    score_seconds=score_seconds,
                    extra=extra,
                )
            )

    _annotate_validation_selection(rows)
    output = write_csv(output_path, rows)
    print_metric_table(
        rows,
        "Experiment 2 metrics (eval / Overall)",
        max_rows=120,
    )
    print_metric_table(
        [row for row in rows if row["is_validation_selected"]],
        "Experiment 2 validation-selected configs (eval / Overall)",
        max_rows=120,
    )
    return output


def main() -> None:
    args = parse_args()
    output = run(args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
