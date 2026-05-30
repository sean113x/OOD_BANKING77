"""OOD_BANKING77 experiment and interactive model entry point."""

from __future__ import annotations

import argparse
import importlib

from experiments.common import (
    all_model_specs,
    calibrate_threshold,
    evaluate_split_rows,
    load_or_fit_model,
    load_standard_splits,
    score_model,
    specs_by_family,
)


EXPERIMENTS = {
    "1": ("Experiment 1: overall method comparison", "experiments.experiment1_overall_comparison"),
    "2": ("Experiment 2: hyperparameter sensitivity", "experiments.experiment2_sensitivity"),
    "3": ("Experiment 3: Near-OOD ratio sweep", "experiments.experiment3_near_ood_difficulty"),
    "4": ("Experiment 4: accuracy vs OOD reliability", "experiments.experiment4_accuracy_vs_ood"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        nargs="?",
        const="menu",
        choices=["menu", "1", "2", "3", "4", "all"],
        help="Run an experiment CSV export. Omit the value to choose from a menu.",
    )
    parser.add_argument("--force-train", action="store_true", help="Retrain models even when saved models exist.")
    parser.add_argument("--list-models", action="store_true", help="Print available interactive model keys.")
    return parser.parse_args()


def run_experiment(selection: str, force_train: bool = False) -> None:
    experiment_ids = list(EXPERIMENTS) if selection == "all" else [selection]
    for experiment_id in experiment_ids:
        title, module_name = EXPERIMENTS[experiment_id]
        print(f"\n{title}")
        module = importlib.import_module(module_name)
        if experiment_id in {"1", "3", "4"}:
            output = module.run(force_train=force_train)
        else:
            output = module.run()
        print(f"Wrote {output}")


def choose_experiment() -> str:
    print("\nSelect an experiment to run.")
    for experiment_id, (title, _) in EXPERIMENTS.items():
        print(f"{experiment_id}. {title}")
    all_option = len(EXPERIMENTS) + 1
    print(f"{all_option}. Run all experiments")

    choice = _choose_number("> ", 1, all_option)
    return "all" if choice == all_option else str(choice)


def list_models() -> None:
    for spec in all_model_specs():
        print(f"{spec.key}: [{spec.family_id}] {spec.display_name} / {spec.score_name}")


def _choose_number(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        value = input(prompt).strip()
        if value.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if value.isdigit() and minimum <= int(value) <= maximum:
            return int(value)
        print(f"Enter a number from {minimum} to {maximum}. Enter q to quit.")


def _label_lookup(train) -> dict[str, str]:
    lookup = {}
    for label, label_text in zip(train["labels"], train["label_texts"]):
        lookup[str(label)] = str(label_text)
        lookup[str(label_text)] = str(label_text)
    return lookup


def _display_validation_metrics(spec, validation, validation_scores, threshold, fit_seconds, loaded) -> None:
    rows = evaluate_split_rows(
        experiment="interactive_validation",
        family=spec.family_name,
        model=spec.display_name,
        score=spec.score_name,
        hyperparameters=spec.hyperparameters,
        threshold_source="validation_best_f1",
        data_split="validation",
        split=validation,
        scores=validation_scores,
        threshold=threshold,
        fit_seconds=fit_seconds,
        extra={"model_source": "loaded" if loaded else "trained"},
    )
    print("\nValidation metrics")
    print(f"model_source={'loaded' if loaded else 'trained'}  time={fit_seconds:.2f}s  threshold={threshold:.6g}")
    for row in rows:
        print(
            f"- {row['metric_split']}: "
            f"AUROC={row['auroc']:.4f}, AUPR={row['aupr']:.4f}, "
            f"F1={row['f1']:.4f}, Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, FPR={row['fpr']:.4f}, Acc={row['accuracy']:.4f}"
        )


def _load_interactive_embedder():
    from BERT.embedding_utils import encode_texts, load_embedding_model

    tokenizer, encoder, device = load_embedding_model()

    def embed(text: str):
        return encode_texts(
            tokenizer,
            encoder,
            [text],
            device,
            batch_size=1,
            show_progress=False,
        )

    return embed


def interactive(force_train: bool = False) -> None:
    train, _, validation, _ = load_standard_splits()
    grouped = specs_by_family()

    print("\nOOD_BANKING77 interactive tester")
    print("Select a model family.")
    for family_id in sorted(grouped):
        print(f"{family_id}. {grouped[family_id][0].family_name}")

    family_id = _choose_number("> ", min(grouped), max(grouped))
    specs = grouped[family_id]

    print("\nSelect a model.")
    for idx, spec in enumerate(specs, start=1):
        print(f"{idx}. {spec.display_name} ({spec.score_name})")

    model_idx = _choose_number("> ", 1, len(specs))
    spec = specs[model_idx - 1]

    print(f"\nPreparing: {spec.display_name}")
    model, loaded, fit_seconds = load_or_fit_model(spec, train, force_train=force_train)
    threshold, validation_scores, _ = calibrate_threshold(spec, model, validation, "validation_best_f1")
    _display_validation_metrics(spec, validation, validation_scores, threshold, fit_seconds, loaded)

    print("\nLoading the embedding encoder once for interactive queries...")
    embed_text = _load_interactive_embedder()
    label_lookup = _label_lookup(train)
    print("\nEnter a query to classify it as ID or OOD. Quit with q, quit, or exit.")
    while True:
        text = input("\nquery> ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break
        if not text:
            continue

        embeddings = embed_text(text)
        scores, labels, _ = score_model(spec, model, embeddings)
        score = float(scores[0])
        predicted = "OOD" if score >= threshold else "ID"
        label = label_lookup.get(str(labels[0]), str(labels[0]))
        print(
            f"{predicted} | score={score:.6g}, threshold={threshold:.6g}, "
            f"nearest/predicted={label}"
        )


def main() -> None:
    args = parse_args()
    try:
        if args.list_models:
            list_models()
            return
        if args.experiment:
            selection = choose_experiment() if args.experiment == "menu" else args.experiment
            run_experiment(selection, force_train=args.force_train)
            return
        interactive(force_train=args.force_train)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
