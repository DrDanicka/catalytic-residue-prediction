from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Literal


Profile = Literal["quick", "full"]


@dataclass(frozen=True)
class Experiment:
    name: str
    command: list[str]
    requires_embeddings: bool = False


def add_common(command: list[str], csv_path: Path, results_csv: Path, seed: int, no_progress: bool) -> list[str]:
    command.extend(["--csv", str(csv_path), "--results-csv", str(results_csv), "--seed", str(seed)])
    if no_progress and command[0] in {"train-embedding-baseline", "train-squidly-like"}:
        command.append("--no-progress")
    return command


def experiments(profile: Profile, csv_path: Path, results_csv: Path, seed: int, no_progress: bool) -> list[Experiment]:
    if profile == "quick":
        window_train = "20000"
        window_eval = "50000"
        embedding_train = "10000"
        embedding_eval = "50000"
        squidly_positions = "15000"
        squidly_batches = "150"
        squidly_sequences = "300"
        squidly_eval = "50000"
        squidly_contrastive_epochs = "2"
        squidly_classifier_epochs = "2"
    else:
        window_train = "100000"
        window_eval = "300000"
        embedding_train = "50000"
        embedding_eval = "300000"
        squidly_positions = "50000"
        squidly_batches = "500"
        squidly_sequences = "1000"
        squidly_eval = "300000"
        squidly_contrastive_epochs = "3"
        squidly_classifier_epochs = "5"

    base: list[Experiment] = [
        Experiment(
            "window_logistic_r10",
            add_common(
                [
                    "train-baseline",
                    "--model",
                    "logistic",
                    "--radius",
                    "10",
                    "--negative-ratio",
                    "10",
                    "--epochs",
                    "5",
                    "--max-train-positions",
                    window_train,
                    "--max-eval-positions",
                    window_eval,
                    "--threshold",
                    "0.5",
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
        ),
        Experiment(
            "window_mlp_r10",
            add_common(
                [
                    "train-baseline",
                    "--model",
                    "mlp",
                    "--radius",
                    "10",
                    "--negative-ratio",
                    "10",
                    "--epochs",
                    "30",
                    "--learning-rate",
                    "0.001",
                    "--mlp-hidden-units",
                    "128",
                    "--max-train-positions",
                    window_train,
                    "--max-eval-positions",
                    window_eval,
                    "--threshold",
                    "0.5",
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
        ),
        Experiment(
            "window_random_forest",
            add_common(
                [
                    "train-baseline",
                    "--model",
                    "random-forest",
                    "--radius",
                    "10",
                    "--negative-ratio",
                    "10",
                    "--n-estimators",
                    "100",
                    "--max-depth",
                    "12",
                    "--max-train-positions",
                    window_train,
                    "--max-eval-positions",
                    window_eval,
                    "--threshold",
                    "0.5",
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
        ),
        Experiment(
            "window_xgboost",
            add_common(
                [
                    "train-baseline",
                    "--model",
                    "xgboost",
                    "--radius",
                    "10",
                    "--negative-ratio",
                    "10",
                    "--n-estimators",
                    "250",
                    "--max-depth",
                    "4",
                    "--learning-rate",
                    "0.03",
                    "--max-train-positions",
                    window_train,
                    "--max-eval-positions",
                    window_eval,
                    "--threshold",
                    "0.5",
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
        ),
    ]

    embedding_model = "facebook/esm2_t6_8M_UR50D"
    embedding: list[Experiment] = [
        Experiment(
            "embedding_logistic_single",
            add_common(
                [
                    "train-embedding-baseline",
                    "--classifier",
                    "logistic",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--epochs",
                    "200",
                    "--max-train-positions",
                    embedding_train,
                    "--max-eval-positions",
                    embedding_eval,
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "embedding_mlp_single",
            add_common(
                [
                    "train-embedding-baseline",
                    "--classifier",
                    "mlp",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--epochs",
                    "50",
                    "--learning-rate",
                    "0.001",
                    "--mlp-hidden-units",
                    "128",
                    "--max-train-positions",
                    embedding_train,
                    "--max-eval-positions",
                    embedding_eval,
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "embedding_xgboost_single",
            add_common(
                [
                    "train-embedding-baseline",
                    "--classifier",
                    "xgboost",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--n-estimators",
                    "300",
                    "--max-depth",
                    "4",
                    "--learning-rate",
                    "0.03",
                    "--max-train-positions",
                    embedding_train,
                    "--max-eval-positions",
                    embedding_eval,
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "embedding_mlp_context_concat_r1",
            add_common(
                [
                    "train-embedding-baseline",
                    "--classifier",
                    "mlp",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--epochs",
                    "50",
                    "--learning-rate",
                    "0.001",
                    "--mlp-hidden-units",
                    "128",
                    "--embedding-context-radius",
                    "1",
                    "--embedding-context-mode",
                    "concat",
                    "--max-train-positions",
                    embedding_train,
                    "--max-eval-positions",
                    embedding_eval,
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "embedding_mlp_context_concat_r2",
            add_common(
                [
                    "train-embedding-baseline",
                    "--classifier",
                    "mlp",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--epochs",
                    "50",
                    "--learning-rate",
                    "0.001",
                    "--mlp-hidden-units",
                    "128",
                    "--embedding-context-radius",
                    "2",
                    "--embedding-context-mode",
                    "concat",
                    "--max-train-positions",
                    embedding_train,
                    "--max-eval-positions",
                    embedding_eval,
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "embedding_mlp_context_mean_r2",
            add_common(
                [
                    "train-embedding-baseline",
                    "--classifier",
                    "mlp",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--epochs",
                    "50",
                    "--learning-rate",
                    "0.001",
                    "--mlp-hidden-units",
                    "128",
                    "--embedding-context-radius",
                    "2",
                    "--embedding-context-mode",
                    "mean",
                    "--max-train-positions",
                    embedding_train,
                    "--max-eval-positions",
                    embedding_eval,
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "embedding_xgboost_context_mean_r2",
            add_common(
                [
                    "train-embedding-baseline",
                    "--classifier",
                    "xgboost",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--n-estimators",
                    "300",
                    "--max-depth",
                    "4",
                    "--learning-rate",
                    "0.03",
                    "--embedding-context-radius",
                    "2",
                    "--embedding-context-mode",
                    "mean",
                    "--max-train-positions",
                    embedding_train,
                    "--max-eval-positions",
                    embedding_eval,
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
    ]

    squidly: list[Experiment] = [
        Experiment(
            "squidly_like_small",
            add_common(
                [
                    "train-squidly-like",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--max-contrastive-positions",
                    squidly_positions,
                    "--contrastive-epochs",
                    squidly_contrastive_epochs,
                    "--contrastive-batches-per-epoch",
                    squidly_batches,
                    "--contrastive-batch-size",
                    "512",
                    "--classifier-epochs",
                    squidly_classifier_epochs,
                    "--max-train-sequences",
                    squidly_sequences,
                    "--max-eval-positions",
                    squidly_eval,
                    "--max-sequence-length",
                    "1500",
                    "--positive-weight",
                    "100",
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "squidly_like_more_pairs",
            add_common(
                [
                    "train-squidly-like",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "20",
                    "--max-contrastive-positions",
                    squidly_positions,
                    "--contrastive-epochs",
                    squidly_contrastive_epochs,
                    "--contrastive-batches-per-epoch",
                    str(int(squidly_batches) + 200),
                    "--contrastive-batch-size",
                    "512",
                    "--classifier-epochs",
                    squidly_classifier_epochs,
                    "--max-train-sequences",
                    squidly_sequences,
                    "--max-eval-positions",
                    squidly_eval,
                    "--max-sequence-length",
                    "1500",
                    "--positive-weight",
                    "100",
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "squidly_like_lower_pos_weight",
            add_common(
                [
                    "train-squidly-like",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--max-contrastive-positions",
                    squidly_positions,
                    "--contrastive-epochs",
                    squidly_contrastive_epochs,
                    "--contrastive-batches-per-epoch",
                    squidly_batches,
                    "--contrastive-batch-size",
                    "512",
                    "--classifier-epochs",
                    squidly_classifier_epochs,
                    "--max-train-sequences",
                    squidly_sequences,
                    "--max-eval-positions",
                    squidly_eval,
                    "--max-sequence-length",
                    "1500",
                    "--positive-weight",
                    "50",
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
        Experiment(
            "squidly_like_longer_sequences",
            add_common(
                [
                    "train-squidly-like",
                    "--embedding-model",
                    embedding_model,
                    "--negative-ratio",
                    "10",
                    "--max-contrastive-positions",
                    squidly_positions,
                    "--contrastive-epochs",
                    squidly_contrastive_epochs,
                    "--contrastive-batches-per-epoch",
                    squidly_batches,
                    "--contrastive-batch-size",
                    "512",
                    "--classifier-epochs",
                    squidly_classifier_epochs,
                    "--max-train-sequences",
                    squidly_sequences,
                    "--max-eval-positions",
                    squidly_eval,
                    "--max-sequence-length",
                    "2500",
                    "--positive-weight",
                    "100",
                ],
                csv_path,
                results_csv,
                seed,
                no_progress,
            ),
            requires_embeddings=True,
        ),
    ]

    return base + embedding + squidly


def run_experiment(experiment: Experiment, python_executable: str, dry_run: bool) -> None:
    command = [python_executable, "-m", "cli", *experiment.command]
    print("\n" + "=" * 100)
    print(f"Experiment: {experiment.name}")
    print("Command:")
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a suite of catalytic residue prediction experiments.")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--csv", type=Path, default=Path("data/residual_amino_database.csv"))
    parser.add_argument("--results-csv", type=Path, default=Path("results/experiment_suite.csv"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--start-at", default=None, help="Experiment name to start from.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected experiment names.")
    parser.add_argument("--list", action="store_true", help="List experiments and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars for embedding experiments.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suite = experiments(args.profile, args.csv, args.results_csv, args.seed, args.no_progress)

    if args.only:
        selected = set(args.only)
        suite = [experiment for experiment in suite if experiment.name in selected]
    if args.start_at:
        names = [experiment.name for experiment in suite]
        if args.start_at not in names:
            raise SystemExit(f"Unknown experiment for --start-at: {args.start_at}")
        suite = suite[names.index(args.start_at) :]
    if args.list:
        for index, experiment in enumerate(suite, start=1):
            extra = " embeddings" if experiment.requires_embeddings else ""
            print(f"{index:02d}. {experiment.name}{extra}")
        return

    print(f"Running {len(suite)} experiments with profile={args.profile}")
    print(f"Results CSV: {args.results_csv}")
    for experiment in suite:
        run_experiment(experiment, sys.executable, args.dry_run)


if __name__ == "__main__":
    main()

