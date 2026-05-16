import argparse
from pathlib import Path

from data.labeling import infer_index_base
from data.load import load_records
from data.splits import split_records
from data.validate import DatasetReport, validate_records
from evaluation.experiment_log import (
    append_experiment_result,
    current_timestamp,
    experiment_row,
    monotonic_seconds,
    new_run_id,
)
from evaluation.metrics import binary_classification_metrics, threshold_sweep_metrics, top_k_metrics
from models.baseline import TrainingConfig, evaluation_sample, train_baseline
from models.embedding_baseline import EmbeddingTrainingConfig, train_embedding_baseline


def print_report(report: DatasetReport) -> None:
    for field, value in report.model_dump().items():
        print(f"{field}: {value}")


def validate_command(args: argparse.Namespace) -> None:
    records = load_records(args.csv)
    report = validate_records(records)
    print_report(report)


def train_baseline_command(args: argparse.Namespace) -> None:
    run_id = new_run_id()
    created_at = current_timestamp()
    started_at = monotonic_seconds()
    records = load_records(args.csv)
    index_base = infer_index_base(records) if args.index_base == "auto" else args.index_base
    train, validation, test = split_records(
        records,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    config = TrainingConfig(
        model_name=args.model,
        radius=args.radius,
        negative_ratio=args.negative_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_train_positions=args.max_train_positions,
        max_eval_positions=args.max_eval_positions,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        mlp_hidden_units=args.mlp_hidden_units,
        n_jobs=args.n_jobs,
    )
    model, training_log = train_baseline(train, index_base, config)

    validation_x, validation_y = evaluation_sample(
        validation,
        index_base,
        config.radius,
        config.max_eval_positions,
        config.seed,
    )
    validation_scores = model.predict_proba(validation_x)
    metrics = binary_classification_metrics(validation_y, validation_scores, threshold=args.threshold)
    metrics.update(threshold_sweep_metrics(validation_y, validation_scores))
    duration_seconds = monotonic_seconds() - started_at

    append_experiment_result(
        args.results_csv,
        experiment_row(
            run_id=run_id,
            created_at=created_at,
            pipeline="window",
            model=config.model_name,
            embedding_model="",
            csv_path=args.csv,
            index_base=index_base,
            train_proteins=len(train),
            validation_proteins=len(validation),
            test_proteins=len(test),
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            threshold=args.threshold,
            duration_seconds=duration_seconds,
            training_log=training_log,
            config=config,
            metrics=metrics,
        ),
    )

    print(f"run_id: {run_id}")
    print(f"index_base: {index_base}")
    print(f"train_proteins: {len(train)}")
    print(f"validation_proteins: {len(validation)}")
    print(f"test_proteins: {len(test)}")
    print(f"model: {config.model_name}")
    print(f"results_csv: {args.results_csv}")
    print(f"training_log: {', '.join(training_log)}")
    for name, value in metrics.items():
        print(f"validation_{name}: {value:.6f}")


def train_embedding_baseline_command(args: argparse.Namespace) -> None:
    run_id = new_run_id()
    created_at = current_timestamp()
    started_at = monotonic_seconds()
    records = load_records(args.csv)
    index_base = infer_index_base(records) if args.index_base == "auto" else args.index_base
    train, validation, test = split_records(
        records,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    config = EmbeddingTrainingConfig(
        classifier=args.classifier,
        embedding_model=args.embedding_model,
        embedding_cache_dir=args.embedding_cache_dir,
        device=args.device,
        max_residues_per_chunk=args.max_residues_per_chunk,
        negative_ratio=args.negative_ratio,
        max_train_positions=args.max_train_positions,
        max_eval_positions=args.max_eval_positions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        mlp_hidden_units=args.mlp_hidden_units,
        embedding_context_radius=args.embedding_context_radius,
        embedding_context_mode=args.embedding_context_mode,
        n_jobs=args.n_jobs,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    _, training_log, validation_positions, validation_y, validation_scores = train_embedding_baseline(
        train,
        validation,
        index_base,
        config,
    )
    metrics = binary_classification_metrics(validation_y, validation_scores, threshold=args.threshold)
    metrics.update(threshold_sweep_metrics(validation_y, validation_scores))
    metrics.update(top_k_metrics(validation_positions, validation_y, validation_scores))
    duration_seconds = monotonic_seconds() - started_at

    append_experiment_result(
        args.results_csv,
        experiment_row(
            run_id=run_id,
            created_at=created_at,
            pipeline="embedding",
            model=config.classifier,
            embedding_model=config.embedding_model,
            csv_path=args.csv,
            index_base=index_base,
            train_proteins=len(train),
            validation_proteins=len(validation),
            test_proteins=len(test),
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            threshold=args.threshold,
            duration_seconds=duration_seconds,
            training_log=training_log,
            config=config,
            metrics=metrics,
        ),
    )

    print(f"run_id: {run_id}")
    print(f"index_base: {index_base}")
    print(f"train_proteins: {len(train)}")
    print(f"validation_proteins: {len(validation)}")
    print(f"test_proteins: {len(test)}")
    print(f"embedding_model: {config.embedding_model}")
    print(f"classifier: {config.classifier}")
    print(f"embedding_cache_dir: {config.embedding_cache_dir}")
    print(f"embedding_context_radius: {config.embedding_context_radius}")
    print(f"embedding_context_mode: {config.embedding_context_mode}")
    print(f"results_csv: {args.results_csv}")
    print(f"training_log: {', '.join(training_log)}")
    for name, value in metrics.items():
        print(f"validation_{name}: {value:.6f}")


def train_squidly_like_command(args: argparse.Namespace) -> None:
    from models.squidly_like import SquidlyLikeConfig, train_squidly_like

    run_id = new_run_id()
    created_at = current_timestamp()
    started_at = monotonic_seconds()
    records = load_records(args.csv)
    index_base = infer_index_base(records) if args.index_base == "auto" else args.index_base
    train, validation, test = split_records(
        records,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    config = SquidlyLikeConfig(
        embedding_model=args.embedding_model,
        embedding_cache_dir=args.embedding_cache_dir,
        device=args.device,
        max_residues_per_chunk=args.max_residues_per_chunk,
        negative_ratio=args.negative_ratio,
        max_contrastive_positions=args.max_contrastive_positions,
        contrastive_epochs=args.contrastive_epochs,
        contrastive_batches_per_epoch=args.contrastive_batches_per_epoch,
        contrastive_batch_size=args.contrastive_batch_size,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        classifier_epochs=args.classifier_epochs,
        learning_rate=args.learning_rate,
        max_train_sequences=args.max_train_sequences,
        max_eval_positions=args.max_eval_positions,
        max_sequence_length=args.max_sequence_length,
        positive_weight=args.positive_weight,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    training_log, validation_positions, validation_y, validation_scores, used_device = train_squidly_like(
        train,
        validation,
        index_base,
        config,
    )
    metrics = binary_classification_metrics(validation_y, validation_scores, threshold=args.threshold)
    metrics.update(threshold_sweep_metrics(validation_y, validation_scores))
    metrics.update(top_k_metrics(validation_positions, validation_y, validation_scores))
    duration_seconds = monotonic_seconds() - started_at

    append_experiment_result(
        args.results_csv,
        experiment_row(
            run_id=run_id,
            created_at=created_at,
            pipeline="squidly_like",
            model="contrastive_projection_bilstm",
            embedding_model=config.embedding_model,
            csv_path=args.csv,
            index_base=index_base,
            train_proteins=len(train),
            validation_proteins=len(validation),
            test_proteins=len(test),
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            threshold=args.threshold,
            duration_seconds=duration_seconds,
            training_log=training_log,
            config=config,
            metrics=metrics,
        ),
    )

    print(f"run_id: {run_id}")
    print(f"index_base: {index_base}")
    print(f"train_proteins: {len(train)}")
    print(f"validation_proteins: {len(validation)}")
    print(f"test_proteins: {len(test)}")
    print(f"embedding_model: {config.embedding_model}")
    print("model: contrastive_projection_bilstm")
    print(f"device: {used_device}")
    print(f"results_csv: {args.results_csv}")
    print(f"training_log: {', '.join(training_log)}")
    for name, value in metrics.items():
        print(f"validation_{name}: {value:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crp")
    subparsers = parser.add_subparsers(required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate and summarize a catalytic residue CSV.")
    validate_parser.add_argument("--csv", type=Path, default=Path("data/residual_amino_database.csv"))
    validate_parser.set_defaults(func=validate_command)

    baseline_parser = subparsers.add_parser("train-baseline", help="Train a window-feature baseline model.")
    baseline_parser.add_argument("--csv", type=Path, default=Path("data/residual_amino_database.csv"))
    baseline_parser.add_argument(
        "--model",
        choices=["logistic", "random-forest", "xgboost", "mlp"],
        default="logistic",
    )
    baseline_parser.add_argument("--index-base", choices=["auto", "zero", "one"], default="auto")
    baseline_parser.add_argument("--train-fraction", type=float, default=0.8)
    baseline_parser.add_argument("--validation-fraction", type=float, default=0.1)
    baseline_parser.add_argument("--radius", type=int, default=10)
    baseline_parser.add_argument("--negative-ratio", type=int, default=10)
    baseline_parser.add_argument("--epochs", type=int, default=5)
    baseline_parser.add_argument("--batch-size", type=int, default=512)
    baseline_parser.add_argument("--learning-rate", type=float, default=0.05)
    baseline_parser.add_argument("--max-train-positions", type=int, default=250_000)
    baseline_parser.add_argument("--max-eval-positions", type=int, default=250_000)
    baseline_parser.add_argument("--n-estimators", type=int, default=100)
    baseline_parser.add_argument("--max-depth", type=int, default=None)
    baseline_parser.add_argument("--mlp-hidden-units", type=int, default=64)
    baseline_parser.add_argument("--n-jobs", type=int, default=1)
    baseline_parser.add_argument("--threshold", type=float, default=0.5)
    baseline_parser.add_argument("--seed", type=int, default=13)
    baseline_parser.add_argument("--results-csv", type=Path, default=Path("results/experiments.csv"))
    baseline_parser.set_defaults(func=train_baseline_command)

    embedding_parser = subparsers.add_parser(
        "train-embedding-baseline",
        help="Train a classifier over per-residue protein language model embeddings.",
    )
    embedding_parser.add_argument("--csv", type=Path, default=Path("data/residual_amino_database.csv"))
    embedding_parser.add_argument(
        "--classifier",
        choices=["logistic", "random-forest", "xgboost", "mlp"],
        default="logistic",
    )
    embedding_parser.add_argument("--embedding-model", default="facebook/esm2_t6_8M_UR50D")
    embedding_parser.add_argument("--embedding-cache-dir", type=Path, default=Path("data/embedding_cache"))
    embedding_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    embedding_parser.add_argument("--max-residues-per-chunk", type=int, default=900)
    embedding_parser.add_argument("--index-base", choices=["auto", "zero", "one"], default="auto")
    embedding_parser.add_argument("--train-fraction", type=float, default=0.8)
    embedding_parser.add_argument("--validation-fraction", type=float, default=0.1)
    embedding_parser.add_argument("--negative-ratio", type=int, default=10)
    embedding_parser.add_argument("--epochs", type=int, default=20)
    embedding_parser.add_argument("--batch-size", type=int, default=512)
    embedding_parser.add_argument("--learning-rate", type=float, default=0.001)
    embedding_parser.add_argument("--max-train-positions", type=int, default=50_000)
    embedding_parser.add_argument("--max-eval-positions", type=int, default=50_000)
    embedding_parser.add_argument("--n-estimators", type=int, default=100)
    embedding_parser.add_argument("--max-depth", type=int, default=None)
    embedding_parser.add_argument("--mlp-hidden-units", type=int, default=128)
    embedding_parser.add_argument("--n-jobs", type=int, default=1)
    embedding_parser.add_argument("--embedding-context-radius", type=int, default=0)
    embedding_parser.add_argument("--embedding-context-mode", choices=["concat", "mean"], default="concat")
    embedding_parser.add_argument("--threshold", type=float, default=0.5)
    embedding_parser.add_argument("--seed", type=int, default=13)
    embedding_parser.add_argument("--results-csv", type=Path, default=Path("results/experiments.csv"))
    embedding_parser.add_argument("--no-progress", action="store_true")
    embedding_parser.set_defaults(func=train_embedding_baseline_command)

    squidly_parser = subparsers.add_parser(
        "train-squidly-like",
        help="Train a Squidly-inspired contrastive projection plus BiLSTM token classifier.",
    )
    squidly_parser.add_argument("--csv", type=Path, default=Path("data/residual_amino_database.csv"))
    squidly_parser.add_argument("--embedding-model", default="facebook/esm2_t6_8M_UR50D")
    squidly_parser.add_argument("--embedding-cache-dir", type=Path, default=Path("data/embedding_cache"))
    squidly_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    squidly_parser.add_argument("--max-residues-per-chunk", type=int, default=900)
    squidly_parser.add_argument("--index-base", choices=["auto", "zero", "one"], default="auto")
    squidly_parser.add_argument("--train-fraction", type=float, default=0.8)
    squidly_parser.add_argument("--validation-fraction", type=float, default=0.1)
    squidly_parser.add_argument("--negative-ratio", type=int, default=10)
    squidly_parser.add_argument("--max-contrastive-positions", type=int, default=50_000)
    squidly_parser.add_argument("--contrastive-epochs", type=int, default=3)
    squidly_parser.add_argument("--contrastive-batches-per-epoch", type=int, default=500)
    squidly_parser.add_argument("--contrastive-batch-size", type=int, default=512)
    squidly_parser.add_argument("--projection-dim", type=int, default=128)
    squidly_parser.add_argument("--projection-hidden-dim", type=int, default=512)
    squidly_parser.add_argument("--projection-dropout", type=float, default=0.1)
    squidly_parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    squidly_parser.add_argument("--lstm-layers", type=int, default=2)
    squidly_parser.add_argument("--lstm-dropout", type=float, default=0.2)
    squidly_parser.add_argument("--classifier-epochs", type=int, default=5)
    squidly_parser.add_argument("--learning-rate", type=float, default=0.001)
    squidly_parser.add_argument("--max-train-sequences", type=int, default=1000)
    squidly_parser.add_argument("--max-eval-positions", type=int, default=50_000)
    squidly_parser.add_argument("--max-sequence-length", type=int, default=1500)
    squidly_parser.add_argument("--positive-weight", type=float, default=100.0)
    squidly_parser.add_argument("--threshold", type=float, default=0.5)
    squidly_parser.add_argument("--seed", type=int, default=13)
    squidly_parser.add_argument("--results-csv", type=Path, default=Path("results/experiments.csv"))
    squidly_parser.add_argument("--no-progress", action="store_true")
    squidly_parser.set_defaults(func=train_squidly_like_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
