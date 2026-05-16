from datetime import UTC, datetime
import csv
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4


EXPERIMENT_FIELDNAMES = [
    "run_id",
    "created_at",
    "pipeline",
    "model",
    "embedding_model",
    "csv",
    "index_base",
    "train_proteins",
    "validation_proteins",
    "test_proteins",
    "train_fraction",
    "validation_fraction",
    "threshold",
    "duration_seconds",
    "training_log",
    "config_json",
    "validation_precision",
    "validation_recall",
    "validation_specificity",
    "validation_f1",
    "validation_mcc",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_positive_rate",
    "validation_best_f1_threshold",
    "validation_best_f1",
    "validation_best_f1_precision",
    "validation_best_f1_recall",
    "validation_best_precision_threshold",
    "validation_best_precision",
    "validation_best_precision_recall",
    "validation_best_precision_f1",
    "validation_top_1_precision",
    "validation_top_1_protein_recall",
    "validation_top_3_precision",
    "validation_top_3_protein_recall",
    "validation_top_5_precision",
    "validation_top_5_protein_recall",
    "validation_top_10_precision",
    "validation_top_10_protein_recall",
]


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]


def current_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def monotonic_seconds() -> float:
    return time.perf_counter()


def serialize_config(config: Any) -> str:
    if hasattr(config, "model_dump"):
        payload = config.model_dump(mode="json")
    else:
        payload = config
    return json.dumps(payload, sort_keys=True)


def append_experiment_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    normalized_row = {field: row.get(field, "") for field in EXPERIMENT_FIELDNAMES}

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized_row)


def experiment_row(
    *,
    run_id: str,
    created_at: str,
    pipeline: str,
    model: str,
    embedding_model: str,
    csv_path: Path,
    index_base: str,
    train_proteins: int,
    validation_proteins: int,
    test_proteins: int,
    train_fraction: float,
    validation_fraction: float,
    threshold: float,
    duration_seconds: float,
    training_log: list[str],
    config: Any,
    metrics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "pipeline": pipeline,
        "model": model,
        "embedding_model": embedding_model,
        "csv": str(csv_path),
        "index_base": index_base,
        "train_proteins": train_proteins,
        "validation_proteins": validation_proteins,
        "test_proteins": test_proteins,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "threshold": threshold,
        "duration_seconds": round(duration_seconds, 3),
        "training_log": " | ".join(training_log),
        "config_json": serialize_config(config),
    }
    for metric_name, metric_value in metrics.items():
        row[f"validation_{metric_name}"] = metric_value
    return row
