import csv

from evaluation.experiment_log import append_experiment_result, experiment_row


def test_append_experiment_result_writes_header_once(tmp_path):
    path = tmp_path / "experiments.csv"
    row = experiment_row(
        run_id="run-1",
        created_at="2026-05-16T10:00:00+00:00",
        pipeline="window",
        model="mlp",
        embedding_model="",
        csv_path=tmp_path / "data.csv",
        index_base="zero",
        train_proteins=8,
        validation_proteins=1,
        test_proteins=1,
        train_fraction=0.8,
        validation_fraction=0.1,
        threshold=0.5,
        duration_seconds=1.25,
        training_log=["loss=0.1"],
        config={"model": "mlp"},
        metrics={"precision": 0.2, "recall": 0.4, "pr_auc": 0.3},
    )

    append_experiment_result(path, row)
    append_experiment_result(path, row | {"run_id": "run-2"})

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert rows[1]["run_id"] == "run-2"
    assert rows[0]["validation_precision"] == "0.2"
    assert rows[0]["training_log"] == "loss=0.1"

