import numpy as np

from evaluation.metrics import threshold_sweep_metrics, top_k_metrics


def test_threshold_sweep_metrics_returns_best_thresholds():
    y_true = np.asarray([0, 1, 0, 1])
    y_score = np.asarray([0.1, 0.8, 0.4, 0.9])

    metrics = threshold_sweep_metrics(y_true, y_score, thresholds=(0.3, 0.7))

    assert metrics["best_f1_threshold"] == 0.7
    assert metrics["best_precision"] == 1.0


def test_top_k_metrics_groups_by_protein():
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    y_true = np.asarray([0, 1, 1, 0])
    y_score = np.asarray([0.9, 0.8, 0.7, 0.1])

    metrics = top_k_metrics(positions, y_true, y_score, k_values=(1,))

    assert metrics["top_1_precision"] == 0.5
    assert metrics["top_1_protein_recall"] == 0.5
