import numpy as np


def threshold_sweep_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> dict[str, float]:
    best_f1 = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    best_precision = {"threshold": 0.0, "precision": -1.0, "recall": 0.0, "f1": 0.0}

    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, y_score, threshold=threshold)
        if metrics["f1"] > best_f1["f1"]:
            best_f1 = {
                "threshold": threshold,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        if metrics["precision"] > best_precision["precision"] and metrics["recall"] > 0:
            best_precision = {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }

    return {
        "best_f1_threshold": best_f1["threshold"],
        "best_f1": best_f1["f1"],
        "best_f1_precision": best_f1["precision"],
        "best_f1_recall": best_f1["recall"],
        "best_precision_threshold": best_precision["threshold"],
        "best_precision": max(best_precision["precision"], 0.0),
        "best_precision_recall": best_precision["recall"],
        "best_precision_f1": best_precision["f1"],
    }


def top_k_metrics(
    positions: list[tuple[int, int]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    grouped: dict[int, list[int]] = {}
    for row, (record_index, _) in enumerate(positions):
        grouped.setdefault(record_index, []).append(row)

    metrics: dict[str, float] = {}
    for k in k_values:
        true_positives = 0
        predicted_positives = 0
        recovered_positive_proteins = 0
        positive_proteins = 0

        for rows in grouped.values():
            group_labels = y_true[rows]
            positives_in_group = int(np.sum(group_labels == 1))
            if positives_in_group > 0:
                positive_proteins += 1

            group_scores = y_score[rows]
            top_count = min(k, len(rows))
            top_local = np.argsort(-group_scores)[:top_count]
            top_labels = group_labels[top_local]
            hits = int(np.sum(top_labels == 1))
            true_positives += hits
            predicted_positives += top_count
            if positives_in_group > 0 and hits > 0:
                recovered_positive_proteins += 1

        metrics[f"top_{k}_precision"] = true_positives / predicted_positives if predicted_positives else 0.0
        metrics[f"top_{k}_protein_recall"] = (
            recovered_positive_proteins / positive_proteins if positive_proteins else 0.0
        )
    return metrics


def binary_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.int8)
    y_pred = (y_score >= threshold).astype(np.int8)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "positive_rate": float(np.mean(y_true)),
    }


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    positives = np.sum(y_sorted == 1)
    if positives == 0:
        return 0.0

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.trapezoid(precision, recall))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    positive_rank_sum = np.sum(ranks[y_true == 1])
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)
