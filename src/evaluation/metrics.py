"""Evaluation Metrics Module for Time-Series Anomaly Detection.

Calculates point-wise classification and ranking metrics (Average Precision, ROC-AUC,
Precision, Recall, F1-Score, Confusion Matrix counts) and distribution shift performance degradation.
Handles edge cases (single-class arrays, zero predicted positives) by returning null (None)
with explicit explanatory fields rather than invalid numbers.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_pointwise_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Calculate point-wise anomaly detection evaluation metrics.

    Positive class (1) represents anomalies; negative class (0) represents normal timestamps.

    Args:
        y_true: 1D binary ground-truth labels array of length N (values in {0, 1}).
        y_score: 1D float continuous anomaly scores of length N (higher = more anomalous).
        y_pred: 1D binary predictions array of length N (values in {0, 1}).

    Returns:
        Dictionary containing:
        - 'pr_auc': Average Precision (AP), primary ranking metric across all thresholds.
        - 'roc_auc': Area Under ROC Curve (or None if undefined).
        - 'precision': Point-wise Precision at frozen threshold.
        - 'recall': Point-wise Recall at frozen threshold.
        - 'f1': Point-wise F1 score at frozen threshold.
        - 'tp', 'fp', 'tn', 'fn': Integer confusion matrix counts.
        - 'total_points': Total observation count N.
        - 'anomalous_points': Ground-truth positive count.
        - 'anomaly_prevalence': Ratio of anomalous timestamps (float).
        - 'predicted_anomalous_points': Total predicted positive count.
        - 'metric_notes': Dict of any caveats or reasons for null metrics.

    Raises:
        ValueError: If input shapes mismatch, are empty, or contain non-finite scores.
    """
    yt = np.asarray(y_true, dtype=np.int64).flatten()
    ys = np.asarray(y_score, dtype=np.float64).flatten()
    yp = np.asarray(y_pred, dtype=np.int64).flatten()

    n = yt.size
    if n == 0 or ys.size != n or yp.size != n:
        raise ValueError(
            f"Input arrays must be non-empty and matching lengths. "
            f"Got y_true={yt.shape}, y_score={ys.shape}, y_pred={yp.shape}."
        )

    if not np.all(np.isfinite(ys)):
        raise ValueError("y_score contains non-finite values (NaN or Inf).")

    unique_true = np.unique(yt)
    if not np.all(np.isin(unique_true, [0, 1])):
        raise ValueError(f"y_true must contain only binary values 0 and 1. Found {unique_true}.")

    unique_pred = np.unique(yp)
    if not np.all(np.isin(unique_pred, [0, 1])):
        raise ValueError(f"y_pred must contain only binary values 0 and 1. Found {unique_pred}.")

    anom_count = int(np.sum(yt == 1))
    normal_count = int(np.sum(yt == 0))
    pred_anom_count = int(np.sum(yp == 1))
    prevalence = float(anom_count / n)

    # Confusion matrix counts
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    # Point-wise precision, recall, F1
    if tp + fp > 0:
        precision = float(tp / (tp + fp))
    else:
        precision = 0.0  # Zero predicted positives convention

    if tp + fn > 0:
        recall = float(tp / (tp + fn))
    else:
        recall = 0.0

    if precision + recall > 0.0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    notes = {}

    # Continuous Ranking Metrics: Average Precision & ROC-AUC
    # Undefined when single class is present (no anomalies or all anomalies)
    if anom_count == 0 or normal_count == 0:
        pr_auc = None
        roc_auc = None
        notes["pr_auc"] = "Undefined: single class present in ground-truth labels."
        notes["roc_auc"] = "Undefined: single class present in ground-truth labels."
    else:
        try:
            pr_auc = float(average_precision_score(yt, ys))
        except Exception as exc:
            pr_auc = None
            notes["pr_auc"] = f"Calculation failed: {exc}"

        try:
            roc_auc = float(roc_auc_score(yt, ys))
        except Exception as exc:
            roc_auc = None
            notes["roc_auc"] = f"Calculation failed: {exc}"

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_points": n,
        "anomalous_points": anom_count,
        "anomaly_prevalence": prevalence,
        "predicted_anomalous_points": pred_anom_count,
        "metric_notes": notes,
    }


def calculate_metric_degradation(
    id_metrics: Dict,
    shift_metrics: Dict,
) -> Dict:
    """Calculate performance degradation under distribution shift: Delta M = M_ID - M_shift.

    Formula:
        Delta M = M_ID - M_shift
    Interpretation:
        Delta M > 0: Performance degraded under distribution shift.
        Delta M < 0: Performance improved under distribution shift.
        Delta M == 0: Performance invariant under distribution shift.

    Args:
        id_metrics: Metrics dictionary for In-Distribution (ID) condition.
        shift_metrics: Metrics dictionary for shifted test condition (Shift 1.25 or 1.50).

    Returns:
        Dictionary mapping metric names to Delta values (or None if either is undefined).
    """
    degradations = {}
    tracked_metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall"]

    for m in tracked_metrics:
        id_val = id_metrics.get(m)
        shift_val = shift_metrics.get(m)

        if id_val is not None and shift_val is not None:
            degradations[f"delta_{m}"] = float(id_val - shift_val)
        else:
            degradations[f"delta_{m}"] = None

    return degradations
