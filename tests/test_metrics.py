"""Unit Tests for Anomaly Detection Evaluation Metrics Module.

Tests point-wise Precision, Recall, F1, Average Precision (PR-AUC), ROC-AUC,
confusion matrix counts, null-safety on edge cases, and degradation calculations.
"""

import unittest
import numpy as np

from src.evaluation.metrics import (
    calculate_pointwise_metrics,
    calculate_metric_degradation,
)


class TestMetrics(unittest.TestCase):
    """Test suite for metrics calculation and robustness."""

    def test_perfect_classification(self):
        """Test ideal detector outputs with 1.0 on all metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        y_score = np.array([0.1, 0.2, 0.0, 0.9, 0.85, 0.3, 0.1, 0.95], dtype=np.float64)
        y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 1], dtype=np.int8)

        metrics = calculate_pointwise_metrics(y_true, y_score, y_pred)

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["tn"], 5)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["total_points"], 8)
        self.assertEqual(metrics["anomalous_points"], 3)
        self.assertAlmostEqual(metrics["anomaly_prevalence"], 3 / 8)

    def test_zero_predicted_positives(self):
        """Test zero predicted positives yields precision=0, recall=0 without division by zero."""
        y_true = np.array([0, 0, 1, 1], dtype=np.int8)
        y_score = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        y_pred = np.array([0, 0, 0, 0], dtype=np.int8)

        metrics = calculate_pointwise_metrics(y_true, y_score, y_pred)

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 2)
        self.assertEqual(metrics["tn"], 2)

    def test_single_class_ground_truth(self):
        """Test all-normal series sets ROC-AUC and PR-AUC to None with note."""
        y_true = np.zeros(100, dtype=np.int8)
        y_score = np.random.uniform(0, 1, 100)
        y_pred = (y_score > 0.9).astype(np.int8)

        metrics = calculate_pointwise_metrics(y_true, y_score, y_pred)

        self.assertIsNone(metrics["pr_auc"])
        self.assertIsNone(metrics["roc_auc"])
        self.assertIn("pr_auc", metrics["metric_notes"])
        self.assertEqual(metrics["anomalous_points"], 0)

    def test_input_validation(self):
        """Test shape mismatch and non-finite scores raise ValueError."""
        y_true = np.array([0, 1, 0])
        y_score = np.array([0.1, 0.2])
        y_pred = np.array([0, 1, 0])

        with self.assertRaises(ValueError):
            calculate_pointwise_metrics(y_true, y_score, y_pred)

        # Non-finite scores
        y_score_nan = np.array([0.1, np.nan, 0.5])
        with self.assertRaises(ValueError):
            calculate_pointwise_metrics(y_true, y_score_nan, y_pred)

    def test_metric_degradation(self):
        """Test degradation calculation Delta M = M_ID - M_shift."""
        id_metrics = {
            "pr_auc": 0.85,
            "roc_auc": 0.90,
            "f1": 0.80,
            "precision": 0.80,
            "recall": 0.80,
        }
        shift_metrics = {
            "pr_auc": 0.70,
            "roc_auc": 0.85,
            "f1": 0.60,
            "precision": 0.50,
            "recall": 0.75,
        }

        deg = calculate_metric_degradation(id_metrics, shift_metrics)

        self.assertAlmostEqual(deg["delta_pr_auc"], 0.15)
        self.assertAlmostEqual(deg["delta_roc_auc"], 0.05)
        self.assertAlmostEqual(deg["delta_f1"], 0.20)
        self.assertAlmostEqual(deg["delta_precision"], 0.30)
        self.assertAlmostEqual(deg["delta_recall"], 0.05)


if __name__ == "__main__":
    unittest.main()
