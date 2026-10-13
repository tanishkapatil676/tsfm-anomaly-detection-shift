"""Baseline Anomaly Detector CPU Smoke Test Suite.

Verifies that Isolation Forest and PyTorch LSTM Autoencoder detectors execute cleanly on CPU,
produce finite 1D anomaly score vectors matching sequence length N, and integrate with
leakage-safe validation thresholding.
"""

import unittest
import numpy as np

from src.models.baselines.isolation_forest import IsolationForestDetector
from src.models.baselines.lstm_ae import LSTMAutoencoderDetector
from src.scoring.threshold import ThresholdSelector
from src.scoring.aligner import align_anomaly_scores


class TestBaselineSmoke(unittest.TestCase):
    """Smoke test suite for Isolation Forest and LSTM Autoencoder CPU baselines."""

    def setUp(self):
        """Generate synthetic univariate time series for baseline smoke testing."""
        np.random.seed(42)
        n_train = 1000
        n_test = 1000

        # Normal training signal (sine wave + Gaussian noise)
        self.train_series = np.sin(np.linspace(0, 40, n_train)) + np.random.normal(0, 0.1, n_train)
        self.val_series = self.train_series[-200:]

        # Test signal with injected amplitude spike anomaly in indices 400..450
        self.test_series = np.sin(np.linspace(40, 80, n_test)) + np.random.normal(0, 0.1, n_test)
        self.test_series[400:450] += 4.0

    def test_isolation_forest_detector_smoke(self):
        """Test Isolation Forest fit, score generation, finite outputs, and thresholding."""
        config = {
            "window_len": 50,
            "stride": 1,
            "n_estimators": 50,
            "random_seed": 42,
            "feature_mode": "summary_stats",
        }
        detector = IsolationForestDetector(config=config)

        # Fit strictly on normal training series
        detector.fit(self.train_series)
        self.assertTrue(detector.is_fitted)

        # Generate scores on validation and test series
        val_scores = detector.predict_score(self.val_series)
        test_scores = detector.predict_score(self.test_series)

        # Assert output properties
        self.assertEqual(val_scores.shape, (200,))
        self.assertEqual(test_scores.shape, (1000,))
        self.assertFalse(np.isnan(test_scores).any())
        self.assertFalse(np.isinf(test_scores).any())

        # Assert score direction: anomalous segment (400..450) has higher mean score than normal test
        anom_mean_score = float(np.mean(test_scores[400:450]))
        normal_mean_score = float(np.mean(test_scores[:300]))
        self.assertGreater(anom_mean_score, normal_mean_score)

        # Fit frozen threshold on validation scores ONLY
        thresh_selector = ThresholdSelector(method="percentile", percentile=99.0)
        tau = thresh_selector.fit(val_scores)
        self.assertGreater(tau, 0.0)

        # Binary prediction using frozen threshold
        binary_preds = thresh_selector.predict_binary(test_scores)
        self.assertEqual(binary_preds.shape, (1000,))

    def test_lstm_autoencoder_detector_smoke(self):
        """Test PyTorch LSTM Autoencoder fit, reconstruction score generation, and CPU safety."""
        config = {
            "window_len": 50,
            "stride": 1,
            "hidden_dim": 16,
            "bottleneck_dim": 8,
            "epochs": 2,  # Fast 2 epochs for smoke test execution
            "batch_size": 32,
            "random_seed": 42,
        }
        detector = LSTMAutoencoderDetector(config=config)

        # Fit strictly on normal training series
        detector.fit(self.train_series, self.val_series)
        self.assertTrue(detector.is_fitted)

        # Generate scores on validation and test series
        val_scores = detector.predict_score(self.val_series)
        test_scores = detector.predict_score(self.test_series)

        # Assert output properties
        self.assertEqual(val_scores.shape, (200,))
        self.assertEqual(test_scores.shape, (1000,))
        self.assertFalse(np.isnan(test_scores).any())
        self.assertFalse(np.isinf(test_scores).any())

        # Fit frozen threshold on validation scores
        thresh_selector = ThresholdSelector(method="percentile", percentile=99.0)
        tau = thresh_selector.fit(val_scores)
        self.assertGreater(tau, 0.0)

        binary_preds = thresh_selector.predict_binary(test_scores)
        self.assertEqual(binary_preds.shape, (1000,))


if __name__ == "__main__":
    unittest.main()
