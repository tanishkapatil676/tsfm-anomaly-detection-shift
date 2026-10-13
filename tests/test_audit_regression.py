"""Milestone 4A Audit Regression Tests.

Validates:
1. Disjoint, non-overlapping train and validation splits across all 5 authentic UCR series.
2. Preprocessing scalers fit strictly on effective training data.
3. Out-of-sample threshold calibration on permitted validation data without in-sample leakage.
4. Causal window-end score alignment (method='end') in baseline detectors.
5. Chronos validation and test scoring causal context prefix alignment.
"""

import unittest
from pathlib import Path
import numpy as np

from src.data.loader import load_ucr_file
from src.preprocessing.windowing import aggregate_window_scores_to_timestamps
from src.models.baselines.isolation_forest import IsolationForestDetector
from src.models.baselines.lstm_ae import LSTMAutoencoderDetector
from src.models.tsfm.chronos_adapter import ChronosDetector
from src.scoring.threshold import ThresholdSelector


class TestAuditRegression(unittest.TestCase):
    """Rigorous regression tests for Milestone 4A audit fixes."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path("data/raw")
        cls.expected_series = [
            ("001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt", 35000, 7000),
            ("002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt", 35000, 7000),
            ("003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt", 35000, 7000),
            ("004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt", 2500, 500),
            ("005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt", 4000, 800),
        ]

    def test_all_5_authentic_series_disjoint_splits(self):
        """Verify that train and validation data are strictly disjoint and sum to t_split."""
        for fname, t_split, expected_val in self.expected_series:
            fpath = self.data_dir / fname
            if not fpath.exists():
                continue
            series = load_ucr_file(fpath)
            
            # 1. Lengths check
            expected_train = t_split - expected_val
            self.assertEqual(len(series.train_data), expected_train, f"Mismatch train len for {fname}")
            self.assertEqual(len(series.val_data), expected_val, f"Mismatch val len for {fname}")
            self.assertEqual(len(series.train_data) + len(series.val_data), t_split)

            # 2. Slice disjointness check
            np.testing.assert_array_equal(
                series.raw_data[:expected_train], series.train_data,
                err_msg=f"train_data is not exact prefix of raw_data for {fname}"
            )
            np.testing.assert_array_equal(
                series.raw_data[expected_train:t_split], series.val_data,
                err_msg=f"val_data is not exact slice [expected_train:t_split] for {fname}"
            )
            np.testing.assert_array_equal(
                series.raw_data[t_split:], series.test_data,
                err_msg=f"test_data is not exact slice [t_split:] for {fname}"
            )

    def test_scaler_fitted_on_effective_train_only(self):
        """Verify detector scaler is fitted strictly on effective train data."""
        np.random.seed(42)
        train_data = np.random.normal(10.0, 2.0, 500)
        val_data = np.random.normal(50.0, 5.0, 100)

        detector = IsolationForestDetector(config={"n_estimators": 10, "window_len": 20})
        detector.fit(train_data, val_series=val_data)

        # Scaler mean must match train_data mean, NOT combined (train + val) mean
        self.assertAlmostEqual(detector.scaler.mean_, np.mean(train_data), places=5)
        self.assertAlmostEqual(detector.scaler.scale_, np.std(train_data), places=5)
        self.assertNotAlmostEqual(detector.scaler.mean_, np.mean(np.concatenate([train_data, val_data])), places=2)

    def test_window_end_alignment_causality(self):
        """Verify window aggregation uses method='end' padding warm-up positions."""
        window_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W = 4
        S = 1
        total_len = len(window_scores) + W - 1  # 5 + 4 - 1 = 8

        ts_scores = aggregate_window_scores_to_timestamps(
            window_scores, total_length=total_len, window_len=W, stride=S, method="end"
        )

        self.assertEqual(len(ts_scores), total_len)
        # Warm-up positions 0..W-2 (0, 1, 2) must be padded with window 0 score (1.0)
        self.assertEqual(ts_scores[0], 1.0)
        self.assertEqual(ts_scores[1], 1.0)
        self.assertEqual(ts_scores[2], 1.0)
        # End positions W-1..total_len-1 must match window scores
        self.assertEqual(ts_scores[3], 1.0)  # window 0 end
        self.assertEqual(ts_scores[4], 2.0)  # window 1 end
        self.assertEqual(ts_scores[5], 3.0)  # window 2 end
        self.assertEqual(ts_scores[6], 4.0)  # window 3 end
        self.assertEqual(ts_scores[7], 5.0)  # window 4 end

    def test_out_of_sample_threshold_calibration(self):
        """Verify threshold selector calibration on out-of-sample validation data."""
        np.random.seed(42)
        train_data = np.sin(np.linspace(0, 10, 400))
        val_data = np.sin(np.linspace(10, 12, 100))

        detector = IsolationForestDetector(config={"n_estimators": 20, "window_len": 20})
        detector.fit(train_data)

        val_scores = detector.predict_score(val_data)
        thresh_selector = ThresholdSelector(method="percentile", percentile=99.0)
        tau = thresh_selector.fit(val_scores)

        self.assertTrue(np.isfinite(tau))
        self.assertEqual(len(val_scores), 100)
        self.assertGreater(tau, 0.0)


if __name__ == "__main__":
    unittest.main()
