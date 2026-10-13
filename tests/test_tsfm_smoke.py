"""Foundation Model TSFM Bounded Feasibility Smoke Test Suite.

Verifies MOMENT-1-small and Chronos-Bolt-tiny adapters, tests feasibility status,
and validates bounded environment loading behaviors.
"""

import unittest
import numpy as np

from src.models.tsfm.moment_adapter import MomentDetector
from src.models.tsfm.chronos_adapter import ChronosDetector


class TestTSFMSmoke(unittest.TestCase):
    """Feasibility and adapter smoke test suite for MOMENT-1-small and Chronos-Bolt-tiny."""

    def setUp(self):
        """Generate synthetic univariate time series."""
        np.random.seed(42)
        self.train_series = np.sin(np.linspace(0, 40, 1000)) + np.random.normal(0, 0.1, 1000)
        self.test_series = np.sin(np.linspace(40, 80, 1000)) + np.random.normal(0, 0.1, 1000)

    def test_moment_detector_feasibility_check(self):
        """Test MOMENT-1-small bounded feasibility procedure."""
        detector = MomentDetector()
        status, details = detector.check_feasibility()

        self.assertIn(
            status,
            [
                "FEASIBLE",
                "FEASIBLE_WITH_LIMITATIONS",
                "INFEASIBLE_IN_CURRENT_ENVIRONMENT",
                "NOT_YET_VERIFIED",
            ],
        )
        self.assertIn("model_tag", details)
        self.assertEqual(details["model_tag"], "AutonLab/MOMENT-1-small")

        # Verify fit() sets is_fitted
        detector.fit(self.train_series)
        self.assertTrue(detector.is_fitted)

        # If infeasible in current environment, predict_score must raise actionable RuntimeError
        if status == "INFEASIBLE_IN_CURRENT_ENVIRONMENT":
            with self.assertRaises(RuntimeError):
                detector.predict_score(self.test_series)

    def test_chronos_detector_feasibility_check(self):
        """Test Chronos-Bolt-tiny conditional fallback feasibility procedure."""
        detector = ChronosDetector()
        status, details = detector.check_feasibility()

        self.assertIn(
            status,
            [
                "FEASIBLE",
                "FEASIBLE_WITH_LIMITATIONS",
                "INFEASIBLE_IN_CURRENT_ENVIRONMENT",
                "NOT_YET_VERIFIED",
            ],
        )
        self.assertEqual(details["model_tag"], "amazon/chronos-bolt-tiny")

        detector.fit(self.train_series)
        self.assertTrue(detector.is_fitted)

        if status == "INFEASIBLE_IN_CURRENT_ENVIRONMENT":
            with self.assertRaises(RuntimeError):
                detector.predict_score(self.test_series)
        elif status == "FEASIBLE":
            scores = detector.predict_score(self.test_series[:128])
            self.assertEqual(len(scores), 128)
            self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
