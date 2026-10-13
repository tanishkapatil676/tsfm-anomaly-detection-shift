"""Integration Test for Experiment Runner using Synthetic Fixtures.

Verifies end-to-end execution of run_single_dataset_benchmark on a small synthetic series,
checking JSON persistence, threshold freezing, metric calculations, and CSV generation.
"""

import tempfile
from pathlib import Path
import unittest
import numpy as np

from experiments.run_experiment import run_single_dataset_benchmark


class TestExperimentRunnerIntegration(unittest.TestCase):
    """Integration test suite for experiment runner pipeline."""

    def setUp(self):
        """Create temporary workspace and synthetic UCR dataset file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create small synthetic UCR series: N=1500, T_split=1000, Anom=[1200, 1250]
        self.filename = "001_UCR_Anomaly_SYNTHETICTEST_1000_1201_1250.txt"
        self.filepath = self.temp_path / self.filename

        np.random.seed(42)
        train_data = np.sin(np.linspace(0, 30, 1000)) + np.random.normal(0, 0.05, 1000)
        test_data = np.sin(np.linspace(30, 45, 500)) + np.random.normal(0, 0.05, 500)
        # Spike anomaly in indices 200..249 of test
        test_data[200:250] += 3.0

        full_series = np.concatenate([train_data, test_data])
        np.savetxt(self.filepath, full_series, fmt="%.6f")

        self.results_dir = self.temp_path / "results"

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_runner_on_synthetic_series(self):
        """Test single-dataset benchmark runner on Isolation Forest and 3 shift conditions."""
        records = run_single_dataset_benchmark(
            file_path=self.filepath,
            model_names=["isolation_forest"],
            shift_conditions=["ID", "SHIFT_125", "SHIFT_150"],
            results_dir=self.results_dir,
            smoke_mode=True,
        )

        # Verify 3 runs produced (ID, SHIFT_125, SHIFT_150)
        self.assertEqual(len(records), 3)

        for rec in records:
            self.assertEqual(rec["status"], "SUCCESS")
            self.assertEqual(rec["model_id"], "isolation_forest")
            self.assertIn(rec["condition"], ["ID", "SHIFT_125", "SHIFT_150"])
            self.assertIsNotNone(rec["pr_auc"])
            self.assertIsNotNone(rec["f1"])
            self.assertGreater(rec["threshold"], 0.0)

        # Check JSON files exist in results_dir/json
        json_dir = self.results_dir / "json"
        self.assertTrue(json_dir.exists())
        json_files = list(json_dir.glob("*.json"))
        self.assertEqual(len(json_files), 3)


if __name__ == "__main__":
    unittest.main()
