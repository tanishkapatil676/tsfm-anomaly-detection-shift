"""Unit tests for dashboard data loading and selection logic.

Validates that all artifacts required by app.py are present, formatted correctly,
and that all dropdown selections resolve to verified benchmark records.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CSV_PATH = PROJECT_ROOT / "results" / "experiment_summary.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_FILES = {
    "001": "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
    "002": "002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt",
    "003": "003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt",
    "004": "004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt",
    "005": "005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt",
}


class TestDashboardLogic(unittest.TestCase):
    """Test suite for app.py backend data loading and presentation logic."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CSV_PATH)

    def test_summary_csv_integrity(self):
        """Verify summary CSV exists, contains exactly 45 rows, and all runs succeeded."""
        self.assertTrue(CSV_PATH.is_file(), f"Missing CSV: {CSV_PATH}")
        self.assertEqual(len(self.df), 45, f"Expected 45 benchmark rows, got {len(self.df)}")
        self.assertTrue((self.df["status"] == "SUCCESS").all(), "Not all benchmark runs succeeded")

    def test_dropdown_coverage(self):
        """Verify every combination of 5 datasets, 3 models, and 3 conditions exists."""
        datasets = ["001", "002", "003", "004", "005"]
        models = ["isolation_forest", "lstm_ae", "chronos_bolt_tiny"]
        conditions = ["ID", "SHIFT_125", "SHIFT_150"]

        df_padded = self.df.copy()
        df_padded["dataset_id_str"] = df_padded["dataset_id"].astype(str).str.zfill(3)

        for d in datasets:
            for m in models:
                for c in conditions:
                    match = df_padded[
                        (df_padded["dataset_id_str"] == d)
                        & (df_padded["model_id"] == m)
                        & (df_padded["condition"] == c)
                    ]
                    self.assertEqual(
                        len(match),
                        1,
                        f"Expected exactly 1 record for dataset={d}, model={m}, condition={c}",
                    )

    def test_publication_plots_exist(self):
        """Verify required publication figures exist in results/plots/."""
        expected_plots = [
            "metric_comparison_ap.png",
            "metric_comparison_f1.png",
            "degradation_summary_ap.png",
        ]
        for p_name in expected_plots:
            p_path = PLOTS_DIR / p_name
            self.assertTrue(p_path.is_file(), f"Missing plot figure: {p_path}")
            self.assertGreater(p_path.stat().st_size, 1000, f"Plot figure {p_name} is empty")

    def test_dataset_raw_series_loading(self):
        """Verify all 5 authentic UCR raw series load and have valid anomaly spans."""
        for d_id, fname in DATASET_FILES.items():
            fpath = DATA_DIR / fname
            self.assertTrue(fpath.is_file(), f"Missing UCR raw file: {fpath}")

            parts = fname[:-4].split("_")
            t_split = int(parts[-3])
            ano_start = int(parts[-2])
            ano_end = int(parts[-1])

            data = np.loadtxt(fpath, dtype=np.float64)
            if data.ndim > 1:
                data = data.flatten()

            test_data = data[t_split:]
            self.assertGreater(len(test_data), 0)
            rel_start = ano_start - t_split
            rel_end = ano_end - t_split

            self.assertGreaterEqual(rel_start, 0)
            self.assertLess(rel_start, len(test_data))
            self.assertGreaterEqual(rel_end, rel_start)

    def test_extreme_imbalance_dataset_005_metrics(self):
        """Verify verified empirical metrics for Dataset 005 (2 anomalies)."""
        df_padded = self.df.copy()
        df_padded["dataset_id_str"] = df_padded["dataset_id"].astype(str).str.zfill(3)

        # Chronos on 005 ID
        row = df_padded[
            (df_padded["dataset_id_str"] == "005")
            & (df_padded["model_id"] == "chronos_bolt_tiny")
            & (df_padded["condition"] == "ID")
        ].iloc[0]

        self.assertEqual(row["tp"], 2)
        self.assertEqual(row["fn"], 0)
        self.assertEqual(row["pr_auc"], 1.0)
        self.assertEqual(row["recall"], 1.0)
        self.assertAlmostEqual(row["precision"], 0.0250, places=3)
        self.assertEqual(row["fp"], 78)


if __name__ == "__main__":
    unittest.main()
