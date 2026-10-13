"""Unit Tests for Results Persistence Utility.

Tests JSON serialization, NumPy type conversion, NaN to null mapping,
roundtrip file loading, and summary CSV generation.
"""

import tempfile
from pathlib import Path
import unittest
import numpy as np

from src.utils.persistence import (
    convert_numpy_to_python,
    save_run_record,
    load_run_record,
    save_summary_table,
)


class TestPersistence(unittest.TestCase):
    """Test suite for results persistence."""

    def setUp(self):
        """Create temporary working directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_numpy_conversion_and_null_safety(self):
        """Test conversion of NumPy types and NaN/Inf to valid JSON types."""
        data = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.1415),
            "bool_val": np.bool_(True),
            "nan_val": np.nan,
            "inf_val": np.inf,
            "arr_val": np.array([1, 2, 3]),
        }

        clean = convert_numpy_to_python(data)

        self.assertIsInstance(clean["int_val"], int)
        self.assertIsInstance(clean["float_val"], float)
        self.assertIsInstance(clean["bool_val"], bool)
        self.assertIsNone(clean["nan_val"])
        self.assertIsNone(clean["inf_val"])
        self.assertEqual(clean["arr_val"], [1, 2, 3])

    def test_save_and_load_run_record(self):
        """Test saving run record to JSON and reloading with full fidelity."""
        record = {
            "run_id": "test_run_001",
            "model_id": "isolation_forest",
            "dataset_id": "001",
            "condition": "SHIFT_125",
            "metrics": {
                "pr_auc": np.float64(0.82),
                "roc_auc": None,
                "tp": np.int64(10),
            },
        }

        saved_path = save_run_record(record, self.temp_path)
        self.assertTrue(saved_path.exists())

        loaded = load_run_record(saved_path)
        self.assertEqual(loaded["run_id"], "test_run_001")
        self.assertEqual(loaded["model_id"], "isolation_forest")
        self.assertEqual(loaded["metrics"]["pr_auc"], 0.82)
        self.assertIsNone(loaded["metrics"]["roc_auc"])
        self.assertEqual(loaded["metrics"]["tp"], 10)

    def test_save_summary_table(self):
        """Test saving CSV and JSON summary tables."""
        records = [
            {"run_id": "run_1", "model": "iForest", "pr_auc": 0.85, "roc_auc": 0.90},
            {"run_id": "run_2", "model": "LSTM-AE", "pr_auc": 0.88, "roc_auc": None},
        ]

        csv_path = self.temp_path / "summary.csv"
        json_path = self.temp_path / "summary.json"

        save_summary_table(records, csv_path, json_path)

        self.assertTrue(csv_path.exists())
        self.assertTrue(json_path.exists())

        # Verify CSV has 3 lines (header + 2 rows)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
