"""Automated Unit Tests for UCR Data Loader and Profiler.

All tests use synthetic memory fixtures and temporary files.
Executable via: `python -m unittest discover tests` or `pytest tests/`.
"""

import json
import hashlib
import tempfile
from pathlib import Path
import unittest
import numpy as np

from src.data.loader import (
    parse_ucr_filename,
    validate_ucr_series,
    load_ucr_file,
    discover_candidate_files,
    UCRSeries,
)
from src.data.profiler import (
    profile_ucr_series,
    save_profile_report,
    plot_ucr_series,
)


class TestUCRDataLoader(unittest.TestCase):
    """Test suite covering UCR data loader, validator, and boundary math."""

    def setUp(self):
        """Create temporary working directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.default_config = {
            "min_train_len": 1000,
            "min_val_len": 200,
            "window_len": 100,
            "val_split_fraction": 0.20,
        }

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _create_synthetic_ucr_file(
        self,
        filename: str,
        series_length: int = 15000,
        t_split: int = 10000,
        anomaly_start: int = 12000,
        anomaly_end: int = 12500,
        inject_nan: bool = False,
        inject_inf: bool = False,
        constant_train: bool = False,
        non_numeric: bool = False,
    ) -> Path:
        """Helper creating synthetic UCR text file on disk."""
        filepath = self.temp_path / filename

        if non_numeric:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("1.23\n2.45\nINVALID_FLOAT_STRING\n4.56\n")
            return filepath

        np.random.seed(42)
        if constant_train:
            train_part = np.ones(t_split, dtype=np.float64) * 5.0
        else:
            train_part = np.sin(np.linspace(0, 50, t_split)) + np.random.normal(0, 0.1, t_split)

        test_len = series_length - t_split
        test_part = np.sin(np.linspace(50, 100, test_len)) + np.random.normal(0, 0.1, test_len)

        raw_data = np.concatenate([train_part, test_part])

        if inject_nan:
            raw_data[500] = np.nan
        if inject_inf:
            raw_data[600] = np.inf

        np.savetxt(filepath, raw_data, fmt="%.6f")
        return filepath

    def test_parse_ucr_filename_valid(self):
        """Test parsing valid UCR filename syntax."""
        filename = "001_UCR_Anomaly_DISTORTED1k_10000_17700_17900.txt"
        meta = parse_ucr_filename(filename)

        self.assertEqual(meta["dataset_id"], "001")
        self.assertEqual(meta["dataset_name"], "DISTORTED1k")
        self.assertEqual(meta["series_id"], "001_UCR_Anomaly_DISTORTED1k")
        self.assertEqual(meta["t_split"], 10000)
        self.assertEqual(meta["source_anomaly_start"], 17700)
        self.assertEqual(meta["source_anomaly_end"], 17900)

    def test_parse_ucr_filename_invalid(self):
        """Test parsing malformed filename raises ValueError."""
        invalid_filenames = [
            "random_filename.txt",
            "001_UCR_Anomaly_missing_splits.txt",
            "001_UCR_Anomaly_name_1000_2000.txt",
        ]
        for fname in invalid_filenames:
            with self.subTest(fname=fname):
                with self.assertRaises(ValueError):
                    parse_ucr_filename(fname)

    def test_load_ucr_file_valid(self):
        """Test loading valid numeric series and checking array splits."""
        fname = "001_UCR_Anomaly_DISTORTED1k_10000_12000_12500.txt"
        fpath = self._create_synthetic_ucr_file(fname, 15000, 10000, 12000, 12500)

        series = load_ucr_file(fpath, self.default_config)

        self.assertTrue(series.validation_result.is_eligible)
        self.assertEqual(series.total_length, 15000)
        self.assertEqual(len(series.train_data), 8000)
        self.assertEqual(len(series.val_data), 2000)
        self.assertEqual(len(series.train_data) + len(series.val_data), 10000)
        self.assertEqual(len(series.test_data), 5000)
        self.assertEqual(len(series.test_labels), 5000)
        self.assertEqual(series.validation_result.rejection_reasons, [])

    def test_load_ucr_file_empty_rejection(self):
        """Test loading empty file is rejected properly."""
        fname = "002_UCR_Anomaly_EMPTY_10000_12000_12500.txt"
        fpath = self.temp_path / fname
        fpath.touch()

        with self.assertRaises(ValueError):
            load_ucr_file(fpath, self.default_config)

    def test_load_ucr_file_non_numeric_rejection(self):
        """Test loading non-numeric content raises ValueError."""
        fname = "003_UCR_Anomaly_NONNUMERIC_10000_12000_12500.txt"
        fpath = self._create_synthetic_ucr_file(fname, non_numeric=True)

        with self.assertRaises(ValueError):
            load_ucr_file(fpath, self.default_config)

    def test_invalid_training_prefix_short(self):
        """Test series with training prefix below min_train_len is rejected."""
        fname = "004_UCR_Anomaly_SHORTTRAIN_500_1200_1300.txt"
        fpath = self._create_synthetic_ucr_file(fname, 2000, 500, 1200, 1300)

        series = load_ucr_file(fpath, self.default_config)

        self.assertFalse(series.validation_result.is_eligible)
        self.assertTrue(
            any("below minimum requirement" in r for r in series.validation_result.rejection_reasons)
        )

    def test_empty_test_segment(self):
        """Test series where t_split equals total length is rejected."""
        fname = "005_UCR_Anomaly_EMPTYTEST_10000_10001_10050.txt"
        fpath = self._create_synthetic_ucr_file(fname, 10000, 10000, 10001, 10050)

        series = load_ucr_file(fpath, self.default_config)

        self.assertFalse(series.validation_result.is_eligible)
        self.assertTrue(
            any("exceeds total sequence length" in r for r in series.validation_result.rejection_reasons)
        )

    def test_0based_anomaly_boundary_conversion(self):
        """Test exact 0-based test-relative index conversion math.

        Source 1-based: START=12001, END=12500, T_split=10000
        Expected test 0-based:
            test_start = 12001 - 1 - 10000 = 200
            test_end_inclusive = 12500 - 1 - 10000 = 499
            test_end_exclusive = 500
        """
        fname = "006_UCR_Anomaly_MATHCHECK_10000_12001_12500.txt"
        fpath = self._create_synthetic_ucr_file(fname, 15000, 10000, 12001, 12500)

        series = load_ucr_file(fpath, self.default_config)

        self.assertEqual(series.test_anomaly_start, 2000)
        self.assertEqual(series.test_anomaly_end_inclusive, 2499)
        self.assertEqual(series.test_anomaly_end_exclusive, 2500)
        self.assertEqual(int(series.test_labels.sum()), 500)
        self.assertEqual(series.test_labels[2000], 1)
        self.assertEqual(series.test_labels[2499], 1)
        self.assertEqual(series.test_labels[1999], 0)
        self.assertEqual(series.test_labels[2500], 0)

    def test_anomaly_at_first_test_obs(self):
        """Test anomaly starting at the very first test timestamp (START = T_split + 1)."""
        fname = "007_UCR_Anomaly_FIRSTOBS_10000_10001_10100.txt"
        fpath = self._create_synthetic_ucr_file(fname, 15000, 10000, 10001, 10100)

        series = load_ucr_file(fpath, self.default_config)

        self.assertTrue(series.validation_result.is_eligible)
        self.assertEqual(series.test_anomaly_start, 0)
        self.assertEqual(series.test_labels[0], 1)

    def test_anomaly_at_final_test_obs(self):
        """Test anomaly ending at final test timestamp (END = N)."""
        fname = "008_UCR_Anomaly_LASTOBS_10000_14901_15000.txt"
        fpath = self._create_synthetic_ucr_file(fname, 15000, 10000, 14901, 15000)

        series = load_ucr_file(fpath, self.default_config)

        self.assertTrue(series.validation_result.is_eligible)
        self.assertEqual(series.test_anomaly_end_inclusive, 4999)
        self.assertEqual(series.test_labels[4999], 1)

    def test_anomaly_extending_outside_segment(self):
        """Test anomaly range extending past file length (END > N)."""
        fname = "009_UCR_Anomaly_OVERFLOW_10000_14901_16000.txt"
        fpath = self._create_synthetic_ucr_file(fname, 15000, 10000, 14901, 16000)

        series = load_ucr_file(fpath, self.default_config)

        self.assertFalse(series.validation_result.is_eligible)
        self.assertTrue(
            any("outside sequence range" in r for r in series.validation_result.rejection_reasons)
        )

    def test_anomaly_in_training_prefix(self):
        """Test anomaly starting in training prefix (START <= T_split) is rejected."""
        fname = "010_UCR_Anomaly_TRAINANOMALY_10000_5000_6000.txt"
        fpath = self._create_synthetic_ucr_file(fname, 15000, 10000, 5000, 6000)

        series = load_ucr_file(fpath, self.default_config)

        self.assertFalse(series.validation_result.is_eligible)
        self.assertTrue(
            any("falls inside training prefix" in r for r in series.validation_result.rejection_reasons)
        )

    def test_missing_or_non_finite_values(self):
        """Test NaNs and Infs cause eligibility rejection."""
        fname_nan = "011_UCR_Anomaly_HASNAN_10000_12000_12500.txt"
        fpath_nan = self._create_synthetic_ucr_file(fname_nan, 15000, 10000, 12000, 12500, inject_nan=True)

        series_nan = load_ucr_file(fpath_nan, self.default_config)
        self.assertFalse(series_nan.validation_result.is_eligible)

        fname_inf = "012_UCR_Anomaly_HASINF_10000_12000_12500.txt"
        fpath_inf = self._create_synthetic_ucr_file(fname_inf, 15000, 10000, 12000, 12500, inject_inf=True)

        series_inf = load_ucr_file(fpath_inf, self.default_config)
        self.assertFalse(series_inf.validation_result.is_eligible)

    def test_discover_candidate_files(self):
        """Test candidate file discovery matching UCR pattern."""
        self._create_synthetic_ucr_file("001_UCR_Anomaly_TESTA_10000_12000_12500.txt")
        self._create_synthetic_ucr_file("002_UCR_Anomaly_TESTB_10000_12000_12500.txt")
        # Non-matching file
        (self.temp_path / "other_file.txt").touch()

        candidates = discover_candidate_files(self.temp_path)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].name, "001_UCR_Anomaly_TESTA_10000_12000_12500.txt")
        self.assertEqual(candidates[1].name, "002_UCR_Anomaly_TESTB_10000_12000_12500.txt")

    def test_full_profiler_integration(self):
        """Test end-to-end load, profile report generation, JSON save, and plot generation."""
        fname = "013_UCR_Anomaly_INTEGRATION_10000_12000_12500.txt"
        fpath = self._create_synthetic_ucr_file(fname, 15000, 10000, 12000, 12500)

        series = load_ucr_file(fpath, self.default_config)
        profile = profile_ucr_series(series)

        self.assertEqual(profile["series_id"], "013_UCR_Anomaly_INTEGRATION")
        self.assertTrue(profile["data_quality_evaluation"]["is_eligible"])

        json_out = self.temp_path / "profile_013.json"
        saved_json = save_profile_report(profile, json_out)
        self.assertTrue(saved_json.exists())

        with open(saved_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["series_id"], "013_UCR_Anomaly_INTEGRATION")

        img_out = self.temp_path / "plot_013.png"
        saved_img = plot_ucr_series(series, img_out)
        self.assertTrue(saved_img.exists())

    def test_authentic_ucr_files_validation(self):
        """Validate all 5 authentic UCR source files against verified cryptographic hashes and boundary math."""
        raw_dir = Path("data/raw")
        expected_meta = [
            {
                "fname": "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
                "sha256": "e064e98e0b41d117f666945459671d3eec014ff9ac2457401befd8a244b9b93f",
                "total_len": 79795,
                "t_split": 35000,
                "test_len": 44795,
                "anom_start": 16999,
                "anom_end_inc": 17619,
                "anom_count": 621,
            },
            {
                "fname": "002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt",
                "sha256": "1c730e485f186aa39859ac0fca707a7977860b8db49112489270990b7f240e84",
                "total_len": 80001,
                "t_split": 35000,
                "test_len": 45001,
                "anom_start": 21599,
                "anom_end_inc": 21899,
                "anom_count": 301,
            },
            {
                "fname": "003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt",
                "sha256": "557cbf5463ea3fc9cc1d17be60157c737de7a01810bc4907ab252e746eb6a7ae",
                "total_len": 80000,
                "t_split": 35000,
                "test_len": 45000,
                "anom_start": 11599,
                "anom_end_inc": 11899,
                "anom_count": 301,
            },
            {
                "fname": "004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt",
                "sha256": "f76d4cbc56328179183f3ce69a92f5a22389135a786d8f0bb1662bcfc748ede6",
                "total_len": 11000,
                "t_split": 2500,
                "test_len": 8500,
                "anom_start": 2899,
                "anom_end_inc": 3099,
                "anom_count": 201,
            },
            {
                "fname": "005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt",
                "sha256": "076a847fcefaf023942db0324186aa976407c43a8d699beaee8b77527bb5ceae",
                "total_len": 8184,
                "t_split": 4000,
                "test_len": 4184,
                "anom_start": 1390,
                "anom_end_inc": 1391,
                "anom_count": 2,
            },
        ]
        for spec in expected_meta:
            fpath = raw_dir / spec["fname"]
            self.assertTrue(fpath.exists(), f"Authentic source file missing: {spec['fname']}")
            
            # Cryptographic SHA256 verification
            file_bytes = fpath.read_bytes()
            computed_sha = hashlib.sha256(file_bytes).hexdigest()
            self.assertEqual(computed_sha, spec["sha256"], f"SHA256 mismatch for {spec['fname']}")

            # Load and structural verification
            series = load_ucr_file(fpath, self.default_config)
            self.assertTrue(series.validation_result.is_eligible, f"Series {spec['fname']} failed validation")
            self.assertEqual(series.total_length, spec["total_len"])
            
            # Verify disjoint train and validation split partitioning
            expected_val_len = int(0.20 * spec["t_split"])
            expected_effective_train_len = spec["t_split"] - expected_val_len
            self.assertEqual(len(series.train_data), expected_effective_train_len)
            self.assertEqual(len(series.val_data), expected_val_len)
            self.assertEqual(len(series.train_data) + len(series.val_data), spec["t_split"])
            
            # Verify exact contiguous and non-overlapping slices
            np.testing.assert_array_equal(series.raw_data[:expected_effective_train_len], series.train_data)
            np.testing.assert_array_equal(
                series.raw_data[expected_effective_train_len:spec["t_split"]], series.val_data
            )
            np.testing.assert_array_equal(series.raw_data[spec["t_split"]:], series.test_data)

            self.assertEqual(len(series.test_data), spec["test_len"])
            self.assertEqual(len(series.test_labels), spec["test_len"])
            self.assertEqual(series.test_anomaly_start, spec["anom_start"])
            self.assertEqual(series.test_anomaly_end_inclusive, spec["anom_end_inc"])
            self.assertEqual(int(series.test_labels.sum()), spec["anom_count"])

            # Verify no off-by-one errors at boundaries
            if spec["anom_start"] > 0:
                self.assertEqual(series.test_labels[spec["anom_start"] - 1], 0)
            if spec["anom_end_inc"] < spec["test_len"] - 1:
                self.assertEqual(series.test_labels[spec["anom_end_inc"] + 1], 0)
            self.assertEqual(series.test_labels[spec["anom_start"]], 1)
            self.assertEqual(series.test_labels[spec["anom_end_inc"]], 1)


if __name__ == "__main__":
    unittest.main()
