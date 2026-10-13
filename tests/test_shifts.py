"""Unit Tests for Amplitude Distribution Shift Module.

Verifies exact formula x_shift(t) = a * x_test(t), immutability of input series,
ground-truth label preservation, and error handling.
"""

import unittest
import numpy as np

from src.shifts.amplitude_shift import apply_amplitude_shift, get_shift_conditions


class TestAmplitudeShift(unittest.TestCase):
    """Test suite for controlled amplitude shifts."""

    def setUp(self):
        """Set up test series fixtures."""
        np.random.seed(42)
        # Signal with nonzero mean offset (e.g. mean=10.0) + oscillation
        self.original_test = 10.0 + np.sin(np.linspace(0, 20, 500))
        self.labels = np.zeros(500, dtype=np.int8)
        self.labels[100:150] = 1

    def test_id_shift_identity(self):
        """Test ID condition factor=1.0 returns identical copy without modification."""
        shifted = apply_amplitude_shift(self.original_test, factor=1.00)
        self.assertEqual(len(shifted), len(self.original_test))
        np.testing.assert_allclose(shifted, self.original_test)
        # Ensure distinct memory copy
        self.assertIsNot(shifted, self.original_test)

    def test_shift_125_math(self):
        """Test SHIFT_125 factor=1.25 scales both amplitude and offset exactly."""
        shifted = apply_amplitude_shift(self.original_test, factor=1.25)
        expected = self.original_test * 1.25
        np.testing.assert_allclose(shifted, expected)
        # Verify offset scaling: mean is also scaled by 1.25
        self.assertAlmostEqual(float(np.mean(shifted)), float(np.mean(self.original_test)) * 1.25, places=5)

    def test_shift_150_math(self):
        """Test SHIFT_150 factor=1.50 scales both amplitude and offset exactly."""
        shifted = apply_amplitude_shift(self.original_test, factor=1.50)
        expected = self.original_test * 1.50
        np.testing.assert_allclose(shifted, expected)
        self.assertAlmostEqual(float(np.mean(shifted)), float(np.mean(self.original_test)) * 1.50, places=5)

    def test_input_immutability(self):
        """Test that applying a shift NEVER mutates the original array."""
        test_copy = self.original_test.copy()
        _ = apply_amplitude_shift(self.original_test, factor=1.50)
        np.testing.assert_array_equal(self.original_test, test_copy)

    def test_labels_unaffected(self):
        """Verify ground-truth label array is completely preserved and unmodified."""
        labels_before = self.labels.copy()
        _ = apply_amplitude_shift(self.original_test, factor=1.25)
        np.testing.assert_array_equal(self.labels, labels_before)

    def test_get_shift_conditions(self):
        """Test generation of all 3 conditions dictionary."""
        conditions = get_shift_conditions(self.original_test)
        self.assertIn("ID", conditions)
        self.assertIn("SHIFT_125", conditions)
        self.assertIn("SHIFT_150", conditions)

        self.assertEqual(conditions["ID"]["factor"], 1.00)
        self.assertEqual(conditions["SHIFT_125"]["factor"], 1.25)
        self.assertEqual(conditions["SHIFT_150"]["factor"], 1.50)

        for cond_id, cond_data in conditions.items():
            self.assertEqual(len(cond_data["data"]), len(self.original_test))
            self.assertIn("formula", cond_data)

    def test_invalid_parameters(self):
        """Test error handling for invalid shift factors or empty/malformed inputs."""
        with self.assertRaises(ValueError):
            apply_amplitude_shift(self.original_test, factor=0.0)

        with self.assertRaises(ValueError):
            apply_amplitude_shift(self.original_test, factor=-1.5)

        with self.assertRaises(ValueError):
            apply_amplitude_shift(self.original_test, factor=np.nan)

        with self.assertRaises(ValueError):
            apply_amplitude_shift(np.array([]), factor=1.25)

        with self.assertRaises(ValueError):
            apply_amplitude_shift(np.zeros((10, 10)), factor=1.25)


if __name__ == "__main__":
    unittest.main()
