"""Controlled Amplitude Distribution Shift Module.

Implements the experimental protocol's exact mathematical distribution shift:
    x_shift(t) = a * x_test(t)
for scale factors a in {1.00 (ID), 1.25 (Shift 1), 1.50 (Shift 2)}.

Note on Mathematical Definition:
This transformation scales both the signal amplitude and any nonzero DC offset
(mean level) present in the unscaled test observations. It is not a centered variance
scaling. The ground-truth anomaly timestamps and labels remain 100% unchanged.
"""

from typing import Dict, Tuple
import numpy as np


def apply_amplitude_shift(
    test_series: np.ndarray,
    factor: float = 1.0,
) -> np.ndarray:
    """Apply controlled multiplicative amplitude shift to held-out test series.

    Formula:
        x_shift(t) = factor * x_test(t)

    Guarantees:
    - Creates a new float64 array copy without mutating the input array.
    - Preserves sequence length and timestamp alignment.
    - Ground-truth labels must NOT be passed to or altered by this function.

    Args:
        test_series: 1D float array of held-out test observations.
        factor: Multiplicative scale factor 'a' (e.g. 1.00, 1.25, 1.50). Must be positive finite.

    Returns:
        1D float64 numpy array containing shifted test series.

    Raises:
        ValueError: If test_series is empty, non-1D, or factor is non-positive / non-finite.
    """
    arr = np.asarray(test_series, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"test_series must be non-empty 1D array. Got shape {arr.shape}.")

    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"Shift factor must be a finite positive number. Got {factor}.")

    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Input series contains non-finite values (NaN: {nan_count}, Inf: {inf_count}).")

    if factor == 1.0:
        return arr.copy()

    return arr * factor


def get_shift_conditions(
    test_series: np.ndarray,
) -> Dict[str, Dict]:
    """Generate the three benchmark test conditions (ID, SHIFT_125, SHIFT_150).

    Args:
        test_series: Original 1D float array of test observations.

    Returns:
        Dictionary mapping condition IDs to a dictionary containing:
        - 'condition_id': str ('ID', 'SHIFT_125', 'SHIFT_150')
        - 'factor': float (1.00, 1.25, 1.50)
        - 'formula': str ('x_shift(t) = a * x_test(t)')
        - 'data': np.ndarray (shifted series copy)
    """
    conditions = {
        "ID": {
            "condition_id": "ID",
            "factor": 1.00,
            "formula": "x_shift(t) = 1.00 * x_test(t)",
            "data": apply_amplitude_shift(test_series, factor=1.00),
        },
        "SHIFT_125": {
            "condition_id": "SHIFT_125",
            "factor": 1.25,
            "formula": "x_shift(t) = 1.25 * x_test(t)",
            "data": apply_amplitude_shift(test_series, factor=1.25),
        },
        "SHIFT_150": {
            "condition_id": "SHIFT_150",
            "factor": 1.50,
            "formula": "x_shift(t) = 1.50 * x_test(t)",
            "data": apply_amplitude_shift(test_series, factor=1.50),
        },
    }
    return conditions
