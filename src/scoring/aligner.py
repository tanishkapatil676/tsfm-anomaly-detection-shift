"""Anomaly Score Alignment and Validation Utility.

Ensures continuous anomaly score arrays match target sequence lengths,
have correct float64 dtypes, and contain zero non-finite values.
"""

from typing import Optional
import numpy as np


def align_anomaly_scores(
    scores: np.ndarray,
    target_length: int,
    pad_value: Optional[float] = None,
) -> np.ndarray:
    """Validate and align continuous anomaly scores to exact target sequence length N.

    Args:
        scores: 1D continuous anomaly score array.
        target_length: Target sequence length N.
        pad_value: Optional fill value for missing positions (defaults to first score).

    Returns:
        1D float64 numpy array of length target_length.

    Raises:
        ValueError: If scores array is empty or contains non-finite values.
    """
    arr = np.asarray(scores, dtype=np.float64).flatten()

    if arr.size == 0:
        raise ValueError("Cannot align empty score array.")

    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Score array contains non-finite values (NaN: {nan_count}, Inf: {inf_count}).")

    if arr.size == target_length:
        return arr

    if arr.size < target_length:
        pad_fill = pad_value if pad_value is not None else float(arr[0])
        pad_size = target_length - arr.size
        padding = np.full(pad_size, pad_fill, dtype=np.float64)
        return np.concatenate([padding, arr])

    # Truncate if arr.size > target_length
    return arr[:target_length]
