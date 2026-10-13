"""Leakage-Safe Preprocessing and Sliding-Window Extraction Utilities.

Provides scaling parameter preservation, sliding-window construction, summary feature
extraction, and overlapping window-to-timestamp score aggregation.
"""

from typing import Tuple, Union
import numpy as np


class StandardScalerPreserver:
    """Scaler that fits mean and std strictly on training observations and transforms test data.

    Prevents data leakage by freezing training scale parameters (mu_train, sigma_train).
    """

    def __init__(self):
        self.mean_: float = 0.0
        self.scale_: float = 1.0
        self.is_fitted: bool = False

    def fit(self, train_series: np.ndarray) -> "StandardScalerPreserver":
        """Fit mean and std parameters strictly on training observations.

        Args:
            train_series: 1D float array of training observations.

        Returns:
            Self instance.
        """
        arr = np.asarray(train_series, dtype=np.float64).flatten()
        if arr.size == 0:
            raise ValueError("Cannot fit scaler on empty training array.")

        self.mean_ = float(np.mean(arr))
        std_val = float(np.std(arr))
        # Prevent division by zero for constant training signals
        self.scale_ = std_val if std_val > 1e-8 else 1.0
        self.is_fitted = True
        return self

    def transform(self, series: np.ndarray) -> np.ndarray:
        """Transform a series using frozen training mean and std parameters.

        Args:
            series: 1D float array of observations.

        Returns:
            Scaled 1D float64 numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before calling transform().")

        arr = np.asarray(series, dtype=np.float64)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, train_series: np.ndarray) -> np.ndarray:
        """Fit scaler on training array and return transformed array."""
        return self.fit(train_series).transform(train_series)


def extract_sliding_windows(
    series: np.ndarray,
    window_len: int = 100,
    stride: int = 1,
) -> np.ndarray:
    """Extract chronological 2D sliding windows from a 1D time series.

    Args:
        series: 1D float array of length N.
        window_len: Window length W.
        stride: Stride step S.

    Returns:
        2D float64 numpy array of shape [N_win, window_len].

    Raises:
        ValueError: If series length N < window_len or parameters are invalid.
    """
    arr = np.asarray(series, dtype=np.float64).flatten()
    n = arr.size

    if window_len <= 0 or stride <= 0:
        raise ValueError(f"window_len ({window_len}) and stride ({stride}) must be positive integers.")

    if n < window_len:
        raise ValueError(
            f"Series length ({n}) is shorter than window length ({window_len})."
        )

    n_windows = (n - window_len) // stride + 1
    # Use NumPy strided view for fast zero-copy memory extraction
    shape = (n_windows, window_len)
    strides = (arr.strides[0] * stride, arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    return windows.copy()


def extract_summary_features(windows: np.ndarray) -> np.ndarray:
    """Extract 5 statistical summary features per window [mean, std, min, max, median].

    Args:
        windows: 2D numpy array of shape [N_win, window_len].

    Returns:
        2D float64 numpy array of shape [N_win, 5].
    """
    if windows.ndim != 2:
        raise ValueError(f"Expected 2D windows array, got {windows.ndim}D.")

    means = np.mean(windows, axis=1, keepdims=True)
    stds = np.std(windows, axis=1, keepdims=True)
    mins = np.min(windows, axis=1, keepdims=True)
    maxs = np.max(windows, axis=1, keepdims=True)
    medians = np.median(windows, axis=1, keepdims=True)

    return np.hstack([means, stds, mins, maxs, medians])


def aggregate_window_scores_to_timestamps(
    window_scores: np.ndarray,
    total_length: int,
    window_len: int = 100,
    stride: int = 1,
    method: str = "mean",
) -> np.ndarray:
    """Map continuous window-level scores back to timestamp-level score array of length N.

    Args:
        window_scores: 1D array of length N_win.
        total_length: Total original sequence length N.
        window_len: Window length W.
        stride: Stride step S.
        method: Aggregation method ('mean' or 'end').

    Returns:
        1D float64 array of continuous timestamp anomaly scores of length N.
    """
    scores = np.asarray(window_scores, dtype=np.float64).flatten()
    n_win = scores.size
    expected_win = (total_length - window_len) // stride + 1

    if n_win != expected_win:
        raise ValueError(
            f"window_scores size ({n_win}) does not match expected window count ({expected_win}) "
            f"for length {total_length}, window_len {window_len}, stride {stride}."
        )

    timestamp_scores = np.zeros(total_length, dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.float64)

    if method == "end":
        # Map each window score to its end timestamp
        for i in range(n_win):
            end_idx = i * stride + window_len - 1
            timestamp_scores[end_idx] = scores[i]
            counts[end_idx] += 1.0

        # Fill warm-up timestamps (0..window_len-2) with first available window score
        timestamp_scores[0 : window_len - 1] = scores[0]
        return timestamp_scores

    # Default: Overlapping window mean aggregation
    for i in range(n_win):
        start_idx = i * stride
        end_idx = start_idx + window_len
        timestamp_scores[start_idx:end_idx] += scores[i]
        counts[start_idx:end_idx] += 1.0

    # Avoid division by zero for unvisited timestamps (if any)
    valid_mask = counts > 0
    timestamp_scores[valid_mask] /= counts[valid_mask]

    # Warm-up / tail fallback for zero-count positions
    if not np.all(valid_mask):
        first_valid_score = scores[0]
        timestamp_scores[~valid_mask] = first_valid_score

    return timestamp_scores
