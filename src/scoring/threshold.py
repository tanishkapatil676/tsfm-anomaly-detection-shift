"""Threshold Selection and Freezing Utility.

Derives decision thresholds tau strictly from permitted normal validation split scores
and converts continuous anomaly scores into binary predictions.
"""

from typing import Union
import numpy as np


class ThresholdSelector:
    """Utility for fitting, freezing, and applying anomaly score decision thresholds.

    Enforces zero test-label leakage: thresholds MUST be computed on validation scores only.
    """

    def __init__(self, method: str = "percentile", percentile: float = 99.0, k_std: float = 3.0):
        """Initialize threshold selector settings.

        Args:
            method: Thresholding method ('percentile' or 'mean_std').
            percentile: Percentile value for 'percentile' method (default 99.0).
            k_std: Standard deviation multiplier for 'mean_std' method (default 3.0).
        """
        self.method = method
        self.percentile = percentile
        self.k_std = k_std
        self.threshold_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, val_scores: np.ndarray) -> float:
        """Derive decision threshold tau strictly from validation split scores.

        Args:
            val_scores: 1D float array of validation scores s(t in D_val).

        Returns:
            Calculated threshold value tau (float).

        Raises:
            ValueError: If validation scores are empty or contain NaNs/Infs.
        """
        arr = np.asarray(val_scores, dtype=np.float64).flatten()

        if arr.size == 0:
            raise ValueError("Cannot fit threshold on empty validation scores array.")

        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"Validation scores contain non-finite values (NaN: {nan_count}, Inf: {inf_count})."
            )

        if self.method == "percentile":
            self.threshold_ = float(np.percentile(arr, self.percentile))
        elif self.method == "mean_std":
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            self.threshold_ = mean_val + self.k_std * std_val
        else:
            raise ValueError(f"Unknown thresholding method '{self.method}'. Expected 'percentile' or 'mean_std'.")

        self.is_fitted = True
        return self.threshold_

    def predict_binary(self, scores: np.ndarray, threshold: float = None) -> np.ndarray:
        """Convert continuous anomaly scores into binary predictions using frozen threshold tau.

        Args:
            scores: 1D float array of continuous anomaly scores.
            threshold: Optional override threshold (uses self.threshold_ if None).

        Returns:
            1D int8 numpy array containing binary predictions (1 for anomaly, 0 for normal).
        """
        tau = threshold if threshold is not None else self.threshold_
        if threshold is None and not self.is_fitted:
            raise RuntimeError("ThresholdSelector must be fitted before calling predict_binary().")

        arr = np.asarray(scores, dtype=np.float64)
        return (arr >= tau).astype(np.int8)
