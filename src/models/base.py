"""Base Anomaly Detector Abstract Base Class.

Defines the shared, clean interface contract for all time-series anomaly detectors
(baselines and foundation models) in the project.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np


class BaseAnomalyDetector(ABC):
    """Abstract Base Class for all Time-Series Anomaly Detectors.

    Contract Requirements:
    1. Input shapes: 1D float numpy arrays of sequence length N.
    2. Output shapes: 1D finite float64 numpy arrays of sequence length N.
    3. Score direction: Higher values indicate higher anomaly probability (s(t) -> +infinity).
    4. Zero test label exposure: fit() receives permitted train/val data only.
    """

    def __init__(self, model_id: str, config: Optional[Dict] = None):
        """Initialize the anomaly detector.

        Args:
            model_id: Identifier string for the model (e.g. 'isolation_forest').
            config: Optional configuration dictionary containing model parameters.
        """
        self.model_id = model_id
        self.config = config or {}
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        train_series: np.ndarray,
        val_series: Optional[np.ndarray] = None,
    ) -> "BaseAnomalyDetector":
        """Fit model parameters strictly on permitted training observations.

        Args:
            train_series: 1D float array of normal training observations.
            val_series: Optional 1D float array of normal validation observations.

        Returns:
            Self instance (fitted).
        """
        pass

    @abstractmethod
    def predict_score(self, series: np.ndarray) -> np.ndarray:
        """Compute un-thresholded continuous anomaly scores for a time series.

        Args:
            series: 1D float array of time-series observations of length N.

        Returns:
            1D float64 numpy array of continuous anomaly scores of length N.
            Higher values indicate higher anomaly likelihood.

        Raises:
            RuntimeError: If model is not fitted prior to prediction.
            ValueError: If input series is invalid (empty, non-finite, wrong shape).
        """
        pass

    def validate_input_series(self, series: np.ndarray, name: str = "series") -> np.ndarray:
        """Validate input series array dimensions and finite values.

        Args:
            series: Array to validate.
            name: Description string for error messages.

        Returns:
            Validated 1D float64 numpy array.

        Raises:
            ValueError: If array is empty, non-1D, or contains NaN/Inf values.
        """
        if not isinstance(series, np.ndarray):
            series = np.asarray(series, dtype=np.float64)

        if series.ndim != 1:
            series = series.flatten()

        if series.size == 0:
            raise ValueError(f"Input {name} is empty (0 observations).")

        nan_count = np.isnan(series).sum()
        inf_count = np.isinf(series).sum()
        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"Input {name} contains non-finite values (NaN: {nan_count}, Inf: {inf_count})."
            )

        return series
