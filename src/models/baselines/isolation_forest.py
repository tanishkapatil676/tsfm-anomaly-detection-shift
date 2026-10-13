"""Isolation Forest Anomaly Detector Baseline.

Implements non-parametric sliding-window Isolation Forest for time-series anomaly detection.
Inverts sklearn path-length scores so higher scores indicate higher anomaly likelihood.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.base import BaseAnomalyDetector
from src.preprocessing.windowing import (
    StandardScalerPreserver,
    extract_sliding_windows,
    extract_summary_features,
    aggregate_window_scores_to_timestamps,
)
from src.scoring.aligner import align_anomaly_scores


class IsolationForestDetector(BaseAnomalyDetector):
    """Sliding-window Isolation Forest Anomaly Detector."""

    def __init__(self, model_id: str = "isolation_forest", config: Optional[Dict] = None):
        """Initialize Isolation Forest detector.

        Args:
            model_id: Identifier string (default 'isolation_forest').
            config: Configuration dictionary with window_len, stride, n_estimators, seed.
        """
        super().__init__(model_id=model_id, config=config)

        self.window_len = self.config.get("window_len", 100)
        self.stride = self.config.get("stride", 1)
        self.n_estimators = self.config.get("n_estimators", 100)
        self.random_state = self.config.get("random_seed", 42)
        self.feature_mode = self.config.get("feature_mode", "summary_stats")  # 'summary_stats' or 'raw_window'

        self.scaler = StandardScalerPreserver()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(
        self,
        train_series: np.ndarray,
        val_series: Optional[np.ndarray] = None,
    ) -> "IsolationForestDetector":
        """Fit scaler and Isolation Forest strictly on permitted training observations.

        Args:
            train_series: 1D float array of normal training observations.
            val_series: Optional 1D float array of validation observations (unused for fitting).

        Returns:
            Self instance (fitted).
        """
        train_arr = self.validate_input_series(train_series, "train_series")

        # Fit training scaler and transform training sequence
        scaled_train = self.scaler.fit_transform(train_arr)

        # Extract sliding windows
        windows = extract_sliding_windows(
            scaled_train, window_len=self.window_len, stride=self.stride
        )

        if self.feature_mode == "summary_stats":
            features = extract_summary_features(windows)
        else:
            features = windows

        # Fit sklearn IsolationForest model
        self.model.fit(features)
        self.is_fitted = True
        return self

    def predict_score(self, series: np.ndarray) -> np.ndarray:
        """Compute timestamp-level continuous anomaly scores for a time series.

        Args:
            series: 1D float array of time-series observations of length N.

        Returns:
            1D float64 numpy array of continuous anomaly scores of length N.
            Higher values indicate higher anomaly probability.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Detector '{self.model_id}' must be fitted before predicting scores.")

        arr = self.validate_input_series(series, "predict_series")
        total_len = arr.size

        # Transform using frozen training scaler
        scaled_series = self.scaler.transform(arr)

        # Extract sliding windows
        windows = extract_sliding_windows(
            scaled_series, window_len=self.window_len, stride=self.stride
        )

        if self.feature_mode == "summary_stats":
            features = extract_summary_features(windows)
        else:
            features = windows

        # sklearn score_samples returns negative average path length
        # Negate so higher value = more anomalous
        raw_scores = -self.model.score_samples(features)

        # Aggregate window-level scores back to timestamps using causal window-end alignment
        timestamp_scores = aggregate_window_scores_to_timestamps(
            raw_scores,
            total_length=total_len,
            window_len=self.window_len,
            stride=self.stride,
            method="end",
        )

        return align_anomaly_scores(timestamp_scores, target_length=total_len)
