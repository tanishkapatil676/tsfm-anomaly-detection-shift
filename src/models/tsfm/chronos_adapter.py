"""Chronos-Bolt-tiny Conditional Fallback TSFM Adapter.

Implements zero-shot rolling 1-step forecasting error anomaly detection adapter for
amazon/chronos-bolt-tiny, used conditionally if MOMENT-1-small is infeasible.
"""

from typing import Dict, Optional, Tuple
import numpy as np

import torch

from src.models.base import BaseAnomalyDetector
from src.preprocessing.windowing import (
    StandardScalerPreserver,
    extract_sliding_windows,
    aggregate_window_scores_to_timestamps,
)
from src.scoring.aligner import align_anomaly_scores


class ChronosDetector(BaseAnomalyDetector):
    """Conditional Fallback Adapter for amazon/chronos-bolt-tiny Foundation Model."""

    MODEL_TAG = "amazon/chronos-bolt-tiny"

    def __init__(self, model_id: str = "chronos_bolt_tiny", config: Optional[Dict] = None):
        """Initialize Chronos-Bolt adapter.

        Args:
            model_id: Identifier string (default 'chronos_bolt_tiny').
            config: Configuration dictionary.
        """
        super().__init__(model_id=model_id, config=config)

        self.context_len = self.config.get("context_len", 64)
        self.stride = self.config.get("stride", 1)
        self.batch_size = self.config.get("batch_size", 512)
        self.device = torch.device("cpu")

        self.scaler = StandardScalerPreserver()
        self.pipeline = None
        self.model = None
        self.train_tail: Optional[np.ndarray] = None
        self.feasibility_status: str = "NOT_YET_VERIFIED"
        self.feasibility_details: Dict = {}

    def check_feasibility(self) -> Tuple[str, Dict]:
        """Conduct bounded feasibility investigation for Chronos-Bolt-tiny via documented API.

        Uses official chronos BaseChronosPipeline interface.

        Returns:
            Tuple of (feasibility_status_string, details_dict).
        """
        details = {
            "model_tag": self.MODEL_TAG,
            "license": "Apache-2.0",
            "pytorch_version": torch.__version__,
            "device": str(self.device),
        }

        try:
            import chronos
            from chronos import BaseChronosPipeline
            details["chronos_version"] = getattr(chronos, "__version__", "unknown")
        except ImportError as err:
            self.feasibility_status = "INFEASIBLE_IN_CURRENT_ENVIRONMENT"
            details["error"] = f"Missing required chronos-forecasting package: {err}"
            self.feasibility_details = details
            return self.feasibility_status, details

        try:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.MODEL_TAG,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            self.model = self.pipeline.model

            # Test minimal inference on synthetic context
            dummy_ctx = torch.randn(1, 32, dtype=torch.float32)
            with torch.no_grad():
                dummy_forecast = self.pipeline.predict(dummy_ctx, prediction_length=1)

            if not torch.isfinite(dummy_forecast).all():
                raise ValueError("Chronos output contains non-finite values.")

            self.feasibility_status = "FEASIBLE"
            details["status_reason"] = (
                f"BaseChronosPipeline loaded successfully. "
                f"Output shape: {list(dummy_forecast.shape)}, finite: True."
            )
        except Exception as exc:
            self.feasibility_status = "INFEASIBLE_IN_CURRENT_ENVIRONMENT"
            details["error"] = f"Chronos pipeline load/inference failed: {type(exc).__name__}: {str(exc)}"

        self.feasibility_details = details
        return self.feasibility_status, details

    def fit(
        self,
        train_series: np.ndarray,
        val_series: Optional[np.ndarray] = None,
    ) -> "ChronosDetector":
        """Fit scaler strictly on permitted normal training observations and store training context tail.

        Args:
            train_series: 1D float array of normal training observations.
            val_series: Optional 1D float array of validation observations.

        Returns:
            Self instance (fitted).
        """
        train_arr = self.validate_input_series(train_series, "train_series")
        scaled_train = self.scaler.fit_transform(train_arr)
        # Store scaled training tail to provide natural causal context prefix for test sequences
        self.train_tail = scaled_train[-self.context_len:].copy()

        if self.feasibility_status == "NOT_YET_VERIFIED":
            self.check_feasibility()

        self.is_fitted = True
        return self

    def predict_score(
        self,
        series: np.ndarray,
        context_prefix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute timestamp-level 1-step forecasting residual anomaly scores with strict causality.

        For each target timestamp t, the forecasting context contains exclusively the
        prior observations [t - context_len ... t - 1]. The target observation x(t) is NEVER
        present in its own context, and zero future observations influence its score.

        Args:
            series: 1D float array of time-series observations of length N to score.
            context_prefix: Optional 1D float array of prior observations strictly preceding series
                (e.g. un-shifted training tail). If None, uses stored training tail from fit().

        Returns:
            1D float64 numpy array of continuous anomaly scores of length N.
            Higher values indicate larger forecasting absolute error (more anomalous).
        """
        if not self.is_fitted:
            raise RuntimeError(f"Detector '{self.model_id}' must be fitted before predicting scores.")

        if self.feasibility_status == "INFEASIBLE_IN_CURRENT_ENVIRONMENT":
            raise RuntimeError(
                f"Cannot predict scores: {self.MODEL_TAG} is INFEASIBLE_IN_CURRENT_ENVIRONMENT. "
                f"Details: {self.feasibility_details.get('error', 'Package or hardware bound failure.')}"
            )

        arr = self.validate_input_series(series, "predict_series")
        total_len = arr.size

        # Scale input series using frozen training scaler
        scaled_series = self.scaler.transform(arr)

        # Establish strictly prior context prefix
        if context_prefix is not None:
            raw_p = self.validate_input_series(context_prefix, "context_prefix")
            scaled_p = self.scaler.transform(raw_p)
            if len(scaled_p) < self.context_len:
                pad = np.full(self.context_len - len(scaled_p), scaled_p[0], dtype=np.float64)
                prefix = np.concatenate([pad, scaled_p])
            else:
                prefix = scaled_p[-self.context_len:]
        elif self.train_tail is not None:
            if len(self.train_tail) < self.context_len:
                pad = np.full(self.context_len - len(self.train_tail), self.train_tail[0], dtype=np.float64)
                prefix = np.concatenate([pad, self.train_tail])
            else:
                prefix = self.train_tail[-self.context_len:]
        else:
            # Fallback for standalone evaluation: warm-up prefix from initial point
            prefix = np.full(self.context_len, scaled_series[0], dtype=np.float64)

        # Concatenate buffer: prefix (length C) + series (length N)
        buffer = np.concatenate([prefix, scaled_series])

        # Extract causal contexts of length context_len ending strictly at t - 1
        # Window i in buffer spans [i : i + C], predicting buffer[i + C] (which is scaled_series[i])
        from numpy.lib.stride_tricks import sliding_window_view
        contexts = sliding_window_view(buffer, self.context_len)[:total_len]

        preds_list = []
        for i in range(0, total_len, self.batch_size):
            batch_ctx = torch.tensor(contexts[i : i + self.batch_size], dtype=torch.float32)
            with torch.no_grad():
                forecast = self.pipeline.predict(batch_ctx, prediction_length=1)
            median_idx = forecast.shape[1] // 2
            batch_median = forecast[:, median_idx, 0].cpu().numpy().astype(np.float64)
            preds_list.append(batch_median)

        preds = np.concatenate(preds_list, axis=0) if preds_list else np.zeros(0, dtype=np.float64)

        # 1-step forecasting absolute error: s_t = |x(t) - x_hat_median(t)|
        pointwise_error = np.abs(scaled_series - preds)

        return align_anomaly_scores(pointwise_error, target_length=total_len)
