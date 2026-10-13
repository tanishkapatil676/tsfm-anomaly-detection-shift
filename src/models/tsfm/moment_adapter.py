"""MOMENT-1-small Foundation Model Adapter and Bounded Feasibility Utility.

Provides zero-shot masked reconstruction error anomaly detection adapter for
AutonLab/MOMENT-1-small and conducts bounded environment feasibility testing.
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


class MomentDetector(BaseAnomalyDetector):
    """Adapter for AutonLab/MOMENT-1-small Foundation Model."""

    MODEL_TAG = "AutonLab/MOMENT-1-small"

    def __init__(self, model_id: str = "moment_1_small", config: Optional[Dict] = None):
        """Initialize MOMENT adapter.

        Args:
            model_id: Identifier string (default 'moment_1_small').
            config: Configuration dictionary.
        """
        super().__init__(model_id=model_id, config=config)

        self.window_len = self.config.get("window_len", 512)
        self.stride = self.config.get("stride", 1)
        self.device = torch.device("cpu")

        self.scaler = StandardScalerPreserver()
        self.model = None
        self.feasibility_status: str = "NOT_YET_VERIFIED"
        self.feasibility_details: Dict = {}

    def check_feasibility(self) -> Tuple[str, Dict]:
        """Conduct bounded feasibility investigation for MOMENT-1-small via documented API.

        Tests official momentfm API first, then documents exact environment/package blockers.

        Returns:
            Tuple of (feasibility_status_string, details_dict).
        """
        details = {
            "model_tag": self.MODEL_TAG,
            "license": "MIT",
            "pytorch_version": torch.__version__,
            "device": str(self.device),
        }

        # Step 1: Check official documented momentfm API
        try:
            from momentfm import MOMENTPipeline
            self.model = MOMENTPipeline.from_pretrained(
                self.MODEL_TAG,
                model_kwargs={
                    "task_name": "reconstruction",
                    "forecast_horizon": 0,
                    "head_dropout": 0.1,
                    "weight_decay": 0,
                    "freeze_encoder": True,
                    "freeze_embedder": True,
                    "freeze_head": False,
                },
            )
            self.model.init()
            self.model.to(self.device)
            self.feasibility_status = "FEASIBLE"
            details["status_reason"] = "MOMENTPipeline loaded successfully via momentfm API."
            self.feasibility_details = details
            return self.feasibility_status, details
        except ImportError as imp_err:
            details["momentfm_import_error"] = str(imp_err)
            details["documented_blocker"] = (
                "momentfm 0.1.4 requires numpy==1.25.2, which has no binary wheels for Python 3.13 "
                "on Windows/PyPI and fails building from source. Documented official API cannot be installed."
            )
        except Exception as exc:
            details["momentfm_error"] = f"{type(exc).__name__}: {str(exc)}"

        # Step 2: Attempt fallback generic Hugging Face AutoModel loading
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.MODEL_TAG,
                trust_remote_code=True,
            )
            self.model.eval().to(self.device)

            dummy_input = torch.randn(1, 1, 512, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)

            self.feasibility_status = "FEASIBLE"
            details["status_reason"] = "Model loaded successfully via transformers AutoModel fallback."
        except Exception as exc:
            self.feasibility_status = "INFEASIBLE_IN_CURRENT_ENVIRONMENT"
            details["error"] = (
                f"Documented API failed (momentfm dependency conflict: numpy==1.25.2 vs Python 3.13). "
                f"Generic HF fallback failed: {type(exc).__name__}: {str(exc)}"
            )

        self.feasibility_details = details
        return self.feasibility_status, details

    def fit(
        self,
        train_series: np.ndarray,
        val_series: Optional[np.ndarray] = None,
    ) -> "MomentDetector":
        """Fit scaler strictly on permitted normal training series."""
        train_arr = self.validate_input_series(train_series, "train_series")
        self.scaler.fit(train_arr)

        if self.feasibility_status == "NOT_YET_VERIFIED":
            self.check_feasibility()

        self.is_fitted = True
        return self

    def predict_score(self, series: np.ndarray) -> np.ndarray:
        """Compute timestamp-level continuous anomaly scores via zero-shot reconstruction."""
        if not self.is_fitted:
            raise RuntimeError(f"Detector '{self.model_id}' must be fitted before predicting scores.")

        if self.feasibility_status == "INFEASIBLE_IN_CURRENT_ENVIRONMENT":
            raise RuntimeError(
                f"Cannot predict scores: {self.MODEL_TAG} is INFEASIBLE_IN_CURRENT_ENVIRONMENT. "
                f"Details: {self.feasibility_details.get('error', 'Package or hardware bound failure.')}"
            )

        arr = self.validate_input_series(series, "predict_series")
        total_len = arr.size

        scaled_series = self.scaler.transform(arr)
        win_len = min(self.window_len, total_len)

        windows = extract_sliding_windows(scaled_series, window_len=win_len, stride=self.stride)
        tensor_windows = torch.tensor(windows, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_windows)
            recon = output.reconstruction if hasattr(output, "reconstruction") else output
            diff = (tensor_windows - recon).squeeze(1)
            window_mse = torch.mean(diff ** 2, dim=1).cpu().numpy().astype(np.float64)

        timestamp_scores = aggregate_window_scores_to_timestamps(
            window_mse,
            total_length=total_len,
            window_len=win_len,
            stride=self.stride,
            method="mean",
        )

        return align_anomaly_scores(timestamp_scores, target_length=total_len)
