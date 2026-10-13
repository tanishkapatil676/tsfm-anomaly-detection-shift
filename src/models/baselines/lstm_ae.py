"""LSTM Autoencoder Anomaly Detector Baseline.

Implements a sequence-to-sequence PyTorch LSTM Autoencoder trained on normal time-series windows.
Anomaly scores are derived from mean squared reconstruction errors.
"""

from typing import Dict, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseAnomalyDetector
from src.preprocessing.windowing import (
    StandardScalerPreserver,
    extract_sliding_windows,
    aggregate_window_scores_to_timestamps,
)
from src.scoring.aligner import align_anomaly_scores


class PyTorchLSTMAutoencoder(nn.Module):
    """PyTorch LSTM Autoencoder Module for sliding window sequence reconstruction."""

    def __init__(
        self,
        window_len: int = 100,
        hidden_dim: int = 32,
        bottleneck_dim: int = 16,
        num_layers: int = 2,
    ):
        """Initialize LSTM Autoencoder layers.

        Args:
            window_len: Sequence window length W.
            hidden_dim: LSTM hidden state dimension H.
            bottleneck_dim: Latent representation dimension.
            num_layers: Number of stacked LSTM layers.
        """
        super().__init__()
        self.window_len = window_len
        self.hidden_dim = hidden_dim

        # Encoder: 1D sequence input (batch, window_len, 1) -> (batch, window_len, hidden_dim)
        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.encoder_proj = nn.Linear(hidden_dim, bottleneck_dim)

        # Decoder: Latent bottleneck -> (batch, window_len, hidden_dim) -> (batch, window_len, 1)
        self.decoder_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass reconstructing window sequence.

        Args:
            x: Tensor of shape (batch, window_len, 1).

        Returns:
            Reconstructed tensor of shape (batch, window_len, 1).
        """
        # Encoder
        enc_out, (hn, cn) = self.encoder_lstm(x)  # enc_out: (batch, window_len, hidden_dim)
        last_hidden = enc_out[:, -1, :]           # (batch, hidden_dim)
        latent = self.encoder_proj(last_hidden)  # (batch, bottleneck_dim)

        # Decoder
        dec_in_latent = self.decoder_proj(latent) # (batch, hidden_dim)
        dec_seq = dec_in_latent.unsqueeze(1).repeat(1, self.window_len, 1) # (batch, window_len, hidden_dim)
        dec_out, _ = self.decoder_lstm(dec_seq)
        reconstruction = self.output_proj(dec_out) # (batch, window_len, 1)

        return reconstruction


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """LSTM Autoencoder Anomaly Detector adhering to BaseAnomalyDetector contract."""

    def __init__(self, model_id: str = "lstm_ae", config: Optional[Dict] = None):
        """Initialize LSTM Autoencoder detector.

        Args:
            model_id: Identifier string (default 'lstm_ae').
            config: Configuration dictionary containing hyperparameters.
        """
        super().__init__(model_id=model_id, config=config)

        self.window_len = self.config.get("window_len", 100)
        self.stride = self.config.get("stride", 1)
        self.hidden_dim = self.config.get("hidden_dim", 32)
        self.bottleneck_dim = self.config.get("bottleneck_dim", 16)
        self.num_layers = self.config.get("num_layers", 2)
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        self.epochs = self.config.get("epochs", 3)
        self.batch_size = self.config.get("batch_size", 256)
        self.random_state = self.config.get("random_seed", 42)
        self.device = torch.device("cpu")  # Strictly CPU for robust reproducibility

        self.scaler = StandardScalerPreserver()
        self.model: Optional[PyTorchLSTMAutoencoder] = None

    def fit(
        self,
        train_series: np.ndarray,
        val_series: Optional[np.ndarray] = None,
    ) -> "LSTMAutoencoderDetector":
        """Fit scaler and train LSTM Autoencoder strictly on permitted training windows.

        Args:
            train_series: 1D float array of normal training observations.
            val_series: Optional 1D float array of validation observations for early stopping.

        Returns:
            Self instance (fitted).
        """
        train_arr = self.validate_input_series(train_series, "train_series")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Fit training scaler and transform training sequence
        scaled_train = self.scaler.fit_transform(train_arr)

        # Extract sliding windows: shape (N_win, window_len)
        windows = extract_sliding_windows(
            scaled_train, window_len=self.window_len, stride=self.stride
        )

        # Convert to PyTorch Tensor: (N_win, window_len, 1)
        tensor_windows = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)
        dataset = TensorDataset(tensor_windows)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Instantiate model
        self.model = PyTorchLSTMAutoencoder(
            window_len=self.window_len,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_score(self, series: np.ndarray) -> np.ndarray:
        """Compute timestamp-level continuous reconstruction error anomaly scores in batches.

        Args:
            series: 1D float array of time-series observations of length N.

        Returns:
            1D float64 numpy array of continuous anomaly scores of length N.
            Higher values indicate larger reconstruction error (more anomalous).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Detector '{self.model_id}' must be fitted before predicting scores.")

        arr = self.validate_input_series(series, "predict_series")
        total_len = arr.size

        # Transform using frozen training scaler
        scaled_series = self.scaler.transform(arr)

        # Extract sliding windows
        windows = extract_sliding_windows(
            scaled_series, window_len=self.window_len, stride=self.stride
        )

        window_mse_list = []
        eval_batch_size = 512
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(windows), eval_batch_size):
                batch_win = torch.tensor(
                    windows[i : i + eval_batch_size], dtype=torch.float32
                ).unsqueeze(-1).to(self.device)
                recon = self.model(batch_win)
                diff = (batch_win - recon).squeeze(-1)
                batch_mse = torch.mean(diff ** 2, dim=1).cpu().numpy().astype(np.float64)
                window_mse_list.append(batch_mse)

        window_mse = np.concatenate(window_mse_list, axis=0) if window_mse_list else np.zeros(0, dtype=np.float64)

        # Aggregate window-level MSE scores back to timestamps using causal window-end alignment
        timestamp_scores = aggregate_window_scores_to_timestamps(
            window_mse,
            total_length=total_len,
            window_len=self.window_len,
            stride=self.stride,
            method="end",
        )

        return align_anomaly_scores(timestamp_scores, target_length=total_len)
