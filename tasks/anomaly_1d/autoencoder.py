"""
autoencoder.py
Mô hình Autoencoder đối xứng vừa (~54,000 params) cho Anomaly Detection 1D.
Kiến trúc: D → 64 → 32 → 16 → 32 → 64 → D
Loss: MSE Reconstruction Error
"""

import torch
import torch.nn as nn
from typing import Tuple


class SmallAutoencoder(nn.Module):
    """
    Symmetric Autoencoder cho time-series anomaly detection.

    Encoder: D_in → 48 → 24 → 12 (bottleneck)
    Decoder: 12 → 24 → 48 → D_in
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Chiều dữ liệu đầu vào D (window_size × n_features).
        """
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, input_dim),
            # Không có activation cuối — dữ liệu đã normalize về [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            x_hat: (batch, input_dim) — reconstructed
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Lấy latent representation z ∈ ℝ^16."""
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tính MSE reconstruction error cho từng sample.
        Dùng để tính ngưỡng τ_A và phát hiện anomaly.

        Returns:
            errors: (batch,) — per-sample MSE.
        """
        x_hat = self.forward(x)
        errors = ((x - x_hat) ** 2).mean(dim=1)
        return errors

    def count_parameters(self) -> int:
        """Đếm tổng số trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """MSE Reconstruction Loss (batch mean)."""
    return nn.functional.mse_loss(x_hat, x, reduction='mean')


def get_model_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten tất cả tham số thành 1D tensor (dùng cho Top-K compression)."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_model_flat_params(model: nn.Module, flat_params: torch.Tensor):
    """Load 1D tensor trở lại model parameters."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
        offset += numel


def get_model_state_dict_copy(model: nn.Module) -> dict:
    """Deep copy state dict (dùng để lưu θ_global đầu round)."""
    return {k: v.clone() for k, v in model.state_dict().items()}
