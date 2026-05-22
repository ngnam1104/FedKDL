"""
local_trainer.py
Huấn luyện cục bộ tại Sensor (Local SGD).

Hỗ trợ:
    - FedAvg / HFL-*:  Loss = MSE reconstruction error
    - FedProx:         Loss = MSE + (μ/2)||θ - θ_global||²  (Proximal term)
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict

from tasks.anomaly_1d.autoencoder import (
    get_model_flat_params, set_model_flat_params, get_model_state_dict_copy,
)


def local_sgd(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 5,
    lr: float = 0.01,
    global_model: Optional[nn.Module] = None,
    mu: float = 0.0,
    device: str = 'cpu',
) -> tuple[torch.Tensor, float]:
    """
    Huấn luyện cục bộ với SGD và trả về model update Δθ cùng với average training loss.

    Args:
        model:        Model cục bộ (sẽ bị modify in-place).
        dataloader:   DataLoader dữ liệu cục bộ của sensor.
        epochs:       Số vòng lặp SGD nội bộ (E = 5).
        lr:           Learning rate (η = 0.01).
        global_model: Model toàn cục đóng băng đầu round (dùng cho FedProx).
                      None → FedAvg mode (μ = 0).
        mu:           Hệ số proximal term FedProx (0.0 = FedAvg, 0.01 = FedProx).
        device:       'cpu' or 'cuda'.

    Returns:
        delta_theta: (total_params,) float tensor — model update = θ_new - θ_old.
        avg_loss: float — Trung bình Loss qua các batch/epoch
    """
    model = model.to(device)
    model.train()

    # Lưu θ_old để tính Δθ sau
    theta_old = get_model_flat_params(model).clone()

    # Đóng băng θ_global cho proximal term (FedProx)
    if mu > 0.0 and global_model is not None:
        global_params = get_model_flat_params(global_model).to(device).detach()
    else:
        global_params = None

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0

    for _epoch in range(epochs):
        for x_batch, _labels in dataloader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            x_hat = model(x_batch)

            # Reconstruction loss
            loss = mse_loss_fn(x_hat, x_batch)

            # FedProx proximal term: (μ/2) ||θ - θ_global||²
            if mu > 0.0 and global_params is not None:
                current_params = get_model_flat_params(model).to(device)
                proximal = (mu / 2.0) * torch.sum((current_params - global_params) ** 2)
                loss = loss + proximal

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

    # Tính model update Δθ = θ_new - θ_old
    theta_new = get_model_flat_params(model).clone()
    delta_theta = theta_new - theta_old
    
    avg_loss = total_loss / max(1, num_batches)

    return delta_theta, avg_loss
