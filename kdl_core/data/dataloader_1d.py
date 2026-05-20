"""
dataloader_1d.py
Dữ liệu chuỗi thời gian 1D cho Anomaly Detection (Scenario 1).
Tạo Synthetic data để test code, với Non-IID Dirichlet partition.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional


# ──────────────────────────────────────────────────────────────────────
#  Synthetic Time-Series Generator
# ──────────────────────────────────────────────────────────────────────

def generate_synthetic_timeseries(
    n_samples: int = 2000,
    n_features: int = 38,
    anomaly_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo dữ liệu chuỗi thời gian tổng hợp mô phỏng SMD/SMAP/MSL.

    Normal data: multivariate Gaussian với tương quan không gian.
    Anomaly data: offset + noise spike tại các segment ngẫu nhiên.

    Returns:
        data:   (n_samples, n_features) float32 — chuỗi thời gian.
        labels: (n_samples,) int — 0=normal, 1=anomaly.
    """
    rng = np.random.RandomState(seed)

    # Normal: correlated multivariate Gaussian
    cov = rng.uniform(0.1, 0.5, (n_features, n_features))
    cov = cov @ cov.T + np.eye(n_features) * 0.5  # positive definite
    data = rng.multivariate_normal(np.zeros(n_features), cov, n_samples).astype(np.float32)

    # Thêm temporal trend
    t = np.linspace(0, 4 * np.pi, n_samples)
    for j in range(n_features):
        data[:, j] += 0.3 * np.sin(t + j * 0.5).astype(np.float32)

    # Anomaly injection — random contiguous segments
    labels = np.zeros(n_samples, dtype=np.int32)
    n_anomaly = int(n_samples * anomaly_ratio)
    n_segments = max(1, n_anomaly // 10)
    seg_len = n_anomaly // n_segments

    for _ in range(n_segments):
        start = rng.randint(0, n_samples - seg_len)
        data[start:start + seg_len] += rng.uniform(3.0, 6.0, n_features).astype(np.float32)
        data[start:start + seg_len] += rng.randn(seg_len, n_features).astype(np.float32) * 2.0
        labels[start:start + seg_len] = 1

    # Normalize to [0, 1] per feature
    data_min = data.min(axis=0, keepdims=True)
    data_max = data.max(axis=0, keepdims=True)
    data = (data - data_min) / (data_max - data_min + 1e-8)

    return data, labels


DATASET_CONFIGS = {
    'SMD':  {'n_features': 38, 'n_samples': 2000},
    'SMAP': {'n_features': 25, 'n_samples': 2000},
    'MSL':  {'n_features': 55, 'n_samples': 2000},
}


def load_real_smd(data_dir="datasets/SMD") -> Tuple[np.ndarray, np.ndarray]:
    import os
    train_file = os.path.join(data_dir, "train", "machine-1-1.txt")
    test_file = os.path.join(data_dir, "test", "machine-1-1.txt")
    label_file = os.path.join(data_dir, "test_label", "machine-1-1.txt")
    
    if not os.path.exists(train_file):
        print(f"[Warning] Real SMD data not found at {train_file}. Falling back to synthetic.")
        return generate_synthetic_timeseries(n_samples=2000, n_features=38, seed=42)
        
    train_data = np.loadtxt(train_file, delimiter=",", dtype=np.float32)
    test_data = np.loadtxt(test_file, delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(label_file, delimiter=",", dtype=np.int32)
    
    # MinMaxScaler based on train data
    d_min = train_data.min(axis=0, keepdims=True)
    d_max = train_data.max(axis=0, keepdims=True)
    train_data = (train_data - d_min) / (d_max - d_min + 1e-8)
    test_data = (test_data - d_min) / (d_max - d_min + 1e-8)
    
    data = np.vstack([train_data, test_data])
    labels = np.concatenate([np.zeros(len(train_data), dtype=np.int32), test_labels])
    
    return data, labels


def load_dataset(name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tải dataset theo tên. Tích hợp dữ liệu thực nếu có, nếu không fallback qua synthetic.
    """
    if name == 'SMD':
        return load_real_smd("datasets/SMD")
    
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[name]
    return generate_synthetic_timeseries(
        n_samples=cfg['n_samples'],
        n_features=cfg['n_features'],
        seed=seed,
    )


# ──────────────────────────────────────────────────────────────────────
#  Sliding Window Dataset
# ──────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Chuyển đổi chuỗi thời gian thành fixed-length vectors qua sliding window.
    Mỗi sample x ∈ ℝ^(window_size × n_features) được flatten thành ℝ^D.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray,
                 window_size: int = 10, stride: int = 1):
        """
        Args:
            data:        (T, n_features) float32.
            labels:      (T,) int — label của mỗi timestep.
            window_size: Độ dài cửa sổ trượt.
            stride:      Bước nhảy.
        """
        self.window_size = window_size
        self.n_features = data.shape[1]
        self.D = window_size * self.n_features

        windows, window_labels = [], []
        T = len(data)
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            win = data[start:end].flatten()     # (D,)
            # Label cửa sổ: 1 nếu có bất kỳ anomaly nào trong window
            lbl = int(labels[start:end].max())
            windows.append(win)
            window_labels.append(lbl)

        self.X = torch.tensor(np.array(windows), dtype=torch.float32)
        self.y = torch.tensor(np.array(window_labels), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────
#  Non-IID Dirichlet Partition
# ──────────────────────────────────────────────────────────────────────

def non_iid_partition(
    dataset: SlidingWindowDataset,
    n_clients: int,
    alpha: float = 0.1,
    seed: int = 0,
) -> Dict[int, List[int]]:
    """
    Phân chia Non-IID theo phân phối Dirichlet(α).

    α nhỏ (0.1) → phân phối rất skewed (mỗi client chỉ thấy 1-2 class).
    α lớn (10)  → gần IID.

    Returns:
        client_indices: dict[client_id → list of sample indices]
    """
    rng = np.random.RandomState(seed)
    labels = dataset.y.numpy()
    n_classes = int(labels.max()) + 1
    n_samples = len(dataset)

    # Group indices by class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(n_classes)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Dirichlet allocation
    client_indices = {i: [] for i in range(n_clients)}
    for c in range(n_classes):
        proportions = rng.dirichlet([alpha] * n_clients)
        indices = class_indices[c]
        if not indices:
            continue
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, cuts)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return client_indices


def make_client_loaders(
    dataset: SlidingWindowDataset,
    client_indices: Dict[int, List[int]],
    batch_size: int = 64,
) -> Dict[int, DataLoader]:
    """
    Tạo DataLoader cho mỗi client từ partition indices.
    """
    from torch.utils.data import Subset
    loaders = {}
    for client_id, indices in client_indices.items():
        if not indices:
            continue
        subset = Subset(dataset, indices)
        loaders[client_id] = DataLoader(subset, batch_size=batch_size,
                                        shuffle=True, drop_last=False)
    return loaders


def make_val_loader(
    dataset: SlidingWindowDataset,
    batch_size: int = 256,
) -> DataLoader:
    """DataLoader toàn bộ dataset cho validation (tính ngưỡng τ_A)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
