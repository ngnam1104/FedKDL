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
    scale = data_max - data_min
    scale[scale == 0] = 1.0
    data = (data - data_min) / scale

    split_idx = int(len(data) * 0.7)
    train_data_split = data[:split_idx]
    train_labels_split = labels[:split_idx]
    
    val_data_split = data[split_idx:]
    val_labels_split = labels[split_idx:]
    
    return train_data_split, train_labels_split, val_data_split, val_labels_split


DATASET_CONFIGS = {
    'SMD':  {'n_features': 38, 'n_samples': 2000},
    'SMAP': {'n_features': 25, 'n_samples': 2000},
    'MSL':  {'n_features': 55, 'n_samples': 2000},
}


def load_real_smd(data_dir="datasets/SMD") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import os
    train_file = os.path.join(data_dir, "train", "machine-1-1.txt")
    test_file = os.path.join(data_dir, "test", "machine-1-1.txt")
    label_file = os.path.join(data_dir, "test_label", "machine-1-1.txt")
    
    if not os.path.exists(train_file):
        print(f"[Warning] Real SMD data not found at {train_file}. Falling back to synthetic.")
        tr_d, tr_l, val_d, val_l = generate_synthetic_timeseries(n_samples=2000, n_features=38, seed=42)
        # return dummy splits
        return tr_d, tr_l, val_d, val_l, val_d, val_l
        
    train_data = np.loadtxt(train_file, delimiter=",", dtype=np.float32)
    test_data = np.loadtxt(test_file, delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(label_file, delimiter=",", dtype=np.int32)
    
    # MinMaxScaler based on train data
    d_min = train_data.min(axis=0, keepdims=True)
    d_max = train_data.max(axis=0, keepdims=True)
    scale = d_max - d_min
    scale[scale == 0] = 1.0
    train_data = (train_data - d_min) / scale
    test_data = (test_data - d_min) / scale
    
    train_labels = np.zeros(len(train_data), dtype=np.int32)
    
    split_idx = int(len(train_data) * 0.7)
    train_data_split = train_data[:split_idx]
    train_labels_split = train_labels[:split_idx]
    
    val_data_split = train_data[split_idx:]
    val_labels_split = train_labels[split_idx:]
    
    return train_data_split, train_labels_split, val_data_split, val_labels_split, test_data, test_labels



def load_real_smap_msl(dataset: str, data_dir: str = "datasets/SMAP_MSL") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SMAP hoac MSL tu file .npy (telemanom format).
    SMAP channels: prefix A-* B-* C-* D-* E-* F-* G-* P-* R-* S-* T-*  (n_features=25)
    MSL  channels: prefix M-*                                              (n_features=55)
    """
    import os, csv
    from pathlib import Path

    train_dir = Path(data_dir) / "data" / "data" / "train"
    test_dir  = Path(data_dir) / "data" / "data" / "test"

    labels_csv = Path(data_dir) / "labeled_anomalies.csv"
    valid_channels = set()
    label_map: dict = {}
    
    if labels_csv.exists():
        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('spacecraft', '').strip() == dataset:
                    chan = row.get('chan_id', '').strip()
                    valid_channels.add(chan)
                    segs = row.get('anomaly_sequences', '[]').strip()
                    try:
                        label_map[chan] = eval(segs)
                    except Exception:
                        pass
    else:
        print(f"[Warning] {labels_csv} not found. Cannot filter channels.")
        data, labels = generate_synthetic_timeseries(n_samples=DATASET_CONFIGS[dataset]['n_samples'], n_features=DATASET_CONFIGS[dataset]['n_features'], seed=42)
        half = len(data) // 2
        return data[:half], np.zeros(half, dtype=np.int32), data[half:], labels[half:]

    def collect_files(d):
        return sorted([f for f in d.glob("*.npy") if f.stem in valid_channels])

    train_files = collect_files(train_dir)
    test_files  = collect_files(test_dir)

    if not train_files:
        print(f"[Warning] No real {dataset} data found in {train_dir}. Using synthetic.")
        cfg = DATASET_CONFIGS[dataset]
        data, labels = generate_synthetic_timeseries(
            n_samples=cfg['n_samples'], n_features=cfg['n_features'], seed=42)
        half = len(data) // 2
        return data[:half], np.zeros(half, dtype=np.int32), data[half:], labels[half:]

    def build_labels(files, segs_map, total_len):
        labels_arr = np.zeros(total_len, dtype=np.int32)
        offset = 0
        for f in files:
            arr = np.load(f)
            n = len(arr)
            chan = f.stem
            if chan in segs_map:
                for start, end in segs_map[chan]:
                    s = max(0, int(start) - offset)
                    e = min(n, int(end) - offset + 1)
                    if s < e:
                        labels_arr[offset + s: offset + e] = 1
            offset += n
        return labels_arr

    def stack_npy(files, split_ratio=0.7):
        arrays = [np.load(f).astype(np.float32) for f in files]
        fixed = []
        for a in arrays:
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            fixed.append(a)
            
        part1 = []
        part2 = []
        for a in fixed:
            split_idx = int(len(a) * split_ratio)
            part1.append(a[:split_idx])
            if split_ratio < 1.0:
                part2.append(a[split_idx:])
            
        return np.concatenate(part1, axis=0), np.concatenate(part2, axis=0) if part2 else None

    train_data, val_data = stack_npy(train_files, split_ratio=0.7)
    test_data, _  = stack_npy(test_files, split_ratio=1.0)

    # MinMaxScaler theo train
    d_min = train_data.min(axis=0, keepdims=True)
    d_max = train_data.max(axis=0, keepdims=True)
    scale = d_max - d_min
    scale[scale == 0] = 1.0
    train_data = (train_data - d_min) / scale
    val_data   = (val_data - d_min) / scale
    test_data  = (test_data  - d_min) / scale

    train_labels = np.zeros(len(train_data), dtype=np.int32)
    val_labels   = np.zeros(len(val_data), dtype=np.int32)
    test_labels  = build_labels(test_files, label_map, len(test_data))

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def load_dataset(name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset theo ten. Su dung du lieu thuc neu co san, khong fallback synthetic.
    Datasets ho tro: SMD, SMAP, MSL
    """
    if name == 'SMD':
        return load_real_smd("datasets/SMD")
    if name in ('SMAP', 'MSL'):
        return load_real_smap_msl(name, "datasets/SMAP_MSL")
    raise ValueError(f"Unknown dataset: {name}. Choose from ['SMD', 'SMAP', 'MSL']")



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
