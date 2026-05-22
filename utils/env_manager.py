"""
utils/env_manager.py
EnvironmentManager: sinh, luu va tai moi truong thi nghiem.
Tach roi hoan toan khoi Simulator de dam bao tinh cong bang.
Giai doan 5: Tach Topology va Data Partition doc lap.
"""
import pickle
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from physics_models.topology import (
    Topology3D, build_feasibility_graph,
    nearest_feasible_association, flat_topology_association, build_clusters
)
from tasks.anomaly_1d.dataloader import (
    load_dataset, SlidingWindowDataset, non_iid_partition
)


@dataclass
class TopologySnapshot:
    """Snapshot chi chua cau hinh Vat ly (Toa do, Lien ket, Association)."""
    N: int
    M: int
    seed: int

    sensor_positions: np.ndarray       # (N, 3)
    fog_positions: np.ndarray          # (M, 3)
    gateway_position: np.ndarray       # (3,)

    feasibility_graph_items: list      # [((tu,iu,tv,iv), {dist, SL_min, TL, NL, R_bps})]

    hfl_association: Dict[int, int]    # sensor_id -> fog_id
    flat_association: Dict[int, int]   # sensor_id -> -1
    clusters: Dict[int, List[int]]     # fog_id -> [sensor_ids]


@dataclass
class DataPartitionSnapshot:
    """Snapshot chi chua chi muc phan chia Data."""
    dataset_name: str
    N: int
    alpha: float
    seed: int

    client_data_indices: Dict[int, List[int]]  # sensor_id -> [sample_idx]
    val_indices: List[int]                     # validation split indices

    input_dim: int                     # window_size * n_features
    n_train_samples: int               # So luong window trong train_ds


class EnvironmentManager:
    ENVS_DIR = Path("environments")

    @staticmethod
    def topo_path(N: int, seed: int) -> Path:
        return EnvironmentManager.ENVS_DIR / "topo" / f"N_{N}" / f"topo_N{N}_seed{seed}.pkl"

    @staticmethod
    def data_path(N: int, dataset: str, alpha: float, seed: int) -> Path:
        alpha_str = str(alpha).replace(".", "p")
        return EnvironmentManager.ENVS_DIR / "data" / dataset / f"N_{N}" / f"data_N{N}_{dataset}_a{alpha_str}_seed{seed}.pkl"

    # --- TOPOLOGY ---
    @classmethod
    def generate_topology(cls, net_cfg, ac_cfg, seed: int) -> TopologySnapshot:
        topology = Topology3D(net_cfg, ac_cfg, seed)
        G_raw = build_feasibility_graph(topology, ac_cfg)

        graph_items = []
        for key, link in G_raw.items():
            graph_items.append((key, {
                "distance": link.distance,
                "SL_min":   link.SL_min,
                "TL":       link.TL,
                "NL":       link.NL,
                "R_bps":    link.R_bps,
            }))

        hfl_assoc  = nearest_feasible_association(topology, G_raw)
        flat_assoc = flat_topology_association(topology, G_raw)
        clusters   = build_clusters(hfl_assoc, topology.M)

        return TopologySnapshot(
            N=net_cfg.N_SENSORS,
            M=net_cfg.M_FOGS,
            seed=seed,
            sensor_positions=topology.sensor_positions.copy(),
            fog_positions=topology.fog_positions.copy(),
            gateway_position=topology.gateway_position.copy(),
            feasibility_graph_items=graph_items,
            hfl_association=dict(hfl_assoc),
            flat_association=dict(flat_assoc),
            clusters={k: list(v) for k, v in clusters.items()},
        )

    @classmethod
    def save_topology(cls, topo: TopologySnapshot):
        path = cls.topo_path(topo.N, topo.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(topo, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [saved] {path.name}  ({os.path.getsize(path)/1024:.1f} KB)")
        return path

    @classmethod
    def load_topology(cls, path: str) -> TopologySnapshot:
        with open(path, "rb") as f:
            return pickle.load(f)

    # --- DATA PARTITION ---
    @classmethod
    def generate_data_partition(
        cls, net_cfg, dataset_name: str, alpha: float, seed: int,
        window_size: int = 10, val_ratio: float = 0.3
    ) -> DataPartitionSnapshot:
        
        train_data, train_labels, test_data, test_labels = load_dataset(dataset_name, seed=seed)
        split_idx = int(len(train_data) * (1.0 - val_ratio))
        train_data_split, val_data_split     = train_data[:split_idx],   train_data[split_idx:]
        train_labels_split, val_labels_split = train_labels[:split_idx], train_labels[split_idx:]

        train_ds = SlidingWindowDataset(train_data_split, train_labels_split, window_size=window_size)
        val_ds   = SlidingWindowDataset(val_data_split,   val_labels_split,   window_size=window_size)

        client_indices = non_iid_partition(
            train_ds, net_cfg.N_SENSORS, alpha=alpha, seed=seed
        )
        # Note: Khong filter valid_sensors o day de data_partition hoan toan doc lap voi Topology!
        
        val_indices = list(range(len(val_ds)))

        return DataPartitionSnapshot(
            dataset_name=dataset_name,
            N=net_cfg.N_SENSORS,
            alpha=alpha,
            seed=seed,
            client_data_indices={int(k): list(v) for k, v in client_indices.items()},
            val_indices=val_indices,
            input_dim=train_ds.D,
            n_train_samples=len(train_ds),
        )

    @classmethod
    def save_data_partition(cls, data_part: DataPartitionSnapshot):
        path = cls.data_path(data_part.N, data_part.dataset_name, data_part.alpha, data_part.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data_part, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [saved] {path.name}  ({os.path.getsize(path)/1024:.1f} KB)")
        return path

    @classmethod
    def load_data_partition(cls, path: str) -> DataPartitionSnapshot:
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def generate_data_partition_2d(
        cls, net_cfg, dataset_name: str, alpha: float, seed: int,
        base_yaml_path: str
    ) -> DataPartitionSnapshot:
        import yaml
        import random
        
        random.seed(seed)
        np.random.seed(seed)
        
        if not os.path.exists(base_yaml_path):
            print(f"Warning: Khong tim thay {base_yaml_path}. Tra ve empty data partition.")
            all_images = []
        else:
            with open(base_yaml_path, 'r') as f:
                base_cfg = yaml.safe_load(f)
            train_path = base_cfg.get('train', '')
            
            if isinstance(train_path, str) and train_path.endswith('.txt'):
                with open(train_path, 'r') as f:
                    all_images = [line.strip() for line in f.readlines()]
            else:
                dataset_dir = Path(base_yaml_path).parent
                original_path = base_cfg.get('path', '')
                img_dir_candidates = [
                    dataset_dir / original_path / train_path,
                    dataset_dir / original_path.split('/')[0] / train_path,
                    dataset_dir / base_yaml_path.split('/')[-1].split('.')[0] / train_path
                ]
                
                img_dir = None
                for candidate in img_dir_candidates:
                    if candidate.exists() and candidate.is_dir():
                        img_dir = candidate
                        break
                
                if img_dir is None:
                    for potential_dir in dataset_dir.glob(f'**/{train_path}'):
                        if potential_dir.is_dir():
                            img_dir = potential_dir
                            break
                            
                all_images = []
                if img_dir is not None and img_dir.exists():
                    all_images = [str(p) for p in img_dir.glob('**/*.jpg')]
                
        num_samples = len(all_images)
        proportions = np.random.dirichlet(np.repeat(alpha, net_cfg.N_SENSORS))
        proportions = proportions / proportions.sum()
        
        # Đảm bảo mỗi thiết bị có ít nhất 2 ảnh (nếu đủ ảnh)
        min_samples = 2 if num_samples >= net_cfg.N_SENSORS * 2 else 0
        remaining_samples = max(0, num_samples - net_cfg.N_SENSORS * min_samples)
        
        client_splits = (proportions * remaining_samples).astype(int)
        client_splits += min_samples
        
        if num_samples > 0:
            client_splits[-1] = num_samples - sum(client_splits[:-1])
        
        # We only need to store the indices (not the strings), 
        # so simulator can lookup the same array from base_yaml.
        # But wait, shuffle changes the order!
        # If we just store indices from 0 to num_samples-1 randomly assigned.
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        client_data_indices = {}
        current_idx = 0
        for i in range(net_cfg.N_SENSORS):
            c_indices = indices[current_idx : current_idx + client_splits[i]]
            client_data_indices[i] = c_indices.tolist()
            current_idx += client_splits[i]
            
        return DataPartitionSnapshot(
            dataset_name=dataset_name,
            N=net_cfg.N_SENSORS,
            alpha=alpha,
            seed=seed,
            client_data_indices=client_data_indices,
            val_indices=[], # YOLO validates via the base yaml's val set
            input_dim=0, # not used for images in the same way
            n_train_samples=num_samples
        )

    # --- HELPERS ---
    @classmethod
    def restore_graph(cls, topo: TopologySnapshot):
        from types import SimpleNamespace
        G = {}
        for key, d in topo.feasibility_graph_items:
            G[key] = SimpleNamespace(**d)
        return G


