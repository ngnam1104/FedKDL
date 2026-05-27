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

    auv_positions: np.ndarray       # (N, 3)
    relay_positions: np.ndarray          # (M, 3)
    gateway_position: np.ndarray       # (3,)

    feasibility_graph_items: list      # [((tu,iu,tv,iv), {dist, SL_min, TL, NL, R_bps})]

    hfl_association: Dict[int, int]    # auv_id -> relay_id
    flat_association: Dict[int, int]   # auv_id -> -1
    clusters: Dict[int, List[int]]     # relay_id -> [auv_ids]


@dataclass
class DataPartitionSnapshot:
    """Snapshot chi chua chi muc phan chia Data."""
    dataset_name: str
    N: int
    alpha: float
    seed: int

    auv_data_indices: Dict[int, List[int]]  # auv_id -> [sample_idx]
    val_indices: List[int]                     # validation split indices

    input_dim: int                     # window_size * n_features
    n_train_samples: int               # So luong window trong train_ds
    public_data_indices: List[int] = None # proxy dataset cho Gateway KD


class EnvironmentManager:
    ENVS_DIR = Path("environments")

    @staticmethod
    def topo_path(task_type: str, N: int, seed: int) -> Path:
        return EnvironmentManager.ENVS_DIR / task_type / "topo" / f"N_{N}" / f"topo_N{N}_seed{seed}.pkl"

    @staticmethod
    def data_path(task_type: str, N: int, dataset: str, alpha: float, seed: int) -> Path:
        alpha_str = str(alpha).replace(".", "p")
        return EnvironmentManager.ENVS_DIR / task_type / "data" / dataset / f"N_{N}" / f"data_N{N}_{dataset}_a{alpha_str}_seed{seed}.pkl"

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
            N=net_cfg.N_AUVS,
            M=net_cfg.M_RELAYS,
            seed=seed,
            auv_positions=topology.auv_positions.copy(),
            relay_positions=topology.relay_positions.copy(),
            gateway_position=topology.gateway_position.copy(),
            feasibility_graph_items=graph_items,
            hfl_association=dict(hfl_assoc),
            flat_association=dict(flat_assoc),
            clusters={k: list(v) for k, v in clusters.items()},
        )

    @classmethod
    def save_topology(cls, topo: TopologySnapshot, task_type: str):
        path = cls.topo_path(task_type, topo.N, topo.seed)
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
        
        train_data, train_labels, val_data, val_labels, test_data, test_labels = load_dataset(dataset_name, seed=seed)
        
        train_ds = SlidingWindowDataset(train_data, train_labels, window_size=window_size)
        val_ds   = SlidingWindowDataset(val_data,   val_labels,   window_size=window_size)

        auv_indices = non_iid_partition(
            train_ds, net_cfg.N_AUVS, alpha=alpha, seed=seed
        )
        # Note: Khong filter valid_auvs o day de data_partition hoan toan doc lap voi Topology!
        
        val_indices = list(range(len(val_ds)))

        return DataPartitionSnapshot(
            dataset_name=dataset_name,
            N=net_cfg.N_AUVS,
            alpha=alpha,
            seed=seed,
            auv_data_indices={int(k): list(v) for k, v in auv_indices.items()},
            val_indices=val_indices,
            input_dim=train_ds.D,
            n_train_samples=len(train_ds),
        )

    @classmethod
    def save_data_partition(cls, data_part: DataPartitionSnapshot, task_type: str):
        path = cls.data_path(task_type, data_part.N, data_part.dataset_name, data_part.alpha, data_part.seed)
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
        cls, net_cfg, topo, dataset_name: str, alpha: float, seed: int,
        base_yaml_path: str
    ) -> DataPartitionSnapshot:
        import yaml
        import random
        from pathlib import Path as PPath
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 1. Thu thập danh sách ảnh
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
                dataset_dir = PPath(base_yaml_path).parent
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
        all_images = list(all_images)
        
        # 2. Đọc file label để phân loại ảnh vào 4 Habitat Bucket
        # URPC2020 classes: holothurian=0, echinus=1, scallop=2, starfish=3
        # Habitat mapping (theo độ sâu):
        # H0 (Scallop - Cạn nhất): 2
        # H1 (Echinus - Tầng 2): 1
        # H2 (Starfish - Tầng 3): 3
        # H3 (Holothurian - Sâu nhất): 0
        HABITAT_TO_URPC = {0: 2, 1: 1, 2: 3, 3: 0}
        URPC_TO_HABITAT = {v: k for k, v in HABITAT_TO_URPC.items()}

        imgs_by_habitat = {h: [] for h in range(4)}
        imgs_noclass = []

        for idx, img_path in enumerate(all_images):
            lbl_path = img_path.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
            lbl_path = lbl_path.rsplit('.', 1)[0] + '.txt'
            counts = {c: 0 for c in range(4)}
            try:
                with open(lbl_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id in counts:
                                counts[cls_id] += 1
                if sum(counts.values()) == 0:
                    imgs_noclass.append(idx)
                    continue
                dominant = max(counts, key=counts.get)
                habitat = URPC_TO_HABITAT.get(dominant, 0)
                imgs_by_habitat[habitat].append(idx)
            except Exception:
                imgs_noclass.append(idx)

        # Phân rải ảnh không nhãn
        for i, idx in enumerate(imgs_noclass):
            imgs_by_habitat[i % 4].append(idx)

        print("  [Habitat Buckets] " +
              " | ".join(f"H{h}({len(imgs_by_habitat[h])})" for h in range(4)))

        # 3. Trích 30% làm Public Dataset (Proxy KD) rải đều 4 habitat
        proxy_per_habitat = max(1, int(num_samples * 0.3) // 4)
        public_indices = []
        for h in range(4):
            pool = list(imgs_by_habitat[h])
            random.shuffle(pool)
            public_indices.extend(pool[:proxy_per_habitat])

        # 4. Gán Habitat cho AUV dựa hoàn toàn vào Độ Sâu (Trục Z)
        auv_pos = topo.auv_positions  # (N, 3)
        auv_habitat = np.zeros(net_cfg.N_AUVS, dtype=int)
        for i in range(net_cfg.N_AUVS):
            z = auv_pos[i, 2]
            if z < 625.0:
                auv_habitat[i] = 0  # H0: Scallop
            elif z < 750.0:
                auv_habitat[i] = 1  # H1: Echinus
            elif z < 875.0:
                auv_habitat[i] = 2  # H2: Starfish
            else:
                auv_habitat[i] = 3  # H3: Holothurian
        
        habitat_count = [int(np.sum(auv_habitat == h)) for h in range(4)]
        print("  [AUV→DepthHabitat] " +
              " | ".join(f"H{h}:{habitat_count[h]}auvs" for h in range(4)))

        # 5. Xác định P_skew từ alpha
        if alpha >= 1000.0:
            p_skew = 0.25  # IID
        elif alpha >= 1.0:
            p_skew = 0.80  # Non-IID vừa
        else:
            p_skew = 0.95  # Non-IID cực đoan

        # 6. Trộn ảnh cho từng AUV
        auv_pool_size = int(num_samples * 0.7)
        n_per_auv = max(1, auv_pool_size // net_cfg.N_AUVS)

        auv_data_indices = {}
        for i in range(net_cfg.N_AUVS):
            h = int(auv_habitat[i])
            dominant_pool = list(imgs_by_habitat[h])
            noise_pool = []
            for hh in range(4):
                if hh != h:
                    noise_pool.extend(imgs_by_habitat[hh])

            random.shuffle(dominant_pool)
            random.shuffle(noise_pool)

            n_dom   = max(1, int(n_per_auv * p_skew))
            n_noise = max(0, n_per_auv - n_dom)

            chosen = dominant_pool[:n_dom] + noise_pool[:n_noise]
            if len(chosen) < n_per_auv:
                extra = (dominant_pool + noise_pool)[len(chosen):n_per_auv]
                chosen += extra

            auv_data_indices[i] = chosen
            
        return DataPartitionSnapshot(
            dataset_name=dataset_name,
            N=net_cfg.N_AUVS,
            alpha=alpha,
            seed=seed,
            auv_data_indices=auv_data_indices,
            val_indices=[], # YOLO validates via the base yaml's val set
            input_dim=0, # not used for images in the same way
            n_train_samples=num_samples,
            public_data_indices=public_indices
        )

    # --- HELPERS ---
    @classmethod
    def restore_graph(cls, topo: TopologySnapshot):
        from types import SimpleNamespace
        G = {}
        for key, d in topo.feasibility_graph_items:
            G[key] = SimpleNamespace(**d)
        return G


