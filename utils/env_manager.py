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
from typing import Dict, List, Optional, Tuple

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
    log_text: str = ""                    # Log metadata for saving to txt


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
    def shrink_image_pool(
        cls,
        data_part: DataPartitionSnapshot,
        all_images: List[str],
        max_pool: int,
        seed: Optional[int] = None,
    ) -> Tuple[DataPartitionSnapshot, List[str]]:
        """
        Thu hẹp pool ảnh + partition xuống max_pool mẫu (dry-test).
        Giữ cấu trúc non-IID: remap chỉ số gốc → 0..max_pool-1 theo cùng seed.
        """
        import random
        from dataclasses import replace

        pool_size = len(all_images)
        if pool_size <= max_pool:
            return data_part, all_images

        rng = random.Random(seed if seed is not None else data_part.seed)
        selected_old = sorted(rng.sample(range(pool_size), max_pool))
        old_to_new = {old: new for new, old in enumerate(selected_old)}
        shrunk_images = [all_images[i] for i in selected_old]

        public_src = data_part.public_data_indices or []
        new_public = [old_to_new[i] for i in public_src if i in old_to_new]

        new_auv: Dict[int, List[int]] = {}
        for sid, idx_list in data_part.auv_data_indices.items():
            remapped = [old_to_new[i] for i in idx_list if i in old_to_new]
            if remapped:
                new_auv[int(sid)] = remapped

        total_auv = sum(len(v) for v in new_auv.values())
        log = (
            f"[dry-test] pool {pool_size} → {max_pool} ảnh | "
            f"public={len(new_public)} | AUV có data={len(new_auv)} | train={total_auv}"
        )
        new_part = replace(
            data_part,
            auv_data_indices=new_auv,
            public_data_indices=new_public,
            n_train_samples=max_pool,
            log_text=(data_part.log_text or "") + "\n" + log,
        )
        return new_part, shrunk_images

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
        base_yaml_path = str(PPath(base_yaml_path).resolve())  # Luôn dùng absolute path
        if not os.path.exists(base_yaml_path):
            raise FileNotFoundError(f"[ENV ERROR] Không tìm thấy {base_yaml_path}")
        
        with open(base_yaml_path, 'r') as f:
            base_cfg = yaml.safe_load(f)
        train_path = base_cfg.get('train', '')
        
        if isinstance(train_path, str) and train_path.endswith('.txt'):
            with open(train_path, 'r') as f:
                all_images = [line.strip() for line in f.readlines() if line.strip()]
        else:
            dataset_dir = PPath(base_yaml_path).parent
            original_path = base_cfg.get('path', '')
            img_dir_candidates = [
                dataset_dir / original_path / train_path,
                dataset_dir / original_path.split('/')[0] / train_path,
                dataset_dir / PPath(base_yaml_path).stem / train_path,
                dataset_dir / train_path,  # fallback nếu không có path prefix
            ]
            
            img_dir = None
            for candidate in img_dir_candidates:
                print(f"  [probe] {candidate}")
                if candidate.exists() and candidate.is_dir():
                    img_dir = candidate
                    print(f"  [found] img_dir = {img_dir}")
                    break
            
            if img_dir is None:
                # Deep scan fallback (không resolve để xuyên qua symlink/bind-mount)
                for potential_dir in dataset_dir.rglob(train_path):
                    if potential_dir.is_dir():
                        img_dir = potential_dir
                        print(f"  [found via rglob] img_dir = {img_dir}")
                        break
                        
            if img_dir is None or not img_dir.exists():
                print(f"  [warn] Không tìm thấy thư mục {train_path} trực tiếp. Thực hiện quét toàn bộ ảnh trong {dataset_dir}...")
                all_images = []
                for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG'):
                    for p in dataset_dir.rglob(ext):
                        if 'train' in p.parts: # Chỉ lấy ảnh trong thư mục train
                            all_images.append(str(p))
                
                if not all_images:
                    raise FileNotFoundError(
                        f"[ENV ERROR] Không tìm thấy ảnh nào trong '{dataset_dir}' (có chứa 'train' trong đường dẫn).\n"
                        f"  Đường dẫn đã thử: {[str(c) for c in img_dir_candidates]}"
                    )
                all_images = sorted(set(all_images))
                print(f"  [found via deep scan] {len(all_images)} ảnh")
            else:
                all_images = []
                for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG'):
                    all_images.extend([str(p) for p in img_dir.glob(f'**/{ext}')])
                all_images = sorted(set(all_images))  # dedup + sort

        num_samples = len(all_images)
        if num_samples == 0:
            raise RuntimeError(
                f"[ENV ERROR] Tìm thấy thư mục ảnh nhưng KHÔNG CÓ ẢNH NÀO (jpg/png) trong '{img_dir}'."
            )
        print(f"  [images] Tìm thấy {num_samples} ảnh trong {img_dir}")
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

        logs = []
        log_str = "  [Habitat Buckets] " + " | ".join(f"H{h}({len(imgs_by_habitat[h])})" for h in range(4))
        print(log_str)
        logs.append(log_str)

        # 3. Trích 20% làm Public Dataset (Proxy KD) rải đều 4 habitat và CÔ LẬP hoàn toàn
        proxy_per_habitat = max(1, int(num_samples * 0.2) // 4)
        public_indices = []
        
        log_str = "\n  [Data Partitioning] Tách 20% Public Data và 80% AUV Data:"
        print(log_str)
        logs.append(log_str)
        for h in range(4):
            pool = imgs_by_habitat[h] # Trỏ trực tiếp vào kho
            old_size = len(pool)
            random.shuffle(pool)
            
            # Đưa vào tập Public (Proxy KD)
            public_indices.extend(pool[:proxy_per_habitat])
            
            # XÓA TRIỆT ĐỂ 20% này khỏi kho, 80% còn lại mới để cho AUV chia nhau
            imgs_by_habitat[h] = pool[proxy_per_habitat:]
            new_size = len(imgs_by_habitat[h])
            log_str = f"    - Habitat {h}: Tổng {old_size} ảnh -> Lấy {proxy_per_habitat} ảnh cho KD -> Còn lại {new_size} ảnh cho AUV"
            print(log_str)
            logs.append(log_str)
        log_str = f"  => Tổng Public (Proxy) Data: {len(public_indices)} ảnh.\n"
        print(log_str)
        logs.append(log_str)

        # 4. Tính toán Gaussian Affinity (Độ bám dính Ecotone) cho từng AUV
        auv_pos = topo.auv_positions  # (N, 3)
        centers = [562.5, 687.5, 812.5, 937.5]  # Tâm 4 vùng sinh thái
        sigma = 40.0  # Hệ số lan truyền (càng lớn vùng biên càng rộng)
        
        affinity_matrix = np.zeros((net_cfg.N_AUVS, 4))
        auv_primary_habitat = np.zeros(net_cfg.N_AUVS, dtype=int)
        
        for i in range(net_cfg.N_AUVS):
            z = auv_pos[i, 2]
            weights = [np.exp(-((z - c)**2) / (2 * sigma**2)) for c in centers]
            
            # Xử lý đặc biệt cho IID (alpha >= 1000)
            if alpha >= 1000.0:
                weights = [0.25, 0.25, 0.25, 0.25]
                
            total_w = sum(weights)
            affinity_matrix[i] = [w / total_w for w in weights]
            
            # Nếu là IID, gán primary habitat xoay vòng để log đẹp, không ảnh hưởng logic rút ảnh
            if alpha >= 1000.0:
                auv_primary_habitat[i] = i % 4
            else:
                auv_primary_habitat[i] = int(np.argmax(affinity_matrix[i]))
            
        habitat_count = [int(np.sum(auv_primary_habitat == h)) for h in range(4)]
        log_str = "\n  [AUV Primary Habitat] " + " | ".join(f"H{h}:{habitat_count[h]}auvs" for h in range(4))
        print(log_str)
        logs.append(log_str)

        # 5. Phân bổ Quantity Skew (Dựa trên Primary Habitat và Dirichlet)
        auv_total_images = np.zeros(net_cfg.N_AUVS, dtype=int)
        
        if alpha >= 1000.0:
            # IID: Tính tổng ảnh khả dụng của CẢ 4 KHO
            total_pool_size = sum(len(imgs_by_habitat[h]) for h in range(4))
            proportions = np.random.dirichlet(np.repeat(alpha, net_cfg.N_AUVS))
            
            min_img = min(20, total_pool_size // net_cfg.N_AUVS) if net_cfg.N_AUVS > 0 else 0
            remaining = max(0, total_pool_size - min_img * net_cfg.N_AUVS)
            
            splits = (proportions * remaining).astype(int) + min_img
            if total_pool_size > 0 and sum(splits) < total_pool_size:
                splits[-1] += total_pool_size - sum(splits)
                
            for i in range(net_cfg.N_AUVS):
                auv_total_images[i] = splits[i]
        else:
            # Non-IID: Phân bổ theo từng vùng
            auvs_in_habitat = {h: [] for h in range(4)}
            for i in range(net_cfg.N_AUVS):
                auvs_in_habitat[int(auv_primary_habitat[i])].append(i)
                
            for h in range(4):
                auv_list = auvs_in_habitat[h]
                n_auvs_in_h = len(auv_list)
                if n_auvs_in_h == 0:
                    continue
                    
                # Tổng lượng ảnh khả dụng của vùng này (không tính proxy)
                pool_size_h = len(imgs_by_habitat[h])
                
                # Dirichlet phân bổ Quantity (SỐ LƯỢNG) cho các AUV trong vùng
                proportions = np.random.dirichlet(np.repeat(alpha, n_auvs_in_h))
                
                min_img = min(20, pool_size_h // n_auvs_in_h) if n_auvs_in_h > 0 else 0
                remaining = max(0, pool_size_h - min_img * n_auvs_in_h)
                
                splits = (proportions * remaining).astype(int) + min_img
                if pool_size_h > 0 and sum(splits) < pool_size_h:
                    splits[-1] += pool_size_h - sum(splits)
                    
                for idx, auv_id in enumerate(auv_list):
                    auv_total_images[auv_id] = splits[idx]

        # 6. Rút ảnh từ các kho dựa vào Ecotone Affinity
        auv_data_indices = {}
        # Shuffle lại các kho ảnh
        for h in range(4):
            random.shuffle(imgs_by_habitat[h])
            
        # Dùng con trỏ để theo dõi ảnh đã lấy từ mỗi kho
        bucket_pointers = [0, 0, 0, 0]
        
        for i in range(net_cfg.N_AUVS):
            total_needed = auv_total_images[i]
            probs = affinity_matrix[i]
            
            # Số lượng cần rút từ 4 kho
            draws = (probs * total_needed).astype(int)
            # Sửa sai số làm tròn
            draws[-1] = total_needed - sum(draws[:-1])
            
            chosen = []
            for h in range(4):
                needed = draws[h]
                if needed <= 0:
                    continue
                    
                pool = imgs_by_habitat[h]
                ptr = bucket_pointers[h]
                
                # Nếu thiếu ảnh trong kho (rất dễ xảy ra nếu 1 kho ít ảnh), bốc random xoay vòng
                if ptr + needed > len(pool):
                    extracted = [pool[(ptr + j) % max(1, len(pool))] for j in range(needed)]
                    bucket_pointers[h] = (ptr + needed) % max(1, len(pool))
                else:
                    extracted = pool[ptr : ptr + needed]
                    bucket_pointers[h] += needed
                    
                chosen.extend(extracted)
                
            random.shuffle(chosen)
            auv_data_indices[i] = chosen
            
        # Thêm thông tin tọa độ và số ảnh vào log
        logs.append("\n  [Chi tiết AUV]")
        for i in range(net_cfg.N_AUVS):
            pos = auv_pos[i]
            n_imgs = len(auv_data_indices[i])
            h = auv_primary_habitat[i]
            logs.append(f"    - AUV {i:2d}: Z={pos[2]:6.1f} | Primary Habitat={h} | Nhận {n_imgs} ảnh")

        return DataPartitionSnapshot(
            dataset_name=dataset_name,
            N=net_cfg.N_AUVS,
            alpha=alpha,
            seed=seed,
            auv_data_indices=auv_data_indices,
            val_indices=[], # YOLO validates via the base yaml's val set
            input_dim=0, # not used for images in the same way
            n_train_samples=num_samples,
            public_data_indices=public_indices,
            log_text="\n".join(logs)
        )

    # --- HELPERS ---
    @classmethod
    def restore_graph(cls, topo: TopologySnapshot):
        from types import SimpleNamespace
        G = {}
        for key, d in topo.feasibility_graph_items:
            G[key] = SimpleNamespace(**d)
        return G


