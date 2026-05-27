import sys
import os
import yaml
import numpy as np
from pathlib import Path

# Thêm đường dẫn thư mục gốc vào sys.path để chạy ở bất cứ đâu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import NetworkConfig, AcousticChannelConfig
from utils.env_manager import EnvironmentManager, TopologySnapshot, DataPartitionSnapshot
from physics_models.topology import Topology3D, build_feasibility_graph, nearest_feasible_association
from tasks.detection_2d.knowledge_compression.knowledge_association import knowledge_aware_association

def get_actual_labels_from_images(base_yaml_path, img_indices):
    with open(base_yaml_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    train_path = base_cfg.get('train', '')
    
    if isinstance(train_path, str) and train_path.endswith('.txt'):
        with open(train_path, 'r') as f:
            all_images = [line.strip() for line in f.readlines()]
    else:
        dataset_dir = Path(base_yaml_path).parent
        original_path = base_cfg.get('path', '')
        # Tìm thư mục train
        img_dir = dataset_dir / original_path / train_path
        all_images = [str(p) for p in img_dir.glob('**/*.jpg')]
        
    hist = np.zeros(4)
    HABITAT_TO_URPC = {0: 2, 1: 1, 2: 3, 3: 0}
    URPC_TO_HABITAT = {v: k for k, v in HABITAT_TO_URPC.items()}
    
    for idx in img_indices:
        img_path = all_images[idx]
        lbl_path = img_path.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\').rsplit('.', 1)[0] + '.txt'
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
                continue
            dominant = max(counts, key=counts.get)
            habitat = URPC_TO_HABITAT.get(dominant, 0)
            hist[habitat] += 1
        except:
            pass
    return hist

def main():
    task_type = '2d'
    dataset = 'URPC'
    n_auvs = 20
    alpha = 1.0
    seed = 42

    print("="*60)
    print("1. ĐANG TẢI MÔI TRƯỜNG TỪ .PKL")
    print("="*60)
    
    topo_path = EnvironmentManager.topo_path(task_type, n_auvs, seed)
    data_path = EnvironmentManager.data_path(task_type, n_auvs, dataset, alpha, seed)
    
    if not topo_path.exists() or not data_path.exists():
        print(f"Chưa tìm thấy môi trường! Hãy chạy: python utils/generate_all_envs.py --dataset {dataset} --n {n_auvs}")
        return
        
    topo_snapshot = EnvironmentManager.load_topology(topo_path)
    data_snapshot = EnvironmentManager.load_data_partition(data_path)
    
    net_cfg = NetworkConfig()
    net_cfg.N_AUVS = n_auvs
    net_cfg.M_RELAYS = topo_snapshot.M
    
    ac_cfg = AcousticChannelConfig()
    
    # Khôi phục đồ thị
    class DummyTopo:
        def __init__(self, n, m, p_auv, p_rel):
            self.N = n
            self.M = m
            self.auv_positions = p_auv
            self.relay_positions = p_rel
            
    dummy_topo = DummyTopo(n_auvs, topo_snapshot.M, topo_snapshot.auv_positions, topo_snapshot.relay_positions)
    G = EnvironmentManager.restore_graph(topo_snapshot)
    
    print("\n[+] Thống kê AUV (20 con):")
    print(f"{'AUV':<4} | {'Z_Depth':<8} | {'Primary Habitat':<16} | {'Affinity P (H0,H1,H2,H3)':<30} | {'Hist (S, E, Star, Holo)'}")
    print("-" * 105)
    
    auv_hists = np.zeros((n_auvs, 4))
    
    centers = [562.5, 687.5, 812.5, 937.5]
    sigma = 40.0
    
    for i in range(n_auvs):
        z = topo_snapshot.auv_positions[i, 2]
        
        weights = [np.exp(-((z - c)**2) / (2 * sigma**2)) for c in centers]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        prob_str = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}, {probs[3]:.2f}]"
        
        primary_h = int(np.argmax(probs))
        h_names = ["Scallop", "Echinus", "Starfish", "Holothurian"]
        h_name = h_names[primary_h]
            
        hist = get_actual_labels_from_images("datasets/URPC2020.yaml", data_snapshot.auv_data_indices[i])
        auv_hists[i] = hist
        
        if i < 20:
            hist_str = f"[{int(hist[0])}, {int(hist[1])}, {int(hist[2])}, {int(hist[3])}]"
            print(f"{i:<4} | {z:<8.1f} | {h_name:<16} | {prob_str:<30} | {hist_str}")

    print("\n" + "="*60)
    print("2. SO SÁNH PHÂN CỤM: VẬT LÝ vs EMD TRI THỨC")
    print("="*60)
    
    # 2.1 Phân cụm vật lý (Nearest Relay)
    physical_assoc = nearest_feasible_association(dummy_topo, G)
    phys_cluster_hists = np.zeros((topo_snapshot.M, 4))
    
    print("\n[A] KỊCH BẢN VẬT LÝ (Nearest Relay - Dùng trong HFL FedAvg)")
    for auv_id, relay_id in physical_assoc.items():
        phys_cluster_hists[relay_id] += auv_hists[auv_id]
        
    for m in range(topo_snapshot.M):
        h = phys_cluster_hists[m]
        total = sum(h)
        if total == 0: continue
        print(f"Relay {m}: Gom được {int(total):<4} ảnh -> Hist: [{int(h[0])}, {int(h[1])}, {int(h[2])}, {int(h[3])}]")
        if h[0] > total * 0.7:
            print(f"   => CẢNH BÁO: Mất cân bằng nặng! Relay {m} bị 'bội thực' Sò Điệp (Scallop) vì toàn AUV cạn.")
    
    # 2.2 Phân cụm EMD (Knowledge-Aware) ở Round 1
    # Mô phỏng: Sau Round 0, các Relay đã học được tri thức từ các AUV mà nó kết nối vật lý.
    # Nên Histogram tri thức của Relay lúc này chính là phys_cluster_hists (đã chuẩn hóa).
    relay_hists = np.zeros((topo_snapshot.M, 4))
    for m in range(topo_snapshot.M):
        total = sum(phys_cluster_hists[m])
        if total > 0:
            relay_hists[m] = phys_cluster_hists[m] / total
        else:
            relay_hists[m] = np.ones(4) / 4.0
            
    # Tăng beta lên 0.8 để thấy rõ hiệu ứng EMD kéo AUV
    emd_assoc = knowledge_aware_association(dummy_topo, G, auv_hists, relay_hists, beta=0.8)
    emd_cluster_hists = np.zeros((topo_snapshot.M, 4))
    
    print("\n[B] KỊCH BẢN TRI THỨC (EMD FedKDL) - Tại Vòng 1 (Sau khi Relay đã có tri thức)")
    for auv_id, relay_id in emd_assoc.items():
        emd_cluster_hists[relay_id] += auv_hists[auv_id]
        
    for m in range(topo_snapshot.M):
        h = emd_cluster_hists[m]
        total = sum(h)
        if total == 0: continue
        print(f"Relay {m}: Gom được {int(total):<4} ảnh -> Hist: [{int(h[0])}, {int(h[1])}, {int(h[2])}, {int(h[3])}]")
        
    print("\n[C] NHỮNG AUV ĐÃ BỊ EMD ÉP ĐỔI CỤM (Hand-over từ Round 0 -> Round 1):")
    changed = 0
    for i in range(n_auvs):
        old_m = physical_assoc.get(i, -1)
        new_m = emd_assoc.get(i, -1)
        if old_m != -1 and new_m != -1 and old_m != new_m:
            changed += 1
            z = topo_snapshot.auv_positions[i, 2]
            print(f"- AUV {i:<2} (Z={z:<6.1f}m): Từ Relay {old_m} chuyển sang Relay {new_m} để giúp cân bằng tri thức!")
    print(f"Tổng cộng có {changed}/{n_auvs} AUVs phải chuyển cụm.")
    
if __name__ == "__main__":
    main()
