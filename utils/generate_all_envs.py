"""
generate_all_envs.py
Sinh truoc toan bo file Topology va Data Partition cho thu nghiem.
Giai doan 5: Decoupled.
"""
import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.env_manager import EnvironmentManager
from config.settings import NetworkConfig, AcousticChannelConfig, FedKDLConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help="Khong chay gi ca, chi in ra")
    parser.add_argument('--n', type=int, help="Chi chay cho N nay (vd: 50)")
    parser.add_argument('--dataset', type=str, help="Chi chay dataset nay (vd: SMD)")
    args = parser.parse_args()

    N_LIST = [50, 100, 150, 200]
    DATASETS = ['SMD', 'SMAP', 'MSL', 'URPC']
    ALPHAS = [0.5, 10000.0]
    SEEDS = [42, 123, 2024]
    
    if args.n:
        N_LIST = [args.n]
    if args.dataset:
        DATASETS = [args.dataset]
        
    os.makedirs(EnvironmentManager.ENVS_DIR, exist_ok=True)
    
    net_cfg = NetworkConfig()
    ac_cfg = AcousticChannelConfig()
    fed_cfg = FedKDLConfig()
    
    print(f"Bắt đầu sinh file cấu hình cho Giai đoạn 5 (Decoupled)...")
    
    # 1. Sinh Topology
    topo_count = 0
    for n in N_LIST:
        net_cfg.N_SENSORS = n
        net_cfg.M_FOGS = max(5, n // 10)
        
        for seed in SEEDS:
            topo_path = EnvironmentManager.topo_path(n, seed)
            if not topo_path.exists():
                if not args.dry_run:
                    topo = EnvironmentManager.generate_topology(net_cfg, ac_cfg, seed)
                    EnvironmentManager.save_topology(topo)
                else:
                    print(f"  [dry-run] se sinh topo_N{n}_seed{seed}.pkl")
                topo_count += 1
            else:
                print(f"  [skip]    {topo_path.name}")
                
    # 2. Sinh Data Partition
    data_count = 0
    for n in N_LIST:
        net_cfg.N_SENSORS = n
        
        for ds in DATASETS:
            for alpha in ALPHAS:
                for seed in SEEDS:
                    data_path = EnvironmentManager.data_path(n, ds, alpha, seed)
                    if not data_path.exists():
                        if not args.dry_run:
                            if ds == 'URPC':
                                data_part = EnvironmentManager.generate_data_partition_2d(
                                    net_cfg, dataset_name=ds, alpha=alpha, seed=seed,
                                    base_yaml_path="datasets/URPC2020.yaml"
                                )
                            else:
                                data_part = EnvironmentManager.generate_data_partition(
                                    net_cfg, dataset_name=ds, alpha=alpha, seed=seed
                                )
                            EnvironmentManager.save_data_partition(data_part)
                        else:
                            print(f"  [dry-run] se sinh data_N{n}_{ds}_a{alpha}_seed{seed}.pkl")
                        data_count += 1
                    else:
                        print(f"  [skip]    {data_path.name}")
                        
    print(f"\n[Hoan thanh] Đã sinh {topo_count} Topologies và {data_count} Data Partitions.")

if __name__ == "__main__":
    main()
