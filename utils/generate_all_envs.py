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
from config.settings import NetworkConfig, AcousticChannelConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help="Khong chay gi ca, chi in ra")
    parser.add_argument('--n', type=int, help="Chi chay cho N nay (vd: 50)")
    parser.add_argument('--dataset', type=str, help="Chi chay dataset nay (vd: SMD)")
    parser.add_argument('--m-relays', type=int, help="Override so luong relay nodes khi sinh topology")
    parser.add_argument('--force-topo', action='store_true', help="Ghi de topology da ton tai")
    parser.add_argument(
        '--find-gateway-seed',
        action='store_true',
        help="Tim seed dau tien ma tat ca relay co uplink kha thi toi gateway",
    )
    parser.add_argument(
        '--seed-search-limit',
        type=int,
        default=10000,
        help="So seed toi da can thu khi dung --find-gateway-seed",
    )
    parser.add_argument('--alphas', nargs='+', type=float, help="Danh sach cac gia tri alpha, vi du: --alphas 0.5 1.0")
    parser.add_argument('--seeds', nargs='+', type=int, help="Danh sach random seed")
    args = parser.parse_args()

    # Chỉ chạy 2D (URPC) giống cấu hình trong run_kdl_experiments.sh
    # if args.dataset == 'URPC':
    DATASETS = ['URPC']
    N_LIST = [50]
    ALPHAS = args.alphas if args.alphas is not None else [0.5]
    SEEDS = args.seeds if args.seeds is not None else [1104]
    task_type = '2d'
    # else:
    #     DATASETS = ['SMD', 'SMAP', 'MSL'] if not args.dataset else [args.dataset]
    #     N_LIST = [50, 100, 150, 200]
    #     ALPHAS = [1.0, 10000.0]
    #     SEEDS = [42, 123, 2024]
    #     task_type = '1d'

    if args.n:
        N_LIST = [args.n]
        
    os.makedirs(EnvironmentManager.ENVS_DIR, exist_ok=True)
    
    net_cfg = NetworkConfig()
    ac_cfg = AcousticChannelConfig()

    if args.find_gateway_seed:
        from physics_models.topology import (
            Topology3D,
            build_feasibility_graph,
            gateway_disconnected_relays,
        )

        net_cfg.N_AUVS = N_LIST[0]
        net_cfg.M_RELAYS = (
            args.m_relays
            if args.m_relays is not None
            else net_cfg.M_RELAYS_2D
        )
        start_seed = SEEDS[0]
        found_seed = None
        for candidate in range(start_seed, start_seed + args.seed_search_limit):
            candidate_topo = Topology3D(net_cfg, ac_cfg, candidate)
            candidate_graph = build_feasibility_graph(candidate_topo, ac_cfg)
            if not gateway_disconnected_relays(candidate_topo, candidate_graph):
                found_seed = candidate
                break
        if found_seed is None:
            raise RuntimeError(
                f"No fully gateway-connected topology found in "
                f"[{start_seed}, {start_seed + args.seed_search_limit})"
            )
        SEEDS = [found_seed]
        print(f"  [seed] All {net_cfg.M_RELAYS} relays reach gateway: {found_seed}")
    
    print(f"Bắt đầu sinh file cấu hình cho Giai đoạn 5 (Decoupled)...")
    
    # 1. Sinh Topology
    topo_count = 0
    for n in N_LIST:
        net_cfg.N_AUVS = n
        
        # Relay count: CLI --m-relays, else M_RELAYS_2D (URPC) / M_RELAYS_1D
        if args.m_relays is not None:
            net_cfg.M_RELAYS = args.m_relays
        elif len(DATASETS) == 1 and DATASETS[0] == 'URPC':
            net_cfg.M_RELAYS = net_cfg.M_RELAYS_2D
        else:
            net_cfg.M_RELAYS = net_cfg.M_RELAYS_1D
        print(f"  [topology] N={n} -> M_RELAYS={net_cfg.M_RELAYS}")
        
        for seed in SEEDS:
            topo_path = EnvironmentManager.topo_path(task_type, n, seed)
            topology_mismatch = False
            if topo_path.exists() and not args.force_topo:
                existing_topo = EnvironmentManager.load_topology(topo_path)
                topology_mismatch = (
                    existing_topo.N != net_cfg.N_AUVS
                    or existing_topo.M != net_cfg.M_RELAYS
                )
                if topology_mismatch:
                    print(
                        f"  [stale]   {topo_path.name}: "
                        f"N={existing_topo.N}, M={existing_topo.M}; "
                        f"expected N={net_cfg.N_AUVS}, M={net_cfg.M_RELAYS}"
                    )

            if args.force_topo or not topo_path.exists() or topology_mismatch:
                if not args.dry_run:
                    topo = EnvironmentManager.generate_topology(net_cfg, ac_cfg, seed)
                    EnvironmentManager.save_topology(topo, task_type)
                else:
                    print(f"  [dry-run] se sinh topo_N{n}_seed{seed}.pkl")
                topo_count += 1
            else:
                print(f"  [skip]    {topo_path.name}")
                
    # 2. Sinh Data Partition
    data_count = 0
    for n in N_LIST:
        net_cfg.N_AUVS = n
        
        for ds in DATASETS:
            for alpha in ALPHAS:
                for seed in SEEDS:
                    data_path = EnvironmentManager.data_path(task_type, n, ds, alpha, seed)
                    if not data_path.exists():
                        if not args.dry_run:
                            if ds == 'URPC':
                                # Bắt buộc phải load topo trước để biết Depth Z
                                topo_path_for_data = EnvironmentManager.topo_path(task_type, n, seed)
                                if not topo_path_for_data.exists():
                                    print(f"  [error] Không tìm thấy topo: {topo_path_for_data.name}. Chạy --force-topo trước!")
                                    continue
                                topo = EnvironmentManager.load_topology(topo_path_for_data)

                                data_part = EnvironmentManager.generate_data_partition_2d(
                                    net_cfg, topo=topo, dataset_name=ds, alpha=alpha, seed=seed,
                                    base_yaml_path="datasets/URPC2020.yaml"
                                )
                            else:
                                print(f"  [error] Dataset {ds} không được hỗ trợ (chỉ hỗ trợ URPC).")
                                continue
                            EnvironmentManager.save_data_partition(data_part, task_type)
                        else:
                            print(f"  [dry-run] se sinh data_N{n}_{ds}_a{alpha}_seed{seed}.pkl")
                        data_count += 1
                    else:
                        print(f"  [skip]    {data_path.name}")
                        
    print(f"\n[Hoan thanh] Đã sinh {topo_count} Topologies và {data_count} Data Partitions.")

if __name__ == "__main__":
    main()
