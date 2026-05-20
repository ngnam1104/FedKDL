"""
main_train.py
CLI Entry Point cho hệ thống FedKDL.
Hỗ trợ chạy kịch bản 1 (Baseline) với các arg.
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from fl_core.simulator import Scenario1Simulator

def parse_args():
    parser = argparse.ArgumentParser(description="FedKDL Simulator")
    parser.add_argument('--scenario', type=int, default=1, help='Kịch bản mô phỏng (1, 2, 3)')
    parser.add_argument('--baseline', type=str, default='hfl_selective', choices=['hfl_selective', 'fedprox'], help='Baseline algorithm')
    parser.add_argument('--rounds', type=int, default=5, help='Số vòng lặp (T)')
    parser.add_argument('--dataset', type=str, default='SMD', choices=['SMD', 'SMAP', 'MSL'], help='Dataset 1D')
    return parser.parse_args()

def plot_results(metrics_df: pd.DataFrame, title_prefix: str):
    """Vẽ biểu đồ F1 và Participation sau khi chạy xong."""
    rounds = metrics_df['round']
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rounds, metrics_df['PA-F1'], marker='o', color='blue')
    plt.title(f'{title_prefix} - PA-F1 Score')
    plt.xlabel('Round')
    plt.ylabel('PA-F1')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(rounds, metrics_df['Participation'] * 100, marker='x', color='red')
    plt.title(f'{title_prefix} - Network Participation (%)')
    plt.xlabel('Round')
    plt.ylabel('Active Sensors (%)')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    save_path = f"results/{title_prefix.lower().replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ tại: {save_path}")

def main():
    args = parse_args()
    
    if args.scenario == 1:
        fed_cfg.DATASETS_1D = [args.dataset]
        sim = Scenario1Simulator(
            net_cfg=network_cfg,
            ac_cfg=acoustic_cfg,
            en_cfg=energy_cfg,
            fed_cfg=fed_cfg,
            baseline=args.baseline
        )
        
        metrics_df, energy_df, latency_df = sim.run(T_rounds=args.rounds)
        
        # Save CSV
        os.makedirs('results', exist_ok=True)
        prefix = f"scen1_{args.baseline}_{args.dataset}"
        metrics_df.to_csv(f"results/{prefix}_metrics.csv", index=False)
        energy_df.to_csv(f"results/{prefix}_energy.csv", index=False)
        latency_df.to_csv(f"results/{prefix}_latency.csv", index=False)
        
        # Plot
        plot_results(metrics_df, title_prefix=f"Scenario 1 - {args.baseline.upper()}")
        
    else:
        print(f"Kịch bản {args.scenario} chưa được triển khai đầy đủ.")

if __name__ == '__main__':
    main()
