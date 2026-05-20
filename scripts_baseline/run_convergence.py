"""
run_convergence.py
Thí nghiệm: Hội tụ của Loss (Convergence Behaviour - Figure 4)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from hfl_core.simulator import Scenario1Simulator

def run_convergence(dry_run=False):
    N_list = [150, 200]
    baselines = ['fedavg', 'fedprox', 'hfl_nocoop', 'hfl_selective', 'hfl_nearest']
    T_rounds = 2 if dry_run else 20
    SEEDS = [42] if dry_run else [42, 123, 2024]
    
    os.makedirs('results/convergence', exist_ok=True)
    
    all_loss_data = {}
    
    for n in N_list:
        network_cfg.N_SENSORS = n
        network_cfg.M_FOGS = max(5, n // 10)
        
        all_loss_data[n] = {}
        
        for baseline in baselines:
            loss_history_seeds = []
            
            for seed in SEEDS:
                print(f"\n=== Chạy Convergence N={n}, Baseline={baseline}, Seed={seed} ===")
                sim = Scenario1Simulator(
                    net_cfg=network_cfg,
                    ac_cfg=acoustic_cfg,
                    en_cfg=energy_cfg,
                    fed_cfg=fed_cfg,
                    baseline=baseline,
                    seed=seed
                )
                
                metrics_df, _, _ = sim.run(T_rounds=T_rounds)
                
                if not metrics_df.empty and 'Train_Loss' in metrics_df.columns:
                    loss_history_seeds.append(metrics_df['Train_Loss'].values)
            
            if loss_history_seeds:
                min_len = min(len(h) for h in loss_history_seeds)
                loss_matrix = np.array([h[:min_len] for h in loss_history_seeds])
                mean_loss = np.mean(loss_matrix, axis=0)
                std_loss = np.std(loss_matrix, axis=0)
                
                all_loss_data[n][baseline] = {
                    'mean': mean_loss,
                    'std': std_loss,
                    'rounds': np.arange(min_len)
                }

    plot_convergence(all_loss_data)

def plot_convergence(all_loss_data):
    style_map = {
        'fedavg': ('pink', 'FedAvg'),
        'fedprox': ('red', 'FedProx'),
        'hfl_nocoop': ('green', 'HFL-NoCoop'),
        'hfl_selective': ('blue', 'HFL-Selective'),
        'hfl_nearest': ('orange', 'HFL-Nearest')
    }
    
    N_list = list(all_loss_data.keys())
    if not N_list:
        return
        
    fig, axes = plt.subplots(1, len(N_list), figsize=(6 * len(N_list), 5))
    if len(N_list) == 1:
        axes = [axes]
        
    for idx, n in enumerate(N_list):
        ax = axes[idx]
        data_n = all_loss_data[n]
        
        for baseline, stats in data_n.items():
            c, l = style_map.get(baseline, ('black', baseline))
            rounds = stats['rounds']
            mean_l = stats['mean']
            std_l = stats['std']
            
            ax.plot(rounds, mean_l, label=l, color=c, linewidth=2)
            ax.fill_between(rounds, np.maximum(0, mean_l - std_l), mean_l + std_l, color=c, alpha=0.2)
            
        ax.set_title(f'Convergence Behaviour (N={n})')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Training Loss (MSE)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    save_path = 'results/convergence/fig4_convergence.png'
    plt.savefig(save_path)
    print(f"\nĐã lưu biểu đồ: {save_path}")

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    run_convergence(dry_run)
