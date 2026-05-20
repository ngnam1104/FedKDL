"""
run_scalability.py
Thí nghiệm 1: Khả năng mở rộng dưới ràng buộc độ phủ âm thanh (Scalability Study).
Tái hiện Figure 5 và Table III của Omeke et al. 2026.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Thêm thư mục gốc vào sys.path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from hfl_core.simulator import Scenario1Simulator

def run_scalability(dry_run=False):
    N_list = [50, 100, 150, 200]
    baselines = ['fedprox', 'hfl_nocoop', 'hfl_selective', 'hfl_nearest']
    T_rounds = 2 if dry_run else 20
    SEEDS = [42] if dry_run else [42, 123, 2024]  # 3 seeds như trong paper
    
    results = []
    
    os.makedirs('results/scalability', exist_ok=True)
    
    for n in N_list:
        # Cập nhật cấu hình
        network_cfg.N_SENSORS = n
        network_cfg.M_FOGS = max(5, n // 10)  # M = N/10
        
        for baseline in baselines:
            seed_metrics = []

            for seed in SEEDS:
                print(f"\n=====================================")
                print(f"Chạy Scale N={n}, Baseline={baseline}, Seed={seed}")
                print(f"=====================================")
                
                sim = Scenario1Simulator(
                    net_cfg=network_cfg,
                    ac_cfg=acoustic_cfg,
                    en_cfg=energy_cfg,
                    fed_cfg=fed_cfg,
                    baseline=baseline,
                    seed=seed
                )
                
                metrics_df, energy_df, latency_df = sim.run(T_rounds=T_rounds)
                
                if not metrics_df.empty:
                    last = metrics_df.iloc[-1]
                    seed_metrics.append({
                        'PA-F1': last['PA-F1'],
                        'Participation': last['Participation'],
                        'Total_Energy': last['Cumul_Energy'],
                        'Tau_Round_s': last['Tau_Round_s'],
                    })

            if seed_metrics:
                paf1_vals = [s['PA-F1'] for s in seed_metrics]
                part_vals = [s['Participation'] for s in seed_metrics]
                eng_vals  = [s['Total_Energy'] for s in seed_metrics]
                tau_vals  = [s['Tau_Round_s'] for s in seed_metrics]
                results.append({
                    'N': n,
                    'Baseline': baseline,
                    'PA-F1_mean':   float(np.mean(paf1_vals)),
                    'PA-F1_std':    float(np.std(paf1_vals)),
                    'Participation': float(np.mean(part_vals)),
                    'Energy_mean':  float(np.mean(eng_vals)),
                    'Energy_std':   float(np.std(eng_vals)),
                    'Energy_Per_Sensor': float(np.mean(eng_vals)) / (n * max(np.mean(part_vals), 1e-6)),
                    'Tau_Round_mean_s': float(np.mean(tau_vals)),
                })
                
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/scalability/summary.csv', index=False)
    print("\nKết quả tổng hợp (Scalability):")
    print(df_results.to_string())
    
    plot_scalability(df_results)

def plot_scalability(df):
    N_list = df['N'].unique()
    baselines = df['Baseline'].unique()
    
    # Map colors/markers
    style_map = {
        'fedprox': ('red', 's', 'FedProx'),
        'hfl_nocoop': ('green', '^', 'HFL-NoCoop'),
        'hfl_selective': ('blue', 'o', 'HFL-Selective'),
        'hfl_nearest': ('orange', 'd', 'HFL-Nearest')
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Participation vs N
    for b in baselines:
        sub = df[df['Baseline'] == b]
        c, m, l = style_map.get(b, ('black', 'o', b))
        axes[0].plot(sub['N'], sub['Participation'], marker=m, color=c, label=l, linewidth=2)
    axes[0].set_title('Reachability (Participation) vs Scale')
    axes[0].set_xlabel('Number of Sensors (N)')
    axes[0].set_ylabel('Participation Fraction')
    axes[0].set_xticks(N_list)
    axes[0].grid(True)
    axes[0].legend()
    
    # 2. PA-F1 vs N (with error bars = ±std)
    for b in baselines:
        sub = df[df['Baseline'] == b]
        c, m, l = style_map.get(b, ('black', 'o', b))
        axes[1].errorbar(sub['N'], sub['PA-F1_mean'], yerr=sub['PA-F1_std'],
                         marker=m, color=c, label=l, linewidth=2, capsize=4)
    axes[1].set_title('Detection Quality (PA-F1) vs Scale')
    axes[1].set_xlabel('Number of Sensors (N)')
    axes[1].set_ylabel('PA-F1 Score')
    axes[1].set_xticks(N_list)
    axes[1].grid(True)
    
    # 3. Energy per sensor vs N (with error bars)
    for b in baselines:
        sub = df[df['Baseline'] == b]
        c, m, l = style_map.get(b, ('black', 'o', b))
        axes[2].errorbar(sub['N'], sub['Energy_Per_Sensor'],
                         yerr=sub.get('Energy_std', pd.Series([0]*len(sub))).values / sub['N'],
                         marker=m, color=c, label=l, linewidth=2, capsize=4)
    axes[2].set_title('Per-Sensor Energy vs Scale')
    axes[2].set_xlabel('Number of Sensors (N)')
    axes[2].set_ylabel('Energy (Joules)')
    axes[2].set_xticks(N_list)
    axes[2].grid(True)
    
    plt.tight_layout()
    save_path = 'results/scalability/fig5_scalability.png'
    plt.savefig(save_path)
    print(f"\nĐã lưu biểu đồ: {save_path}")

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    run_scalability(dry_run)
