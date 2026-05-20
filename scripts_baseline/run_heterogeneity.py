"""
run_heterogeneity.py
Thí nghiệm 2: Độ nhạy với Dữ liệu Non-IID (Heterogeneity Study).
Tái hiện Figure 7 của Omeke et al. 2026.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from hfl_core.simulator import Scenario1Simulator

def run_heterogeneity(dry_run=False):
    alpha_list = [0.1, 10000.0]  # Strongly Non-IID vs Near-IID (paper Fig. 7)
    alpha_labels = ['Strong non-IID\n(alpha=0.1)', 'Near-IID\n(alpha=10000)']
    baselines = ['fedprox', 'hfl_nocoop', 'hfl_selective', 'hfl_nearest']
    T_rounds = 2 if dry_run else 20
    SEEDS = [42] if dry_run else [42, 123, 2024]  # 3 seeds như trong paper
    
    network_cfg.N_SENSORS = 100
    network_cfg.M_FOGS = 10
    
    results = []
    os.makedirs('results/heterogeneity', exist_ok=True)
    
    for alpha, label in zip(alpha_list, alpha_labels):
        fed_cfg.NON_IID_ALPHA = alpha
        
        for baseline in baselines:
            seed_metrics = []

            for seed in SEEDS:
                print(f"\n=====================================")
                print(f"Chạy Heterogeneity alpha={alpha}, Baseline={baseline}, Seed={seed}")
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
                        'Total_Energy': last['Cumul_Energy'],
                    })

            if seed_metrics:
                paf1_vals = [s['PA-F1'] for s in seed_metrics]
                eng_vals  = [s['Total_Energy'] for s in seed_metrics]
                results.append({
                    'Alpha_Label': label,
                    'Baseline': baseline,
                    'PA-F1_mean':  float(np.mean(paf1_vals)),
                    'PA-F1_std':   float(np.std(paf1_vals)),
                    'Energy_mean': float(np.mean(eng_vals)),
                    'Energy_std':  float(np.std(eng_vals)),
                })
                
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/heterogeneity/summary.csv', index=False)
    print("\nKết quả tổng hợp (Heterogeneity):")
    print(df_results.to_string())
    
    plot_heterogeneity(df_results)

def plot_heterogeneity(df):
    labels = df['Alpha_Label'].unique()
    baselines = df['Baseline'].unique()
    
    x = np.arange(len(labels))
    width = 0.2
    
    style_map = {
        'fedprox': ('red', 'FedProx'),
        'hfl_nocoop': ('green', 'HFL-NoCoop'),
        'hfl_selective': ('blue', 'HFL-Selective'),
        'hfl_nearest': ('orange', 'HFL-Nearest')
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. PA-F1 (with error bars = ±std)
    for i, b in enumerate(baselines):
        c, l = style_map.get(b, ('black', b))
        sub = df[df['Baseline'] == b]
        y   = sub['PA-F1_mean'].values
        yerr = sub['PA-F1_std'].values
        if len(y) < len(x):
            y    = np.pad(y,    (0, len(x)-len(y)))
            yerr = np.pad(yerr, (0, len(x)-len(yerr)))
        axes[0].bar(x + (i - 1.5) * width, y, width, label=l, color=c,
                    yerr=yerr, capsize=4, error_kw={'elinewidth': 1.5})
        
    axes[0].set_title('Detection Quality under Data Heterogeneity')
    axes[0].set_ylabel('PA-F1 Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    
    # 2. Total Energy (Log Scale, with error bars)
    for i, b in enumerate(baselines):
        c, l = style_map.get(b, ('black', b))
        sub = df[df['Baseline'] == b]
        y    = sub['Energy_mean'].values
        yerr = sub['Energy_std'].values
        if len(y) < len(x):
            y    = np.pad(y,    (0, len(x)-len(y)))
            yerr = np.pad(yerr, (0, len(x)-len(yerr)))
        axes[1].bar(x + (i - 1.5) * width, y, width, label=l, color=c,
                    yerr=yerr, capsize=4, error_kw={'elinewidth': 1.5})
        
    axes[1].set_title('Total Communication Energy')
    axes[1].set_ylabel('Energy (Joules) - Log scale')
    axes[1].set_yscale('log')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    
    plt.tight_layout()
    save_path = 'results/heterogeneity/fig7_heterogeneity.png'
    plt.savefig(save_path)
    print(f"\nĐã lưu biểu đồ: {save_path}")

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    run_heterogeneity(dry_run)
