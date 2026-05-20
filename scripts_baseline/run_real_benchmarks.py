"""
run_real_benchmarks.py
Thí nghiệm 3: Dữ liệu Thực tế (Real Benchmarks).
Tái hiện Figure 8 và Table IV của Omeke et al. 2026.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from hfl_core.simulator import Scenario1Simulator

def run_real_benchmarks(dry_run=False):
    datasets = ['SMD', 'SMAP', 'MSL']
    baselines = ['centralised', 'fedavg', 'fedprox', 'hfl_nocoop', 'hfl_selective', 'hfl_nearest']
    T_rounds = 2 if dry_run else 30  # T=30 cho real benchmarks theo paper
    SEEDS = [42] if dry_run else [42, 123, 2024]  # 3 seeds như trong paper
    
    # Cấu hình mặc định cho thí nghiệm benchmark
    network_cfg.N_SENSORS = 100
    network_cfg.M_FOGS = 10
    
    results = []
    os.makedirs('results/real_benchmarks', exist_ok=True)
    
    for dataset in datasets:
        fed_cfg.DATASETS_1D = [dataset]
        
        for baseline in baselines:
            seed_metrics = []

            for seed in SEEDS:
                print(f"\n=====================================")
                print(f"Chạy Benchmark Dataset={dataset}, Baseline={baseline}, Seed={seed}")
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
                        'Tau_Round_s': last['Tau_Round_s'],
                    })

            if seed_metrics:
                paf1_vals = [s['PA-F1'] for s in seed_metrics]
                eng_vals  = [s['Total_Energy'] for s in seed_metrics]
                tau_vals  = [s['Tau_Round_s'] for s in seed_metrics]
                results.append({
                    'Dataset': dataset,
                    'Baseline': baseline,
                    'PA-F1_mean':  float(np.mean(paf1_vals)),
                    'PA-F1_std':   float(np.std(paf1_vals)),
                    'Energy_mean': float(np.mean(eng_vals)),
                    'Energy_std':  float(np.std(eng_vals)),
                    'Tau_Round_mean_s': float(np.mean(tau_vals)),
                })
                
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/real_benchmarks/summary.csv', index=False)
    print("\nKết quả tổng hợp (Real Benchmarks):")
    print(df_results.to_string())
    
    plot_real_benchmarks(df_results)

def plot_real_benchmarks(df):
    datasets = df['Dataset'].unique()
    baselines = df['Baseline'].unique()
    
    x = np.arange(len(datasets))
    width = 0.12
    
    style_map = {
        'centralised': ('grey', 'Centralised (Oracle)'),
        'fedavg': ('pink', 'FedAvg'),
        'fedprox': ('red', 'FedProx'),
        'hfl_nocoop': ('green', 'HFL-NoCoop'),
        'hfl_selective': ('blue', 'HFL-Selective'),
        'hfl_nearest': ('orange', 'HFL-Nearest')
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. PA-F1 (with error bars = ±std)
    for i, b in enumerate(baselines):
        c, l = style_map.get(b, ('black', b))
        sub = df[df['Baseline'] == b]
        y    = sub['PA-F1_mean'].values
        yerr = sub['PA-F1_std'].values
        if len(y) < len(x):
            y    = np.pad(y,    (0, len(x)-len(y)))
            yerr = np.pad(yerr, (0, len(x)-len(yerr)))
        axes[0].bar(x + (i - 2.5) * width, y, width, label=l, color=c,
                    yerr=yerr, capsize=3, error_kw={'elinewidth': 1.5})
        
    axes[0].set_title('Detection Quality Across Real Benchmarks (PA-F1)')
    axes[0].set_ylabel('PA-F1 Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    
    min_f1 = df['PA-F1_mean'].min()
    max_f1 = df['PA-F1_mean'].max()
    padding = (max_f1 - min_f1) * 0.5 if max_f1 > min_f1 else 0.05
    axes[0].set_ylim(max(0, min_f1 - padding), min(1.05, max_f1 + padding))
    
    axes[0].legend(loc='lower right')
    
    # 2. Total Energy (Log Scale, with error bars)
    for i, b in enumerate(baselines):
        if b == 'centralised':
            continue  # Tránh vẽ energy=0 trên thang log
        c, l = style_map.get(b, ('black', b))
        sub = df[df['Baseline'] == b]
        y    = sub['Energy_mean'].values
        yerr = sub['Energy_std'].values
        if len(y) < len(x):
            y    = np.pad(y,    (0, len(x)-len(y)))
            yerr = np.pad(yerr, (0, len(x)-len(yerr)))
        # Thay 0 bằng 1e-1 để vẽ log
        y = np.where(y == 0, 1e-1, y)
        axes[1].bar(x + (i - 2.5) * width, y, width, label=l, color=c,
                    yerr=yerr, capsize=3, error_kw={'elinewidth': 1.5})
        
    axes[1].set_title('Communication Cost Across Real Benchmarks')
    axes[1].set_ylabel('Energy (Joules) - Log scale')
    axes[1].set_yscale('log')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    
    plt.tight_layout()
    save_path = 'results/real_benchmarks/fig8_real_benchmarks.png'
    plt.savefig(save_path)
    print(f"\nĐã lưu biểu đồ: {save_path}")

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    run_real_benchmarks(dry_run)
