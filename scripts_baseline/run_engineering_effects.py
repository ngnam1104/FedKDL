"""
run_engineering_effects.py
Thí nghiệm: Hiệu quả Kỹ thuật (Engineering Effects - Figure 6)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from hfl_core.simulator import Scenario1Simulator

def run_engineering_effects(dry_run=False):
    N_list = [150, 200]
    baselines = ['fedavg', 'fedprox', 'hfl_nocoop', 'hfl_selective', 'hfl_nearest']
    T_rounds = 2 if dry_run else 20
    SEEDS = [42] if dry_run else [42, 123, 2024]
    
    os.makedirs('results/engineering', exist_ok=True)
    
    results = []
    
    # Compress mode: rho_s = 0.05 (Default)
    # Uncompress mode: rho_s = 1.0
    
    for n in N_list:
        network_cfg.N_SENSORS = n
        network_cfg.M_FOGS = max(5, n // 10)
        
        for baseline in baselines:
            for is_compressed in [True, False]:
                comp_ratio = 0.05 if is_compressed else 1.0
                fed_cfg.RHO_S = comp_ratio
                
                eng_vals = []
                f1_vals = []
                
                for seed in SEEDS:
                    print(f"\n=== Chạy Engineering N={n}, Baseline={baseline}, Compress={is_compressed}, Seed={seed} ===")
                    sim = Scenario1Simulator(
                        net_cfg=network_cfg,
                        ac_cfg=acoustic_cfg,
                        en_cfg=energy_cfg,
                        fed_cfg=fed_cfg,
                        baseline=baseline,
                        seed=seed
                    )
                    
                    metrics_df, _, _ = sim.run(T_rounds=T_rounds)
                    
                    if not metrics_df.empty:
                        last = metrics_df.iloc[-1]
                        eng_vals.append(last['Cumul_Energy'])
                        f1_vals.append(last['PA-F1'])
                        
                if eng_vals:
                    results.append({
                        'N': n,
                        'Baseline': baseline,
                        'Compressed': is_compressed,
                        'Energy_mean': np.mean(eng_vals),
                        'PA-F1_mean': np.mean(f1_vals),
                    })
                    
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/engineering/summary.csv', index=False)
    
    # Restore default
    fed_cfg.RHO_S = 0.05
    plot_engineering_effects(df_results)

def plot_engineering_effects(df):
    N_list = [150, 200]
    hfl_baselines = ['hfl_nocoop', 'hfl_selective', 'hfl_nearest']
    
    style_map = {
        'hfl_nocoop': 'green',
        'hfl_selective': 'blue',
        'hfl_nearest': 'orange'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Fig 6a: Total Energy of HFL protocols (Compressed) with F1 on top
    width = 0.25
    x = np.arange(len(N_list))
    
    for i, b in enumerate(hfl_baselines):
        c = style_map.get(b, 'black')
        sub = df[(df['Baseline'] == b) & (df['Compressed'] == True) & (df['N'].isin(N_list))]
        if sub.empty: continue
        
        y = sub['Energy_mean'].values
        f1 = sub['PA-F1_mean'].values
        
        bars = axes[0].bar(x + (i - 1) * width, y, width, label=b, color=c)
        
        # Add F1 text on top
        for bar, f1_score in zip(bars, f1):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                         f'{f1_score:.2f}', ha='center', va='bottom', fontsize=9)
                         
    axes[0].set_title('Total Energy vs Scale (with F1 scores)')
    axes[0].set_ylabel('Total Energy (Joules)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'N={n}' for n in N_list])
    axes[0].legend()
    
    # Fig 6b: Energy Savings (%) 
    all_baselines = ['fedavg', 'fedprox', 'hfl_nocoop', 'hfl_selective', 'hfl_nearest']
    savings_data = {b: [] for b in all_baselines}
    
    for b in all_baselines:
        for n in N_list:
            sub_comp = df[(df['Baseline'] == b) & (df['Compressed'] == True) & (df['N'] == n)]
            sub_uncomp = df[(df['Baseline'] == b) & (df['Compressed'] == False) & (df['N'] == n)]
            
            if not sub_comp.empty and not sub_uncomp.empty:
                e_comp = sub_comp['Energy_mean'].values[0]
                e_uncomp = sub_uncomp['Energy_mean'].values[0]
                saving = max(0, (e_uncomp - e_comp) / max(1e-6, e_uncomp) * 100)
                savings_data[b].append(saving)
            else:
                savings_data[b].append(0.0)
                
    x_save = np.arange(len(N_list))
    width_s = 0.15
    for i, b in enumerate(all_baselines):
        axes[1].bar(x_save + (i - 2) * width_s, savings_data[b], width_s, label=b)
        
    axes[1].set_title('Energy Savings via Compression (%)')
    axes[1].set_ylabel('Energy Saving (%)')
    axes[1].set_xticks(x_save)
    axes[1].set_xticklabels([f'N={n}' for n in N_list])
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    
    plt.tight_layout()
    save_path = 'results/engineering/fig6_engineering.png'
    plt.savefig(save_path)
    print(f"\nĐã lưu biểu đồ: {save_path}")

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    run_engineering_effects(dry_run)
