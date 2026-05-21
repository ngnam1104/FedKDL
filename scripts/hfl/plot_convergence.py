"""
plot_convergence.py
Đọc logs JSON để vẽ biểu đồ hội tụ (Figure 4).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.plot_styles import setup_global_plot_style, get_style

def plot_convergence():
    setup_global_plot_style()
    N_list = [150, 200]
    os.makedirs("results/convergence", exist_ok=True)
    all_loss_data = defaultdict(lambda: defaultdict(list))
    
    log_files = glob.glob("results/logs/*.json")
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        meta = data.get("metadata", {})
        n = meta.get("N")
        if n not in N_list: continue
        baseline = meta.get("baseline")
        if meta.get("rho_s") != 0.05: continue
            
        metrics = data.get("metrics", {})
        losses = metrics.get("Train_Loss") or metrics.get("loss") or []
        if losses:
            all_loss_data[n][baseline].append(losses)
            
    if not all_loss_data:
        print("[Warning] Không tìm thấy dữ liệu hội tụ.")
        return

    fig, axes = plt.subplots(1, len(N_list), figsize=(6 * len(N_list), 5))
    if len(N_list) == 1: axes = [axes]
        
    for idx, n in enumerate(N_list):
        ax = axes[idx]
        data_n = all_loss_data[n]
        
        for baseline, loss_list in data_n.items():
            if not loss_list: continue
                
            min_len = min(len(h) for h in loss_list)
            loss_matrix = np.array([h[:min_len] for h in loss_list])
            mean_loss = np.mean(loss_matrix, axis=0)
            std_loss = np.std(loss_matrix, axis=0)
            rounds = np.arange(min_len)
            c, m, l = get_style(baseline)
            ax.plot(rounds, mean_loss, label=l, color=c, marker=m, linewidth=2, markevery=5)
            ax.fill_between(rounds, np.maximum(0, mean_loss - std_loss), mean_loss + std_loss, color=c, alpha=0.2)
            
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
    plot_convergence()
