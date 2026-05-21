"""
plot_od_scalability.py
Đọc logs JSON (results/logs_kdl) để vẽ biểu đồ khả năng mở rộng cho Kịch bản 3.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import glob
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_scalability():
    log_files = glob.glob("results/logs_kdl/*.json")
    if not log_files:
        print("[Warning] Không có file JSON nào trong results/logs_kdl.")
        return

    energy_data = defaultdict(lambda: defaultdict(list))
    alive_data = defaultdict(lambda: defaultdict(list))
    map_data = defaultdict(lambda: defaultdict(list))

    # Cố định alpha để đánh giá scalability
    target_alpha = "0p1"
    
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        n = meta.get("N")
        alpha = meta.get("alpha")
        
        if alpha != target_alpha: continue
        
        metrics = data.get("metrics", {})
        
        if 'energy_cumul_J' in metrics and metrics['energy_cumul_J']:
            energy_data[baseline][n].append(metrics['energy_cumul_J'][-1])
        if 'alive' in metrics and metrics['alive']:
            alive_data[baseline][n].append(metrics['alive'][-1] / max(1, n)) # Participation fraction
        if 'map' in metrics and metrics['map']:
            map_data[baseline][n].append(metrics['map'][-1])

    if not energy_data:
        print(f"[Warning] Không tìm thấy dữ liệu scalability cho alpha={target_alpha}")
        return

    os.makedirs("results/scenario3", exist_ok=True)
    
    N_sorted = sorted({n for b in energy_data for n in energy_data[b]})

    colors = {"baseline_od": "orange", "fedkdl": "green"}
    labels = {"baseline_od": "Bottleneck OD", "fedkdl": "FedKDL (Ours)"}
    markers = {"baseline_od": "s", "fedkdl": "o"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for baseline in energy_data.keys():
        e_means = [np.mean(energy_data[baseline][n]) if energy_data[baseline][n] else np.nan for n in N_sorted]
        a_means = [np.mean(alive_data[baseline][n]) if alive_data[baseline][n] else np.nan for n in N_sorted]
        m_means = [np.mean(map_data[baseline][n]) if map_data[baseline][n] else np.nan for n in N_sorted]
        
        c = colors.get(baseline, 'black')
        l = labels.get(baseline, baseline)
        m = markers.get(baseline, 'o')

        axes[0].plot(N_sorted, m_means, label=l, color=c, marker=m, linewidth=2, markersize=8)
        axes[1].plot(N_sorted, a_means, label=l, color=c, marker=m, linewidth=2, markersize=8)
        axes[2].plot(N_sorted, e_means, label=l, color=c, marker=m, linewidth=2, markersize=8)

    axes[0].set_title("Final mAP vs Scale")
    axes[0].set_xlabel("Number of AUVs (N)")
    axes[0].set_ylabel("mAP@0.5:0.95")
    
    axes[1].set_title("Survival Fraction vs Scale")
    axes[1].set_xlabel("Number of AUVs (N)")
    axes[1].set_ylabel("Survival Rate")
    
    axes[2].set_title("Total Energy vs Scale")
    axes[2].set_xlabel("Number of AUVs (N)")
    axes[2].set_ylabel("Energy (Joules)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(N_list := N_sorted)
        ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = "results/scenario3/fedkdl_scalability.png"
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Lưu biểu đồ tại {save_path}")

if __name__ == "__main__":
    plot_scalability()
