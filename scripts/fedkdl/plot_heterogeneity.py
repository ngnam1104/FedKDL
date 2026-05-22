"""
plot_heterogeneity.py
Đọc logs JSON (results/logs_kdl) để vẽ biểu đồ độ nhạy với Dữ liệu Non-IID cho Kịch bản 3.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_heterogeneity():
    log_files = glob.glob("results/logs_kdl/*.json")
    if not log_files:
        print("[Warning] Không có file JSON nào trong results/logs_kdl.")
        return

    results = []

    # Map labels cho alpha
    alpha_map = {"0p1": "Strong non-IID\n(alpha=0.1)", "10000p0": "Near-IID\n(alpha=10000)"}

    # Chọn 1 scale cố định (VD: N=100)
    target_N = 100
    
    data_points = []
    
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        n = meta.get("N")
        alpha = meta.get("alpha")
        
        if n != target_N: continue
        
        metrics = data.get("metrics", {})
        
        map_val = metrics.get('mAP50-95', metrics.get('map'))
        e_cumul_val = metrics.get('e_cumul', metrics.get('energy_cumul_J'))
        
        if e_cumul_val and map_val:
            data_points.append({
                'alpha': alpha,
                'baseline': baseline,
                'map': map_val[-1],
                'energy': e_cumul_val[-1]
            })

    if not data_points:
        print(f"[Warning] Không tìm thấy dữ liệu heterogeneity cho N={target_N}")
        return

    os.makedirs("results/scenario3", exist_ok=True)
    
    alphas = ["0p1", "10000p0"]
    baselines = ["baseline_od", "fedkdl"]
    
    labels = [alpha_map[a] for a in alphas]
    
    x = np.arange(len(labels))
    width = 0.35
    
    colors = {"baseline_od": "orange", "fedkdl": "green"}
    names = {"baseline_od": "Bottleneck OD", "fedkdl": "FedKDL (Ours)"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, b in enumerate(baselines):
        c = colors.get(b, 'black')
        l = names.get(b, b)
        
        y_map = [next((d['map'] for d in data_points if d['baseline'] == b and d['alpha'] == a), 0) for a in alphas]
        y_energy = [next((d['energy'] for d in data_points if d['baseline'] == b and d['alpha'] == a), 0) for a in alphas]

        axes[0].bar(x + (i - 0.5) * width, y_map, width, label=l, color=c)
        axes[1].bar(x + (i - 0.5) * width, y_energy, width, label=l, color=c)

    axes[0].set_title('Detection Quality under Data Heterogeneity')
    axes[0].set_ylabel('mAP@0.5:0.95')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].set_title('Total Communication Energy')
    axes[1].set_ylabel('Energy (Joules) - Log scale')
    axes[1].set_yscale('log')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = "results/scenario3/fedkdl_heterogeneity.png"
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Lưu biểu đồ tại {save_path}")

if __name__ == "__main__":
    plot_heterogeneity()
