"""
plot_od_comparison.py
Đọc logs JSON (results/logs_kdl) để vẽ biểu đồ so sánh Kịch bản 3: FedKDL vs baseline_od.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_comparison():
    log_files = glob.glob("results/logs_kdl/*.json")
    if not log_files:
        print("[Warning] Không có file JSON nào trong results/logs_kdl.")
        return

    # Chọn 1 cấu hình cố định để plot trace (VD: N=50, alpha=0.1, seed=42)
    target_N = 50
    target_alpha = "0p1"
    target_seed = 42
    
    results = {}
    
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        n = meta.get("N")
        alpha = meta.get("alpha")
        seed = meta.get("seed")
        
        if n == target_N and alpha == target_alpha and seed == target_seed:
            results[baseline] = data.get("metrics", {})

    if not results:
        print(f"[Warning] Không tìm thấy dữ liệu trace cho N={target_N}, alpha={target_alpha}, seed={target_seed}")
        return

    os.makedirs("results/scenario3", exist_ok=True)

    colors = {"baseline_od": "orange", "fedkdl": "green"}
    labels = {"baseline_od": "Bottleneck OD (No compression)", "fedkdl": "FedKDL (Ours)"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for key, hist in results.items():
        if 'round' not in hist: continue
        map_val = hist.get('mAP50-95', hist.get('map', []))
        e_cumul_val = hist.get('e_cumul', hist.get('energy_cumul_J', []))
        if len(map_val) > 0: axes[0].plot(hist['round'], map_val, label=labels.get(key, key), color=colors.get(key, 'black'), marker='o')
        if len(hist.get('alive', [])) > 0: axes[1].plot(hist['round'], hist['alive'], label=labels.get(key, key), color=colors.get(key, 'black'), marker='o')
        if len(e_cumul_val) > 0: axes[2].plot(hist['round'], e_cumul_val, label=labels.get(key, key), color=colors.get(key, 'black'), marker='o')

    axes[0].set_title("mAP@0.5:0.95 vs Round")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("mAP")
    
    axes[1].set_title("Alive AUVs vs Round")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Alive AUVs")
    
    axes[2].set_title("Cumulative Energy (J)")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("E (J)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = "results/scenario3/fedkdl_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Lưu biểu đồ tại {save_path}")

if __name__ == "__main__":
    plot_comparison()
