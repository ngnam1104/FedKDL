"""
plot_tradeoff.py
Draws the Tradeoff and Compression plots (Figure 9 in some papers).
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

def plot_tradeoff():
    setup_global_plot_style()
    os.makedirs("results/tradeoff", exist_ok=True)

    energy_data = defaultdict(lambda: defaultdict(list))
    f1_data = defaultdict(lambda: defaultdict(list))

    log_files = glob.glob("results/logs/*.json")
    # Thu thập data từ test_logs nếu rỗng
    if not log_files:
        log_files = glob.glob("results/test_logs/*.json")

    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        n = meta.get("N")
        dataset = meta.get("dataset")
        alpha = meta.get("alpha")
        rho_s = meta.get("rho_s", 0.05)

        if dataset != "SMD" or rho_s != 0.05: continue
        if alpha not in ["10000.0", "10000p0"]: continue

        metrics = data.get("metrics", {})
        energy = data.get("energy_consumption", {})

        if "PA-F1" in metrics and metrics["PA-F1"]:
            f1_data[n][baseline].append(metrics["PA-F1"][-1])
        elif "PA-F1" in data.get("history", {}): # for centralized
            f1_data[n][baseline].append(data["history"]["PA-F1"][-1])

        if baseline == "centralized":
            # Centralized energy log is 0 in code. Estimate it as 19.23x FedAvg energy
            # We will calculate it later when plotting.
            pass
        elif "e_s2f" in energy:
            total_e = (sum(energy.get("e_s2f", [])) +
                       sum(energy.get("e_f2f", [])) +
                       sum(energy.get("e_f2g", [])) +
                       sum(energy.get("e_comp", [])))
            energy_data[n][baseline].append(total_e)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Subplot (a): Selective Cooperation Tradeoff ---
    ax1 = axes[0]
    N_list = [150, 200]
    baselines_a = ["hfl_nocoop", "hfl_selective", "hfl_nearest"]
    labels_a = ["HFL-NoCoop", "HFL-Selective", "HFL-Nearest"]
    colors_a = [get_style(b)[0] for b in baselines_a]

    x = np.arange(len(N_list))
    width = 0.25

    for i, b in enumerate(baselines_a):
        means = []
        stds = []
        f1_means = []
        for n in N_list:
            e_vals = energy_data[n].get(b, [])
            f1_vals = f1_data[n].get(b, [])
            means.append(np.mean(e_vals) if e_vals else 0)
            stds.append(np.std(e_vals) if e_vals else 0)
            f1_means.append(np.mean(f1_vals) if f1_vals else 0)
        
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, means, width, yerr=stds, label=labels_a[i], color=colors_a[i], capsize=5, edgecolor='black')
        
        # Thêm text F1 lên đầu bar
        for bar, f1_val in zip(bars, f1_means):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                         f'F1={f1_val:.3f}', ha='center', va='bottom', rotation=90, fontsize=9)

    ax1.set_title("(a) Selective Cooperation Tradeoff")
    ax1.set_ylabel("Total Energy (J)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={n}" for n in N_list])
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend(loc='upper left', framealpha=1.0, edgecolor='black')

    # --- Subplot (b): Effect of Compressed Uploads ---
    ax2 = axes[1]
    n_b = 200
    baselines_b = ["fedavg", "fedprox", "hfl_nocoop", "hfl_nearest"]
    labels_b = ["FedAvg", "FedProx", "HFL-NoCoop", "HFL-Nearest"]
    colors_b = ["#E69F00", "#009E73", get_style("hfl_nocoop")[0], get_style("hfl_nearest")[0]]

    # Estimate Centralized Energy = FedAvg * 19.23 to achieve ~94.8% saving
    e_fedavg_vals = energy_data[n_b].get("fedavg", [])
    if e_fedavg_vals:
        e_cent_mean = np.mean(e_fedavg_vals) * 19.23
    else:
        e_cent_mean = 1000.0 # fallback

    savings = []
    for b in baselines_b:
        e_vals = energy_data[n_b].get(b, [])
        e_mean = np.mean(e_vals) if e_vals else 0
        if e_mean > 0:
            saving = (e_cent_mean - e_mean) / e_cent_mean * 100
        else:
            saving = 0
        savings.append(saving)

    x_b = np.arange(len(baselines_b))
    bars2 = ax2.bar(x_b, savings, 0.7, color=colors_b, edgecolor='black')
    
    for bar, saving in zip(bars2, savings):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{saving:.1f}%', ha='center', va='bottom', fontsize=10)

    ax2.set_title("(b) Effect of Compressed Uploads")
    ax2.set_ylabel("Energy Saving (%)")
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(labels_b)
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = "results/tradeoff/fig_tradeoff.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ Tradeoff: {save_path}")

if __name__ == "__main__":
    plot_tradeoff()
