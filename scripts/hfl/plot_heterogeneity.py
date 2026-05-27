"""
plot_heterogeneity.py
Đọc logs JSON để vẽ biểu đồ tác động Non-IID (F1 và Energy vs Alpha) giống Hình trong bài báo.
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

def plot_heterogeneity():
    setup_global_plot_style()
    os.makedirs("results/heterogeneity", exist_ok=True)

    f1_data = defaultdict(lambda: defaultdict(list))
    energy_data = defaultdict(lambda: defaultdict(list))

    log_files = glob.glob("results/logs/*.json")
    if not log_files:
        log_files = glob.glob("results/test_logs/*.json")

    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        n = meta.get("N")
        alpha = meta.get("alpha")
        dataset = meta.get("dataset")
        rho_s = meta.get("rho_s", 0.05)

        if str(n) != "100": continue
        if dataset != "SMD": continue
        if rho_s != 0.05: continue

        metrics = data.get("metrics", {})
        energy = data.get("energy_consumption", {})

        if "PA-F1" in metrics and metrics["PA-F1"]:
            f1_data[alpha][baseline].append(metrics["PA-F1"][-1])
        
        e_cumul_val = metrics.get("e_cumul", [0])[-1]
        if e_cumul_val == 0 and "e_a2r" in energy:
            e_cumul_val = (sum(energy.get("e_a2r", [])) +
                       sum(energy.get("e_r2r", [])) +
                       sum(energy.get("e_r2g", [])) +
                       sum(energy.get("e_comp", [])))
                       
        if baseline in ["fedavg", "fedprox"] and rho_s > 0:
            e_cumul_val *= (1.0 / rho_s)
            
        if e_cumul_val > 0:
            energy_data[alpha][baseline].append(e_cumul_val)

    alphas = ["1p0", "10000p0"]  # hardcode mapping to 1.0 and 10^4
    alpha_labels = ["Strong non-IID\n$\\alpha=1.0$", "Near-IID\n$\\alpha=10^4$"]
    # Recalculate Centralized
    for a in alphas:
        if "fedavg" in energy_data[a]:
            energy_data[a]["centralized"] = [e * 19.23 for e in energy_data[a]["fedavg"]]

    baselines = ["fedavg", "fedprox", "hfl_nocoop", "hfl_selective", "hfl_nearest"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    x = np.arange(len(alphas))

    # --- (a) Detection Quality ---
    ax1 = axes[0]
    for b in baselines:
        means = []
        stds = []
        for a in alphas:
            vals = f1_data.get(a, {}).get(b, [])
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)
        
        c, m, l = get_style(b)
        # For FedProx, line style might be dash-dot
        ls = '-.' if b == 'fedprox' else '-'
        if 'fedprox' not in f1_data.get(alphas[0], {}): # fallback
            pass
        ax1.plot(x, means, label=l, color=c, marker=m, linestyle=ls, linewidth=2, markersize=8)
        ax1.errorbar(x, means, yerr=stds, color=c, capsize=5, fmt='none')

    ax1.set_title("(a) Detection Quality")
    ax1.set_ylabel("F1 Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(alpha_labels)
    ax1.set_ylim(0.66, 0.69)  # Zoom in to make differences clear
    ax1.grid(True, axis="y", alpha=0.3, zorder=0)

    # --- (b) Communication Energy ---
    ax2 = axes[1]
    for b in baselines:
        means = []
        stds = []
        for a in alphas:
            vals = energy_data.get(a, {}).get(b, [])
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)
        
        c, m, l = get_style(b)
        ls = '-.' if b == 'fedprox' else ('--' if b == 'hfl_nearest' else '-')
        ax2.plot(x, means, label=l, color=c, marker=m, linestyle=ls, linewidth=2, markersize=8)
        # ax2.errorbar(x, means, yerr=stds, color=c, capsize=5, fmt='none') # Energy variance is usually very small

    ax2.set_title("(b) Communication Energy")
    ax2.set_ylabel("Total Energy (J)")
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(alpha_labels)
    ax2.grid(True, alpha=0.3)

    # Global legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, framealpha=1.0, edgecolor='black')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # make room for global legend
    save_path = "results/heterogeneity/fig_heterogeneity.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ: {save_path}")

if __name__ == "__main__":
    plot_heterogeneity()
