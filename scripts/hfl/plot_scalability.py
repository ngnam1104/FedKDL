"""
plot_scalability.py
Đọc logs JSON để vẽ biểu đồ khả năng mở rộng (Figure 5).
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

def plot_scalability():
    setup_global_plot_style()
    os.makedirs("results/scalability", exist_ok=True)

    participation_data = defaultdict(lambda: defaultdict(list))
    energy_data = defaultdict(lambda: defaultdict(list))
    latency_data = defaultdict(lambda: defaultdict(list))

    log_files = glob.glob("results/logs/*.json")
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        n = meta.get("N")

        if meta.get("dataset") != "SMD": continue
        if meta.get("rho_s") != 0.05: continue

        metrics = data.get("metrics", {})
        energy = data.get("energy_consumption", {})
        latency = data.get("latency_history", {})

        # Participation — last round
        if "Participation" in metrics and metrics["Participation"]:
            participation_data[baseline][n].append(metrics["Participation"][-1])

        # Energy — tổng toàn bộ round
        if "e_s2f" in energy:
            total_e = (sum(energy.get("e_s2f", [])) +
                       sum(energy.get("e_f2f", [])) +
                       sum(energy.get("e_f2g", [])) +
                       sum(energy.get("e_comp", [])))
            energy_data[baseline][n].append(total_e)

        # Latency — dùng tau_round_s (tổng hợp) thay cho breakdown
        if "tau_round_s" in latency and latency["tau_round_s"]:
            latency_data[baseline][n].append(np.mean(latency["tau_round_s"]))

    if not participation_data:
        print("[Warning] Không tìm thấy dữ liệu scalability.")
        return

    N_sorted = sorted({n for b in participation_data for n in participation_data[b]})

    # --- Fig 5a: Participation ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for baseline, bdata in participation_data.items():
        means = [np.mean(bdata[n]) * 100 if bdata[n] else np.nan for n in N_sorted]
        c, m, l = get_style(baseline)
        ax.plot(N_sorted, means, label=l, color=c, marker=m, linewidth=2, markersize=8)
    ax.set_title("Network Participation vs Scale")
    ax.set_xlabel("Number of AUVs (N)")
    ax.set_ylabel("Participation Rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/scalability/fig5a_participation.png", dpi=150)
    plt.close()

    # --- Fig 5b: Energy ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for baseline, bdata in energy_data.items():
        means = [np.mean(bdata[n]) if bdata[n] else np.nan for n in N_sorted]
        c, m, l = get_style(baseline)
        ax.plot(N_sorted, means, label=l, color=c, marker=m, linewidth=2, markersize=8)
    ax.set_title("Total Energy Consumption vs Scale")
    ax.set_xlabel("Number of AUVs (N)")
    ax.set_ylabel("Energy (Joules)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/scalability/fig5b_energy.png", dpi=150)
    plt.close()

    # --- Fig 5c: Latency ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for baseline, bdata in latency_data.items():
        means = [np.mean(bdata[n]) if bdata[n] else np.nan for n in N_sorted]
        c, m, l = get_style(baseline)
        ax.plot(N_sorted, means, label=l, color=c, marker=m, linewidth=2, markersize=8)
    ax.set_title("Average Round Latency vs Scale")
    ax.set_xlabel("Number of AUVs (N)")
    ax.set_ylabel("Latency (Seconds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/scalability/fig5c_latency.png", dpi=150)
    plt.close()

    print("\nDa luu bieu do Scalability (Fig 5a, 5b, 5c).")

if __name__ == "__main__":
    plot_scalability()
