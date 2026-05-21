"""
plot_heterogeneity.py
Đọc logs JSON để vẽ biểu đồ tác động Non-IID (Figure 6).
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

    # Cau truc: alpha -> baseline -> list cac PA-F1 history (nhieu seed)
    f1_data = defaultdict(lambda: defaultdict(list))

    log_files = glob.glob("results/logs/*.json")
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        n = meta.get("N")
        alpha = meta.get("alpha")

        if str(n) != "200": continue
        if meta.get("dataset") != "SMD": continue
        if meta.get("rho_s") != 0.05: continue

        metrics = data.get("metrics", {})
        if "PA-F1" in metrics and metrics["PA-F1"]:
            f1_data[alpha][baseline].append(metrics["PA-F1"])

    if not f1_data:
        print("[Warning] Khong tim thay du lieu heterogeneity.")
        return

    alphas = sorted(f1_data.keys())

    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 5))
    if len(alphas) == 1:
        axes = [axes]

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        data_a = f1_data[alpha]

        for baseline, f1_list in data_a.items():
            if not f1_list: continue
            min_len = min(len(h) for h in f1_list)
            f1_matrix = np.array([h[:min_len] for h in f1_list])
            mean_f1 = np.mean(f1_matrix, axis=0)
            std_f1 = np.std(f1_matrix, axis=0)
            rounds = np.arange(min_len)
            c, m, l = get_style(baseline)
            ax.plot(rounds, mean_f1, label=l, color=c, marker=m, linewidth=2, markevery=5)
            ax.fill_between(rounds,
                            np.maximum(0, mean_f1 - std_f1),
                            np.minimum(1, mean_f1 + std_f1),
                            color=c, alpha=0.2)

        try:
            a_val = str(float(alpha.replace("p", ".")))
        except Exception:
            a_val = alpha
        title_suffix = "(Non-IID)" if float(a_val) < 1.0 else "(IID)"
        ax.set_title(f"PA-F1 (alpha={a_val}) {title_suffix}")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("PA-F1 Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    save_path = "results/heterogeneity/fig6_heterogeneity.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do: {save_path}")

if __name__ == "__main__":
    plot_heterogeneity()
