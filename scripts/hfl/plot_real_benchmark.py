"""
plot_real_benchmark.py
Đọc logs JSON để vẽ biểu đồ benchmark dữ liệu thực (Figure 7 & 8).
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

def plot_real_benchmark():
    setup_global_plot_style()
    os.makedirs("results/real_benchmark", exist_ok=True)

    # Cau truc: dataset -> baseline -> list PA-F1 cuoi (nhieu seed)
    f1_data = defaultdict(lambda: defaultdict(list))

    log_files = glob.glob("results/logs/*.json")
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        dataset = meta.get("dataset")
        n = meta.get("N")
        alpha = meta.get("alpha")

        if str(n) != "200": continue
        if alpha not in ("0p1", "0.1"): continue
        if meta.get("rho_s") != 0.05: continue

        metrics = data.get("metrics", {})
        if "PA-F1" in metrics and metrics["PA-F1"]:
            f1_data[dataset][baseline].append(metrics["PA-F1"][-1])

    if not f1_data:
        print("[Warning] Khong tim thay du lieu real benchmark.")
        return

    datasets = sorted(f1_data.keys())
    baselines = ["hfl_selective", "hfl_nearest", "hfl_nocoop", "fedprox", "fedavg"]

    x = np.arange(len(datasets))
    width = 0.15

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, baseline in enumerate(baselines):
        means = []
        for ds in datasets:
            vals = f1_data[ds].get(baseline, [])
            means.append(np.mean(vals) if vals else 0.0)
        c, m, l = get_style(baseline)
        offset = (i - 2) * width
        ax.bar(x + offset, means, width, label=l, color=c)

    ax.set_title("PA-F1 Score on Real Datasets (N=200, alpha=0.1)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("PA-F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()

    save_path = "results/real_benchmark/fig7_8_real_benchmark.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do: {save_path}")

if __name__ == "__main__":
    plot_real_benchmark()
