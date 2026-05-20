"""
plot_metrics.py
Vẽ biểu đồ tổng hợp từ nhiều file JSON kết quả.
Hỗ trợ cả Scenario 1 (1D anomaly) và Scenario 2 & 3 (OD).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE = [
    '#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e',
]


def load_results(results_dir: str) -> dict:
    """Đọc tất cả file JSON trong thư mục → dict[name → history]."""
    data = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.json'):
            with open(os.path.join(results_dir, fname)) as f:
                data[fname.replace('.json', '')] = json.load(f)
    return data


def plot_panel(ax, histories: dict, key: str, ylabel: str, title: str, log=False):
    for idx, (name, hist) in enumerate(histories.items()):
        if key not in hist or not hist[key]:
            continue
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(hist['round'], hist[key], label=name, color=color, linewidth=1.8)
    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    if log:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')


def main(results_dir: str, output_path: str):
    histories = load_results(results_dir)
    if not histories:
        print(f"Không tìm thấy kết quả trong '{results_dir}'.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("FedKDL — Simulation Results", fontsize=14, fontweight='bold')

    plot_panel(axes[0, 0], histories, 'map',            'mAP@0.5:0.95',  'Detection Accuracy')
    plot_panel(axes[0, 1], histories, 'alive',          'Alive AUVs',    'Network Survival')
    plot_panel(axes[1, 0], histories, 'avg_payload_kb', 'Payload (KB)',   'Avg Payload / Round', log=True)
    plot_panel(axes[1, 1], histories, 'energy_cumul_J', 'Energy (J)',     'Cumulative Energy Consumed')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Biểu đồ đã lưu tại: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Thư mục chứa JSON")
    parser.add_argument("--output", default="results/summary_plot.png", help="Đường dẫn ảnh output")
    args = parser.parse_args()
    main(args.results_dir, args.output)
