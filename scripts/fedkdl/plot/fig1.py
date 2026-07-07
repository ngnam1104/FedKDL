"""Hình 1: Connectivity and Participation (Kịch bản 1).

Panel (a) — Participating AUVs vs. Network Size (N = 30 … 100)
    Source  : scalability_physics_summary.csv
    X-axis  : Number of AUVs in Network (30, 40, 50, 60, 70, 80, 90, 100)
    Y-axis  : Participating AUVs (mean ± std shading)
    Methods : FedAvg (Flat), FedProx (Flat), FedKDL

Panel (b) — Connectivity-constrained learning at N = 30
    Sources : fedavg_metrics.csv, fedprox_metrics.csv, fedkdl_metrics.csv
    X-axis  : Communication Round
    Y-axis  : mAP@0.5
    + zoom inset on converged region

Usage:
    python scripts/fedkdl/plot/fig1.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import (
    COLORS, MARKERS, SCALABILITY_SUMMARY,
    L, T, plot_learning,
    save_figure, setup_style,
)

# Methods for panel (a) — must match 'baseline' column in scalability_physics_summary.csv
PART_METHODS = [
    ("fedkdl",  "fedkdl"),    # (csv_key, color_key)
    ("fedavg",  "fedavg"),
    ("fedprox", "fedprox"),
]

# Methods for panel (b)
KEYS_B = ["fedavg", "fedprox", "fedkdl"]


def draw(lang: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    # ── (a) Participating AUVs vs. Network Size ──────────────────────────────
    if SCALABILITY_SUMMARY.exists():
        summ = pd.read_csv(SCALABILITY_SUMMARY)
        for idx, (csv_key, color_key) in enumerate(PART_METHODS):
            rows = summ[summ["baseline"] == csv_key].sort_values("N_AUV")
            if rows.empty:
                continue
            x     = rows["N_AUV"].to_numpy()
            y     = rows["participants_mean"].to_numpy()
            y_std = rows["participants_std"].fillna(0).to_numpy()
            color  = COLORS.get(color_key, f"C{idx}")
            marker = MARKERS[idx % len(MARKERS)]
            axes[0].plot(x, y,
                         label=L(color_key, lang),
                         color=color, marker=marker,
                         linewidth=2)
            axes[0].fill_between(x, y - y_std, y + y_std,
                                 color=color, alpha=0.15)
        axes[0].set_xlabel(T("Number of AUVs in Network", lang))
        axes[0].set_ylabel(T("Connected AUVs", lang))
        axes[0].set_xlim(left=28)
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                       fontsize=8.5, framealpha=0.95)
    else:
        axes[0].text(0.5, 0.5, "scalability_physics_summary.csv not found",
                     transform=axes[0].transAxes, ha="center", va="center")

    axes[0].text(0.5, -0.12, "(a)", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=11, fontweight="bold")

    # ── (b) Learning curves at N = 30 ────────────────────────────────────────
    plot_learning(axes[1], KEYS_B, lang,
                  legend_loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    axes[1].text(0.5, -0.12, "(b)", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=11, fontweight="bold")

    save_figure(fig, "K1_fig1_connectivity_participation", lang)


def main() -> None:
    setup_style()
    for lang in ("en", "vi"):
        print(f"[{lang.upper()}] Fig 1: Connectivity & Participation")
        draw(lang)


if __name__ == "__main__":
    main()
