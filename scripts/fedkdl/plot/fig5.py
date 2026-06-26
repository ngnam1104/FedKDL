"""Hình 5: LoRA Rank/Capacity Ablation (Kịch bản 2).

Single panel — mAP@0.5 learning curves for different LoRA rank configs
    Sources : fedkdl_r24_metrics.csv   → LoRA r2/4
              fedkdl_r44_metrics.csv   → LoRA r4/4
              fedkdl_metrics.csv       → FedKDL (default rank)
              fedkdl_32bit_metrics.csv → 32-bit Full FT (upper bound)
    X-axis  : Communication Round
    Y-axis  : mAP@0.5

Usage:
    python scripts/fedkdl/plot/fig5.py
"""

import matplotlib.pyplot as plt

from plot_common import (
    T, add_zoom_inset, plot_learning, save_figure, setup_style,
)

KEYS = ["fedkdl_r24", "fedkdl_r44", "fedkdl", "fedkdl_32bit"]


def draw(lang: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plot_learning(ax, KEYS, lang, legend_loc="upper center", bbox_to_anchor=(0.5, -0.24))
    add_zoom_inset(ax, KEYS, lang, zoom_start_frac=0.60, loc="lower right")
    ax.set_title(T("LoRA Rank Ablation Learning Curves", lang), loc="center", pad=12, fontweight="bold")
    save_figure(fig, "K2_fig5_rank_ablation", lang)


def main() -> None:
    setup_style()
    for lang in ("en", "vi"):
        print(f"[{lang.upper()}] Fig 5: LoRA Rank Ablation")
        draw(lang)


if __name__ == "__main__":
    main()
