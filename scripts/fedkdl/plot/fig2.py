"""Hình 2: Learning under AUV Mobility (Kịch bản 1).

Single panel — mAP@0.5 learning curves for three mobility levels
    Sources : fedkdl_metrics.csv         → FedKDL (5 m/round)
              fedkdl_v50_metrics.csv     → FedKDL (50 m/round)
              fedkdl_v100_metrics.csv    → FedKDL (100 m/round)
    X-axis  : Communication Round
    Y-axis  : mAP@0.5

Usage:
    python scripts/fedkdl/plot/fig2.py
"""

import matplotlib.pyplot as plt

from plot_common import (
    COLORS,
    T, save_figure, setup_style, add_zoom_inset
)

KEYS = ["fedkdl", "fedkdl_v50", "fedkdl_v100"]

# Velocity labels shown in legend (same in both languages)
VEL_LABELS = {
    "fedkdl":      "FedKDL (5 m/round)",
    "fedkdl_v50":  "FedKDL (50 m/round)",
    "fedkdl_v100": "FedKDL (100 m/round)",
}


def draw(lang: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    from plot_common import plot_learning
    plot_learning(ax, KEYS, lang, override_labels=VEL_LABELS, legend_loc="upper center", bbox_to_anchor=(0.5, -0.24))
    add_zoom_inset(ax, KEYS, lang, zoom_start_frac=0.55, loc="lower right", override_labels=VEL_LABELS)
    ax.set_title(T("Learning under AUV Mobility", lang), loc="center", pad=12, fontweight="bold")
    save_figure(fig, "K1_fig2_mobility", lang)


def main() -> None:
    setup_style()
    for lang in ("en", "vi"):
        print(f"[{lang.upper()}] Fig 2: Mobility")
        draw(lang)


if __name__ == "__main__":
    main()
