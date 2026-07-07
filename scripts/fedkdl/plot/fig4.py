"""Hình 4: Compression-Method Learning and Cost Breakdown (Kịch bản 2).

Panel (a) — Learning curves (mAP@0.5 vs. round) + zoom inset
    Sources : fedavg_hfl, naive_lora, flora, top_k, fedkdl metrics CSVs

Panel (b) — Grouped cost bars (normalized): Loss | λ_E·Energy | λ_τ·Latency
    Same methods as (a).  All quantities normalized for visual comparison.

Usage:
    python scripts/fedkdl/plot/fig4.py
"""

import matplotlib.pyplot as plt

from plot_common import (
    L, T,
    add_zoom_inset, grouped_cost_bars, plot_learning,
    save_figure, setup_style,
)

KEYS = ["fedavg_hfl", "naive_lora", "flora", "top_k", "fedkdl"]


def draw(lang: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    # ── (a) Learning curves + zoom ───────────────────────────────────────────
    plot_learning(axes[0], KEYS, lang,
                  legend_loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    add_zoom_inset(axes[0], KEYS, lang, zoom_start_frac=0.60, loc="lower right")
    axes[0].text(0.5, -0.12, "(a)", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=11, fontweight="bold")

    # ── (b) Grouped cost bars: Loss | Energy | Latency (normalized) ──────────
    grouped_cost_bars(axes[1], KEYS, lang,
                      legend_loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    axes[1].text(0.5, -0.12, "(b)", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=11, fontweight="bold")

    save_figure(fig, "K2_fig4_compression_learning", lang)


def main() -> None:
    setup_style()
    for lang in ("en", "vi"):
        print(f"[{lang.upper()}] Fig 4: Compression + Objective")
        draw(lang)


if __name__ == "__main__":
    main()
