"""Hình 6: Non-IID Baselines (Kịch bản 3).

Panel (a) — Non-IID algorithm comparison (learning curves) + zoom inset
    Sources : centralized, fedavg_hfl, fedprox_hfl, scaffold, fedkdl metrics CSVs

Panel (b) — Grouped cost bars (normalized): Loss | λ_E·Energy | λ_τ·Latency
    FL methods only (Centralized excluded — no FL transmission cost)

Usage:
    python scripts/fedkdl/plot/fig6.py
"""

import matplotlib.pyplot as plt

from plot_common import (
    L, T, summary_row,
    add_zoom_inset, grouped_cost_bars, plot_learning,
    save_figure, setup_style,
)

KEYS      = ["centralized", "fedavg_hfl", "fedprox_hfl", "scaffold", "fedkdl"]


def draw(lang: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    # ── (a) Learning curves + zoom inset ────────────────────────────────────
    plot_learning(axes[0], KEYS, lang,
                  legend_loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    add_zoom_inset(axes[0], KEYS, lang, zoom_start_frac=0.55, loc="lower right")
    axes[0].text(0.5, -0.12, "(a)", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=11, fontweight="bold")

    # ── (b) Grouped cost bars: Loss | Energy | Latency (normalized) ──────────
    COST_KEYS = sorted(KEYS, key=lambda k: summary_row(k)["avg_joint_cost"], reverse=True)
    grouped_cost_bars(axes[1], COST_KEYS, lang,
                      legend_loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    axes[1].text(0.5, -0.12, "(b)", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=11, fontweight="bold")

    save_figure(fig, "K3_fig6_noniid_baselines", lang)


def main() -> None:
    setup_style()
    for lang in ("en", "vi"):
        print(f"[{lang.upper()}] Fig 6: Non-IID Baselines")
        draw(lang)


if __name__ == "__main__":
    main()
