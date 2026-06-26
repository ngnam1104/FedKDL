"""Hình 8: Gateway-Refinement Learning Curves (Kịch bản 4).

Single panel — mAP@0.5 learning curves for gateway-side refinement variants
    Sources : fedkdl_nokd_metrics.csv  → No Gateway KD
              proxy_ft_metrics.csv     → Proxy-set FT
              logit_kd_metrics.csv     → Logit/Box KD
              fedkdl_metrics.csv       → FedKDL (full, logit + box KD)
    X-axis  : Communication Round
    Y-axis  : mAP@0.5

Usage:
    python scripts/fedkdl/plot/fig8.py
"""

import matplotlib.pyplot as plt

from plot_common import (
    T, add_zoom_inset, grouped_map_bars, plot_learning, save_figure, setup_style,
)

KEYS = ["fedkdl_nokd", "proxy_ft", "logit_kd", "fedkdl"]


def draw(lang: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    # ── (a) Learning curves + zoom ───────────────────────────────────────────
    plot_learning(axes[0], KEYS, lang, legend_loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    add_zoom_inset(axes[0], KEYS, lang, zoom_start_frac=0.60, loc="lower right")
    axes[0].set_title(T("(a)", lang) + " " + T("Gateway Refinement Learning Curves", lang), loc="center", fontweight="bold")

    # ── (b) Grouped map bars: mAP50 | mAP50-95 ──────────────────────────────
    grouped_map_bars(axes[1], KEYS, lang, legend_loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    axes[1].set_title(T("(b)", lang) + " " + T("mAP Comparison", lang), loc="center", fontweight="bold")

    save_figure(fig, "K4_fig8_gateway_refinement", lang)


def main() -> None:
    setup_style()
    for lang in ("en", "vi"):
        print(f"[{lang.upper()}] Fig 8: Gateway Refinement")
        draw(lang)


if __name__ == "__main__":
    main()
