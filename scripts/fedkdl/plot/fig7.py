"""Hình 7: Relay Cooperation Ablation (Kịch bản 3).

Panel (a) — Cooperation learning curves + zoom inset
    Sources : fedkdl_nocoop, fedkdl_selective, fedkdl metrics CSVs

Panel (b) — Cột ghép dual Y-axis (BOTH metrics as bars, no line)
    Same 3 cooperation configs
    Y-left  : Total objective cost (bars, blue)
    Y-right : Peak mAP@0.5 (bars, red)

Convention: "cột ghép" — cả hai trục đều vẽ bằng CỘT, không dùng đường.

Usage:
    python scripts/fedkdl/plot/fig7.py
"""

import matplotlib.pyplot as plt

from plot_common import (
    L, T,
    add_zoom_inset, grouped_dual_bar_map, plot_learning,
    save_figure, setup_style,
)

KEYS = ["fedkdl_nocoop", "fedkdl_selective", "fedkdl"]


def draw(lang: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    # ── (a) Learning curves + zoom inset ────────────────────────────────────
    plot_learning(axes[0], KEYS, lang, legend_loc="upper center", bbox_to_anchor=(0.5, -0.24))
    add_zoom_inset(axes[0], KEYS, lang, zoom_start_frac=0.55, loc="lower right")
    axes[0].set_title(T("(a)", lang) + " " + T("Relay Cooperation Learning Curves", lang), loc="center", fontweight="bold")

    # ── (b) Grouped dual bars: Objective Cost & mAP ────────────────────────
    grouped_dual_bar_map(axes[1], KEYS, lang, legend_loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
                         cost_col="avg_joint_cost",
                         cost_label="Objective Cost")
    axes[1].set_title(T("(b)", lang) + " " + T("Objective Cost vs. Mean Average Precision (mAP@0.5)", lang), loc="center", fontweight="bold")

    save_figure(fig, "K3_fig7_cooperation_ablation", lang)


def main() -> None:
    setup_style()
    for lang in ("en", "vi"):
        print(f"[{lang.upper()}] Fig 7: Cooperation Ablation")
        draw(lang)


if __name__ == "__main__":
    main()
