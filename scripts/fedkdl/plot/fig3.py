"""Hình 3: Average Uplink Payload Comparison (Kịch bản 2).

Biểu đồ cột ghép với hai trục tung (Dual Y-axis grouped bars)
    Sources : fedavg_hfl, naive_lora, flora, top_k, fedkdl metrics CSVs
    X-axis  : Algorithm
    Y-left  : Average uplink payload (KiB/AUV)  — cột màu xanh
    Y-right : Peak mAP@0.5                      — cột màu đỏ

Convention (FedKDL_final_experiment_scenarios.md §2):
    "cột ghép": cả hai metrics đều vẽ bằng CỘT, không dùng đường
    "trục tung trái": payload  |  "trục tung phải": mAP@0.5

Usage:
    python scripts/fedkdl/plot/fig3.py
"""

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    L, T, summary_row, save_figure, setup_style,
)

KEYS = ["fedavg_hfl", "naive_lora", "flora", "top_k", "fedkdl"]

# Bar colours
COLOR_PAYLOAD = "#2C7BB6"   # clear blue (not purple)
COLOR_MAP     = "#C44E52"   # seaborn-deep red

BAR_W = 0.38          # width of each bar
BAR_GAP = 0.04        # gap between the two bars of the same method


def draw(lang: str) -> None:
    rows     = [summary_row(k) for k in KEYS]
    payload  = np.array([r["avg_payload_kb"] / 1024.0 for r in rows])  # KiB → MB
    peak_map = np.array([r["peak_mAP50"]               for r in rows])
    labels   = [L(k, lang) for k in KEYS]

    n = len(KEYS)
    x = np.arange(n)

    fig, ax_left = plt.subplots(figsize=(8.5, 5.5))
    ax_right = ax_left.twinx()

    # ── Payload bars (left Y) ────────────────────────────────────────────────
    ax_left.bar(
        x - BAR_W / 2 - BAR_GAP / 2, payload, BAR_W,
        color=COLOR_PAYLOAD, edgecolor="white", linewidth=0.6,
        label=T("Average Uplink Payload (MB/AUV)", lang),
    )

    # ── mAP bars (right Y) ──────────────────────────────────────────────────
    ax_right.bar(
        x + BAR_W / 2 + BAR_GAP / 2, peak_map, BAR_W,
        color=COLOR_MAP, edgecolor="white", linewidth=0.6, alpha=0.88,
        label="mAP@0.5",
    )

    # ── Axis labels & ticks ──────────────────────────────────────────────────
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=15, ha="right")
    ax_left.set_ylabel(T("Average Uplink Payload (MB/AUV)", lang),
                       color=COLOR_PAYLOAD)
    ax_right.set_ylabel(T("Mean Average Precision (mAP@0.5)", lang), color=COLOR_MAP)
    ax_left.tick_params(axis="y", colors=COLOR_PAYLOAD)
    ax_right.tick_params(axis="y", colors=COLOR_MAP)

    # Headroom
    ax_left.set_ylim(0, max(payload) * 1.22)
    ax_right.set_ylim(
        max(0, min(peak_map) - 0.04),
        max(peak_map) + 0.04,
    )

    # Value labels on payload bars
    for i, v in enumerate(payload):
        ax_left.text(
            x[i] - BAR_W / 2 - BAR_GAP / 2,
            v + max(payload) * 0.015,
            f"{v:.2f} MB", ha="center", va="bottom",
            fontsize=8, color=COLOR_PAYLOAD,
        )

    # Value labels on mAP bars
    for i, v in enumerate(peak_map):
        ax_right.text(
            x[i] + BAR_W / 2 + BAR_GAP / 2,
            v + (max(peak_map) - min(peak_map)) * 0.05 + 0.002,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=8, color=COLOR_MAP,
        )

    # ── Combined legend ──────────────────────────────────────────────────────
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.24),
                   ncol=2, framealpha=0.9)

    ax_left.set_xlim(-0.6, n - 0.4)
    ax_left.set_axisbelow(True)

    save_figure(fig, "K2_fig3_payload_comparison", lang)


def main() -> None:
    setup_style()
    for lang in ("en", "vi"):
        print(f"[{lang.upper()}] Fig 3: Payload Comparison (grouped bars)")
        draw(lang)


if __name__ == "__main__":
    main()
