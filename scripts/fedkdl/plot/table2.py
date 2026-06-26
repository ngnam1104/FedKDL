"""Bảng 2: Objective Components and Peak Accuracy (Scenarios 2 + 3).

Gộp toàn bộ các phương pháp ở Kịch bản 2 và 3, xếp theo peak mAP@0.5 giảm dần.

Outputs:
  - results/metrics_final/tables_paper/table2_objective_summary.csv
  - results/metrics_final/tables_paper/table2_objective_summary.md
  - results/metrics_final/tables_paper/table2_objective_summary.tex  (LaTeX)

Usage:
    python scripts/fedkdl/plot/table2.py
"""

import numpy as np

from plot_common import (
    LABELS_EN, summary_row, save_table, save_table_latex, save_table_pdf, setup_style,
)

ALL_KEYS = [
    "centralized",
    "fedkdl",
    "scaffold",
    "fedavg_hfl",
    "fedprox_hfl",
    "flora",
    "top_k",
    "naive_lora",
]

CAPTION = (
    "Mean Average Precision (mAP@0.5), peak validation loss, and physical-layer cost components "
    "for all methods in Scenarios~2 and~3, sorted by peak mAP@0.5 (descending). "
    r"Energy (J) and latency (s) are per-round averages; "
    r"Total Obj.\ $= \lambda_E E + \lambda_\tau \tau$ with "
    r"$\lambda_E = 0.005$, $\lambda_\tau = 0.01$. "
    r"``---'' indicates centralized training with no FL transmission cost."
)
LABEL = "tab:objective_summary"


def build() -> None:
    data_rows = []
    for k in ALL_KEYS:
        try:
            r = summary_row(k)
            data_rows.append({
                "key":            k,
                "peak_mAP50":     r["peak_mAP50"],
                "peak_loss":      r["peak_loss"],
                "avg_energy_j":   r["avg_energy_j"],
                "avg_tau_s":      r["avg_tau_s"],
                "avg_joint_cost": r["avg_joint_cost"],
            })
        except FileNotFoundError:
            print(f"  [WARN] Skipping {k} — file not found")

    if not data_rows:
        return

    # Sort by peak mAP descending
    data_rows.sort(key=lambda x: x["peak_mAP50"], reverse=True)

    table_rows = []
    for r in data_rows:
        def fmt(v, fmt_str):
            return fmt_str.format(v) if not np.isnan(float(v)) else "---"

        table_rows.append({
            "Method":       LABELS_EN[r["key"]],
            "Mean Average Precision (mAP@0.5)": f"{r['peak_mAP50']:.4f}",
            "Peak Loss":    fmt(r["peak_loss"],    "{:.4f}"),
            "Energy (J)":   fmt(r["avg_energy_j"], "{:.1f}"),
            "Latency (s)":  fmt(r["avg_tau_s"],    "{:.1f}"),
            "Total Obj.":   fmt(r["avg_joint_cost"],"{:.2f}"),
        })

    save_table("table2_objective_summary", table_rows)
    save_table_latex("table2_objective_summary", table_rows,
                     caption=CAPTION, label=LABEL)
    save_table_pdf("table2_objective_summary", table_rows,
                   title="Table 2: Objective Summary (Scenarios 2 & 3)")


def main() -> None:
    setup_style()
    print("Generating Table 2: Objective Summary")
    build()


if __name__ == "__main__":
    main()
