"""Bảng 1: Payload Breakdown for Scenario 2 methods.

Outputs:
  - results/metrics_final/tables_paper/table1_payload_breakdown.csv
  - results/metrics_final/tables_paper/table1_payload_breakdown.md
  - results/metrics_final/tables_paper/table1_payload_breakdown.tex  (LaTeX)

Usage:
    python scripts/fedkdl/plot/table1.py
"""

from plot_common import (
    LABELS_EN, summary_row, save_table, save_table_latex, save_table_pdf, setup_style,
)

KEYS = ["fedavg_hfl", "naive_lora", "flora", "top_k", "fedkdl"]

NOTES = {
    "fedavg_hfl":  "Full model",
    "naive_lora":  "LoRA only",
    "flora":       "FLoRA",
    "top_k":       "Sparse upload",
    "fedkdl":      "Delta-INT8 LoRA + Head + BN",
}

CAPTION = (
    "Average uplink payload (MB per AUV per round) and peak mAP@0.5 "
    "for each compression / PEFT method in Scenario~2."
)
LABEL = "tab:payload_breakdown"


def build() -> None:
    rows = []
    for k in KEYS:
        r = summary_row(k)
        payload_mb = r["avg_payload_kb"] / 1024.0
        rows.append({
            "Method":               LABELS_EN[k],
            "Avg Payload (MB/AUV)": f"{payload_mb:.2f}",
            "Mean Average Precision (mAP@0.5)": f"{r['peak_mAP50']:.4f}",
            "Compression Note":     NOTES.get(k, ""),
        })
    save_table("table1_payload_breakdown", rows)
    save_table_latex("table1_payload_breakdown", rows,
                     caption=CAPTION, label=LABEL)
    save_table_pdf("table1_payload_breakdown", rows,
                   title="Table 1: Payload Breakdown (Scenario 2)")


def main() -> None:
    setup_style()
    print("Generating Table 1: Payload Breakdown")
    build()


if __name__ == "__main__":
    main()
