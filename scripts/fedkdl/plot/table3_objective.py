"""Bảng 3: Objective Summary.

Outputs:
  - results/metrics_final/tables_paper/table3_objective_summary.csv
  - results/metrics_final/tables_paper/table3_objective_summary.md
  - results/metrics_final/tables_paper/table3_objective_summary.tex
  - results/metrics_final/tables_paper/table3_objective_summary.pdf
"""

from plot_common import (
    LABELS_EN, LAMBDA_E, LAMBDA_TAU, summary_row,
    save_table, save_table_latex, save_table_pdf, setup_style, T, L
)

KEYS = [
    "centralized", "fedavg_hfl", "fedprox_hfl", "scaffold",
    "naive_lora", "flora", "top_k", "fedkdl"
]

CAPTION = (
    "Peak mAP@0.5, average validation loss, and average weighted physical-layer cost components "
    "for all methods in Scenarios~2 and~3, sorted by peak mAP@0.5 (descending). "
    r"Total Objective $= \overline{\mathcal{L}} + \lambda_E \bar{E} + \lambda_\tau \bar{\tau}$ with "
    r"$\lambda_E = 0.0005$ and $\lambda_\tau = 0.001$ (all averaged over training rounds)."
)
LABEL = "tab:objective_summary"

def fmt(val: float, fmt_str: str) -> str:
    import numpy as np
    return "---" if np.isnan(val) else fmt_str.format(val)

def build(lang: str = "en") -> None:
    data_rows = [summary_row(k) for k in KEYS]
    data_rows.sort(key=lambda r: r["peak_mAP50"], reverse=True)

    table_rows = []
    for r in data_rows:
        weighted_energy = r["avg_energy_j"] * LAMBDA_E
        weighted_latency = r["avg_tau_s"] * LAMBDA_TAU
        
        table_rows.append({
            T("Method", lang):       L(r["file_key"], lang),
            T("Mean Average Precision (mAP@0.5)", lang):      f"{r['peak_mAP50']:.4f}",
            T("Avg. Loss", lang):    fmt(r["avg_loss"],    "{:.4f}"),
            T("Weighted Energy ($\\lambda_E E$)", lang): fmt(weighted_energy, "{:.2f}"),
            T("Weighted Latency ($\\lambda_\\tau \\tau$)", lang): fmt(weighted_latency, "{:.2f}"),
            T("Total Objective", lang):   fmt(r["avg_joint_cost"],"{:.2f}"),
        })
    
    save_table("table3_objective_summary", table_rows, lang)
    save_table_latex("table3_objective_summary", table_rows, caption=T(CAPTION, lang), label=LABEL, lang=lang)
    
    setup_style()
    save_table_pdf("table3_objective_summary", table_rows, title=T(CAPTION, lang), lang=lang)

if __name__ == "__main__":
    build("en")
