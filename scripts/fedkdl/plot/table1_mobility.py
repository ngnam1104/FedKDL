"""Bảng 1: Mobility mAP.

Outputs:
  - results/metrics_final/tables_paper/table1_mobility.csv
  - results/metrics_final/tables_paper/table1_mobility.md
  - results/metrics_final/tables_paper/table1_mobility.tex
  - results/metrics_final/tables_paper/table1_mobility.pdf
"""

from plot_common import (
    LABELS_EN, summary_row, save_table, save_table_latex, save_table_pdf, setup_style, T, L
)

KEYS = ["fedkdl", "fedkdl_v50", "fedkdl_v100"]

CAPTION = (
    "Peak and final Mean Average Precision (mAP) under varying AUV mobility speeds."
)
LABEL = "tab:mobility_map"


def build(lang: str = "en") -> None:
    rows = []
    for k in KEYS:
        r = summary_row(k)
        rows.append({
            T("Mobility Setting", lang):     L(k, lang),
            T("Peak mAP@0.5", lang):         f"{r['peak_mAP50']:.4f}",
            T("Peak mAP@0.5:0.95", lang):    f"{r['peak_mAP50_95']:.4f}",
            T("Final Validation Loss", lang):f"{r['final_loss']:.4f}",
        })
    
    save_table("table1_mobility", rows, lang)
    save_table_latex("table1_mobility", rows, caption=T(CAPTION, lang), label=LABEL, lang=lang)
    
    setup_style()
    save_table_pdf("table1_mobility", rows, title=T(CAPTION, lang), lang=lang)

if __name__ == "__main__":
    build("en")
