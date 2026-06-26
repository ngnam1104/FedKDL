"""plot_all.py — Vẽ tất cả 8 hình + 2 bảng trong một lần chạy.

Mỗi hình được xuất ra cả tiếng Anh (.images/en) và tiếng Việt (.images/vi).
Chạy từng hình riêng lẻ: python scripts/fedkdl/plot/fig1.py (hoặc fig2..fig8)

Usage:
    python scripts/fedkdl/plot/plot_all.py
"""

import sys
from pathlib import Path

# Ensure plot/ directory is on the path so fig*.py can import plot_common
sys.path.insert(0, str(Path(__file__).parent))

from plot_common import setup_style, TABLE_DIR

from fig1   import draw as draw1
from fig2   import draw as draw2
from fig3   import draw as draw3
from fig4   import draw as draw4
from fig5   import draw as draw5
from fig6   import draw as draw6
from fig7   import draw as draw7
from fig8   import draw as draw8
from table1_mobility import build as build_table1
from table2_payload import build as build_table2
from table3_objective import build as build_table3


ALL_FIGURES = [
    ("Fig 1: Connectivity & Participation", draw1),
    ("Fig 2: Mobility",                     draw2),
    ("Fig 3: Payload Comparison",           draw3),
    ("Fig 4: Compression + Objective",      draw4),
    ("Fig 5: LoRA Rank Ablation",           draw5),
    ("Fig 6: Non-IID Baselines",            draw6),
    ("Fig 7: Cooperation Ablation",         draw7),
    ("Fig 8: Gateway Refinement",           draw8),
]


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    for lang in ("en", "vi"):
        print(f"\n{'='*60}")
        print(f"  Figures [{lang.upper()}]")
        print(f"{'='*60}")
        for title, draw_fn in ALL_FIGURES:
            print(f"\n[{lang.upper()}] {title}")
            try:
                draw_fn(lang)
            except FileNotFoundError as exc:
                print(f"  [SKIP] {exc}")

    print(f"\n{'='*60}")
    print("  Tables")
    print(f"{'='*60}")
    for lang in ("en", "vi"):
        print(f"\n[{lang.upper()}] Building Tables...")
        try:
            build_table1(lang)
        except Exception as exc:
            print(f"  [ERROR] Table 1 ({lang}): {exc}")
        try:
            build_table2(lang)
        except Exception as exc:
            print(f"  [ERROR] Table 2 ({lang}): {exc}")
        try:
            build_table3(lang)
        except Exception as exc:
            print(f"  [ERROR] Table 3 ({lang}): {exc}")

    print("\nDone!")


if __name__ == "__main__":
    main()
