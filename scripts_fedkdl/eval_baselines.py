"""
eval_baselines.py
So sánh nhanh các baseline sau khi chạy Scenario 2 & 3.
Đọc kết quả từ dict history hoặc file JSON và in bảng tổng kết.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np


def summarize(name: str, history: dict) -> dict:
    """Tính các chỉ số tổng kết từ history dict."""
    rounds = history.get('round', [])
    if not rounds:
        return {}
    return {
        'baseline': name,
        'final_mAP': history['map'][-1] if history.get('map') else 0,
        'peak_mAP': max(history['map']) if history.get('map') else 0,
        'final_alive': history['alive'][-1] if history.get('alive') else 0,
        'survival_rate_%': 100 * history['alive'][-1] / max(history['alive'][0], 1) if history.get('alive') else 0,
        'avg_payload_KB': np.mean(history['avg_payload_kb']) if history.get('avg_payload_kb') else 0,
        'total_energy_J': history['energy_cumul_J'][-1] if history.get('energy_cumul_J') else 0,
        'rounds_run': len(rounds),
    }


def print_table(summaries: list):
    if not summaries:
        print("Không có dữ liệu.")
        return
    headers = list(summaries[0].keys())
    col_w = max(18, max(len(h) for h in headers) + 2)
    row_w = 16

    # Header
    print("\n" + "=" * (len(headers) * row_w + col_w))
    print(f"{'Baseline':<{col_w}}" + "".join(f"{h:>{row_w}}" for h in headers[1:]))
    print("=" * (len(headers) * row_w + col_w))

    for s in summaries:
        line = f"{s['baseline']:<{col_w}}"
        for k in headers[1:]:
            v = s[k]
            if isinstance(v, float):
                line += f"{v:>{row_w}.3f}"
            else:
                line += f"{str(v):>{row_w}}"
        print(line)
    print("=" * (len(headers) * row_w + col_w) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare FedKDL baselines.")
    parser.add_argument("--results-dir", default="results", help="Thư mục chứa file JSON kết quả")
    args = parser.parse_args()

    summaries = []
    results_dir = args.results_dir
    for fname in os.listdir(results_dir):
        if fname.endswith('.json'):
            fpath = os.path.join(results_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            name = fname.replace('.json', '')
            summaries.append(summarize(name, data))

    if not summaries:
        print(f"Không tìm thấy file JSON trong '{results_dir}'. Chạy scenario scripts trước.")
    else:
        print_table(sorted(summaries, key=lambda x: -x['final_mAP']))
