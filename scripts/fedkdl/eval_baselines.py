"""
eval_baselines.py
So sánh nhanh các baseline 2D sau khi chạy Scenario 2 & 3.
Đọc kết quả từ results/logs_kdl/*.json (schema từ utils.log_export).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import os

import numpy as np
import pandas as pd


def _metrics_dict(data: dict) -> dict:
    metrics = data.get("metrics", {})
    if isinstance(metrics, list):
        return pd.DataFrame(metrics).to_dict(orient="list") if metrics else {}
    return metrics


def summarize(name: str, history: dict) -> dict:
    """Tính các chỉ số tổng kết từ history dict (cột theo round)."""
    rounds = history.get("round", [])
    if not rounds:
        return {}
    mAP = history.get("map") or history.get("mAP") or []
    alive = history.get("alive") or []
    energy = history.get("energy_cumul_J") or history.get("e_cumul") or []
    payload = history.get("avg_payload_kb") or []
    return {
        "baseline": name,
        "final_mAP": mAP[-1] if mAP else 0,
        "peak_mAP": max(mAP) if mAP else 0,
        "final_alive": alive[-1] if alive else 0,
        "survival_rate_%": 100 * alive[-1] / max(alive[0], 1) if alive else 0,
        "avg_payload_KB": float(np.mean(payload)) if payload else 0,
        "total_energy_J": energy[-1] if energy else 0,
        "rounds_run": len(rounds),
    }


def print_table(summaries: list):
    if not summaries:
        print("Không có dữ liệu.")
        return
    headers = list(summaries[0].keys())
    col_w = max(18, max(len(h) for h in headers) + 2)
    row_w = 16

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
    parser = argparse.ArgumentParser(description="Evaluate and compare FedKDL 2D baselines.")
    parser.add_argument(
        "--results-dir",
        default="results/logs_kdl",
        help="Thư mục chứa file JSON kết quả",
    )
    args = parser.parse_args()

    summaries = []
    if not os.path.isdir(args.results_dir):
        print(f"Không tìm thấy thư mục '{args.results_dir}'.")
        raise SystemExit(1)

    for fname in sorted(os.listdir(args.results_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(args.results_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        name = meta.get("baseline", fname.replace(".json", ""))
        metrics = _metrics_dict(data)
        row = summarize(name, metrics)
        if row:
            summaries.append(row)

    if not summaries:
        print(
            f"Không tìm thấy file JSON hợp lệ trong '{args.results_dir}'. "
            "Chạy run_kdl_experiments.sh trước."
        )
    else:
        print_table(sorted(summaries, key=lambda x: -x["final_mAP"]))
