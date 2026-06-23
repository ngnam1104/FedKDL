"""Shared data loading and publication plotting helpers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PHYSICS_RAW = RESULTS / "scalability_physics_raw.csv"
PHYSICS_SUMMARY = RESULTS / "scalability_physics_summary.csv"
LEARNING_DIR = RESULTS / "learning_curves"
VIETNAMESE_MODE = True  # Toggle this to False for English plots

if VIETNAMESE_MODE:
    OUT_DIR = ROOT / ".images" / "vi"
else:
    OUT_DIR = RESULTS / "plots"

TRANSLATIONS = {
    "Communication Round": "Vòng Huấn Luyện (Round)",
    "Mean Average Precision (mAP@0.5)": "Độ Chính Xác Trung Bình (mAP@0.5)",
    "Validation Loss": "Hàm Mất Mát (Validation Loss)",
    "Number of AUVs": "Số Lượng AUVs",
    "Participation Rate (%)": "Tỉ Lệ Tham Gia (%)",
    "Participating AUVs": "Số Lượng AUVs Tham Gia",
    "Joint Objective Cost": "Giá Trị Hàm Mục Tiêu",
    "Baseline Method": "Các Phương Pháp Đề Xuất",
    "Cost Contribution at N=60": "Thành Phần Chi Phí tại N=60",
    "Payload per AUV (MB)": "Khối Lượng Dữ Liệu Mỗi AUV (MB)",
    "Best mAP@0.5": "mAP@0.5 Tốt Nhất",
    "Compression / PEFT Method": "Phương Pháp Nén / PEFT",
    "Transmission Payload per AUV (MB)": "Khối Lượng Dữ Liệu Truyền Mỗi AUV (MB)",
    "Relay Cooperation Strategy": "Chiến Lược Hợp Tác Relay",
    "Mean Average Precision": "Độ Chính Xác Trung Bình (mAP@0.5)",
    "Detection Quality (mAP)": "Chất Lượng Nhận Diện (mAP)",
    "FedKDL-Nearest": "FedKDL (Gần Nhất)",
}

def T(text: str) -> str:
    """Translate text to Vietnamese if VIETNAMESE_MODE is True."""
    if VIETNAMESE_MODE and text in TRANSLATIONS:
        return TRANSLATIONS[text]
    return text

LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedavg_hfl": "FedAvg-HFL",
    "fedprox_hfl": "FedProx-HFL",
    "topk_grad": "Top-K",
    "topk_grad_10": "Top-K 10%",
    "topk_grad_20": "Top-K 20%",
    "flora": "FLORA",
    "naive_lora": "Naive LoRA",
    "scaffold": "SCAFFOLD",
    "fedkdl": "FedKDL",
    "fedkdl_nocoop": "FedKDL (Không Hợp Tác)" if VIETNAMESE_MODE else "FedKDL-NoCoop",
    "fedkdl_selective": "FedKDL (Lựa Chọn)" if VIETNAMESE_MODE else "FedKDL-Selective",
    "fedkdl_nokd": "Không dùng KD" if VIETNAMESE_MODE else "No KD",
    "logit_kd": "KD qua Logit" if VIETNAMESE_MODE else "Logit KD",
    "fedkdl_proxy_ft": "Tinh Chỉnh trên tập Proxy" if VIETNAMESE_MODE else "Proxy fine-tuning",
    "centralized": "Học Tập Trung" if VIETNAMESE_MODE else "Centralized",
}

COLORS = {
    "fedavg": "#C44E52",
    "fedprox": "#DD8452",
    "fedavg_hfl": "#8172B3",
    "topk_grad": "#CCB974",
    "topk_grad_10": "#B7A04E",
    "topk_grad_20": "#9E8732",
    "flora": "#64B5CD",
    "naive_lora": "#937860",
    "scaffold": "#DA8BC3",
    "fedkdl": "#2A9D8F",
    "fedkdl_nocoop": "#4C72B0",
    "fedkdl_selective": "#55A868",
    "fedkdl_nokd": "#4C72B0",
    "logit_kd": "#8172B3",
    "fedkdl_proxy_ft": "#DD8452",
    "centralized": "#333333",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    return pd.read_csv(PHYSICS_RAW)


def load_summary():
    return pd.read_csv(PHYSICS_SUMMARY)


def load_learning(baseline):
    path = LEARNING_DIR / f"results_{baseline}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing real learning curve: {path}. "
            "Mock learning curves are intentionally not used."
        )
    frame = pd.read_csv(path)
    round_col = "Round" if "Round" in frame else "epoch"
    map50_col = (
        "mAP50" if "mAP50" in frame else "metrics/mAP50(B)"
    )
    map5095_col = (
        "mAP50-95"
        if "mAP50-95" in frame
        else "metrics/mAP50-95(B)"
    )
    if "loss" in frame:
        loss = frame["loss"]
    else:
        loss = sum(
            frame[column]
            for column in (
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
            )
        )
    return pd.DataFrame(
        {
            "round": frame[round_col],
            "map50": frame[map50_col],
            "map5095": frame[map5095_col],
            "loss": loss,
            "precision": frame.get("Prec", frame.get("metrics/precision(B)")),
            "recall": frame.get("Rec", frame.get("metrics/recall(B)")),
        }
    )


def physics_series(summary, baseline, metric):
    rows = summary[summary["baseline"] == baseline].sort_values("N_AUV")
    return (
        rows["N_AUV"].to_numpy(),
        rows[f"{metric}_mean"].to_numpy(),
        rows[f"{metric}_std"].fillna(0).to_numpy(),
    )


def plot_mean_std(ax, summary, baselines, metric):
    for index, baseline in enumerate(baselines):
        x, mean, std = physics_series(summary, baseline, metric)
        color = COLORS.get(baseline, f"C{index}")
        ax.plot(
            x,
            mean,
            label=LABELS.get(baseline, baseline),
            color=color,
            marker=MARKERS[index % len(MARKERS)],
            linewidth=2,
        )
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)


def save(fig, filename):
    fig.tight_layout()
    if VIETNAMESE_MODE:
        filename = f"{filename}_vi"
    png = OUT_DIR / f"{filename}.png"
    pdf = OUT_DIR / f"{filename}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def plot_learning_panels(baselines, filename, title):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for index, baseline in enumerate(baselines):
        frame = load_learning(baseline)
        label = LABELS.get(baseline, baseline)
        color = COLORS.get(baseline, f"C{index}")
        marker = MARKERS[index % len(MARKERS)]
        markevery = max(1, len(frame) // 10)
        axes[0].plot(
            frame["round"], frame["map50"], label=label, color=color,
            marker=marker, markevery=markevery, linewidth=2,
        )
        axes[1].plot(
            frame["round"], frame["loss"], label=label, color=color,
            marker=marker, markevery=markevery, linewidth=2,
        )
    axes[0].set(xlabel=T("Communication Round"), ylabel=T("Mean Average Precision (mAP@0.5)"))
    axes[1].set(xlabel=T("Communication Round"), ylabel=T("Validation Loss"))
    axes[0].legend()
    axes[1].legend()
    save(fig, filename)
