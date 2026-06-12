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
OUT_DIR = RESULTS / "plots"

LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedavg_hfl": "FedAvg-HFL",
    "fedprox_hfl": "FedProx-HFL",
    "topk_grad": "Top-K",
    "flora": "FLORA",
    "naive_lora": "Naive LoRA",
    "scaffold": "SCAFFOLD",
    "fedkdl": "FedKDL",
    "fedkdl_nocoop": "FedKDL-NoCoop",
    "fedkdl_selective": "FedKDL-Selective",
    "fedkdl_nokd": "No KD",
    "logit_kd": "Logit KD",
    "fedkdl_proxy_ft": "Proxy fine-tuning",
    "centralized": "Centralized",
}

COLORS = {
    "fedavg": "#C44E52",
    "fedprox": "#DD8452",
    "fedavg_hfl": "#8172B3",
    "topk_grad": "#CCB974",
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
    axes[0].set(xlabel="Communication Round", ylabel="Mean Average Precision (mAP@0.5)")
    axes[1].set(xlabel="Communication Round", ylabel="Validation Loss")
    axes[0].legend()
    axes[1].legend()
    save(fig, filename)
