"""Shared data loading and publication plotting helpers.

All per-figure scripts (fig1.py … fig8.py) import from here.
plot_all.py calls all eight figure functions in sequence.

Structure
---------
- ROOT / results / metrics_final / <key>_metrics.csv   ← per-method data
- ROOT / results / metrics_final / tables_paper /       ← CSV + MD tables
- ROOT / .images / en /                                 ← English figures
- ROOT / .images / vi /                                 ← Vietnamese figures
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[3]   # FedKDL project root
METRICS_DIR = ROOT / "results" / "metrics_final"
SCALABILITY_SUMMARY = ROOT / "results" / "scalability_physics_summary.csv"
TABLE_DIR   = ROOT / ".images" / "tables"
EN_DIR      = ROOT / ".images" / "en"
VI_DIR      = ROOT / ".images" / "vi"


# ── Labels ───────────────────────────────────────────────────────────────────
LABELS_EN: dict[str, str] = {
    "centralized":      "Centralized",
    "fedavg":           "FedAvg (Flat)",
    "fedprox":          "FedProx (Flat)",
    "fedavg_hfl":       "FedAvg-HFL",
    "fedprox_hfl":      "FedProx-HFL",
    "scaffold":         "SCAFFOLD",
    "naive_lora":       "Naive LoRA",
    "flora":            "FlexLoRA",
    "top_k":            "Top-K",
    "fedkdl":           "FedKDL",
    "fedkdl_nokd":      "No Refinement",
    "proxy_ft":         "Proxy-set FT",
    "logit_kd":         "Logit KD",
    "fedkdl_nocoop":    "No Coop",
    "fedkdl_selective": "Selective Coop",
    "fedkdl_r24":       "LoRA (Backbone r=2, Neck r=4)",
    "fedkdl_r44":       "LoRA (Backbone r=4, Neck r=4)",
    "fedkdl_32bit":     "FedKDL 32-bit",
    "fedkdl_v50":       "FedKDL (50 m/r)",
    "fedkdl_v100":      "FedKDL (100 m/r)",
}

LABELS_VI: dict[str, str] = {
    "centralized":      "Học Tập Trung",
    "fedavg":           "FedAvg (Phẳng)",
    "fedprox":          "FedProx (Phẳng)",
    "fedavg_hfl":       "FedAvg-HFL",
    "fedprox_hfl":      "FedProx-HFL",
    "scaffold":         "SCAFFOLD",
    "naive_lora":       "Naive LoRA",
    "flora":            "FlexLoRA",
    "top_k":            "Top-K",
    "fedkdl":           "FedKDL",
    "fedkdl_nokd":      "Không Tinh Chỉnh",
    "proxy_ft":         "Tinh chỉnh Proxy",
    "logit_kd":         "Logit KD",
    "fedkdl_nocoop":    "Không Hợp Tác",
    "fedkdl_selective": "Hợp Tác Lựa Chọn",
    "fedkdl_r24":       "LoRA (Backbone r=2, Neck r=4)",
    "fedkdl_r44":       "LoRA (Backbone r=4, Neck r=4)",
    "fedkdl_32bit":     "FedKDL 32-bit",
    "fedkdl_v50":       "FedKDL (50 m/r)",
    "fedkdl_v100":      "FedKDL (100 m/r)",
}

# Vietnamese translations for axis / legend strings
TRANSLATIONS: dict[str, str] = {
    "Communication Round":                       "Vòng Huấn Luyện (Round)",
    "Mean Average Precision (mAP@0.5)":          "Độ Chính Xác Trung Bình (mAP@0.5)",
    "Mean Average Precision (mAP@0.5:0.95)":     "Độ Chính Xác Trung Bình (mAP@0.5:0.95)",
    "Validation Loss":                           "Hàm Mất Mát (Validation Loss)",
    "Number of AUVs in Network":                 "Số Lượng AUV Trong Mạng",
    "Connected AUVs":                            "Số Lượng AUV Kết Nối",
    "Average Uplink Payload (KiB/AUV)":          "Khối Lượng Truyền Trung Bình (KiB/AUV)",
    "Average Uplink Payload (MB/AUV)":           "Khối Lượng Truyền Trung Bình (MB/AUV)",
    "Peak mAP@0.5":                              "mAP@0.5 Cao Nhất",
    "Method":                                    "Phương Pháp",
    "Energy Cost ($\\lambda_E E$, norm.)":       "Chi Phí Năng Lượng ($\\lambda_E E$, chuẩn hóa)",
    "Latency Cost ($\\lambda_\\tau \\tau$, norm.)": "Chi Phí Độ Trễ ($\\lambda_\\tau \\tau$, chuẩn hóa)",
    "Validation Loss (norm.)":                   "Mất Mát Xác Nhận (chuẩn hóa)",
    "Normalized Value":                          "Giá Trị Chuẩn Hóa",
    "Total Objective":                           "Hàm Mục Tiêu Tổng",
    "Cost Components":                           "Các Thành Phần Chi Phí",
    "Objective Cost":                            "Chi Phí Mục Tiêu",
    "Avg. Loss":                                 "Loss Trung Bình",
    "Mobility Setting":                          "Cấu Hình Di Động",
    "Final Validation Loss":                     "Mất Mát Xác Nhận Cuối",
    "Component":                                 "Thành Phần",
    "Precision":                                 "Độ Chính Xác (Kiểu Dữ Liệu)",
    "Params":                                    "Số Lượng Tham Số",
    "Payload (MB)":                              "Khối Lượng Truyền (MB)",
    "Raw images":                                "Ảnh Gốc",
    "Full model":                                "Toàn Bộ Mô Hình",
    "LoRA factors":                              "Các Hệ Số LoRA",
    "Detection head":                            "Đầu Phát Hiện",
    "BN affine":                                 "Thành Phần BN Affine",
    "BN running":                                "Thành Phần BN Running",
    "BN running buffers":                        "Thành Phần BN Running",
    "Values":                                    "Giá Trị",
    "Indices":                                   "Chỉ Số",
    "**Total**":                                 "**Tổng Cộng**",
    "Peak Loss":                                 "Mất Mát (Loss) Cao Nhất",
    "Weighted Energy ($\\lambda_E E$)":          "Năng Lượng Có Trọng Số ($\\lambda_E E$)",
    "Weighted Latency ($\\lambda_\\tau \\tau$)": "Độ Trễ Có Trọng Số ($\\lambda_\\tau \\tau$)",
    "Energy Cost ($\\lambda_E E$)":              "Chi Phí Năng Lượng ($\\lambda_E E$)",
    "Latency Cost ($\\lambda_\\tau \\tau$)":     "Chi Phí Độ Trễ ($\\lambda_\\tau \\tau$)",
    "(a)":                                       "(a)",
    "(b)":                                       "(b)",
    "Connected AUVs vs. Network Size":           "Số lượng AUV kết nối theo Quy mô mạng",
    "Connectivity-constrained Learning (N = 30)":"Huấn luyện với hạn chế kết nối (N = 30)",
    "Learning under AUV Mobility":               "Huấn luyện dưới tác động di chuyển của AUV",
    "Average Uplink Payload vs. Mean Average Precision (mAP@0.5)": "Khối lượng truyền tải và độ chính xác trung bình (mAP@0.5)",
    "Compression-Method Learning Curves":        "Đường cong huấn luyện của các phương pháp nén",
    "Objective Cost Breakdown":                  "Phân rã chi phí mục tiêu",
    "LoRA Rank Ablation Learning Curves":        "Đường cong huấn luyện khi thay đổi hạng LoRA",
    "Non-IID Baselines Learning Curves":         "Đường cong huấn luyện của các baseline Non-IID",
    "Relay Cooperation Learning Curves":         "Đường cong huấn luyện khi hợp tác chuyển tiếp",
    "Objective Cost vs. Mean Average Precision (mAP@0.5)": "Chi phí mục tiêu và độ chính xác trung bình (mAP@0.5)",
    "mAP Comparison":                            "So sánh mAP",
    "Gateway Refinement Learning Curves":        "Đường cong huấn luyện với tinh chỉnh tại Gateway",
    "Mean Average Precision (mAP)":               "Độ Chính Xác Trung Bình (mAP)",
    "mAP@0.5":                                    "mAP@0.5",
    "mAP@0.5:0.95":                              "mAP@0.5:0.95",
}


# ── Visual palette ────────────────────────────────────────────────────────────
COLORS: dict[str, str] = {
    "centralized":      "#222222",
    "fedavg":           "#E6A817",   # yellow/amber
    "fedprox":          "#55A868",   # green
    "fedavg_hfl":       "#E6A817",   # yellow/amber
    "fedprox_hfl":      "#55A868",   # green
    "scaffold":         "#DD8452",   # orange
    "naive_lora":       "#937860",
    "flora":            "#64B5CD",
    "top_k":            "#CCB974",
    "fedkdl":           "#D62728",
    "fedkdl_nokd":      "#2C7BB6",
    "proxy_ft":         "#DD8452",
    "logit_kd":         "#55A868",
    "fedkdl_nocoop":    "#2C7BB6",
    "fedkdl_selective": "#DD8452",
    "fedkdl_r24":       "#2C7BB6",
    "fedkdl_r44":       "#DD8452",
    "fedkdl_32bit":     "#55A868",
    "fedkdl_v50":       "#DD8452",
    "fedkdl_v100":      "#55A868",
}

MARKERS = ("o", "s", "^", "D", "v", "P", "X")

# λ coefficients matching paper Table (settings.py)
LAMBDA_E   = 0.0005   # energy objective weight
LAMBDA_TAU = 0.001    # latency objective weight


# ── Style ─────────────────────────────────────────────────────────────────────
def setup_style() -> None:
    """Apply publication-quality rcParams and create output directories.

    Targets a clean IEEE/Springer look:
    - Computer Modern serif via matplotlib's built-in mathtext
    - Slightly larger base font so figures read well at half-column width
    - Tight grid, no top/right spines for bar charts (done per-ax if needed)
    - High-res PNG (300 dpi) + lossless PDF for inclusion in LaTeX
    """
    plt.rcParams.update({
        # Font — serif body, math rendered via matplotlib mathtext (CM-like)
        "font.family":          "serif",
        "font.serif":           ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset":     "dejavuserif",   # closest to CM without full LaTeX
        "font.size":            11,
        "axes.labelsize":       12,
        "axes.titlesize":       12,
        "axes.titleweight":     "bold",
        "legend.fontsize":      10,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        # Grid
        "axes.grid":            True,
        "grid.alpha":           0.30,
        "grid.linestyle":       "--",
        "grid.color":           "#CCCCCC",
        # Legend
        "legend.frameon":       True,
        "legend.framealpha":    0.90,
        "legend.edgecolor":     "#AAAAAA",
        # Lines
        "lines.linewidth":      2.0,
        "lines.markersize":     6,
        # Figure
        "figure.dpi":           100,
        "savefig.dpi":          150,
        "savefig.bbox":         "tight",
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.edgecolor":       "#444444",
        "axes.linewidth":       0.8,
    })
    EN_DIR.mkdir(parents=True, exist_ok=True)
    VI_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ── Translation helpers ───────────────────────────────────────────────────────
def T(text: str, lang: str) -> str:
    """Return Vietnamese translation when lang='vi', else original."""
    if lang == "vi" and text in TRANSLATIONS:
        return TRANSLATIONS[text]
    return text


def L(key: str, lang: str) -> str:
    """Return display label for a method key in the requested language."""
    labels = LABELS_VI if lang == "vi" else LABELS_EN
    return labels.get(key, key)


def out_dir(lang: str) -> Path:
    """Return the output directory for the given language."""
    return VI_DIR if lang == "vi" else EN_DIR


# ── Data loading ──────────────────────────────────────────────────────────────
def load_metrics(key: str) -> pd.DataFrame:
    """Load a per-method metrics CSV from results/metrics_final/.

    Returns a tidy DataFrame with columns:
        round, map50, map5095, loss, payload_kb,
        tau_s, joint_cost, energy_j
    A round-0 anchor (map50=0) is prepended if first round > 0.
    Rows with missing round or map50 are dropped.
    """
    path = METRICS_DIR / f"{key}_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    frame = pd.read_csv(path)

    round_col   = "Round"    if "Round"    in frame else "epoch"
    map50_col   = "mAP50"   if "mAP50"   in frame else "metrics/mAP50(B)"
    map5095_col = "mAP50-95" if "mAP50-95" in frame else "metrics/mAP50-95(B)"

    missing = pd.Series(np.nan, index=frame.index)

    def nc(name: str) -> pd.Series:
        return pd.to_numeric(frame[name], errors="coerce") if name in frame else missing

    if "loss" in frame:
        loss = pd.to_numeric(frame["loss"], errors="coerce")
    else:
        loss = sum(
            pd.to_numeric(frame[col], errors="coerce")
            for col in ("val/box_loss", "val/cls_loss", "val/dfl_loss")
            if col in frame
        )

    df = pd.DataFrame({
        "round":      pd.to_numeric(frame[round_col],   errors="coerce"),
        "map50":      pd.to_numeric(frame[map50_col],   errors="coerce"),
        "map5095":    pd.to_numeric(frame[map5095_col], errors="coerce"),
        "loss":       loss,
        "payload_kb": nc("avg_payload_kb"),
        "tau_s":      nc("tau_round_s"),
        "joint_cost": nc("joint_cost_round"),
        "energy_j":   nc("e_total"),
    }).dropna(subset=["round", "map50"])

    # Prepend a round-0 anchor (mAP=0) when the first round is not already 0.
    if not df.empty and df["round"].iloc[0] > 0:
        anchor = pd.DataFrame([{
            "round":      0.0,
            "map50":      0.0,
            "map5095":    0.0,
            "loss":       np.nan,
            "payload_kb": np.nan,
            "tau_s":      np.nan,
            "joint_cost": np.nan,
            "energy_j":   np.nan,
        }])
        df = pd.concat([anchor, df], ignore_index=True)

    return df


def summary_row(key: str) -> dict:
    """Return a dict of peak/final/avg statistics for a method."""
    data = load_metrics(key)
    peak_idx  = data["map50"].idxmax()
    final_idx = data["round"].idxmax()
    avg_loss = float(data["loss"].dropna().mean())
    
    if key == "centralized":
        try:
            phys_df = pd.read_csv(SCALABILITY_SUMMARY)
            cent = phys_df[(phys_df["baseline"] == "centralized") & (phys_df["N_AUV"] == 30)].iloc[0]
            avg_energy_j = float(cent["e_comm_j_mean"])
            avg_tau_s = float(cent["tau_a2g_s_mean"])
            avg_payload_kb = float(cent["payload_per_auv_kb_mean"])
        except Exception:
            avg_energy_j = np.nan
            avg_tau_s = np.nan
            avg_payload_kb = np.nan
    else:
        avg_energy_j = float(data["energy_j"].mean())
        avg_tau_s = float(data["tau_s"].mean())
        avg_payload_kb = float(data["payload_kb"].mean())
    
    # Recalculate joint cost because CSV might have been generated with older lambda values
    avg_joint_cost = avg_loss + LAMBDA_E * avg_energy_j + LAMBDA_TAU * avg_tau_s

    return {
        "file_key":       key,
        "peak_round":     int(data.loc[peak_idx, "round"]),
        "peak_mAP50":     float(data.loc[peak_idx, "map50"]),
        "peak_mAP50_95":  float(data.loc[peak_idx, "map5095"]),
        "peak_loss":      float(data.loc[peak_idx, "loss"]),
        "final_mAP50":    float(data.loc[final_idx, "map50"]),
        "final_loss":     float(data.loc[final_idx, "loss"]),
        "avg_payload_kb": avg_payload_kb,
        "avg_tau_s":      avg_tau_s,
        "avg_joint_cost": avg_joint_cost,
        "avg_energy_j":   avg_energy_j,
        "avg_loss":       avg_loss,
    }


# ── Reusable plot helpers ─────────────────────────────────────────────────────
def plot_learning(ax, keys: Iterable[str], lang: str,
                  override_labels: dict | None = None,
                  **kwargs) -> None:
    """Plot mAP@0.5 learning curves for each key on *ax*.

    Round-0 anchor (mAP=0) is prepended automatically by load_metrics() when
    the first logged round is > 0, so all curves share the same origin.
    Markers are placed every N real rounds (the anchor row is skipped).
    """
    keys_list = list(keys)
    for idx, key in enumerate(keys_list):
        data      = load_metrics(key)
        real_rows = (data["round"] > 0).sum()
        marker    = MARKERS[idx % len(MARKERS)]
        markevery = (1, max(1, real_rows // 8))  # (start, step): skip anchor at idx 0
        label = (
            override_labels[key]
            if override_labels and key in override_labels
            else L(key, lang)
        )
        ax.plot(
            data["round"], data["map50"],
            label=label,
            color=COLORS.get(key, f"C{idx}"),
            marker=marker, markevery=markevery,
            linewidth=2,
        )
    ax.set_xlabel(T("Communication Round", lang))
    ax.set_ylabel(T("Mean Average Precision (mAP@0.5)", lang))
    
    legend_loc = kwargs.get("legend_loc", "lower right")
    bbox = kwargs.get("bbox_to_anchor", None)
    n_cols = kwargs.get("ncol", min(len(keys_list), 4))

    if bbox:
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox, ncol=n_cols,
                  fontsize=8.5, framealpha=0.95, handlelength=2.2)
    else:
        ax.legend(loc=legend_loc, ncol=n_cols,
                  fontsize=8.5, framealpha=0.95, handlelength=2.2)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def add_zoom_inset(ax, keys: list[str], lang: str,
                   zoom_start_frac: float = 0.55,
                   loc: str = "lower right",
                   override_labels: dict | None = None) -> None:
    """Add a zoomed inset showing the converged region of learning curves.

    Uses ax.inset_axes() (built-in, no mpl_toolkits required) to avoid
    memory issues on large figures.  Connection lines are drawn with
    ax.indicate_inset_zoom().
    """
    all_data: dict[str, pd.DataFrame] = {}
    for key in keys:
        try:
            all_data[key] = load_metrics(key)
        except FileNotFoundError:
            pass
    if not all_data:
        return

    max_round = max(d["round"].max() for d in all_data.values())
    x_start   = zoom_start_frac * max_round

    # Inset position in axes-fraction coords: [x0, y0, width, height]
    # "lower right": bottom-right corner with small margin
    POSITIONS = {
        "lower right": [0.55, 0.09, 0.40, 0.38],
        "upper right": [0.58, 0.57, 0.40, 0.38],
        "lower left":  [0.03, 0.05, 0.40, 0.38],
        "upper left":  [0.03, 0.57, 0.40, 0.38],
    }
    bounds = POSITIONS.get(loc, POSITIONS["lower right"])
    axins  = ax.inset_axes(bounds)

    for idx, key in enumerate(keys):
        if key not in all_data:
            continue
        data   = all_data[key]
        zoom   = data[data["round"] >= x_start]
        if zoom.empty:
            continue
        label = (
            override_labels[key]
            if override_labels and key in override_labels
            else L(key, lang)
        )
        marker    = MARKERS[idx % len(MARKERS)]
        # ~4 markers across the zoom window
        n_zoom_pts = len(zoom)
        markevery  = max(1, n_zoom_pts // 4)
        axins.plot(
            zoom["round"], zoom["map50"],
            color=COLORS.get(key, f"C{idx}"),
            linewidth=1.4, label=label,
            marker=marker, markevery=markevery,
            markersize=4,
        )

    # Compute zoom y-range
    zoom_frames = [
        d[d["round"] >= x_start]
        for d in all_data.values()
        if not d[d["round"] >= x_start].empty
    ]
    if zoom_frames:
        all_zoom = pd.concat(zoom_frames)
        ymin = float(all_zoom["map50"].min())
        ymax = float(all_zoom["map50"].max())
        pad  = max((ymax - ymin) * 0.30, 0.005)
        axins.set_xlim(x_start, float(max_round))
        axins.set_ylim(ymin - pad, ymax + pad)

    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.20, linestyle="--")
    axins.set_xlabel("")
    axins.set_ylabel("")

    # Draw zoom indicator (works in mpl >= 3.3)
    try:
        ax.indicate_inset_zoom(axins, edgecolor="0.5", linewidth=0.8)
    except Exception:
        pass  # older matplotlib fallback: just leave the inset unlabelled


def grouped_cost_bars(ax, keys: list[str], lang: str, **kwargs) -> None:
    """Grouped bars: Validation Loss | λ_E·Energy | λ_τ·Latency.

    All three quantities are plotted as absolute values.
    LAMBDA_E and LAMBDA_TAU inherently scale Energy and Latency to be
    comparable to Loss.
    """
    rows      = [summary_row(k) for k in keys]
    loss_raw  = np.array([r["avg_loss"]     for r in rows])
    energy_c  = np.array([r["avg_energy_j"] for r in rows]) * LAMBDA_E
    latency_c = np.array([r["avg_tau_s"]    for r in rows]) * LAMBDA_TAU

    x = np.arange(len(keys))
    w = 0.25

    # Colors: yellow-green palette, no purple
    COLOR_LOSS    = "#C44E52"  # deep red
    COLOR_ENERGY  = "#2C7BB6"  # blue
    COLOR_LATENCY = "#55A868"  # green

    bars_loss    = ax.bar(x - w, loss_raw,    w,
                          label=T("Validation Loss", lang),
                          color=COLOR_LOSS,    edgecolor="white", linewidth=0.5)
    bars_energy  = ax.bar(x,     energy_c,  w,
                          label=T("Energy Cost ($\\lambda_E E$)", lang),
                          color=COLOR_ENERGY,  edgecolor="white", linewidth=0.5)
    bars_latency = ax.bar(x + w, latency_c, w,
                          label=T("Latency Cost ($\\lambda_\\tau \\tau$)", lang),
                          color=COLOR_LATENCY, edgecolor="white", linewidth=0.5)

    # Value labels on top of every bar
    all_vals = np.concatenate([loss_raw, energy_c, latency_c])
    y_max    = float(all_vals.max())
    for bars, vals in ((bars_loss, loss_raw), (bars_energy, energy_c), (bars_latency, latency_c)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90,
            )

    # Expand y-axis so labels fit
    ax.set_ylim(0, y_max * 1.20)

    ax.set_xticks(x)
    ax.set_xticklabels([L(k, lang) for k in keys], rotation=15, ha="right")
    ax.set_ylabel(T("Objective Cost", lang))

    legend_loc = kwargs.get("legend_loc", "upper right")
    bbox = kwargs.get("bbox_to_anchor", None)
    n_cols = kwargs.get("ncol", 1)

    if bbox:
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox, ncol=n_cols,
                  fontsize=8.5, framealpha=0.9)
    else:
        ax.legend(loc=legend_loc, ncol=n_cols, fontsize=8.5, framealpha=0.9)


def grouped_map_bars(ax, keys: list[str], lang: str, **kwargs) -> None:
    """Grouped bars: Peak mAP@0.5 | Peak mAP@0.5:0.95."""
    rows = [summary_row(k) for k in keys]
    map50 = np.array([r["peak_mAP50"] for r in rows])
    map5095 = np.array([r["peak_mAP50_95"] for r in rows])

    x = np.arange(len(keys))
    w = 0.35

    COLOR_MAP50 = "#C44E52"    # red
    COLOR_MAP5095 = "#2C7BB6"  # blue

    bars_map50 = ax.bar(x - w/2, map50, w,
                        label=T("mAP@0.5", lang),
                        color=COLOR_MAP50, edgecolor="white", linewidth=0.5)
    bars_map5095 = ax.bar(x + w/2, map5095, w,
                          label=T("mAP@0.5:0.95", lang),
                          color=COLOR_MAP5095, edgecolor="white", linewidth=0.5)

    # Value labels
    y_max = max(map50.max(), map5095.max())
    for bars, vals in ((bars_map50, map50), (bars_map5095, map5095)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=0,
            )

    ax.set_ylim(0, y_max + 0.12)
    ax.set_xticks(x)
    ax.set_xticklabels([L(k, lang) for k in keys], rotation=15, ha="right")
    ax.set_ylabel(T("Mean Average Precision (mAP)", lang))

    legend_loc = kwargs.get("legend_loc", "upper right")
    bbox = kwargs.get("bbox_to_anchor", None)
    n_cols = kwargs.get("ncol", 1)

    if bbox:
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox, ncol=n_cols,
                  fontsize=8.5, framealpha=0.9)
    else:
        ax.legend(loc=legend_loc, ncol=n_cols,
                  fontsize=8.5, framealpha=0.9)
    ax.set_axisbelow(True)


def grouped_dual_bar_map(ax_left, keys: list[str], lang: str,
                         cost_col: str = "avg_joint_cost",
                         cost_label: str = "Objective Cost",
                         **kwargs) -> None:
    """Two grouped bars per method (cost left Y, peak mAP right Y) — NO line.

    Both metrics are rendered as bars, satisfying the "cột ghép" convention.
    """
    rows     = [summary_row(k) for k in keys]
    costs    = np.array([r[cost_col]    for r in rows])
    peak_map = np.array([r["peak_mAP50"] for r in rows])
    labels   = [L(k, lang) for k in keys]

    x = np.arange(len(keys))
    w = 0.36

    COLOR_COST = "#2C7BB6"   # blue
    COLOR_MAP  = "#C44E52"   # red

    bars_cost = ax_left.bar(x - w / 2 - 0.02, costs, w,
                            color=COLOR_COST, edgecolor="white", linewidth=0.5,
                            label=T(cost_label, lang))

    ax_right = ax_left.twinx()
    bars_map  = ax_right.bar(x + w / 2 + 0.02, peak_map, w,
                             color=COLOR_MAP, edgecolor="white", linewidth=0.5,
                             alpha=0.88, label="mAP@0.5")

    # Expand ylim first so annotation fits
    ax_left.set_ylim(0, max(costs) * 1.25)
    mlo, mhi = min(peak_map), max(peak_map)
    ax_right.set_ylim(max(0, mlo - 0.04), mhi + 0.05)

    # Value labels on bars
    c_yrange = ax_left.get_ylim()[1] - ax_left.get_ylim()[0]
    for bar, v in zip(bars_cost, costs):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + c_yrange * 0.01,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=7, color=COLOR_COST,
        )
    m_yrange = ax_right.get_ylim()[1] - ax_right.get_ylim()[0]
    for bar, v in zip(bars_map, peak_map):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + m_yrange * 0.01,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=7, color=COLOR_MAP,
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=15, ha="right")
    ax_left.set_ylabel(T(cost_label, lang), color=COLOR_COST)
    ax_right.set_ylabel(T("Mean Average Precision (mAP@0.5)", lang), color=COLOR_MAP)
    ax_left.tick_params(axis="y", colors=COLOR_COST)
    ax_right.tick_params(axis="y", colors=COLOR_MAP)

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()

    legend_loc = kwargs.get("legend_loc", "best")
    bbox = kwargs.get("bbox_to_anchor", None)
    n_cols = kwargs.get("ncol", 1)

    if bbox:
        ax_left.legend(h1 + h2, l1 + l2, loc=legend_loc, bbox_to_anchor=bbox,
                       ncol=n_cols, fontsize=9)
    else:
        ax_left.legend(h1 + h2, l1 + l2, loc=legend_loc, ncol=n_cols, fontsize=9)


# ── Save helpers ─────────────────────────────────────────────────────────────
def save_figure(fig, filename: str, lang: str) -> None:
    """Save figure as PNG (150 dpi) + PDF (vector) to language output dir."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except Exception:
            pass  # inset_axes may raise; ignore and save anyway
    d = out_dir(lang)
    png_path = d / f"{filename}.png"
    pdf_path = d / f"{filename}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {png_path}")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved {pdf_path}")
    plt.close(fig)


def save_table(name: str, rows: list[dict], lang: str = "en") -> None:
    """Write rows to the language tables dir as both CSV and Markdown."""
    if not rows:
        return
    fields   = list(rows[0].keys())
    d = out_dir(lang) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    csv_path = d / f"{name}.csv"
    md_path  = d / f"{name}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(fields) + " |\n")
        fh.write("| " + " | ".join("---" for _ in fields) + " |\n")
        for row in rows:
            vals = []
            for f in fields:
                v = row[f]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    vals.append("")
                else:
                    vals.append(str(v))
            fh.write("| " + " | ".join(vals) + " |\n")

    print(f"  Saved {csv_path}")
    print(f"  Saved {md_path}")


def save_table_latex(name: str, rows: list[dict],
                     caption: str = "", label: str = "", lang: str = "en") -> None:
    """Write rows as a publication-ready LaTeX table (booktabs) to language tables dir."""
    if not rows:
        return
    fields  = list(rows[0].keys())
    n_cols  = len(fields)
    col_fmt = "l" + "r" * (n_cols - 1)

    d = out_dir(lang) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    tex_path = d / f"{name}.tex"

    def esc(s: str) -> str:
        return (s.replace("_", r"\_")
                 .replace("&", r"\&")
                 .replace("%", r"\%")
                 .replace("#", r"\#"))

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    if caption:
        lines.append(f"  \\caption{{{esc(caption)}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"    \toprule")

    header = " & ".join(f"\\textbf{{{esc(f)}}}" for f in fields)
    lines.append(f"    {header} \\\\")
    lines.append(r"    \midrule")

    for row in rows:
        vals = []
        for f in fields:
            v = row[f]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append("---")
            else:
                vals.append(esc(str(v)))
        lines.append("    " + " & ".join(vals) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  Saved {tex_path}")


def save_table_pdf(name: str, rows: list[dict], title: str = "", lang: str = "en") -> None:
    """Render a table as a standalone, publication-quality PDF using matplotlib.

    The PDF is saved alongside the other table outputs in the language tables dir.
    """
    if not rows:
        return
    fields = list(rows[0].keys())
    n_rows = len(rows)
    n_cols = len(fields)

    # Build cell data (header + body)
    cell_text = [[str(row.get(f, "")) for f in fields] for row in rows]

    # Column width proportional to content
    col_widths = []
    for f in fields:
        max_len = max(len(f), max(len(str(r.get(f, ""))) for r in rows))
        col_widths.append(max(0.10, max_len * 0.015))

    fig_w = max(6.0, sum(col_widths) + 0.4)
    fig_h = max(1.5, (n_rows + 1) * 0.38 + (0.5 if title else 0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=fields,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Style header
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2C7BB6")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#FFFFFF")

    # Style alternating rows
    for i in range(1, n_rows + 1):
        bg = "#F0F4FA" if i % 2 == 0 else "#FFFFFF"
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#DDDDDD")

    # Scale row heights
    tbl.scale(1, 1.4)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=0.98)

    d = out_dir(lang) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    pdf_path = d / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {pdf_path}")
