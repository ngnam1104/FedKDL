from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load files
# =========================
file1 = Path("D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/results/lora_vs_nolora/results.csv")
file2 = Path("D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/results/lora_vs_nolora/results_yolov12_fullfinetune.csv")

df_lora = pd.read_csv(file1)
df_full = pd.read_csv(file2)

# =========================
# Normalize column names
# =========================
df_lora.columns = [c.strip() for c in df_lora.columns]
df_full.columns = [c.strip() for c in df_full.columns]

# =========================
# Detect epoch columns
# =========================
epoch_col_lora = [c for c in df_lora.columns if "epoch" in c.lower()][0]
epoch_col_full = [c for c in df_full.columns if "epoch" in c.lower()][0]

# =========================
# Fix Full Finetune epochs
# Phase 1: 2 -> 86
# Phase 2: 1 -> 113
# =========================
epochs = df_full[epoch_col_full].tolist()

fixed_epochs = []
offset = 0
prev = epochs[0]

for e in epochs:
    if e < prev:
        offset += prev
    fixed_epochs.append(e + offset)
    prev = e

df_full["fixed_epoch"] = fixed_epochs

# =========================
# Truncate LoRA by Full FT max epoch
# =========================
max_epoch = df_full["fixed_epoch"].max()

df_lora_plot = df_lora[
    df_lora[epoch_col_lora] <= max_epoch
].copy()

# =========================
# Candidate metric names
# =========================
metric_map = {
    "Precision": [
        c for c in df_lora.columns
        if "precision" in c.lower()
    ],

    "Recall": [
        c for c in df_lora.columns
        if "recall" in c.lower()
    ],

    "mAP50": [
        c for c in df_lora.columns
        if "map50" in c.lower() and "95" not in c.lower()
    ],

    "mAP50-95": [
        c for c in df_lora.columns
        if "95" in c.lower()
    ],
}

# =========================
# Plot comparisons
# =========================
output_paths = []

save_dir = Path(
    "D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/results/lora_vs_nolora"
)

for metric_name, cols in metric_map.items():

    if not cols:
        continue

    col = cols[0]

    plt.figure(figsize=(8, 5))

    # LoRA
    plt.plot(
        df_lora_plot[epoch_col_lora],
        df_lora_plot[col],
        label="LoRA"
    )

    # Full Finetune
    if col in df_full.columns:
        plt.plot(
            df_full["fixed_epoch"],
            df_full[col],
            label="Full Finetune"
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison")

    plt.legend()
    plt.grid(True)

    out = save_dir / f"{metric_name.replace('/', '_')}_comparison.png"

    plt.savefig(out, bbox_inches="tight")
    plt.close()

    output_paths.append(out)

# =========================
# Summary table
# =========================
summary_rows = []

for metric_name, cols in metric_map.items():

    if not cols:
        continue

    col = cols[0]

    # LoRA
    lora_best = df_lora_plot[col].max()
    lora_final = df_lora_plot[col].iloc[-1]

    # Full FT
    full_best = None
    full_final = None

    if col in df_full.columns:
        full_best = df_full[col].max()
        full_final = df_full[col].iloc[-1]

    summary_rows.append({
        "Metric": metric_name,

        "LoRA Best": lora_best,
        "LoRA Final": lora_final,

        "FullFinetune Best": full_best,
        "FullFinetune Final": full_final
    })

summary_df = pd.DataFrame(summary_rows)

# =========================
# Save summary
# =========================
summary_csv = save_dir / "metrics_summary.csv"

summary_df.to_csv(summary_csv, index=False)

# =========================
# Save fixed full-ft csv
# =========================
fixed_csv = save_dir / "results_yolov12_fullfinetune_fixed.csv"

df_full.to_csv(fixed_csv, index=False)

# =========================
# Print outputs
# =========================
print(summary_df)

print("\nGenerated files:")
for p in output_paths:
    print(p)

print("\nSummary CSV:")
print(summary_csv)

print("\nFixed Full FT CSV:")
print(fixed_csv)