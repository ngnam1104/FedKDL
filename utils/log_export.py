"""
Chuẩn hóa export log JSON sau mỗi lần train — dùng bởi main_trainer*.py và scripts plot.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def build_experiment_bundle(sim: Any, history: List[dict], metadata: dict) -> dict:
    """
    Gom metrics theo round + energy/latency trackers thành schema thống nhất.

    Aliases cho scripts:
      - 1D: Train_Loss, Participation
      - 2D: map, energy_cumul_J
    """
    metrics_df = pd.DataFrame(history) if history else pd.DataFrame()
    energy_df = sim.energy_tracker.get_dataframe()
    latency_df = sim.latency_tracker.get_dataframe()

    metrics: Dict[str, list] = (
        metrics_df.to_dict(orient="list") if not metrics_df.empty else {}
    )

    if "loss" in metrics and "Train_Loss" not in metrics:
        metrics["Train_Loss"] = metrics["loss"]

    n_auvs = metadata.get("N")
    if n_auvs and "alive" in metrics and "Participation" not in metrics:
        metrics["Participation"] = [a / n_auvs for a in metrics["alive"]]

    if "mAP" in metrics and "map" not in metrics:
        metrics["map"] = metrics["mAP"]
    if "e_cumul" in metrics and "energy_cumul_J" not in metrics:
        metrics["energy_cumul_J"] = metrics["e_cumul"]

    return {
        "metadata": metadata,
        "metrics": metrics,
        "latency_history": (
            latency_df.to_dict(orient="list") if not latency_df.empty else {}
        ),
        "energy_consumption": (
            energy_df.to_dict(orient="list") if not energy_df.empty else {}
        ),
    }
