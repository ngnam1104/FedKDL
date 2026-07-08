"""Parse real training logs into compact demo replay events."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
AUV_PATTERN = re.compile(r"\[LocalSGD\]\[AUV\s+(\d+)\]", re.IGNORECASE)
PROGRESS_PATTERN = re.compile(
    r"^\s*(?P<epoch>\d+)\s*/\s*(?P<epochs>\d+)"
    r"\s+\S+"
    r"\s+(?P<box>[0-9]+(?:\.[0-9]+)?)"
    r"\s+(?P<cls>[0-9]+(?:\.[0-9]+)?)"
    r"\s+(?P<dfl>[0-9]+(?:\.[0-9]+)?)"
    r"\s+(?P<instances>\d+)"
    r"\s+(?P<size>\d+):\s+\d+%"
    r".*?\s(?P<batch>\d+)\s*/\s*(?P<batches>\d+)"
)
KD_EPOCH_PATTERN = re.compile(
    r"\[KD Epoch\s+(?P<epoch>\d+)\]\s+"
    r"Supervised:\s+(?P<supervised>[0-9]+(?:\.[0-9]+)?)\s+\|\s+"
    r"KD Only:\s+(?P<kd_only>[0-9]+(?:\.[0-9]+)?)\s+\|\s+"
    r"Box:\s+(?P<box>[0-9]+(?:\.[0-9]+)?)\s+\|\s+"
    r"KL:\s+(?P<kl>[0-9]+(?:\.[0-9]+)?)"
    r"(?:\s+\|\s+LoRA_Proj:\s+(?P<lora>[0-9]+(?:\.[0-9]+)?))?"
    r".*?Total:\s+(?P<total>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
KD_SUMMARY_PATTERN = re.compile(
    r"\[Gateway KD\]\s+Summary\s+\|\s+"
    r"Box=(?P<box>[0-9]+(?:\.[0-9]+)?),\s+"
    r"KL=(?P<kl>[0-9]+(?:\.[0-9]+)?),\s+"
    r"(?:LoRA_Proj=(?P<lora>[0-9]+(?:\.[0-9]+)?),\s+)?"
    r"KD/Sup=(?P<kd_ratio>[0-9]+(?:\.[0-9]+)?),\s+"
    r"KD Contrib=(?P<kd_contrib>[0-9]+(?:\.[0-9]+)?),\s+"
    r"Total=(?P<total>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def clean_log_segment(segment: str) -> str:
    text = ANSI_PATTERN.sub("", segment)
    return text.replace("\x1b[K", "").strip()


def parse_training_log(
    log_path: Path,
    *,
    max_rounds: int = 10,
    max_events_per_auv_round: int = 36,
    max_kd_progress_events_per_round: int = 24,
) -> dict[str, Any]:
    """Return downsampled batch-loss events from an Ultralytics/FedKDL log."""
    log_path = Path(log_path)
    if not log_path.exists():
        return {
            "available": False,
            "source": str(log_path),
            "rounds": [],
            "events": [],
            "reason": "log file not found",
        }

    events: list[dict[str, Any]] = []
    kd_events: list[dict[str, Any]] = []
    current_round = 0
    total_rounds = 0
    current_auv: int | None = None
    in_gateway_kd = False
    step = 0
    kd_step = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            for raw_segment in raw_line.replace("\r", "\n").split("\n"):
                line = clean_log_segment(raw_segment)
                if not line:
                    continue

                if "[Simulator]" in line and "/" in line:
                    numbers = re.findall(r"(\d+)\s*/\s*(\d+)", line)
                    if numbers:
                        current_round = int(numbers[-1][0])
                        total_rounds = int(numbers[-1][1])
                        current_auv = None
                        if current_round > max_rounds:
                            return _finalize(
                                log_path,
                                events,
                                kd_events,
                                total_rounds,
                                max_events_per_auv_round,
                                max_kd_progress_events_per_round,
                            )
                    continue

                if current_round > 0 and current_round <= max_rounds:
                    if "[Gateway KD] Distilling" in line:
                        in_gateway_kd = True
                        current_auv = None
                        continue
                    if "[Gateway KD] Done" in line or "[Simulator] Evaluating" in line:
                        in_gateway_kd = False

                    if in_gateway_kd:
                        match_kd_progress = PROGRESS_PATTERN.search(line)
                        if match_kd_progress:
                            box = float(match_kd_progress.group("box"))
                            cls = float(match_kd_progress.group("cls"))
                            dfl = float(match_kd_progress.group("dfl"))
                            kd_step += 1
                            kd_events.append({
                                "type": "progress",
                                "step": kd_step,
                                "round": current_round,
                                "epoch": int(match_kd_progress.group("epoch")),
                                "epochs": int(match_kd_progress.group("epochs")),
                                "batch": int(match_kd_progress.group("batch")),
                                "batches": int(match_kd_progress.group("batches")),
                                "box_loss": round(box, 4),
                                "cls_loss": round(cls, 4),
                                "dfl_loss": round(dfl, 4),
                                "loss": round(box + cls + dfl, 4),
                            })
                            continue

                    match_kd_epoch = KD_EPOCH_PATTERN.search(line)
                    if match_kd_epoch:
                        kd_events.append({
                            "type": "epoch",
                            "round": current_round,
                            "epoch": int(match_kd_epoch.group("epoch")),
                            "supervised": round(float(match_kd_epoch.group("supervised")), 4),
                            "kd_only": round(float(match_kd_epoch.group("kd_only")), 4),
                            "box": round(float(match_kd_epoch.group("box")), 4),
                            "kl": round(float(match_kd_epoch.group("kl")), 4),
                            "lora_proj": round(float(match_kd_epoch.group("lora") or 0.0), 4),
                            "total": round(float(match_kd_epoch.group("total")), 4),
                        })
                        continue

                    match_kd_summary = KD_SUMMARY_PATTERN.search(line)
                    if match_kd_summary:
                        kd_events.append({
                            "type": "summary",
                            "round": current_round,
                            "box": round(float(match_kd_summary.group("box")), 4),
                            "kl": round(float(match_kd_summary.group("kl")), 4),
                            "lora_proj": round(float(match_kd_summary.group("lora") or 0.0), 4),
                            "kd_ratio": round(float(match_kd_summary.group("kd_ratio")), 4),
                            "kd_contrib": round(float(match_kd_summary.group("kd_contrib")), 4),
                            "total": round(float(match_kd_summary.group("total")), 4),
                        })
                        continue

                match_auv = AUV_PATTERN.search(line)
                if match_auv:
                    current_auv = int(match_auv.group(1))
                    continue

                if current_round <= 0 or current_round > max_rounds or current_auv is None:
                    continue

                match_progress = PROGRESS_PATTERN.search(line)
                if not match_progress:
                    continue

                box = float(match_progress.group("box"))
                cls = float(match_progress.group("cls"))
                dfl = float(match_progress.group("dfl"))
                step += 1
                events.append({
                    "step": step,
                    "round": current_round,
                    "auv_id": current_auv,
                    "epoch": int(match_progress.group("epoch")),
                    "epochs": int(match_progress.group("epochs")),
                    "batch": int(match_progress.group("batch")),
                    "batches": int(match_progress.group("batches")),
                    "box_loss": round(box, 4),
                    "cls_loss": round(cls, 4),
                    "dfl_loss": round(dfl, 4),
                    "loss": round(box + cls + dfl, 4),
                })

    return _finalize(
        log_path,
        events,
        kd_events,
        total_rounds,
        max_events_per_auv_round,
        max_kd_progress_events_per_round,
    )


def _finalize(
    log_path: Path,
    events: list[dict[str, Any]],
    kd_events: list[dict[str, Any]],
    total_rounds: int,
    max_events_per_auv_round: int,
    max_kd_progress_events_per_round: int,
) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(int(event["round"]), int(event["auv_id"]))].append(event)

    compact: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = grouped[key]
        if len(items) <= max_events_per_auv_round:
            compact.extend(items)
            continue
        stride = max(1, len(items) // max_events_per_auv_round)
        selected = items[::stride]
        if selected[-1] is not items[-1]:
            selected.append(items[-1])
        compact.extend(selected[: max_events_per_auv_round + 1])

    compact.sort(key=lambda item: (item["round"], item["step"]))
    compact_kd = _compact_kd_events(kd_events, max_kd_progress_events_per_round)
    return {
        "available": bool(compact),
        "source": str(log_path),
        "source_name": log_path.name,
        "rounds": sorted({int(event["round"]) for event in compact}),
        "total_rounds": total_rounds,
        "event_count": len(compact),
        "events": compact,
        "kd_events": compact_kd,
    }


def _compact_kd_events(
    kd_events: list[dict[str, Any]],
    max_progress_per_round: int,
) -> list[dict[str, Any]]:
    grouped_progress: dict[int, list[dict[str, Any]]] = defaultdict(list)
    non_progress: list[dict[str, Any]] = []
    for event in kd_events:
        if event.get("type") == "progress":
            grouped_progress[int(event["round"])].append(event)
        else:
            non_progress.append(event)

    compact: list[dict[str, Any]] = []
    for round_id, items in grouped_progress.items():
        if len(items) <= max_progress_per_round:
            compact.extend(items)
            continue
        stride = max(1, len(items) // max_progress_per_round)
        selected = items[::stride]
        if selected[-1] is not items[-1]:
            selected.append(items[-1])
        compact.extend(selected[: max_progress_per_round + 1])

    compact.extend(non_progress)
    compact.sort(key=lambda item: (
        int(item.get("round", 0)),
        int(item.get("step", 10**9)),
        0 if item.get("type") == "epoch" else 1 if item.get("type") == "progress" else 2,
    ))
    return compact
