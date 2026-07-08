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


def clean_log_segment(segment: str) -> str:
    text = ANSI_PATTERN.sub("", segment)
    return text.replace("\x1b[K", "").strip()


def parse_training_log(
    log_path: Path,
    *,
    max_rounds: int = 10,
    max_events_per_auv_round: int = 36,
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
    current_round = 0
    total_rounds = 0
    current_auv: int | None = None
    step = 0

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
                                total_rounds,
                                max_events_per_auv_round,
                            )
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

    return _finalize(log_path, events, total_rounds, max_events_per_auv_round)


def _finalize(
    log_path: Path,
    events: list[dict[str, Any]],
    total_rounds: int,
    max_events_per_auv_round: int,
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
    return {
        "available": bool(compact),
        "source": str(log_path),
        "source_name": log_path.name,
        "rounds": sorted({int(event["round"]) for event in compact}),
        "total_rounds": total_rounds,
        "event_count": len(compact),
        "events": compact,
    }
