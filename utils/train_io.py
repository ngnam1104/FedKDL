"""
Đường dẫn & ghi artifact sau train: JSON (plot) + stdout log (debug / tư liệu).
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TextIO


@dataclass(frozen=True)
class ExperimentPaths:
    """Đường dẫn chuẩn cho một lần chạy train."""

    stem: str
    json_path: Path
    stdout_log_path: Path


def build_experiment_paths(
    *,
    task: str,
    out_dir: str | Path,
    log_dir: str | Path,
    N: int,
    dataset: str,
    alpha_str: str,
    baseline: str,
    seed: int,
    rho_s: Optional[float] = None,
) -> ExperimentPaths:
    """
    Tạo tên file thống nhất giữa JSON, stdout log và runner bash/ps1.

    1D: log_N{n}_{ds}_a{alpha}_{baseline}_rho{rho}_seed{seed}
    2D: log_N{n}_{ds}_a{alpha}_{baseline}_seed{seed}
    """
    if task == "1D":
        rho_str = str(rho_s).replace(".", "p")
        stem = f"log_N{N}_{dataset}_a{alpha_str}_{baseline}_rho{rho_str}_seed{seed}"
    else:
        stem = f"log_N{N}_{dataset}_a{alpha_str}_{baseline}_seed{seed}"

    out_dir = Path(out_dir)
    log_dir = Path(log_dir)
    return ExperimentPaths(
        stem=stem,
        json_path=out_dir / f"{stem}.json",
        stdout_log_path=log_dir / f"{stem}.stdout.log",
    )


@contextmanager
def tee_stdout_to_file(log_path: Path):
    """
    Mirror toàn bộ print() ra console và file .stdout.log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console = sys.stdout
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(console, log_file)
        try:
            yield log_file
        finally:
            sys.stdout = console


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        return self.streams[0].isatty() if self.streams else False


def save_json_result(
    json_path: Path,
    result_data: dict,
    *,
    encoder_cls: Optional[type] = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        kwargs: dict[str, Any] = {"indent": 2, "ensure_ascii": False}
        if encoder_cls is not None:
            kwargs["cls"] = encoder_cls
        json.dump(result_data, f, **kwargs)


def print_artifact_summary(paths: ExperimentPaths, *, extra: Optional[dict] = None) -> None:
    """Print clearly 2 artifacts after training."""
    print("\n" + "=" * 60)
    print("[Artifacts] Training results saved:")
    print(f"  (1) JSON metrics (for plotting scripts)")
    print(f"      -> {paths.json_path.resolve()}")
    print(f"      - metadata, metrics (by round), energy_consumption, latency_history")
    print(f"  (2) Stdout log (console replay / debug)")
    print(f"      -> {paths.stdout_log_path.resolve()}")
    print(f"      - Round-by-round print, payload KB, runtime errors, Gateway KD, ...")
    if extra:
        for k, v in extra.items():
            print(f"  • {k}: {v}")
    print("=" * 60 + "\n")


def run_trainer_with_artifacts(
    paths: ExperimentPaths,
    train_fn: Callable[[], dict],
    *,
    encoder_cls: Optional[type] = None,
) -> dict:
    """
    Chạy train_fn trong tee stdout, lưu JSON, in summary.
    train_fn() phải trả về dict đã build sẵn (build_experiment_bundle).
    """
    with tee_stdout_to_file(paths.stdout_log_path):
        result_data = train_fn()

    result_data.setdefault("metadata", {})["artifacts"] = {
        "json_path": str(paths.json_path.resolve()),
        "stdout_log_path": str(paths.stdout_log_path.resolve()),
    }

    save_json_result(paths.json_path, result_data, encoder_cls=encoder_cls)
    print_artifact_summary(paths)
    return result_data
