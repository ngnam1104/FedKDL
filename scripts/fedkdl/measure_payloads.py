"""
Measure actual learning payload sizes from the instantiated 2D models.

This script loads StudentModel with the current config, so it catches accidental
LoRA injection or payload-key changes instead of relying on hand calculations.

Usage:
    python scripts/fedkdl/measure_payloads.py --student-ckpt yolo12n.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import fed_cfg
from tasks.detection_2d.baselines import parse_baseline_config, STANDARD_BASELINES
from tasks.detection_2d.knowledge_compression.int8_quantization import pack_payload
from tasks.detection_2d.models.lora import LoRAConv2d
from tasks.detection_2d.models.yolo_wrapper import StudentModel


def _tensor_bytes(state: dict[str, torch.Tensor]) -> int:
    return int(
        sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v))
    )


def _topk_payload_bytes(state: dict[str, torch.Tensor], rho: float) -> int:
    total_params = sum(
        v.numel()
        for v in state.values()
        if torch.is_tensor(v) and torch.is_floating_point(v)
    )
    if total_params <= 0:
        return 0
    k = max(1, int(total_params * rho))
    header_bytes = 5  # float32 scale + int8 zero-point, matching SparseINT8Payload
    return int(k * (1 + 4) + header_bytes)  # int8 value + int32 index


def _component_counts(student: StudentModel, state: dict[str, torch.Tensor]) -> dict[str, int]:
    head_idx = len(student.yolo.model.model) - 1
    head_prefix = f"model.{head_idx}."
    counts = {"lora": 0, "head": 0, "bn": 0, "other": 0}
    for key, value in state.items():
        if "lora_" in key:
            counts["lora"] += value.numel()
        elif key.startswith(head_prefix):
            counts["head"] += value.numel()
        elif "bn" in key or "running" in key or "tracked" in key:
            counts["bn"] += value.numel()
        else:
            counts["other"] += value.numel()
    return counts


def _head_lora_modules(student: StudentModel) -> list[str]:
    head_idx = len(student.yolo.model.model) - 1
    head_prefix = f"model.{head_idx}."
    return [
        name
        for name, module in student.yolo.model.named_modules()
        if isinstance(module, LoRAConv2d) and name.startswith(head_prefix)
    ]


def _load_student(baseline: str, ckpt: str, nc: int) -> StudentModel:
    cfg = parse_baseline_config(baseline)
    return StudentModel(
        ckpt=ckpt,
        rank=fed_cfg.LORA_RANK,
        nc=nc,
        full_param=cfg.full_param,
        use_lora=cfg.use_lora,
        lora_targets=list(getattr(fed_cfg, "LORA_TARGETS", ("Conv",))),
        lora_strategy=getattr(fed_cfg, "LORA_STRATEGY", "adaptive"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-ckpt", default="yolo12n.pt")
    parser.add_argument("--nc", type=int, default=4)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(STANDARD_BASELINES),
    )
    args = parser.parse_args()

    print("[PayloadMeasure]")
    print(f"  ckpt={args.student_ckpt}")
    print(f"  LORA_RANK={fed_cfg.LORA_RANK}")
    print(f"  LORA_STRATEGY={getattr(fed_cfg, 'LORA_STRATEGY', 'adaptive')}")
    print(f"  LORA_TARGETS={getattr(fed_cfg, 'LORA_TARGETS', ('Conv',))}")
    print(f"  TARGET_PAYLOAD_KB={fed_cfg.TARGET_PAYLOAD_KB}")
    print()
    print(
        "| baseline | mode | bytes | KiB | lora params | head params | bn params | other params | head LoRA |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    cache: dict[tuple[bool, bool], tuple[StudentModel, dict[str, torch.Tensor]]] = {}
    for baseline in args.baselines:
        cfg = parse_baseline_config(baseline)
        key = (cfg.full_param, cfg.use_lora)
        if key not in cache:
            student = _load_student(baseline, args.student_ckpt, args.nc)
            cache[key] = (student, student.trainable_state_dict())
        student, state = cache[key]

        if cfg.topk_grad:
            rho = cfg.topk_ratio if cfg.topk_ratio is not None else fed_cfg.RHO_S
            payload_bytes = _topk_payload_bytes(state, rho)
            mode = f"topk-int8 rho={rho:.2f}"
        elif cfg.use_int8:
            payload, _ = pack_payload(state)
            payload_bytes = len(payload)
            mode = "int8/delta-shape"
        else:
            payload_bytes = _tensor_bytes(state)
            mode = "float-state"

        counts = _component_counts(student, state)
        head_lora = _head_lora_modules(student)
        print(
            f"| {baseline} | {mode} | {payload_bytes:,} | {payload_bytes / 1024.0:.2f} | "
            f"{counts['lora']:,} | {counts['head']:,} | {counts['bn']:,} | "
            f"{counts['other']:,} | {len(head_lora)} |"
        )

        if head_lora:
            print(f"[WARN] {baseline} has LoRA inside head: {head_lora[:10]}")


if __name__ == "__main__":
    main()
