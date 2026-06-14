"""
payload_breakdown.py
Standalone script to produce a detailed payload composition table for the paper.

Answers the reviewer question: "0.49 MB gồm những gì?"

This script does NOT modify any training code. It loads the student model,
extracts the trainable state dict, and classifies every key into one of:
  - LoRA factors (Delta-INT8)
  - Detection head (Delta-INT8)
  - BN affine (FP32)
  - BN running buffers (FP32)
  - Quantization metadata (FP32 scale + INT32 zero-point per non-BN key)

Usage:
    python scripts/fedkdl/payload_breakdown.py [--student-ckpt yolo12n_warmup.pt]
"""

import sys
import struct
import argparse
import json
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple

import torch

# ── Ensure project root is importable ──
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def classify_key(key: str) -> str:
    """Classify a state_dict key into a payload component category."""
    # BN running statistics
    if 'running_mean' in key or 'running_var' in key or 'tracked' in key:
        return 'bn_running'
    # BN affine parameters (weight/bias of BatchNorm layers)
    if 'bn' in key and ('weight' in key or 'bias' in key):
        return 'bn_affine'
    # Catch any other BN-related keys
    if 'bn' in key or 'running' in key:
        return 'bn_running'
    # LoRA factors
    if 'lora_A' in key or 'lora_B' in key:
        return 'lora_factors'
    # Everything else → detection head (cv2, cv3, dfl, etc.)
    return 'detection_head'


def is_bn_key(key: str) -> bool:
    """Check if a key is treated as BN by pack_payload (raw FP32, no INT8)."""
    return 'bn' in key or 'running' in key or 'tracked' in key


def compute_breakdown(state_dict: Dict[str, torch.Tensor]) -> dict:
    """
    Simulate the exact byte-level packing of pack_payload() and report per-component sizes.
    
    pack_payload logic (from int8_quantization.py):
      - BN keys: raw FP32 bytes (4 bytes/param)
      - Non-BN keys: 8-byte header (scale:f32 + zero_point:i32) + INT8 data (1 byte/param)
    """
    categories = OrderedDict([
        ('lora_factors',    {'keys': [], 'n_params': 0, 'bytes': 0, 'precision': 'Delta-INT8'}),
        ('detection_head',  {'keys': [], 'n_params': 0, 'bytes': 0, 'precision': 'Delta-INT8'}),
        ('bn_affine',       {'keys': [], 'n_params': 0, 'bytes': 0, 'precision': 'FP32'}),
        ('bn_running',      {'keys': [], 'n_params': 0, 'bytes': 0, 'precision': 'FP32'}),
        ('quant_metadata',  {'keys': [], 'n_params': 0, 'bytes': 0, 'precision': 'FP32 scale + INT32 zp'}),
    ])
    
    n_quantized_keys = 0
    
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        numel = tensor.numel()
        cat = classify_key(key)
        
        if is_bn_key(key):
            # BN: raw FP32 bytes, NO header
            byte_size = numel * 4
            categories[cat]['keys'].append(key)
            categories[cat]['n_params'] += numel
            categories[cat]['bytes'] += byte_size
        else:
            # Non-BN: 8-byte header + INT8 data
            header_bytes = 8  # struct.pack('fi', scale, zero_point)
            data_bytes = numel * 1  # INT8 = 1 byte per param
            categories[cat]['keys'].append(key)
            categories[cat]['n_params'] += numel
            categories[cat]['bytes'] += data_bytes
            n_quantized_keys += 1
    
    # Account for quantization metadata (8 bytes per non-BN key)
    total_meta_bytes = n_quantized_keys * 8
    categories['quant_metadata']['bytes'] = total_meta_bytes
    categories['quant_metadata']['n_params'] = n_quantized_keys
    
    return categories


def print_table(categories: dict):
    """Print the reviewer-facing markdown table."""
    total_bytes = sum(c['bytes'] for c in categories.values())
    total_kb = total_bytes / 1024.0
    
    print("\n" + "=" * 75)
    print("  PAYLOAD COMPOSITION BREAKDOWN (FedKDL Delta-INT8)")
    print("=" * 75)
    
    # Markdown table
    print(f"\n| {'Component':<22} | {'Precision':>28} | {'Params':>10} | {'Payload':>12} |")
    print(f"|{'-'*24}|{'-'*30}|{'-'*12}|{'-'*14}|")
    
    display_names = {
        'lora_factors':   'LoRA factors',
        'detection_head': 'Detection head',
        'bn_affine':      'BN affine',
        'bn_running':     'BN running buffers',
        'quant_metadata': 'Quant. metadata',
    }
    
    for cat_id, cat in categories.items():
        name = display_names.get(cat_id, cat_id)
        kb = cat['bytes'] / 1024.0
        pct = 100.0 * cat['bytes'] / total_bytes if total_bytes > 0 else 0
        precision = cat['precision']
        n_params = cat['n_params']
        print(f"| {name:<22} | {precision:>28} | {n_params:>10,} | {kb:>8.1f} KB |")
    
    print(f"|{'-'*24}|{'-'*30}|{'-'*12}|{'-'*14}|")
    total_params = sum(c['n_params'] for c in categories.values() if c['precision'] != 'FP32 scale + INT32 zp')
    print(f"| {'**Total**':<22} | {'—':>28} | {total_params:>10,} | {total_kb:>8.1f} KB |")
    print()
    
    # Detailed key listing per category
    print("-" * 75)
    print("  KEY LISTING PER CATEGORY")
    print("-" * 75)
    for cat_id, cat in categories.items():
        if cat_id == 'quant_metadata':
            continue
        name = display_names.get(cat_id, cat_id)
        n_keys = len(cat['keys'])
        kb = cat['bytes'] / 1024.0
        print(f"\n[{name}] — {n_keys} keys, {kb:.1f} KB")
        for k in cat['keys']:
            print(f"    {k}")
    
    return total_bytes, total_kb


def export_json(categories: dict, output_path: str):
    """Export breakdown as JSON for downstream analysis."""
    result = {}
    for cat_id, cat in categories.items():
        result[cat_id] = {
            'precision': cat['precision'],
            'n_params': cat['n_params'],
            'bytes': cat['bytes'],
            'kb': round(cat['bytes'] / 1024.0, 2),
            'n_keys': len(cat['keys']),
            'keys': cat['keys'],
        }
    result['total'] = {
        'bytes': sum(c['bytes'] for c in categories.values()),
        'kb': round(sum(c['bytes'] for c in categories.values()) / 1024.0, 2),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Exported] {output_path}")


def cross_validate_with_pack_payload(state_dict: Dict[str, torch.Tensor], expected_total_bytes: int):
    """Verify our breakdown matches the real pack_payload output byte-for-byte."""
    from tasks.detection_2d.knowledge_compression.int8_quantization import pack_payload
    actual_bytes, actual_kb = pack_payload(state_dict)
    
    print(f"\n[Cross-validation]")
    print(f"  pack_payload() actual output  = {len(actual_bytes):,} bytes = {actual_kb:.1f} KB")
    print(f"  Breakdown sum (computed)      = {expected_total_bytes:,} bytes = {expected_total_bytes / 1024.0:.1f} KB")
    diff = abs(len(actual_bytes) - expected_total_bytes)
    if diff == 0:
        print(f"  ✅ EXACT MATCH")
    else:
        print(f"  ⚠️  DIFF = {diff} bytes ({diff / 1024.0:.2f} KB)")
        print(f"      This is expected if INT8 quantization pads/rounds differently.")


def main():
    parser = argparse.ArgumentParser("FedKDL Payload Breakdown")
    parser.add_argument(
        "--student-ckpt", type=str, default="yolo12n_warmup.pt",
        help="Student checkpoint to load (default: yolo12n_warmup.pt)"
    )
    parser.add_argument(
        "--output", type=str, default="results/payload_breakdown.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--cross-validate", action="store_true", default=True,
        help="Run pack_payload() to cross-validate byte count"
    )
    args = parser.parse_args()
    
    ckpt_path = Path(args.student_ckpt)
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    # Load student model with LoRA
    from config.settings import fed_cfg
    from tasks.detection_2d.models.yolo_wrapper import StudentModel
    
    print(f"[Loading] {ckpt_path} with LoRA rank={fed_cfg.LORA_RANK}")
    student = StudentModel(
        str(ckpt_path),
        rank=fed_cfg.LORA_RANK,
        nc=4,  # URPC2020
        full_param=False,
        use_lora=True,
    )
    
    state_dict = student.trainable_state_dict()
    print(f"[State dict] {len(state_dict)} keys, "
          f"{sum(v.numel() for v in state_dict.values()):,} total params")
    
    # Compute breakdown
    categories = compute_breakdown(state_dict)
    total_bytes, total_kb = print_table(categories)
    
    # Cross-validate
    if args.cross_validate:
        cross_validate_with_pack_payload(state_dict, total_bytes)
    
    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_json(categories, str(output_path))
    
    # Compare with config constant
    config_kb = fed_cfg.LORA_INT8_PAYLOAD_BYTES_2D / 1024.0
    print(f"\n[Config] LORA_INT8_PAYLOAD_BYTES_2D = {fed_cfg.LORA_INT8_PAYLOAD_BYTES_2D:,} bytes = {config_kb:.1f} KB")
    print(f"[Script] Computed total = {total_bytes:,} bytes = {total_kb:.1f} KB")


if __name__ == "__main__":
    main()
