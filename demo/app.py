import base64
import csv
import io
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

try:
    from demo.live_jobs import LiveRoundJobManager
    from demo.log_replay import parse_training_log
except ImportError:
    from live_jobs import LiveRoundJobManager
    from log_replay import parse_training_log

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
STATIC_DIR = DEMO_DIR / "static"

MODEL_CANDIDATES = [
    DEMO_DIR / "student_lora_best.pt",
    DEMO_DIR / "yolo12n_lora_centralized.pt",
    DEMO_DIR / "yolo12n_centralized.pt",
    DEMO_DIR / "best.pt",
    DEMO_DIR / "yolo12n_warmup.pt",
    REPO_ROOT / "yolo12n_warmup.pt",
    Path("yolo12n.pt"),
]

DETECTION_CONF_PRIMARY = 0.25
DETECTION_CONF_THRESHOLDS = (0.25, 0.05, 0.01, 0.001)
DETECTION_MAX_DET = 8

TOPOLOGY_CANDIDATES = [
    REPO_ROOT / "environments/2d/topo/N_30/topo_N30_seed1107.pkl",
    REPO_ROOT / "environments/2d/topo/hfl/N_30/topo_hfl_N30_seed1107.pkl",
    REPO_ROOT / "environments/2d/topo/N_30/topo_N30_seed1109.pkl",
    REPO_ROOT / "environments/2d/topo/hfl/N_30/topo_hfl_N30_seed1109.pkl",
]

DATA_PARTITION_CANDIDATES = [
    REPO_ROOT / "environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1107.pkl",
    REPO_ROOT / "environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl",
]

CASE_CONFIGS = {
    "fedavg_flat": {
        "title": "FedAvg Flat",
        "metrics": DEMO_DIR / "fedavg_metrics.csv",
        "loss": DEMO_DIR / "fedavg_loss_matrix.csv",
        "log": DEMO_DIR / "fedavg_flat_train.log",
        "flow": "flat_full_model",
        "payload_label": "Full model update",
    },
    "fedavg_hfl": {
        "title": "FedAvg HFL",
        "metrics": DEMO_DIR / "fedavg_hfl_results.csv",
        "loss": DEMO_DIR / "fedavg_hfl_loss_matrix.csv",
        "log": DEMO_DIR / "fedavg_hfl_train.log",
        "flow": "hfl_full_model",
        "payload_label": "Full model update",
    },
    "fedkdl": {
        "title": "FedKDL",
        "metrics": DEMO_DIR / "fedkdl_metrics.csv",
        "loss": DEMO_DIR / "fedkdl_loss_matrix.csv",
        "log": DEMO_DIR / "fedkdl_train.log",
        "flow": "relay_lora_int8",
        "payload_label": "LoRA INT8 update",
    },
}

CENTRALIZED_METRICS = DEMO_DIR / "centralized_metrics.csv"

FALLBACK_METRICS = {
    "fedavg_flat": {
        "mAP50": (0.3568, 0.4300, 0.4750),
        "mAP50-95": (0.1765, 0.2200, 0.2500),
        "loss": (5.2581, 4.6200, 4.3500),
        "avg_payload_kb": 10123.8047,
        "tau_round_s": 6381.2118,
        "tau_comp": 387.15,
        "e_total": 7340.8818,
    },
    "fedavg_hfl": {
        "mAP50": (0.3702, 0.4393, 0.4900),
        "mAP50-95": (0.1888, 0.2298, 0.2600),
        "loss": (5.2966, 4.5910, 4.3084),
        "avg_payload_kb": 10123.8047,
        "tau_round_s": 12374.8172,
        "tau_comp": 387.15,
        "e_total": 25285.5241,
    },
    "fedkdl": {
        "mAP50": (0.4970, 0.5281, 0.5500),
        "mAP50-95": (0.2712, 0.2927, 0.3100),
        "loss": (4.4029, 4.0982, 3.9617),
        "avg_payload_kb": 504.8711,
        "tau_round_s": 1091.9652,
        "tau_comp": 193.575,
        "e_total": 2496.2488,
    },
}

SIMULATION_PHASES = {
    "centralized": [
        {"id": "capture", "label": "Capture raw images", "duration_ms": 500},
        {"id": "uplink_a2r", "label": "AUV to Relay: raw images", "duration_ms": 4500},
        {"id": "relay_forward", "label": "Relay forwards image batches", "duration_ms": 500},
        {"id": "uplink_r2g", "label": "Relay to Gateway: raw images", "duration_ms": 4500},
        {"id": "gateway_train", "label": "Gateway centralized training", "duration_ms": 2200},
    ],
    "fedavg_flat": [
        {"id": "train", "label": "Local full-model training", "duration_ms": 1800},
        {"id": "uplink_direct", "label": "AUV to Gateway: full model", "duration_ms": 2400},
        {"id": "gateway_aggregate", "label": "Gateway FedAvg", "duration_ms": 450},
        {"id": "downlink_direct", "label": "Gateway broadcasts full model", "duration_ms": 2400},
    ],
    "fedavg_hfl": [
        {"id": "train", "label": "Local full-model training", "duration_ms": 1800},
        {"id": "uplink_a2r", "label": "AUV to Relay: full model", "duration_ms": 1800},
        {"id": "relay_aggregate", "label": "Relay aggregation", "duration_ms": 450},
        {"id": "uplink_r2g", "label": "Relay to Gateway: full model", "duration_ms": 1800},
        {"id": "gateway_aggregate", "label": "Gateway FedAvg", "duration_ms": 450},
        {"id": "downlink_g2r", "label": "Broadcast Gateway to Relay", "duration_ms": 1800},
        {"id": "downlink_r2a", "label": "Broadcast Relay to AUV", "duration_ms": 1800},
    ],
    "fedkdl": [
        {"id": "train", "label": "Local LoRA training", "duration_ms": 800},
        {"id": "uplink_a2r", "label": "AUV to Relay: INT8 LoRA", "duration_ms": 450},
        {"id": "relay_aggregate", "label": "Relay SVD aggregation", "duration_ms": 250},
        {"id": "relay_cooperate", "label": "Relay-to-Relay cooperation", "duration_ms": 400},
        {"id": "uplink_r2g", "label": "Relay to Gateway: INT8 LoRA", "duration_ms": 450},
        {"id": "gateway_kd", "label": "Gateway aggregation", "duration_ms": 450},
        {"id": "downlink_g2r", "label": "Broadcast Gateway to Relay", "duration_ms": 450},
        {"id": "downlink_r2a", "label": "Broadcast Relay to AUV", "duration_ms": 450},
    ],
}

app = FastAPI(title="FedKDL Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Any | None = None
_model_path: Path | None = None
_model_device = "0" if torch is not None and torch.cuda.is_available() else "cpu"
_live_jobs = LiveRoundJobManager(REPO_ROOT, max_workers=1)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "inf"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _case_log_path(case_name: str) -> Path | None:
    cfg = CASE_CONFIGS.get(case_name)
    if not cfg:
        return None
    path = cfg.get("log")
    if isinstance(path, Path) and path.exists():
        return path
    demo_log = DEMO_DIR / f"{case_name}_train.log"
    if demo_log.exists():
        return demo_log
    search_dir = REPO_ROOT / "results" / "final_exper" / {
        "fedavg_flat": "flat_fedavg",
        "fedavg_hfl": "fedavg_hfl",
        "fedkdl": "fedkdl",
    }.get(case_name, case_name)
    if not search_dir.is_dir():
        return None
    candidates = sorted(search_dir.glob("raw_*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


@lru_cache(maxsize=16)
def _training_log_replay(case_name: str, max_rounds: int = 10) -> dict[str, Any]:
    path = _case_log_path(case_name)
    if path is None:
        return {
            "available": False,
            "case": case_name,
            "rounds": [],
            "events": [],
            "reason": "No raw training log found for this case.",
        }
    payload = parse_training_log(path, max_rounds=max_rounds)
    payload["case"] = case_name
    payload["title"] = CASE_CONFIGS[case_name]["title"]
    payload["max_rounds"] = max_rounds
    return payload


def _is_lora_conv2d(module: Any) -> bool:
    return (
        module.__class__.__name__ == "LoRAConv2d"
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and hasattr(module, "scaling")
        and hasattr(module, "weight")
        and hasattr(module, "in_channels")
        and hasattr(module, "out_channels")
    )


def _count_lora_conv2d(model: Any) -> int:
    return sum(1 for module in model.model.modules() if _is_lora_conv2d(module))


def _bake_lora_for_inference(model: Any) -> int:
    if torch is None:
        return 0

    before = _count_lora_conv2d(model)
    if before:
        print(f"[FedKDL Demo] Found {before} LoRAConv2d layers before inference bake.")
    baked = 0
    for parent_module in list(model.model.modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not _is_lora_conv2d(child_module):
                continue
            with torch.no_grad():
                lora_weight = (child_module.lora_B @ child_module.lora_A).view(
                    child_module.weight.shape
                ) * child_module.scaling
                new_conv = torch.nn.Conv2d(
                    in_channels=child_module.in_channels,
                    out_channels=child_module.out_channels,
                    kernel_size=child_module.kernel_size,
                    stride=child_module.stride,
                    padding=child_module.padding,
                    dilation=child_module.dilation,
                    groups=child_module.groups,
                    bias=child_module.bias is not None,
                    padding_mode=child_module.padding_mode,
                )
                new_conv.weight.copy_(child_module.weight + lora_weight)
                if child_module.bias is not None:
                    new_conv.bias.copy_(child_module.bias)
            setattr(parent_module, child_name, new_conv)
            baked += 1
    after = _count_lora_conv2d(model)
    if before:
        print(f"[FedKDL Demo] LoRAConv2d layers after bake: {after}.")
    return baked


def _load_model(warmup: bool = False) -> Any:
    global _model, _model_path
    if _model is not None:
        return _model
    if YOLO is None or torch is None:
        raise HTTPException(
            status_code=503,
            detail="Detection runtime is unavailable; replay and topology APIs remain active.",
        )
    from detection_2d.compat import register as register_detection_compat

    register_detection_compat()
    _model_path = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[-1])
    print(f"[FedKDL Demo] Loading detector: {_model_path}")
    _model = YOLO(str(_model_path))
    baked = _bake_lora_for_inference(_model)
    if baked:
        print(f"[FedKDL Demo] Baked {baked} LoRA layers for inference.")
    if warmup:
        started_at = time.perf_counter()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _model.predict(dummy, imgsz=640, conf=0.25, device=_model_device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[FedKDL Demo] Detector warmup complete on {_model_device} "
            f"in {time.perf_counter() - started_at:.2f}s."
        )
    return _model


def _metric_row(case_name: str, round_id: int) -> dict[str, str]:
    if case_name == "centralized":
        rows = _read_csv_rows(CENTRALIZED_METRICS)
        if not rows:
            return {}
        candidates = [row for row in rows if _safe_int(row.get("Round")) == round_id]
        if candidates:
            return candidates[0]
        positive_rounds = [row for row in rows if _safe_int(row.get("Round")) > 0]
        return positive_rounds[min(max(round_id - 1, 0), len(positive_rounds) - 1)] if positive_rounds else rows[0]
    cfg = CASE_CONFIGS[case_name]
    rows = _read_csv_rows(cfg["metrics"])
    if not rows:
        return {}
    candidates = [row for row in rows if _safe_int(row.get("Round")) == round_id]
    if candidates:
        return candidates[0]
    positive_rounds = [row for row in rows if _safe_int(row.get("Round")) > 0]
    return positive_rounds[min(max(round_id - 1, 0), len(positive_rounds) - 1)] if positive_rounds else rows[0]


def _available_rounds(case_name: str) -> list[int]:
    if case_name == "centralized":
        rows = _read_csv_rows(CENTRALIZED_METRICS)
        rounds = sorted({_safe_int(row.get("Round")) for row in rows if _safe_int(row.get("Round")) > 0})
        return rounds if rounds else list(range(1, 41))
    rows = _read_csv_rows(CASE_CONFIGS[case_name]["metrics"])
    rounds = sorted({_safe_int(row.get("Round")) for row in rows if _safe_int(row.get("Round")) > 0})
    return rounds if rounds else list(range(1, 41))


def _loss_items(case_name: str, round_id: int) -> list[dict[str, Any]]:
    rows = _read_csv_rows(CASE_CONFIGS[case_name]["loss"])
    if not rows:
        return []
    matching_rows = [row for row in rows if _safe_int(row.get("Round")) == round_id]
    if "AUV" in rows[0]:
        items = []
        for row in matching_rows:
            components = {
                "box": _safe_float(row.get("box_loss")),
                "cls": _safe_float(row.get("cls_loss")),
                "dfl": _safe_float(row.get("dfl_loss")),
            }
            items.append({
                "id": _safe_int(row.get("AUV")),
                "name": f"AUV_{_safe_int(row.get('AUV'))}",
                "loss": round(sum(components.values()), 4),
                "components": components,
            })
        return sorted(items, key=lambda item: item["id"])

    row = matching_rows[0] if matching_rows else rows[
        min(max(round_id - 1, 0), len(rows) - 1)
    ]
    items = []
    for key, value in row.items():
        if not key.startswith("AUV_"):
            continue
        loss = _safe_float(value, default=-1.0)
        if loss >= 0.0:
            auv_id = _safe_int(key.replace("AUV_", ""))
            items.append({"id": auv_id, "name": key, "loss": round(loss, 4)})
    return sorted(items, key=lambda item: item["id"])


def _fallback_value(case_name: str, key: str, round_id: int) -> float:
    value = FALLBACK_METRICS[case_name][key]
    if isinstance(value, tuple):
        return float(value[min(max(round_id - 1, 0), len(value) - 1)])
    return float(value)


def _metric_value(case_name: str, metric: dict[str, str], key: str, round_id: int) -> float:
    value = _safe_float(metric.get(key), default=float("nan"))
    if np.isfinite(value):
        return value
    return _fallback_value(case_name, key, round_id)


def _fallback_topology() -> dict[str, Any]:
    flat_connected_ids = {1, 2, 3, 12, 13, 15, 20, 21, 24, 27, 28}
    relays = [
        {
            "id": relay_id,
            "x_pct": round(8.0 + relay_id * (84.0 / 7.0), 2),
            "y_pct": 36.0 + (relay_id % 2) * 5.0,
        }
        for relay_id in range(8)
    ]
    auvs = []
    for auv_id in range(30):
        row, column = divmod(auv_id, 10)
        auvs.append({
            "id": auv_id,
            "x_pct": round(7.0 + column * (86.0 / 9.0) + (row % 2) * 1.5, 2),
            "y_pct": 58.0 + row * 10.0 + (column % 2) * 2.0,
            "connected": True,
            "relay_id": auv_id % 8,
            "hfl_connected": True,
            "hfl_relay_id": auv_id % 8,
            "flat_connected": auv_id in flat_connected_ids,
        })
    return {
        "source": "fallback_N30_M8",
        "seed": 1107,
        "auv_count": 30,
        "relay_count": 8,
        "connected_count": 30,
        "flat_connected_count": len(flat_connected_ids),
        "relay_links": [
            {"from": relay_id, "to": (relay_id + 1) % 8}
            for relay_id in range(8)
        ],
        "cooperation_pairs": [
            {"from": (relay_id + 1) % 8, "to": relay_id}
            for relay_id in range(8)
        ],
        "auvs": auvs,
        "relays": relays,
    }


def _project_topology(snapshot: Any, source_path: Path) -> dict[str, Any]:
    auv_positions = np.asarray(snapshot.auv_positions, dtype=float)
    relay_positions = np.asarray(snapshot.relay_positions, dtype=float)
    all_positions = np.vstack([auv_positions, relay_positions])
    horizontal_extent = max(
        float(np.max(all_positions[:, 0])),
        float(np.max(all_positions[:, 1])),
        1.0,
    )
    depth_extent = max(float(np.max(auv_positions[:, 2])), 1.0)

    def project(position: np.ndarray, node_id: int, is_auv: bool) -> tuple[float, float]:
        projected_horizontal = 0.70 * position[0] + 0.30 * position[1]
        x_pct = 5.0 + 90.0 * projected_horizontal / horizontal_extent
        y_pct = 16.0 + 68.0 * position[2] / depth_extent
        if is_auv:
            x_pct += ((node_id % 3) - 1) * 0.9
            y_pct += ((node_id % 4) - 1.5) * 0.45
        return (
            round(float(np.clip(x_pct, 4.0, 96.0)), 2),
            round(float(np.clip(y_pct, 17.0, 84.0)), 2),
        )

    association = {
        int(auv_id): int(relay_id)
        for auv_id, relay_id in dict(snapshot.hfl_association).items()
    }
    flat_connected = {int(auv_id) for auv_id in dict(snapshot.flat_association)}
    cluster_sizes = {
        relay_id: sum(1 for assigned_relay in association.values() if assigned_relay == relay_id)
        for relay_id in range(int(snapshot.M))
    }
    graph = dict(snapshot.feasibility_graph_items)
    relay_links = []
    seen_relay_links = set()
    for key, link in graph.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 4
            or key[0] != "relay"
            or key[2] != "relay"
        ):
            continue
        source = int(key[1])
        target = int(key[3])
        undirected_key = tuple(sorted((source, target)))
        if source == target or undirected_key in seen_relay_links:
            continue
        seen_relay_links.add(undirected_key)
        relay_links.append({
            "from": undirected_key[0],
            "to": undirected_key[1],
            "distance": round(float(link["distance"]), 2),
        })

    cooperation_pairs = []
    for relay_id in range(int(snapshot.M)):
        if cluster_sizes.get(relay_id, 0) <= 0:
            continue
        candidates = []
        for other_id in range(int(snapshot.M)):
            if other_id == relay_id or cluster_sizes.get(other_id, 0) <= 0:
                continue
            key_forward = ("relay", relay_id, "relay", other_id)
            key_backward = ("relay", other_id, "relay", relay_id)
            key = key_forward if key_forward in graph else (key_backward if key_backward in graph else None)
            if key is not None:
                candidates.append((other_id, float(graph[key]["distance"])))
        if candidates:
            candidates.sort(key=lambda item: item[1])
            partner_id, distance = candidates[0]
            # Cooperation is receive-based: relay_id receives partner_id's state.
            cooperation_pairs.append({
                "from": partner_id,
                "to": relay_id,
                "distance": round(distance, 2),
            })
    relays = []
    for relay_id, position in enumerate(relay_positions):
        x_pct, y_pct = project(position, relay_id, False)
        gateway_key = ("relay", relay_id, "gateway", 0)
        relays.append({
            "id": relay_id,
            "x_pct": x_pct,
            "y_pct": y_pct,
            "gateway_connected": gateway_key in graph,
        })

    auvs = []
    for auv_id, position in enumerate(auv_positions):
        x_pct, y_pct = project(position, auv_id, True)
        relay_id = association.get(auv_id)
        auvs.append({
            "id": auv_id,
            "x_pct": x_pct,
            "y_pct": y_pct,
            "connected": relay_id is not None,
            "relay_id": relay_id,
            "hfl_connected": relay_id is not None,
            "hfl_relay_id": relay_id,
            "flat_connected": auv_id in flat_connected,
        })

    return {
        "source": source_path.name,
        "seed": int(getattr(snapshot, "seed", 1109)),
        "auv_count": int(getattr(snapshot, "N", len(auvs))),
        "relay_count": int(getattr(snapshot, "M", len(relays))),
        "connected_count": len(association),
        "flat_connected_count": len(flat_connected),
        "relay_links": relay_links,
        "cooperation_pairs": cooperation_pairs,
        "auvs": auvs,
        "relays": relays,
    }


@lru_cache(maxsize=1)
def _load_topology_snapshot() -> tuple[Any | None, Path | None]:
    for path in TOPOLOGY_CANDIDATES:
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                snapshot = pickle.load(handle)
            if int(snapshot.N) == 30 and int(snapshot.M) == 8:
                return snapshot, path
        except Exception as exc:
            print(f"[FedKDL Demo] Cannot load topology {path}: {exc}")
    return None, None


@lru_cache(maxsize=1)
def _load_demo_topology() -> dict[str, Any]:
    snapshot, path = _load_topology_snapshot()
    if snapshot is not None and path is not None:
        return _project_topology(snapshot, path)
    return _fallback_topology()


def _topology_for_scenario(topology: dict[str, Any], case_name: str) -> dict[str, Any]:
    use_flat = case_name == "fedavg_flat"
    auvs = []
    for auv in topology["auvs"]:
        item = dict(auv)
        if use_flat:
            item["connected"] = bool(item.get("flat_connected", False))
            item["relay_id"] = None
        else:
            item["connected"] = bool(item.get("hfl_connected", item.get("connected", False)))
            item["relay_id"] = item.get("hfl_relay_id", item.get("relay_id"))
        auvs.append(item)
    connected_count = sum(1 for auv in auvs if auv["connected"])
    return {
        **topology,
        "auvs": auvs,
        "connected_count": connected_count,
        "use_relays": not use_flat,
        "connection_mode": "flat" if use_flat else "hfl",
    }


def _find_train_images() -> list[Path]:
    candidates = [
        REPO_ROOT / "datasets/URPC2020/URPC2020/train/images",
        REPO_ROOT / "datasets/URPC2020/train/images",
    ]
    image_dir = next((path for path in candidates if path.is_dir()), None)
    if image_dir is None:
        return []
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
        images.extend(image_dir.rglob(pattern))
    return sorted(set(images))


@lru_cache(maxsize=1)
def _sample_detection_images() -> tuple[Path, ...]:
    images = _find_train_images()
    if not images:
        return ()
    stride = max(1, len(images) // 24)
    selected = images[::stride][:24]
    preferred_indices = [7, 9]
    preferred = [selected[index] for index in preferred_indices if index < len(selected)]
    selected = preferred + [path for path in selected if path not in preferred]
    return tuple(selected)


@lru_cache(maxsize=1)
def _partitioned_image_paths() -> dict[int, tuple[Path, ...]]:
    partition_path = next((path for path in DATA_PARTITION_CANDIDATES if path.exists()), None)
    all_images = _find_train_images()
    if partition_path is None or not all_images:
        return {}
    try:
        with partition_path.open("rb") as handle:
            partition = pickle.load(handle)
        return {
            int(auv_id): tuple(
                all_images[int(index)]
                for index in indices
                if 0 <= int(index) < len(all_images)
            )
            for auv_id, indices in dict(partition.auv_data_indices).items()
        }
    except Exception as exc:
        print(f"[FedKDL Demo] Cannot load AUV image partitions: {exc}")
        return {}


@lru_cache(maxsize=1)
def _centralized_raw_manifest() -> dict[str, Any]:
    partition_path = next((path for path in DATA_PARTITION_CANDIDATES if path.exists()), None)
    image_paths = _partitioned_image_paths()
    if partition_path is None or not image_paths:
        fallback_images = 4713
        fallback_kb = fallback_images * 350.0
        return {
            "source": "fallback_350KB_per_image",
            "image_count": fallback_images,
            "total_bytes": int(fallback_kb * 1024),
            "per_auv": {},
        }

    try:
        per_auv = {}
        total_bytes = 0
        image_count = 0
        for auv_id, paths in image_paths.items():
            byte_count = sum(path.stat().st_size for path in paths)
            valid_count = len(paths)
            per_auv[int(auv_id)] = {
                "images": valid_count,
                "bytes": byte_count,
                "sample_names": [path.name for path in paths[:3]],
            }
            total_bytes += byte_count
            image_count += valid_count
        return {
            "source": partition_path.name,
            "image_count": image_count,
            "total_bytes": total_bytes,
            "per_auv": per_auv,
        }
    except Exception as exc:
        print(f"[FedKDL Demo] Cannot measure centralized raw payload: {exc}")
        fallback_images = 4713
        fallback_kb = fallback_images * 350.0
        return {
            "source": "fallback_350KB_per_image",
            "image_count": fallback_images,
            "total_bytes": int(fallback_kb * 1024),
            "per_auv": {},
        }


def _centralized_link_latency(
    topology: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, float]:
    snapshot, _ = _load_topology_snapshot()
    if snapshot is None or not manifest["per_auv"]:
        raw_payload_kb = manifest["total_bytes"] / 1024.0
        full_model_kb = FALLBACK_METRICS["fedavg_hfl"]["avg_payload_kb"]
        hfl_comm = (
            FALLBACK_METRICS["fedavg_hfl"]["tau_round_s"]
            - FALLBACK_METRICS["fedavg_hfl"]["tau_comp"]
        )
        estimated = raw_payload_kb / full_model_kb * hfl_comm
        return {"tau_a2r": estimated / 2.0, "tau_r2g": estimated / 2.0, "tau_round": estimated}

    graph = dict(snapshot.feasibility_graph_items)
    relay_bytes: dict[int, int] = {}
    max_a2r = 0.0
    per_relay_a2r: dict[int, float] = {}
    for auv in topology["auvs"]:
        if not auv["connected"] or auv["relay_id"] is None:
            continue
        auv_id = int(auv["id"])
        relay_id = int(auv["relay_id"])
        byte_count = int(manifest["per_auv"].get(auv_id, {}).get("bytes", 0))
        relay_bytes[relay_id] = relay_bytes.get(relay_id, 0) + byte_count
        link = graph.get(("auv", auv_id, "relay", relay_id))
        if link and float(link["R_bps"]) > 0:
            delay = byte_count * 8.0 / float(link["R_bps"]) + float(link["distance"]) / 1500.0
            per_relay_a2r[relay_id] = max(per_relay_a2r.get(relay_id, 0.0), delay)
            max_a2r = max(max_a2r, delay)

    max_r2g = 0.0
    relay_totals = []
    for relay_id, byte_count in relay_bytes.items():
        link = graph.get(("relay", relay_id, "gateway", 0))
        r2g = 0.0
        if link and float(link["R_bps"]) > 0:
            r2g = byte_count * 8.0 / float(link["R_bps"]) + float(link["distance"]) / 1500.0
        max_r2g = max(max_r2g, r2g)
        relay_totals.append(per_relay_a2r.get(relay_id, 0.0) + r2g)
    return {
        "tau_a2r": max_a2r,
        "tau_r2g": max_r2g,
        "tau_round": max(relay_totals, default=0.0),
    }


def _demo_losses(case_name: str, round_id: int) -> list[dict[str, Any]]:
    losses = _loss_items(case_name, round_id)
    return losses[:30]


def _network_node_details(
    case_name: str,
    topology: dict[str, Any],
    losses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _centralized_raw_manifest()
    loss_by_id = {int(item["id"]): float(item["loss"]) for item in losses}
    loss_components_by_id = {
        int(item["id"]): item.get("components")
        for item in losses
        if item.get("components")
    }
    per_auv = manifest.get("per_auv", {})
    auv_details = []
    for auv in topology["auvs"]:
        auv_id = int(auv["id"])
        local_data = per_auv.get(auv_id, {})
        if case_name == "centralized":
            model_state = "No local model"
            transmitted_object = "Raw JPEG/PNG images"
        elif case_name == "fedkdl":
            model_state = "YOLO12n student with LoRA"
            transmitted_object = "INT8 LoRA update + head/BN state"
        else:
            model_state = "YOLO12n full-parameter model"
            transmitted_object = "FP32 full-model state"
        if auv["connected"]:
            route = (
                "Direct to Gateway"
                if not topology["use_relays"]
                else f"Relay {auv['relay_id']} -> Gateway"
            )
        else:
            route = "Disconnected in this round"
        sample_count = min(3, len(local_data.get("sample_names", [])))
        auv_details.append({
            "id": auv_id,
            "connected": bool(auv["connected"]),
            "relay_id": auv.get("relay_id"),
            "route": route,
            "image_count": int(local_data.get("images", 0)),
            "local_loss": loss_by_id.get(auv_id),
            "loss_components": loss_components_by_id.get(auv_id),
            "model_state": model_state,
            "transmitted_object": transmitted_object,
            "sample_names": local_data.get("sample_names", []),
            "sample_image_urls": [
                f"/api/demo/auv-image/{auv_id}/{slot}"
                for slot in range(sample_count)
            ],
        })

    neighbors: dict[int, set[int]] = {
        int(relay["id"]): set() for relay in topology["relays"]
    }
    if case_name == "fedkdl":
        for pair in topology.get("cooperation_pairs", []):
            if isinstance(pair, dict):
                source, target = pair.get("from"), pair.get("to")
            else:
                source, target = pair
            neighbors[int(source)].add(int(target))
            neighbors[int(target)].add(int(source))
    relay_details = []
    for relay in topology["relays"]:
        relay_id = int(relay["id"])
        members = [
            int(auv["id"])
            for auv in topology["auvs"]
            if auv["connected"] and auv.get("relay_id") == relay_id
        ]
        relay_details.append({
            "id": relay_id,
            "auv_ids": members,
            "image_count": sum(
                int(per_auv.get(auv_id, {}).get("images", 0))
                for auv_id in members
            ),
            "cooperation_neighbors": sorted(neighbors[relay_id]),
            "aggregation": (
                "Bypassed in flat topology"
                if case_name == "fedavg_flat"
                else "SVD LoRA aggregation" if case_name == "fedkdl"
                else "Sample-weighted FedAvg" if case_name != "centralized"
                else "Raw image forwarding"
            ),
            "gateway_link": (
                "Not used"
                if case_name == "fedavg_flat"
                else "Relay to Gateway acoustic uplink"
                if relay.get("gateway_connected", True)
                else "No feasible direct Gateway link"
            ),
        })
    return auv_details, relay_details


def _simulation_payload(case_name: str, round_id: int) -> dict[str, Any]:
    round_id = min(max(round_id, 1), 40)
    base_topology = _load_demo_topology()
    if case_name == "centralized":
        topology = _topology_for_scenario(base_topology, "centralized")
        manifest = _centralized_raw_manifest()
        metric = _metric_row("centralized", round_id)
        auv_details, relay_details = _network_node_details(
            case_name,
            topology,
            [],
        )
        connected_ids = {int(auv["id"]) for auv in topology["auvs"] if auv["connected"]}
        if manifest["per_auv"]:
            transmitted_bytes = sum(
                int(values["bytes"])
                for auv_id, values in manifest["per_auv"].items()
                if int(auv_id) in connected_ids
            )
            transmitted_images = sum(
                int(values["images"])
                for auv_id, values in manifest["per_auv"].items()
                if int(auv_id) in connected_ids
            )
        else:
            transmitted_bytes = manifest["total_bytes"]
            transmitted_images = manifest["image_count"]
        raw_payload_kb = transmitted_bytes / 1024.0
        latency = _centralized_link_latency(topology, manifest)
        return {
            "case": case_name,
            "title": "Centralized upload through relays",
            "round": round_id,
            "topology": {**topology, "use_relays": True},
            "auv_details": auv_details,
            "relay_details": relay_details,
            "phases": SIMULATION_PHASES[case_name],
            "losses": [],
            "payload": {
                "name": "Raw URPC images",
                "encoding": "Original JPEG/PNG bytes",
                "contents": f"{transmitted_images:,} AUV-owned training images",
                "source": manifest["source"],
            },
            "metrics": {
                "uplink_payload_kb": raw_payload_kb,
                "downlink_payload_kb": 0.0,
                "train_latency_s": _safe_float(metric.get("time"), 185.0),
                "tau_a2r": latency["tau_a2r"],
                "tau_r2r": 0.0,
                "tau_r2g": latency["tau_r2g"],
                "tau_svd": 0.0,
                "communication_latency_s": latency["tau_round"],
                "round_latency_s": latency["tau_round"] + _safe_float(metric.get("time"), 185.0),
                "energy_j": 0.0,
                "mAP50": _safe_float(metric.get("mAP50"), _safe_float(metric.get("metrics/mAP50(B)"), 0.0)),
                "mAP50_95": _safe_float(metric.get("metrics/mAP50-95(B)"), 0.0),
                "precision": _safe_float(metric.get("metrics/precision(B)"), 0.0),
                "recall": _safe_float(metric.get("metrics/recall(B)"), 0.0),
                "loss": _safe_float(metric.get("loss"), 0.0),
                "pre_gateway_mAP50": _safe_float(metric.get("mAP50"), _safe_float(metric.get("metrics/mAP50(B)"), 0.0)),
            },
        }

    if case_name not in CASE_CONFIGS:
        raise ValueError(f"Unknown simulation case: {case_name}")
    topology = _topology_for_scenario(base_topology, case_name)
    metric = _metric_row(case_name, round_id)
    losses = _demo_losses(case_name, round_id)
    auv_details, relay_details = _network_node_details(
        case_name,
        topology,
        losses,
    )
    payload_kb = _metric_value(case_name, metric, "avg_payload_kb", round_id)
    round_latency = _metric_value(case_name, metric, "tau_round_s", round_id)
    train_latency = _metric_value(case_name, metric, "tau_comp", round_id)
    is_fedkdl = case_name == "fedkdl"
    payload = {
        "name": "INT8 LoRA update" if is_fedkdl else "FP32 full-model state",
        "encoding": "INT8 tensors + quantization metadata" if is_fedkdl else "Float32 state_dict",
        "contents": (
            "LoRA A/B, partial detection head, and BatchNorm state"
            if is_fedkdl
            else "All YOLO model parameters and buffers"
        ),
        "source": CASE_CONFIGS[case_name]["metrics"].name,
    }
    return {
        "case": case_name,
        "title": CASE_CONFIGS[case_name]["title"],
        "round": round_id,
        "topology": topology,
        "auv_details": auv_details,
        "relay_details": relay_details,
        "phases": SIMULATION_PHASES[case_name],
        "losses": losses,
        "payload": payload,
        "metrics": {
            "uplink_payload_kb": payload_kb,
            "downlink_payload_kb": payload_kb,
            "train_latency_s": train_latency,
            "tau_a2r": _safe_float(metric.get("tau_a2r"), round_latency - train_latency),
            "tau_r2r": _safe_float(metric.get("tau_r2r"), 0.0),
            "tau_r2g": _safe_float(metric.get("tau_r2g"), 0.0),
            "tau_svd": _safe_float(metric.get("tau_svd"), 0.0),
            "communication_latency_s": max(round_latency - train_latency, 0.0),
            "round_latency_s": round_latency,
            "energy_j": _metric_value(case_name, metric, "e_total", round_id),
            "mAP50": _metric_value(case_name, metric, "mAP50", round_id),
            "mAP50_95": _metric_value(case_name, metric, "mAP50-95", round_id),
            "precision": _safe_float(metric.get("precision"), _safe_float(metric.get("metrics/precision(B)"), 0.0)),
            "recall": _safe_float(metric.get("recall"), _safe_float(metric.get("metrics/recall(B)"), 0.0)),
            "loss": _metric_value(case_name, metric, "loss", round_id),
            "pre_gateway_mAP50": _safe_float(
                metric.get("pre_kd_mAP50"),
                _metric_value(case_name, metric, "mAP50", round_id),
            ),
        },
    }


@app.get("/api/auvs")
def get_auvs():
    losses = _loss_items("fedkdl", 1) or _loss_items("fedavg_hfl", 1)
    auvs = []
    for item in losses[:12]:
        auvs.append({
            "id": item["id"],
            "name": f"AUV {item['id']}",
            "battery": max(18, min(96, int(100 - item["loss"] * 10))),
            "status": "Active",
        })
    return {"auvs": auvs or [
        {"id": 1, "name": "AUV 1", "battery": 85, "status": "Active"},
        {"id": 2, "name": "AUV 2", "battery": 73, "status": "Active"},
    ]}


@app.get("/api/demo/summary")
def demo_summary():
    cases = {
        "centralized": {
            "title": "Centralized",
            "rounds": _available_rounds("centralized"),
            "metrics_file": CENTRALIZED_METRICS.name,
            "loss_file": None,
            "log_file": None,
            "log_replay_available": False,
            "live_demo_available": CENTRALIZED_METRICS.exists(),
        }
    }
    cases.update({
        name: {
            "title": cfg["title"],
            "rounds": _available_rounds(name),
            "metrics_file": cfg["metrics"].name,
            "loss_file": cfg["loss"].name if cfg["loss"] is not None else None,
            "log_file": _case_log_path(name).name if _case_log_path(name) else None,
            "log_replay_available": _case_log_path(name) is not None,
            "live_demo_available": _case_log_path(name) is not None,
        }
        for name, cfg in CASE_CONFIGS.items()
    })
    return {
        "cases": cases,
        "ml_available": YOLO is not None and torch is not None and cv2 is not None,
        "model_path": str(_model_path or next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[-1])),
    }


@app.post("/api/demo/live-round/start")
def start_live_round(baseline: str = "fedkdl"):
    if baseline not in {"fedkdl", "fedavg_hfl"}:
        raise HTTPException(status_code=400, detail=f"Unsupported live baseline: {baseline}")
    if YOLO is None or torch is None:
        raise HTTPException(
            status_code=503,
            detail="Live training requires the Torch and Ultralytics runtime.",
        )

    topo_path = next((path for path in TOPOLOGY_CANDIDATES if path.exists()), None)
    data_path = next((path for path in DATA_PARTITION_CANDIDATES if path.exists()), None)
    if topo_path is None or data_path is None:
        raise HTTPException(status_code=503, detail="Live topology/data snapshot is unavailable.")

    command = [
        sys.executable,
        str(REPO_ROOT / "main_trainer_od.py"),
        "--topo",
        str(topo_path),
        "--data",
        str(data_path),
        "--baseline",
        baseline,
        "--rounds",
        "1",
        "--out-dir",
        str(REPO_ROOT / "results" / "demo_live"),
        "--log-dir",
        str(REPO_ROOT / "results" / "demo_live_logs"),
    ]
    try:
        return _live_jobs.start(command, baseline)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/demo/live-round/{job_id}")
def get_live_round(job_id: str):
    job = _live_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Live training job not found.")
    return job


@app.post("/api/demo/live-round/{job_id}/cancel")
def cancel_live_round(job_id: str):
    job = _live_jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Live training job not found.")
    return job


@app.get("/api/demo/training-log/{case_name}")
def training_log_replay(case_name: str, max_rounds: int = 10):
    if case_name not in CASE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown case: {case_name}")
    max_rounds = min(max(int(max_rounds), 1), 40)
    payload = _training_log_replay(case_name, max_rounds)
    if not payload.get("available"):
        raise HTTPException(
            status_code=404,
            detail=payload.get("reason", "Training log replay is unavailable."),
        )
    return payload


@app.get("/api/demo/round/{case_name}/{round_id}")
def demo_round(case_name: str, round_id: int):
    if case_name not in CASE_CONFIGS:
        return {"error": f"Unknown case: {case_name}"}
    return _simulation_payload(case_name, round_id)


@app.get("/api/demo/centralized")
def centralized_demo():
    return _simulation_payload("centralized", 1)


@app.get("/api/demo/scenario/{case_name}/{round_id}")
def simulation_scenario(case_name: str, round_id: int):
    if case_name not in SIMULATION_PHASES:
        return {"error": f"Unknown scenario: {case_name}"}
    return _simulation_payload(case_name, round_id)


@app.get("/api/demo/auv-image/{auv_id}/{slot}")
def auv_sample_image(auv_id: int, slot: int):
    image_paths = _partitioned_image_paths().get(auv_id, ())
    if slot < 0 or slot >= min(3, len(image_paths)):
        raise HTTPException(status_code=404, detail="AUV sample image not found")
    return FileResponse(image_paths[slot])


@app.get("/api/demo/sample-images")
def sample_detection_images():
    images = _sample_detection_images()
    return {
        "images": [
            {
                "id": index,
                "name": path.name,
                "url": f"/api/demo/sample-image/{index}",
            }
            for index, path in enumerate(images)
        ],
    }


@app.get("/api/demo/sample-image/{image_id}")
def sample_detection_image(image_id: int):
    images = _sample_detection_images()
    if image_id < 0 or image_id >= len(images):
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(images[image_id])


async def _run_detection(file: UploadFile) -> dict[str, Any]:
    if cv2 is None or torch is None or YOLO is None:
        raise HTTPException(
            status_code=503,
            detail="Detection dependencies are unavailable on this server.",
        )
    model = _load_model()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_started_at = time.perf_counter()
    confidence_threshold = DETECTION_CONF_THRESHOLDS[-1]
    results = []
    for threshold in DETECTION_CONF_THRESHOLDS:
        confidence_threshold = threshold
        results = model.predict(
            img_np,
            imgsz=640,
            conf=confidence_threshold,
            max_det=DETECTION_MAX_DET,
            device=_model_device,
            verbose=False,
        )
        if results and len(results[0].boxes) > 0:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    server_inference_ms = (time.perf_counter() - inference_started_at) * 1000.0
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    detections = []

    if results:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id] if model.names else str(cls_id)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 120), 2)
            cv2.putText(
                img_bgr,
                f"{label} {conf:.2f}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 120),
                2,
            )
            detections.append({"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]})
    detections.sort(key=lambda item: item["confidence"], reverse=True)

    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return {
        "model_path": str(_model_path),
        "detections": detections,
        "image_b64": base64.b64encode(buffer).decode("utf-8"),
        "server_inference_ms": server_inference_ms,
        "confidence_threshold": confidence_threshold,
        "fallback_threshold_used": confidence_threshold < DETECTION_CONF_PRIMARY,
    }


@app.post("/api/detect")
async def detect_global_model(file: UploadFile = File(...)):
    return await _run_detection(file)


@app.post("/api/detect/{auv_id}")
async def detect_objects(auv_id: int, file: UploadFile = File(...)):
    response = await _run_detection(file)
    response["auv_id"] = auv_id
    return response


@app.on_event("startup")
def preload_detector():
    global _model
    try:
        _load_model(warmup=True)
    except Exception as exc:
        _model = None
        print(f"[FedKDL Demo] Detector preload failed; lazy load will retry: {exc}")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
