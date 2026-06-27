import base64
import csv
import io
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO


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

TOPOLOGY_CANDIDATES = [
    REPO_ROOT / "environments/2d/topo/N_30/topo_N30_seed1109.pkl",
    REPO_ROOT / "environments/2d/topo/hfl/N_30/topo_hfl_N30_seed1109.pkl",
]

CASE_CONFIGS = {
    "fedavg": {
        "title": "FedAvg HFL",
        "metrics": DEMO_DIR / "fedavg_hfl_results.csv",
        "loss": DEMO_DIR / "fedavg_hfl_loss_matrix.csv",
        "flow": "full_model",
        "payload_label": "Full model update",
    },
    "fedkdl": {
        "title": "FedKDL",
        "metrics": DEMO_DIR / "fedkdl_metrics.csv",
        "loss": DEMO_DIR / "fedkdl_loss_matrix.csv",
        "flow": "relay_lora_int8",
        "payload_label": "LoRA INT8 update",
    },
}

FALLBACK_METRICS = {
    "fedavg": {
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
        {"id": "uplink_direct", "label": "Upload raw images to Gateway", "duration_ms": 10000},
        {"id": "gateway_store", "label": "Store the centralized dataset", "duration_ms": 500},
    ],
    "fedavg": [
        {"id": "train", "label": "Local full-model training", "duration_ms": 1800},
        {"id": "uplink_a2r", "label": "AUV to Relay: full model", "duration_ms": 1800},
        {"id": "relay_aggregate", "label": "Relay aggregation", "duration_ms": 450},
        {"id": "uplink_r2g", "label": "Relay to Gateway: full model", "duration_ms": 1800},
        {"id": "gateway_aggregate", "label": "Gateway FedAvg", "duration_ms": 450},
        {"id": "downlink_g2r", "label": "Broadcast Gateway to Relay", "duration_ms": 800},
        {"id": "downlink_r2a", "label": "Broadcast Relay to AUV", "duration_ms": 800},
    ],
    "fedkdl": [
        {"id": "train", "label": "Local LoRA training", "duration_ms": 800},
        {"id": "uplink_a2r", "label": "AUV to Relay: INT8 LoRA", "duration_ms": 450},
        {"id": "relay_aggregate", "label": "Relay SVD aggregation", "duration_ms": 250},
        {"id": "uplink_r2g", "label": "Relay to Gateway: INT8 LoRA", "duration_ms": 450},
        {"id": "gateway_kd", "label": "Gateway aggregation and KD", "duration_ms": 450},
        {"id": "downlink_g2r", "label": "Broadcast Gateway to Relay", "duration_ms": 300},
        {"id": "downlink_r2a", "label": "Broadcast Relay to AUV", "duration_ms": 300},
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

_model: YOLO | None = None
_model_path: Path | None = None


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_model() -> YOLO:
    global _model, _model_path
    if _model is not None:
        return _model
    _model_path = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[-1])
    print(f"[FedKDL Demo] Loading detector: {_model_path}")
    _model = YOLO(str(_model_path))
    return _model


def _metric_row(case_name: str, round_id: int) -> dict[str, str]:
    cfg = CASE_CONFIGS[case_name]
    rows = _read_csv_rows(cfg["metrics"])
    if not rows:
        return {}
    candidates = [row for row in rows if _safe_int(row.get("Round")) == round_id]
    if candidates:
        return candidates[0]
    positive_rounds = [row for row in rows if _safe_int(row.get("Round")) > 0]
    return positive_rounds[min(max(round_id - 1, 0), len(positive_rounds) - 1)] if positive_rounds else rows[0]


def _loss_row(case_name: str, round_id: int) -> dict[str, str]:
    cfg = CASE_CONFIGS[case_name]
    rows = _read_csv_rows(cfg["loss"])
    if not rows:
        return {}
    candidates = [row for row in rows if _safe_int(row.get("Round")) == round_id]
    if candidates:
        return candidates[0]
    return rows[min(max(round_id - 1, 0), len(rows) - 1)]


def _available_rounds(case_name: str) -> list[int]:
    rows = _read_csv_rows(CASE_CONFIGS[case_name]["metrics"])
    rounds = sorted({_safe_int(row.get("Round")) for row in rows if _safe_int(row.get("Round")) > 0})
    return rounds[:3] if rounds else [1, 2, 3]


def _loss_items(case_name: str, round_id: int) -> list[dict[str, Any]]:
    row = _loss_row(case_name, round_id)
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
        })
    return {
        "source": "fallback_N30_M8",
        "seed": 1109,
        "auv_count": 30,
        "relay_count": 8,
        "connected_count": 30,
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
    relays = []
    for relay_id, position in enumerate(relay_positions):
        x_pct, y_pct = project(position, relay_id, False)
        relays.append({"id": relay_id, "x_pct": x_pct, "y_pct": y_pct})

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
        })

    return {
        "source": source_path.name,
        "seed": int(getattr(snapshot, "seed", 1109)),
        "auv_count": int(getattr(snapshot, "N", len(auvs))),
        "relay_count": int(getattr(snapshot, "M", len(relays))),
        "connected_count": len(association),
        "auvs": auvs,
        "relays": relays,
    }


def _load_demo_topology() -> dict[str, Any]:
    for path in TOPOLOGY_CANDIDATES:
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                snapshot = pickle.load(handle)
            if int(snapshot.N) == 30 and int(snapshot.M) == 8:
                return _project_topology(snapshot, path)
        except Exception as exc:
            print(f"[FedKDL Demo] Cannot load topology {path}: {exc}")
    return _fallback_topology()


def _demo_losses(case_name: str, round_id: int) -> list[dict[str, Any]]:
    losses = _loss_items(case_name, round_id)
    if losses:
        return losses[:30]
    base = _fallback_value(case_name, "loss", round_id)
    offsets = (0.18, -0.08, 0.06, -0.12, 0.21, 0.03, -0.04, 0.11, -0.16, 0.08)
    return [
        {
            "id": auv_id,
            "name": f"AUV_{auv_id}",
            "loss": round(base + offsets[auv_id % len(offsets)] + 0.01 * (auv_id // 10), 4),
        }
        for auv_id in range(30)
    ]


def _build_flow(case_name: str, losses: list[dict[str, Any]]) -> dict[str, Any]:
    active_ids = [item["id"] for item in losses]
    if case_name == "fedkdl":
        relays = []
        for relay_id in range(4):
            members = [auv_id for auv_id in active_ids if auv_id % 4 == relay_id][:7]
            relays.append({"id": relay_id, "name": f"Relay {relay_id}", "auv_ids": members})
        return {
            "type": "relay_lora_int8",
            "steps": ["AUV local LoRA train", "AUV -> Relay INT8 update", "Relay SVD aggregation", "Gateway KD / aggregation"],
            "relays": relays,
        }
    return {
        "type": "full_model",
        "steps": ["AUV local full-model train", "AUV -> Gateway full update", "Gateway FedAvg aggregation", "Broadcast global model"],
        "relays": [],
    }


def _round_payload(case_name: str, metric: dict[str, str]) -> float:
    payload = _safe_float(metric.get("avg_payload_kb"))
    if payload > 0:
        return payload
    return 504.9 if case_name == "fedkdl" else 20480.0


def _case_round_payload(case_name: str, round_id: int) -> dict[str, Any]:
    cfg = CASE_CONFIGS[case_name]
    metric = _metric_row(case_name, round_id)
    losses = _loss_items(case_name, round_id)
    payload_kb = _round_payload(case_name, metric)
    return {
        "case": case_name,
        "title": cfg["title"],
        "round": round_id,
        "flow": _build_flow(case_name, losses),
        "metrics": {
            "mAP50": _safe_float(metric.get("mAP50")),
            "mAP50_95": _safe_float(metric.get("mAP50-95")),
            "loss": _safe_float(metric.get("loss")),
            "payload_kb": payload_kb,
            "latency_s": _safe_float(metric.get("tau_round_s")),
            "energy_j": _safe_float(metric.get("e_total")),
            "alive": _safe_float(metric.get("alive")),
            "kd_active": str(metric.get("kd_active", "")).lower() == "true",
            "kd_accepted": str(metric.get("gateway_kd_accepted", "")).lower() == "true",
        },
        "losses": losses,
        "payload_label": cfg["payload_label"],
    }


def _simulation_payload(case_name: str, round_id: int) -> dict[str, Any]:
    round_id = min(max(round_id, 1), 3)
    topology = _load_demo_topology()
    if case_name == "centralized":
        raw_images = 5543
        avg_image_kb = 350.0
        raw_payload_kb = raw_images * avg_image_kb
        fedavg_payload = FALLBACK_METRICS["fedavg"]["avg_payload_kb"]
        fedavg_comm = (
            FALLBACK_METRICS["fedavg"]["tau_round_s"]
            - FALLBACK_METRICS["fedavg"]["tau_comp"]
        )
        estimated_comm_s = raw_payload_kb / fedavg_payload * fedavg_comm
        return {
            "case": case_name,
            "title": "Centralized raw-data upload",
            "round": 1,
            "topology": {**topology, "use_relays": False},
            "phases": SIMULATION_PHASES[case_name],
            "losses": [],
            "metrics": {
                "uplink_payload_kb": raw_payload_kb,
                "downlink_payload_kb": 0.0,
                "train_latency_s": 0.0,
                "communication_latency_s": estimated_comm_s,
                "round_latency_s": estimated_comm_s,
                "energy_j": 0.0,
                "mAP50": 0.7128,
            },
        }

    metric = _metric_row(case_name, round_id)
    payload_kb = _metric_value(case_name, metric, "avg_payload_kb", round_id)
    round_latency = _metric_value(case_name, metric, "tau_round_s", round_id)
    train_latency = _metric_value(case_name, metric, "tau_comp", round_id)
    return {
        "case": case_name,
        "title": "FedAvg full-model HFL" if case_name == "fedavg" else "FedKDL LoRA/INT8 HFL",
        "round": round_id,
        "topology": {**topology, "use_relays": True},
        "phases": SIMULATION_PHASES[case_name],
        "losses": _demo_losses(case_name, round_id),
        "metrics": {
            "uplink_payload_kb": payload_kb,
            "downlink_payload_kb": payload_kb,
            "train_latency_s": train_latency,
            "communication_latency_s": max(round_latency - train_latency, 0.0),
            "round_latency_s": round_latency,
            "energy_j": _metric_value(case_name, metric, "e_total", round_id),
            "mAP50": _metric_value(case_name, metric, "mAP50", round_id),
        },
    }


@app.get("/api/auvs")
def get_auvs():
    losses = _loss_items("fedkdl", 1) or _loss_items("fedavg", 1)
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
    return {
        "cases": {
            name: {
                "title": cfg["title"],
                "rounds": _available_rounds(name),
                "metrics_file": cfg["metrics"].name,
                "loss_file": cfg["loss"].name,
            }
            for name, cfg in CASE_CONFIGS.items()
        },
        "model_path": str(_model_path or next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[-1])),
    }


@app.get("/api/demo/round/{case_name}/{round_id}")
def demo_round(case_name: str, round_id: int):
    if case_name not in CASE_CONFIGS:
        return {"error": f"Unknown case: {case_name}"}
    return _case_round_payload(case_name, round_id)


@app.get("/api/demo/centralized")
def centralized_demo():
    raw_images = 5543
    avg_image_kb = 350.0
    raw_payload_kb = raw_images * avg_image_kb
    return {
        "title": "Centralized Raw-Data Upload",
        "steps": ["AUV captures raw images", "Raw images sent to Gateway", "Gateway trains YOLO12n-LoRA", "Detector used for inference"],
        "raw_images": raw_images,
        "avg_image_kb": avg_image_kb,
        "payload_kb": raw_payload_kb,
        "payload_mb": raw_payload_kb / 1024.0,
        "note": "Centralized training is an accuracy upper-bound, but raw image upload is impractical over acoustic links.",
    }


@app.get("/api/demo/scenario/{case_name}/{round_id}")
def simulation_scenario(case_name: str, round_id: int):
    if case_name not in SIMULATION_PHASES:
        return {"error": f"Unknown scenario: {case_name}"}
    return _simulation_payload(case_name, round_id)


async def _run_detection(file: UploadFile) -> dict[str, Any]:
    model = _load_model()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    results = model.predict(img_np, conf=0.25, verbose=False)
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

    _, buffer = cv2.imencode(".jpg", img_bgr)
    return {
        "model_path": str(_model_path),
        "detections": detections,
        "image_b64": base64.b64encode(buffer).decode("utf-8"),
    }


@app.post("/api/detect")
async def detect_global_model(file: UploadFile = File(...)):
    return await _run_detection(file)


@app.post("/api/detect/{auv_id}")
async def detect_objects(auv_id: int, file: UploadFile = File(...)):
    response = await _run_detection(file)
    response["auv_id"] = auv_id
    return response


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
