import base64
import csv
import io
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
    DEMO_DIR / "yolo12n_lora_centralized.pt",
    DEMO_DIR / "yolo12n_centralized.pt",
    DEMO_DIR / "best.pt",
    DEMO_DIR / "yolo12n_warmup.pt",
    REPO_ROOT / "yolo12n_warmup.pt",
    Path("yolo12n.pt"),
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


@app.post("/api/detect/{auv_id}")
async def detect_objects(auv_id: int, file: UploadFile = File(...)):
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
        "auv_id": auv_id,
        "model_path": str(_model_path),
        "detections": detections,
        "image_b64": base64.b64encode(buffer).decode("utf-8"),
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
