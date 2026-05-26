import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO

app = FastAPI(title="FedKDL Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model (fallback to base model if pretrained doesn't exist)
model_path = "../yolo11n_pretrained.pt" if os.path.exists("../yolo11n_pretrained.pt") else "yolo11n.pt"
print(f"[FastAPI] Loading model: {model_path}")
model = YOLO(model_path)

@app.get("/api/sensors")
def get_sensors():
    # Mock data for demonstration
    return {
        "sensors": [
            {"id": 1, "name": "AUV Alpha", "battery": 85, "status": "Active"},
            {"id": 2, "name": "AUV Beta", "battery": 62, "status": "Active"},
            {"id": 3, "name": "AUV Gamma", "battery": 15, "status": "Low Battery"},
            {"id": 4, "name": "AUV Delta", "battery": 90, "status": "Active"},
        ]
    }

@app.post("/api/detect/{sensor_id}")
async def detect_objects(sensor_id: int, file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    
    # YOLO prediction
    results = model.predict(img_np, conf=0.25, verbose=False)
    
    # Draw bounding boxes (OpenCV uses BGR, so convert)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    detections = []
    
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id] if model.names else str(cls_id)
            
            # Draw box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 120), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(img_bgr, text, (x1, max(15, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)
                        
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    # Encode result image to base64
    _, buffer = cv2.imencode('.jpg', img_bgr)
    b64_image = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "sensor_id": sensor_id,
        "detections": detections,
        "image_b64": b64_image
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
