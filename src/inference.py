from fastapi import FastAPI, UploadFile, File
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time
import os
from torchvision import models

app = FastAPI(title="RLVS Violence Detection API")

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'rlvs_predictions_total',
    'Total predictions made',
    ['result']
)
INFERENCE_LATENCY = Histogram(
    'rlvs_inference_latency_seconds',
    'Inference latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)
VIOLENCE_PROBABILITY = Gauge(
    'rlvs_latest_violence_probability',
    'Latest violence probability'
)
MODEL_ACCURACY = Gauge(
    'rlvs_model_avg_accuracy',
    'Model average accuracy'
)
MODEL_FORGETTING = Gauge(
    'rlvs_model_avg_forgetting',
    'Model average forgetting'
)
NV_ACCURACY = Gauge(
    'rlvs_nonviolence_accuracy',
    'NonViolence accuracy'
)
V_ACCURACY = Gauge(
    'rlvs_violence_accuracy',
    'Violence accuracy'
)

# Set known metrics from thesis results
MODEL_ACCURACY.set(93.85)
MODEL_FORGETTING.set(3.42)
NV_ACCURACY.set(93.17)
V_ACCURACY.set(94.54)

CLASS_NAMES = ["NonViolence", "Violence"]
MODEL_PATH  = os.environ.get("MODEL_PATH", "/app/models/model_final.pth")
model       = None
device      = "cpu"

# -------------------------
# Model Definition
# -------------------------
class ResNetWithHidden(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base    = models.resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        old_conv     = self.base.conv1
        self.base.conv1 = nn.Conv2d(
            6, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding, bias=False
        )

    def forward(self, x):
        return self.base(x)

# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            m  = ResNetWithHidden(num_classes=2)
            sd = torch.load(MODEL_PATH, map_location="cpu")
            m.load_state_dict(sd, strict=False)
            m.eval()
            model = m
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Model load error: {e}")
    else:
        print(f"Model file not found: {MODEL_PATH}")

# -------------------------
# Preprocess
# -------------------------
def preprocess_bytes(data: bytes) -> torch.Tensor:
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame = cv2.resize(frame, (224, 224))
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Motion channel — zeros for single image
    motion = np.zeros_like(rgb)
    stacked = np.concatenate([motion, rgb], axis=2).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
    return tensor

# -------------------------
# Endpoints
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start    = time.time()
    contents = await file.read()

    if model is None:
        return {"error": "Model not loaded", "prediction": None}

    tensor = preprocess_bytes(contents)

    with torch.no_grad():
        output       = model(tensor)
        probs        = torch.softmax(output, dim=1)
        violence_prob = probs[0][1].item()

    result  = "Violence" if violence_prob > 0.5 else "NonViolence"
    latency = time.time() - start

    PREDICTION_COUNTER.labels(result=result).inc()
    INFERENCE_LATENCY.observe(latency)
    VIOLENCE_PROBABILITY.set(violence_prob)

    return {
        "prediction":          result,
        "violence_probability": round(violence_prob, 4),
        "nonviolence_probability": round(1 - violence_prob, 4),
        "latency_ms":          round(latency * 1000, 2)
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
        "device":       device
    }

@app.get("/")
async def root():
    return {
        "message":  "RLVS Violence Detection API",
        "docs":     "/docs",
        "health":   "/health",
        "metrics":  "/metrics",
        "predict":  "POST /predict"
    }