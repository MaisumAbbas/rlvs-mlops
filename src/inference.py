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
import tempfile
from torchvision import models

app = FastAPI(title="RLVS Violence Detection API")

# -------------------------
# Prometheus Metrics
# -------------------------
PREDICTION_COUNTER = Counter(
    'rlvs_predictions_total',
    'Total predictions made',
    ['result']
)
INFERENCE_LATENCY = Histogram(
    'rlvs_inference_latency_seconds',
    'Inference latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
VIOLENCE_PROBABILITY = Gauge(
    'rlvs_latest_violence_probability',
    'Latest violence probability score'
)
MODEL_ACCURACY = Gauge(
    'rlvs_model_avg_accuracy',
    'Model average accuracy from thesis results'
)
MODEL_FORGETTING = Gauge(
    'rlvs_model_avg_forgetting',
    'Model average forgetting from thesis results'
)
NV_ACCURACY = Gauge(
    'rlvs_nonviolence_accuracy',
    'NonViolence class accuracy'
)
V_ACCURACY = Gauge(
    'rlvs_violence_accuracy',
    'Violence class accuracy'
)

# Pre-set known metrics from thesis ER+MixUp best run
MODEL_ACCURACY.set(93.85)
MODEL_FORGETTING.set(3.42)
NV_ACCURACY.set(93.17)
V_ACCURACY.set(94.54)

# -------------------------
# Config
# -------------------------
CLASS_NAMES = ["NonViolence", "Violence"]
MODEL_PATH  = os.environ.get("MODEL_PATH", "/app/models/model_final.pth")
DEVICE      = "cpu"  # EC2 free tier — no GPU
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

model = None

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
            padding=old_conv.padding,
            bias=False
        )

    def forward(self, x):
        return self.base(x)

# -------------------------
# Startup — Load Model
# -------------------------
@app.on_event("startup")
async def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found at {MODEL_PATH}")
        print("[WARNING] API will run but /predict will return error")
        return
    try:
        m  = ResNetWithHidden(num_classes=2)
        sd = torch.load(MODEL_PATH, map_location=DEVICE)
        m.load_state_dict(sd, strict=False)
        m.eval()
        model = m
        print(f"[OK] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        model = None

# -------------------------
# Preprocess — Single Image
# -------------------------
def preprocess_image(data: bytes) -> torch.Tensor:
    """
    Single image bytes → 6-channel tensor.
    Motion channels are zeros (no temporal info available).
    """
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

    frame  = cv2.resize(frame, (224, 224))
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    motion = np.zeros_like(rgb)  # no motion for single image

    stacked = np.concatenate([motion, rgb], axis=2).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
    return tensor

# -------------------------
# Preprocess — Video
# -------------------------
def preprocess_video(data: bytes) -> torch.Tensor:
    """
    Video bytes → 6-channel tensor (motion map + avg RGB).
    Matches exactly how train.py built the cache.
    """
    tmp_path = None
    try:
        # Write to temp file — cv2 needs a file path
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, dir="/tmp"
        ) as f:
            f.write(data)
            tmp_path = f.name

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 2:
            cap.release()
            return preprocess_image(data)

        frames_per_video = 16
        if total_frames < frames_per_video:
            indices = list(range(total_frames)) + \
                      [total_frames - 1] * (frames_per_video - total_frames)
        else:
            indices = np.linspace(
                0, total_frames - 1, frames_per_video
            ).astype(int)

        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            return preprocess_image(data)

        # Motion map — same logic as train.py make_motion_cache
        motion = np.zeros_like(frames[0], dtype=np.float32)
        for i in range(1, len(frames)):
            diff    = cv2.absdiff(frames[i], frames[i - 1]).astype(np.float32)
            motion += diff
        motion /= max(1, len(frames))
        motion  = motion.astype(np.uint8)
        avg     = np.mean(frames, axis=0).astype(np.uint8)

        # motion first (0:3), RGB second (3:6) — same as train.py
        stacked = np.concatenate([motion, avg], axis=2).astype(np.float32) / 255.0
        tensor  = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
        return tensor

    except Exception as e:
        print(f"[WARNING] Video preprocess failed: {e}, falling back to image mode")
        return preprocess_image(data)

    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# -------------------------
# POST /predict
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {
            "error":      "Model not loaded",
            "prediction": None,
            "hint":       f"Expected model at {MODEL_PATH}"
        }

    start    = time.time()
    contents = await file.read()

    if not contents:
        return {"error": "Empty file received", "prediction": None}

    # Detect input type from filename extension
    filename   = (file.filename or "").lower()
    ext        = os.path.splitext(filename)[1]
    input_type = "video" if ext in VIDEO_EXTS else "image"

    try:
        if input_type == "video":
            tensor = preprocess_video(contents)
        else:
            tensor = preprocess_image(contents)
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}", "prediction": None}

    try:
        with torch.no_grad():
            output        = model(tensor)
            probs         = torch.softmax(output, dim=1)
            violence_prob = float(probs[0][1].item())
            nv_prob       = float(probs[0][0].item())
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}", "prediction": None}

    result  = "Violence" if violence_prob > 0.5 else "NonViolence"
    latency = time.time() - start

    # Update Prometheus metrics
    PREDICTION_COUNTER.labels(result=result).inc()
    INFERENCE_LATENCY.observe(latency)
    VIOLENCE_PROBABILITY.set(violence_prob)

    return {
        "prediction":               result,
        "violence_probability":     round(violence_prob, 4),
        "nonviolence_probability":  round(nv_prob, 4),
        "latency_ms":               round(latency * 1000, 2),
        "input_type":               input_type,
        "model":                    "ResNet18-6ch-ER+MixUp"
    }

# -------------------------
# GET /metrics  (Prometheus scrape endpoint)
# -------------------------
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

# -------------------------
# GET /health
# -------------------------
@app.get("/health")
async def health():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
        "device":       DEVICE
    }

# -------------------------
# GET /
# -------------------------
@app.get("/")
async def root():
    return {
        "service":  "RLVS Violence Detection API",
        "variant":  "ER+MixUp (Motion-Aware Reservoir Sampling)",
        "docs":     "/docs",
        "health":   "/health",
        "metrics":  "/metrics",
        "predict":  "POST /predict — accepts image or video file"
    }