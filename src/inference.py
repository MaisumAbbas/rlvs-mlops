from fastapi import FastAPI, UploadFile, File
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import torch
import cv2
import numpy as np
import time
import os

app = FastAPI(title="RLVS Violence Detection API")

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'predictions_total', 
    'Total predictions made',
    ['result', 'variant']
)
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Time spent doing inference',
    ['variant']
)
VIOLENCE_DETECTED_GAUGE = Gauge(
    'violence_probability',
    'Latest violence probability score'
)
MODEL_ACCURACY_GAUGE = Gauge(
    'model_accuracy',
    'Current model accuracy from last evaluation'
)

# Load model on startup
model = None
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/model_final.pth")

@app.on_event("startup")
async def load_model():
    global model
    # Load your ResNet18 model here
    # model = make_model(num_classes=2, base_weights=MODEL_PATH)
    # model.eval()
    print(f"Model loaded from {MODEL_PATH}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    
    # Read video/image bytes
    contents = await file.read()
    
    # Preprocess (your existing logic)
    # tensor = preprocess_video(contents)
    
    # Inference
    with torch.no_grad():
        # output = model(tensor)
        # probs = torch.softmax(output, dim=1)
        # violence_prob = probs[0][1].item()
        violence_prob = 0.5  # placeholder
    
    result = "Violence" if violence_prob > 0.5 else "NonViolence"
    latency = time.time() - start
    
    # Update Prometheus metrics
    PREDICTION_COUNTER.labels(result=result, variant="ER-MixUp").inc()
    INFERENCE_LATENCY.labels(variant="ER-MixUp").observe(latency)
    VIOLENCE_DETECTED_GAUGE.set(violence_prob)
    
    return {
        "prediction": result,
        "violence_probability": violence_prob,
        "latency_ms": latency * 1000
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
async def root():
    return {"message": "RLVS Violence Detection API", "docs": "/docs"}