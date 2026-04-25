"""
Custom Prometheus metrics jo inference k ilawa log karne hain.
Jaise model accuracy, dataset stats, etc.
inference.py ke saath milke kaam karta hai.
"""
from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry

# Shared registry
REGISTRY = CollectorRegistry()

# Model performance gauges
MODEL_AVG_ACCURACY = Gauge(
    'rlvs_model_avg_accuracy',
    'Current deployed model average accuracy (%)',
    registry=REGISTRY
)
MODEL_AVG_FORGETTING = Gauge(
    'rlvs_model_avg_forgetting',
    'Current deployed model average forgetting (%)',
    registry=REGISTRY
)
NV_ACCURACY = Gauge(
    'rlvs_nonviolence_accuracy',
    'NonViolence class accuracy (%)',
    registry=REGISTRY
)
V_ACCURACY = Gauge(
    'rlvs_violence_accuracy',
    'Violence class accuracy (%)',
    registry=REGISTRY
)

# Request metrics
PREDICTION_COUNTER = Counter(
    'rlvs_predictions_total',
    'Total number of predictions',
    ['result', 'variant'],
    registry=REGISTRY
)
INFERENCE_LATENCY = Histogram(
    'rlvs_inference_latency_seconds',
    'Inference latency in seconds',
    ['variant'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY
)
VIOLENCE_PROBABILITY = Gauge(
    'rlvs_latest_violence_probability',
    'Latest violence probability score from model',
    registry=REGISTRY
)

# Drift / data metrics
BUFFER_SIZE_GAUGE = Gauge(
    'rlvs_replay_buffer_size',
    'Current replay buffer size',
    registry=REGISTRY
)
MOTION_MAGNITUDE_GAUGE = Gauge(
    'rlvs_avg_motion_magnitude',
    'Average motion magnitude of last batch',
    registry=REGISTRY
)

def update_model_metrics(avg_acc, avg_forget, nv_acc, v_acc):
    """Call this after each evaluation to update Prometheus gauges."""
    MODEL_AVG_ACCURACY.set(avg_acc)
    MODEL_AVG_FORGETTING.set(avg_forget)
    NV_ACCURACY.set(nv_acc)
    V_ACCURACY.set(v_acc)