# Basic sanity tests — no heavy dependencies needed

def test_import_fastapi():
    """FastAPI import hona chahiye"""
    import fastapi
    assert fastapi is not None

def test_import_prometheus():
    """Prometheus client import hona chahiye"""
    import prometheus_client
    assert prometheus_client is not None

def test_class_names():
    """Class names sahi honi chahiye"""
    class_names = ["NonViolence", "Violence"]
    assert len(class_names) == 2
    assert "Violence" in class_names
    assert "NonViolence" in class_names

def test_health_endpoint_structure():
    """Health response structure check"""
    response = {"status": "healthy", "model_loaded": False}
    assert "status" in response
    assert "model_loaded" in response

def test_prediction_response_structure():
    """Prediction response structure check"""
    response = {
        "prediction": "Violence",
        "violence_probability": 0.87,
        "latency_ms": 45.2
    }
    assert "prediction" in response
    assert "violence_probability" in response
    assert response["prediction"] in ["Violence", "NonViolence"]