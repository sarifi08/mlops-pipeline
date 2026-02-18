"""
Fraud Detection API — serves predictions with A/B testing.

Features:
    - FastAPI endpoint for real-time predictions
    - A/B testing between model versions (user-based split)
    - Prometheus metrics for monitoring
    - Request validation with Pydantic
    - Health check endpoint

Deployment:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional
import pandas as pd
import joblib
import hashlib
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import time


# ── Pydantic Models (request validation) ──────────────────────────────────────

class Transaction(BaseModel):
    """
    Incoming transaction to score for fraud.
    All features the model expects.
    """
    user_id: str = Field(..., description="Unique user identifier for A/B testing")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    merchant_risk: float = Field(..., ge=0, le=1, description="Merchant risk score")
    card_age_days: int = Field(..., ge=0, description="Card age in days")
    distance_km: float = Field(..., ge=0, description="Distance from usual location")
    num_recent_txns: int = Field(..., ge=0, description="Transactions in last hour")
    is_international: int = Field(..., ge=0, le=1, description="0=domestic, 1=international")


class PredictionResponse(BaseModel):
    """API response with prediction and metadata."""
    model_config = ConfigDict(protected_namespaces=())

    is_fraud: bool
    fraud_probability: float
    model_version: str
    latency_ms: float
    timestamp: str


# ── Prometheus Metrics ─────────────────────────────────────────────────────────

prediction_counter = Counter(
    "fraud_predictions_total",
    "Total predictions made",
    ["model_version", "prediction"]
)

prediction_latency = Histogram(
    "fraud_prediction_latency_seconds",
    "Time to make prediction",
    ["model_version"]
)

fraud_rate_gauge = Gauge(
    "fraud_detection_rate",
    "Current fraud detection rate",
    ["model_version"]
)

# Rolling window for fraud rate calculation
recent_predictions = {"model_a": [], "model_b": []}


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_models():
    """
    Load both model versions for A/B testing.

    In production:
        - Model A = current production model
        - Model B = new candidate model being tested

    For this demo:
        - Model A = saved model
        - Model B = same model (you'd train a different one in practice)
    """
    try:
        model_a = joblib.load("model/fraud_model.pkl")
        model_b = joblib.load("model/fraud_model.pkl")  # In reality, load different model
        print("✅ Models loaded successfully")
        return model_a, model_b
    except FileNotFoundError:
        print("⚠️  Model file not found. Run train.py first.")
        return None, None


MODEL_A, MODEL_B = load_models()


# ── A/B Testing Logic ──────────────────────────────────────────────────────────

def assign_model(user_id: str) -> str:
    """
    Assign user to model A or B consistently.

    Uses hash of user_id so:
        - Same user always gets same model (consistent experience)
        - 50/50 distribution across population (fair comparison)

    This is exactly how Netflix, Spotify, etc do A/B testing.
    """
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "model_a" if hash_value % 2 == 0 else "model_b"


# ── FastAPI Application ────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with A/B testing",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """Health check endpoint — used by load balancers."""
    if MODEL_A is None or MODEL_B is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": MODEL_A is not None and MODEL_B is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """
    Predict if a transaction is fraudulent.

    A/B Testing:
        - User is consistently assigned to model A or B
        - Both models are evaluated on same traffic
        - Metrics tracked separately per model
    """
    if MODEL_A is None or MODEL_B is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    start_time = time.time()

    # A/B assignment
    assigned_model = assign_model(transaction.user_id)
    model = MODEL_A if assigned_model == "model_a" else MODEL_B

    # Prepare features
    features = pd.DataFrame([{
        "amount": transaction.amount,
        "hour": transaction.hour,
        "merchant_risk": transaction.merchant_risk,
        "card_age_days": transaction.card_age_days,
        "distance_km": transaction.distance_km,
        "num_recent_txns": transaction.num_recent_txns,
        "is_international": transaction.is_international,
    }])

    # Predict
    fraud_prob = float(model.predict_proba(features)[0][1])
    is_fraud = fraud_prob >= 0.5

    latency = (time.time() - start_time) * 1000  # ms

    # Update metrics
    prediction_counter.labels(
        model_version=assigned_model,
        prediction="fraud" if is_fraud else "legit"
    ).inc()

    prediction_latency.labels(model_version=assigned_model).observe(latency / 1000)

    # Update rolling fraud rate (last 1000 predictions)
    recent_predictions[assigned_model].append(1 if is_fraud else 0)
    if len(recent_predictions[assigned_model]) > 1000:
        recent_predictions[assigned_model].pop(0)

    if recent_predictions[assigned_model]:
        fraud_rate = sum(recent_predictions[assigned_model]) / len(recent_predictions[assigned_model])
        fraud_rate_gauge.labels(model_version=assigned_model).set(fraud_rate)

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        model_version=assigned_model,
        latency_ms=round(latency, 2),
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.
    Scraped by Prometheus for monitoring dashboards.
    """
    return Response(generate_latest(), media_type="text/plain")


@app.get("/ab-stats")
def ab_stats():
    """
    Current A/B test statistics.
    Shows how models are performing in real-time.
    """
    stats = {}
    for model_name, preds in recent_predictions.items():
        if preds:
            stats[model_name] = {
                "total_predictions": len(preds),
                "fraud_rate": sum(preds) / len(preds),
                "fraud_count": sum(preds),
                "legit_count": len(preds) - sum(preds),
            }
        else:
            stats[model_name] = {
                "total_predictions": 0,
                "fraud_rate": 0.0,
                "fraud_count": 0,
                "legit_count": 0,
            }

    return {
        "timestamp": datetime.now().isoformat(),
        "model_a": stats.get("model_a", {}),
        "model_b": stats.get("model_b", {}),
        "note": "User-based A/B split — each user consistently sees same model"
    }


# ── Example Usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Fraud Detection API...")
    print("   Health check: http://localhost:8000/health")
    print("   Predict:      POST http://localhost:8000/predict")
    print("   Metrics:      http://localhost:8000/metrics")
    print("   A/B Stats:    http://localhost:8000/ab-stats")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
