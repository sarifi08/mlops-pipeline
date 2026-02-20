"""
Fraud Detection API — serves predictions with A/B testing.

Features:
    - FastAPI endpoint for real-time predictions
    - A/B testing between model versions (user-based split)
    - Prometheus metrics for monitoring
    - Data drift detection endpoint
    - Model explainability endpoint (feature contributions)
    - Structured JSON logging with request tracing
    - CORS middleware for cross-origin requests
    - Request/response middleware with latency tracking
    - Health check with dependency status

Deployment:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import joblib
import hashlib
import uuid
from datetime import datetime, timezone
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import time

from config.settings import settings
from config.logging import get_logger
from monitoring.drift import DriftDetector
from model.train import generate_fraud_data

logger = get_logger(__name__)


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

    request_id: str
    is_fraud: bool
    fraud_probability: float
    model_version: str
    latency_ms: float
    timestamp: str


class ExplainResponse(BaseModel):
    """Explainability response — shows which features drove the prediction."""
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    is_fraud: bool
    fraud_probability: float
    feature_contributions: Dict[str, float]
    top_risk_factors: List[Dict[str, Any]]
    model_version: str


class DriftResponse(BaseModel):
    """Data drift detection results."""
    drift_detected: bool
    overall_drift_score: float
    feature_reports: Dict[str, Any]
    alerts: List[str]
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
    ["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

fraud_rate_gauge = Gauge(
    "fraud_detection_rate",
    "Current fraud detection rate",
    ["model_version"]
)

request_counter = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

active_requests = Gauge(
    "http_requests_active",
    "Currently active requests"
)

# Rolling window for fraud rate calculation
recent_predictions: Dict[str, list] = {"model_a": [], "model_b": []}

# Drift detector (initialized with training data)
drift_detector = DriftDetector()


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
        model_a = joblib.load(settings.model_path)
        model_b = joblib.load(settings.model_path)  # In reality, load different model
        logger.info(
            "Models loaded successfully",
            extra={"model_path": settings.model_path}
        )

        # Initialize drift detector with training data distribution
        X_ref, _ = generate_fraud_data(n_samples=5000)
        drift_detector.set_reference(X_ref)
        logger.info("Drift detector initialized with reference distribution")

        return model_a, model_b
    except FileNotFoundError:
        logger.warning("Model file not found — run train.py first", extra={"path": settings.model_path})
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
    description=(
        "Real-time fraud detection with A/B testing, model explainability, "
        "and data drift monitoring. Production-grade MLOps pipeline."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware — allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """
    Middleware that adds:
    - Request ID for tracing
    - Request latency tracking
    - Structured access logging
    """
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    active_requests.inc()
    start = time.time()

    response = await call_next(request)

    latency = (time.time() - start) * 1000
    active_requests.dec()

    # Track request metrics
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    # Structured access log
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency, 2),
        },
    )

    response.headers["X-Request-ID"] = request_id
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Health check endpoint — used by load balancers and Kubernetes.

    Returns dependency status for each component.
    """
    models_loaded = MODEL_A is not None and MODEL_B is not None
    drift_ready = drift_detector.reference_data is not None

    status = "healthy" if models_loaded else "degraded"

    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "dependencies": {
            "models_loaded": models_loaded,
            "drift_detector_ready": drift_ready,
            "ab_testing_enabled": settings.ab_test_enabled,
        },
        "environment": settings.environment,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction, request: Request):
    """
    Predict if a transaction is fraudulent.

    A/B Testing:
        - User is consistently assigned to model A or B
        - Both models are evaluated on same traffic
        - Metrics tracked separately per model
    """
    if MODEL_A is None or MODEL_B is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
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
    is_fraud = fraud_prob >= settings.prediction_threshold

    latency = (time.time() - start_time) * 1000  # ms

    # Update Prometheus metrics
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

    # Feed to drift detector
    drift_detector.add_production_sample(features.iloc[0].to_dict())

    # Structured prediction log
    logger.info(
        "Prediction made",
        extra={
            "request_id": request_id,
            "user_id": transaction.user_id,
            "model_version": assigned_model,
            "fraud_probability": round(fraud_prob, 4),
            "is_fraud": is_fraud,
            "latency_ms": round(latency, 2),
            "amount": transaction.amount,
        },
    )

    return PredictionResponse(
        request_id=request_id,
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        model_version=assigned_model,
        latency_ms=round(latency, 2),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/explain", response_model=ExplainResponse)
def explain_prediction(transaction: Transaction, request: Request):
    """
    Explain a fraud prediction — which features drove the decision.

    Uses feature contribution analysis (based on Random Forest feature importances
    weighted by the transaction's feature values relative to population statistics).

    This helps investigators understand WHY a transaction was flagged.
    """
    if MODEL_A is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    # Prepare features
    feature_dict = {
        "amount": transaction.amount,
        "hour": transaction.hour,
        "merchant_risk": transaction.merchant_risk,
        "card_age_days": transaction.card_age_days,
        "distance_km": transaction.distance_km,
        "num_recent_txns": transaction.num_recent_txns,
        "is_international": transaction.is_international,
    }
    features = pd.DataFrame([feature_dict])

    # Get prediction
    fraud_prob = float(MODEL_A.predict_proba(features)[0][1])
    is_fraud = fraud_prob >= settings.prediction_threshold

    # Feature contribution analysis
    # Use feature importances weighted by z-score of each feature
    importances = MODEL_A.feature_importances_
    feature_names = list(feature_dict.keys())

    # Get reference stats for z-score calculation
    ref_stats = drift_detector.reference_stats

    contributions = {}
    for i, name in enumerate(feature_names):
        if name in ref_stats:
            ref_mean = ref_stats[name]["mean"]
            ref_std = ref_stats[name]["std"]
            if ref_std > 0:
                z_score = (feature_dict[name] - ref_mean) / ref_std
            else:
                z_score = 0.0
            # Contribution = importance * how unusual the value is
            contributions[name] = round(float(importances[i] * abs(z_score)), 4)
        else:
            contributions[name] = round(float(importances[i]), 4)

    # Sort by contribution (highest risk factors first)
    sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    top_risk_factors = [
        {
            "feature": name,
            "contribution": score,
            "value": feature_dict[name],
            "interpretation": _interpret_feature(name, feature_dict[name], ref_stats.get(name, {})),
        }
        for name, score in sorted_contributions[:5]
    ]

    logger.info(
        "Explanation generated",
        extra={
            "request_id": request_id,
            "is_fraud": is_fraud,
            "top_feature": sorted_contributions[0][0] if sorted_contributions else "unknown",
        },
    )

    return ExplainResponse(
        request_id=request_id,
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        feature_contributions=contributions,
        top_risk_factors=top_risk_factors,
        model_version="model_a",
    )


def _interpret_feature(name: str, value: float, stats: Dict) -> str:
    """Generate human-readable interpretation of a feature's contribution."""
    if not stats:
        return f"{name} = {value}"

    mean = stats.get("mean", 0)
    std = stats.get("std", 1)

    if std == 0:
        return f"{name} = {value}"

    z = (value - mean) / std

    interpretations = {
        "amount": f"${value:.0f} ({'very high' if z > 2 else 'high' if z > 1 else 'normal' if z > -1 else 'low'})",
        "hour": f"Hour {int(value)} ({'late night — risky' if value in [0,1,2,3,22,23] else 'normal hours'})",
        "merchant_risk": f"Risk {value:.2f} ({'high risk merchant' if value > 0.7 else 'moderate' if value > 0.4 else 'low risk'})",
        "card_age_days": f"{int(value)} days ({'very new card — risky' if value < 30 else 'new' if value < 90 else 'established'})",
        "distance_km": f"{value:.0f}km ({'very far from home' if z > 2 else 'far' if z > 1 else 'near home'})",
        "num_recent_txns": f"{int(value)} recent txns ({'burst activity' if z > 2 else 'high' if z > 1 else 'normal'})",
        "is_international": f"{'International' if value == 1 else 'Domestic'} transaction",
    }

    return interpretations.get(name, f"{name} = {value} (z-score: {z:.1f})")


@app.get("/drift", response_model=DriftResponse)
def check_drift():
    """
    Check for data drift in recent production data.

    Compares the distribution of recent predictions against
    the training data distribution using PSI and KS tests.

    Call this periodically or after observing anomalies.
    """
    if drift_detector.reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")

    if len(drift_detector.production_buffer) < 50:
        return DriftResponse(
            drift_detected=False,
            overall_drift_score=0.0,
            feature_reports={},
            alerts=[f"Need at least 50 samples, have {len(drift_detector.production_buffer)}"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    try:
        report = drift_detector.check_drift()
        return DriftResponse(
            drift_detected=report.drift_detected,
            overall_drift_score=report.overall_drift_score,
            feature_reports=report.feature_reports,
            alerts=report.alerts,
            timestamp=report.timestamp,
        )
    except Exception as e:
        logger.error("Drift check failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")


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
                "fraud_rate": round(sum(preds) / len(preds), 4),
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_a": stats.get("model_a", {}),
        "model_b": stats.get("model_b", {}),
        "ab_test_enabled": settings.ab_test_enabled,
        "traffic_split": settings.ab_test_traffic_split,
        "note": "User-based A/B split — each user consistently sees same model",
    }


# ── Example Usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Fraud Detection API", extra={
        "host": settings.api_host,
        "port": settings.api_port,
        "environment": settings.environment,
    })

    print("\n🚀 Starting Fraud Detection API...")
    print(f"   Environment: {settings.environment}")
    print(f"   Docs:        http://localhost:{settings.api_port}/docs")
    print(f"   Health:      http://localhost:{settings.api_port}/health")
    print(f"   Predict:     POST http://localhost:{settings.api_port}/predict")
    print(f"   Explain:     POST http://localhost:{settings.api_port}/explain")
    print(f"   Drift:       http://localhost:{settings.api_port}/drift")
    print(f"   Metrics:     http://localhost:{settings.api_port}/metrics")
    print(f"   A/B Stats:   http://localhost:{settings.api_port}/ab-stats")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
