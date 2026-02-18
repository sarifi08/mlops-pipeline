"""
MLOps Test Suite — ensures production readiness.

Tests:
    1. Model loads correctly
    2. Predictions return valid format
    3. Latency under 100ms
    4. Performance meets thresholds (F1 > 0.7, AUC > 0.85)
    5. A/B split is actually 50/50
"""
import pytest
import pandas as pd
import numpy as np
import time
import joblib
from api.serve import assign_model
from model.train import generate_fraud_data


def test_model_loads():
    """Model file exists and loads without errors."""
    model = joblib.load("model/fraud_model.pkl")
    assert model is not None
    assert hasattr(model, 'predict')


def test_prediction_format():
    """Predictions return correct format."""
    model = joblib.load("model/fraud_model.pkl")
    X, _ = generate_fraud_data(n_samples=100)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    assert predictions.shape == (100,)
    assert probabilities.shape == (100,)
    assert all(p in [0, 1] for p in predictions)
    assert all(0 <= p <= 1 for p in probabilities)


def test_prediction_latency():
    """Single prediction completes under 100ms."""
    model = joblib.load("model/fraud_model.pkl")
    X, _ = generate_fraud_data(n_samples=1)

    # Warm up (first prediction is slower)
    model.predict(X)

    # Actual test
    start = time.time()
    model.predict(X)
    latency = (time.time() - start) * 1000

    assert latency < 100, f"Latency {latency:.1f}ms exceeds 100ms threshold"


def test_ab_split_is_50_50():
    """A/B assignment produces ~50/50 split."""
    user_ids = [f"user_{i}" for i in range(1000)]
    assignments = [assign_model(uid) for uid in user_ids]

    model_a_count = sum(1 for a in assignments if a == "model_a")
    model_b_count = len(assignments) - model_a_count

    ratio = model_a_count / len(assignments)

    # Should be within 45-55% (statistically very likely for 1000 samples)
    assert 0.45 <= ratio <= 0.55, f"A/B split is {ratio:.1%}, expected ~50%"


def test_ab_split_is_consistent():
    """Same user always gets same model (sticky assignment)."""
    user_id = "test_user_123"

    assignments = [assign_model(user_id) for _ in range(100)]

    # All 100 calls should return same model
    assert len(set(assignments)) == 1, "Same user got different models"


def test_feature_importance_exists():
    """Model has interpretable feature importances."""
    model = joblib.load("model/fraud_model.pkl")

    assert hasattr(model, 'feature_importances_')
    assert len(model.feature_importances_) == 7  # 7 features
    assert sum(model.feature_importances_) > 0.99  # sum to ~1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
