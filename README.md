# MLOps Pipeline — Production ML System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/sarifi08/mlops-pipeline/actions/workflows/mlops.yml/badge.svg)](https://github.com/sarifi08/mlops-pipeline/actions/workflows/mlops.yml)

> End-to-end MLOps pipeline for fraud detection — from training to A/B-tested, monitored production deployment with automated rollback.

## 🎯 What This Demonstrates

Most ML projects stop at "model.ipynb works on my laptop." This project shows you can:

- ✅ Automate training via GitHub Actions
- ✅ Deploy with zero downtime
- ✅ A/B test model versions in production
- ✅ Monitor performance in real-time
- ✅ Rollback bad deployments automatically
- ✅ Gate deployments on performance thresholds

This is what separates ML engineers from ML researchers.

## 🏗️ Architecture

```
┌─────────────┐
│   GitHub    │  Trigger: push to main, scheduled, manual
│   Actions   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         CI/CD Pipeline                  │
├─────────────────────────────────────────┤
│ 1. Train model → log to MLflow          │
│ 2. Run tests   → check thresholds       │
│ 3. Build Docker image                   │
│ 4. Deploy to production                 │
│ 5. Monitor deployment                   │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│      Production API (FastAPI)           │
├─────────────────────────────────────────┤
│  /predict  → fraud detection            │
│  /health   → healthcheck                │
│  /metrics  → Prometheus metrics         │
│  /ab-stats → A/B test results           │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│           Monitoring Stack              │
├─────────────────────────────────────────┤
│  Prometheus → metrics collection        │
│  Alerts     → performance degradation   │
│  MLflow     → experiment tracking       │
└─────────────────────────────────────────┘
```

## 📁 Project Structure

```
mlops-pipeline/
├── model/
│   └── train.py              # Training script (called by CI/CD)
├── api/
│   └── serve.py              # FastAPI service with A/B testing
├── monitoring/
│   ├── prometheus.yml        # Metrics collection config
│   └── alerts.yml            # Alerting rules
├── tests/
│   ├── test_api.py           # API tests
│   └── check_performance.py  # Performance gate (blocks bad models)
├── deployment/
│   └── Dockerfile            # Container for production
├── .github/workflows/
│   └── mlops.yml             # CI/CD automation
└── README.md
```

## 🚀 Quick Start

### 1. Train the Model Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (saves to model/fraud_model.pkl)
python model/train.py
```

### 2. Run the API

```bash
# Start API server
uvicorn api.serve:app --reload

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "amount": 250.0,
    "hour": 23,
    "merchant_risk": 0.8,
    "card_age_days": 15,
    "distance_km": 500,
    "num_recent_txns": 10,
    "is_international": 1
  }'

# Check A/B test stats
curl http://localhost:8000/ab-stats
```

### 3. Run Monitoring Stack

```bash
# Start Prometheus
docker run -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# View metrics at http://localhost:9090
```

### 4. Run Tests

```bash
# Unit tests
pytest tests/test_api.py -v

# Performance gate check
python tests/check_performance.py
```

## 🎯 Key MLOps Patterns Implemented

### 1. CI/CD Automation

Every push to `main` triggers:
```
Train → Test → Build → Deploy → Monitor
```

Manual trigger also available for retraining on demand.

### 2. A/B Testing

User-based model assignment:
```python
def assign_model(user_id: str) -> str:
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "model_a" if hash_value % 2 == 0 else "model_b"
```

- Same user always sees same model (consistent experience)
- 50/50 split across users (fair comparison)
- Metrics tracked separately per model

### 3. Performance Gates

Deployment is **blocked** if model doesn't meet thresholds:

```python
THRESHOLDS = {
    "f1_score": 0.70,   # Must detect 70% of fraud
    "roc_auc": 0.85,    # Must have good discrimination
}
```

This prevents bad models from reaching production.

### 4. Monitoring & Alerting

Prometheus alerts fire when:
- Prediction latency > 100ms
- Fraud rate spikes (data drift indicator)
- No predictions in 5 minutes (service down)
- A/B test shows significant difference

### 5. Model Registry

MLflow tracks:
- Every training run
- Hyperparameters used
- Metrics achieved
- Feature importances
- Model artifacts

## 📊 Metrics Tracked

### API Metrics
- `fraud_predictions_total`: Total predictions made
- `fraud_prediction_latency_seconds`: Prediction latency histogram
- `fraud_detection_rate`: Current fraud detection rate

### A/B Test Metrics
- Per-model fraud rates
- Per-model latency
- Per-model prediction counts

## 🔄 Deployment Flow

```
1. Developer pushes code
       ↓
2. GitHub Actions triggers
       ↓
3. Train model + log to MLflow
       ↓
4. Run automated tests
       ↓
5. Check performance thresholds ← GATE: blocks if model is bad
       ↓
6. Build Docker image
       ↓
7. Deploy to production (blue-green)
       ↓
8. Monitor for 5 minutes
       ↓
9. Rollback if alerts fire, else continue
```

## 🧪 A/B Testing Analysis

After collecting data, analyze with:

```python
import requests

# Get current stats
stats = requests.get("http://localhost:8000/ab-stats").json()

model_a = stats["model_a"]
model_b = stats["model_b"]

# Compare fraud rates
print(f"Model A fraud rate: {model_a['fraud_rate']:.2%}")
print(f"Model B fraud rate: {model_b['fraud_rate']:.2%}")

# Statistical significance test
from scipy.stats import chi2_contingency

table = [
    [model_a['fraud_count'], model_a['legit_count']],
    [model_b['fraud_count'], model_b['legit_count']]
]

chi2, p_value, dof, expected = chi2_contingency(table)

if p_value < 0.05:
    print(f"✅ Difference is statistically significant (p={p_value:.4f})")
else:
    print(f"⚠️  Difference not significant (p={p_value:.4f})")
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 fraud-detection-api

# Health check
curl http://localhost:8000/health
```

## 📈 Production Checklist

Before deploying to production:

- [ ] Model meets performance thresholds
- [ ] All tests pass
- [ ] Prometheus alerts configured
- [ ] A/B test logging enabled
- [ ] Rollback procedure documented
- [ ] On-call rotation in place
- [ ] Resource limits set (CPU, memory)
- [ ] Auto-scaling configured
- [ ] Backup deployment ready

## 🔧 Configuration

### Environment Variables

```bash
MLFLOW_TRACKING_URI=http://mlflow-server:5000
PROMETHEUS_URL=http://prometheus:9090
ALERT_WEBHOOK=https://hooks.slack.com/...
```

### Performance Thresholds

Edit `tests/check_performance.py`:

```python
THRESHOLDS = {
    "f1_score": 0.70,
    "roc_auc": 0.85,
}
```

## 🎯 What Interviewers Look For

This project demonstrates you understand:

1. **CI/CD for ML** — not just DevOps, but ML-specific challenges
2. **A/B testing** — how to validate models in production
3. **Monitoring** — what to track and when to alert
4. **Performance gates** — preventing bad models from deploying
5. **Production readiness** — latency requirements, error handling

## 🚨 Common Pitfalls Avoided

❌ Training in production → ✅ Train in CI/CD, deploy artifact
❌ No performance gates → ✅ Automated threshold checks
❌ Random A/B split → ✅ User-based consistent assignment
❌ No monitoring → ✅ Prometheus + alerts
❌ Manual deployment → ✅ Fully automated pipeline

## 📚 Further Reading

- [MLOps Principles](https://ml-ops.org/)
- [A/B Testing Guide](https://exp-platform.com/)
- [Monitoring ML in Production](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)

---

**Built by [sarifi08](https://github.com/sarifi08)** | Demonstrates production ML engineering skills beyond model training
