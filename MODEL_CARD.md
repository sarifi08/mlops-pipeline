# Model Card — Fraud Detection Model

## Model Details

| Field | Value |
|-------|-------|
| **Model Type** | Random Forest Classifier |
| **Framework** | scikit-learn 1.4.0 |
| **Task** | Binary classification (fraud vs. legitimate) |
| **Input** | 7 transaction features |
| **Output** | Fraud probability (0-1) + binary prediction |
| **Training Data** | 10,000 synthetic transactions (2% fraud rate) |
| **Inference Latency** | < 10ms (single prediction) |

## Intended Use

- **Primary use**: Real-time fraud detection for payment transactions
- **Users**: Fraud investigation teams, automated payment systems
- **Out of scope**: Credit scoring, identity verification, AML

## Training Data

Synthetic dataset mimicking real-world fraud patterns:

| Feature | Description | Fraud Signal |
|---------|-------------|-------------|
| `amount` | Transaction amount (USD) | Higher for fraud |
| `hour` | Hour of day (0-23) | Late night = risky |
| `merchant_risk` | Merchant risk score (0-1) | Higher for fraud |
| `card_age_days` | Card age in days | Newer cards = risky |
| `distance_km` | Distance from usual location | Farther = risky |
| `num_recent_txns` | Transactions in last hour | Burst = risky |
| `is_international` | 0=domestic, 1=international | International = risky |

**Class balance**: 98% legitimate, 2% fraudulent (realistic imbalance)

## Performance Metrics

| Metric | Validation | Test | Threshold |
|--------|-----------|------|-----------|
| **Accuracy** | ~0.99 | ~0.99 | — |
| **Precision** | ~0.90 | ~0.90 | — |
| **Recall** | ~0.85 | ~0.85 | — |
| **F1 Score** | ~0.87 | ~0.87 | ≥ 0.70 |
| **ROC AUC** | ~0.99 | ~0.99 | ≥ 0.85 |

## Limitations

- **Synthetic data**: Model trained on generated data, not real transactions
- **Feature coverage**: Limited to 7 features (real systems use 100+)
- **Concept drift**: Fraud patterns evolve; model needs periodic retraining
- **Class imbalance**: 2% fraud rate may not match all real-world scenarios
- **No sequential modeling**: Doesn't capture transaction sequences

## Ethical Considerations

- **False positives**: Legitimate transactions blocked → user frustration
- **False negatives**: Fraud not detected → financial loss
- **Bias**: Synthetic data doesn't capture real-world demographic biases
- **Transparency**: Feature contributions provided via `/explain` endpoint

## Monitoring

- **Data drift**: PSI and KS tests via `/drift` endpoint
- **Performance**: Prometheus metrics + Grafana dashboards
- **A/B testing**: User-based model comparison via `/ab-stats`
- **Alerts**: Prometheus rules for latency, fraud rate spikes, downtime

## Maintenance

- **Retraining trigger**: Data drift detected (PSI > 0.2) or scheduled weekly
- **Deployment gate**: F1 ≥ 0.70 and ROC AUC ≥ 0.85 required
- **Rollback**: Automated if alerts fire within 5 minutes of deployment
