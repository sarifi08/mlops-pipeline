"""
Performance Threshold Checker — gates deployment.

This runs in CI/CD and FAILS the build if model performance
is below production thresholds.

Prevents deploying bad models to production.
"""
import sys
from model.train import FraudDetectionModel, generate_fraud_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


# Production thresholds — model must beat these to deploy
THRESHOLDS = {
    "f1_score": 0.70,
    "roc_auc": 0.85,
}


def check_performance():
    """
    Load model, test on held-out data, check against thresholds.
    Exit code 1 if model fails thresholds (blocks deployment).
    """
    print("=" * 60)
    print("  CHECKING MODEL PERFORMANCE THRESHOLDS")
    print("=" * 60)

    # Load model
    try:
        model = FraudDetectionModel.load("model/fraud_model.pkl")
    except FileNotFoundError:
        print("❌ Model file not found. Run training first.")
        sys.exit(1)

    # Generate test data
    X, y = generate_fraud_data(n_samples=5000)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nModel Performance:")
    print(f"  F1 Score: {f1:.3f}  (threshold: {THRESHOLDS['f1_score']})")
    print(f"  ROC AUC:  {auc:.3f}  (threshold: {THRESHOLDS['roc_auc']})")

    # Check thresholds
    passed = True

    if f1 < THRESHOLDS["f1_score"]:
        print(f"\n❌ FAIL: F1 score {f1:.3f} below threshold {THRESHOLDS['f1_score']}")
        passed = False

    if auc < THRESHOLDS["roc_auc"]:
        print(f"\n❌ FAIL: ROC AUC {auc:.3f} below threshold {THRESHOLDS['roc_auc']}")
        passed = False

    if passed:
        print("\n✅ PASS: Model meets all production thresholds")
        print("   Deployment approved.")
        sys.exit(0)
    else:
        print("\n⛔ Model below production thresholds.")
        print("   Deployment BLOCKED.")
        print("   Review model before deploying to production.")
        sys.exit(1)


if __name__ == "__main__":
    check_performance()
