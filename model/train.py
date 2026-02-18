"""
Fraud Detection Model — the ML component we'll put into production.

Scenario:
    A payment processor needs to detect fraudulent transactions in real-time.
    Every transaction needs a prediction within 100ms.

Model:
    Random Forest classifier on transaction features.
    Simple model on purpose — the focus is MLOps, not ML complexity.

This file handles:
    - Training the model
    - Logging experiments to MLflow
    - Registering models in MLflow Model Registry
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path


class FraudDetectionModel:
    """
    Fraud detection classifier with MLflow tracking.

    Why Random Forest?
        - Fast inference (<10ms typical)
        - Handles mixed feature types well
        - Interpretable feature importances
        - Good baseline performance
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # use all cores
        )
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        experiment_name: str = "fraud-detection"
    ) -> Dict[str, float]:
        """
        Train the model and log everything to MLflow.

        Returns:
            Dictionary of validation metrics
        """
        # Set up MLflow
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "random_state": self.model.random_state,
            })

            # Train
            print("Training model...")
            self.model.fit(X_train, y_train)
            self.feature_names = list(X_train.columns)

            # Evaluate
            y_pred = self.model.predict(X_val)
            y_prob = self.model.predict_proba(X_val)[:, 1]

            metrics = {
                "accuracy":  accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "recall":    recall_score(y_val, y_pred),
                "f1":        f1_score(y_val, y_pred),
                "roc_auc":   roc_auc_score(y_val, y_prob),
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log feature importances
            importances = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            mlflow.log_dict(importances, "feature_importances.json")

            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name="fraud-detection-model"
            )

            print(f"✅ Model trained | F1: {metrics['f1']:.3f} | AUC: {metrics['roc_auc']:.3f}")
            return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud (0 or 1)."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability (0-1)."""
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        """Save model to disk (for deployment)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FraudDetectionModel":
        """Load model from disk."""
        return joblib.load(path)


# ── Synthetic Data Generator ──────────────────────────────────────────────────

def generate_fraud_data(n_samples: int = 10000, fraud_rate: float = 0.02) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic fraud detection dataset.

    Features (what a payment processor would see):
        - amount: transaction amount in dollars
        - hour: hour of day (0-23)
        - merchant_risk: risk score of merchant (0-1)
        - card_age_days: how old is the card
        - distance_km: distance from usual location
        - num_recent_txns: transactions in last hour
        - is_international: 0 or 1

    Target:
        - is_fraud: 0 or 1
    """
    np.random.seed(42)

    # Legit transactions (98%)
    n_legit = int(n_samples * (1 - fraud_rate))
    legit_data = pd.DataFrame({
        "amount":          np.random.lognormal(3, 1, n_legit),  # $20-200 typical
        "hour":            np.random.randint(0, 24, n_legit),
        "merchant_risk":   np.random.beta(2, 5, n_legit),       # mostly low risk
        "card_age_days":   np.random.randint(30, 2000, n_legit),
        "distance_km":     np.random.gamma(2, 5, n_legit),      # near home
        "num_recent_txns": np.random.poisson(2, n_legit),       # few recent
        "is_international":np.random.choice([0,1], n_legit, p=[0.95, 0.05]),
    })

    # Fraudulent transactions (2%)
    n_fraud = n_samples - n_legit
    fraud_data = pd.DataFrame({
        "amount":          np.random.lognormal(5, 1.5, n_fraud),  # $150-1000 larger
        "hour":            np.random.choice([0,1,2,3,22,23], n_fraud),  # late night
        "merchant_risk":   np.random.beta(5, 2, n_fraud),         # higher risk
        "card_age_days":   np.random.randint(1, 90, n_fraud),     # newer cards
        "distance_km":     np.random.gamma(10, 20, n_fraud),      # far from home
        "num_recent_txns": np.random.poisson(8, n_fraud),         # burst activity
        "is_international":np.random.choice([0,1], n_fraud, p=[0.3, 0.7]),  # often international
    })

    # Combine
    X = pd.concat([legit_data, fraud_data], ignore_index=True)
    y = pd.Series([0]*n_legit + [1]*n_fraud)

    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)

    return X, y


# ── Training Script ────────────────────────────────────────────────────────────

def train_model():
    """
    Main training script — called by CI/CD pipeline.

    This is what GitHub Actions will run whenever:
    - New code is pushed
    - New training data is added
    - Manual trigger
    """
    print("=" * 60)
    print("  FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    # Generate data (in production this would load from database)
    print("\n[1/4] Loading data...")
    X, y = generate_fraud_data(n_samples=10000, fraud_rate=0.02)
    print(f"  Loaded {len(X)} transactions ({y.sum()} fraudulent, {(y==0).sum()} legit)")

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Train
    print("\n[2/4] Training model...")
    model = FraudDetectionModel(n_estimators=100, max_depth=10)
    val_metrics = model.train(X_train, y_train, X_val, y_val)

    # Test set evaluation (final check before deployment)
    print("\n[3/4] Evaluating on test set...")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    test_metrics = {
        "test_accuracy":  accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall":    recall_score(y_test, y_test_pred),
        "test_f1":        f1_score(y_test, y_test_pred),
        "test_roc_auc":   roc_auc_score(y_test, y_test_prob),
    }

    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Save for deployment
    print("\n[4/4] Saving model for deployment...")
    model.save("model/fraud_model.pkl")

    # Production readiness check
    meets_threshold = test_metrics["test_f1"] >= 0.7 and test_metrics["test_roc_auc"] >= 0.85
    if meets_threshold:
        print("\n✅ Model meets production thresholds! Ready for deployment.")
    else:
        print("\n⚠️  Model below production thresholds. Review before deploying.")

    return test_metrics


if __name__ == "__main__":
    train_model()
