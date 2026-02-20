"""
Centralized Configuration — no more hardcoded values scattered everywhere.

Uses pydantic-settings for:
    - Environment variable loading
    - .env file support
    - Validation and type safety
    - Default values with overrides

Usage:
    from config.settings import settings
    print(settings.model_path)
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application configuration — loaded from environment variables.

    Environment variables override defaults. Prefix: none (direct names).
    Example: LOG_LEVEL=DEBUG python api/serve.py
    """

    # ── Application ────────────────────────────────────────────
    app_name: str = "Fraud Detection API"
    environment: str = Field(default="development", description="development | staging | production")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # ── API ────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: Optional[str] = Field(default=None, description="API key for authentication (None = disabled)")
    cors_origins: list[str] = ["*"]
    rate_limit_per_minute: int = Field(default=60, description="Max requests per minute per client")

    # ── Model ──────────────────────────────────────────────────
    model_path: str = "model/fraud_model.pkl"
    model_version: str = "1.0.0"
    prediction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── A/B Testing ────────────────────────────────────────────
    ab_test_enabled: bool = True
    ab_test_traffic_split: float = Field(default=0.5, ge=0.0, le=1.0, description="Fraction of traffic to model B")

    # ── MLflow ─────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="mlruns", description="MLflow tracking URI")
    mlflow_experiment_name: str = "fraud-detection"

    # ── Training ───────────────────────────────────────────────
    train_samples: int = 10000
    fraud_rate: float = 0.02
    test_size: float = 0.3
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 10

    # ── Performance Thresholds ─────────────────────────────────
    threshold_f1: float = Field(default=0.70, description="Minimum F1 score for deployment")
    threshold_roc_auc: float = Field(default=0.85, description="Minimum ROC AUC for deployment")

    # ── Monitoring ─────────────────────────────────────────────
    prometheus_enabled: bool = True
    drift_detection_enabled: bool = True
    drift_window_size: int = Field(default=1000, description="Rolling window for drift detection")
    drift_threshold: float = Field(default=0.05, description="P-value threshold for drift alerts")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — loaded once, reused everywhere."""
    return Settings()


# Convenience alias
settings = get_settings()
