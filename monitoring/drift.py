"""
Data Drift Detection — catches when production data diverges from training data.

Why drift detection matters:
    - Model trained on distribution X, but production sees distribution Y
    - Performance degrades silently without retraining
    - Common causes: seasonality, fraud pattern evolution, data pipeline bugs

Methods implemented:
    1. Population Stability Index (PSI) — distribution shift detection
    2. Kolmogorov-Smirnov test — per-feature statistical test
    3. Feature statistics monitoring — mean/std/min/max tracking

Usage:
    from monitoring.drift import DriftDetector

    detector = DriftDetector()
    detector.set_reference(training_data)
    report = detector.check_drift(production_batch)
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftReport:
    """Results from a drift detection check."""
    timestamp: str
    drift_detected: bool
    overall_drift_score: float
    feature_reports: Dict[str, Dict[str, Any]]
    psi_scores: Dict[str, float]
    ks_test_results: Dict[str, Dict[str, float]]
    alerts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "drift_detected": self.drift_detected,
            "overall_drift_score": round(self.overall_drift_score, 4),
            "feature_reports": self.feature_reports,
            "psi_scores": {k: round(v, 4) for k, v in self.psi_scores.items()},
            "ks_test_results": self.ks_test_results,
            "alerts": self.alerts,
            "num_features_drifted": sum(
                1 for v in self.psi_scores.values() if v > 0.2
            ),
        }


class DriftDetector:
    """
    Statistical drift detection for production ML systems.

    Compares incoming production data against a reference distribution
    (typically the training data) to detect distribution shifts.
    """

    # PSI thresholds (industry standard)
    PSI_NO_DRIFT = 0.1        # < 0.1: no significant drift
    PSI_MODERATE_DRIFT = 0.2  # 0.1-0.2: moderate drift, investigate
    PSI_SIGNIFICANT_DRIFT = 0.25  # > 0.25: significant drift, retrain

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        self.production_buffer: List[Dict] = []
        self.drift_history: List[DriftReport] = []

    def set_reference(self, data: pd.DataFrame) -> None:
        """
        Set the reference distribution (typically training data).
        All future drift checks compare against this.
        """
        self.reference_data = data.copy()
        self.reference_stats = {}

        for col in data.columns:
            self.reference_stats[col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "median": float(data[col].median()),
                "q25": float(data[col].quantile(0.25)),
                "q75": float(data[col].quantile(0.75)),
            }

        logger.info(
            "Reference distribution set",
            extra={"n_samples": len(data), "n_features": len(data.columns)}
        )

    def add_production_sample(self, sample: Dict[str, float]) -> None:
        """Add a single production observation to the buffer."""
        self.production_buffer.append(sample)

    def calculate_psi(
        self, reference: np.ndarray, production: np.ndarray
    ) -> float:
        """
        Population Stability Index (PSI).

        Measures how much the production distribution has shifted
        from the reference distribution.

        PSI = Σ (P_i - Q_i) * ln(P_i / Q_i)

        Interpretation:
            < 0.1:  No significant shift
            0.1-0.2: Moderate shift — monitor
            > 0.25:  Significant shift — retrain

        Used by banks, insurance companies, and fraud detection systems.
        """
        # Create bins from reference data
        bins = np.linspace(
            min(reference.min(), production.min()) - 1e-6,
            max(reference.max(), production.max()) + 1e-6,
            self.n_bins + 1,
        )

        # Calculate proportions in each bin
        ref_counts, _ = np.histogram(reference, bins=bins)
        prod_counts, _ = np.histogram(production, bins=bins)

        # Convert to proportions (add small epsilon to avoid division by zero)
        eps = 1e-6
        ref_proportions = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
        prod_proportions = (prod_counts + eps) / (prod_counts.sum() + eps * len(prod_counts))

        # PSI formula
        psi = np.sum(
            (prod_proportions - ref_proportions)
            * np.log(prod_proportions / ref_proportions)
        )

        return float(psi)

    def ks_test(
        self, reference: np.ndarray, production: np.ndarray
    ) -> Dict[str, float]:
        """
        Kolmogorov-Smirnov test — detects if two samples come from
        the same distribution.

        Returns:
            statistic: max distance between CDFs (0-1)
            p_value: probability that distributions are the same
                     < 0.05 = likely different distributions
        """
        statistic, p_value = stats.ks_2samp(reference, production)
        return {
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": p_value < 0.05,
        }

    def check_drift(
        self, production_data: Optional[pd.DataFrame] = None
    ) -> DriftReport:
        """
        Run full drift detection on production data vs reference.

        Args:
            production_data: DataFrame to check. If None, uses buffered samples.

        Returns:
            DriftReport with per-feature analysis and alerts.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        # Use provided data or buffer
        if production_data is not None:
            prod_df = production_data
        elif self.production_buffer:
            prod_df = pd.DataFrame(self.production_buffer)
            self.production_buffer = []  # clear buffer
        else:
            raise ValueError("No production data to check. Provide data or add samples.")

        # Ensure same columns
        common_cols = [c for c in self.reference_data.columns if c in prod_df.columns]

        psi_scores = {}
        ks_results = {}
        feature_reports = {}
        alerts = []

        for col in common_cols:
            ref_values = self.reference_data[col].values.astype(float)
            prod_values = prod_df[col].values.astype(float)

            # PSI
            psi = self.calculate_psi(ref_values, prod_values)
            psi_scores[col] = psi

            # KS test
            ks = self.ks_test(ref_values, prod_values)
            ks_results[col] = ks

            # Feature statistics comparison
            ref_stats = self.reference_stats[col]
            prod_mean = float(prod_values.mean())
            prod_std = float(prod_values.std())

            feature_reports[col] = {
                "reference_mean": ref_stats["mean"],
                "production_mean": round(prod_mean, 4),
                "mean_shift": round(abs(prod_mean - ref_stats["mean"]), 4),
                "reference_std": round(ref_stats["std"], 4),
                "production_std": round(prod_std, 4),
                "psi": round(psi, 4),
                "psi_status": self._psi_status(psi),
                "ks_p_value": ks["p_value"],
            }

            # Generate alerts
            if psi > self.PSI_SIGNIFICANT_DRIFT:
                alerts.append(
                    f"🚨 SIGNIFICANT DRIFT in '{col}': PSI={psi:.3f} "
                    f"(threshold={self.PSI_SIGNIFICANT_DRIFT})"
                )
            elif psi > self.PSI_MODERATE_DRIFT:
                alerts.append(
                    f"⚠️  MODERATE DRIFT in '{col}': PSI={psi:.3f} "
                    f"(threshold={self.PSI_MODERATE_DRIFT})"
                )

        # Overall drift score (average PSI)
        overall_score = np.mean(list(psi_scores.values())) if psi_scores else 0.0
        drift_detected = any(psi > self.PSI_MODERATE_DRIFT for psi in psi_scores.values())

        report = DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            drift_detected=drift_detected,
            overall_drift_score=overall_score,
            feature_reports=feature_reports,
            psi_scores=psi_scores,
            ks_test_results=ks_results,
            alerts=alerts,
        )

        self.drift_history.append(report)

        if drift_detected:
            logger.warning(
                "Data drift detected",
                extra={
                    "overall_score": round(overall_score, 4),
                    "drifted_features": [
                        k for k, v in psi_scores.items() if v > self.PSI_MODERATE_DRIFT
                    ],
                },
            )
        else:
            logger.info(
                "No significant drift detected",
                extra={"overall_score": round(overall_score, 4)},
            )

        return report

    def _psi_status(self, psi: float) -> str:
        """Human-readable PSI interpretation."""
        if psi < self.PSI_NO_DRIFT:
            return "✅ No drift"
        elif psi < self.PSI_MODERATE_DRIFT:
            return "⚠️ Minor shift"
        elif psi < self.PSI_SIGNIFICANT_DRIFT:
            return "⚠️ Moderate drift"
        else:
            return "🚨 Significant drift"

    def get_history(self) -> List[Dict[str, Any]]:
        """Return drift check history."""
        return [r.to_dict() for r in self.drift_history]
