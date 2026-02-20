"""
Load Testing — validates API can handle production traffic.

Uses Locust to simulate realistic traffic patterns.

Run headless:  make load-test
Run with UI:   make load-test-ui
                → Open http://localhost:8089

Key metrics to watch:
    - P95 latency < 100ms
    - Error rate < 0.1%
    - Throughput > 100 RPS
"""
from locust import HttpUser, task, between
import random


class FraudDetectionUser(HttpUser):
    """
    Simulates a client making fraud detection requests.

    Mix of request patterns:
        - 80% normal predictions
        - 10% health checks
        - 5% A/B stats checks
        - 5% suspicious transactions (high fraud probability)
    """
    wait_time = between(0.1, 0.5)  # 2-10 requests per second per user

    def on_start(self):
        """Generate a consistent user_id for this simulated user."""
        self.user_id = f"load_test_user_{random.randint(1, 10000)}"

    @task(80)
    def predict_normal(self):
        """Normal transaction — should predict legit."""
        self.client.post("/predict", json={
            "user_id": self.user_id,
            "amount": round(random.uniform(10, 200), 2),
            "hour": random.randint(8, 20),
            "merchant_risk": round(random.uniform(0.0, 0.3), 2),
            "card_age_days": random.randint(90, 1500),
            "distance_km": round(random.uniform(0, 20), 1),
            "num_recent_txns": random.randint(0, 5),
            "is_international": 0,
        })

    @task(5)
    def predict_suspicious(self):
        """Suspicious transaction — should predict fraud."""
        self.client.post("/predict", json={
            "user_id": self.user_id,
            "amount": round(random.uniform(500, 5000), 2),
            "hour": random.choice([0, 1, 2, 3, 23]),
            "merchant_risk": round(random.uniform(0.7, 1.0), 2),
            "card_age_days": random.randint(1, 30),
            "distance_km": round(random.uniform(200, 1000), 1),
            "num_recent_txns": random.randint(8, 20),
            "is_international": 1,
        })

    @task(10)
    def health_check(self):
        """Health check — should always return 200."""
        self.client.get("/health")

    @task(5)
    def ab_stats(self):
        """A/B test stats — lightweight read."""
        self.client.get("/ab-stats")
