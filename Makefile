# MLOps Pipeline — Makefile
# One command to do anything. No more scrolling through READMEs.
#
# Usage:
#   make help        # show all commands
#   make setup       # first-time setup
#   make train       # train the model
#   make serve       # run the API
#   make up          # start full stack (API + Prometheus + Grafana + MLflow)
#   make test        # run all tests
#   make lint        # check code quality

.PHONY: help setup train serve test lint format up down logs clean check load-test

# Default: show help
help: ## Show this help message
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║            MLOps Pipeline — Command Reference           ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ──────────────────────────────────────────────────────
setup: ## First-time project setup
	@echo "🔧 Setting up MLOps Pipeline..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
	cp -n .env.example .env 2>/dev/null || true
	@echo "✅ Setup complete! Run 'make train' to train the model."

install: ## Install production dependencies only
	pip install -r requirements.txt

install-dev: ## Install all dependencies (production + development)
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# ── Training ───────────────────────────────────────────────────
train: ## Train the fraud detection model
	@echo "🧠 Training model..."
	python model/train.py

# ── API ────────────────────────────────────────────────────────
serve: ## Run the API locally (with auto-reload)
	@echo "🚀 Starting API server..."
	uvicorn api.serve:app --reload --host 0.0.0.0 --port 8000

serve-prod: ## Run the API in production mode
	uvicorn api.serve:app --host 0.0.0.0 --port 8000 --workers 4

# ── Docker Stack ───────────────────────────────────────────────
up: ## Start full stack (API + Prometheus + Grafana + MLflow)
	@echo "🐳 Starting full MLOps stack..."
	docker compose up -d --build
	@echo ""
	@echo "Services running:"
	@echo "  API:        http://localhost:8000"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  MLflow:     http://localhost:5000"

down: ## Stop all services
	docker compose down

logs: ## Tail logs from all services
	docker compose logs -f

restart: ## Restart all services
	docker compose restart

# ── Testing ────────────────────────────────────────────────────
test: ## Run all tests with coverage
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v

check: ## Run performance threshold checks
	@echo "📊 Checking model performance..."
	python tests/check_performance.py

# ── Code Quality ───────────────────────────────────────────────
lint: ## Run all linters
	@echo "🔍 Running linters..."
	ruff check .
	mypy model/ api/ config/ --ignore-missing-imports

format: ## Auto-format code
	@echo "✨ Formatting code..."
	ruff format .
	ruff check --fix .

# ── Load Testing ───────────────────────────────────────────────
load-test: ## Run load tests with Locust (headless, 60s)
	@echo "⚡ Running load tests..."
	locust -f tests/load_test.py --headless -u 50 -r 10 --run-time 60s --host http://localhost:8000

load-test-ui: ## Run load tests with Locust web UI
	@echo "📊 Starting Locust UI at http://localhost:8089..."
	locust -f tests/load_test.py --host http://localhost:8000

# ── Drift Detection ───────────────────────────────────────────
drift-check: ## Run data drift detection
	@echo "📉 Checking for data drift..."
	python -c "from monitoring.drift import DriftDetector; from model.train import generate_fraud_data; \
		d = DriftDetector(); X, _ = generate_fraud_data(10000); d.set_reference(X); \
		X2, _ = generate_fraud_data(1000); report = d.check_drift(X2); \
		print('Drift detected:', report.drift_detected); \
		[print(a) for a in report.alerts] if report.alerts else print('✅ No drift detected')"

# ── Cleanup ────────────────────────────────────────────────────
clean: ## Remove generated files and caches
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "✅ Clean"

clean-all: clean ## Remove everything including model artifacts
	rm -f model/fraud_model.pkl
	rm -rf mlruns/
	docker compose down -v 2>/dev/null || true
	@echo "✅ Deep clean complete"
