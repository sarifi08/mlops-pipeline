# Fraud Detection API - Production Docker Image
# Multi-stage build for smaller image size

# ── Stage 1: Build dependencies ───────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install dependencies in a virtual env for clean copy
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production image ─────────────────────────────────
FROM python:3.10-slim

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config/ ./config/
COPY api/ ./api/
COPY model/ ./model/
COPY monitoring/ ./monitoring/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run API server with multiple workers for production
CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
