"""
Structured Logging — production-grade logging with JSON output.

Why structured logging?
    - Machine-parseable (ELK stack, CloudWatch, Datadog)
    - Correlation IDs for request tracing
    - Consistent format across all modules
    - Context-rich (model version, latency, user_id)

Usage:
    from config.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Prediction made", extra={"user_id": "123", "latency_ms": 5.2})
"""
import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any
from config.settings import settings


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for production environments.

    Output format:
    {"timestamp": "...", "level": "INFO", "module": "api.serve", "message": "...", "extra": {...}}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields (user_id, latency, model_version, etc.)
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "relativeCreated",
                "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "filename", "module", "pathname", "thread", "threadName",
                "process", "processName", "levelname", "levelno",
                "message", "msecs", "taskName",
            ):
                log_data[key] = value

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{color}{timestamp} [{record.levelname:8s}]{self.RESET} {record.name}: {record.getMessage()}"


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    In production: JSON format (machine-parseable)
    In development: Pretty format (human-readable)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        if settings.environment == "production":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(PrettyFormatter())

        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    return logger
