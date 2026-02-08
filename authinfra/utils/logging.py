"""
Structured logging for AuthInfra.

Prefer JSON output for machine-readable logs. Falls back to standard key=value
or message-only if JSON is not available or disabled.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Extra fields set by caller (e.g. logger.info("msg", extra={"key": "val"}))
        for k, v in record.__dict__.items():
            if k not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            ):
                payload[k] = v
        return json.dumps(payload, default=str)


def configure_logging(
    level: int | str = logging.INFO,
    json_logs: bool = True,
    stream: Any = None,
) -> None:
    """
    Configure root logger for authinfra.
    Default: JSON to stderr. Set json_logs=False for human-readable output.
    """
    stream = stream or sys.stderr
    root = logging.getLogger("authinfra")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(stream)
        if json_logs:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under authinfra for the given name."""
    if name.startswith("authinfra"):
        return logging.getLogger(name)
    return logging.getLogger(f"authinfra.{name}")
