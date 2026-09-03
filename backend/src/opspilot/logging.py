"""Structured logging configuration (structlog).

JSON logs in non-local environments, colourised key/value logs locally. A
``request_id`` contextvar is bound by the API middleware so every log line
emitted while handling a request carries it.
"""

from __future__ import annotations

import logging
import sys

import structlog

from opspilot.config import Settings

_configured = False


def configure_logging(settings: Settings) -> None:
    """Idempotently configure stdlib logging + structlog."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer()
        if settings.app_env == "local"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
