"""Request-scoped middleware: assign a request id and bind it to the logger."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from opspilot.logging import get_logger

_log = get_logger("opspilot.request")


class RequestContextMiddleware:
    """Pure-ASGI middleware (cheaper than BaseHTTPMiddleware)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        status_holder: dict[str, int] = {}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            _log.info(
                "request.completed",
                method=scope.get("method"),
                path=scope.get("path"),
                status=status_holder.get("status"),
                duration_ms=round(elapsed_ms, 2),
            )
            structlog.contextvars.clear_contextvars()
