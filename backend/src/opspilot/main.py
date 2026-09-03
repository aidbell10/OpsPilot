"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from opspilot import __version__
from opspilot.api.middleware import RequestContextMiddleware
from opspilot.api.routes import (
    feedback_router,
    health_router,
    incidents_router,
    search_router,
)
from opspilot.config import Settings, get_settings
from opspilot.logging import configure_logging, get_logger


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    log = get_logger("opspilot.startup")
    log.info(
        "app.starting",
        env=settings.app_env,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        embedding_dim=settings.embedding_dim,
    )
    yield
    log.info("app.stopping")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="OpsPilot AI",
        version=__version__,
        summary="Agentic Incident Investigation & RAG Evaluation Platform",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health_router)
    app.include_router(incidents_router)
    app.include_router(search_router)
    app.include_router(feedback_router)

    return app


app = create_app()
