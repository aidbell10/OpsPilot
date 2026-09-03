"""Integration test fixtures — a real PostgreSQL with pgvector.

Resolution order for the database:

1. ``OPSPILOT_TEST_DATABASE_URL`` env var (e.g. the running compose ``db``)
2. an ephemeral ``pgvector/pgvector`` container via testcontainers (needs Docker)
3. otherwise every integration test is skipped

Migrations are applied once per session with ``alembic upgrade head``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

_PGVECTOR_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    explicit = os.environ.get("OPSPILOT_TEST_DATABASE_URL")
    if explicit:
        yield explicit
        return

    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover
        pytest.skip("testcontainers not installed")

    try:
        container = PostgresContainer(_PGVECTOR_IMAGE, driver="psycopg")
        with container as pg:
            yield pg.get_connection_url()
    except Exception as exc:  # pragma: no cover - Docker missing / unusable
        pytest.skip(f"Docker not available for integration tests: {exc}")


@pytest.fixture(scope="session")
def migrated_engine(postgres_url: str) -> Iterator[Engine]:
    os.environ["OPSPILOT_DATABASE_URL"] = postgres_url

    from opspilot.config import get_settings

    get_settings.cache_clear()

    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", postgres_url)
    command.upgrade(cfg, "head")

    engine = create_engine(postgres_url, future=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(migrated_engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=migrated_engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def clean_db(migrated_engine: Engine) -> Iterator[None]:
    """Truncate all data tables before a test that needs a blank slate."""
    with migrated_engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE services, documents, document_chunks, deployments, incidents, "
                "historical_incidents, evaluation_cases, evaluation_runs, evaluation_results, "
                "user_feedback RESTART IDENTITY CASCADE"
            )
        )
    yield
