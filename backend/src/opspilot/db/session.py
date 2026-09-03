"""Engine and session lifecycle.

The engine is created lazily and cached so importing this module has no side
effects. ``get_db`` is the FastAPI dependency; ``session_scope`` is a plain
context manager for scripts, ingestion, and evaluation runs.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from opspilot.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.sync_database_url,
        pool_pre_ping=True,
        future=True,
        echo=False,
        # Fail fast: a readiness probe must not hang when the DB is unreachable.
        connect_args={"connect_timeout": 5},
    )


@lru_cache(maxsize=1)
def _get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session."""
    session = _get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = _get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Drop cached engine/sessionmaker — used by tests that switch databases."""
    get_engine.cache_clear()
    _get_sessionmaker.cache_clear()
