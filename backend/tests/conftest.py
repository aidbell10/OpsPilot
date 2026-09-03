"""Shared test fixtures.

The default environment for tests uses the deterministic fake providers so the
whole suite runs with no network, no model downloads, and no API keys.
Integration tests opt into a real PostgreSQL via the ``postgres_url`` fixture.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Set BEFORE opspilot.config is imported anywhere.
os.environ.setdefault("OPSPILOT_APP_ENV", "ci")
os.environ.setdefault("OPSPILOT_LLM_PROVIDER", "fake")
os.environ.setdefault("OPSPILOT_EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("OPSPILOT_LOG_LEVEL", "WARNING")


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    """Clear cached settings / engine / providers around every test."""
    from opspilot.config import get_settings
    from opspilot.db.session import reset_engine_cache
    from opspilot.providers.factory import reset_provider_cache

    get_settings.cache_clear()
    reset_provider_cache()
    reset_engine_cache()
    yield
    get_settings.cache_clear()
    reset_provider_cache()
    reset_engine_cache()


@pytest.fixture
def set_env(monkeypatch: pytest.MonkeyPatch):
    """Helper to set OPSPILOT_* env vars and drop the settings cache."""
    from opspilot.config import get_settings

    def _set(**values: str) -> None:
        for key, value in values.items():
            monkeypatch.setenv(f"OPSPILOT_{key.upper()}", value)
        get_settings.cache_clear()

    return _set
