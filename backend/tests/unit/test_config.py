from __future__ import annotations

from collections.abc import Callable

import pytest

from opspilot.config import Settings, get_settings


@pytest.mark.unit
def test_defaults_are_offline_safe() -> None:
    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_provider == "fake"
    assert s.embedding_provider in {"fake", "local"}
    assert s.embedding_dim == 384
    assert s.chunk_overlap < s.chunk_size


@pytest.mark.unit
def test_env_overrides(set_env: Callable[..., None]) -> None:
    set_env(chunk_size="256", chunk_overlap="32", retrieval_top_k="5")
    s = get_settings()
    assert s.chunk_size == 256
    assert s.chunk_overlap == 32
    assert s.retrieval_top_k == 5


@pytest.mark.unit
def test_overlap_must_be_smaller_than_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPSPILOT_CHUNK_SIZE", "128")
    monkeypatch.setenv("OPSPILOT_CHUNK_OVERLAP", "128")
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings()


@pytest.mark.unit
def test_anthropic_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPSPILOT_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("OPSPILOT_ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Settings()
