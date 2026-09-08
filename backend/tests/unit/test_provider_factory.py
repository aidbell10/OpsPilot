from __future__ import annotations

from collections.abc import Callable

import pytest

from opspilot.providers.factory import get_rerank_provider

pytestmark = pytest.mark.unit


def test_reranker_is_none_by_default() -> None:
    assert get_rerank_provider() is None


def test_fake_reranker_selected_by_env(set_env: Callable[..., None]) -> None:
    set_env(reranker="fake")
    get_rerank_provider.cache_clear()
    provider = get_rerank_provider()
    assert provider is not None
    assert provider.model == "fake-rerank-v1"


def test_cross_encoder_selection_is_lazy(set_env: Callable[..., None]) -> None:
    # building the provider must not import torch / download a model — that is
    # deferred to the first rerank() call.
    set_env(reranker="cross_encoder")
    get_rerank_provider.cache_clear()
    provider = get_rerank_provider()
    assert provider is not None
    assert provider.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
