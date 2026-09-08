"""Provider selection based on :class:`opspilot.config.Settings`."""

from __future__ import annotations

from functools import lru_cache

from opspilot.config import RerankerName, Settings, get_settings
from opspilot.providers.base import EmbeddingProvider, LLMProvider, RerankProvider


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return _build_embedding_provider(get_settings())


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    return _build_llm_provider(get_settings())


@lru_cache(maxsize=1)
def get_rerank_provider() -> RerankProvider | None:
    """The configured reranker, or ``None`` when ``OPSPILOT_RERANKER=none``."""
    return _build_rerank_provider(get_settings())


def _build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "fake":
        from opspilot.providers.fake import FakeEmbeddingProvider

        return FakeEmbeddingProvider(model="fake-embed-v1", dim=settings.embedding_dim)

    from opspilot.providers.local_embeddings import LocalEmbeddingProvider

    return LocalEmbeddingProvider(model=settings.embedding_model, dim=settings.embedding_dim)


def _build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        from opspilot.providers.fake import FakeLLMProvider

        return FakeLLMProvider(model="fake-llm-v1")

    if not settings.anthropic_api_key:  # defensive; also checked in Settings
        raise RuntimeError("anthropic provider selected but no API key configured")

    from opspilot.providers.anthropic_llm import AnthropicLLMProvider

    return AnthropicLLMProvider(api_key=settings.anthropic_api_key, model=settings.llm_model)


def _build_rerank_provider(settings: Settings) -> RerankProvider | None:
    return build_rerank_provider(settings.reranker, settings)


def build_rerank_provider(reranker: RerankerName, settings: Settings) -> RerankProvider | None:
    """Build a reranker by name (lets the eval CLI override ``settings.reranker``)."""
    if reranker == "none":
        return None

    if reranker == "fake":
        from opspilot.providers.fake import FakeRerankProvider

        return FakeRerankProvider(model="fake-rerank-v1")

    from opspilot.providers.local_rerank import LocalCrossEncoderProvider

    return LocalCrossEncoderProvider(model=settings.reranker_model)


def reset_provider_cache() -> None:
    get_embedding_provider.cache_clear()
    get_llm_provider.cache_clear()
    get_rerank_provider.cache_clear()
