"""Provider abstraction for embeddings and LLM generation.

Callers depend only on the :class:`EmbeddingProvider` / :class:`LLMProvider`
protocols in :mod:`opspilot.providers.base`. Concrete implementations
(``fake``, ``local``, ``anthropic``) are selected via
:func:`opspilot.providers.factory.get_embedding_provider` /
:func:`opspilot.providers.factory.get_llm_provider`.
"""

from opspilot.providers.base import (
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    LLMResult,
)
from opspilot.providers.factory import get_embedding_provider, get_llm_provider

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "LLMProvider",
    "LLMResult",
    "get_embedding_provider",
    "get_llm_provider",
]
