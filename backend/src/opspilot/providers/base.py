"""Protocols and result types for the provider abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vectors plus the token accounting needed for cost reporting."""

    vectors: list[list[float]]
    model: str
    total_tokens: int = 0

    @property
    def dim(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Raw text response plus token usage."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into fixed-dimension vectors."""

    name: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts. Order of ``vectors`` matches ``texts``."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generates a completion for a prompt.

    Structured-output handling (JSON schema / tool use) lives in the
    ``generation`` package, layered on top of this minimal contract, so that
    swapping providers never touches retrieval or orchestration code.
    """

    name: str
    model: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResult: ...
