"""Application configuration.

All runtime configuration is sourced from environment variables (prefixed
``OPSPILOT_``) or a local ``.env`` file. Nothing here reaches out to a network
or a database at import time — :func:`get_settings` is a pure, cached accessor.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PG_DSN = TypeAdapter(PostgresDsn)

LLMProviderName = Literal["fake", "anthropic"]
EmbeddingProviderName = Literal["fake", "local"]
RetrievalStrategyName = Literal["vector", "lexical", "hybrid"]
AppEnv = Literal["local", "ci", "staging", "prod"]


class Settings(BaseSettings):
    """Typed, validated view of the process environment."""

    model_config = SettingsConfigDict(
        env_prefix="OPSPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Application ---
    app_env: AppEnv = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Database ---
    # Stored as a plain string (SQLAlchemy wants one) but validated as a real
    # PostgreSQL DSN on load.
    database_url: str = "postgresql+psycopg://opspilot:opspilot@localhost:5432/opspilot"

    @field_validator("database_url")
    @classmethod
    def _validate_dsn(cls, v: str) -> str:
        _PG_DSN.validate_python(v)
        return v

    # --- LLM provider (generation) ---
    llm_provider: LLMProviderName = "fake"
    llm_model: str = "claude-sonnet-5"
    anthropic_api_key: str | None = None
    llm_max_tokens: int = Field(default=1024, ge=1, le=8192)

    # --- Embedding provider (semantic retrieval) ---
    embedding_provider: EmbeddingProviderName = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, ge=8, le=4096)

    # --- Retrieval / chunking ---
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)
    retrieval_top_k: int = Field(default=8, ge=1, le=100)
    # Phase 5: which retrieval arm(s) serve a query. `hybrid` = vector + lexical
    # fused with RRF; the evaluation runner can override this per run to compare.
    retrieval_strategy: RetrievalStrategyName = "hybrid"
    # Candidates pulled from each arm before RRF fusion (hybrid only).
    retrieval_candidate_k: int = Field(default=30, ge=1, le=200)
    # RRF smoothing constant (Cormack et al. 2009 use 60).
    rrf_k: int = Field(default=60, ge=1, le=1000)

    # --- Prompting ---
    prompt_version: str = "v1"

    def model_post_init(self, _context: object) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "OPSPILOT_ANTHROPIC_API_KEY is required when OPSPILOT_LLM_PROVIDER=anthropic"
            )

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy-ready URL string."""
        return str(self.database_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that repeated FastAPI dependency resolution is free. Call
    ``get_settings.cache_clear()`` in tests that manipulate the environment.
    """
    return Settings()
