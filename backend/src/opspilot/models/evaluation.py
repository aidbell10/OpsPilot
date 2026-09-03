"""Evaluation dataset, runs, and per-case results.

Phase 4 is where these get exercised. The schema is created now so migrations
stay linear and the evaluation harness has stable storage from the start.
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from opspilot.db.base import Base, JsonDict, TimestampMixin, UUIDPrimaryKeyMixin
from opspilot.db.types import str_enum
from opspilot.models.enums import Difficulty, EvalSplit, RetrievalStrategy


class EvaluationCase(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_cases"

    case_id: Mapped[str] = mapped_column(String(64), index=True)  # stable human id
    dataset_version: Mapped[str] = mapped_column(String(32), index=True)
    split: Mapped[EvalSplit] = mapped_column(str_enum(EvalSplit))

    query: Mapped[str] = mapped_column(Text)
    answerable: Mapped[bool] = mapped_column(Boolean)
    service_name: Mapped[str | None] = mapped_column(String(100), default=None)

    required_document_ids: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    required_evidence_ids: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    expected_root_cause: Mapped[str | None] = mapped_column(Text, default=None)
    acceptable_actions: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    forbidden_claims: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    difficulty: Mapped[Difficulty] = mapped_column(str_enum(Difficulty))
    category: Mapped[str] = mapped_column(String(50), default="general")

    __table_args__ = (
        UniqueConstraint("dataset_version", "case_id", name="uq_evaluation_cases_version_case"),
    )


class EvaluationRun(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "evaluation_runs"

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC)
    )
    git_sha: Mapped[str | None] = mapped_column(String(40), default=None)
    dataset_version: Mapped[str] = mapped_column(String(32), index=True)
    split: Mapped[EvalSplit] = mapped_column(str_enum(EvalSplit))

    llm_model: Mapped[str] = mapped_column(String(100))
    embedding_model: Mapped[str] = mapped_column(String(100))
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer)
    top_k: Mapped[int] = mapped_column(Integer)
    retrieval_strategy: Mapped[RetrievalStrategy] = mapped_column(str_enum(RetrievalStrategy))
    reranker: Mapped[str | None] = mapped_column(String(100), default=None)
    prompt_version: Mapped[str] = mapped_column(String(20))

    notes: Mapped[str] = mapped_column(Text, default="")
    aggregate_metrics: Mapped[JsonDict] = mapped_column(default=dict)

    results: Mapped[list[EvaluationResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class EvaluationResult(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "evaluation_results"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(String(64), index=True)

    retrieval_metrics: Mapped[JsonDict] = mapped_column(default=dict)
    generation_metrics: Mapped[JsonDict] = mapped_column(default=dict)
    latency_ms: Mapped[JsonDict] = mapped_column(default=dict)
    cost: Mapped[JsonDict] = mapped_column(default=dict)
    raw_output: Mapped[JsonDict] = mapped_column(default=dict)

    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    run: Mapped[EvaluationRun] = relationship(back_populates="results")

    __table_args__ = (Index("uq_evaluation_results_run_case", "run_id", "case_id", unique=True),)
