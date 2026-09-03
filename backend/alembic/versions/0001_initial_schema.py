"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-02

Creates the pgvector / pg_trgm extensions and the full OpsPilot data model:
services, documents, document_chunks (with vector + generated tsvector columns),
deployments, incidents, historical_incidents, and the evaluation +
user_feedback tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from opspilot.db.types import str_enum
from opspilot.models.document import EMBEDDING_DIM
from opspilot.models.enums import (
    DeploymentStatus,
    Difficulty,
    DocumentType,
    Environment,
    EvalSplit,
    IncidentStatus,
    RetrievalStrategy,
    Severity,
)

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = sa.Uuid(as_uuid=True)
_TS = sa.DateTime(timezone=True)
_JSONB = postgresql.JSONB
_ARRAY_STR = postgresql.ARRAY(sa.String())


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # --- services ---------------------------------------------------------
    op.create_table(
        "services",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("repo_url", sa.String(300), nullable=True),
        sa.Column("tier", sa.Integer(), server_default="2", nullable=False),
        sa.Column("owning_team", sa.String(100), server_default="platform", nullable=False),
    )
    op.create_index("ix_services_name", "services", ["name"], unique=True)

    # --- documents -------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("document_type", str_enum(DocumentType), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "service_id",
            _UUID,
            sa.ForeignKey("services.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("service_name", sa.String(100), nullable=True),
        sa.Column("version", sa.String(50), nullable=True),
        sa.Column("environment", str_enum(Environment), nullable=True),
        sa.Column("source_path", sa.String(400), nullable=True),
        sa.Column("doc_timestamp", _TS, nullable=True),
        sa.Column("meta", _JSONB, server_default="{}", nullable=False),
    )
    op.create_index("ix_documents_document_type", "documents", ["document_type"])
    op.create_index("ix_documents_service_id", "documents", ["service_id"])
    op.create_index("ix_documents_service_name", "documents", ["service_name"])

    # --- document_chunks ----------------------------------------------------
    op.create_table(
        "document_chunks",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column(
            "document_id",
            _UUID,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("document_type", str_enum(DocumentType), nullable=False),
        sa.Column("service_name", sa.String(100), nullable=True),
        sa.Column("version", sa.String(50), nullable=True),
        sa.Column("environment", str_enum(Environment), nullable=True),
        sa.Column("chunk_timestamp", _TS, nullable=True),
        sa.Column("meta", _JSONB, server_default="{}", nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_document_type", "document_chunks", ["document_type"])
    op.create_index("ix_document_chunks_service_name", "document_chunks", ["service_name"])
    op.create_index(
        "uq_document_chunks_document_id_chunk_index",
        "document_chunks",
        ["document_id", "chunk_index"],
        unique=True,
    )
    op.create_index(
        "ix_document_chunks_service_name_document_type",
        "document_chunks",
        ["service_name", "document_type"],
    )
    op.create_index(
        "ix_document_chunks_content_tsv",
        "document_chunks",
        ["content_tsv"],
        postgresql_using="gin",
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    # --- deployments ------------------------------------------------------
    op.create_table(
        "deployments",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column(
            "service_id",
            _UUID,
            sa.ForeignKey("services.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("service_name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column(
            "environment",
            str_enum(Environment),
            server_default=Environment.PRODUCTION.value,
            nullable=False,
        ),
        sa.Column("deployed_at", _TS, nullable=False),
        sa.Column(
            "status",
            str_enum(DeploymentStatus),
            server_default=DeploymentStatus.SUCCEEDED.value,
            nullable=False,
        ),
        sa.Column("change_summary", sa.Text(), server_default="", nullable=False),
        sa.Column("changed_components", _JSONB, server_default="{}", nullable=False),
    )
    op.create_index("ix_deployments_service_name", "deployments", ["service_name"])
    op.create_index("ix_deployments_version", "deployments", ["version"])
    op.create_index(
        "uq_deployments_service_name_version",
        "deployments",
        ["service_name", "version"],
        unique=True,
    )

    # --- incidents -------------------------------------------------------
    op.create_table(
        "incidents",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "service_id",
            _UUID,
            sa.ForeignKey("services.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("reported_service", sa.String(100), nullable=True),
        sa.Column("reported_version", sa.String(50), nullable=True),
        sa.Column("environment", str_enum(Environment), nullable=True),
        sa.Column(
            "status",
            str_enum(IncidentStatus),
            server_default=IncidentStatus.OPEN.value,
            nullable=False,
        ),
        sa.Column("analysis", _JSONB, nullable=True),
    )

    # --- historical_incidents -------------------------------------------
    op.create_table(
        "historical_incidents",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("service_name", sa.String(100), nullable=False),
        sa.Column("occurred_at", _TS, nullable=False),
        sa.Column(
            "severity",
            str_enum(Severity),
            server_default=Severity.SEV3.value,
            nullable=False,
        ),
        sa.Column("symptoms", sa.Text(), nullable=False),
        sa.Column("root_cause", sa.Text(), nullable=False),
        sa.Column("resolution", sa.Text(), nullable=False),
        sa.Column("related_deployment_version", sa.String(50), nullable=True),
        sa.Column(
            "linked_document_id",
            _UUID,
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("meta", _JSONB, server_default="{}", nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
    )
    op.create_index(
        "ix_historical_incidents_service_name", "historical_incidents", ["service_name"]
    )

    # --- evaluation_cases ----------------------------------------------
    op.create_table(
        "evaluation_cases",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("case_id", sa.String(64), nullable=False),
        sa.Column("dataset_version", sa.String(32), nullable=False),
        sa.Column("split", str_enum(EvalSplit), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False),
        sa.Column("service_name", sa.String(100), nullable=True),
        sa.Column("required_document_ids", _ARRAY_STR, server_default="{}", nullable=False),
        sa.Column("required_evidence_ids", _ARRAY_STR, server_default="{}", nullable=False),
        sa.Column("expected_root_cause", sa.Text(), nullable=True),
        sa.Column("acceptable_actions", _ARRAY_STR, server_default="{}", nullable=False),
        sa.Column("forbidden_claims", _ARRAY_STR, server_default="{}", nullable=False),
        sa.Column("difficulty", str_enum(Difficulty), nullable=False),
        sa.Column("category", sa.String(50), server_default="general", nullable=False),
        sa.UniqueConstraint("dataset_version", "case_id", name="uq_evaluation_cases_version_case"),
    )
    op.create_index("ix_evaluation_cases_case_id", "evaluation_cases", ["case_id"])
    op.create_index("ix_evaluation_cases_dataset_version", "evaluation_cases", ["dataset_version"])

    # --- evaluation_runs ----------------------------------------------
    op.create_table(
        "evaluation_runs",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("created_at", _TS, nullable=False),
        sa.Column("git_sha", sa.String(40), nullable=True),
        sa.Column("dataset_version", sa.String(32), nullable=False),
        sa.Column("split", str_enum(EvalSplit), nullable=False),
        sa.Column("llm_model", sa.String(100), nullable=False),
        sa.Column("embedding_model", sa.String(100), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("retrieval_strategy", str_enum(RetrievalStrategy), nullable=False),
        sa.Column("reranker", sa.String(100), nullable=True),
        sa.Column("prompt_version", sa.String(20), nullable=False),
        sa.Column("notes", sa.Text(), server_default="", nullable=False),
        sa.Column("aggregate_metrics", _JSONB, server_default="{}", nullable=False),
    )
    op.create_index("ix_evaluation_runs_dataset_version", "evaluation_runs", ["dataset_version"])

    # --- evaluation_results ------------------------------------------
    op.create_table(
        "evaluation_results",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column(
            "run_id",
            _UUID,
            sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("case_id", sa.String(64), nullable=False),
        sa.Column("retrieval_metrics", _JSONB, server_default="{}", nullable=False),
        sa.Column("generation_metrics", _JSONB, server_default="{}", nullable=False),
        sa.Column("latency_ms", _JSONB, server_default="{}", nullable=False),
        sa.Column("cost", _JSONB, server_default="{}", nullable=False),
        sa.Column("raw_output", _JSONB, server_default="{}", nullable=False),
        sa.Column("passed", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("total_latency_ms", sa.Float(), server_default="0", nullable=False),
    )
    op.create_index("ix_evaluation_results_run_id", "evaluation_results", ["run_id"])
    op.create_index("ix_evaluation_results_case_id", "evaluation_results", ["case_id"])
    op.create_index(
        "uq_evaluation_results_run_case",
        "evaluation_results",
        ["run_id", "case_id"],
        unique=True,
    )

    # --- user_feedback ---------------------------------------------
    op.create_table(
        "user_feedback",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column(
            "incident_id",
            _UUID,
            sa.ForeignKey("incidents.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("case_id", sa.String(64), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("helpful", sa.Boolean(), nullable=True),
        sa.Column("comment", sa.Text(), server_default="", nullable=False),
    )
    op.create_index("ix_user_feedback_incident_id", "user_feedback", ["incident_id"])


def downgrade() -> None:
    for table in (
        "user_feedback",
        "evaluation_results",
        "evaluation_runs",
        "evaluation_cases",
        "historical_incidents",
        "incidents",
        "deployments",
        "document_chunks",
        "documents",
        "services",
    ):
        op.drop_table(table)
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS vector")
