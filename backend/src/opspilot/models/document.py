"""Knowledge-base documents and their embedded chunks."""

from __future__ import annotations

import datetime as dt
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from opspilot.config import get_settings
from opspilot.db.base import Base, JsonDict, TimestampMixin, UUIDPrimaryKeyMixin
from opspilot.db.types import str_enum
from opspilot.models.enums import DocumentType, Environment

# Static at process start. Changing the embedding model's dimensionality
# requires a new migration (the pgvector column type is fixed).
EMBEDDING_DIM: int = get_settings().embedding_dim


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    document_type: Mapped[DocumentType] = mapped_column(str_enum(DocumentType), index=True)
    title: Mapped[str] = mapped_column(String(300))
    content: Mapped[str] = mapped_column(Text)

    service_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("services.id", ondelete="SET NULL"), default=None, index=True
    )
    service_name: Mapped[str | None] = mapped_column(String(100), default=None, index=True)
    version: Mapped[str | None] = mapped_column(String(50), default=None)
    environment: Mapped[Environment | None] = mapped_column(str_enum(Environment), default=None)
    source_path: Mapped[str | None] = mapped_column(String(400), default=None)
    doc_timestamp: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    meta: Mapped[JsonDict] = mapped_column(default=dict)

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentChunk(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    # Denormalised metadata copied from the parent document so retrieval
    # filtering never needs a join.
    document_type: Mapped[DocumentType] = mapped_column(str_enum(DocumentType), index=True)
    service_name: Mapped[str | None] = mapped_column(String(100), default=None, index=True)
    version: Mapped[str | None] = mapped_column(String(50), default=None)
    environment: Mapped[Environment | None] = mapped_column(str_enum(Environment), default=None)
    chunk_timestamp: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    meta: Mapped[JsonDict] = mapped_column(default=dict)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), default=None)

    # Generated full-text search vector (Phase 5 uses it; created now so the
    # schema is stable). English config over the chunk content.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "uq_document_chunks_document_id_chunk_index",
            "document_id",
            "chunk_index",
            unique=True,
        ),
        Index("ix_document_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_document_chunks_service_name_document_type", "service_name", "document_type"),
    )
