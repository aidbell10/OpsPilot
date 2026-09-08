"""Evidence search (Phase 5/6: vector / lexical / hybrid-RRF + optional rerank)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from opspilot.config import RetrievalStrategyName, get_settings
from opspilot.db.session import get_db
from opspilot.models.enums import DocumentType, RetrievalStrategy
from opspilot.providers.factory import get_embedding_provider, get_rerank_provider
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.strategy import retrieve_chunks
from opspilot.schemas.analysis import EvidenceItem
from opspilot.schemas.search import SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    service: str | None = None
    document_type: DocumentType | None = None
    version: str | None = None
    strategy: RetrievalStrategyName | None = Field(
        default=None, description="Override OPSPILOT_RETRIEVAL_STRATEGY for this request."
    )
    rerank: bool | None = Field(
        default=None,
        description="Force the cross-encoder reranker on/off for this request "
        "(default: whether OPSPILOT_RERANKER is configured).",
    )


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    strategy = RetrievalStrategy.from_name(request.strategy or settings.retrieval_strategy)
    reranker = None if request.rerank is False else get_rerank_provider()
    if request.rerank is True and reranker is None:
        raise HTTPException(
            status_code=422, detail="rerank=true requested but OPSPILOT_RERANKER=none"
        )

    query_vector = embedding_provider.embed([request.query]).vectors[0]
    matches = retrieve_chunks(
        db,
        strategy=strategy,
        query_embedding=query_vector,
        query_text=request.query,
        top_k=request.top_k,
        filters=ChunkFilters(
            service_name=request.service,
            document_type=request.document_type,
            version=request.version,
        ),
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.rrf_k,
        reranker=reranker,
        rerank_candidate_k=settings.rerank_candidate_k,
    )
    return SearchResponse(
        query=request.query,
        strategy=strategy.value,
        reranker=reranker.model if reranker is not None else None,
        results=[
            EvidenceItem(
                chunk_id=str(m.chunk_id),
                document_id=str(m.document_id),
                title=m.title,
                source_path=m.source_path,
                document_type=m.document_type,
                service_name=m.service_name,
                version=m.version,
                content=m.content,
                score=m.score,
            )
            for m in matches
        ],
    )
