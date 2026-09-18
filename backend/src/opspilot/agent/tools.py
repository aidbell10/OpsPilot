"""Read-only typed tools available to the Phase 8 investigation agent.

Every tool is a plain SQLAlchemy ``select`` (or the existing retrieval/
generation helpers) behind a validated Pydantic argument schema — there is no
code path here to raw SQL, a shell, a deployment, or any mutation. This is
the hard prohibition from the plan enforced by construction: the agent's only
capabilities are the five functions registered in :data:`TOOLS`, all reads.

``query_metrics`` (sketched in the original roadmap) is deliberately not
implemented: this project has no time-series/metrics backend, and fabricating
one to give the tool something to return would violate "no fabricated
numbers, ever" (see docs/experiments.md). Five real, DB-backed tools beat six
where one always lies.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from opspilot.config import Settings
from opspilot.generation.prompt import format_chunk, format_related_incident
from opspilot.models.deployment import Deployment
from opspilot.models.enums import DocumentType, RetrievalStrategy
from opspilot.models.historical_incident import HistoricalIncident
from opspilot.models.service import Service
from opspilot.providers.base import EmbeddingProvider, RerankProvider
from opspilot.retrieval.filters import ChunkFilters
from opspilot.retrieval.semantic import search_historical_incidents
from opspilot.retrieval.strategy import retrieve_chunks
from opspilot.schemas.analysis import RelatedIncident
from opspilot.telemetry.cost import CostAccumulator


class ToolError(Exception):
    """A tool's expected, recoverable failure (not found, no data) — never a crash."""


@dataclass(frozen=True, slots=True)
class ToolContext:
    embedding_provider: EmbeddingProvider
    rerank_provider: RerankProvider | None
    settings: Settings
    cost: CostAccumulator


# --------------------------------------------------------------------------
# Argument schemas
# --------------------------------------------------------------------------
class SearchDocsArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    service_name: str | None = None
    document_type: DocumentType | None = None
    top_k: int = Field(default=8, ge=1, le=20)


class FindSimilarIncidentsArgs(BaseModel):
    symptoms: str = Field(min_length=1, max_length=2000)
    service_name: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class GetIncidentLogsArgs(BaseModel):
    service_name: str
    error_code: str | None = None
    incident_id: str | None = None

    @model_validator(mode="after")
    def _require_one_lookup_key(self) -> GetIncidentLogsArgs:
        if not self.error_code and not self.incident_id:
            raise ValueError("one of error_code or incident_id is required")
        return self


class GetDeploymentArgs(BaseModel):
    service_name: str
    version: str


class GetServiceDependenciesArgs(BaseModel):
    service_name: str


# --------------------------------------------------------------------------
# Implementations — each returns the observation text, or raises ToolError
# --------------------------------------------------------------------------
def _search_docs(session: Session, args: SearchDocsArgs, ctx: ToolContext) -> str:
    embed_result = ctx.embedding_provider.embed([args.query])
    ctx.cost.add_embedding(ctx.embedding_provider.model, embed_result.total_tokens)

    strategy = RetrievalStrategy.from_name(ctx.settings.retrieval_strategy)
    chunks = retrieve_chunks(
        session,
        strategy=strategy,
        query_embedding=embed_result.vectors[0],
        query_text=args.query,
        top_k=args.top_k,
        filters=ChunkFilters(service_name=args.service_name, document_type=args.document_type),
        candidate_k=ctx.settings.retrieval_candidate_k,
        rrf_k=ctx.settings.rrf_k,
        reranker=ctx.rerank_provider,
        rerank_candidate_k=ctx.settings.rerank_candidate_k,
    )
    if not chunks:
        return "No matching document chunks found."
    return "\n\n".join(format_chunk(c) for c in chunks)


def _find_similar_incidents(
    session: Session, args: FindSimilarIncidentsArgs, ctx: ToolContext
) -> str:
    embed_result = ctx.embedding_provider.embed([args.symptoms])
    ctx.cost.add_embedding(ctx.embedding_provider.model, embed_result.total_tokens)

    matches = search_historical_incidents(
        session, embed_result.vectors[0], top_k=args.top_k, service_name=args.service_name
    )
    if not matches:
        return "No similar historical incidents found."
    related = [
        RelatedIncident(
            incident_id=str(m.incident_id),
            title=m.title,
            service_name=m.service_name,
            occurred_at=dt.datetime.fromisoformat(m.occurred_at),
            severity=m.severity,
            root_cause=m.root_cause,
            resolution=m.resolution,
            score=m.score,
        )
        for m in matches
    ]
    return "\n\n".join(format_related_incident(r) for r in related)


def _get_incident_logs(session: Session, args: GetIncidentLogsArgs, _ctx: ToolContext) -> str:
    """Raw log lines for a past incident — NOT its root cause or resolution.

    Deliberately narrower than :func:`_find_similar_incidents`: a real logs
    tool gives you telemetry to correlate, not the answer. Root cause and
    resolution stay behind similarity search, which is explicitly a lookup
    over *resolved* incidents.
    """
    stmt = select(HistoricalIncident).where(HistoricalIncident.service_name == args.service_name)
    if args.incident_id:
        stmt = stmt.where(HistoricalIncident.meta["incident_id"].astext == args.incident_id)
    else:
        stmt = stmt.where(HistoricalIncident.meta["error_code"].astext == args.error_code)

    incident = session.execute(stmt).scalars().first()
    if incident is None:
        raise ToolError(
            f"no historical incident found for service={args.service_name!r} "
            f"error_code={args.error_code!r} incident_id={args.incident_id!r}"
        )

    log_lines = incident.meta.get("log_lines") or []
    incident_ref = incident.meta.get("incident_id", str(incident.id))
    if not log_lines:
        return f"Incident {incident_ref} has no recorded log lines."

    header = (
        f"Logs for {args.service_name} incident {incident_ref} "
        f"(occurred {incident.occurred_at.isoformat()}, related deployment "
        f"{incident.related_deployment_version or 'n/a'}):"
    )
    return header + "\n" + "\n".join(log_lines)


def _get_deployment(session: Session, args: GetDeploymentArgs, _ctx: ToolContext) -> str:
    deployment = session.execute(
        select(Deployment).where(
            Deployment.service_name == args.service_name, Deployment.version == args.version
        )
    ).scalar_one_or_none()
    if deployment is None:
        raise ToolError(f"no deployment found for {args.service_name} {args.version}")

    changed = json.dumps(deployment.changed_components) if deployment.changed_components else "{}"
    return (
        f"Deployment {args.service_name} {args.version}: status={deployment.status.value}, "
        f"environment={deployment.environment.value}, "
        f"deployed_at={deployment.deployed_at.isoformat()}\n"
        f"change_summary: {deployment.change_summary or 'n/a'}\n"
        f"changed_components: {changed}"
    )


def _get_service_dependencies(
    session: Session, args: GetServiceDependenciesArgs, _ctx: ToolContext
) -> str:
    service = session.execute(
        select(Service).where(Service.name == args.service_name)
    ).scalar_one_or_none()
    if service is None:
        raise ToolError(f"no service named {args.service_name!r}")

    all_services = session.execute(select(Service.name, Service.depends_on)).all()
    dependents = sorted(name for name, deps in all_services if args.service_name in (deps or []))

    return "\n".join(
        [
            f"Service {args.service_name} (tier {service.tier}, owned by {service.owning_team}):",
            f"depends on: {', '.join(service.depends_on) or '(none)'}",
            f"depended on by: {', '.join(dependents) or '(none)'}",
        ]
    )


# --------------------------------------------------------------------------
# Registry + dispatch
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    # Every registered fn actually takes its own args_model's type as the
    # second parameter, not bare BaseModel — dispatch() validates against
    # args_model before calling, so the narrower type is always honored at
    # runtime. A heterogeneous registry can't express that in the type
    # system without existential types, hence Any here.
    fn: Callable[[Session, Any, ToolContext], str]


TOOLS: dict[str, ToolSpec] = {
    "search_docs": ToolSpec(
        name="search_docs",
        description=(
            "Hybrid semantic+lexical search over runbooks, postmortems, architecture docs, "
            "deployment notes, and known-error docs. Use this to find written knowledge about "
            "a symptom, error code, function, or service."
        ),
        args_model=SearchDocsArgs,
        fn=_search_docs,
    ),
    "find_similar_incidents": ToolSpec(
        name="find_similar_incidents",
        description=(
            "Semantic search over RESOLVED past incidents by symptom similarity. Returns each "
            "match's root cause and resolution — use this to find an analogous solved incident."
        ),
        args_model=FindSimilarIncidentsArgs,
        fn=_find_similar_incidents,
    ),
    "get_incident_logs": ToolSpec(
        name="get_incident_logs",
        description=(
            "Raw log lines for one past incident, looked up by service_name plus either "
            "error_code or incident_id. Does NOT include root cause or resolution — use "
            "find_similar_incidents for that."
        ),
        args_model=GetIncidentLogsArgs,
        fn=_get_incident_logs,
    ),
    "get_deployment": ToolSpec(
        name="get_deployment",
        description="Exact lookup of one deployment record by service_name and version.",
        args_model=GetDeploymentArgs,
        fn=_get_deployment,
    ),
    "get_service_dependencies": ToolSpec(
        name="get_service_dependencies",
        description=(
            "What a service depends on, and what depends on it — use this to reason about "
            "blast radius across service boundaries."
        ),
        args_model=GetServiceDependenciesArgs,
        fn=_get_service_dependencies,
    ),
}


def render_tool_specs() -> str:
    """Render every tool's name, description, and argument schema for the system prompt."""
    blocks = []
    for spec in TOOLS.values():
        schema = json.dumps(spec.args_model.model_json_schema(), indent=2)
        blocks.append(f"### {spec.name}\n{spec.description}\nArguments schema:\n{schema}")
    return "\n\n".join(blocks)


def dispatch(
    name: str, raw_arguments: dict[str, object], session: Session, ctx: ToolContext
) -> tuple[bool, str, str | None]:
    """Validate arguments and run a tool call. Never raises.

    Returns ``(ok, content, error)`` — exactly one of ``content``/``error`` is
    meaningful, matching :class:`opspilot.schemas.agent.Observation`.
    """
    spec = TOOLS.get(name)
    if spec is None:
        return False, "", f"unknown tool {name!r}; available tools: {', '.join(sorted(TOOLS))}"

    try:
        args = spec.args_model.model_validate(raw_arguments)
    except ValidationError as exc:
        return False, "", f"invalid arguments for {name}: {exc}"

    try:
        content = spec.fn(session, args, ctx)
    except ToolError as exc:
        return False, "", str(exc)
    return True, content, None
