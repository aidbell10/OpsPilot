# OpsPilot AI

**Agentic Incident Investigation & RAG Evaluation Platform**

OpsPilot is an internal engineering tool for a *fictional* software company. An engineer
submits an incident — for example:

> "Checkout requests started returning HTTP 500 errors after deployment v2.14.0."

…and OpsPilot investigates the available evidence (runbooks, postmortems, deployment
records, known-error docs, logs, metrics) and returns a structured, **citation-backed**
result:

1. incident classification
2. likely root-cause hypothesis
3. confidence / evidence-sufficiency assessment
4. supporting evidence (with exact chunk/document citations)
5. similar historical incidents
6. recommended investigative steps
7. safe remediation recommendations (human-approval only)
8. an explicit **"insufficient evidence"** response when a reliable diagnosis is not possible

> All data in this project is synthetic. OpsPilot never assumes access to real employer,
> customer, or proprietary data.

This repository is a portfolio project. Its goal is to demonstrate real AI/ML engineering:
hybrid retrieval, Reciprocal Rank Fusion, reranking (pretrained **and** a fine-tuned
PyTorch cross-encoder), structured LLM output with verified citations, a versioned
evaluation harness that produces **measured** numbers, read-only agent tooling, and
production concerns (Docker, CI/CD, observability, cost/latency tradeoffs).

---

## Architecture

The pipeline is deliberately built from transparent Python. A framework (LangGraph)
is introduced only for agent control flow, in a later phase.

```
                 ┌─────────────┐
   engineer ───▶ │  Next.js UI │  (Phase 10)
                 └──────┬──────┘
                        │  HTTP
                 ┌──────▼───────┐
                 │  FastAPI API │
                 └──────┬───────┘
                        │
              ┌─────────▼──────────┐
              │  Investigation     │
              │  orchestrator      │
              └───┬────────────┬───┘
                  │            │
   normal path    │            │  (complex incidents, Phase 8)
                  │            └──────────────┐
        ┌─────────▼─────────┐        ┌────────▼─────────┐
        │  hybrid retrieval │        │  LangGraph agent │
        │  (vector + FTS)   │        │  read-only tools │
        │        │ RRF      │        └────────┬─────────┘
        │        ▼          │                 │
        │  cross-encoder    │                 │
        │  reranking        │                 │
        └─────────┬─────────┘                 │
                  │  top evidence             │
        ┌─────────▼─────────┐                 │
        │ context builder   │◀────────────────┘
        └─────────┬─────────┘
        ┌─────────▼─────────┐
        │  LLM generation   │  (structured Pydantic output)
        └─────────┬─────────┘
        ┌─────────▼─────────┐
        │ citation verifier │  → answer OR abstention
        └───────────────────┘
```

Retrieval target (built up over Phases 3 → 7):

```
query ─┬─▶ semantic retrieval (pgvector) ─┐
       │                                  ├─▶ Reciprocal Rank Fusion ─▶ cross-encoder rerank ─▶ top evidence ─▶ LLM
       └─▶ lexical retrieval (Postgres FTS)┘
```

Separation of concerns (one package each): `ingestion`, `retrieval`, `generation`,
`evaluation`, `agent`, `observability`, `providers`, `telemetry`.

---

## Stack

| Layer          | Choice |
|----------------|--------|
| Backend        | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic |
| Database       | PostgreSQL 16, `pgvector` (HNSW), PostgreSQL full-text search |
| Embeddings     | local Sentence-Transformers (`BAAI/bge-small-en-v1.5`, 384-dim) via PyTorch |
| LLM            | Anthropic Claude, behind a provider abstraction (Vercel AI Gateway pluggable later) |
| ML             | PyTorch + Hugging Face — fine-tuned cross-encoder reranker (Phase 7) |
| Agent          | LangGraph (Phase 8), read-only typed tools only |
| Frontend       | Next.js, React, TypeScript (Phase 10) |
| Infra          | Docker, Docker Compose, GitHub Actions |
| Observability  | OpenTelemetry (Phase 11) |
| Testing        | pytest — unit, integration, retrieval regression, agent, adversarial |

A deterministic **fake** provider (hash-based embeddings, templated generation) lets the
entire ingestion → retrieval → evaluation pipeline and CI run offline, with no API keys.

---

## Data model

`services`, `documents`, `document_chunks` (vector + generated `tsvector`), `deployments`,
`incidents`, `historical_incidents`, `evaluation_cases`, `evaluation_runs`,
`evaluation_results`, `user_feedback`.

Citations always reference **real stored** `document_chunks` / `documents` ids — the LLM is
never trusted to invent citation strings; a verifier drops or flags anything unresolvable.

---

## Local setup

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/),
[`uv`](https://docs.astral.sh/uv/), Python 3.12 (uv can install it).

```bash
# 1. start PostgreSQL + pgvector
docker compose up -d db

# 2. backend deps
cd backend
uv sync --group dev            # add --extra ml for the local embedding model

# 3. apply migrations
cp ../.env.example ../.env     # defaults are offline-safe
uv run alembic upgrade head

# 4. run the API
uv run uvicorn opspilot.main:app --reload
# GET http://localhost:8000/health        -> {"status":"ok",...}
# GET http://localhost:8000/health/ready  -> checks db + pgvector + migrations
# GET http://localhost:8000/docs          -> OpenAPI

# …or run everything in containers
docker compose up
```

### Tests

```bash
cd backend
uv run pytest -m "not integration"   # fast: no infra, no network  (runs in CI)
uv run pytest                        # full: needs `docker compose up -d db`
uv run pytest -m slow                # downloads the embedding model
```

Quality gate: `uv run ruff check .` · `uv run ruff format --check .` · `uv run mypy src`.

Point integration tests at an already-running database with
`OPSPILOT_TEST_DATABASE_URL=postgresql+psycopg://opspilot:opspilot@localhost:5432/opspilot`;
otherwise they spin up an ephemeral `pgvector` container via testcontainers, and skip if
Docker is unavailable.

---

## Project status

| Phase | Scope | State |
|------:|-------|-------|
| 1  | Foundation: scaffold, config, DB + migrations, health checks, provider abstraction, tests, CI skeleton | ✅ done |
| 2  | Deterministic synthetic knowledge base (coherent fictional company, planted ground truth) | ⏳ next |
| 3  | Baseline RAG: ingestion, chunking, pgvector retrieval, structured cited answers, abstention | ⏳ |
| 4  | Versioned evaluation harness (retrieval / generation / reliability / performance / cost metrics) | ⏳ |
| 5  | Hybrid retrieval (vector + FTS + RRF) with a measured vector-vs-lexical-vs-hybrid experiment | ⏳ |
| 6  | Pretrained cross-encoder reranking, measured | ⏳ |
| 7  | Fine-tuned PyTorch cross-encoder reranker (hard negatives, loss curves, A/B/C comparison) | ⏳ |
| 8  | LangGraph agent with read-only tools and hard budgets | ⏳ |
| 9  | "Does the agent help?" experiment + complexity router | ⏳ |
| 10 | Next.js frontend incl. a first-class evaluation dashboard | ⏳ |
| 11 | OpenTelemetry instrumentation | ⏳ |
| 12 | Full test matrix incl. adversarial (prompt injection in retrieved docs) | ⏳ |
| 13 | CI/CD pipelines | ⏳ |
| 14 | Dockerized deployment (AWS-straightforward) | ⏳ |

Benchmark results, screenshots, and a demo link will be added here as the measured phases land.
Numbers in this project are always produced by real experiments — never hand-written.

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/evaluation.md`](docs/evaluation.md)
- [`docs/ml-training.md`](docs/ml-training.md)
- [`docs/security.md`](docs/security.md)
- [`docs/deployment.md`](docs/deployment.md)
- [`docs/experiments.md`](docs/experiments.md)

## Limitations

- Synthetic data only; retrieval quality claims generalise only as far as the synthetic
  corpus is representative.
- Single-node Postgres; no sharding or read replicas.
- Costs are *estimated* from a configurable pricebook, not billed amounts.
