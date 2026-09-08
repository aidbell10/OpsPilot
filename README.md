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

### Ingest the synthetic knowledge base (Phase 3)

```bash
python -m data.generator             # -> data/generated/  (see below)
cd backend
uv run python -m opspilot.ingestion  # chunk, embed, and upsert into Postgres — idempotent
# or: make ingest / make ingest-dry-run (loads + chunks without touching the DB)
```

Then try it:

```bash
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "auth rejecting valid tokens after key rotation", "top_k": 3}'
# retrieval strategy defaults to OPSPILOT_RETRIEVAL_STRATEGY (hybrid); override per request:
#   -d '{"query": "...", "strategy": "lexical", "service": "auth", "version": "v5.0.6"}'

curl -X POST localhost:8000/incidents/analyze -H 'content-type: application/json' \
  -d '{"description": "auth is rejecting valid tokens with invalid signature errors", "service": "auth"}'
```

With the default `fake` LLM provider, `/incidents/analyze` retrieves real evidence but always
abstains on generation (the fake provider doesn't emit valid structured JSON — that's the
point: a parse/validation failure degrades to an explicit "insufficient evidence" response
instead of a 500). Set `OPSPILOT_LLM_PROVIDER=anthropic` and `OPSPILOT_ANTHROPIC_API_KEY` for
real generation.

### Evaluation harness (Phase 4)

```bash
cd backend
uv run python -m opspilot.evaluation load-cases          # ground_truth.json -> evaluation_cases
uv run python -m opspilot.evaluation run                 # pipeline over the dev split
uv run python -m opspilot.evaluation run --strategy lexical   # vector | lexical | hybrid
uv run python -m opspilot.evaluation report              # comparison table across all runs
# or: make eval-load / make eval-run / make eval-report
# make eval-experiment  -> one run each of vector, lexical, hybrid, then the report
```

Retrieval metrics (Recall@K, MRR, nDCG@10, evidence coverage) are real under any embedding
provider. Generation metrics (citation precision, hallucination rate, abstention P/R/F1) are
only meaningful with `OPSPILOT_LLM_PROVIDER=anthropic` — under the default `fake` provider
every case abstains by design, which is an honest reflection of the offline provider, not of
system quality. See [`docs/evaluation.md`](docs/evaluation.md) for the full methodology and
field-mapping decisions.

### Tests

```bash
cd backend
uv run pytest -m "not integration"   # fast: no infra, no network  (runs in CI)
uv run pytest                        # full: needs `docker compose up -d db`
uv run pytest -m slow                # downloads the embedding model
```

### Synthetic knowledge base (Phase 2)

```bash
python -m data.generator             # -> data/generated/  (deterministic, seed 42)
python -m data.generator --check     # fail if data/generated/ is stale
make gen  /  make gen-check  /  make gen-test
```

Meridian Retail — 7 services, ~115 releases, ~95 documents, ~25 historical
incidents, and 12 planted ground-truth chains with ~22 evaluation cases
(straightforward / multi-hop / unanswerable / adversarial). `data/generated/` is
git-ignored and rebuilt on demand. See [`data/generator/README.md`](data/generator/README.md).

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
| 2  | Deterministic synthetic knowledge base (coherent fictional company, planted ground truth) | ✅ done |
| 3  | Baseline RAG: ingestion, chunking, pgvector retrieval, structured cited answers, abstention | ✅ done |
| 4  | Versioned evaluation harness (retrieval / generation / reliability / performance / cost metrics) | ✅ done |
| 5  | Hybrid retrieval (vector + FTS + RRF), metadata filters, strategy-per-eval-run; measured vector-vs-lexical-vs-hybrid experiment | ✅ done ([Experiment 1](docs/experiments.md)) |
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
