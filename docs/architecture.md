# Architecture

## Principles

1. **Transparent core.** Retrieval, scoring, RRF, citation handling, ranking, and evaluation
   metrics are ordinary, unit-tested Python. No framework hides this logic.
2. **Separation of concerns.** Each stage is its own package with a narrow interface:
   `ingestion → retrieval → generation → evaluation`, plus `agent`, `providers`,
   `observability`, `telemetry`.
3. **Deterministic where it matters.** Synthetic data generation, the fake providers, and
   the evaluation harness are seedable/reproducible so experiments are comparable.
4. **Untrusted retrieved content.** Document text is data, never instructions.

## Request flow (normal incident)

```
POST /incidents/analyze
  → orchestrator
    → hybrid retrieval        (semantic pgvector + lexical FTS → RRF)
    → cross-encoder rerank    (≈20 candidates → ≈6 evidence chunks)
    → context builder         (each chunk tagged with its real id)
    → LLM generation          (Pydantic-validated structured output)
    → citation verifier       (every claim resolves to a retrieved chunk)
    → answer  OR  abstention  ("insufficient evidence")
```

**Phase 3 status:** this is the vector-only slice of that pipeline —
`opspilot.retrieval.semantic` (pgvector cosine top-K, no lexical/RRF/rerank yet),
`opspilot.generation.prompt` (evidence-grounded prompt, JSON-schema instructions),
`opspilot.generation.parser` (JSON parse + Pydantic validation + citation
verification, degrading to abstention on any failure — including the offline
`fake` LLM provider, which never emits valid JSON on purpose). Ingestion
(`opspilot.ingestion`) loads `data/generated/`, chunks with `tiktoken`
(`chunk_size`/`chunk_overlap` in tokens), embeds via the configured
`EmbeddingProvider`, and idempotently upserts `services` / `documents` /
`document_chunks` / `deployments` / `historical_incidents`
(`uv run python -m opspilot.ingestion`, or `make ingest`).

## Components

| Package | Responsibility | Phase |
|---------|----------------|-------|
| `config` | env-driven `Settings`, validated on load | 1 |
| `db` | engine/session lifecycle, declarative base, column types | 1 |
| `models` | SQLAlchemy ORM (one module per aggregate) | 1 |
| `providers` | `EmbeddingProvider` / `LLMProvider` protocols + `fake` / `local` / `anthropic` impls | 1 |
| `telemetry` | token + cost accounting; OpenTelemetry wiring | 1 / 11 |
| `ingestion` | load synthetic docs, chunk (token-bounded, overlap), persist | 3 |
| `retrieval` | `semantic`, `lexical`, `fusion` (RRF), `rerank`, `filters` | 3 / 5 / 6 / 7 |
| `generation` | context construction, structured generation, citation verification | 3 |
| `evaluation` | dataset loading, metrics, run persistence, comparison reports | 4 |
| `agent` | LangGraph orchestrator + read-only typed tools + budgets | 8 |

## Data model

See [`backend/src/opspilot/models/`](../backend/src/opspilot/models). Key points:

- `document_chunks` carries a `vector(384)` embedding (HNSW, cosine) **and** a generated
  `content_tsv tsvector` (GIN) — the same rows serve both retrieval arms.
- Chunk metadata (`service_name`, `version`, `environment`, `document_type`, timestamp) is
  denormalised onto the chunk so retrieval filtering needs no join.
- `evaluation_runs` records the full configuration (models, chunk params, top-k, strategy,
  reranker, prompt version, git SHA) alongside aggregate metrics; `evaluation_results` stores
  per-case retrieval/generation/latency/cost breakdowns.

## Why these choices

- **Local embeddings** over a hosted embedding API: reproducible vectors, zero per-token
  cost, CI-runnable, and it exercises the PyTorch/HF stack the project is meant to show.
- **pgvector + Postgres FTS in one database**: hybrid retrieval without a second datastore;
  metadata filters are plain SQL predicates.
- **Provider abstraction**: swapping Claude for a Vercel AI Gateway route (Phase 9 model
  sweeps) never touches retrieval or orchestration code.
