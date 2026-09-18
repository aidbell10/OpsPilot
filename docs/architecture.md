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
    → cross-encoder rerank    (optional: ≈20 candidates → top_k evidence chunks)
    → context builder         (each chunk tagged with its real id)
    → LLM generation          (Pydantic-validated structured output)
    → citation verifier       (every claim resolves to a retrieved chunk)
    → answer  OR  abstention  ("insufficient evidence")
```

**Phase 5/6 status:** retrieval is `vector` / `lexical` / `hybrid` (RRF), selected by
`OPSPILOT_RETRIEVAL_STRATEGY` (default `hybrid`), optionally followed by a cross-encoder
rerank when `OPSPILOT_RERANKER` is set (default `none`). Both are dispatched in one place —
`opspilot.retrieval.strategy.retrieve_chunks`, called by the API routes and (as its two
component steps, for per-stage timing) the evaluation runner, so they cannot drift:

* `opspilot.retrieval.semantic` — pgvector cosine top-K (HNSW index).
* `opspilot.retrieval.lexical` — PostgreSQL FTS over the generated `content_tsv` (GIN index);
  `websearch_to_tsquery` + `ts_rank_cd`. Chunks with no lexical overlap are excluded, so this
  arm can return fewer than *k* results.
* `opspilot.retrieval.fusion` — Reciprocal Rank Fusion (`1/(k+rank)`, k=60); rank-only, so the
  cosine and `ts_rank_cd` scales never need reconciling.
* `opspilot.retrieval.hybrid` — pulls `retrieval_candidate_k` from each arm, RRF-fuses, keeps
  top-K.
* `opspilot.retrieval.filters` — `ChunkFilters` (service / doc type / version / environment /
  timestamp window); the same object is applied identically to both arms.
* `opspilot.retrieval.rerank` — re-scores the retrieved candidate list with a
  `RerankProvider` (`fake`, or a local sentence-transformers `CrossEncoder` —
  `cross-encoder/ms-marco-MiniLM-L-6-v2` by default) and keeps the best `retrieval_top_k`.
  The cross-encoder sees (query, chunk) jointly, so it is more accurate than the bi-encoder
  but only affordable over a ~20-candidate set. **Phase 7** fine-tunes this model on the
  planted ground truth (`ml/train_reranker.py`) — the fine-tuned checkpoint is just another
  `reranker_model` value (a local path instead of a HF hub id); no runtime code changed to
  support it, since `LocalCrossEncoderProvider(model=...)` already accepted an arbitrary
  string. See [`docs/ml-training.md`](ml-training.md).

Generation is unchanged:
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
| `providers` | `EmbeddingProvider` / `LLMProvider` / `RerankProvider` protocols + `fake` / `local` / `anthropic` impls | 1 / 6 |
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
