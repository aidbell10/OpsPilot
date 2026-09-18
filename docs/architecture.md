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

## Agentic investigation (Phase 8)

```
POST /incidents/investigate
  → orchestrator.investigate
    → decide   (LLM: call a tool, or answer)  ─┐
    → act      (run the tool, record an        │  loop until
                observation, or a clean         │  final_answer or
                failure — never a crash)        │  budget exhausted
    └────────────────────────────────────────┘
    → citation-verified AgentFinding  OR  abstention
```

An alternative to `/incidents/analyze`'s one fixed retrieve-then-generate pass: the LLM
chooses which of five **read-only** tools to call and in what order
(`opspilot.agent.tools`: `search_docs`, `find_similar_incidents`, `get_incident_logs`,
`get_deployment`, `get_service_dependencies`), building its own evidence trail before
answering. `opspilot.agent.graph` is a two-node LangGraph `StateGraph` (`decide` / `act`) —
the whole control-flow contribution LangGraph makes to this project; the decision protocol,
citation verification, and every tool are plain, unit-tested Python underneath it, per the
"transparent core" principle above.

* **No native tool-calling.** `LLMProvider.complete` stays `(system, user) -> text` — the same
  contract Phase 3 established, unchanged so a future Phase-9 provider only ever needs to
  implement one method. The agent's "which tool next" decision is a Pydantic-validated JSON
  object (`opspilot.schemas.agent.AgentDecision`), parsed by `opspilot.agent.parser` with the
  exact same philosophy as `generation.parser`: malformed JSON is an immediate abstention
  (never a crash), and an uncited "sufficient evidence" claim is downgraded to an abstention.
  An *unknown tool name*, unlike malformed JSON, is treated as recoverable — it's recorded as a
  failed observation and the model gets another turn, the same way a human's typo isn't fatal.
* **Hard budgets, checked before every LLM call** (`opspilot.agent.budget.BudgetTracker`):
  `max_tool_calls`, `max_cost_usd` (reusing the Phase-1 `CostAccumulator`), `max_seconds`.
  Once exhausted, the next `decide` call is told this is its last turn; if the model still asks
  for a tool anyway, that attempt is overridden into an abstention rather than honored — `act`
  can never run once the budget is spent, regardless of what the model wants.
* **Read-only by construction, not by policy.** Every tool is a parameterized SQLAlchemy
  `select` (or the existing retrieval/generation helpers) behind a validated argument schema;
  there is no code path here to raw SQL, a shell, a deployment, or any mutation. Remediation is
  always a `recommended_actions` string for a human to review — the agent never claims to have
  performed one.
* **`query_metrics`** (in the original roadmap sketch) is deliberately not implemented: this
  project has no metrics/time-series backend, and fabricating one to give the tool something
  to return would violate the "no fabricated numbers, ever" rule. Five real tools beat six
  where one always lies.
* `services.depends_on` (migration `0002`) is new: present in the Phase 2 generator's
  `ServiceRecord` since Phase 1 but never persisted until `get_service_dependencies` needed a
  real column instead of re-reading `data/generated/services.json`.

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
