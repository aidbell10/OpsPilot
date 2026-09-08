# Evaluation methodology

> Status: **Phase 4 harness is implemented** (`backend/src/opspilot/evaluation/`). This
> document describes what is actually built, not a plan.

## Dataset

The ground-truth dataset comes straight from the Phase 2 generator's `data/generated/
ground_truth.json` — 22 cases as of `gen2.0.0-seed42`: **18 answerable, 4 unanswerable**.
Categories: 15 `root_cause`, 4 `abstention`, 1 `prompt_injection`, 1 `duplicate_charge`,
1 `data_exposure`. Difficulty: 11 `multi_hop`, 5 `straightforward`, 4 `unanswerable`,
2 `adversarial`. Each answerable case is engineered against one of the 12 planted ground-truth
chains from Phase 2 (a deployment changes a function → a later incident → a log line with that
function's error code → a related historical incident → a troubleshooting runbook), so the
required evidence is *known*, not guessed.

Target composition as the corpus grows (per the original roadmap, unchanged): ~140 cases,
~90 dev / ~50 held-out test, split straightforward / multi-hop / unanswerable / adversarial.
**The test set is never tuned against**, once one exists.

### Loading (`opspilot.evaluation.loader`)

```bash
cd backend
uv run python -m opspilot.evaluation load-cases           # or: make eval-load
uv run python -m opspilot.evaluation load-cases --dry-run  # count only, no DB writes
```

Three field-name decisions, made because the `evaluation_cases` schema (written in Phase 1,
before the generator's exact output shape existed) doesn't quite match `ground_truth.json`:

* **`required_document_source_paths` → `required_document_ids`.** These are `Document
  .source_path` natural keys, not UUIDs — a `Document`'s id doesn't exist, and isn't stable
  across re-ingests, until ingestion runs. Retrieval metrics compare directly against
  `ChunkMatch.source_path`, so no id-resolution lookup is needed at evaluation time either;
  the natural key round-trips end to end.
* **`required_evidence_snippets` → `required_evidence_ids`.** Repurposed: this column holds
  literal substrings expected to appear in relevant evidence text (error codes, function
  names, version strings — e.g. `PAY-50231`, `authorize_payment`, `v5.0.6`), used for an
  evidence-coverage/grounding check. It is not a list of ids. The column name is a Phase-1
  relic kept to avoid a rename-only migration.
* **`chain_id` is not persisted.** No metric groups by chain, and it is recoverable from
  `case_id`'s prefix (e.g. `gt-01-...`) if that ever changes.

**`split`**: every case currently loads as `EvalSplit.DEV`. The generator emits no split field,
and with only 22 cases across 12 chains (several near-duplicate queries share a chain, e.g.
`gt-01-payments-auth-timeout-{01,02,adv}`), any dev/test split today would either be
statistically meaningless or leak chain siblings across the boundary — the opposite of a
held-out guarantee. A real split (grouped by `chain_id`, never splitting a chain across dev
and test) is Phase-4-follow-up work once the corpus grows toward its target size.

**`dataset_version`**: `f"gen{generator_version}-seed{seed}"` read from `manifest.json`
(currently `gen2.0.0-seed42`) — stable across identical rebuilds of the same corpus, and
changes whenever the generator's output semantics or seed change.

## Metrics (`opspilot.evaluation.metrics`)

Pure functions, unit-tested against hand-computed values in
`tests/unit/test_evaluation_metrics.py`.

**Retrieval** (compared against `required_document_ids`, i.e. source paths)
- `recall_at_k` — fraction of required documents present anywhere in the retrieved set.
- `evidence_coverage` — fraction of `required_evidence_ids` (snippets) found as a
  case-insensitive substring in any retrieved chunk's content.
- `mrr` — reciprocal rank of the first retrieved chunk belonging to a required document.
- `ndcg_at_10` — binary-relevance nDCG (chunk's document ∈ required set).

An empty required set (every unanswerable case) is *vacuously* satisfied — these report `1.0`
for such cases. Whether the system behaved correctly on an unanswerable case is judged by the
abstention metrics below, not by these.

**Generation**
- `citation_precision` — of the model's *verified* citations (already guaranteed to be real,
  retrieved chunk ids by `opspilot.generation.parser.parse_llm_output` — fabricated citations
  never reach this metric), the fraction that belong to a required document. `None`
  (undefined, excluded from aggregation) when there are no citations to score or nothing is
  required — treating that as `0` would penalize a correctly-abstained case for citing
  nothing, which is backwards.
- `hallucinated_forbidden_claim` — a **substring-based heuristic**: true if any of the case's
  `forbidden_claims` strings appears (case-insensitively) in the model's `hypothesis` +
  `root_cause` text. **This is not a semantic faithfulness judge.** It catches an exact or
  near-exact forbidden phrase but will miss a paraphrased hallucination and can, in principle,
  false-positive on an incidental substring match. A real LLM-judge or NLI-based faithfulness
  check is future work (not yet built) — do not read `hallucination_rate` as a calibrated
  hallucination measurement, only as a coarse smoke signal.

**Reliability** — a confusion matrix where the positive class is "should abstain"
(`answerable=False`) vs. "did abstain" (`evidence_sufficient=False` in the model's output):
abstention precision, recall, F1.

**Performance / cost** — retrieval-stage and generation-stage wall-clock (`time.perf_counter`),
persisted per case (`latency_ms`) and aggregated as p50/p95 (`statistics.quantiles`); embedding
+ LLM token usage and estimated cost via `opspilot.telemetry.cost` (fed by the real token
counts the providers return, not estimates of the run itself).

## Runner (`opspilot.evaluation.runner`)

```bash
cd backend
uv run python -m opspilot.evaluation run --dataset-version gen2.0.0-seed42 --split dev
uv run python -m opspilot.evaluation run --strategy vector   # vector | lexical | hybrid
# or: make eval-run  (defaults to the manifest's dataset_version, split=dev, and
#     OPSPILOT_RETRIEVAL_STRATEGY); make eval-experiment runs all three strategies + report
```

**`--strategy`** (Phase 5) selects the retrieval arm for the run and is recorded on the
`EvaluationRun` row (`retrieval_strategy`), so `vector` / `lexical` / `hybrid` runs sit
side by side in the report. It defaults to `OPSPILOT_RETRIEVAL_STRATEGY` (`hybrid`). All
three call the same `opspilot.retrieval.strategy.retrieve_chunks` dispatch the API uses;
only the retrieval arm changes, everything downstream (prompt, generation, metrics) is
identical.

For each loaded case: embeds the query once, runs `opspilot.retrieval.semantic.search_chunks`
/ `search_historical_incidents` (the exact functions `POST /search` and
`POST /incidents/analyze` use), builds the prompt with `opspilot.generation.prompt
.build_user_prompt`, calls the configured `LLMProvider`, and parses/validates/citation-checks
the response with `opspilot.generation.parser.parse_llm_output` — the same function the API
uses. Nothing about retrieval or generation is reimplemented; only the thin endpoint-level
convenience wrappers (which don't expose token counts needed for cost accounting) are bypassed
in favor of the lower-level functions they call.

**`passed`** — an answerable case passes if `recall_at_k == 1.0` (every required document was
retrieved) *and* the model did not abstain *and* no forbidden claim was detected. An
unanswerable case passes if the model abstained. This is deliberately strict: partial credit
belongs in the individual metrics, not in a boolean.

One `EvaluationRun` row is written per invocation (config snapshot: models, chunk size/overlap,
top-K, `retrieval_strategy` (the `--strategy` for the run), `reranker=null`, prompt version,
dataset version/split, git SHA when available) with `aggregate_metrics` (means, abstention
P/R/F1, latency p50/p95, total + mean cost) filled in after every case runs. Runs are an append-only history, not idempotent —
re-running is how a system change gets compared against the past.

## ⚠️ Fake vs. real LLM provider — read this before trusting any generation number

The default `.env` uses `OPSPILOT_LLM_PROVIDER=fake`. The fake provider **never emits valid
JSON** (see `opspilot.providers.fake.FakeLLMProvider` — it deliberately echoes a
recognizably-synthetic string), so under the default configuration **every single case
abstains**, regardless of whether the evidence was actually sufficient. This is the *correct,
honest* behavior of the abstention path (a parse failure degrades to an explicit "insufficient
evidence" response rather than fabricating an answer) — it is not a bug, and it is not a
measurement of system quality.

Retrieval metrics (`recall_at_k`, `mrr`, `ndcg_at_10`, `evidence_coverage`) computed under the
fake or local embedding provider are genuine and meaningful — they don't depend on the LLM
provider at all. **Generation-quality numbers (citation precision, hallucination rate, pass
rate, abstention P/R/F1) computed under the fake LLM provider measure the fake provider's
behavior, not the system's.** Never present a fake-provider run's generation metrics as if they
characterized real answer quality.

To run a real (**paid**) benchmark:

```bash
export OPSPILOT_LLM_PROVIDER=anthropic
export OPSPILOT_ANTHROPIC_API_KEY=sk-...
cd backend
uv run python -m opspilot.evaluation run --dataset-version gen2.0.0-seed42 --split dev
```

This calls the real Anthropic API once per case (22 cases in the current dataset) — costs
scale with `dataset_version`/split size and `OPSPILOT_LLM_MODEL`. Nothing in this repository
runs that command automatically; it is always an explicit, opt-in action.

## Reports (`opspilot.evaluation.report`)

```bash
cd backend
uv run python -m opspilot.evaluation report   # or: make eval-report
```

Renders a plain-text comparison table across every persisted `EvaluationRun`, one row per run,
keyed by `strategy` / `reranker` / dataset version / git SHA. Phase 5 adds the `lexical` and
`hybrid_rrf` rows next to the Phase 3 `vector` baseline; later phases (reranking, fine-tuned
reranker, agent) slot in the same way with no change here. **Reported numbers are computed from
real system output only — never hand-written.**
