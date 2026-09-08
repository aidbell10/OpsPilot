# Experiments

A log of controlled experiments. Each entry links to the `evaluation_runs` rows that back
it. **No numbers are entered here by hand** — they are copied from harness output.

| # | Question | Phase | Result |
|--:|----------|-------|--------|
| 1 | vector-only vs lexical-only vs hybrid (RRF) retrieval | 5 | harness verified end-to-end; real-embedding numbers pending |
| 2 | hybrid vs hybrid + pretrained cross-encoder reranker | 6 | pending |
| 3 | pretrained vs fine-tuned cross-encoder reranker | 7 | pending |
| 4 | one-shot RAG vs deterministic retrieve→verify→answer vs LangGraph agent | 9 | pending |

## Experiment 1 — vector vs lexical vs hybrid retrieval

**How to reproduce** (needs `docker compose up -d db`, an ingested corpus, and loaded cases):

```bash
make db-up
cd backend && uv run python -m opspilot.ingestion && uv run python -m opspilot.evaluation load-cases
make eval-experiment      # one run each: --strategy vector | lexical | hybrid, then the report
```

The LLM provider is irrelevant to this experiment (retrieval metrics don't touch it), so the
free `fake` LLM is fine. The **embedding** provider is not: the `fake` embeddings are
token-hash vectors, so a fake-embedding "vector" arm is not a real semantic retriever and its
numbers must not be published as the result. A real run needs `uv sync --extra ml` +
`OPSPILOT_EMBEDDING_PROVIDER=local` (downloads `BAAI/bge-small-en-v1.5` once), then re-ingest
and re-run.

**Status (2026-09-07):** the harness was exercised end to end against a live pgvector DB — all
three strategies run, persist their own `EvaluationRun` row, and land side by side in
`opspilot.evaluation report`. The fake-embedding smoke run gave hybrid a small recall@k edge
over both single arms (0.90 vs 0.87 vector / 0.86 lexical), which only confirms RRF is fusing
both lists; the real-embedding numbers are the ones that will be recorded here. None are
entered by hand.

## Template

```
### Experiment N — <question>

- Date / git SHA:
- Dataset version / split:
- Fixed config:
- Variants compared:
- Metrics table (Recall@5, MRR, nDCG@10, evidence coverage@5, groundedness,
  hallucination rate, abstention F1, p50/p95 latency, cost/query):
- Conclusion (including where the "better" option is actually worse):
```
