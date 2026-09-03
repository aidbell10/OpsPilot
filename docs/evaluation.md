# Evaluation methodology

> Status: harness lands in **Phase 4**. This document defines the design up front so the
> data model (`evaluation_cases`, `evaluation_runs`, `evaluation_results`) is stable.

## Dataset

A versioned dataset (`dataset_version` on every row). Target composition (~140 cases):

| Category | Count | Purpose |
|----------|------:|---------|
| straightforward answerable | ~70 | baseline correctness |
| multi-source / multi-hop | ~30 | evidence must be combined across documents |
| unanswerable | ~20 | abstention must fire |
| adversarial / noisy | ~20 | prompt injection, conflicting/stale evidence |

Split: ~90 development, ~50 held-out test. **The test set is never tuned against.**

Each case:

```
case_id, query, answerable, service,
required_document_ids, required_evidence_ids,
expected_root_cause, acceptable_actions, forbidden_claims,
difficulty, category
```

Ground truth is *engineered into the synthetic corpus* (Phase 2): a deployment changes a
function → a later incident → a log line with that function's error code → a related
historical incident → a troubleshooting runbook. The chain gives known-correct evidence ids.

## Metrics

**Retrieval** — Recall@K, evidence coverage@K, MRR, nDCG@10
**Generation** — groundedness, hallucination rate, citation precision
**Reliability** — abstention precision, abstention recall, abstention F1
**Performance** — retrieval / reranking / generation / total latency; p50, p95
**Cost** — embedding token usage, LLM token usage, estimated variable cost per query

## Run record

Every run persists: git SHA (when available), LLM model, embedding model, chunk size, chunk
overlap, top-K, retrieval strategy, reranker, prompt version, dataset version — plus all
aggregate metrics. This makes any two system versions directly comparable.

## Reports

A comparison report renders metrics across runs (vector baseline → hybrid → reranked →
fine-tuned reranker → agent). Reported numbers are computed from real system output only.
