# Experiments

A log of controlled experiments. Each entry links to the `evaluation_runs` rows that back
it. **No numbers are entered here by hand** — they are copied from harness output.

| # | Question | Phase | Result |
|--:|----------|-------|--------|
| 1 | vector-only vs lexical-only vs hybrid (RRF) retrieval | 5 | hybrid best on nDCG@10 & MRR; lexical ≈ hybrid on recall on this corpus (see below) |
| 2 | hybrid vs hybrid + pretrained cross-encoder reranker | 6 | pending |
| 3 | pretrained vs fine-tuned cross-encoder reranker | 7 | pending |
| 4 | one-shot RAG vs deterministic retrieve→verify→answer vs LangGraph agent | 9 | pending |

### Experiment 1 — vector vs lexical vs hybrid retrieval

- **Date / git SHA:** 2026-09-07 / `82112f0`
- **Dataset version / split:** `gen2.0.0-seed42` / dev (22 cases: 18 answerable, 4 unanswerable)
- **Fixed config:** embeddings `BAAI/bge-small-en-v1.5` (384-d, local); `chunk_size=512`,
  `chunk_overlap=64`; `retrieval_top_k=8`; hybrid `candidate_k=30`, `rrf_k=60`; LLM = `fake`
  (this experiment measures retrieval only — generation metrics under the fake LLM are the
  documented degenerate case and are ignored here)
- **Variants compared:** `--strategy vector` | `lexical` | `hybrid`
- **Metrics** (`evaluation_runs` rows `2026-09-08T01:15:{05,18,30}Z`, copied from
  `opspilot.evaluation report` + the run's `aggregate_metrics`):

  | strategy | Recall@8 | MRR | nDCG@10 | evidence coverage | latency p50 / p95 (ms) |
  |----------|---------:|----:|--------:|------------------:|-----------------------:|
  | vector      | 0.864 | 0.977 | 0.970 | 1.000 | 22.2 / 107.9 |
  | lexical     | 0.882 | 1.000 | 0.973 | 1.000 | 24.1 /  65.0 |
  | hybrid (RRF)| 0.882 | 1.000 | **0.987** | 1.000 | 25.8 / 115.4 |

- **Conclusion.** Hybrid wins on ranking quality (nDCG@10 0.987, clear of both arms) and ties
  lexical for the best Recall@8 and MRR — RRF keeps lexical's exact-match wins while letting
  the vector arm reorder the rest. But **lexical alone essentially matches hybrid on recall/MRR
  here**, and beats plain vector: the planted ground truth is built around exact error codes,
  function names, and version strings (`PAY-50231`, `authorize_payment`, `v5.0.6`), which is
  lexical FTS's home turf. On a corpus with more paraphrase and synonymy the vector arm should
  pull ahead and hybrid's margin over lexical should widen — this result is specific to a
  small, lexically-loaded synthetic corpus and shouldn't be read as "lexical ≈ hybrid" in
  general. Hybrid also costs ~15% more p50 latency (two index scans + fusion) for the nDCG
  gain. Recommendation: keep `hybrid` as the default; revisit once the corpus grows and once
  the reranker (Phase 6) sits on top.

**Reproduce:**

```bash
make db-up
cd backend
uv sync --extra ml                      # torch + sentence-transformers (once)
export OPSPILOT_EMBEDDING_PROVIDER=local
uv run python -m opspilot.ingestion      # first run downloads BAAI/bge-small-en-v1.5
uv run python -m opspilot.evaluation load-cases
cd .. && make eval-experiment            # vector, lexical, hybrid, then the report
```

Retrieval metrics don't touch the LLM, so `OPSPILOT_LLM_PROVIDER=fake` is fine and free. The
**embedding** provider does matter: `fake` embeddings are token-hash vectors, not a real
semantic retriever — never publish a fake-embedding vector arm as a result.

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
