# Experiments

A log of controlled experiments. Each entry links to the `evaluation_runs` rows that back
it. **No numbers are entered here by hand** — they are copied from harness output.

| # | Question | Phase | Result |
|--:|----------|-------|--------|
| 1 | vector-only vs lexical-only vs hybrid (RRF) retrieval | 5 | hybrid best on nDCG@10 & MRR; lexical ≈ hybrid on recall on this corpus (see below) |
| 2 | hybrid vs hybrid + pretrained cross-encoder reranker | 6 | rerank: Recall@8 0.88 → 0.92, nDCG@10 flat, +830 ms/query — recall gain at a steep CPU-latency cost |
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

> **Reproduced 2026-09-07 at `31563c1`** on a fresh ingest: vector 0.864 / lexical 0.891 /
> hybrid 0.882 Recall@8; nDCG@10 0.970 / 0.973 / 0.988. Run-to-run drift of ~±1 pt on Recall@8
> comes from HNSW's approximate nearest-neighbour search (vector) and uuid tie-breaks between
> equal-`ts_rank_cd` chunks (lexical, and hybrid via it) — the qualitative ordering is stable.

### Experiment 2 — pretrained cross-encoder reranker on top of hybrid

- **Date / git SHA:** 2026-09-07 / `31563c1`
- **Dataset version / split:** `gen2.0.0-seed42` / dev (22 cases)
- **Fixed config:** as Experiment 1, plus `rerank_candidate_k=20`; reranker
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, CPU); retrieval strategy fixed at `hybrid`
- **Variants compared:** `--reranker none` vs `--reranker cross_encoder`
- **Metrics** (`evaluation_runs` rows `2026-09-08T01:30:{47,59}Z`):

  | pipeline | Recall@8 | MRR | nDCG@10 | evidence coverage | mean rerank ms | latency p50 / p95 (ms) |
  |----------|---------:|----:|--------:|------------------:|---------------:|-----------------------:|
  | hybrid                 | 0.882 | 1.000 | **0.988** | 1.000 |   0 |  26 /  116 |
  | hybrid + cross-encoder  | **0.920** | 1.000 | 0.986 | 1.000 | 819 | 858 / 1081 |

- **Conclusion — the reranker buys recall, not ranking, and the bill is latency.**
  Re-scoring a 20-candidate pool and keeping the top 8 pulls a relevant chunk that hybrid
  ranked 9th–20th up into the prompt: **Recall@8 0.88 → 0.92** (+3.8 pts, and the same +3–4 pt
  gain showed on an earlier run, so it's real, not noise). But **nDCG@10 does not improve**
  (0.988 → 0.986 — a rounding-error regression): among the items hybrid already had in the top
  10, RRF's order is as good as the cross-encoder's here. And the cost is brutal —
  **~819 ms/query of rerank compute on CPU**, taking p50 latency from 26 ms to 858 ms (~33×),
  p95 past 1 s. MRR is already 1.0 for both (the first relevant chunk is always rank 1), so the
  reranker can't help there.
- **Recommendation.** Keep `OPSPILOT_RERANKER=none` as the default. The recall gain is
  worth having, but not at 33× latency for a portfolio demo — revisit when (a) the corpus is
  large enough that top-8 recall genuinely hurts, (b) it runs on GPU or with a smaller
  candidate pool / batched inference, (c) the **fine-tuned** cross-encoder (Phase 7, Experiment
  3) can be compared against this off-the-shelf baseline, and (d) a real-LLM run can answer the
  question this experiment can't: does the extra recall actually produce better, better-grounded
  answers, or just more context?

**Reproduce:** as Experiment 1, then `make eval-experiment-rerank` (hybrid with vs without the
cross-encoder). First run downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB).

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
