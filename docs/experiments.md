# Experiments

A log of controlled experiments. Each entry links to the `evaluation_runs` rows that back
it. **No numbers are entered here by hand** — they are copied from harness output.

| # | Question | Phase | Result |
|--:|----------|-------|--------|
| 1 | vector-only vs lexical-only vs hybrid (RRF) retrieval | 5 | hybrid best on nDCG@10 & MRR; lexical ≈ hybrid on recall on this corpus (see below) |
| 2 | hybrid vs hybrid + pretrained cross-encoder reranker | 6 | rerank: Recall@8 0.88 → 0.92, nDCG@10 flat, +830 ms/query — recall gain at a steep CPU-latency cost |
| 3 | no reranker vs pretrained vs fine-tuned cross-encoder | 7 | fine-tune ≈ pretrained on this 5-case held-out slice (nDCG@10 0.986 vs 0.978); too little data to claim a win — see below |
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

### Experiment 3 — pretrained vs fine-tuned cross-encoder reranker

- **Date / git SHA:** 2026-09-18 / `dcef229`
- **Not an `evaluation_runs` entry**, unlike Experiments 1–2: the 22-case dev-split dataset is
  not chain-split (see `opspilot/evaluation/loader.py`'s docstring — splitting it today would
  leak chain siblings across dev/test), and the fine-tuned model in this experiment was trained
  on 9 of those 12 chains, so scoring it against all 22 cases would leak training data into the
  test metric. Instead this experiment holds out 2 chains **entirely** from `ml/build_dataset.py`
  (5 cases: `chain-01-payments-auth-timeout`, `chain-06-orders-event-ordering-race`) and
  evaluates all three configs only on those, via `ml/eval_experiment3.py` — a standalone script,
  but it calls the exact same `retrieve_candidates` / `rerank_chunks` / metrics functions the
  harness does. Numbers below are copied verbatim from `ml/reports/experiment3_results.json`.
- **Dataset:** `ml/datasets/{train,dev}.jsonl` — 108 train pairs / 9 chains, 52 dev pairs / 2
  chains (45/24 positive, chunked exactly as ingestion chunks, hard negatives = other documents
  from the same service, easy negatives = documents from unrelated services). See
  `docs/ml-training.md` for the full dataset/training write-up.
- **Fixed config:** as Experiment 2 (hybrid retrieval, `rerank_candidate_k=20`,
  `BAAI/bge-small-en-v1.5` embeddings); reranker base model for both B and C is
  `cross-encoder/ms-marco-MiniLM-L-6-v2` — C continues training from B's weights (see
  `docs/ml-training.md` for why train-from-scratch wasn't viable at this dataset size)
- **Variants compared:** A. no reranker | B. pretrained cross-encoder | C. fine-tuned
  cross-encoder (`ml/checkpoints/opspilot-ce-v1`)
- **Metrics** (n=5 held-out cases; identical across two repeated runs — HNSW is exact enough at
  this corpus size (108 chunks) that these numbers show no run-to-run drift, unlike the 22-case
  runs in Experiments 1–2):

  | config | Recall@8 | MRR | nDCG@10 | rerank p50 / p95 (ms) |
  |--------|---------:|----:|--------:|-----------------------:|
  | A. no reranker | 0.760 | 1.000 | 0.984 | 0 / 0 |
  | B. pretrained cross-encoder | **0.880** | 1.000 | 0.978 | 1022 / 1813 |
  | C. fine-tuned cross-encoder | 0.840 | 1.000 | **0.986** | 448 / 548 |

- **Conclusion — fine-tuning did not beat the pretrained baseline at this scale, and that's the
  honest result.** C edges out B on nDCG@10 (0.986 vs 0.978) but loses on Recall@8 (0.840 vs
  0.880 — one fewer required-document hit across 5 cases); B and C both clearly beat no
  reranker on recall (confirming Experiment 2's finding: reranking buys recall, not ranking).
  With n=5 cases, a one-document swing moves Recall@8 by 0.04–0.08 — **these deltas are not
  statistically distinguishable**; the fair reading is "fine-tuning on 108 pairs from 9 chains
  neither clearly helped nor hurt." The latency gap (C ~2× faster than B) is almost certainly a
  **measurement artifact, not a real difference**: B always loads and runs first in the process
  in this script, absorbing PyTorch/MKL's one-time thread-pool and kernel warm-up cost — the two
  checkpoints are architecturally identical (same MiniLM-L6 size, same tokenizer, same
  `max_length`), so a real 2× latency difference between them has no mechanism. Don't cite the
  latency numbers here as a finding.
- **Why fine-tune from the pretrained checkpoint instead of from scratch:** 108 training pairs
  from 9 incident chains is nowhere near enough to learn cross-encoder ranking from a random
  init (MS MARCO trains on ~500K+ labeled pairs). Continuing training from an already-strong
  ranker is domain *adaptation* (nudging it toward OpsPilot's error codes / service names /
  config keys), a realistic ask at this data scale — training from scratch would not have been
  a meaningful comparison.
- **What this experiment actually demonstrates:** a working, reproducible fine-tuning pipeline
  (seeded, versioned config, logged loss curve, checkpoint metadata, drop-in via the existing
  `RerankProvider` seam) — not a proven quality win. The planted-chain corpus (12 chains) is the
  bottleneck: 9 training chains / 2 held-out chains is too small to detect anything but a large
  effect. **Recommendation:** keep `cross-encoder/ms-marco-MiniLM-L-6-v2` (pretrained) as the
  default reranker choice (also still `OPSPILOT_RERANKER=none` per Experiment 2); revisit
  fine-tuning if the generator grows the corpus toward the ~35–55 chain target, which would let
  train/held-out actually separate signal from noise.

**Reproduce:**

```bash
cd backend
uv sync --group dev --extra ml            # adds datasets, matplotlib, accelerate
uv run python ../ml/build_dataset.py      # -> ml/datasets/{train,dev,test}.jsonl, split.json
uv run python ../ml/train_reranker.py     # -> ml/checkpoints/opspilot-ce-v1 (~2-3 min on CPU)
make db-up && cd backend && uv run python -m opspilot.ingestion
uv run python ../ml/eval_experiment3.py   # -> ml/reports/experiment3_results.json
```

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
