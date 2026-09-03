# Experiments

A log of controlled experiments. Each entry links to the `evaluation_runs` rows that back
it. **No numbers are entered here by hand** — they are copied from harness output.

| # | Question | Phase | Result |
|--:|----------|-------|--------|
| 1 | vector-only vs lexical-only vs hybrid (RRF) retrieval | 5 | pending |
| 2 | hybrid vs hybrid + pretrained cross-encoder reranker | 6 | pending |
| 3 | pretrained vs fine-tuned cross-encoder reranker | 7 | pending |
| 4 | one-shot RAG vs deterministic retrieve→verify→answer vs LangGraph agent | 9 | pending |

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
