# ML training — fine-tuned cross-encoder reranker

> Status: **Phase 7**. Placeholder; filled in with the real training run.

## Goal

Show that the project involves genuine model training, not only API calls: fine-tune a
compact cross-encoder on incident→evidence relevance and measure whether it beats the
pretrained baseline on OpsPilot's own evaluation set.

## Dataset

Built from the planted ground truth in the synthetic corpus:

- **positive**: (incident query, relevant evidence chunk)
- **hard negative**: (incident query, plausible-but-wrong chunk) — e.g. same service, same
  error family, different root cause; or the right document but the wrong section.

Split: train / dev / test, by incident (no chunk leakage across splits).

## Training

- PyTorch + Sentence-Transformers / Hugging Face `CrossEncoder`.
- Reproducible script: fixed seeds, logged config (base model, LR, batch size, epochs,
  negative ratio), saved checkpoint + metadata, loss curves.
- Evaluation script reused from the main harness.

## Comparison

| Config | Recall@K | MRR | nDCG@10 | rerank latency (p50/p95) |
|--------|---------:|----:|--------:|-------------------------:|
| A. hybrid, no reranker | — | — | — | — |
| B. hybrid + pretrained cross-encoder | — | — | — | — |
| C. hybrid + fine-tuned cross-encoder | — | — | — | — |

Written up with the tradeoffs it introduces (latency, model size, maintenance).
