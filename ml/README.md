# ml/

PyTorch cross-encoder reranker training (**Phase 7**).

| Path | Contents |
|------|----------|
| `train_reranker.py` | reproducible training entrypoint (fixed seeds, logged config) |
| `datasets/` | labeled (query, positive, hard-negative) pairs built from planted ground truth |
| `configs/` | training configs (base model, LR, batch size, epochs, negative ratio) |
| `checkpoints/` | saved models + metadata — **git-ignored** |
| `reports/` | loss curves and the A/B/C comparison write-up |

See [`docs/ml-training.md`](../docs/ml-training.md).
