# ml/

PyTorch cross-encoder reranker training (**Phase 7**, complete).

| Path | Contents |
|------|----------|
| `build_dataset.py` | builds `datasets/{train,dev,test}.jsonl` + `split.json` from `data/generated/` |
| `train_reranker.py` | reproducible training entrypoint (fixed seeds, logged config) |
| `eval_experiment3.py` | A/B/C comparison (no reranker / pretrained / fine-tuned) on the held-out test chains |
| `datasets/` | labeled (query, passage, label) pairs — **git-ignored**, rebuild with `make ml-dataset` |
| `configs/` | training configs (base model, LR, batch size, epochs) |
| `checkpoints/` | saved models + metadata — **git-ignored**, rebuild with `make ml-train` |
| `reports/` | loss curve, log history, Experiment 3 results — **git-ignored** |

```bash
make ml-dataset      # data/generated -> ml/datasets/
make ml-train         # ml/datasets -> ml/checkpoints/opspilot-ce-v1 (~2-3 min CPU)
make ml-experiment3   # A/B/C comparison against the held-out test chains (needs db-up + ingest)
make ml-test / ml-lint   # unit tests + ruff/mypy for this directory
```

See [`docs/ml-training.md`](../docs/ml-training.md) for the dataset methodology, training
config, and the honest read on the results; [`docs/experiments.md`](../docs/experiments.md)
Experiment 3 for the measured numbers.
