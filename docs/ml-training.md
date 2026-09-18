# ML training — fine-tuned cross-encoder reranker

> Status: **Phase 7, complete.** See `docs/experiments.md` Experiment 3 for the measured
> comparison against the Phase 6 pretrained baseline.

## Goal

Show that the project involves genuine model training, not only API calls: fine-tune a
compact cross-encoder on incident→evidence relevance and measure whether it beats the
pretrained baseline (`cross-encoder/ms-marco-MiniLM-L-6-v2`, Phase 6) on OpsPilot's own
evaluation set. The honest headline result: **at this corpus's scale it doesn't clearly win**
— see "Honest tradeoffs" below. The pipeline itself (dataset construction, seeded reproducible
training, loss curves, checkpoint metadata, drop-in evaluation) is the deliverable.

## Dataset (`ml/build_dataset.py` → `ml/datasets/{train,dev,test}.jsonl`, `split.json`)

Built entirely from the planted ground truth already in `data/generated/` (Phase 2) — no new
synthetic data, no hand-labeling:

- **positive** (query, chunk): for each of a ground-truth case's
  `required_document_source_paths`, the document is chunked with
  `opspilot.ingestion.chunking.chunk_text` using the *same* `chunk_size`/`chunk_overlap` as
  ingestion, and the chunk containing one of the case's `required_evidence_snippets`
  (case-insensitive substring) is the positive passage. (0 of 90 required-document lookups
  needed the single-chunk fallback — every snippet landed inside some chunk.)
- **hard negative**: another document from the *same service*, not required by this case — the
  confusion a reranker actually has to resolve (same service vocabulary, wrong incident). Up to
  5 per case, deterministically sampled (`random.Random(f"{seed}:{case_id}")`).
- **easy negative**: a document from an unrelated service. Up to 2 per case — regularization; a
  pretrained cross-encoder already handles cross-service negatives fine.

Only the 18 *answerable* ground-truth cases are used (the 4 unanswerable cases have no required
evidence to build a positive from).

**Split — by chain, not by case**, so no chunk or near-duplicate query crosses a split
boundary:

| split | chains | cases | pairs (pos/neg) |
|-------|-------:|------:|-----------------|
| train | 8 | 9 | 108 (45 / 63) |
| dev   | 2 (`chain-02-auth-jwks-stale`, `chain-09-inventory-kafka-rebalance`) | 4 | 52 (24 / 28) |
| test  | 2 (`chain-01-payments-auth-timeout`, `chain-06-orders-event-ordering-race`) | 5 | 63 (30 / 33) |

The chain assignment is **hand-picked, not randomly drawn**: with only 12 planted chains, a
random split risks a 1-chain (or 0-multi-hop-chain) test set — too small to say anything.
`dev` picks the training checkpoint (best `eval_loss`); `test`'s 5 cases are used only in
Experiment 3, entirely outside `ml/train_reranker.py` — the fine-tuned model never sees a
`test.jsonl` pair or a `test`-chain query during training.

## Training (`ml/train_reranker.py`, config `ml/configs/reranker.json`)

- **Continues from the pretrained checkpoint** (`cross-encoder/ms-marco-MiniLM-L-6-v2`), not a
  random init. 108 training pairs is nowhere near MS MARCO's ~500K+ — nowhere close to enough
  to learn cross-encoder ranking from scratch, but plausible for nudging an already-strong
  ranker toward OpsPilot's vocabulary (error codes, service names, config keys like
  `PAYMENTS_AUTH_TIMEOUT_MS`). A from-scratch run at this scale would not have been a meaningful
  comparison, so it wasn't run.
- **Library:** `sentence_transformers.cross_encoder` (the sentence-transformers v6
  `CrossEncoderTrainer`, a thin `transformers.Trainer` subclass) +
  `BinaryCrossEntropyLoss` (the dataset is binary-labeled pairs, not ranked triples — BCE over
  a single relevance logit, `num_labels=1`, is the natural loss).
- **Reproducibility:** fixed seed (42) applied to `random`, `numpy`, `torch`, and
  `transformers.set_seed`; full config (base model, LR, batch size, epochs, warmup, max_length)
  logged to `ml/configs/reranker.json` and echoed into the checkpoint's
  `training_metadata.json` alongside the git SHA, dataset sizes, final eval metrics, and
  library versions (torch/sentence-transformers/transformers).
- **Hyperparameters:** `lr=2e-5`, `batch_size=8`, `epochs=6`, `weight_decay=0.01`,
  `warmup_ratio=0.1`, `max_length=256`. `eval_strategy="epoch"` against the dev set;
  `load_best_model_at_end=True` on `eval_loss` (best was epoch 5 of 6: `eval_loss=0.598`,
  `dev_accuracy=0.885`, `dev_f1=0.870`).
- **Loss curve:** `ml/reports/loss_curve.png` (train BCE loss per logging step + dev loss per
  epoch), raw log history in `ml/reports/log_history.json`. Train loss is high-variance
  step-to-step (batch size 8 over 108 examples — a handful of steps per epoch, each easily
  dominated by one hard example) but dev loss trends down through epoch 5 before flattening.
- **Runtime:** ~140 s on CPU for the full 6-epoch run — small enough that no GPU/accelerator
  path was needed.
- **Checkpoint:** `ml/checkpoints/opspilot-ce-v1/` (git-ignored, like `data/generated/` and the
  HF cache — rebuild with `make ml-train`). Saved via `CrossEncoder.save_pretrained`, so it
  loads through the *exact same* Phase 6 seam as the pretrained model —
  `LocalCrossEncoderProvider(model="ml/checkpoints/opspilot-ce-v1")` — no runtime code changed
  for Phase 7; `opspilot.providers.factory.build_rerank_provider` already took an arbitrary
  `reranker_model` string.

## Evaluation

Reused the Phase 4 harness's own functions (`retrieve_candidates`, `rerank_chunks`, the
`opspilot.evaluation.metrics` module) but **not** `opspilot.evaluation.runner` /
`evaluation_runs` — those are scoped to the versioned 22-case dev-split dataset, which is not
chain-split (see `opspilot/evaluation/loader.py`) and therefore isn't safe to score a
chain-fine-tuned reranker against without leaking training chains into the metric.
`ml/eval_experiment3.py` runs the three-way comparison (no reranker / pretrained / fine-tuned)
against only the 5 cases in `ml/build_dataset.py`'s held-out test chains. See
`docs/experiments.md` Experiment 3 for the full numbers and the honest read on them.

## Honest tradeoffs

- **The fine-tune did not beat the pretrained baseline** on the held-out slice — it traded a
  small nDCG@10 gain (0.986 vs 0.978) for a small Recall@8 loss (0.840 vs 0.880), with n=5
  cases too small to call either difference real.
- **The bottleneck is corpus size, not the method.** 12 planted chains (9 train / 2 held out)
  is enough to prove the training *pipeline* works end to end, not enough to detect a
  genuine ranking improvement over a strong pretrained baseline. The roadmap's 35–55-chain
  target would make this comparison statistically meaningful.
- **Maintenance cost is real and mostly not about training.** Retraining is cheap (~2–3 min on
  CPU) and reproducible, but every retrain needs a fresh, honest held-out re-evaluation (never
  reuse dev-tuned numbers as if they were test numbers), and the checkpoint (~90 MB) has to be
  version-pinned and reproduced from a logged config, not just "the file that happens to be in
  `ml/checkpoints/`".
- **Latency is not this experiment's finding to make.** The pretrained-vs-fine-tuned latency
  gap measured in Experiment 3 is almost certainly a process warm-up artifact (whichever model
  runs first in the script absorbs PyTorch/MKL's one-time init cost) — the two checkpoints are
  architecturally identical, so don't cite a real speed difference between them from this run.
