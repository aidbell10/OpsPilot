"""Fine-tune a compact cross-encoder reranker on OpsPilot's planted ground truth.

Continues training from the same pretrained checkpoint Experiment 2 (Phase 6)
measured (``cross-encoder/ms-marco-MiniLM-L-6-v2``) rather than a random init:
the labeled set built by ``ml/build_dataset.py`` has ~108 training pairs from
9 planted incident chains — nowhere near enough to learn passage ranking from
scratch, but plausible for nudging an already-strong ranker toward OpsPilot's
vocabulary (error codes, service names, config keys). See
``docs/ml-training.md`` for the honest read on what this can and can't show
at this scale.

Uses ``sentence_transformers.cross_encoder``'s ``CrossEncoderTrainer`` (a thin
``transformers.Trainer`` subclass) with ``BinaryCrossEntropyLoss`` — the
dataset is binary-labeled (query, passage) pairs, not ranked triples, so BCE
over a relevance logit is the natural loss for a ``num_labels=1`` cross-encoder.

Usage: ``uv run python ../ml/train_reranker.py [--config ml/configs/reranker.json]``
(run from ``backend/`` so the ``ml`` extra + editable ``opspilot`` install are on
the path — see ``make ml-train``).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ML_DIR / "configs" / "reranker.json"
DATASET_DIR = ML_DIR / "datasets"
CHECKPOINT_DIR = ML_DIR / "checkpoints"
REPORT_DIR = ML_DIR / "reports"


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _set_all_seeds(seed: int) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def _plot_loss_curve(log_history: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss = [e["loss"] for e in log_history if "loss" in e]
    eval_epochs = [e["epoch"] for e in log_history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(train_steps, train_loss, marker="o", label="train loss (per logging step)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train BCE loss")

    if eval_loss:
        ax2 = ax1.twiny()
        ax2.plot(
            eval_epochs, eval_loss, marker="s", color="tab:orange", label="dev loss (per epoch)"
        )
        ax2.set_xlabel("epoch")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_title("Cross-encoder fine-tune: loss curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-name", type=str, default="opspilot-ce-v1")
    args = parser.parse_args(argv)

    config: dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    _set_all_seeds(seed)

    train_rows = _load_jsonl(DATASET_DIR / "train.jsonl")
    dev_rows = _load_jsonl(DATASET_DIR / "dev.jsonl")
    if not train_rows or not dev_rows:
        print("No dataset found — run `make ml-dataset` first.", file=sys.stderr)
        return 1

    from datasets import Dataset  # type: ignore[attr-defined]
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    train_dataset = Dataset.from_list(
        [
            {"query": r["query"], "passage": r["passage"], "label": float(r["label"])}
            for r in train_rows
        ]
    )
    dev_dataset = Dataset.from_list(
        [
            {"query": r["query"], "passage": r["passage"], "label": float(r["label"])}
            for r in dev_rows
        ]
    )

    model = CrossEncoder(config["base_model"], num_labels=1, max_length=int(config["max_length"]))
    loss = BinaryCrossEntropyLoss(model)
    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=[[r["query"], r["passage"]] for r in dev_rows],
        labels=[int(r["label"]) for r in dev_rows],
        name="dev",
    )

    output_dir = CHECKPOINT_DIR / args.output_name
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_train_batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1,
        seed=seed,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    start = datetime.now(UTC)
    trainer.train()
    end = datetime.now(UTC)

    final_eval = trainer.evaluate()
    trainer.save_model(str(output_dir))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
    )
    _plot_loss_curve(trainer.state.log_history, REPORT_DIR / "loss_curve.png")

    import sentence_transformers
    import torch
    import transformers

    metadata = {
        "output_name": args.output_name,
        "output_dir": str(output_dir),
        "config": config,
        "git_sha": _git_sha(),
        "started_at": start.isoformat(),
        "finished_at": end.isoformat(),
        "n_train_pairs": len(train_rows),
        "n_dev_pairs": len(dev_rows),
        "final_eval_metrics": final_eval,
        "versions": {
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__,
        },
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"saved fine-tuned checkpoint -> {output_dir}")
    print(f"final eval metrics: {final_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
