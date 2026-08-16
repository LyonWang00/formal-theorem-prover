"""Run the fixed Round-1 post-training evaluations as separate GPU processes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lean_prover.lean_training.expert_iteration.config import (
    load_expert_iteration_config,
)
from lean_prover.lean_training.expert_iteration.evaluation_adapter import (
    BenchmarkPipelineAdapter,
)
from lean_prover.lean_training.expert_iteration.utils import write_json_atomic
from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval-loss")
    eval_parser.add_argument("--model", default=None)
    eval_parser.add_argument("--adapter", default=None)
    eval_parser.add_argument("--train-file", default=None)
    eval_parser.add_argument("--output", required=True)

    replay_parser = subparsers.add_parser("discovery-replay")
    replay_parser.add_argument("--model", default=None)
    replay_parser.add_argument("--adapter", default=None)
    replay_parser.add_argument("--dataset", required=True)
    replay_parser.add_argument("--output", required=True)
    replay_parser.add_argument("--seed", type=int, default=4401)
    replay_parser.add_argument("--samples-per-statement", type=int, default=4)
    return parser.parse_args()


def evaluate_loss(args: argparse.Namespace) -> dict[str, object]:
    config = load_expert_iteration_config(args.config)
    training = config.training
    model = args.model or config.initial_model.merged_sft0_path
    if not model:
        raise ValueError("an M0 model path is required")
    output = Path(args.output)
    trainer_config = SFTTrainConfig(
        model_name_or_path=model,
        adapter_path=args.adapter,
        train_file=args.train_file or config.data.train_path,
        validation_file=config.data.eval_path,
        output_dir=str(output / "trainer_runtime"),
        max_seq_length=training.max_seq_length,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_validation_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        num_train_epochs=training.num_train_epochs,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        gradient_checkpointing=False,
        optim=training.optimizer,
        lora_r=training.lora_rank,
        lora_alpha=training.lora_alpha,
        lora_dropout=training.lora_dropout,
        seed=config.seed,
        data_seed=config.seed,
    )
    started = time.monotonic()
    trainer, _ = build_trainer(trainer_config)
    raw_metrics = trainer.evaluate()
    metrics = {
        "model": model,
        "adapter": args.adapter,
        "eval_size": len(trainer.eval_dataset),
        "eval_loss": raw_metrics.get("eval_loss"),
        "eval_token_accuracy": raw_metrics.get(
            "eval_mean_token_accuracy", raw_metrics.get("eval_token_accuracy")
        ),
        "eval_runtime": raw_metrics.get("eval_runtime"),
        "wall_seconds": round(time.monotonic() - started, 4),
    }
    write_json_atomic(output / "eval_metrics.json", metrics)
    return metrics


def discovery_replay(args: argparse.Namespace) -> dict[str, object]:
    config = load_expert_iteration_config(args.config)
    model = args.model or config.initial_model.merged_sft0_path
    if not model:
        raise ValueError("an M0 model path is required")
    return BenchmarkPipelineAdapter(config).run(
        role="discovery_replay",
        dataset_path=args.dataset,
        base_model=model,
        adapter_path=args.adapter,
        output_dir=args.output,
        pass_k=[1, args.samples_per_statement],
        samples_per_statement=args.samples_per_statement,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    if args.command == "eval-loss":
        result = evaluate_loss(args)
    else:
        result = discovery_replay(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
