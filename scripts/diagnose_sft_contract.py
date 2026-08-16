#!/usr/bin/env python3
"""Inspect the installed TRL prompt/completion label contract without loading a model."""

from __future__ import annotations

import argparse
import json
import random
from importlib.metadata import version
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def percentile(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]


def distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": round(sum(values) / len(values), 4),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(Path(args.train_file).expanduser())
    if not rows:
        raise ValueError("training dataset is empty")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rng = random.Random(args.seed)
    sample_indices = sorted(rng.sample(range(len(rows)), min(args.samples, len(rows))))
    samples: list[dict[str, Any]] = []
    total_lengths: list[int] = []
    completion_lengths: list[int] = []
    truncated_examples = 0
    all_masked_examples = 0
    prefix_mismatches = 0
    eos_missing = 0
    non_ignored_label_tokens_total = 0
    unattested = 0
    for index, row in enumerate(rows):
        prompt = str(row[args.prompt_field])
        original_completion = str(row[args.completion_field])
        completion = original_completion
        if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        prefix_matches = full_ids[: len(prompt_ids)] == prompt_ids
        prefix_mismatches += int(not prefix_matches)
        completion_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        raw_total_length = len(full_ids)
        total_lengths.append(raw_total_length)
        completion_lengths.append(sum(completion_mask))
        if raw_total_length > args.max_seq_length:
            truncated_examples += 1
        input_ids = full_ids[: args.max_seq_length]
        completion_mask = completion_mask[: args.max_seq_length]
        labels = [token_id if mask else -100 for token_id, mask in zip(input_ids, completion_mask, strict=True)]
        valid_indices = [position for position, label in enumerate(labels) if label != -100]
        all_masked_examples += int(not valid_indices)
        non_ignored_label_tokens_total += len(valid_indices)
        ends_with_eos = bool(
            valid_indices
            and tokenizer.eos_token_id is not None
            and labels[valid_indices[-1]] == tokenizer.eos_token_id
        )
        eos_missing += int(not ends_with_eos)
        unattested += int(row.get("pantograph_verified") is not True)
        if index in sample_indices:
            samples.append(
                {
                    "index": index,
                    "id": row.get("id"),
                    "statement": row.get("lean_statement"),
                    "proof": original_completion,
                    "formatted_text": prompt + completion,
                    "decoded": tokenizer.decode(input_ids, skip_special_tokens=False),
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": [1] * len(input_ids),
                    "completion_token_interval": (
                        [valid_indices[0], valid_indices[-1] + 1]
                        if valid_indices
                        else None
                    ),
                    "masked_ratio": round(1 - len(valid_indices) / max(1, len(labels)), 6),
                    "valid_label_tokens": len(valid_indices),
                    "ends_with_eos": ends_with_eos,
                    "prompt_prefix_matches_full": prefix_matches,
                }
            )
    return {
        "trl_version": version("trl"),
        "transformers_version": version("transformers"),
        "completion_only_loss": True,
        "train_file": str(Path(args.train_file).expanduser().resolve()),
        "tokenizer": str(Path(args.tokenizer).expanduser().resolve()),
        "prompt_format": "plain_text_lean_sections_v1",
        "num_examples": len(rows),
        "max_seq_length": args.max_seq_length,
        "mean_total_tokens": distribution(total_lengths)["mean"],
        "p50_total_tokens": distribution(total_lengths)["p50"],
        "p90_total_tokens": distribution(total_lengths)["p90"],
        "p95_total_tokens": distribution(total_lengths)["p95"],
        "p99_total_tokens": distribution(total_lengths)["p99"],
        "max_total_tokens": distribution(total_lengths)["max"],
        "truncated_examples": truncated_examples,
        "truncated_ratio": truncated_examples / len(rows),
        "mean_completion_tokens": distribution(completion_lengths)["mean"],
        "non_ignored_label_tokens_total": non_ignored_label_tokens_total,
        "all_labels_ignored_examples": all_masked_examples,
        "prompt_prefix_mismatches": prefix_mismatches,
        "examples_missing_labeled_eos": eos_missing,
        "examples_without_pantograph_attestation": unattested,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template": tokenizer.chat_template,
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--completion_field", default="completion")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = diagnose(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
