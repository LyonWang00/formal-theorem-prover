#!/usr/bin/env python3
"""Freeze a selected GRPO JSONL into hash-bound generation shard manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from lean_prover.lean_training.evaluation.rollout import read_rollout_records


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--pass-k", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--generation-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.data.open(encoding="utf-8") if line.strip()]
    records = read_rollout_records(a.data, rollout_kind="grpo")
    if len(records) != len(rows):
        raise ValueError(f"runtime record count drift: {len(records)} != {len(rows)}")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate problem IDs")
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    a.output.mkdir(parents=True, exist_ok=False)
    shard_rows = [[] for _ in range(a.shards)]
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        prompt = str(record.prompt)
        prompt_tokens = len(tok(prompt, add_special_tokens=True)["input_ids"])
        cap = min(a.max_new_tokens, a.max_model_len - prompt_tokens)
        if cap <= 0:
            raise ValueError(f"no completion budget for row {index}")
        shard_rows[index % a.shards].append({
            "global_index": index,
            "problem_id": str(record.problem_id),
            "statement_hash": str(record.statement_hash),
            "manifest_statement_hash": str(row["statement_hash"]),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens": prompt_tokens,
            "effective_max_new_tokens": cap,
            "max_new_tokens_ceiling": a.max_new_tokens,
            "max_model_len": a.max_model_len,
            "generation_backend": "vllm",
            "generation_batch_size": a.generation_batch_size,
            "sampling_seed_base": a.seed,
            "temperature": a.temperature,
            "top_p": a.top_p,
        })
    reports = []
    for index, values in enumerate(shard_rows):
        path = a.output / f"shard_{index}.jsonl"
        dump(path, values)
        reports.append({"shard_index": index, "path": path.name, "count": len(values), "sha256": sha(path)})
    report = {
        "schema": "adaptive_e2_generation_shards_v1", "status": "frozen",
        "data": str(a.data), "data_sha256": sha(a.data), "problem_count": len(rows),
        "attempts_per_problem": a.pass_k, "expected_attempt_count": len(rows) * a.pass_k,
        "max_model_len": a.max_model_len, "max_new_tokens_ceiling": a.max_new_tokens,
        "generation_backend": "vllm", "generation_batch_size": a.generation_batch_size,
        "sampling_seed_base": a.seed, "temperature": a.temperature, "top_p": a.top_p,
        "shard_count": a.shards, "shards": reports,
    }
    (a.output / "SHARDS_FROZEN.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
