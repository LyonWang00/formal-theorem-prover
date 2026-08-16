#!/usr/bin/env python3
"""Compute reference-completion proxy scores for the 2932 missing M0 rows."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.prepare_stage2_sft_phase0 import (
    ATTEMPTS,
    GENERATIONS,
    OLD_LD,
    OLD_WB,
    domain,
    primary_tactic,
    proof,
    qualified_theorem,
    read_json,
    read_jsonl,
    record_id,
    source_kind,
    statement_tokens,
    theorem_group,
)
from scripts.prepare_task0b_light import CONTRACT_DIR, MERGED, OUTPUT, canonical_hash


def existing_result_ids(project: Path, rows: list[dict[str, Any]]) -> set[str]:
    import re

    normalized = lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
    prompt_map: dict[str, str | None] = {}
    for row in rows:
        key = normalized(row.get("prompt"))
        current = prompt_map.get(key)
        row_id = record_id(row)
        prompt_map[key] = row_id if current in (None, row_id) else ""
    statement_map = {
        str(generation.get("statement_id") or ""): prompt_map.get(
            normalized(generation.get("prompt"))
        )
        for generation in read_jsonl(project / GENERATIONS)
    }
    return {
        str(statement_map.get(str(attempt.get("problem_id") or "")) or "")
        for attempt in read_jsonl(project / ATTEMPTS)
        if statement_map.get(str(attempt.get("problem_id") or ""))
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_one(
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    eos_token_id: int,
) -> tuple[float, float, int, int]:
    prompt = str(row.get("prompt") or "")
    completion = proof(row)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    target_ids = completion_ids + [eos_token_id]
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        shift_logits = output.logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)
        losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shift_labels)
        loss = float(losses[mask].mean().cpu())
        accuracy = float(
            shift_logits.argmax(dim=-1)[mask].eq(shift_labels[mask]).float().mean().cpu()
        )
    del output, shift_logits, shift_labels, losses, mask, input_ids, labels, attention_mask
    return loss, accuracy, len(completion_ids), len(target_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    project = args.project.resolve()
    contract = read_json(project / CONTRACT_DIR / "generation_contract.json")
    hash_payload = dict(contract)
    declared_hash = hash_payload.pop("generation_config_sha256")
    resolved_hash = canonical_hash(hash_payload)
    if resolved_hash != declared_hash:
        raise RuntimeError("generation contract hash mismatch before proxy scoring")
    print(f"[generation-contract] resolved_sha256={resolved_hash}", flush=True)

    rows = read_jsonl(project / OLD_WB) + read_jsonl(project / OLD_LD)
    existing = existing_result_ids(project, rows)
    missing = [row for row in rows if record_id(row) not in existing]
    if len(existing) != 68 or len(missing) != 2932:
        raise RuntimeError(f"expected 68 existing and 2932 missing, got {len(existing)}/{len(missing)}")
    missing.sort(key=lambda row: (int(row.get("total_tokens") or 0), record_id(row)))
    output_path = project / OUTPUT / "proxy_scores.jsonl"
    completed = {
        row["record_id"] for row in read_jsonl(output_path)
    } if output_path.is_file() else set()
    pending = [row for row in missing if record_id(row) not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print(json.dumps({"proxy_scores": "already_complete", "rows": len(completed)}, indent=2))
        return

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str((project / MERGED).resolve()),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str((project / MERGED).resolve()), trust_remote_code=True
    )
    start = time.monotonic()
    for index, row in enumerate(pending, start=1):
        loss, accuracy, completion_length, labels_with_eos = score_one(
            model, tokenizer, row, int(contract["eos_token_id"])
        )
        append_jsonl(
            output_path,
            {
                "record_id": record_id(row),
                "theorem_group_id": theorem_group(row),
                "qualified_theorem": qualified_theorem(row),
                "source": source_kind(row),
                "domain": domain(row),
                "primary_tactic": primary_tactic(row),
                "completion_loss": loss,
                "token_accuracy": accuracy,
                "proof_token_length": completion_length,
                "labels_with_eos": labels_with_eos,
                "statement_tokens": statement_tokens(row),
                "total_tokens": int(row.get("total_tokens") or 0),
                "prompt_sha256": canonical_hash(str(row.get("prompt") or "")),
                "generation_contract_sha256": resolved_hash,
                "proxy_model": str((project / MERGED).resolve()),
            },
        )
        if index % 25 == 0 or index == len(pending):
            elapsed = max(time.monotonic() - start, 1e-6)
            print(
                json.dumps(
                    {
                        "scored_this_run": index,
                        "pending_this_run": len(pending),
                        "rows_per_second": round(index / elapsed, 3),
                        "cuda_allocated_mib": round(
                            torch.cuda.memory_allocated() / 2**20, 2
                        )
                        if torch.cuda.is_available()
                        else 0.0,
                    }
                ),
                flush=True,
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_rows = read_jsonl(output_path)
    print(json.dumps({"proxy_scores": "complete", "rows": len(final_rows)}, indent=2))


if __name__ == "__main__":
    main()
