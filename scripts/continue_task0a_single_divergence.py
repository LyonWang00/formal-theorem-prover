#!/usr/bin/env python3
"""Resume Task 0A diagnostics in isolated, one-model phases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.diagnose_task0a_single_divergence import (
    ADAPTER_REL,
    BASE_FP32_REL,
    BASE_REL,
    MERGED_BF16_REL,
    MERGED_FP32_REL,
    REPAIR_REL,
    STATEMENT_ID,
    common_prefix,
    free_model,
    load_model,
    read_jsonl,
    repeat_greedy_generation,
    run_forward,
    token_info,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--phase", choices=("bf16", "fp32-unmerged", "fp32-merged"), required=True
    )
    args = parser.parse_args()
    project = args.project.resolve()
    repair = project / REPAIR_REL
    partial = repair / "task0a_root_cause_diagnostics.partial.json"
    results = json.loads(partial.read_text(encoding="utf-8"))

    left_row = next(
        row
        for row in read_jsonl(repair / "results/unmerged/generations.jsonl")
        if row["statement_id"] == STATEMENT_ID
    )
    right_row = next(
        row
        for row in read_jsonl(repair / "results/merged/generations.jsonl")
        if row["statement_id"] == STATEMENT_ID
    )
    prompt = left_row["prompt"]
    base = project / BASE_REL
    adapter = project / ADAPTER_REL
    merged_bf16 = project / MERGED_BF16_REL
    base_fp32 = project / BASE_FP32_REL
    merged_fp32 = project / MERGED_FP32_REL

    base_tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    merged_tokenizer = AutoTokenizer.from_pretrained(
        str(merged_bf16), trust_remote_code=True
    )
    for tokenizer in (base_tokenizer, merged_tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    text_fork = results["audit"]["fork"]
    known_left = int(text_fork["unmerged_next"]["id"])
    known_right = int(text_fork["merged_next"]["id"])

    if args.phase == "bf16":
        results["bf16"] = {}
        for side, model_path, adapter_path, tokenizer in (
            ("unmerged", base, adapter, base_tokenizer),
            ("merged", merged_bf16, None, merged_tokenizer),
        ):
            print(f"Loading BF16 {side}", flush=True)
            model = load_model(
                model_path,
                dtype=torch.bfloat16,
                adapter_path=adapter_path,
                cpu_only=False,
            )
            determinism = repeat_greedy_generation(
                model,
                tokenizer,
                prompt,
                repeats=2,
                left_target=known_left,
                right_target=known_right,
            )
            results["bf16"][side] = {"determinism": determinism}
            free_model(model)
            model = None
            write_json(partial, results)

        left_ids = results["bf16"]["unmerged"]["determinism"][
            "token_sequences"
        ][0]
        right_ids = results["bf16"]["merged"]["determinism"]["token_sequences"][0]
        fork_index = common_prefix(left_ids, right_ids)
        left_target = int(left_ids[fork_index])
        right_target = int(right_ids[fork_index])
        results["audit"]["fork_actual_generation"] = {
            "completion_token_index_zero_based": fork_index,
            "generated_token_number_one_based": fork_index + 1,
            "prompt_token_count": len(
                results["audit"]["tokenizer_contracts"]["base"]["prompt_ids"]
            ),
            "common_prefix_ids": left_ids[:fork_index],
            "common_prefix_text": base_tokenizer.decode(
                left_ids[:fork_index], skip_special_tokens=False
            ),
            "unmerged_next": token_info(base_tokenizer, left_target),
            "merged_next": token_info(base_tokenizer, right_target),
        }
        for side, frozen_row in (("unmerged", left_row), ("merged", right_row)):
            determinism = results["bf16"][side]["determinism"]
            scores = determinism.pop("first_repeat_step_scores")
            results["bf16"][side]["fork_logits_actual"] = scores[fork_index]
            replay = determinism["completions"][0]
            frozen = frozen_row["raw_output"]
            results["bf16"][side]["frozen_output_reproduced"] = replay == frozen
            results["bf16"][side]["frozen_output_sha256"] = hashlib.sha256(
                frozen.encode("utf-8")
            ).hexdigest()
        write_json(partial, results)
        print(json.dumps(results["audit"]["fork_actual_generation"], ensure_ascii=False, indent=2))
        return

    actual_fork = results["audit"]["fork_actual_generation"]
    common_ids = [int(value) for value in actual_fork["common_prefix_ids"]]
    left_target = int(actual_fork["unmerged_next"]["id"])
    right_target = int(actual_fork["merged_next"]["id"])
    if args.phase == "fp32-unmerged":
        side = "unmerged"
        model_path = base_fp32
        adapter_path = adapter
        tokenizer = base_tokenizer
    else:
        side = "merged"
        model_path = merged_fp32
        adapter_path = None
        tokenizer = merged_tokenizer
    print(f"Loading FP32 {side} for one forward only", flush=True)
    model = load_model(
        model_path,
        dtype=torch.float32,
        adapter_path=adapter_path,
        cpu_only=True,
    )
    results.setdefault("fp32_single_step", {})[side] = run_forward(
        model,
        tokenizer,
        prompt,
        common_ids,
        left_target=left_target,
        right_target=right_target,
    )
    free_model(model)
    model = None
    write_json(partial, results)
    if args.phase == "fp32-merged":
        write_json(repair / "task0a_root_cause_diagnostics.json", results)
    print(json.dumps({"phase": args.phase, "completed": True}, indent=2))


if __name__ == "__main__":
    main()
