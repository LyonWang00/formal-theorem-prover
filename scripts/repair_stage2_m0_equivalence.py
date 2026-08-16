#!/usr/bin/env python3
"""Repair the M0 merged/unmerged equivalence check with a matched backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.check_stage2_m0_equivalence import (
    ADAPTER,
    BASE,
    FROZEN_NAME,
    ROOT,
    SOURCE,
    compare,
    run_evaluation,
)
from scripts.run_r_random_replication_evaluation import (
    read_json,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
)


def optional_sha256(path: Path) -> str | None:
    return sha256(path) if path.is_file() else None


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--precision", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--probe-statement-id", default=None)
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    repair_name = (
        "m0_checkpoint_equivalence_repair_fp32"
        if args.precision == "float32"
        else "m0_checkpoint_equivalence_repair"
    )
    quantum_text = os.environ.get("M0_EQUIVALENCE_LOGIT_QUANTUM", "")
    if quantum_text:
        repair_name += "_rounded_" + quantum_text.replace(".", "p")
    if args.probe_statement_id:
        repair_name += "_probe_" + args.probe_statement_id[-8:]
    repair = root / "shared" / repair_name
    stage = repair / "results"
    logs = repair / "logs"
    base = (
        root / "shared/checkpoints/Qwen2.5-1.5B-Instruct-FP32-VIEW"
        if args.precision == "float32"
        else project / BASE
    )
    adapter = project / ADAPTER
    merged = root / (
        "shared/checkpoints/M0-ADDON-B-FROZEN-MERGED-FP32"
        if args.precision == "float32"
        else "shared/checkpoints/M0-ADDON-B-FROZEN-MERGED"
    )
    source = project / SOURCE

    source_config = project / (
        "outputs/stage2_sft_incremental_ablation/shared/"
        "m0_checkpoint_equivalence/greedy_generation_config.json"
    )
    config_payload = read_json(source_config)
    generation = config_payload["discovery"]["generation"]
    generation.update(
        {
            "backend": "transformers",
            "samples_per_statement": 1,
            "temperature": 0.0,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "batch_size": 1,
            "load_in_4bit": False,
        }
    )
    config = repair / "matched_transformers_bf16_config.json"
    write_json(config, config_payload)
    manifest = repair / "greedy24_manifest.jsonl"
    selected_rows = read_jsonl(source)[:24]
    if args.probe_statement_id:
        selected_rows = [
            row
            for row in selected_rows
            if str(row.get("statement_id") or "") == args.probe_statement_id
        ]
        if not selected_rows:
            baseline_generations = root / (
                "shared/m0_checkpoint_equivalence_repair/results/"
                "unmerged/generations.jsonl"
            )
            baseline_row = next(
                (
                    row
                    for row in read_jsonl(baseline_generations)
                    if str(row.get("statement_id") or "")
                    == args.probe_statement_id
                ),
                None,
            )
            if baseline_row is not None:
                probe_prompt = str(baseline_row.get("prompt") or "")
                selected_rows = [
                    row
                    for row in read_jsonl(source)[:24]
                    if str(row.get("prompt") or "") == probe_prompt
                ]
        if len(selected_rows) != 1:
            raise RuntimeError(
                f"probe statement alignment is {len(selected_rows)} != 1"
            )
    write_jsonl(manifest, selected_rows)

    base_config = read_json(base / "config.json")
    adapter_config = read_json(adapter / "adapter_config.json")
    merged_config = read_json(merged / "config.json")
    merge_manifest = read_json(merged / "merge_manifest.json")
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    tokenizer_identity = {
        name: {
            "base": optional_sha256(base / name),
            "adapter": optional_sha256(adapter / name),
            "merged": optional_sha256(merged / name),
        }
        for name in tokenizer_files
    }
    diagnostics = {
        "frozen_model": FROZEN_NAME,
        "dtype": {
            "base_config": base_config.get("dtype")
            or base_config.get("torch_dtype"),
            "merged_config": merged_config.get("dtype")
            or merged_config.get("torch_dtype"),
            "matched_inference_dtype": f"torch.{args.precision}",
            "load_in_4bit": False,
        },
        "tokenizer": tokenizer_identity,
        "generation_config": {
            "backend": "transformers",
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "seed": 20261701,
            "padding_side": "left",
            "skip_special_tokens": True,
            "config_sha256": canonical_hash(config_payload),
            "logit_quantum": float(quantum_text) if quantum_text else None,
        },
        "merge": {
            "method": "PEFT merge_and_unload(safe_merge=True)",
            "manifest": merge_manifest,
            "merged_model_sha256": sha256(merged / "model.safetensors"),
        },
        "backend": {
            "previous": "vLLM dynamic LoRA versus vLLM merged",
            "repaired": (
                "Transformers AutoModelForCausalLM + PEFT dynamic LoRA versus "
                "Transformers AutoModelForCausalLM merged; same "
                f"{args.precision} dtype on CPU"
            ),
            "reason": (
                "The previous vLLM paths used different kernels for dynamic LoRA "
                "and merged dense weights. Greedy decoding amplified small numeric "
                "differences into three Pantograph outcome mismatches."
            ),
        },
    }
    write_json(repair / "diagnostics.json", diagnostics)

    run_evaluation(
        project=project,
        config=config,
        manifest=manifest,
        model=base,
        adapter=adapter,
        output=stage / "unmerged",
        log=logs / "transformers_unmerged.log",
    )
    run_evaluation(
        project=project,
        config=config,
        manifest=manifest,
        model=merged,
        adapter=None,
        output=stage / "merged",
        log=logs / "transformers_merged.log",
    )
    if args.probe_statement_id:
        left_attempts = read_jsonl(stage / "unmerged/attempts.jsonl")
        right_attempts = read_jsonl(stage / "merged/attempts.jsonl")
        if len(left_attempts) != 1 or len(right_attempts) != 1:
            raise RuntimeError("probe attempts are not 1/1")
        probe = {
            "statement_id": args.probe_statement_id,
            "logit_quantum": float(quantum_text) if quantum_text else None,
            "unmerged_success": bool(left_attempts[0].get("success")),
            "merged_success": bool(right_attempts[0].get("success")),
            "pantograph_outcome_match": bool(left_attempts[0].get("success"))
            == bool(right_attempts[0].get("success")),
            "unmerged_status": left_attempts[0].get("status"),
            "merged_status": right_attempts[0].get("status"),
        }
        write_json(repair / "probe_summary.json", probe)
        print(json.dumps(probe, indent=2))
        raise SystemExit(0 if probe["pantograph_outcome_match"] else 2)
    report = compare(project, stage)
    report["repair_diagnostics"] = diagnostics
    write_json(repair / "equivalence_summary.json", report)
    markdown = f"""# Checkpoint equivalence repair report

## Result

Status: **{report['status']}**

The repair uses the same Transformers backend and {args.precision} dtype for dynamic
base+adapter inference and merged inference. It keeps the frozen 24 prompts,
greedy decoding, tokenizer contract, seed, and 256-token limit unchanged.

| Check | Result |
|---|---:|
| First-token match | {report['first_token_matches']}/24 |
| Major common-prefix match | {report['major_common_prefix_matches']}/24 |
| Proof-extraction outcome match | {report['proof_extraction_outcome_matches']}/24 |
| Exact extracted-proof match | {report['exact_extracted_proof_matches']}/24 |
| Pantograph outcome match | {report['pantograph_outcome_matches']}/24 |
| Pantograph status match | {report['pantograph_status_matches']}/24 |
| Mean common-prefix tokens | {report['mean_common_prefix_tokens']:.3f} |

## Checks and repair

- dtype: base and merged are loaded as {args.precision}; 4-bit quantization is disabled.
- tokenizer: tokenizer JSON/config/chat-template hashes are recorded in
  `diagnostics.json`; both inference paths use their frozen local tokenizer.
- generation: greedy, temperature 0, top-p 0.95, max-new-tokens 256, seed
  20261701.
- merge: PEFT `merge_and_unload(safe_merge=True)` with frozen base and adapter
  hashes recorded by `merge_manifest.json`.
- backend: replaced the asymmetric vLLM dynamic-LoRA/dense-kernel comparison
  with matched Transformers+PEFT BF16 paths.

No Trainer or training process was created.
"""
    (repair / "checkpoint_equivalence_report.md").write_text(
        markdown, encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"details", "repair_diagnostics"}},
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
