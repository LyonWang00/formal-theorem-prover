#!/usr/bin/env python3
"""Freeze matched canonical evaluation contracts for difficulty ablation arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PARENT_HASH = "bd8f5d25051cb155aaf74b1f25082ddca5b16b44afce7fe066c12c2ab8697386"
EVAL_SEED = 20261815
PARAMETER_KEYS = (
    "backend", "checkpoint_type", "dtype", "load_in_4bit", "prompt_format", "temperature", "top_p",
    "top_k", "do_sample", "repetition_penalty", "max_new_tokens", "eos_token_id", "pad_token_id",
    "stop_tokens", "candidates_per_statement",
)
MODELS = {
    "A_medium_hard": "A-MERGED", "B_medium_only": "B-MERGED",
    "C_add_repair": "C-MERGED", "D_add_replay": "D-MERGED",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("generation_config_sha256", None)
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/difficulty_ablation"
    parent_path = project / "outputs/task0b_light/generation_contract.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8-sig"))
    runtime_template = json.loads((project / "outputs/expert_iteration/round0/evaluation/runtime_config.json").read_text(encoding="utf-8-sig"))
    if parent.get("generation_config_sha256") != PARENT_HASH:
        raise RuntimeError("frozen Task0B generation contract drifted")
    results = {}
    for label, merged_name in MODELS.items():
        model = root / label / "checkpoint" / merged_name
        required = [model / name for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merge_manifest.json")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete merged checkpoint {label}: {missing}")
        for filename, key in (("tokenizer.json", "tokenizer_sha256"), ("tokenizer_config.json", "tokenizer_config_sha256"), ("chat_template.jinja", "chat_template_sha256")):
            if sha256(model / filename) != parent[key]:
                raise RuntimeError(f"{label} tokenizer/formatter drift: {filename}")
        resolved = dict(parent)
        resolved.update({
            "contract_name": f"ei_difficulty_ablation_{label}_merged_v1", "frozen_model": label,
            "checkpoint_path": str(model), "checkpoint_model_sha256": sha256(model / "model.safetensors"),
            "tokenizer_path": str(model), "parent_generation_contract_path": str(parent_path),
            "parent_generation_contract_sha256": PARENT_HASH, "checkpoint_provenance": str(model / "merge_manifest.json"),
            "evaluation_seed": EVAL_SEED,
        })
        resolved["generation_config_sha256"] = contract_hash(resolved)
        if any(resolved.get(key) != parent.get(key) for key in PARAMETER_KEYS):
            raise RuntimeError(f"canonical generation drift for {label}")
        if resolved["backend"] != "transformers" or resolved["checkpoint_type"] != "merged" or resolved["dtype"] != "bfloat16":
            raise RuntimeError(f"non-canonical path for {label}")
        eval_root = root / label / "evaluation"
        contract_path = eval_root / "generation_contract.json"
        write_json(contract_path, resolved)
        runtime = json.loads(json.dumps(runtime_template))
        runtime["run_name"] = f"ei_difficulty_ablation_{label}_core_eval"
        runtime["output_dir"] = str(eval_root / "runtime")
        generation = runtime["discovery"]["generation"]
        generation.update({
            "backend": "transformers", "samples_per_statement": 2, "temperature": parent["temperature"],
            "top_p": parent["top_p"], "max_new_tokens": parent["max_new_tokens"], "load_in_4bit": False,
            "generation_contract_path": str(contract_path), "generation_contract_sha256": resolved["generation_config_sha256"],
        })
        runtime["verification"]["num_workers"] = 1
        runtime["verification"]["queue_maxsize"] = 1
        runtime["runtime"]["verification"]["workers"] = 1
        runtime["runtime"]["memory"]["max_queue_size"] = 1
        runtime_path = eval_root / "runtime_config.json"
        write_json(runtime_path, runtime)
        write_json(eval_root / "contract_audit.json", {
            "passed": True, "model": label, "model_sha256": resolved["checkpoint_model_sha256"],
            "parent_hash": PARENT_HASH, "resolved_hash": resolved["generation_config_sha256"],
            "parameters_unchanged": {key: resolved[key] for key in PARAMETER_KEYS}, "adapter_path": None,
            "checkpoint_type": "merged", "backend": "transformers", "dtype": "bfloat16",
            "evaluation_seed": EVAL_SEED, "verification_workers": 1,
        })
        results[label] = {"checkpoint": str(model), "model_sha256": resolved["checkpoint_model_sha256"], "generation_config_sha256": resolved["generation_config_sha256"], "runtime_config": str(runtime_path)}
    h0_reference = project / "outputs/expert_iteration/ei_ablation/comparisons/H0_matched_seed/evaluation"
    write_json(root / "comparisons/evaluation_contracts.json", {
        "evaluation_seed": EVAL_SEED, "models": results,
        "H0_baseline": {"path": str(h0_reference), "reuse_reason": "identical frozen H0 checkpoint, generation parameters, seed, datasets, and corrected source-faithful LD orchestration"},
    })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
