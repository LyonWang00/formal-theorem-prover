"""Freeze a canonical merged-checkpoint evaluation contract for a Phase-C arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PARAMETER_KEYS = (
    "backend", "checkpoint_type", "dtype", "load_in_4bit", "prompt_format",
    "temperature", "top_p", "top_k", "do_sample", "repetition_penalty",
    "max_new_tokens", "eos_token_id", "pad_token_id", "stop_tokens",
    "candidates_per_statement",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("generation_config_sha256", None)
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--phase-root", default="outputs/stage2_data_ratio_ablation/phaseC1_new_wb")
    parser.add_argument("--model-name", default="S2-New-WB")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional merged checkpoint path, absolute or relative to phase-root.",
    )
    parser.add_argument("--contract-name", default="stage2_phase_c1_s2_new_wb_merged_v1")
    parser.add_argument("--run-name", default="stage2_phase_c1_s2_new_wb_eval")
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / args.phase_root
    parent_path = project / "outputs/task0b_light/generation_contract.json"
    parent_runtime_path = project / "outputs/task0b_light/task0b_light_runtime_config.json"
    if args.model_path:
        supplied_model = Path(args.model_path)
        model = supplied_model if supplied_model.is_absolute() else root / supplied_model
    else:
        model = root / f"checkpoints/{args.model_name}-MERGED"
    out_contract = root / "evaluation/generation_contract.json"
    out_runtime = root / "evaluation/runtime_config.json"

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    runtime = json.loads(parent_runtime_path.read_text(encoding="utf-8"))
    required = ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merge_manifest.json")
    missing = [str(model / name) for name in required if not (model / name).is_file()]
    if missing:
        raise FileNotFoundError(f"merged evaluation checkpoint incomplete: {missing}")
    frozen_parent_hash = "bd8f5d25051cb155aaf74b1f25082ddca5b16b44afce7fe066c12c2ab8697386"
    if parent["generation_config_sha256"] != frozen_parent_hash:
        raise RuntimeError("parent Task0B generation contract drifted")
    for file_name, key in (
        ("tokenizer.json", "tokenizer_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("chat_template.jinja", "chat_template_sha256"),
    ):
        if sha256_file(model / file_name) != parent[key]:
            raise RuntimeError(f"evaluation {file_name} differs from frozen canonical asset")

    resolved = dict(parent)
    resolved.update({
        "contract_name": args.contract_name,
        "frozen_model": args.model_name,
        "checkpoint_path": str(model),
        "checkpoint_model_sha256": sha256_file(model / "model.safetensors"),
        "tokenizer_path": str(model),
        "parent_generation_contract_path": str(parent_path),
        "parent_generation_contract_sha256": parent["generation_config_sha256"],
        "checkpoint_provenance": str(model / "merge_manifest.json"),
    })
    resolved["generation_config_sha256"] = contract_hash(resolved)
    for key in PARAMETER_KEYS:
        if resolved.get(key) != parent.get(key):
            raise RuntimeError(f"generation parameter drift at {key}")
    if resolved["checkpoint_type"] != "merged" or resolved["backend"] != "transformers":
        raise RuntimeError("evaluation is not canonical merged Transformers")
    write_json(out_contract, resolved)

    runtime["run_name"] = args.run_name
    runtime["output_dir"] = str(root / "evaluation/runtime")
    generation = runtime["discovery"]["generation"]
    generation.update({
        "generation_contract_path": str(out_contract),
        "generation_contract_sha256": resolved["generation_config_sha256"],
        "backend": "transformers",
        "samples_per_statement": 2,
        "temperature": parent["temperature"],
        "top_p": parent["top_p"],
        "max_new_tokens": parent["max_new_tokens"],
        "load_in_4bit": False,
    })
    write_json(out_runtime, runtime)
    write_json(root / "evaluation/contract_audit.json", {
        "passed": True,
        "parent_contract": str(parent_path),
        "parent_hash": parent["generation_config_sha256"],
        "resolved_contract": str(out_contract),
        "resolved_hash": resolved["generation_config_sha256"],
        "parameters_unchanged": {key: resolved[key] for key in PARAMETER_KEYS},
        "only_model_identity_changed": True,
        "adapter_path_for_evaluation": None,
        "checkpoint_type": "merged",
    })
    print(json.dumps({
        "model": str(model),
        "model_sha256": resolved["checkpoint_model_sha256"],
        "parent_contract_sha256": parent["generation_config_sha256"],
        "resolved_contract_sha256": resolved["generation_config_sha256"],
        "parameters_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
