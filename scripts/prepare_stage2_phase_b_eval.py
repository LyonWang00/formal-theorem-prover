"""Freeze S2-Hard-Replay merged-checkpoint evaluation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PARAMETER_KEYS = (
    "backend",
    "checkpoint_type",
    "dtype",
    "load_in_4bit",
    "prompt_format",
    "temperature",
    "top_p",
    "top_k",
    "do_sample",
    "repetition_penalty",
    "max_new_tokens",
    "eos_token_id",
    "pad_token_id",
    "stop_tokens",
    "candidates_per_statement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=Path("outputs/stage2_data_ablation/phaseB_hard_replay"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("generation_config_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    root = args.phase_root if args.phase_root.is_absolute() else project / args.phase_root
    original_path = project / "outputs/task0b_light/generation_contract.json"
    runtime_path = project / "outputs/task0b_light/task0b_light_runtime_config.json"
    model = root / "checkpoints/S2-Hard-Replay-MERGED"
    output_contract_path = root / "evaluation/generation_contract.json"
    output_runtime_path = root / "evaluation/runtime_config.json"
    # A previous interrupted preparation may have written the derived contract
    # before writing the runtime configuration.  Recomputing both artifacts is
    # deterministic, so permit an idempotent repair of that partial state.
    original = json.loads(original_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    required = (
        model / "model.safetensors",
        model / "tokenizer.json",
        model / "tokenizer_config.json",
        model / "chat_template.jinja",
        model / "merge_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"merged S2 evaluation checkpoint incomplete: {missing}")
    if original["generation_config_sha256"] != (
        "bd8f5d25051cb155aaf74b1f25082ddca5b16b44afce7fe066c12c2ab8697386"
    ):
        raise RuntimeError("parent Task0B generation contract drifted")
    if sha256_file(model / "tokenizer.json") != original["tokenizer_sha256"]:
        raise RuntimeError("S2 merged tokenizer.json differs from frozen tokenizer")
    if sha256_file(model / "tokenizer_config.json") != original["tokenizer_config_sha256"]:
        raise RuntimeError("S2 merged tokenizer config differs from frozen tokenizer")
    if sha256_file(model / "chat_template.jinja") != original["chat_template_sha256"]:
        raise RuntimeError("S2 merged chat template differs from frozen formatter")

    resolved = dict(original)
    resolved.update(
        {
            "contract_name": "stage2_phase_b_s2_hard_replay_merged_v1",
            "frozen_model": "S2-Hard-Replay",
            "checkpoint_path": str(model),
            "checkpoint_model_sha256": sha256_file(model / "model.safetensors"),
            "tokenizer_path": str(model),
            "parent_generation_contract_path": str(original_path),
            "parent_generation_contract_sha256": original[
                "generation_config_sha256"
            ],
            "checkpoint_provenance": str(model / "merge_manifest.json"),
        }
    )
    resolved["generation_config_sha256"] = contract_hash(resolved)
    for key in PARAMETER_KEYS:
        if resolved.get(key) != original.get(key):
            raise RuntimeError(f"generation parameter drift at {key}")
    if resolved["checkpoint_type"] != "merged" or resolved["backend"] != "transformers":
        raise RuntimeError("S2 evaluation is not canonical merged Transformers")
    write_json(output_contract_path, resolved)

    runtime["run_name"] = "stage2_phase_b_s2_hard_replay_eval"
    runtime["output_dir"] = str(root / "evaluation/runtime")
    generation = runtime["discovery"]["generation"]
    generation["generation_contract_path"] = str(output_contract_path)
    generation["generation_contract_sha256"] = resolved[
        "generation_config_sha256"
    ]
    generation["backend"] = "transformers"
    generation["samples_per_statement"] = 2
    generation["temperature"] = original["temperature"]
    generation["top_p"] = original["top_p"]
    generation["max_new_tokens"] = original["max_new_tokens"]
    generation["load_in_4bit"] = False
    write_json(output_runtime_path, runtime)
    write_json(
        root / "evaluation/contract_audit.json",
        {
            "passed": True,
            "parent_contract": str(original_path),
            "parent_hash": original["generation_config_sha256"],
            "resolved_contract": str(output_contract_path),
            "resolved_hash": resolved["generation_config_sha256"],
            "parameters_unchanged": {key: resolved[key] for key in PARAMETER_KEYS},
            "only_model_identity_changed": True,
            "adapter_path_for_evaluation": None,
            "checkpoint_type": "merged",
        },
    )
    print(
        json.dumps(
            {
                "model": str(model),
                "model_sha256": resolved["checkpoint_model_sha256"],
                "parent_contract_sha256": original["generation_config_sha256"],
                "resolved_contract_sha256": resolved["generation_config_sha256"],
                "parameters_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
