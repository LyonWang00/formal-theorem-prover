#!/usr/bin/env python3
"""Fingerprint base, adapter, merged, tokenizer, and training-state artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifact(path_value: str | None, *, hash_weights: bool) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    report: dict[str, Any] = {
        "configured_path": path_value,
        "resolved_path": str(path.resolve()) if path.exists() else path_value,
        "exists": path.exists(),
    }
    if not path.is_dir():
        return report
    configs = {}
    for name in (
        "config.json",
        "adapter_config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "trainer_state.json",
        "training_config.json",
    ):
        item = path / name
        if item.is_file():
            configs[name] = {"size": item.stat().st_size, "sha256": sha256(item)}
    weights = []
    for item in sorted(path.glob("*.safetensors")):
        weights.append(
            {
                "name": item.name,
                "size": item.stat().st_size,
                "sha256": sha256(item) if hash_weights else None,
            }
        )
    manifest_payload = json.dumps(
        {"configs": configs, "weights": weights},
        sort_keys=True,
        separators=(",", ":"),
    )
    report.update(
        {
            "configs": configs,
            "weights": weights,
            "total_weight_bytes": sum(item["size"] for item in weights),
            "checkpoint_manifest_hash": hashlib.sha256(
                manifest_payload.encode("utf-8")
            ).hexdigest(),
        }
    )
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        payload = json.loads(adapter_config.read_text(encoding="utf-8-sig"))
        report["adapter_config"] = {
            "r": payload.get("r"),
            "lora_alpha": payload.get("lora_alpha"),
            "lora_dropout": payload.get("lora_dropout"),
            "target_modules": payload.get("target_modules"),
            "base_model_name_or_path": payload.get("base_model_name_or_path"),
        }
    return report


def training_metrics(adapter_path: Path) -> dict[str, Any]:
    state_path = adapter_path / "trainer_state.json"
    if not state_path.is_file():
        return {"available": False}
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    history = list(state.get("log_history") or [])
    train_rows = [row for row in history if "loss" in row and "eval_loss" not in row]
    eval_rows = [row for row in history if "eval_loss" in row]
    return {
        "available": True,
        "initial_train_loss": train_rows[0].get("loss") if train_rows else None,
        "final_logged_train_loss": train_rows[-1].get("loss") if train_rows else None,
        "best_eval_loss": min((row["eval_loss"] for row in eval_rows), default=None),
        "final_eval_loss": eval_rows[-1].get("eval_loss") if eval_rows else None,
        "optimizer_steps": state.get("global_step"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "train_epochs": state.get("epoch"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--sft_adapter", required=True)
    parser.add_argument("--merged_checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--selected_checkpoint", required=True)
    parser.add_argument("--checkpoint_strategy", default="fixed_anchor")
    parser.add_argument("--hash_weights", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    adapter_path = Path(args.sft_adapter).expanduser()
    report = {
        "checkpoint_strategy": args.checkpoint_strategy,
        "adapter_enabled_for_selected_checkpoint": (
            Path(args.selected_checkpoint).resolve() != Path(args.merged_checkpoint).resolve()
        ),
        "base_model": inspect_artifact(args.base_model, hash_weights=False),
        "sft_adapter": inspect_artifact(args.sft_adapter, hash_weights=args.hash_weights),
        "merged_checkpoint": inspect_artifact(args.merged_checkpoint, hash_weights=False),
        "tokenizer": inspect_artifact(args.tokenizer, hash_weights=False),
        "selected_checkpoint": inspect_artifact(args.selected_checkpoint, hash_weights=False),
        "training_metrics": training_metrics(adapter_path),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
