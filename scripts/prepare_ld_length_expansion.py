#!/usr/bin/env python3
"""Prepare only the full evaluations authorized by the frozen LD canary gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_PATHS = {
    "M0-ZERO": None,
    "MIX-A-WB100": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-A-WB100/final",
    "MIX-B-LD25": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-B-LD25/final",
    "MIX-C-LD50": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-C-LD50/final",
    "MIX-E-LD100": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-E-LD100/final",
}
BASE_MODEL = "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"
PAIRED_FULL_SEED = 10260901


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_dir(root: Path, dataset: str, model: str) -> Path:
    evaluation = root / "outputs/wb_ld_small_sft_ablation/evaluation"
    if model == "M0-ZERO":
        return evaluation / "zero_step" / (
            "ld_holdout" if dataset == "ld" else "wb_gate150"
        )
    return evaluation / (
        "ld_holdout" if dataset == "ld" else "wb_gate150"
    ) / model


def validate_baseline(path: Path, *, statements: int) -> dict[str, Any]:
    generations = read_jsonl(path / "generations.jsonl")
    attempts = read_jsonl(path / "attempts.jsonl")
    expected_candidates = statements * 4
    if len(generations) != expected_candidates or len(attempts) != expected_candidates:
        raise ValueError(
            f"incomplete L256 baseline at {path}: "
            f"{len(generations)} generations, {len(attempts)} attempts"
        )
    keys = Counter(
        (str(row["statement_id"]), int(row["sample_index"]))
        for row in generations
    )
    if len(keys) != expected_candidates or set(keys.values()) != {1}:
        raise ValueError(f"duplicate or missing candidate key in {path}")
    if {int(row["max_new_tokens"]) for row in generations} != {256}:
        raise ValueError(f"baseline is not exclusively L256: {path}")
    if {int(row["generation_seed"]) for row in generations} != {PAIRED_FULL_SEED}:
        raise ValueError(f"baseline seed mismatch: {path}")
    return {
        "path": str(path),
        "statements": statements,
        "candidates": expected_candidates,
        "generation_seed": PAIRED_FULL_SEED,
        "max_new_tokens": 256,
    }


def tagged_manifest(
    source: Path,
    target: Path,
    *,
    dataset: str,
    expected_rows: int,
) -> dict[str, Any]:
    rows = read_jsonl(source)
    if len(rows) != expected_rows:
        raise ValueError(f"{source} has {len(rows)} rows, expected {expected_rows}")
    tagged: list[dict[str, Any]] = []
    for source_row in rows:
        row = json.loads(json.dumps(source_row))
        row["data_role"] = "benchmark"
        metadata = row.setdefault("metadata", {})
        metadata["length_ablation_dataset"] = dataset
        metadata.setdefault("length_ablation", {}).update(
            {
                "dataset": (
                    "LD_HOLDOUT_SOURCE_DISJOINT_128"
                    if dataset == "ld"
                    else "WB_GATE_150"
                ),
                "parent_manifest": str(source),
            }
        )
        tagged.append(row)
    write_jsonl(target, tagged)
    target.chmod(0o444)
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "frozen_path": str(target),
        "frozen_sha256": sha256(target),
        "rows": len(tagged),
        "read_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/length_ablation"),
    )
    args = parser.parse_args()
    project = args.root.resolve()
    output = (project / args.output).resolve()
    expansion = read_json(output / "comparisons/expansion_decision.json")
    positive_pairs = expansion["positive_model_length_pairs"]
    positive_models = sorted({pair["model"] for pair in positive_pairs})
    if not positive_models:
        write_json(
            output / "audit/full_expansion_audit.json",
            {
                "authorized": False,
                "reason": "No pre-registered LD canary gate was positive.",
                "runs": [],
            },
        )
        print(json.dumps({"authorized": False, "runs": []}, indent=2))
        return

    manifests = {
        "ld": tagged_manifest(
            project
            / "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl",
            output / "manifests/full_ld_holdout_128.jsonl",
            dataset="ld",
            expected_rows=128,
        ),
        "wb": tagged_manifest(
            project / "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
            output / "manifests/full_wb_gate_150.jsonl",
            dataset="wb",
            expected_rows=150,
        ),
    }
    baselines: dict[str, Any] = {}
    for model in positive_models:
        baselines[f"{model}:ld:L256"] = validate_baseline(
            baseline_dir(project, "ld", model),
            statements=128,
        )
    for pair in positive_pairs:
        model = pair["model"]
        key = f"{model}:wb:L256"
        if key not in baselines:
            baselines[key] = validate_baseline(
                baseline_dir(project, "wb", model),
                statements=150,
            )

    runs: list[dict[str, Any]] = []
    for model in positive_models:
        for length in (512, 1024):
            runs.append(
                {
                    "dataset": "ld",
                    "model": model,
                    "length": length,
                    "seed": PAIRED_FULL_SEED,
                    "manifest": manifests["ld"]["frozen_path"],
                    "expected_wb": 0,
                    "expected_ld": 128,
                    "base_model": str(project / BASE_MODEL),
                    "adapter": (
                        str(project / MODEL_PATHS[model])
                        if MODEL_PATHS[model]
                        else None
                    ),
                    "output": str(
                        output / "expansion/ld" / model / f"L{length}"
                    ),
                }
            )
    for pair in positive_pairs:
        model = pair["model"]
        length = int(str(pair["length"]).removeprefix("L"))
        runs.append(
            {
                "dataset": "wb",
                "model": model,
                "length": length,
                "seed": PAIRED_FULL_SEED,
                "manifest": manifests["wb"]["frozen_path"],
                "expected_wb": 150,
                "expected_ld": 0,
                "base_model": str(project / BASE_MODEL),
                "adapter": (
                    str(project / MODEL_PATHS[model])
                    if MODEL_PATHS[model]
                    else None
                ),
                "output": str(
                    output / "expansion/wb" / model / f"L{length}"
                ),
            }
        )
    plan = {
        "authorized": True,
        "gate_source": str(output / "comparisons/expansion_decision.json"),
        "positive_pairs": positive_pairs,
        "positive_models": positive_models,
        "paired_full_seed": PAIRED_FULL_SEED,
        "manifests": manifests,
        "reused_l256_baselines": baselines,
        "runs": runs,
    }
    write_json(output / "manifests/expansion_plan.json", plan)
    write_json(output / "audit/full_expansion_audit.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
