"""Run a real Base/M0/B2 comparison without altering prior experiment outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from lean_prover.lean_training.evaluation.expanded_validation import (
    PRIMARY_SEED,
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
if __package__:
    from scripts.build_b2_expanded_validation import directory_identity
    from scripts import run_b2_expanded_validation as staged
else:
    from build_b2_expanded_validation import directory_identity
    import run_b2_expanded_validation as staged


DEFAULT_ROOT = Path("outputs/base_m0_b2_comparison")
DEFAULT_SOURCE = Path("outputs/b2_expanded_validation")
DEFAULT_CONFIG = Path("configs/expert_iteration.round1.yaml")
RAW_BASE = Path("models/Qwen2.5-1.5B-Instruct")

DATASETS = {
    "full500": ("full500.jsonl", "discovery_replay", 500),
    "strict_unseen_primary": ("strict_unseen_discovery.jsonl", "discovery_replay", 200),
    "monitor_primary": ("monitor_minif2f_valid_64.jsonl", "monitor", 64),
    "benchmark": ("benchmark_minif2f_test_96.jsonl", "benchmark", 96),
}


def fixed_dataset(
    output_root: Path,
    source_root: Path,
    filename: str,
) -> Path:
    source = source_root / "datasets" / filename
    destination = output_root / "datasets" / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"fixed dataset drift: {destination}")
    else:
        shutil.copyfile(source, destination)
    return destination


def build_manifest(
    output_root: Path,
    source_root: Path,
    comparison_config: Path,
) -> dict[str, Any]:
    staged.MODELS["Base"] = {"base": RAW_BASE, "adapter": None}
    source_manifest = json.loads(
        (source_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    datasets: dict[str, Any] = {}
    for stage, (filename, _role, expected_rows) in DATASETS.items():
        path = fixed_dataset(output_root, source_root, filename)
        actual_rows = len(read_jsonl(path))
        if actual_rows != expected_rows:
            raise ValueError(
                f"{stage} expected {expected_rows} rows, found {actual_rows}"
            )
        datasets[stage] = {
            "path": str(path.resolve()),
            "rows": actual_rows,
            "sha256": sha256_file(path),
            "source_sha256": sha256_file(source_root / "datasets" / filename),
        }
    manifest = {
        "experiment": "base_m0_b2_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_results": str(source_root.resolve()),
        "comparison_config": {
            "path": str(comparison_config.resolve()),
            "sha256": sha256_file(comparison_config),
        },
        "models": {
            "Base": {
                **directory_identity(
                    RAW_BASE,
                    ("model*.safetensors", "config.json", "tokenizer.json"),
                ),
                "role": "raw_Qwen2.5_1.5B_Instruct_before_M0",
            },
            "M0": source_manifest["models"]["M0"],
            "B2": source_manifest["models"]["B2"],
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "samples_per_statement": 4,
            "statement_batch_size": 100,
            "seed": PRIMARY_SEED,
            "seed_derivation_version": source_manifest["generation"][
                "seed_derivation_version"
            ],
        },
        "verification": {
            "workers": 2,
            "timeout_seconds": 30,
            "max_queue_size": 32,
            "cache_backend": "sqlite",
        },
        "datasets": datasets,
        "reuse": {
            "full500_M0_B2": str(
                (source_root / "evaluations/full500").resolve()
            ),
            "strict_unseen_primary_M0_B2": str(
                (source_root / "evaluations/strict_unseen_primary").resolve()
            ),
            "monitor_primary_M0_B2": str(
                (source_root / "evaluations/monitor_primary").resolve()
            ),
            "benchmark": "new evaluation for Base, M0, and B2",
        },
        "forbidden_actions": {
            "training": False,
            "proof_bank_updates": False,
            "round2": False,
            "m2": False,
        },
    }
    write_json_atomic(output_root / "run_manifest.json", manifest)
    return manifest


def run_one(
    *,
    root: Path,
    config: Path,
    dataset: Path,
    stage: str,
    model: str,
    role: str,
    max_attempts: int,
) -> dict[str, Any]:
    return staged.run_model_dataset(
        root=root,
        config=config,
        dataset=dataset,
        stage=stage,
        model_name=model,
        role=role,
        seed=PRIMARY_SEED,
        max_attempts=max_attempts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=Path("configs/base_m0_b2_comparison.yaml"),
    )
    parser.add_argument(
        "--phase",
        choices=("manifest", "preflight", "base_primary", "benchmark", "all"),
        default="all",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    staged.MODELS["Base"] = {"base": RAW_BASE, "adapter": None}
    build_manifest(args.root, args.source_root, args.comparison_config)
    if args.phase == "manifest":
        return
    if staged.suspicious_processes():
        raise RuntimeError(
            "pre-existing GPU/Pantograph/trainer processes: "
            f"{staged.suspicious_processes()}"
        )

    if args.phase in ("preflight", "all"):
        dataset = fixed_dataset(
            args.root, args.source_root, "preflight_smoke5.jsonl"
        )
        result = run_one(
            root=args.root,
            config=args.config,
            dataset=dataset,
            stage="preflight_smoke5",
            model="Base",
            role="discovery_replay",
            max_attempts=args.max_attempts,
        )
        if result["candidate_count"] != 20:
            raise RuntimeError("Base preflight did not produce 20 candidates")
        write_json_atomic(
            args.root / "preflight.json",
            {"passed": True, "model": "Base", "metrics": result},
        )
        if args.phase == "preflight":
            return

    if args.phase in ("base_primary", "all"):
        for stage in ("full500", "strict_unseen_primary", "monitor_primary"):
            filename, role, _rows = DATASETS[stage]
            run_one(
                root=args.root,
                config=args.config,
                dataset=args.root / "datasets" / filename,
                stage=stage,
                model="Base",
                role=role,
                max_attempts=args.max_attempts,
            )
        if args.phase == "base_primary":
            return

    if args.phase in ("benchmark", "all"):
        filename, role, _rows = DATASETS["benchmark"]
        results = {}
        for model in ("Base", "M0", "B2"):
            results[model] = run_one(
                root=args.root,
                config=args.config,
                dataset=args.root / "datasets" / filename,
                stage="benchmark",
                model=model,
                role=role,
                max_attempts=args.max_attempts,
            )
        write_json_atomic(
            args.root / "benchmark_completion.json",
            {
                "status": "completed",
                "reason": "explicit three-model comparison requested by user",
                "metrics": results,
            },
        )


if __name__ == "__main__":
    main()
