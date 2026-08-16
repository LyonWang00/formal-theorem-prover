"""Run one staged generation/Lean-verification evaluation for an ablation arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lean_prover.lean_training.expert_iteration.config import load_expert_iteration_config
from lean_prover.lean_training.expert_iteration.evaluation_adapter import BenchmarkPipelineAdapter


def dataset_uses_source_faithful_templates(path: str | Path) -> bool:
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("preassembled_source_template"):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--role", choices=("monitor", "benchmark", "discovery_replay"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--samples-per-statement", type=int, default=4)
    args = parser.parse_args()
    if dataset_uses_source_faithful_templates(args.dataset):
        raise ValueError(
            "source-faithful LeanDojo data cannot use the generic benchmark "
            "orchestrator; use run_wb_ld_ablation_evaluation.py with "
            "--role ld_holdout so Pantograph is grouped by record imports"
        )
    config = load_expert_iteration_config(args.config)
    result = BenchmarkPipelineAdapter(config).run(
        role=args.role,
        dataset_path=args.dataset,
        base_model=args.model,
        adapter_path=args.adapter,
        output_dir=args.output,
        pass_k=sorted({1, 2, args.samples_per_statement}),
        samples_per_statement=args.samples_per_statement,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
