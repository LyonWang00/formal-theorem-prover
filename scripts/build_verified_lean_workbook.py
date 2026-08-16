#!/usr/bin/env python3
"""Reconstruct, audit, attest, and split Lean-Workbook in the target environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-parquet", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--lean-project", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--current-train", required=True)
    parser.add_argument("--current-eval", required=True)
    parser.add_argument("--current-discovery", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--eval-size", type=int, default=160)
    parser.add_argument("--discovery-size", type=int, default=700)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="rebuild reports/splits only when every compatible result is cached",
    )
    args = parser.parse_args()
    lean_project = Path(args.lean_project)
    if args.reports_only:
        from lean_prover.lean_training.data.verified_builder import (
            _build_verified_dataset_with_pool,
        )

        _build_verified_dataset_with_pool(
            pool=None,
            raw_parquet=Path(args.raw_parquet),
            source_commit=args.source_commit,
            lean_project=lean_project,
            output_dir=Path(args.output_dir),
            current_files={
                "train": Path(args.current_train),
                "eval": Path(args.current_eval),
                "discovery": Path(args.current_discovery),
            },
            timeout=args.timeout,
            seed=args.seed,
            train_size=args.train_size,
            eval_size=args.eval_size,
            discovery_size=args.discovery_size,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            max_seq_length=args.max_seq_length,
        )
        return
    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=str(lean_project),
            imports=("Mathlib",),
            timeout=args.timeout,
            warmup_timeout=args.startup_timeout,
            num_workers=args.num_workers,
            queue_maxsize=8,
            max_worker_restarts=3,
            max_task_retries=1,
            shutdown_timeout=15,
        )
    )
    # Keep the spawn entry point lightweight.  Importing the raw-data builder
    # only after worker startup prevents every worker from importing the parquet
    # and audit stack while Pantograph is loading Mathlib.
    with pool:
        from lean_prover.lean_training.data.verified_builder import (
            _build_verified_dataset_with_pool,
        )

        _build_verified_dataset_with_pool(
            pool=pool,
            raw_parquet=Path(args.raw_parquet),
            source_commit=args.source_commit,
            lean_project=lean_project,
            output_dir=Path(args.output_dir),
            current_files={
                "train": Path(args.current_train),
                "eval": Path(args.current_eval),
                "discovery": Path(args.current_discovery),
            },
            timeout=args.timeout,
            seed=args.seed,
            train_size=args.train_size,
            eval_size=args.eval_size,
            discovery_size=args.discovery_size,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()
