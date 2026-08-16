"""Command-line orchestration for Lean dataset preparation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_prover.lean_training.data.preparation import *


def parse_args() -> argparse.Namespace:
    """Parse dataset, filtering, validation, and output CLI options.

    Returns:
        The preparation command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Prepare Lean-Workbook training data and miniF2F benchmark data."
    )
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--train_dataset_config", default=None)
    parser.add_argument(
        "--train_data_kind",
        default=None,
        help="Optional schema kind override, e.g. lean-workbook for a local file.",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--train_output", default=None)
    parser.add_argument("--train_rejected_output", default=None)
    parser.add_argument("--validation_output", default=None)
    parser.add_argument("--grpo_train_output", default=None)
    parser.add_argument("--grpo_validation_output", default=None)
    parser.add_argument(
        "--grpo_exclude_sft_file",
        action="append",
        default=[],
        help=(
            "Prepared SFT JSONL whose statement hashes must be excluded from GRPO; "
            "repeat for train and validation files when both count as prior exposure."
        ),
    )
    parser.add_argument("--validation_ratio", type=float, default=0.02)
    parser.add_argument("--benchmark_dataset_name", default=None)
    parser.add_argument("--benchmark_dataset_config", default=None)
    parser.add_argument(
        "--benchmark_data_kind",
        default=None,
        help="Optional schema kind override, e.g. minif2f for a local file.",
    )
    parser.add_argument("--benchmark_split", default="test")
    parser.add_argument("--benchmark_sample_size", type=int, default=None)
    parser.add_argument("--benchmark_output", default=None)
    parser.add_argument("--benchmark_rejected_output", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Deprecated alias for --max_seq_length.",
    )
    parser.add_argument("--filter_overlength", action="store_true")
    parser.add_argument("--min_completion_tokens", type=int, default=1)
    parser.add_argument(
        "--verify_with_pantograph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter generic normalized records through warm Pantograph workers.",
    )
    parser.add_argument(
        "--lean_project_path",
        default=str(Path(__file__).resolve().parents[3] / "lean_project"),
    )
    parser.add_argument("--lean_timeout", type=int, default=120)
    parser.add_argument("--verification_workers", type=int, default=1)
    parser.add_argument(
        "--pantograph_imports",
        default="Mathlib",
        help="Comma-separated imports used to start the validation server.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for sampling and train/validation splitting. "
            "When omitted, each run uses a fresh random order."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Prepare requested SFT, GRPO, and/or benchmark JSONL artifacts."""
    from lean_prover.lean_training.data.training import (
        build_generation_prompt,
        build_grpo_evaluation_record,
        build_grpo_training_record,
        build_sft_evaluation_record,
        build_sft_training_record,
        exclude_statement_overlaps,
        filter_grpo_records_by_prompt_length,
        filter_sft_records_by_token_length,
        load_statement_hashes,
    )

    args = parse_args()
    if args.max_length is not None:
        print("WARNING: --max_length is deprecated; use --max_seq_length")
        args.max_seq_length = args.max_length
    if not args.train_output and not args.benchmark_output and not args.grpo_train_output:
        raise ValueError(
            "provide --train_output, --grpo_train_output, and/or --benchmark_output"
        )

    if args.train_output:
        if not args.train_dataset_name:
            raise ValueError("--train_dataset_name is required with --train_output")
        train_dataset = load_dataset_source(
            args.train_dataset_name,
            split=args.train_split,
            config_name=args.train_dataset_config,
            revision=args.dataset_revision,
        )
        train_dataset = sample_dataset(train_dataset, None, args.seed)
        train_kind = normalize_dataset_name(
            args.train_data_kind or args.train_dataset_name
        )
        train_records = normalize_records(
            train_dataset,
            dataset_kind=train_kind,
            source_name=args.train_dataset_name,
            require_proof=True,
            limit=args.train_sample_size,
        )
        pantograph_imports = tuple(
            item.strip() for item in args.pantograph_imports.split(",") if item.strip()
        )
        if args.verify_with_pantograph and should_validate_dataset_with_pantograph(
            train_kind
        ):
            before_count = len(train_records)
            train_records = validate_records_with_pantograph(
                train_records,
                lean_project_path=args.lean_project_path,
                imports=pantograph_imports,
                timeout=args.lean_timeout,
                require_proof=True,
                num_workers=args.verification_workers,
                rejected_output_path=(
                    Path(args.train_rejected_output)
                    if args.train_rejected_output
                    else None
                ),
            )
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "train",
                        "before": before_count,
                        "after": len(train_records),
                    },
                    ensure_ascii=False,
                )
            )
        elif args.verify_with_pantograph:
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "train",
                        "data_kind": train_kind,
                        "skipped": True,
                        "reason": "specialized normalizer does not require Lean compile validation",
                    },
                    ensure_ascii=False,
                )
            )
        train_records = filter_sft_records_by_token_length(
            train_records,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            max_seq_length=args.max_seq_length,
            min_completion_tokens=args.min_completion_tokens,
            enabled=args.filter_overlength,
        )
        print_token_length_summary(
            "train_text_before_validation_split",
            token_length_summary(
                (
                    build_generation_prompt(record) + record.proof.strip()
                    for record in train_records
                ),
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                max_length=args.max_seq_length,
            ),
        )
        validation_records: list[NormalizedExample] = []
        if args.validation_output:
            train_records, validation_records = split_training_validation_records(
                train_records,
                validation_ratio=args.validation_ratio,
                seed=None if args.seed is None else args.seed + 1,
            )
        write_jsonl(
            (build_sft_training_record(record) for record in train_records),
            Path(args.train_output),
        )
        print(f"wrote {len(train_records)} training records to {args.train_output}")
        if args.validation_output:
            write_jsonl(
                (build_sft_training_record(record) for record in validation_records),
                Path(args.validation_output),
            )
            print(
                f"wrote {len(validation_records)} in-training validation records "
                f"to {args.validation_output}"
            )

    if args.grpo_train_output:
        if not args.train_dataset_name:
            raise ValueError("--train_dataset_name is required with --grpo_train_output")
        train_kind = normalize_dataset_name(
            args.train_data_kind or args.train_dataset_name
        )
        if train_kind != "lean-workbook":
            raise ValueError(
                "GRPO preparation currently supports only Lean-Workbook; "
                f"got {args.train_dataset_name!r} ({train_kind})"
            )
        grpo_dataset = load_dataset_source(
            args.train_dataset_name,
            split=args.train_split,
            config_name=args.train_dataset_config,
            revision=args.dataset_revision,
        )
        grpo_dataset = sample_dataset(grpo_dataset, None, args.seed)
        grpo_records = normalize_records(
            grpo_dataset,
            dataset_kind=train_kind,
            source_name=args.train_dataset_name,
            require_proof=True,
            limit=args.train_sample_size,
        )
        if args.grpo_exclude_sft_file:
            excluded_hashes = load_statement_hashes(args.grpo_exclude_sft_file)
            grpo_records, overlap_records = exclude_statement_overlaps(
                grpo_records,
                excluded_hashes,
            )
            print(
                "GRPO_SFT_OVERLAP_FILTER_STATS "
                + json.dumps(
                    {
                        "sft_files": args.grpo_exclude_sft_file,
                        "excluded_statement_hashes": len(excluded_hashes),
                        "excluded_records": len(overlap_records),
                        "kept_records": len(grpo_records),
                    },
                    ensure_ascii=False,
                )
            )
            if not grpo_records:
                raise ValueError(
                    "all GRPO records overlap the supplied SFT files; choose a "
                    "disjoint GRPO source/split or omit --grpo_exclude_sft_file"
                )
        if args.verify_with_pantograph:
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "grpo_train",
                        "data_kind": train_kind,
                        "skipped": True,
                        "reason": "Lean-Workbook uses specialized normalization for GRPO preparation",
                    },
                    ensure_ascii=False,
                )
            )
        grpo_records = filter_grpo_records_by_prompt_length(
            grpo_records,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            max_seq_length=args.max_seq_length,
            enabled=args.filter_overlength,
        )
        print_token_length_summary(
            "grpo_prompt_before_validation_split",
            token_length_summary(
                (build_generation_prompt(record) for record in grpo_records),
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                max_length=args.max_seq_length,
            ),
        )
        grpo_validation_records: list[NormalizedExample] = []
        if args.grpo_validation_output:
            grpo_records, grpo_validation_records = split_training_validation_records(
                grpo_records,
                validation_ratio=args.validation_ratio,
                seed=None if args.seed is None else args.seed + 1,
            )
        write_jsonl(
            (build_grpo_training_record(record) for record in grpo_records),
            Path(args.grpo_train_output),
        )
        print(f"wrote {len(grpo_records)} GRPO training records to {args.grpo_train_output}")
        if args.grpo_validation_output:
            write_jsonl(
                (
                    build_grpo_evaluation_record(record)
                    for record in grpo_validation_records
                ),
                Path(args.grpo_validation_output),
            )
            print(
                f"wrote {len(grpo_validation_records)} GRPO validation records "
                f"to {args.grpo_validation_output}"
            )

    if args.benchmark_output:
        if not args.benchmark_dataset_name:
            raise ValueError("--benchmark_dataset_name is required with --benchmark_output")
        benchmark_dataset = load_dataset_source(
            args.benchmark_dataset_name,
            split=args.benchmark_split,
            config_name=args.benchmark_dataset_config,
            revision=args.dataset_revision,
        )
        benchmark_dataset = sample_dataset(benchmark_dataset, None, args.seed)
        benchmark_kind = normalize_dataset_name(
            args.benchmark_data_kind or args.benchmark_dataset_name
        )
        benchmark_records = normalize_records(
            benchmark_dataset,
            dataset_kind=benchmark_kind,
            source_name=args.benchmark_dataset_name,
            require_proof=False,
            limit=args.benchmark_sample_size,
        )
        pantograph_imports = tuple(
            item.strip() for item in args.pantograph_imports.split(",") if item.strip()
        )
        if args.verify_with_pantograph and should_validate_dataset_with_pantograph(
            benchmark_kind
        ):
            before_count = len(benchmark_records)
            benchmark_records = validate_records_with_pantograph(
                benchmark_records,
                lean_project_path=args.lean_project_path,
                imports=pantograph_imports,
                timeout=args.lean_timeout,
                require_proof=False,
                num_workers=args.verification_workers,
                rejected_output_path=(
                    Path(args.benchmark_rejected_output)
                    if args.benchmark_rejected_output
                    else None
                ),
            )
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "benchmark",
                        "before": before_count,
                        "after": len(benchmark_records),
                    },
                    ensure_ascii=False,
                )
            )
        elif args.verify_with_pantograph:
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "benchmark",
                        "data_kind": benchmark_kind,
                        "skipped": True,
                        "reason": "specialized normalizer does not require Lean compile validation",
                    },
                    ensure_ascii=False,
                )
            )
        print_token_length_summary(
            "benchmark_prompt",
            token_length_summary(
                (build_generation_prompt(record) for record in benchmark_records),
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                max_length=args.max_seq_length,
            ),
        )
        write_jsonl(
            (build_sft_evaluation_record(record) for record in benchmark_records),
            Path(args.benchmark_output),
        )
        print(f"wrote {len(benchmark_records)} benchmark records to {args.benchmark_output}")


if __name__ == "__main__":
    main()
