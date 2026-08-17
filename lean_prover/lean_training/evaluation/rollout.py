"""Unified Prover-only rollout over canonical SFT or GRPO final data.

This module intentionally reuses the benchmark generator, proof extraction,
Pantograph worker pool, persistence, and statistics.  It only adapts the two
minimal trainer formats into proof-free evaluation records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lean_prover.lean_training.data.preparation import (
    strip_dataset_placeholder_proof,
)
from lean_prover.lean_training.evaluation.benchmark import (
    BenchmarkRecord,
    build_argument_parser,
    run_pipeline,
    split_prompt_sections,
)


DECLARATION_RE = re.compile(r"(?m)^\s*(?:theorem|lemma|example)\b")
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)\s*$")


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {source}:{line_number}")
            yield row


def _prompt_components(
    row: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("rollout row has no prompt")
    sections = split_prompt_sections(prompt)
    explicit_statement = str(row.get("lean_statement") or "").strip()
    section_statement = sections.get("lean_statement", "").strip()
    if explicit_statement:
        statement = strip_dataset_placeholder_proof(explicit_statement)
    elif section_statement:
        statement = strip_dataset_placeholder_proof(section_statement)
    else:
        match = DECLARATION_RE.search(prompt)
        if match is None:
            raise ValueError("rollout prompt has no theorem/lemma/example declaration")
        statement = strip_dataset_placeholder_proof(prompt[match.start() :])

    imports_value = row.get("imports") or (row.get("preamble") or {}).get("imports")
    imports: list[str] = []
    if isinstance(imports_value, str):
        imports_value = [imports_value]
    if isinstance(imports_value, Iterable):
        imports.extend(str(item).strip() for item in imports_value if str(item).strip())
    for line in prompt.splitlines():
        import_match = IMPORT_RE.match(line)
        if import_match:
            imports.append(import_match.group(1))
    imports = list(dict.fromkeys(imports or ["Mathlib"]))

    context_value = row.get("context_lines") or (row.get("preamble") or {}).get(
        "context_lines"
    )
    context: list[str] = []
    if isinstance(context_value, str):
        context.extend(line for line in context_value.splitlines() if line.strip())
    elif isinstance(context_value, Iterable):
        context.extend(str(item) for item in context_value if str(item).strip())
    if not context and not section_statement:
        declaration_match = DECLARATION_RE.search(prompt)
        prefix = prompt[: declaration_match.start()] if declaration_match else ""
        context.extend(
            line.rstrip()
            for line in prefix.splitlines()
            if line.strip() and IMPORT_RE.match(line) is None
        )
    return statement.strip(), tuple(imports), tuple(dict.fromkeys(context))


def read_rollout_records(
    path: str | Path,
    *,
    rollout_kind: str,
    limit: int | None = None,
) -> list[BenchmarkRecord]:
    """Read minimal final SFT/GRPO rows without exposing an SFT completion."""

    if rollout_kind not in {"sft", "grpo"}:
        raise ValueError("rollout_kind must be sft or grpo")
    records: list[BenchmarkRecord] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(_iter_jsonl(path)):
        if rollout_kind == "grpo":
            leaked = [
                key
                for key in ("proof", "completion", "reference_proof")
                if str(row.get(key) or "").strip()
            ]
            if leaked:
                raise ValueError(f"GRPO final row contains proof fields: {leaked}")
        prompt = str(row.get("prompt") or "")
        statement, imports, context = _prompt_components(row)
        statement_digest = hashlib.sha256(statement.encode("utf-8")).hexdigest()
        problem_id = str(
            row.get("id")
            or row.get("record_id")
            or f"{rollout_kind}-{statement_digest[:20]}"
        )
        if problem_id in seen_ids:
            problem_id = f"{problem_id}-{row_index}"
        seen_ids.add(problem_id)
        records.append(
            BenchmarkRecord(
                problem_id=problem_id,
                prompt=prompt,
                lean_statement=statement,
                imports=imports,
                context_lines=context,
                statement_hash=statement_digest,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_argument_parser()
    parser.description = (
        "Prover-only SFT/GRPO rollout with parallel Pantograph verification."
    )
    parser.add_argument("--rollout_kind", choices=("sft", "grpo"), required=True)
    parser.add_argument(
        "--data_file",
        default=None,
        help="Canonical final_data JSONL. Defaults to sft_data.jsonl or grpo_data.jsonl.",
    )
    parser.add_argument("--num_rollout_samples", type=int, default=None)
    args = parser.parse_args(argv)
    args.imports = tuple(item.strip() for item in args.imports.split(",") if item.strip())
    if args.pass_k < 1:
        parser.error("--pass_k must be positive")
    repo_root = Path(__file__).resolve().parents[3]
    if args.data_file is None:
        args.data_file = str(
            repo_root
            / "lean_prover"
            / "Dataset"
            / "final_data"
            / f"{args.rollout_kind}_data.jsonl"
        )
    args.benchmark_file = args.data_file
    args.num_benchmark_samples = args.num_rollout_samples
    args.workflow_kind = f"{args.rollout_kind}_rollout"
    return args


def main() -> None:
    args = parse_args()
    records = read_rollout_records(
        args.data_file,
        rollout_kind=args.rollout_kind,
        limit=args.num_rollout_samples,
    )
    summary = run_pipeline(args, records=records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
