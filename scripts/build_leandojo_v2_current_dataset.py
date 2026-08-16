"""Build and sample the current-mathlib LeanDojo-v2 candidate dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.data.leandojo_v2_current import (
    ADAPTER_VERSION,
    CURRENT_ENVIRONMENT_HASH,
    CURRENT_LEAN_VERSION,
    CURRENT_MATHLIB_COMMIT,
    LEANDOJO_V2_COMMIT,
    iter_current_mathlib_records,
    quarantine_reason,
    sha256_text,
    validate_current_record,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - index) + ordered[upper] * (index - lower)
    )


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values, default=0.0),
    }


def _successful_source_files(paths: list[Path]) -> list[str]:
    source_files: list[str] = []
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8")):
            if (
                int(item.get("returncode", 1)) == 0
                and item.get("ast_exists")
                and item.get("dep_paths_exists")
            ):
                source_files.append(str(item["source_file"]).replace("\\", "/"))
    return list(dict.fromkeys(source_files))


def _balanced_sample(
    records: list[dict[str, Any]],
    *,
    sample_size: int,
    max_per_file: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_file[str(row["source_file"])].append(row)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for source_file, rows in by_file.items():
        rows = list(rows)
        rng.shuffle(rows)
        # Interleave proof styles and proof-length bands before the file cap.
        rows.sort(
            key=lambda row: (
                str(row["proof_style"]),
                int(row["metadata"]["proof_token_estimate"]) // 64,
                rng.random(),
            )
        )
        queues[source_file] = deque(rows[:max_per_file])

    file_order = sorted(queues)
    rng.shuffle(file_order)
    selected: list[dict[str, Any]] = []
    while file_order and len(selected) < sample_size:
        next_order: list[str] = []
        for source_file in file_order:
            queue = queues[source_file]
            if queue and len(selected) < sample_size:
                selected.append(queue.popleft())
            if queue:
                next_order.append(source_file)
        file_order = next_order
    if len(selected) != sample_size:
        raise RuntimeError(
            f"only {len(selected)} records remain after the "
            f"{max_per_file}-per-file cap; need {sample_size}"
        )
    return selected


def _statistics(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    proof_tokens = [
        float(row["metadata"]["proof_token_estimate"]) for row in rows
    ]
    statement_lengths = [float(len(str(row["statement"]))) for row in rows]
    tactic_steps = [float(len(row.get("tactic_trace") or [])) for row in rows]
    premise_counts = [float(len(row.get("premises") or [])) for row in rows]
    same_file = sum(
        1
        for row in rows
        for premise in row.get("premises") or []
        if premise.get("is_same_file")
    )
    all_premises = sum(len(row.get("premises") or []) for row in rows)
    file_counts = Counter(str(row["source_file"]) for row in rows)
    module_counts = Counter(
        str(row["source_file"]).split("/", 2)[1]
        if "/" in str(row["source_file"])
        else str(row["source_file"])
        for row in rows
    )
    tactic_records = [
        row for row in rows if row.get("proof_style") in {"tactic", "mixed"}
    ]
    single_tactic = sum(
        len(row.get("tactic_trace") or []) == 1 for row in tactic_records
    )
    multi_tactic = sum(
        len(row.get("tactic_trace") or []) > 1 for row in tactic_records
    )
    pair_hashes = [
        sha256_text(f"{row['statement']}\0{row['proof']}") for row in rows
    ]
    return {
        "seed": seed,
        "record_count": len(rows),
        "source_file_count": len(file_counts),
        "source_file_distribution": dict(sorted(file_counts.items())),
        "top_level_module_distribution": dict(sorted(module_counts.items())),
        "proof_token_estimator": "lean-lexeme-regex-v1",
        "proof_token_distribution": _distribution(proof_tokens),
        "statement_character_distribution": _distribution(statement_lengths),
        "tactic_step_distribution": _distribution(tactic_steps),
        "premise_count_distribution": _distribution(premise_counts),
        "same_file_premise_ratio": (
            same_file / all_premises if all_premises else 0.0
        ),
        "proof_style_distribution": dict(
            sorted(Counter(str(row["proof_style"]) for row in rows).items())
        ),
        "single_tactic_ratio_among_tactic_records": (
            single_tactic / len(tactic_records) if tactic_records else 0.0
        ),
        "multi_step_ratio_among_tactic_records": (
            multi_tactic / len(tactic_records) if tactic_records else 0.0
        ),
        "duplicate_id_count": len(rows) - len({row["id"] for row in rows}),
        "duplicate_qualified_name_count": len(rows)
        - len({row["qualified_name"] for row in rows}),
        "duplicate_statement_proof_pair_count": len(rows) - len(set(pair_hashes)),
    }


def build_dataset(
    *,
    repo_root: Path,
    trace_results: list[Path],
    output_root: Path,
    sample_size: int,
    max_per_file: int,
    seed: int,
) -> dict[str, Any]:
    source_files = _successful_source_files(trace_results)
    candidates: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_pairs: set[str] = set()
    traced_declarations = 0
    for record in iter_current_mathlib_records(
        repo_root=repo_root,
        source_files=source_files,
    ):
        traced_declarations += 1
        schema_errors = validate_current_record(record)
        reason = (
            "schema_error:" + ",".join(schema_errors)
            if schema_errors
            else quarantine_reason(record)
        )
        pair_hash = sha256_text(f"{record['statement']}\0{record['proof']}")
        if reason is None and record["qualified_name"] in seen_names:
            reason = "duplicate_qualified_name"
        if reason is None and pair_hash in seen_pairs:
            reason = "duplicate_statement_proof_pair"
        if reason is None:
            seen_names.add(record["qualified_name"])
            seen_pairs.add(pair_hash)
            candidates.append(record)
        else:
            quarantined.append(
                {
                    **record,
                    "metadata": {
                        **record["metadata"],
                        "quarantine_reason": reason,
                    },
                }
            )

    processed = output_root / "processed"
    _write_jsonl(processed / "current_mathlib_candidates.jsonl", candidates)
    _write_jsonl(
        processed / "current_mathlib_quarantined.jsonl", quarantined
    )
    sample = _balanced_sample(
        candidates,
        sample_size=sample_size,
        max_per_file=max_per_file,
        seed=seed,
    )
    sample_dir = output_root / "sample500"
    _write_json(sample_dir / "sample_ids.json", [row["id"] for row in sample])
    _write_jsonl(sample_dir / "sample_manifest.jsonl", sample)
    statistics_payload = _statistics(sample, seed)
    statistics_payload.update(
        {
            "sampling_method": "seeded_without_replacement_round_robin_by_file",
            "max_records_per_source_file": max_per_file,
            "candidate_count": len(candidates),
            "quarantined_count": len(quarantined),
            "traced_declaration_count": traced_declarations,
            "quarantine_reason_counts": dict(
                sorted(
                    Counter(
                        row["metadata"]["quarantine_reason"]
                        for row in quarantined
                    ).items()
                )
            ),
            "environment": {
                "mathlib_commit": CURRENT_MATHLIB_COMMIT,
                "lean_version": CURRENT_LEAN_VERSION,
                "leandojo_v2_commit": LEANDOJO_V2_COMMIT,
                "environment_hash": CURRENT_ENVIRONMENT_HASH,
                "adapter_version": ADAPTER_VERSION,
            },
        }
    )
    _write_json(sample_dir / "sample_statistics.json", statistics_payload)
    lines = [
        "# LeanDojo-v2 current-mathlib sample statistics",
        "",
        f"- Seed: `{seed}`",
        f"- Sampling: without replacement, round-robin by source file",
        f"- Per-file cap: `{max_per_file}`",
        f"- Traced declarations: `{traced_declarations}`",
        f"- Eligible candidates: `{len(candidates)}`",
        f"- Quarantined: `{len(quarantined)}`",
        f"- Sample size: `{len(sample)}`",
        f"- Source files: `{statistics_payload['source_file_count']}`",
        "",
        "## Proof tokens",
        "",
        *[
            f"- {key}: `{value:.2f}`"
            for key, value in statistics_payload[
                "proof_token_distribution"
            ].items()
        ],
        "",
        "## Proof styles",
        "",
        *[
            f"- {key}: `{value}`"
            for key, value in statistics_payload[
                "proof_style_distribution"
            ].items()
        ],
        "",
        "## Quarantine reasons",
        "",
        *[
            f"- {key}: `{value}`"
            for key, value in statistics_payload[
                "quarantine_reason_counts"
            ].items()
        ],
    ]
    (sample_dir / "sample_statistics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return statistics_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--trace-results", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--max-per-file", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    summary = build_dataset(
        repo_root=args.repo_root,
        trace_results=args.trace_results,
        output_root=args.output_root,
        sample_size=args.sample_size,
        max_per_file=args.max_per_file,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
