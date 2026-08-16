#!/usr/bin/env python3
"""Validate Codex-authored difficulty labels and materialize audited pools.

This script never infers or changes a difficulty label.  Every label and every
trainability decision must already be present in ``difficulty_labels.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


DIFFICULTIES = ("easy", "medium", "difficult", "uncertain", "exclude")
CONFIDENCES = ("high", "medium", "low")
EVIDENCE_FIELDS = (
    "statement_complexity",
    "proof_structure",
    "tactic_complexity",
    "premise_dependency",
    "context_dependency",
    "fit_for_subgoal_whole_proof",
)


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


def normalized_statement(row: dict[str, Any]) -> str:
    value = str(
        row.get("statement")
        or row.get("training_statement")
        or row.get("lean_statement")
        or ""
    )
    return re.sub(r"\s+", " ", value).strip()


def default_timeout_verified(row: dict[str, Any]) -> bool:
    """Validate the immutable nested Pantograph receipt in the final pool."""

    verification = row.get("verification")
    return bool(
        row.get("verification_status") == "verified_default_timeout"
        and isinstance(verification, dict)
        and verification.get("compile_success") is True
        and verification.get("timed_out") is False
        and not verification.get("error_category")
    )


def protected_identities(root: Path) -> tuple[set[str], set[str]]:
    paths = (
        root
        / "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl",
        root / "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        root / "outputs/expert_sft_anchor_ablation/gates/discovery_gate_150.jsonl",
        root
        / "outputs/expert_sft_anchor_ablation/gates_stratified/anchor_gate_150.jsonl",
        root
        / "outputs/expert_sft_anchor_ablation/gates_stratified/discovery_gate_150.jsonl",
        root / "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
        root / "outputs/b2_expanded_validation/datasets/benchmark_minif2f_test_96.jsonl",
        root / "outputs/b2_expanded_validation/datasets/full500.jsonl",
        root / "outputs/b2_expanded_validation/datasets/strict_unseen_discovery.jsonl",
    )
    identities: set[str] = set()
    statements: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            for key in ("id", "record_id", "theorem_group_id", "qualified_name"):
                if row.get(key):
                    identities.add(str(row[key]))
            statement = normalized_statement(row)
            if statement:
                statements.add(statement)
    return identities, statements


def percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return float(ordered[round((len(ordered) - 1) * quantile)])


def distribution(labels: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        rows = [row for row in labels if row["difficulty"] == difficulty]
        proof_tokens = [int(row["metrics"]["proof_tokens"]) for row in rows]
        tactic_steps = [int(row["metrics"]["tactic_steps"]) for row in rows]
        premise_counts = [int(row["metrics"]["premise_count"]) for row in rows]
        result[difficulty] = {
            "count": len(rows),
            "proof_tokens": {
                "mean": statistics.fmean(proof_tokens) if proof_tokens else 0.0,
                "p50": percentile(proof_tokens, 0.50),
                "p90": percentile(proof_tokens, 0.90),
            },
            "tactic_steps": {
                "mean": statistics.fmean(tactic_steps) if tactic_steps else 0.0,
                "p50": percentile(tactic_steps, 0.50),
                "p90": percentile(tactic_steps, 0.90),
            },
            "premise_count": {
                "mean": statistics.fmean(premise_counts) if premise_counts else 0.0,
                "p50": percentile(premise_counts, 0.50),
                "p90": percentile(premise_counts, 0.90),
            },
            "proof_style": dict(
                Counter(str(row["metrics"]["proof_style"]) for row in rows)
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--difficulty-dir",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/difficulty"),
    )
    args = parser.parse_args()
    project = args.root.resolve()
    output = (project / args.difficulty_dir).resolve()
    pool_path = output / "frozen_classification_pool.jsonl"
    labels_path = output / "difficulty_labels.jsonl"
    pool = read_jsonl(pool_path)
    labels = read_jsonl(labels_path)
    pool_by_id = {str(row["id"]): row for row in pool}
    if len(pool_by_id) != len(pool):
        raise ValueError("frozen classification pool contains duplicate sample IDs")
    label_ids = [str(row.get("sample_id") or "") for row in labels]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("a frozen sample has more than one difficulty label")
    if set(label_ids) - set(pool_by_id):
        raise ValueError("difficulty labels contain IDs outside the frozen pool")
    versions = {str(row.get("classification_version") or "") for row in labels}
    if len(versions) != 1 or "" in versions:
        raise ValueError(f"classification version is missing or mixed: {versions}")
    metrics_by_id = {
        str(row["sample_id"]): row
        for row in read_jsonl(output / "review_metrics.jsonl")
    }
    problems: list[str] = []
    for label in labels:
        sample_id = str(label.get("sample_id") or "")
        if label.get("difficulty") not in DIFFICULTIES:
            problems.append(f"{sample_id}: invalid difficulty")
        if label.get("confidence") not in CONFIDENCES:
            problems.append(f"{sample_id}: invalid confidence")
        if label.get("reviewed_by") != "codex":
            problems.append(f"{sample_id}: reviewed_by must be codex")
        if not str(label.get("classified_at") or "").strip():
            problems.append(f"{sample_id}: classified_at is missing")
        reason = str(label.get("reason_summary") or "").strip()
        if len(reason) < 40:
            problems.append(f"{sample_id}: reason_summary is not specific enough")
        evidence = label.get("evidence") or {}
        for field in EVIDENCE_FIELDS:
            if len(str(evidence.get(field) or "").strip()) < 15:
                problems.append(f"{sample_id}: incomplete evidence.{field}")
        if not isinstance(label.get("trainable_for_short_whole_proof"), bool):
            problems.append(f"{sample_id}: explicit trainability judgment missing")
        elif (
            label["trainable_for_short_whole_proof"]
            and label.get("difficulty") != "easy"
        ):
            problems.append(
                f"{sample_id}: only a manually labeled easy sample can be trainable"
            )
        expected_metrics = metrics_by_id.get(sample_id) or {}
        supplied_metrics = label.get("metrics") or {}
        for field in (
            "statement_tokens",
            "proof_tokens",
            "tactic_steps",
            "premise_count",
            "same_file_premise_count",
            "proof_style",
        ):
            if supplied_metrics.get(field) != expected_metrics.get(field):
                problems.append(f"{sample_id}: metrics.{field} drift")
    if problems:
        write_json(output / "label_validation_errors.json", problems)
        raise ValueError(
            f"{len(problems)} label validation errors; see label_validation_errors.json"
        )

    protected_ids, protected_statements = protected_identities(project)
    merged_by_difficulty: dict[str, list[dict[str, Any]]] = {
        value: [] for value in DIFFICULTIES
    }
    hard_leaks: list[str] = []
    for label in labels:
        sample_id = str(label["sample_id"])
        source = pool_by_id[sample_id]
        identities = {
            str(source.get(key))
            for key in ("id", "theorem_group_id", "qualified_name")
            if source.get(key)
        }
        if identities & protected_ids or normalized_statement(source) in protected_statements:
            hard_leaks.append(sample_id)
        merged = json.loads(json.dumps(source))
        merged["difficulty_annotation"] = label
        merged_by_difficulty[label["difficulty"]].append(merged)
    if hard_leaks:
        raise ValueError(f"hard evaluation leakage detected: {hard_leaks[:10]}")
    filenames = {
        "easy": "easy_pool.jsonl",
        "medium": "medium_pool.jsonl",
        "difficult": "difficult_pool.jsonl",
        "uncertain": "uncertain_pool.jsonl",
        "exclude": "excluded_pool.jsonl",
    }
    for difficulty, rows in merged_by_difficulty.items():
        write_jsonl(output / filenames[difficulty], rows)

    easy_labels = [row for row in labels if row["difficulty"] == "easy"]
    trainable_easy = [
        row
        for row in easy_labels
        if row["trainable_for_short_whole_proof"]
    ]
    easy_sources = {
        str(pool_by_id[row["sample_id"]].get("source_file") or "")
        for row in trainable_easy
    }
    easy_domains = {
        str(metrics_by_id[row["sample_id"]].get("mathlib_domain") or "unknown")
        for row in trainable_easy
    }
    verification_ok = all(
        default_timeout_verified(pool_by_id[row["sample_id"]])
        for row in trainable_easy
    )
    audit_path = output / "consistency_audit.json"
    consistency = read_json(audit_path) if audit_path.is_file() else {}
    consistency_counts = consistency.get("random_rereview_counts") or {}
    gate_checks = {
        "reviewed_at_least_300": len(labels) >= 300,
        "easy_at_least_180": len(easy_labels) >= 180,
        "trainable_easy_at_least_160": len(trainable_easy) >= 160,
        "easy_source_files_at_least_10": len(easy_sources) >= 10,
        "easy_domains_at_least_5": len(easy_domains) >= 5,
        "all_trainable_easy_default_timeout_verified": verification_ok,
        "zero_hard_evaluation_leaks": not hard_leaks,
        "all_label_reasons_complete": not problems,
        "consistency_audit_passed": bool(consistency.get("passed")),
        "consistency_version_matches": (
            consistency.get("classification_version") == next(iter(versions))
        ),
        "consistency_rereview_easy_at_least_20": (
            int(consistency_counts.get("easy") or 0) >= 20
        ),
        "consistency_rereview_medium_at_least_20": (
            int(consistency_counts.get("medium") or 0) >= 20
        ),
        "consistency_rereview_difficult_at_least_20": (
            int(consistency_counts.get("difficult") or 0) >= 20
        ),
    }
    summary = {
        "classification_version": next(iter(versions)),
        "frozen_pool_sha256": sha256(pool_path),
        "labels_sha256": sha256(labels_path),
        "reviewed": len(labels),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in labels)),
        "trainable_easy": len(trainable_easy),
        "trainable_easy_source_files": len(easy_sources),
        "trainable_easy_domains": len(easy_domains),
        "distributions": distribution(labels),
        "gate_checks": gate_checks,
        "pilot_gate_passed": all(gate_checks.values()),
    }
    write_json(output / "difficulty_summary.json", summary)
    write_json(output / "pilot_gate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
