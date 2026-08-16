#!/usr/bin/env python3
"""Freeze a stratified LeanDojo review pool and generate classification dossiers.

This script deliberately does not assign difficulty labels. It only prepares the
material that Codex must read before making each final judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TACTIC_NAMES = (
    "simp",
    "simpa",
    "rw",
    "exact",
    "apply",
    "rfl",
    "norm_num",
    "ring",
    "ring_nf",
    "linarith",
    "nlinarith",
    "aesop",
    "omega",
    "constructor",
    "cases",
    "induction",
    "refine",
    "have",
    "show",
    "suffices",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_statement(row: dict[str, Any]) -> str:
    value = str(
        row.get("lean_statement")
        or row.get("statement")
        or row.get("training_statement")
        or row.get("statement_hash_normalized")
        or row.get("statement_hash")
        or ""
    )
    return re.sub(r"\s+", " ", value).strip()


def default_timeout_verified(row: dict[str, Any]) -> bool:
    """Read the frozen verification receipt, not the stale retrace-era flag."""

    verification = row.get("verification")
    return bool(
        row.get("verification_status") == "verified_default_timeout"
        and isinstance(verification, dict)
        and verification.get("compile_success") is True
        and verification.get("timed_out") is False
        and not verification.get("error_category")
    )


def source_domain(row: dict[str, Any]) -> str:
    source_file = str(row.get("source_file") or "")
    parts = source_file.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "Mathlib" else "unknown"


def bin_value(value: int, *, low: int, high: int) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def tactic_signature(row: dict[str, Any]) -> list[str]:
    proof = str(row.get("proof") or row.get("training_proof") or "")
    result = [
        tactic for tactic in TACTIC_NAMES if re.search(rf"\b{tactic}\b", proof)
    ]
    return result or ["term_or_other"]


def review_metrics(row: dict[str, Any]) -> dict[str, Any]:
    proof = str(row.get("proof") or row.get("training_proof") or "")
    statement = str(row.get("statement") or row.get("training_statement") or "")
    premises = list(row.get("premises") or [])
    trace = list(row.get("tactic_trace") or [])
    same_file = sum(bool(item.get("is_same_file")) for item in premises)
    proof_tokens = int(
        row.get("label_tokens")
        or (row.get("metadata") or {}).get("proof_token_estimate")
        or len(proof.split())
    )
    statement_tokens = int(row.get("statement_tokens") or len(statement.split()))
    statement_structure = (
        statement.count("→")
        + statement.count("∀")
        + statement.count("∃")
        + statement.count(":=")
        + len(re.findall(r"\([^)]*:[^)]*\)", statement))
    )
    tactics = tactic_signature(row)
    typeclass_binders = len(re.findall(r"\[[^\]]+\]", statement))
    implicit_binders = len(re.findall(r"\{[^}]+\}", statement))
    coercion_cues = statement.count("↑") + proof.count("↑")
    return {
        "proof_tokens": proof_tokens,
        "statement_tokens": statement_tokens,
        "tactic_steps": len(trace),
        "premise_count": len(premises),
        "same_file_premise_count": same_file,
        "proof_style": str(row.get("proof_style") or "unknown"),
        "mathlib_domain": source_domain(row),
        "source_file": str(row.get("source_file") or "unknown"),
        "tactic_signature": tactics,
        "statement_structure_count": statement_structure,
        "typeclass_binder_count": typeclass_binders,
        "implicit_binder_count": implicit_binders,
        "coercion_cue_count": coercion_cues,
        "proof_length_bin": bin_value(proof_tokens, low=12, high=40),
        "statement_length_bin": bin_value(statement_tokens, low=24, high=60),
        "tactic_step_bin": bin_value(len(trace), low=0, high=2),
        "premise_count_bin": bin_value(len(premises), low=2, high=7),
        "has_same_file_premise": bool(same_file),
    }


def protected_identities(root: Path) -> tuple[set[str], set[str]]:
    paths = [
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
    ]
    ids: set[str] = set()
    statements: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            for key in ("id", "record_id", "theorem_group_id", "qualified_name"):
                if row.get(key):
                    ids.add(str(row[key]))
            statement = normalized_statement(row)
            if statement:
                statements.add(statement)
    return ids, statements


def select_review_pool(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    metrics_by_id = {
        str(row["id"]): review_metrics(row)
        for row in rows
    }
    tactic_frequency = Counter(
        tactic
        for row in rows
        for tactic in metrics_by_id[str(row["id"])]["tactic_signature"]
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # Guarantee source-file breadth before filling multi-dimensional strata.
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[metrics_by_id[str(row["id"])]["source_file"]].append(row)
    for source_file, values in sorted(by_file.items()):
        rng.shuffle(values)
        row = min(
            values,
            key=lambda item: (
                tactic_frequency[
                    metrics_by_id[str(item["id"])]["tactic_signature"][0]
                ],
                metrics_by_id[str(item["id"])]["proof_tokens"],
                str(item["id"]),
            ),
        )
        selected.append(row)
        selected_ids.add(str(row["id"]))

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["id"]) in selected_ids:
            continue
        metrics = metrics_by_id[str(row["id"])]
        rare_tactic = min(
            metrics["tactic_signature"],
            key=lambda tactic: (tactic_frequency[tactic], tactic),
        )
        signature = (
            metrics["mathlib_domain"],
            metrics["proof_style"],
            metrics["proof_length_bin"],
            metrics["statement_length_bin"],
            metrics["tactic_step_bin"],
            metrics["premise_count_bin"],
            metrics["has_same_file_premise"],
            rare_tactic,
        )
        buckets[signature].append(row)
    keys = sorted(buckets, key=repr)
    rng.shuffle(keys)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    while keys and len(selected) < count:
        remaining = []
        for key in keys:
            if len(selected) >= count:
                break
            row = buckets[key].pop()
            selected.append(row)
            selected_ids.add(str(row["id"]))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    if len(selected) != count:
        raise ValueError(f"selected {len(selected)} rows, expected {count}")
    rng.shuffle(selected)
    return selected


def render_dossier(row: dict[str, Any], metrics: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    recovery = metadata.get("context_recovery") or {}
    premises = list(row.get("premises") or [])
    trace = list(row.get("tactic_trace") or [])
    lines = [
        f"# Classification dossier: {row['id']}",
        "",
        "## Identity",
        "",
        f"- Sample ID: `{row['id']}`",
        f"- Qualified name: `{row.get('qualified_name') or 'unknown'}`",
        f"- Source file: `{row.get('source_file') or 'unknown'}`",
        f"- Mathlib domain: `{metrics['mathlib_domain']}`",
        f"- Namespace stack: `{row.get('namespace_stack') or []}`",
        f"- Verification status: `{row.get('verification_status')}`",
        "- Pantograph timeout policy: `30 seconds (default)`",
        f"- Verification time: `{(row.get('verification') or {}).get('elapsed_seconds')}` seconds",
        "",
        "## Complete theorem statement",
        "",
        "```lean",
        str(row.get("statement") or row.get("training_statement") or ""),
        "```",
        "",
        "## Complete reference proof",
        "",
        "```lean",
        str(row.get("proof") or row.get("training_proof") or ""),
        "```",
        "",
        "## Auxiliary metrics (not an automatic label)",
        "",
        f"- Proof style: `{metrics['proof_style']}`",
        f"- Proof tokens: `{metrics['proof_tokens']}`",
        f"- Statement tokens: `{metrics['statement_tokens']}`",
        f"- Tactic steps: `{metrics['tactic_steps']}`",
        f"- Premises: `{metrics['premise_count']}`",
        f"- Same-file premises: `{metrics['same_file_premise_count']}`",
        f"- Tactic signature: `{metrics['tactic_signature']}`",
        f"- Typeclass binder cues: `{metrics['typeclass_binder_count']}`",
        f"- Implicit binder cues: `{metrics['implicit_binder_count']}`",
        f"- Explicit coercion cues: `{metrics['coercion_cue_count']}`",
        "- These counts are review aids, not a typeclass/coercion difficulty judgment.",
        "",
        "## Tactic trace",
        "",
    ]
    if trace:
        for index, step in enumerate(trace, 1):
            lines.extend(
                [
                    f"### Step {index}",
                    "",
                    f"- Tactic: `{step.get('tactic') or step.get('tactic_code') or step}`",
                    "",
                    "State before:",
                    "```text",
                    str(step.get("state_before") or "unavailable"),
                    "```",
                    "State after:",
                    "```text",
                    str(step.get("state_after") or "unavailable"),
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(["No tactic trace (term-style proof or unavailable trace).", ""])
    lines.extend(["## Premises", ""])
    if premises:
        for premise in premises:
            lines.append(
                "- "
                f"`{premise.get('qualified_name')}`; "
                f"source=`{premise.get('source_file')}`; "
                f"same_file={bool(premise.get('is_same_file'))}; "
                f"type=`{premise.get('statement') or premise.get('type') or 'unavailable'}`"
            )
    else:
        lines.append("No recorded premises.")
    lines.extend(
        [
            "",
            "## Recovered context",
            "",
            f"- Imports: `{row.get('imports') or recovery.get('imports') or []}`",
            f"- Sections: `{row.get('section_context') or recovery.get('sections') or []}`",
            f"- Open namespaces: `{row.get('open_namespaces') or recovery.get('open_namespaces') or []}`",
            f"- Open scopes: `{row.get('open_scopes') or recovery.get('scopes') or []}`",
            f"- Variables/hypotheses: `{row.get('variables') or recovery.get('variables') or []}`",
            f"- Explicit hypotheses: `{row.get('hypotheses') or []}`",
            f"- Local notations: `{row.get('local_notations') or recovery.get('local_notations') or []}`",
            f"- Local instances: `{row.get('local_instances') or recovery.get('local_instances') or []}`",
            f"- Local attributes: `{row.get('local_attributes') or recovery.get('local_attributes') or []}`",
            f"- Active commands: `{recovery.get('active_commands') or []}`",
            f"- Context recovery status: `{recovery.get('recovery_status') or 'unknown'}`",
            "",
            "## Codex judgment (to be completed by actual review)",
            "",
            "- Difficulty:",
            "- Confidence:",
            "- Reason summary:",
            "- Statement complexity evidence:",
            "- Proof structure evidence:",
            "- Tactic complexity evidence:",
            "- Premise dependency evidence:",
            "- Context dependency evidence:",
            "- Typeclass/coercion complexity evidence:",
            "- Fit for subgoal whole-proof evidence:",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/leandojo_v2_dataset_build/final_pool/"
            "leandojo_v2_final_train_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/difficulty"),
    )
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    root = args.root.resolve()
    input_path = (root / args.input).resolve()
    output = (root / args.output).resolve()
    rows = read_jsonl(input_path)
    protected_ids, protected_statements = protected_identities(root)
    eligible = []
    excluded_overlap = 0
    excluded_status = 0
    for row in rows:
        identities = {
            str(row.get(key))
            for key in ("id", "theorem_group_id", "qualified_name")
            if row.get(key)
        }
        if identities & protected_ids or normalized_statement(row) in protected_statements:
            excluded_overlap += 1
            continue
        if not default_timeout_verified(row):
            excluded_status += 1
            continue
        eligible.append(row)
    selected = select_review_pool(eligible, count=args.count, seed=args.seed)
    frozen_path = output / "frozen_classification_pool.jsonl"
    write_jsonl(frozen_path, selected)
    digest = sha256(frozen_path)
    (output / "frozen_classification_pool.sha256").write_text(
        f"{digest}  {frozen_path.name}\n", encoding="utf-8"
    )
    metrics_rows = []
    for index, row in enumerate(selected):
        metrics = review_metrics(row)
        metrics_rows.append({"sample_id": row["id"], **metrics})
        dossier_path = output / "dossiers" / f"{row['id']}.md"
        dossier_path.parent.mkdir(parents=True, exist_ok=True)
        dossier_path.write_text(render_dossier(row, metrics), encoding="utf-8")
    write_jsonl(output / "review_metrics.jsonl", metrics_rows)
    for batch_index in range(0, len(selected), 25):
        batch = []
        for row in selected[batch_index : batch_index + 25]:
            metrics = review_metrics(row)
            batch.append(
                {
                    "sample_id": row["id"],
                    "qualified_name": row.get("qualified_name"),
                    "source_file": row.get("source_file"),
                    "mathlib_domain": metrics["mathlib_domain"],
                    "statement": row.get("statement"),
                    "proof": row.get("proof"),
                    "proof_style": metrics["proof_style"],
                    "proof_tokens": metrics["proof_tokens"],
                    "statement_tokens": metrics["statement_tokens"],
                    "tactic_steps": metrics["tactic_steps"],
                    "tactic_trace": row.get("tactic_trace"),
                    "premises": row.get("premises"),
                    "same_file_premise_count": metrics[
                        "same_file_premise_count"
                    ],
                    "imports": row.get("imports"),
                    "namespace_stack": row.get("namespace_stack"),
                    "variables": row.get("variables"),
                    "hypotheses": row.get("hypotheses"),
                    "open_namespaces": row.get("open_namespaces"),
                    "open_scopes": row.get("open_scopes"),
                    "local_notations": row.get("local_notations"),
                    "local_instances": row.get("local_instances"),
                    "local_attributes": row.get("local_attributes"),
                    "context_recovery": (row.get("metadata") or {}).get(
                        "context_recovery"
                    ),
                    "verification_status": row.get("verification_status"),
                    "verification": row.get("verification"),
                    "typeclass_binder_count": metrics[
                        "typeclass_binder_count"
                    ],
                    "implicit_binder_count": metrics["implicit_binder_count"],
                    "coercion_cue_count": metrics["coercion_cue_count"],
                }
            )
        write_jsonl(
            output / "review_batches" / f"batch_{batch_index // 25:03d}.jsonl",
            batch,
        )
    metric_lookup = {str(row["sample_id"]): row for row in metrics_rows}
    distributions = {
        "domains": dict(
            Counter(metric_lookup[str(row["id"])]["mathlib_domain"] for row in selected)
        ),
        "source_files": dict(
            Counter(metric_lookup[str(row["id"])]["source_file"] for row in selected)
        ),
        "proof_styles": dict(
            Counter(metric_lookup[str(row["id"])]["proof_style"] for row in selected)
        ),
        "proof_length_bins": dict(
            Counter(
                metric_lookup[str(row["id"])]["proof_length_bin"] for row in selected
            )
        ),
        "tactic_step_bins": dict(
            Counter(
                metric_lookup[str(row["id"])]["tactic_step_bin"] for row in selected
            )
        ),
        "premise_count_bins": dict(
            Counter(
                metric_lookup[str(row["id"])]["premise_count_bin"] for row in selected
            )
        ),
        "same_file_dependency": dict(
            Counter(
                str(metric_lookup[str(row["id"])]["has_same_file_premise"])
                for row in selected
            )
        ),
    }
    report = [
        "# Frozen manual difficulty-classification pool",
        "",
        f"- Source candidates: {len(rows)}",
        f"- Eligible after protected-set/status audit: {len(eligible)}",
        f"- Selected for actual Codex review: {len(selected)}",
        f"- Protected overlaps excluded: {excluded_overlap}",
        f"- Non-default-verification rows excluded: {excluded_status}",
        f"- Source files covered: {len(distributions['source_files'])}",
        f"- Mathlib domains covered: {len(distributions['domains'])}",
        f"- SHA-256: `{digest}`",
        "",
        "Selection used stratification only. No difficulty label has been assigned "
        "by this script.",
        "",
        "```json",
        json.dumps(distributions, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (output / "sampling_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    frozen_path.chmod(0o444)
    print(
        json.dumps(
            {
                "selected": len(selected),
                "sha256": digest,
                "dossiers": len(selected),
                "review_batches": (len(selected) + 24) // 25,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
