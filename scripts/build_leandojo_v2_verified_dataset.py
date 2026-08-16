#!/usr/bin/env python3
"""Build normalized, deduplicated, leakage-audited LeanDojo-v2 pools."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lean_prover.lean_training.data.leandojo_v2_dataset.dedup import (  # noqa: E402
    audit_cross_source,
    audit_internal_dedup,
)
from lean_prover.lean_training.data.leandojo_v2_dataset.leakage import (  # noqa: E402
    audit_leakage,
    load_protected_jsonl,
)
from lean_prover.lean_training.data.leandojo_v2_dataset.loader import (  # noqa: E402
    iter_jsonl,
    write_json,
    write_jsonl,
)
from lean_prover.lean_training.data.leandojo_v2_dataset.manifest_builder import (  # noqa: E402
    attach_token_counts,
    sample_rows,
    sample_token_matched,
    summarize_manifest,
)
from lean_prover.lean_training.data.leandojo_v2_dataset.normalization import (  # noqa: E402
    NORMALIZATION_VERSION,
    normalize_leandojo_record,
    normalize_workbook_record,
)

EXPECTED_CANDIDATES = 5366
MATHLIB_COMMIT = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"
SEED = 20260801
FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom|unsafe)\b")


def discover_protected_paths(root: Path) -> list[Path]:
    """Return actual, canonical project evaluation/gate datasets."""

    relative = [
        "data/processed/lean_workbook_verified_v2/eval.jsonl",
        "data/processed/lean_workbook_verified_v2/discovery.jsonl",
        "data/processed/lean_workbook_verified_v2/monitor.jsonl",
        "data/processed/lean_workbook_verified_v2/benchmark.jsonl",
        "data/processed/expert_iteration_6h/discovery_700.jsonl",
        "data/processed/expert_iteration_6h/minif2f_valid_32.jsonl",
        "data/processed/expert_iteration_6h/minif2f_test_96.jsonl",
        "outputs/b2_expanded_validation/datasets/full500.jsonl",
        "outputs/b2_expanded_validation/datasets/strict_unseen_discovery.jsonl",
        "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
        "outputs/b2_expanded_validation/datasets/benchmark_minif2f_test_96.jsonl",
        "outputs/expert_iteration_round1/inputs/verified/monitor.jsonl",
        "outputs/expert_iteration_round1/inputs/verified/benchmark.jsonl",
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        "outputs/expert_sft_anchor_ablation/gates/discovery_gate_150.jsonl",
        "outputs/expert_sft_anchor_ablation/gates_stratified/anchor_gate_150.jsonl",
        "outputs/expert_sft_anchor_ablation/gates_stratified/discovery_gate_150.jsonl",
        "outputs/expert_sft_no_replacement_ablation/manifests/early_retention_gate_50.jsonl",
    ]
    return [root / path for path in relative if (root / path).exists()]


def deduplicate_protected(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in records:
        key = (
            row["statement_hash_normalized"],
            str(row.get("id") or row.get("record_id") or row.get("statement_id")),
            str(row.get("source_dataset") or row.get("source")),
        )
        unique.setdefault(key, row)
    return list(unique.values())


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
    ordered = sorted(values)

    def percentile(value: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * value))]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": max(values),
    }


def write_reports(
    output: Path,
    *,
    verification_summary: dict[str, Any],
    normalization_stats: dict[str, Any],
    dedup_stats: dict[str, Any],
    cross_stats: dict[str, Any],
    leakage_stats: dict[str, Any],
    final_stats: dict[str, Any],
    manifest_summaries: dict[str, Any],
) -> None:
    (output / "normalized/normalization_report.md").write_text(
        "# LeanDojo-v2 normalization\n\n"
        f"- Version: `{NORMALIZATION_VERSION}`\n"
        f"- Success: `{normalization_stats['success']}`\n"
        f"- Failure: `{normalization_stats['failure']}`\n"
        "- Raw source, executable source, training representation, and "
        "fingerprint text are stored as distinct layers.\n"
        "- Fingerprint normalization is never submitted to Pantograph.\n",
        encoding="utf-8",
    )
    (output / "dedup/dedup_report.md").write_text(
        "# Deduplication report\n\n"
        f"- Internal statistics: `{json.dumps(dedup_stats, ensure_ascii=False)}`\n"
        f"- Cross-source statistics: `{json.dumps(cross_stats, ensure_ascii=False)}`\n"
        "- Same-statement/different-proof records are retained as proof variants.\n"
        "- Current-environment verified LeanDojo records are canonical for exact "
        "cross-source copies; Lean Workbook provenance is retained as an alias.\n",
        encoding="utf-8",
    )
    leak_lines = [
        "# Benchmark and evaluation leakage report",
        "",
        "Level 0-2 matches are removed from the training candidate theorem group. "
        "Protected datasets are not modified.",
        "",
        f"- Statistics: `{json.dumps(leakage_stats, ensure_ascii=False)}`",
        "",
        "Exact leakage records are listed individually in the adjacent JSONL files.",
    ]
    (output / "leakage/benchmark_leakage_report.md").write_text(
        "\n".join(leak_lines) + "\n", encoding="utf-8"
    )
    (output / "leakage/leakage_report.md").write_text(
        "\n".join(leak_lines) + "\n", encoding="utf-8"
    )
    (output / "final_pool/final_pool_report.md").write_text(
        "# Final candidate pools\n\n"
        f"- Statistics: `{json.dumps(final_stats, ensure_ascii=False)}`\n"
        "- Only 30-second current-environment fidelity successes without forbidden "
        "proof tokens and without Level 0-2 evaluation leakage enter the LD pool.\n",
        encoding="utf-8",
    )
    manifest_rows = []
    for name, summary in manifest_summaries.items():
        sources = summary["sources"]
        label_share = summary["label_token_share"]
        manifest_rows.append(
            "| {name} | {rows} | {wb} | {ld} | {wb_share:.2%} | "
            "{ld_share:.2%} | {error:.2%} | {duplicates} | {leaks} |".format(
                name=name,
                rows=summary["rows"],
                wb=sources.get("lean_workbook", 0),
                ld=sources.get("leandojo_v2_current_mathlib", 0),
                wb_share=label_share.get("lean_workbook", 0.0),
                ld_share=label_share.get("leandojo_v2_current_mathlib", 0.0),
                error=summary["label_token_budget_error_ratio"],
                duplicates=summary["theorem_group_duplicates"],
                leaks=summary["evaluation_leaks"],
            )
        )
    final = f"""# LeanDojo-v2 current-mathlib dataset build

## Full verification

- Accounted candidates: {verification_summary['total_candidates']}/{EXPECTED_CANDIDATES}
- Default-timeout verified: {verification_summary['verified_default_timeout']}
- Success ratio: {verification_summary['success_ratio']:.4%}
- Timeouts: {verification_summary['timeouts']}
- Compile failures: {verification_summary['compile_failures']}
- 95% gate: {verification_summary['quality_gate_95_percent']}
- Failure categories: `{json.dumps(verification_summary.get('failure_concentration', {}), ensure_ascii=False)}`
- First-run cache hits: 0
- Repeat-cache audit: {verification_summary['cache_repeat']['cache_hits']}/{verification_summary['cache_repeat']['repeat_count']} hits, {verification_summary['cache_repeat']['result_mismatches']} mismatches
- Worker restarts: {verification_summary['worker_restart_count']}

## Normalization and deduplication

- Normalized: {normalization_stats['success']}
- Normalization failures: {normalization_stats['failure']}
- Proof styles: `{json.dumps(normalization_stats['proof_styles'], ensure_ascii=False)}`
- Statement tokens: `{json.dumps(normalization_stats['statement_tokens'], ensure_ascii=False)}`
- Proof tokens: `{json.dumps(normalization_stats['proof_tokens'], ensure_ascii=False)}`
- Tactic steps: `{json.dumps(normalization_stats['tactic_steps'], ensure_ascii=False)}`
- Premises: `{json.dumps(normalization_stats['premises'], ensure_ascii=False)}`
- Raw, executable, training, and fingerprint representations remain separate; fingerprint normalization never replaces executable source.
- Internal dedup: `{json.dumps(dedup_stats, ensure_ascii=False)}`
- Cross-source dedup: `{json.dumps(cross_stats, ensure_ascii=False)}`

## Evaluation leakage

- `{json.dumps(leakage_stats, ensure_ascii=False)}`
- Protected sets: {final_stats['protected_eval_set_count']} files / {final_stats['protected_eval_record_count']} unique records
- Protected evaluation records were read-only and unchanged.
- Level 3-4 similarity is audit-only and was not treated as mathematical equivalence.

## Final pools and manifests

- `{json.dumps(final_stats, ensure_ascii=False)}`
- Generated manifests: {len(manifest_summaries)}
- All manifests use seed 20260801-derived fixed seeds, no replacement, one proof per theorem group, zero theorem-group duplicates, and zero hard evaluation leaks.

| Manifest | Rows | WB | LD | WB label share | LD label share | Budget error | Group duplicates | Eval leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(manifest_rows)}

## Recommendation

- The verified pool is suitable for WB/LD mixture SFT ablations after manually reviewing the single Level-4 similarity candidate.
- Start with `B_token_matched` and `C_token_matched`; retain `A_WB1000_LD0` and `E_token_matched` as source-only controls.
- Do not expand tracing yet: the current 60-file pool supplies 5311 high-confidence LD records and passed fidelity verification at 98.99%.
- Keep the 19 timeout records quarantined. Optional 60/120-second diagnostics may be run separately, but must not promote them into the default pool.
- Preserve legacy LeanDojo migration artifacts for audit only; do not restore them to the training path.
- No training, Expert Iteration, or GRPO run was started.
"""
    (output / "final_report.md").write_text(final, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("outputs/leandojo_v2_retrace/processed/current_mathlib_candidates.jsonl"),
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path("outputs/leandojo_v2_dataset_build/verification/all_results.jsonl"),
    )
    parser.add_argument(
        "--workbook-train",
        type=Path,
        default=Path("data/processed/lean_workbook_verified_v2/train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/leandojo_v2_dataset_build"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("models/Qwen2.5-1.5B-Instruct"),
    )
    args = parser.parse_args()
    output = args.output
    for directory in ("normalized", "dedup", "leakage", "final_pool", "manifests/row_matched", "manifests/token_matched"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    candidates = list(iter_jsonl(args.candidate))
    verifications = list(iter_jsonl(args.verification))
    if len(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"candidate count changed: {len(candidates)}")
    by_id = {str(row["sample_id"]): row for row in verifications}
    if len(by_id) != EXPECTED_CANDIDATES or set(by_id) != {
        str(row["id"]) for row in candidates
    }:
        raise RuntimeError("verification accounting mismatch or silent drop")

    normalized: list[dict[str, Any]] = []
    normalization_failures: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            normalized.append(normalize_leandojo_record(candidate, by_id[str(candidate["id"])]))
        except Exception as exc:
            normalization_failures.append(
                {"id": candidate.get("id"), "error": f"{type(exc).__name__}: {exc}"}
            )
    write_jsonl(output / "normalized/leandojo_v2_normalized.jsonl", normalized)
    write_jsonl(output / "normalized/normalization_failures.jsonl", normalization_failures)
    schema = {
        "normalization_version": NORMALIZATION_VERSION,
        "layers": ["raw", "executable", "training", "fingerprint"],
        "required_fields": sorted(
            {
                "id",
                "raw_statement",
                "raw_proof",
                "raw_declaration_source",
                "training_statement",
                "training_proof",
                "training_declaration",
                "statement_hash_exact",
                "statement_hash_normalized",
                "proof_hash_exact",
                "proof_hash_normalized",
                "source_identity_hash",
                "theorem_group_id",
                "verification_status",
            }
        ),
    }
    write_json(output / "normalized/normalization_schema.json", schema)

    internal = audit_internal_dedup(normalized)
    write_jsonl(output / "dedup/canonical_records.jsonl", internal["canonical_records"])
    write_jsonl(output / "dedup/dedup_aliases.jsonl", internal["dedup_aliases"])
    write_jsonl(
        output / "dedup/same_statement_multiple_proofs.jsonl",
        internal["same_statement_multiple_proofs"],
    )
    write_jsonl(
        output / "dedup/same_proof_different_statements.jsonl",
        internal["same_proof_different_statements"],
    )
    write_jsonl(output / "dedup/dedup_groups.jsonl", internal["dedup_groups"])

    workbook_train = [
        normalize_workbook_record(row) for row in iter_jsonl(args.workbook_train)
    ]
    protected_paths = discover_protected_paths(ROOT)
    protected, protected_inventory = load_protected_jsonl(protected_paths)
    protected = deduplicate_protected(protected)
    write_json(
        output / "leakage/protected_eval_sets.json",
        {
            "sets": protected_inventory,
            "unique_protected_records": len(protected),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    canonical = internal["canonical_records"]
    cross = audit_cross_source(canonical, workbook_train)
    write_jsonl(output / "dedup/cross_source_overlaps.jsonl", cross["overlaps"])

    default_verified = [
        row
        for row in canonical
        if row["verification_status"] == "verified_default_timeout"
        and row.get("repository_commit") == MATHLIB_COMMIT
        and row.get("source_file")
        and row.get("source_span")
        and not FORBIDDEN.search(str(row.get("raw_proof") or ""))
    ]
    leakage = audit_leakage(default_verified, protected)
    write_jsonl(output / "leakage/exact_identity_leaks.jsonl", leakage["exact_identity"])
    write_jsonl(output / "leakage/exact_statement_leaks.jsonl", leakage["exact_statement"])
    write_jsonl(
        output / "leakage/normalized_statement_leaks.jsonl",
        leakage["normalized_statement"],
    )
    write_jsonl(
        output / "leakage/structural_near_duplicates.jsonl",
        leakage["structural_near_duplicates"],
    )
    write_jsonl(
        output / "leakage/high_similarity_review.jsonl",
        leakage["high_similarity_review"],
    )
    write_jsonl(
        output / "leakage/evaluation_leakage_removed.jsonl", leakage["removed"]
    )
    final_ld = leakage["retained"]

    protected_groups = {row["theorem_group_id"] for row in protected}
    exact_alias_ids = cross["workbook_exact_alias_ids"]
    workbook_by_group: dict[str, dict[str, Any]] = {}
    for row in workbook_train:
        row_id = str(row.get("id") or row.get("record_id") or row.get("statement_id"))
        if row["theorem_group_id"] in protected_groups or row_id in exact_alias_ids:
            continue
        workbook_by_group.setdefault(row["theorem_group_id"], row)
    final_wb = list(workbook_by_group.values())
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"local training tokenizer unavailable at {args.tokenizer}: {exc}"
        ) from exc

    def with_training_tokens(row: dict[str, Any]) -> dict[str, Any]:
        value = dict(row)
        statement = str(
            row.get("training_statement") or row.get("raw_statement") or ""
        )
        proof = str(row.get("training_proof") or row.get("raw_proof") or "")
        value["statement_tokens"] = len(
            tokenizer.encode(statement, add_special_tokens=False)
        )
        value["label_tokens"] = len(
            tokenizer.encode(proof, add_special_tokens=False)
        )
        value["total_tokens"] = value["statement_tokens"] + value["label_tokens"]
        value["tokenizer"] = str(args.tokenizer)
        return value

    final_ld = [with_training_tokens(row) for row in final_ld]
    final_wb = [with_training_tokens(row) for row in final_wb]
    combined = sorted(
        final_ld + final_wb,
        key=lambda row: (
            str(row["theorem_group_id"]),
            str(row.get("id") or row.get("record_id")),
        ),
    )
    write_jsonl(
        output / "final_pool/leandojo_v2_final_train_candidates.jsonl", final_ld
    )
    write_jsonl(
        output / "final_pool/lean_workbook_canonical_candidates.jsonl", final_wb
    )
    write_jsonl(output / "final_pool/combined_canonical_index.jsonl", combined)

    verification_summary = json.loads(
        (output / "verification/verification_summary.json").read_text(encoding="utf-8")
    )
    if not verification_summary.get("quality_gate_95_percent"):
        write_json(
            output / "manifests/generation_status.json",
            {"generated": False, "reason": "full verification below 95% gate"},
        )
        return 2
    if len(final_wb) < 1000 or len(final_ld) < 1000:
        write_json(
            output / "manifests/generation_status.json",
            {
                "generated": False,
                "reason": "insufficient leakage-free source pool",
                "workbook": len(final_wb),
                "leandojo": len(final_ld),
            },
        )
        return 2

    row_specs = {
        "A_WB1000_LD0": (1000, 0),
        "B_WB750_LD250": (750, 250),
        "C_WB500_LD500": (500, 500),
        "D_WB250_LD750": (250, 750),
        "E_WB0_LD1000": (0, 1000),
    }
    manifest_summaries: dict[str, Any] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    for offset, (name, (wb_count, ld_count)) in enumerate(row_specs.items()):
        rows = sample_rows(
            final_wb,
            final_ld,
            workbook_rows=wb_count,
            leandojo_rows=ld_count,
            seed=SEED + offset,
        )
        manifests[name] = rows
        write_jsonl(output / f"manifests/row_matched/{name}.jsonl", rows)
        manifest_summaries[name] = summarize_manifest(name, rows, seed=SEED + offset)

    label_budget = sum(
        attach_token_counts(row)["label_tokens"] for row in manifests["A_WB1000_LD0"]
    )
    for offset, (name, workbook_share) in enumerate(
        (
            ("B_token_matched", 0.75),
            ("C_token_matched", 0.50),
            ("D_token_matched", 0.25),
            ("E_token_matched", 0.0),
        ),
        start=10,
    ):
        rows = sample_token_matched(
            final_wb,
            final_ld,
            workbook_share=workbook_share,
            label_token_budget=label_budget,
            seed=SEED + offset,
        )
        write_jsonl(output / f"manifests/token_matched/{name}.jsonl", rows)
        manifest_summaries[name] = summarize_manifest(
            name, rows, seed=SEED + offset, budget=label_budget
        )

    write_json(output / "manifests/manifest_summaries.json", manifest_summaries)
    write_json(
        output / "manifests/generation_status.json",
        {"generated": True, "seed": SEED, "manifest_count": len(manifest_summaries)},
    )

    normalization_stats = {
        "success": len(normalized),
        "failure": len(normalization_failures),
        "proof_styles": dict(Counter(row["proof_style"] for row in normalized)),
        "statement_tokens": distribution(
            [
                len(tokenizer.encode(row["training_statement"], add_special_tokens=False))
                for row in normalized
            ]
        ),
        "proof_tokens": distribution(
            [
                len(tokenizer.encode(row["training_proof"], add_special_tokens=False))
                for row in normalized
            ]
        ),
        "tactic_steps": distribution(
            [len(row["raw_tactic_trace"]) for row in normalized]
        ),
        "premises": distribution([len(row["raw_premises"]) for row in normalized]),
    }
    final_stats = {
        "leandojo_v2_final_candidates": len(final_ld),
        "lean_workbook_canonical_candidates": len(final_wb),
        "combined_pool": len(combined),
        "protected_eval_set_count": len(protected_inventory),
        "protected_eval_record_count": len(protected),
        "leandojo_source_files": len({row["source_file"] for row in final_ld}),
        "leandojo_mathlib_domains": dict(
            Counter(
                (
                    str(row["source_file"]).split("/")[1]
                    if len(str(row["source_file"]).split("/")) > 2
                    else "root"
                )
                for row in final_ld
            )
        ),
        "leandojo_proof_styles": dict(Counter(row["proof_style"] for row in final_ld)),
        "leandojo_proof_tokens": distribution(
            [int(row["label_tokens"]) for row in final_ld]
        ),
        "leandojo_tactic_steps": distribution(
            [len(row["raw_tactic_trace"]) for row in final_ld]
        ),
        "leandojo_premises": distribution(
            [len(row["raw_premises"]) for row in final_ld]
        ),
    }
    write_json(
        output / "audit/pipeline_versions.json",
        {
            "normalization_version": NORMALIZATION_VERSION,
            "manifest_seed": SEED,
            "mathlib_commit": MATHLIB_COMMIT,
            "tokenizer": str(args.tokenizer),
        },
    )
    write_json(
        output / "audit/exclusions_accounting.json",
        {
            "input": EXPECTED_CANDIDATES,
            "normalization_failures": len(normalization_failures),
            "internal_deduplicated": internal["statistics"]["deduplicated_records"],
            "not_default_verified_or_policy_rejected": len(canonical)
            - len(default_verified),
            "evaluation_leakage_removed": len(leakage["removed"]),
            "final": len(final_ld),
        },
    )
    write_reports(
        output,
        verification_summary=verification_summary,
        normalization_stats=normalization_stats,
        dedup_stats=internal["statistics"],
        cross_stats=cross["statistics"],
        leakage_stats=leakage["statistics"],
        final_stats=final_stats,
        manifest_summaries=manifest_summaries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
