"""Audit EI Round 0 training-yield constraints without starting a Trainer."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = read_json(root / "discovery/discovery_summary.json")
    outcomes = read_jsonl(root / "discovery/statement_outcomes.jsonl")
    successes = read_jsonl(root / "success_bank/success_bank.jsonl")
    failures = read_jsonl(root / "failure_bank/failure_bank.jsonl")
    manifest = read_jsonl(root / "discovery/discovery_manifest.jsonl")
    if (
        summary.get("verified_candidates") != 4000
        or len(outcomes) != 500
        or len(successes) != 722
        or len(failures) != 3278
    ):
        raise RuntimeError("Discovery artifacts are incomplete or drifted")

    success_statement_ids = {
        str(row["statement_id"]) for row in successes if row.get("pantograph_verified")
    }
    by_statement_failure_layers: dict[str, Counter[str]] = defaultdict(Counter)
    for row in failures:
        by_statement_failure_layers[str(row["statement_id"])][
            str(row.get("failure_layer") or "unknown")
        ] += 1
    unsolved_ids = {
        str(row["statement_id"])
        for row in outcomes
        if row.get("discovery_bucket") == "unsolved"
    }
    partial_ids = {
        str(row["statement_id"])
        for row in outcomes
        if row.get("discovery_bucket") == "frontier"
    }
    near_miss_unsolved_ids = {
        statement_id
        for statement_id in unsolved_ids
        if by_statement_failure_layers[statement_id]["near_miss"] > 0
    }
    repair_only_unsolved_ids = {
        statement_id
        for statement_id in unsolved_ids
        if by_statement_failure_layers[statement_id]["near_miss"] == 0
        and by_statement_failure_layers[statement_id]["repair_candidate"] > 0
    }

    required_total = 500
    required_success = 400
    required_frontier = 100
    strict_generated_only_total = (len(success_statement_ids) // 5) * 5
    strict_generated_success = strict_generated_only_total * 4 // 5
    strict_generated_frontier = strict_generated_only_total - strict_generated_success
    max_80_20_with_reference_frontier = min(
        1000,
        (len(success_statement_ids) // 4) * 5,
        ((len(success_statement_ids) + len(near_miss_unsolved_ids)) // 5) * 5,
    )
    max_80_20_success = int(max_80_20_with_reference_frontier * 0.8)
    max_80_20_frontier = max_80_20_with_reference_frontier - max_80_20_success

    payload = {
        "status": "BLOCKED_EMPIRICAL_YIELD_CONFLICT",
        "trainer_started": False,
        "grpo_started": False,
        "discovery": {
            "statements": 500,
            "candidates": 4000,
            "candidate_successes": len(successes),
            "solved_statements": len(success_statement_ids),
            "partial_success_statements": len(partial_ids),
            "zero_success_statements": len(unsolved_ids),
            "zero_success_with_near_miss": len(near_miss_unsolved_ids),
            "zero_success_repair_only": len(repair_only_unsolved_ids),
        },
        "hard_gates": {
            "requested_rows_min": 500,
            "requested_rows_max": 1000,
            "success_share": 0.8,
            "frontier_share": 0.2,
            "theorem_group_duplicates": 0,
            "max_repeat": 1,
            "unverified_targets": 0,
        },
        "infeasibility_proof": {
            "unique_success_statements_available": len(success_statement_ids),
            "unique_success_statements_required_for_500_at_80pct": required_success,
            "success_statement_shortfall": required_success - len(success_statement_ids),
            "success_bank_candidate_rows": len(successes),
            "candidate_rows_discarded_by_theorem_group_uniqueness": (
                len(successes) - len(success_statement_ids)
            ),
            "strict_500_80_20_possible": len(success_statement_ids) >= required_success,
        },
        "safe_options": {
            "A_preserve_generated_only_and_80_20": {
                "rows": strict_generated_only_total,
                "success_bank": strict_generated_success,
                "frontier_partial_success": strict_generated_frontier,
                "violated_requirement": "row minimum 500",
                "advantages": [
                    "all targets are Pantograph-verified generated proofs",
                    "no theorem-group duplicates",
                    "preserves intended EI causal test",
                ],
            },
            "B_preserve_80_20_and_add_reference_frontier": {
                "rows": max_80_20_with_reference_frontier,
                "success_bank": max_80_20_success,
                "frontier_verified_reference": max_80_20_frontier,
                "violated_requirement": "row minimum 500",
                "advantages": [
                    "no theorem-group duplicates",
                    "all targets can be Pantograph-attested",
                    "uses only discovery-derived near-miss statements for replay",
                ],
            },
            "C_preserve_500_rows_and_uniqueness": {
                "rows": required_total,
                "success_bank": len(success_statement_ids),
                "frontier_verified_reference": required_total
                - len(success_statement_ids),
                "realized_success_share": len(success_statement_ids) / required_total,
                "realized_frontier_share": (
                    required_total - len(success_statement_ids)
                )
                / required_total,
                "violated_requirement": "80/20 source mix",
                "additional_risk": (
                    "not every zero-success statement is a near miss; repair-only rows "
                    "would change the experiment into reference-proof replay"
                ),
            },
            "D_preserve_500_rows_and_80_20": {
                "rows": required_total,
                "success_bank": required_success,
                "frontier": required_frontier,
                "violated_requirement": (
                    "requires 147 additional solved theorem groups, theorem duplicates, "
                    "or a new Discovery sample"
                ),
                "authorized": False,
            },
        },
        "recommended_option": "A_preserve_generated_only_and_80_20",
        "recommended_reason": (
            "It is the only option that cleanly tests whether verified generated proofs "
            "improve H0 without adding source-reference targets; its 250 rows are below "
            "the requested scale but remain consistent with the conservative EI objective."
        ),
        "manifest_reference_attestations": {
            "rows": len(manifest),
            "assembled_source_hash_present": sum(
                bool(row.get("reference_assembled_source_hash")) for row in manifest
            ),
            "attestation_id_present": sum(
                bool(row.get("reference_attestation_id")) for row in manifest
            ),
        },
    }
    write_json(root / "training_manifest/feasibility_report.json", payload)
    report = f"""# EI Round 0 training feasibility gate

Status: **BLOCKED_EMPIRICAL_YIELD_CONFLICT**

Discovery produced 722 successful candidates but only **253 unique solved theorem
groups**.  The training gate requires 500--1000 rows, an 80/20 Success/Frontier
mix, theorem-group duplicates=0, and max repeat=1.  A 500-row 80/20 manifest
therefore requires 400 unique solved theorem groups, leaving a shortfall of
**{required_success - len(success_statement_ids)}**.

No Trainer was started.  No unverified proof was admitted.  No theorem was
duplicated to manufacture the requested row count.

## Safe choices

1. **Recommended: 250 rows (200 Success + 50 partial-success Frontier).**
   Preserves generated-only targets, 80/20, and theorem uniqueness; relaxes the
   500-row minimum.
2. **{max_80_20_with_reference_frontier} rows ({max_80_20_success} generated Success +
   {max_80_20_frontier} verified-reference near-miss Frontier).** Preserves 80/20 and
   uniqueness but still relaxes the 500-row minimum and weakens the causal test.
3. **500 rows (253 generated Success + 247 verified-reference rows).** Preserves
   scale and uniqueness but changes the mix to 50.6/49.4 and includes repair-only
   statements.
4. **500 rows at 80/20.** Not possible without a new Discovery sample, theorem
   duplicates, or another unapproved training source.
"""
    (root / "training_manifest/TRAINING_GATE_BLOCKED.md").write_text(
        report, encoding="utf-8"
    )
    status_path = root / "status.json"
    status = read_json(status_path)
    status.update(
        {
            "status": "TRAINING_GATE_BLOCKED_EMPIRICAL_YIELD_CONFLICT",
            "trainer_started": False,
            "grpo_started": False,
            "training_feasibility_report": str(
                root / "training_manifest/feasibility_report.json"
            ),
        }
    )
    write_json(status_path, status)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
