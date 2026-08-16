#!/usr/bin/env python3
"""Write the final human-readable report for campaign 0164."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--commit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    commit = json.loads(args.commit_report.read_text(encoding="utf-8"))
    batch_stats = []
    per_record_results: dict[str, list[dict]] = {}
    for value in audit["selected_batch_dirs"]:
        batch = Path(value)
        candidates = rows(batch / "candidate_manifest.jsonl")
        results = rows(batch / "verification_results.jsonl")
        attempted = {str(row["record_id"]) for row in candidates}
        succeeded = {
            str(row["record_id"])
            for row in results
            if row.get("success") is True or row.get("pantograph_verified") == "success"
        }
        batch_stats.append((batch.name, len(attempted), len(succeeded)))
        for result in results:
            per_record_results.setdefault(str(result["record_id"]), []).append(result)

    failure_taxonomy: Counter[str] = Counter()
    for attempts in per_record_results.values():
        if any(row.get("success") is True or row.get("pantograph_verified") == "success" for row in attempts):
            continue
        text = "\n".join(str(row.get("diagnostics") or "") for row in attempts).lower()
        if "timeout" in text or "maximum number of heartbeats" in text:
            category = "timeout_or_heartbeat"
        elif "unknown identifier" in text or "unknown constant" in text:
            category = "unknown_identifier_or_missing_context"
        elif "no goals to be solved" in text:
            category = "no_goals"
        elif "rewrite failed" in text or "did not find an occurrence" in text:
            category = "rewrite_mismatch"
        elif "unexpected token" in text or "parser" in text:
            category = "syntax_error"
        elif "unsolved goals" in text:
            category = "unsolved_goals"
        else:
            category = "other_lean_compilation"
        failure_taxonomy[category] += 1

    lines = [
        "# NuminaMath local repair campaign 0164 report",
        "",
        "## Outcome",
        "",
        f"- Contract window: `{contract['stable_start']}` → `{contract['deadline']}` ({contract['duration_hours']} hours).",
        f"- Canonical success: **{audit['canonical_success_before']:,} → {commit['success_after']:,}**.",
        f"- Canonical fail: **{audit['canonical_fail_before']:,} → {commit['fail_after']:,}**.",
        f"- Newly promoted, locally Pantograph-verified records: **{commit['promoted_to_success']:,}**.",
        f"- Unique attempted records: **{audit['unique_attempted_record_ids']:,}** across {audit['selected_batch_count']} fully verified batches.",
        "- Writeback: one atomic canonical update after the final audit.",
        "",
        "## Safety and quality gates",
        "",
        f"- Frozen cloud 10,000 overlap: attempted `{audit['cloud_overlap_attempted']}`, promoted `{audit['cloud_overlap_promoted']}`.",
        f"- `need_decompose` overlap: attempted `{audit['need_decompose_overlap_attempted']}`, promoted `{audit['need_decompose_overlap_promoted']}`.",
        f"- Forbidden `sorry` / `admit` / `axiom` in promoted proofs: `{audit['forbidden_success_proofs']}`.",
        f"- Pantograph success carrying a `sorry` warning: `{audit['sorry_warning_successes']}`.",
        f"- Cloud frozen SHA256: `{audit['cloud_frozen_sha256']}`.",
        f"- `need_decompose` SHA256: `{audit['need_decompose_sha256']}`.",
        "- One Pantograph worker was used; no Trainer or GPU training was started.",
        "",
        "## Method",
        "",
        "The highest-yield lane reused a shortest locally verified proof only when statements were structurally equivalent (alpha-renaming, layout, safe notation, or reversed atomic inequality notation). Every transferred proof was recompiled locally. Namespace and local-shadow failures were reviewed and repaired in small follow-up batches; semantic mismatches, deterministic timeouts, and context-dependent helper proofs were not promoted.",
        "",
        "## Batch results",
        "",
        "| Batch | Attempted records | Verified success |",
        "|---|---:|---:|",
    ]
    for name, attempted, succeeded in batch_stats:
        lines.append(f"| `{name}` | {attempted} | {succeeded} |")
    lines.extend([
        "",
        "## Remaining attempted failures",
        "",
    ])
    for category, count in sorted(failure_taxonomy.items()):
        lines.append(f"- `{category}`: {count}")
    lines.extend([
        "",
        "## Frozen exclusions",
        "",
        "The cloud repair file and the existing `need_decompose` file remained byte-identical throughout the campaign. The unverified preparation directory `batch_0275_verified_notation_duplicates_v2` was deliberately excluded from writeback; its corrected replacement was `batch_0275b_verified_notation_duplicates_v2`.",
        "",
    ])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
