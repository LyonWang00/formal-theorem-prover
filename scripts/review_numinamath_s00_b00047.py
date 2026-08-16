#!/usr/bin/env python3
"""Frozen strict review for the six-record NuminaMath shard s00/b00047."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "numinamath_b46_review_base", ROOT / "scripts/review_numinamath_s00_b00046.py"
)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
base = previous.base
scale = previous.scale

base.BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00047"
base.BATCH_ID = base.BATCH.name
base.MANIFEST = base.BATCH / "candidate_manifest.jsonl"
base.RECEIPTS = base.BATCH / "verification_results.jsonl"
base.OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00047_resolved.jsonl"
base.REPORT = base.OUTPUT.with_name("review_s00_b00047_resolved_report.json")
base.VERSION = "numinamath_s00_b00047_strict_frozen_v1"

# All six items fail the calibrated semantic gate: the existential conjunction
# split leaks binders, the reverse iff and conjunction commutation are logical
# repackaging, the cosine equality is a scalar mechanical bound, and the sole
# clean-looking order refinement is an exact target already retained in
# reviewed history (cebc5a10... / the same Shapiro-sum inequality).
previous.SELECTED_IFF_FORWARD = set()
previous.SELECTED_AND = set()
previous.SELECTED_LT_TO_LE = set()
previous.SELECTED_ORDER_REFINEMENTS = set()


def main() -> None:
    """Six-row equivalent of the inherited 500-row append-safe writer."""
    if base.OUTPUT.exists() or base.REPORT.exists():
        raise FileExistsError("resolved review output already exists")
    ordered = [str(row["record_id"]) for _, row in scale.rows(base.MANIFEST)]
    wanted = set(ordered)
    if len(ordered) != 6 or len(wanted) != 6:
        raise ValueError("manifest must contain 6 unique record_ids")
    packets = {
        row["record_id"]: row
        for _, row in scale.rows(scale.PACKETS)
        if row.get("record_id") in wanted
    }
    raw_all = [row for _, row in scale.rows(scale.RAW)]
    raw = {row["record_id"]: row for row in raw_all if row.get("record_id") in wanted}
    parent_ids = {str(row["parent_id"]) for row in packets.values()}
    parents = {
        row["record_id"]: row
        for _, row in scale.rows(scale.PARENTS)
        if row.get("record_id") in parent_ids
    }
    if set(packets) != wanted or set(raw) != wanted or set(parents) != parent_ids:
        raise ValueError("packet/raw/parent join incomplete")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_all:
        if row.get("upstream_source"):
            groups[str(row["upstream_source"])].append(
                {"record_id": str(row["record_id"]), "method": str(row.get("question_type", ""))}
            )
    for rid in ordered:
        expected = "parent:" + str(packets[rid]["parent_id"])
        if str(raw[rid].get("upstream_source", "")) != expected:
            raise AssertionError(f"raw upstream mismatch {rid}")

    receipt_rows = [row for _, row in scale.rows(base.RECEIPTS)]
    receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipt_rows:
        if row.get("record_id") in wanted:
            receipts[str(row["record_id"])].append(row)
    if len(receipt_rows) != 6 or set(receipts) != wanted:
        raise ValueError(f"Pantograph batch incomplete: rows={len(receipt_rows)}, ids={len(receipts)}")

    equality_choice: dict[str, tuple[str, str | None]] = {}
    for packet in packets.values():
        if str(packet["method"]).startswith("eq_to_le_"):
            equality_choice.setdefault(
                str(packet["parent_id"]),
                scale.equality_class(packet, parents[str(packet["parent_id"])]),
            )
    output = []
    decisions, tiers, methods, receipt_stats = Counter(), Counter(), Counter(), Counter()
    eligible: dict[str, list[str]] = defaultdict(list)
    passes: dict[str, list[str]] = defaultdict(list)
    for number, rid in enumerate(ordered, 1):
        packet, raw_row = packets[rid], raw[rid]
        proof_hash = str(packet["candidate"]["proof_sha256"])
        statement_hash = str(packet["diff"]["statement"]["after_sha256"])
        success = scale.compatible_success(receipts[rid], proof_hash)
        receipt_stats["compatible_success" if success else "fail"] += 1
        decision, tier, basis = previous.classify(
            packet, raw_row, parents[str(packet["parent_id"])], success, equality_choice
        )
        hypothetical, _, _ = previous.classify(
            packet, raw_row, parents[str(packet["parent_id"])], True, equality_choice
        )
        upstream = str(raw_row["upstream_source"])
        if hypothetical == "pass":
            eligible[upstream].append(rid)
        if decision == "pass":
            if not success:
                raise AssertionError(f"pass without success {rid}")
            passes[upstream].append(rid)
        siblings = [item for item in groups[upstream] if item["record_id"] != rid]
        sibling_text = ", ".join(f"{item['method']}:{item['record_id']}" for item in siblings) or "none"
        receipt_text = (
            "compatible Pantograph success"
            if success
            else "Pantograph fail: " + base.b5.error_excerpt(receipts[rid])
        )
        method = str(packet["method"])
        reason = (
            f"Row {number}; frozen raw upstream_source={upstream}. Compared parent goal "
            f"`{packet['parent']['goal']}` with candidate goal `{packet['candidate']['goal']}`, "
            f"problem text, proof, reviewed-history exact/near duplicates, and complete global "
            f"siblings [{sibling_text}]. Decision basis: {basis}. Exact statement/proof receipt: "
            f"{receipt_text}."
        )
        output.append(
            {
                "record_id": rid,
                "quality_decision": decision,
                "quality_tier": tier,
                "manual_reviewed": True,
                "reviewer": "codex-quality-adjudicator",
                "review_reason": reason,
                "reviewed_statement_sha256": statement_hash,
                "reviewed_proof_sha256": proof_hash,
                "review_version": base.VERSION,
                "parent_id": upstream,
                "upstream_source": upstream,
                "method": method,
                "siblings": siblings,
                "pantograph_compatible_success": success,
            }
        )
        decisions[decision] += 1
        tiers[tier] += 1
        methods[(method, decision)] += 1
    if any(len(items) > 1 for items in eligible.values()) or any(len(items) > 1 for items in passes.values()):
        raise AssertionError("multiple retained siblings")
    if len(output) != 6 or len({row["record_id"] for row in output}) != 6:
        raise AssertionError("coverage failure")
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in output
    )
    report = {
        "schema_version": "numinamath_shard_review_report_v1",
        "review_version": base.VERSION,
        "batch_id": base.BATCH_ID,
        "rows": 6,
        "unique_record_ids": 6,
        "candidate_manifest_sha256": scale.sha_file(base.MANIFEST),
        "receipts_sha256": scale.sha_file(base.RECEIPTS),
        "packets_sha256": scale.sha_file(scale.PACKETS),
        "raw_sha256": scale.sha_file(scale.RAW),
        "receipt_status": dict(sorted(receipt_stats.items())),
        "final_decisions": dict(sorted(decisions.items())),
        "final_tiers": dict(sorted(tiers.items())),
        "by_method_decision": {f"{method}|{decision}": count for (method, decision), count in sorted(methods.items())},
        "raw_upstream_parent_groups": len({raw[rid]["upstream_source"] for rid in ordered}),
        "max_eligible_per_upstream": max(map(len, eligible.values()), default=0),
        "max_pass_per_upstream": max(map(len, passes.values()), default=0),
        "global_siblings_embedded": True,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    base.b5.safe_write(base.OUTPUT, payload)
    base.b5.safe_write(
        base.REPORT, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    base.classify = previous.classify
    main()
