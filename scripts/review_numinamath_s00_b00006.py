#!/usr/bin/env python3
"""Strict raw-upstream review for NuminaMath shard s00/b00006."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00006"
MANIFEST = BATCH / "candidate_manifest.jsonl"
RECEIPTS = BATCH / "verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00006.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00006_report.json")
VERSION = "numinamath_s00_b00006_strict_upstream_review_v1"

base_spec = importlib.util.spec_from_file_location(
    "numinamath_b5_review_helpers", ROOT / "scripts/review_numinamath_s00_b00005.py"
)
base = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base
base_spec.loader.exec_module(base)
scale = base.scale

SELECTED_AND = {
    "2791fea5-9da7-57ee-9141-e8ac3f5fbd3a::and_right_81e0e4bf48dc",
    "c76ee110-8b17-589d-a8a1-4546d6bcb67c::and_left_7c5141069b5b",
    "3470a354-4be3-585e-9f4e-aba97d3fd2e1::and_right_8f77e211acba",
    "bfbcdf76-b055-5368-93ba-18c2905876f7::and_right_04a7e39991c5",
    "9654a5fd-fe04-5f16-b768-c412f8e118cb::and_right_6ad90d6ac0ee",
    "4efdbe9e-63b1-50c7-a8b3-95a0eaddd97d::and_left_d5aac30f0180",
    "d3965f72-7932-5ff3-9a21-a807f9a9b4bf::and_left_c07a7de1ee84",
    "10cb3895-aa76-5309-8f95-85a84f9ee042::and_left_24b1d1ea52b8",
    "90546044-3ddd-5040-a39b-0ffd36e7a41f::and_left_77d95d57245b",
    "57c00235-b195-52f1-9c91-5e3c28836f01::and_right_c449f2e6a80c",
    "48e9dacf-7ea3-512b-a509-f9807d9349c3::and_right_c2a259cb4f7b",
    "291f0075-26b4-5f7f-8dfe-bf8918c527e3::and_right_cd9de20df8f2",
    "cca09e81-521a-5214-970f-86f9611486fd::and_right_19a2a050cacf",
    "7834fbde-9845-559f-b686-99b940e89141::and_left_88c027cd0be3",
    "6faa7843-0f54-50a5-b1fc-c0a77133d906::and_left_8ddc3302218b",
}

TRIVIAL_EQUALITY_BOUNDS = {
    # Both are closed numerical trigonometric identities with no variable
    # content; replacing equality by <= is a tautological weakening.
    "a4a28165-7e37-5491-bf77-0885fb2790a6::eq_to_le_forward_5ef96de6ef90",
    "f6b3eb52-dbce-507f-99fe-f514fced247d::eq_to_le_forward_14fc0f76ac5f",
}

SELECTED_CONTRAPOSITIVE = {
    "683730c6-bc68-5213-8baa-c303c4b47bb4::implication_contrapositive_c150c2b8dfab",
}

HARD_REJECT = set(scale.HARD_REJECT) | {"lt_to_ne"}


def classify(
    packet: Mapping[str, Any], raw: Mapping[str, Any], parent: Mapping[str, Any],
    success: bool, equality_choice: Mapping[str, tuple[str, str | None]],
) -> tuple[str, str, str]:
    rid = str(packet["record_id"])
    method = str(packet["method"])
    pid = str(packet["parent_id"])
    parent_goal = str(packet["parent"]["goal"])
    contaminated = scale.contaminated(packet, raw)
    problem_ok = scale.problem_ok(method, raw)
    reasons = bool(packet.get("reason_codes"))

    if method in HARD_REJECT:
        return "reject", "extremely_low", "pure formal reordering, symmetry, negation weakening, or proof-only variation"

    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject", "extremely_low", "closed trigonometric numeral identity weakened to a tautological one-sided bound"
        if chosen is None:
            labels = {
                "bare_value_to_numeric_bound": "bare variable/function value weakened from an exact numeral to a scalar bound",
                "arbitrary_expression_order": "arbitrary ordering of two expression-valued equality sides",
                "invalid_scope_or_prop_cut": "equality cut inside a quantifier, Prop, set description, or declaration",
                "closed_numeric_relaxation": "closed numerical equality mechanically weakened",
                "not_top_level_equality": "parent is not a safe top-level equality",
            }
            return "reject", "extremely_low" if category == "invalid_scope_or_prop_cut" else "low", labels.get(category, category)
        if method != chosen:
            return "reject", "low", f"paired equality direction; sole natural direction is {chosen}"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "candidate has scope, declaration, or problem-text pollution"
        basis = "substantive set containment" if category == "substantive_set_containment" else "natural nontrivial numerical bound"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            tier = "extremely_low" if contaminated or reasons else "low"
            return "reject", tier, "not the sole retained global sibling: scalar answer, weaker paired bound, lost binder, negation-scope cut, or pollution"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "selected-looking component still has unresolved scope/declaration pollution"
        basis = "sole retained well-scoped substantive conjunction component after global sibling comparison"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "iff_forward":
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "generator split an iff under exists/forall or a polluted declaration"
        basis = "sole retained nontrivial solution-classification, parameter-range, or completeness direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "iff_reverse":
        return "reject", "extremely_low" if contaminated or reasons else "low", (
            "reverse/substitution sibling is not retained; the global forward classification direction owns this upstream"
        )

    if method == "implication_contrapositive":
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low", "forall/exists/conjunction-internal arrow was mistaken for a safe top-level implication"
        if contaminated or reasons or not problem_ok or scale.top_arrow(parent_goal) is None:
            return "reject", "extremely_low", "contrapositive has unresolved scope or declaration pollution"
        basis = "complete well-scoped top-level contrapositive with substantive digit/divisibility content"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "le_to_lt_or_eq":
        if contaminated or reasons or not problem_ok or not base.top_order(parent_goal, ("≤", "≥")):
            return "reject", "extremely_low", "not a clean complete top-level non-strict order relation"
        basis = "substantive top-level order refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "lt_to_le":
        if contaminated or reasons or not problem_ok or not base.top_order(parent_goal, ("<", ">")):
            return "reject", "extremely_low", "strict relation was cut inside a quantifier, Prop, or declaration"
        basis = "natural reusable non-strict consequence of a substantive strict relation"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    return "reject", "extremely_low", f"method {method} is outside the calibrated acceptance scale"


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("review output already exists")
    ordered = [str(row["record_id"]) for _, row in scale.rows(MANIFEST)]
    wanted = set(ordered)
    if len(ordered) != 500 or len(wanted) != 500:
        raise ValueError("manifest must contain 500 unique record_ids")

    packets = {row["record_id"]: row for _, row in scale.rows(scale.PACKETS) if row.get("record_id") in wanted}
    raw_all = [row for _, row in scale.rows(scale.RAW)]
    raw = {row["record_id"]: row for row in raw_all if row.get("record_id") in wanted}
    parent_ids = {str(row["parent_id"]) for row in packets.values()}
    parents = {row["record_id"]: row for _, row in scale.rows(scale.PARENTS) if row.get("record_id") in parent_ids}
    if set(packets) != wanted or set(raw) != wanted or set(parents) != parent_ids:
        raise ValueError("packet/raw/parent join is incomplete")

    sibling_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_all:
        upstream = row.get("upstream_source")
        if upstream:
            sibling_groups[str(upstream)].append({"record_id": str(row["record_id"]), "method": str(row.get("question_type", ""))})
    for rid in ordered:
        upstream = str(raw[rid].get("upstream_source", ""))
        expected = "parent:" + str(packets[rid]["parent_id"])
        if upstream != expected:
            raise AssertionError(f"raw upstream mismatch for {rid}: {upstream!r} != {expected!r}")

    receipt_rows = [row for _, row in scale.rows(RECEIPTS)]
    receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipt_rows:
        if row.get("record_id") in wanted:
            receipts[str(row["record_id"])].append(row)
    if len(receipt_rows) != 500 or set(receipts) != wanted:
        raise ValueError(f"Pantograph batch incomplete: rows={len(receipt_rows)}, ids={len(receipts)}")

    equality_choice: dict[str, tuple[str, str | None]] = {}
    for packet in packets.values():
        if str(packet["method"]).startswith("eq_to_le_"):
            pid = str(packet["parent_id"])
            equality_choice.setdefault(pid, scale.equality_class(packet, parents[pid]))

    output_rows: list[dict[str, Any]] = []
    decisions, tiers, methods, receipt_stats = Counter(), Counter(), Counter(), Counter()
    eligible: dict[str, list[str]] = defaultdict(list)
    passes: dict[str, list[str]] = defaultdict(list)
    for number, rid in enumerate(ordered, 1):
        packet, raw_row = packets[rid], raw[rid]
        proof_hash = str(packet["candidate"]["proof_sha256"])
        statement_hash = str(packet["diff"]["statement"]["after_sha256"])
        success = scale.compatible_success(receipts[rid], proof_hash)
        receipt_stats["compatible_success" if success else "fail"] += 1
        decision, tier, basis = classify(packet, raw_row, parents[str(packet["parent_id"])], success, equality_choice)
        hypothetical, _, _ = classify(packet, raw_row, parents[str(packet["parent_id"])], True, equality_choice)
        upstream = str(raw_row["upstream_source"])
        if hypothetical == "pass":
            eligible[upstream].append(rid)
        if decision == "pass":
            if not success:
                raise AssertionError(f"pass without compatible success: {rid}")
            passes[upstream].append(rid)
        siblings = [item for item in sibling_groups[upstream] if item["record_id"] != rid]
        sibling_text = ", ".join(f"{x['method']}:{x['record_id']}" for x in siblings) or "none"
        receipt_text = "compatible Pantograph success" if success else "Pantograph fail: " + base.error_excerpt(receipts[rid])
        reason = (
            f"Row {number}; frozen raw upstream_source={upstream}. Compared parent goal `{packet['parent']['goal']}` "
            f"with candidate goal `{packet['candidate']['goal']}`, problem text, proof, and complete global siblings "
            f"[{sibling_text}]. Decision basis: {basis}. Exact statement/proof receipt: {receipt_text}."
        )
        method = str(packet["method"])
        output_rows.append({
            "record_id": rid, "quality_decision": decision, "quality_tier": tier,
            "manual_reviewed": True, "reviewer": "codex-quality-adjudicator", "review_reason": reason,
            "reviewed_statement_sha256": statement_hash, "reviewed_proof_sha256": proof_hash,
            "review_version": VERSION, "parent_id": upstream, "upstream_source": upstream,
            "method": method, "siblings": siblings, "pantograph_compatible_success": success,
        })
        decisions[decision] += 1; tiers[tier] += 1; methods[(method, decision)] += 1

    eligible_bad = {key: value for key, value in eligible.items() if len(value) > 1}
    pass_bad = {key: value for key, value in passes.items() if len(value) > 1}
    if eligible_bad or pass_bad:
        raise AssertionError(f"multiple retained siblings: eligible={eligible_bad}, pass={pass_bad}")
    if len(output_rows) != 500 or len({row["record_id"] for row in output_rows}) != 500:
        raise AssertionError("coverage failure")

    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in output_rows)
    report = {
        "schema_version": "numinamath_shard_review_report_v1", "review_version": VERSION,
        "batch_id": "numinamath_expand_s00_b00006", "rows": 500, "unique_record_ids": 500,
        "candidate_manifest_sha256": scale.sha_file(MANIFEST), "receipts_sha256": scale.sha_file(RECEIPTS),
        "packets_sha256": scale.sha_file(scale.PACKETS), "raw_sha256": scale.sha_file(scale.RAW),
        "receipt_status": dict(sorted(receipt_stats.items())), "final_decisions": dict(sorted(decisions.items())),
        "final_tiers": dict(sorted(tiers.items())),
        "by_method_decision": {f"{m}|{d}": n for (m, d), n in sorted(methods.items())},
        "raw_upstream_parent_groups": len({raw[rid]["upstream_source"] for rid in ordered}),
        "max_eligible_per_upstream": max(map(len, eligible.values()), default=0),
        "max_pass_per_upstream": max(map(len, passes.values()), default=0),
        "global_siblings_embedded": True, "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    base.safe_write(OUTPUT, payload)
    base.safe_write(REPORT, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
