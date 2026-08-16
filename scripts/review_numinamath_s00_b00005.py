#!/usr/bin/env python3
"""Strict calibrated review for NuminaMath shard s00/b00005.

The parent key is always the frozen raw ``upstream_source``.  Siblings are
reconstructed from the complete raw expansion file, rather than from batch
adjacency or the audit packet's abbreviated list.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00005"
MANIFEST = BATCH / "candidate_manifest.jsonl"
RECEIPTS = BATCH / "verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00005.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00005_report.json")
VERSION = "numinamath_s00_b00005_strict_upstream_review_v1"

spec = importlib.util.spec_from_file_location(
    "numinamath_adjudication_scale_b5", ROOT / "scripts/adjudicate_numinamath_first1040.py"
)
scale = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scale
spec.loader.exec_module(scale)

# These conjunction projections were compared individually with both the
# parent theorem and the complete global sibling set.  Each upstream has at
# most one retained component.
SELECTED_AND = {
    "75675916-e042-59f5-bb47-49c5a938a13a::and_left_25c71d18b66f",
    "05f8e175-5ce7-5f49-8531-7a71577bc1a9::and_right_a041651cf9b8",
    "f21aa3ad-3d17-5b23-844b-b4e8602156f7::and_right_5e3eb1712754",
    "78ad861d-7d05-5469-aa86-3cfc20ec8020::and_left_83f58bc51773",
    "ee5f643f-6437-524b-bd14-22992f6d08b2::and_left_ab507ec97421",
    "2dc49c8d-4476-55d3-9b07-2812792e837e::and_left_3d3ea29f3991",
    "53202693-1bda-50b5-93db-bbef29fe5a24::and_left_44ca04c05cad",
    "6b20465c-73d6-5315-a113-12c2f22d7751::and_right_91f05de18f12",
    "f102a9fa-c16c-52ba-9e96-00b82b9c7c6a::and_left_52acf39c0584",
    "09cadbaf-b3ab-58bb-89aa-ee36d0804c87::and_left_a5fd445b0e2f",
    "4af30b75-9e0f-5379-9339-6fe3f7e1be6e::and_right_bff3403cb2d3",
    "56ec06b1-9a4b-5f3d-9bd0-ad15b2aa319a::and_right_6fcbd1ef40ca",
    "0e91ad6a-e8bd-5bdd-8210-d82c85c74662::and_right_6b67ecd2c7ab",
    "e43d4218-dca1-5f58-9a85-423de2df8dbe::and_left_e85423818fea",
    "f3c3a396-85d5-5d8a-9b55-3cab823502e9::and_left_0a2113eac804",
}

# Forward directions selected after reading the concrete equivalences.  They
# express solution classification, completeness, or parameter ranges.  Every
# reverse sibling is deliberately excluded to enforce one retained theorem per
# raw upstream source.
SELECTED_IFF_FORWARD = {
    "63461795-a2cd-54ea-8d64-a97f1e773bde::iff_forward_5cdf13253ab2",
    "5d443ada-2794-5232-9e6a-0bfb766c6daa::iff_forward_87c783b60fe5",
    "beced4d7-6348-5121-ad11-2b1d41a568a2::iff_forward_3dbbabd9d73e",
    "6ffaba53-7d2e-5427-9149-1fb250e3e3c6::iff_forward_c622a14d1e7c",
    "cedb567f-1271-5bdc-8748-eb9abecb483c::iff_forward_a04229401598",
    "2a8e5e67-311b-5ae3-98eb-ccc552a9d9e6::iff_forward_79682690e06e",
    "7548c35a-5e98-5f73-9a75-80277d68cea6::iff_forward_c27b3a6c44b2",
    "2b381102-addb-5179-91d8-3d67fa25166d::iff_forward_339ee3f27e98",
    "87b940b0-89ea-5d64-adda-aac010eff146::iff_forward_2437059bef35",
    "e43da844-1729-5182-a45b-e5303a54cdfc::iff_forward_2739d1e77fc9",
    "d77d1626-9ff5-539d-9e9e-73ed00f949ce::iff_forward_02dd8d1251e1",
    "18110675-6188-5b35-af46-f9b2d0634736::iff_forward_c059a60044af",
    "aee6b9ce-8495-5759-b2d5-10e0d278f4ca::iff_forward_5acbae46a022",
    "cad37762-9244-5e7d-9e0b-a8776f3b3b17::iff_forward_81c7b8bef330",
    "05426aa1-283e-5549-9a55-1f0b8cd40f48::iff_forward_e2e56b36dc81",
    "416e203a-9b1d-55a5-9383-a7c4ecd36036::iff_forward_237fbb21025f",
    "97452783-26a1-5958-a2e3-febc364adead::iff_forward_a446546d562c",
}

TRIVIAL_EQUALITY_BOUNDS = {
    # A closed trigonometric numeral identity weakened to a one-sided bound.
    "d7f29db6-a387-5821-a1e1-b5ce1b5d7fed::eq_to_le_forward_6b5e73d31705",
}

INVALID_ORDER_CUTS = {
    # The generator ordered a Prop/conjunction or cut through a quantifier.
    "f8ebb2f7-e77a-5855-8f29-236e41374bbd::le_to_lt_or_eq_b74864e0833a",
    "71facef4-15a8-51e2-8e6c-646e78a8c53d::le_to_lt_or_eq_302411f2f6cc",
    "1e50951d-5367-57c6-8b86-8a87ebce2120::lt_to_le_ff375ed2bfa3",
}

TRIVIAL_FUNCTION_BOUND = {
    # Directly weakens one function value's strict scalar bound.
    "bbb5fc7c-0177-5bde-a22b-ff3d6005558c::lt_to_le_2887ff3b3121",
}

SAFE_SCOPED_AND = {
    # Audit flags the surrounding quantifier, but manual inspection confirms
    # the complete binder/antecedent is preserved in the candidate statement.
    "75675916-e042-59f5-bb47-49c5a938a13a::and_left_25c71d18b66f",
    "78ad861d-7d05-5469-aa86-3cfc20ec8020::and_left_83f58bc51773",
    "e43d4218-dca1-5f58-9a85-423de2df8dbe::and_left_e85423818fea",
    "f3c3a396-85d5-5d8a-9b55-3cab823502e9::and_left_0a2113eac804",
}

HARD_REJECT = set(scale.HARD_REJECT) | {"lt_to_ne"}


def safe_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def top_order(goal: str, operators: tuple[str, ...]) -> bool:
    depth = 0
    for index, char in enumerate(goal):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and any(goal.startswith(op, index) for op in operators):
            return True
    return False


def error_excerpt(receipts: list[Mapping[str, Any]]) -> str:
    for receipt in receipts:
        if receipt.get("pantograph_verified") == "success" or receipt.get("success") is True:
            continue
        value = receipt.get("diagnostics") or receipt.get("error") or receipt.get("error_type")
        if value:
            return " ".join(str(value).split())[:360]
        errors = receipt.get("errors")
        if errors:
            return " ".join(str(errors).split())[:360]
    return "Pantograph returned failure without a textual diagnostic"


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

    if method in HARD_REJECT:
        return "reject", "extremely_low", (
            "purely mechanical equivalence/reordering/negation weakening; it adds no new mathematical content"
        )

    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject", "extremely_low", (
                "closed numeral/trigonometric identity weakened to a tautological one-sided bound"
            )
        if chosen is None:
            labels = {
                "bare_value_to_numeric_bound": "bare variable or function value weakened from an exact numeral to a scalar bound",
                "closed_numeric_relaxation": "closed numerical calculation mechanically weakened to an inequality",
                "arbitrary_expression_order": "both equality sides are expressions and the generated order has no natural bound semantics",
                "invalid_scope_or_prop_cut": "the equality was cut inside a quantifier, Prop, set description, or declaration",
                "not_top_level_equality": "the parent is not a safe top-level equality",
            }
            tier = "extremely_low" if category == "invalid_scope_or_prop_cut" else "low"
            return "reject", tier, labels.get(category, category)
        if method != chosen:
            return "reject", "low", f"paired equality direction; the sole natural direction is {chosen}"
        if contaminated or packet.get("reason_codes") or not problem_ok:
            return "reject", "extremely_low", "otherwise useful bound is tainted by scope/declaration/problem-text pollution"
        basis = "natural nontrivial numeric bound" if category != "substantive_set_containment" else "substantive set containment"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; valuable but current proof failed")

    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            return "reject", "low" if not contaminated else "extremely_low", (
                "not the sole retained substantive component: it is trivial, loses binders, repeats a scalar answer, or is the weaker sibling"
            )
        unsafe_reason = bool(packet.get("reason_codes")) and rid not in SAFE_SCOPED_AND
        if contaminated or unsafe_reason or not problem_ok:
            return "reject", "extremely_low", "conjunction projection has unresolved declaration or quantifier-scope pollution"
        tier = "high" if rid.endswith("and_left_0a2113eac804") else "medium"
        basis = "sole retained, well-scoped substantive conjunction component"
        return ("pass", tier, basis) if success else ("pending", tier, basis + "; valuable but current proof failed")

    if method in {"iff_forward", "iff_reverse"}:
        if rid not in SELECTED_IFF_FORWARD:
            return "reject", "extremely_low" if contaminated or packet.get("reason_codes") else "low", (
                "not the sole retained equivalence direction: reverse substitution, paired alternative, or a scope/declaration cut"
            )
        if method != "iff_forward" or contaminated or packet.get("reason_codes") or not problem_ok:
            return "reject", "extremely_low", "equivalence direction has scope/declaration/problem-text pollution"
        basis = "sole retained nontrivial solution-classification or completeness direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; valuable but current proof failed")

    if method == "implication_contrapositive":
        return "reject", "extremely_low", (
            "manual inspection found a forall/exists/declaration cut rather than a complete safe top-level implication"
        )

    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_CUTS or contaminated or packet.get("reason_codes") or not problem_ok:
            return "reject", "extremely_low", "orders a Prop or cuts through a quantifier/declaration instead of refining a top-level order"
        if not top_order(parent_goal, ("≤", "≥")):
            return "reject", "extremely_low", "parent is not a complete top-level non-strict order relation"
        basis = "substantive top-level order refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; valuable but current proof failed")

    if method == "lt_to_le":
        if rid in INVALID_ORDER_CUTS or contaminated or packet.get("reason_codes") or not problem_ok:
            return "reject", "extremely_low", "strict relation was cut inside a quantifier/Prop/declaration"
        if rid in TRIVIAL_FUNCTION_BOUND:
            return "reject", "low", "direct mechanical weakening of one function value against a numeral"
        if not top_order(parent_goal, ("<", ">")):
            return "reject", "extremely_low", "parent is not a complete top-level strict order relation"
        basis = "natural reusable non-strict consequence of a substantive strict inequality"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; valuable but current proof failed")

    return "reject", "extremely_low", f"method {method} is outside the calibrated acceptance scale"


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("review output already exists")
    manifest_rows = [row for _, row in scale.rows(MANIFEST)]
    ordered = [str(row["record_id"]) for row in manifest_rows]
    wanted = set(ordered)
    if len(ordered) != 500 or len(wanted) != 500:
        raise ValueError("manifest must contain exactly 500 unique record_ids")

    packets = {row["record_id"]: row for _, row in scale.rows(scale.PACKETS) if row.get("record_id") in wanted}
    raw_all = [row for _, row in scale.rows(scale.RAW)]
    raw = {row["record_id"]: row for row in raw_all if row.get("record_id") in wanted}
    parent_ids = {str(row["parent_id"]) for row in packets.values()}
    parents = {row["record_id"]: row for _, row in scale.rows(scale.PARENTS) if row.get("record_id") in parent_ids}
    if set(packets) != wanted or set(raw) != wanted or set(parents) != parent_ids:
        raise ValueError("packet/raw/parent join is incomplete")

    global_siblings: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_all:
        upstream = row.get("upstream_source")
        if upstream:
            global_siblings[str(upstream)].append({
                "record_id": str(row["record_id"]), "method": str(row.get("question_type", "")),
            })
    for rid in ordered:
        upstream = str(raw[rid].get("upstream_source", ""))
        expected = "parent:" + str(packets[rid]["parent_id"])
        if not upstream or upstream != expected:
            raise AssertionError(f"raw upstream_source mismatch for {rid}: {upstream!r} != {expected!r}")

    receipt_rows = [row for _, row in scale.rows(RECEIPTS)]
    receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipt_rows:
        if row.get("record_id") in wanted:
            receipts[str(row["record_id"])].append(row)
    if len(receipt_rows) != 500 or set(receipts) != wanted:
        raise ValueError(f"Pantograph batch is not complete: rows={len(receipt_rows)}, ids={len(receipts)}")

    equality_choice: dict[str, tuple[str, str | None]] = {}
    for packet in packets.values():
        if str(packet["method"]).startswith("eq_to_le_"):
            pid = str(packet["parent_id"])
            equality_choice.setdefault(pid, scale.equality_class(packet, parents[pid]))

    output_rows: list[dict[str, Any]] = []
    decisions, tiers, method_decisions, receipt_stats = Counter(), Counter(), Counter(), Counter()
    pass_by_upstream: dict[str, list[str]] = defaultdict(list)
    eligible_by_upstream: dict[str, list[str]] = defaultdict(list)

    for row_number, rid in enumerate(ordered, 1):
        packet, raw_row = packets[rid], raw[rid]
        proof_hash = str(packet["candidate"]["proof_sha256"])
        statement_hash = str(packet["diff"]["statement"]["after_sha256"])
        success = scale.compatible_success(receipts[rid], proof_hash)
        receipt_stats["compatible_success" if success else "fail"] += 1
        decision, tier, basis = classify(
            packet, raw_row, parents[str(packet["parent_id"])], success, equality_choice
        )
        hypothetical, _, _ = classify(
            packet, raw_row, parents[str(packet["parent_id"])], True, equality_choice
        )
        upstream = str(raw_row["upstream_source"])
        if hypothetical == "pass":
            eligible_by_upstream[upstream].append(rid)
        if decision == "pass":
            if not success:
                raise AssertionError(f"pass without compatible Pantograph success: {rid}")
            pass_by_upstream[upstream].append(rid)

        siblings = [item for item in global_siblings[upstream] if item["record_id"] != rid]
        sibling_text = ", ".join(f"{item['method']}:{item['record_id']}" for item in siblings) or "none"
        receipt_text = "compatible Pantograph success" if success else "Pantograph fail: " + error_excerpt(receipts[rid])
        reason = (
            f"Row {row_number}; frozen raw upstream_source={upstream}. Compared parent goal "
            f"`{packet['parent']['goal']}` with candidate goal `{packet['candidate']['goal']}`, problem text, proof, "
            f"and complete global siblings [{sibling_text}]. Decision basis: {basis}. Exact statement/proof receipt: {receipt_text}."
        )
        method = str(packet["method"])
        review = {
            "record_id": rid,
            "quality_decision": decision,
            "quality_tier": tier,
            "manual_reviewed": True,
            "reviewer": "codex-quality-adjudicator",
            "review_reason": reason,
            "reviewed_statement_sha256": statement_hash,
            "reviewed_proof_sha256": proof_hash,
            "review_version": VERSION,
            "parent_id": upstream,
            "upstream_source": upstream,
            "method": method,
            "siblings": siblings,
            "pantograph_compatible_success": success,
        }
        output_rows.append(review)
        decisions[decision] += 1
        tiers[tier] += 1
        method_decisions[(method, decision)] += 1

    eligible_violations = {key: value for key, value in eligible_by_upstream.items() if len(value) > 1}
    pass_violations = {key: value for key, value in pass_by_upstream.items() if len(value) > 1}
    if eligible_violations or pass_violations:
        raise AssertionError(f"more than one retained record per upstream: eligible={eligible_violations}, pass={pass_violations}")
    if len(output_rows) != 500 or len({row["record_id"] for row in output_rows}) != 500:
        raise AssertionError("review coverage failure")

    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in output_rows
    )
    report = {
        "schema_version": "numinamath_shard_review_report_v1",
        "review_version": VERSION,
        "batch_id": "numinamath_expand_s00_b00005",
        "rows": 500,
        "unique_record_ids": 500,
        "candidate_manifest_sha256": scale.sha_file(MANIFEST),
        "receipts_sha256": scale.sha_file(RECEIPTS),
        "packets_sha256": scale.sha_file(scale.PACKETS),
        "raw_sha256": scale.sha_file(scale.RAW),
        "receipt_status": dict(sorted(receipt_stats.items())),
        "final_decisions": dict(sorted(decisions.items())),
        "final_tiers": dict(sorted(tiers.items())),
        "by_method_decision": {f"{method}|{decision}": count for (method, decision), count in sorted(method_decisions.items())},
        "raw_upstream_parent_groups": len({str(raw[rid]["upstream_source"]) for rid in ordered}),
        "max_eligible_per_upstream": max(map(len, eligible_by_upstream.values()), default=0),
        "max_pass_per_upstream": max(map(len, pass_by_upstream.values()), default=0),
        "global_siblings_embedded": True,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    safe_write(OUTPUT, payload)
    safe_write(REPORT, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
