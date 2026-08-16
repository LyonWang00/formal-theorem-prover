#!/usr/bin/env python3
"""Strict raw-upstream review for NuminaMath shard s00/b00009."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00009"
MANIFEST = BATCH / "candidate_manifest.jsonl"
RECEIPTS = BATCH / "verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00009.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00009_report.json")
VERSION = "numinamath_s00_b00009_strict_upstream_review_v1"

helper_spec = importlib.util.spec_from_file_location(
    "numinamath_b8_review_helpers", ROOT / "scripts/review_numinamath_s00_b00008.py"
)
helper = importlib.util.module_from_spec(helper_spec)
sys.modules[helper_spec.name] = helper
helper_spec.loader.exec_module(helper)
scale = helper.scale
b5 = helper.b5

SELECTED_AND = {
    "f4e6b5c5-4290-576a-a645-b897674aef46::and_right_f8f277b491e8",
    "dbe83154-d97a-5f78-8be4-fb5f2c7a36ed::and_right_aa219e3c4527",
    "24b03010-199c-542d-9499-3ad348133f21::and_left_da5603c4aa8f",
    "7b143989-95df-5560-8f40-b6cacb18e68f::and_right_1c3900651a10",
    "cc9afc15-cc7c-5ce1-9f9c-93a8cc4c98af::and_right_5458cb3de988",
    "c89c157a-e025-5ae2-a2fd-a6ab1b615ada::and_right_438e0aa492bf",
    "44ba98db-ae02-5be9-b78b-1cc37f4b6799::and_right_8baa0b406822",
    "ea5445e5-7b3a-5238-8011-5584d691d245::and_right_22d8b1b49612",
    "df2ddf57-cba0-5c36-90ff-4655ec00877e::and_left_707dc971cf79",
    "0b9b26a7-a854-5f0a-a689-74b62f586e83::and_right_4d3e3c8a8db2",
    "9fe9c58a-724c-5050-8f2a-98d79535ea2a::and_left_d81d086cbea3",
}

TRIVIAL_EQUALITY_BOUNDS = {
    "101a9153-e188-5213-b593-7caa98e1f689::eq_to_le_forward_0bb25911e8d9",
    "70448f2d-c1c7-5c3c-bc5f-c9fc01e94848::eq_to_le_forward_06dd1b46636f",
    "2fb4a006-35e9-51a1-bd54-f05c8cac0261::eq_to_le_forward_8540a8b23dab",
    "ad53eb1b-cf89-56cb-a4bd-fc26a5d5b883::eq_to_le_forward_7d89ad39a73b",
    "94d398ca-926b-5f77-85b8-0d5932f5fb85::eq_to_le_forward_63895a9e1ba0",
    "9a1a9e3c-5a3d-5101-bff3-2a89c8ecd64d::eq_to_le_forward_29233663721f",
    "e3bd446f-6dbe-54d2-a076-bdffaf078495::eq_to_le_forward_d181f30a7635",
    "bd167c7a-d2c2-5333-b722-b9918ae1c4b7::eq_to_le_forward_9ba027911107",
    "b380f54f-3528-5251-9d3f-a8c75867349c::eq_to_le_forward_5fc343f3fae8",
    "68dd7f1b-4bef-58fa-b932-928b51099723::eq_to_le_forward_1c03aeae475a",
    "6e5f36b6-2e2d-5487-9383-13d80dafc1e7::eq_to_le_forward_423dfec8194d",
    "1e26a5b6-c9fb-58f5-94d3-d17317954454::eq_to_le_forward_1343aa7b994c",
    "9389b201-729d-511e-a5f1-b2a1d0711930::eq_to_le_forward_ce9f17768df0",
}

SELECTED_CONTRAPOSITIVE = {
    "b138fc58-9a76-5d77-a371-3796887a7791::implication_contrapositive_640c98a3063c",
}

INVALID_ORDER_REFINEMENTS = {
    "8d433b5f-756b-59f6-beac-53a316874e7e::le_to_lt_or_eq_da1d580f8d55",
    "86f97dc5-99d4-5116-8d38-d2edc7d86006::le_to_lt_or_eq_7059e38b3ec6",
    "fbf41beb-ae63-581b-b082-e8cb3fe006f3::le_to_lt_or_eq_7a7e25bec25a",
    "bd46c9a6-ea4a-5e9e-a80d-e772250253b1::le_to_lt_or_eq_a1b0e35c61de",
    "05e21a57-3f1a-5d69-97f0-c8a7b1be8e2a::le_to_lt_or_eq_9d1c8491b454",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "5567586c-36cc-5c96-add1-84e7db956fb8::lt_to_le_c08caef62b69",
    "cf8f4101-46e7-51c5-a9f0-23867e2c553c::lt_to_le_7aaa22f57d28",
    "fba9ed76-ec12-5faa-8275-86eb08a32a7f::lt_to_le_6813884045fb",
    "09d3a834-eec4-5179-ae0c-bb4817399e9d::lt_to_le_8204019fa298",
    "f49c89a2-c177-5be1-923d-940f44798b8d::lt_to_le_c7d2fdecf802",
}

TRIVIAL_IFF_FORWARD = {
    "996516f4-8b37-5e1c-b360-4e379842ab69::iff_forward_4d7b6c824e0c",
}

HARD_REJECT = set(scale.HARD_REJECT) | {"lt_to_ne"}


def classify(packet: Mapping[str, Any], raw: Mapping[str, Any], parent: Mapping[str, Any],
             success: bool, equality_choice: Mapping[str, tuple[str, str | None]]) -> tuple[str, str, str]:
    rid = str(packet["record_id"]); method = str(packet["method"]); pid = str(packet["parent_id"])
    parent_goal = str(packet["parent"]["goal"])
    contaminated = scale.contaminated(packet, raw); problem_ok = scale.problem_ok(method, raw)
    reasons = bool(packet.get("reason_codes"))
    if method in HARD_REJECT:
        return "reject", "extremely_low", "pure reordering, symmetry, negation weakening, or proof-only variation"
    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject", "low", "pure computed answer, remainder, closed value, physical ratio, or function-value equality made less informative"
        if chosen is None:
            labels = {"bare_value_to_numeric_bound": "bare variable/function value weakened to a scalar bound",
                      "arbitrary_expression_order": "arbitrary ordering of expression-valued equality sides",
                      "invalid_scope_or_prop_cut": "equality cut inside a quantifier, Prop, set, conjunction, or declaration",
                      "closed_numeric_relaxation": "closed numerical equality mechanically weakened",
                      "not_top_level_equality": "parent is not a safe top-level equality"}
            return "reject", "extremely_low" if category == "invalid_scope_or_prop_cut" else "low", labels.get(category, category)
        if method != chosen:
            return "reject", "low", f"paired equality direction; sole natural direction is {chosen}"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "candidate has scope, declaration, or problem-text pollution"
        basis = "natural nontrivial numerical or algebraic bound"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            return "reject", "extremely_low" if contaminated or reasons else "low", "not the sole substantive global sibling: scalar answer, weak component, lost binder, or pollution"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "selected-looking component has unresolved scope/declaration pollution"
        basis = "sole retained well-scoped substantive range, divisibility, periodicity, or inequality component"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method == "iff_forward":
        if rid in TRIVIAL_IFF_FORWARD:
            return "reject", "low", "forward direction merely specializes a universal hypothesis at one index"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "iff was split under quantifier/negation or a polluted declaration"
        basis = "sole retained nontrivial classification, completeness, construction, or parameter direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "iff_reverse":
        return "reject", "extremely_low" if contaminated or reasons else "low", "reverse sibling is not retained; global forward direction owns this upstream"
    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low", "quantifier-internal arrow, negation cut, or disjunction repackaging is not substantive"
        if contaminated or reasons or not problem_ok or scale.top_arrow(parent_goal) is None:
            return "reject", "extremely_low", "contrapositive has unresolved scope pollution"
        basis = "complete top-level contrapositive proving a substantive nonexistence result"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok or not b5.top_order(parent_goal, ("≤", "≥")):
            return "reject", "extremely_low", "order token was cut inside exists/conjunction/Prop, or a bare parameter bound was merely restated"
        basis = "substantive top-level order refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "low", "closed/function-value weakening or quantifier-scope cut"
        if contaminated or reasons or not problem_ok or not b5.top_order(parent_goal, ("<", ">")):
            return "reject", "extremely_low", "strict relation was cut inside a quantifier, Prop, or declaration"
        basis = "natural reusable non-strict consequence of a substantive strict relation"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    return "reject", "extremely_low", f"method {method} is outside the calibrated acceptance scale"


def main() -> None:
    if OUTPUT.exists() or REPORT.exists(): raise FileExistsError("review output already exists")
    ordered = [str(row["record_id"]) for _, row in scale.rows(MANIFEST)]; wanted = set(ordered)
    if len(ordered) != 500 or len(wanted) != 500: raise ValueError("manifest must contain 500 unique record_ids")
    packets = {row["record_id"]: row for _, row in scale.rows(scale.PACKETS) if row.get("record_id") in wanted}
    raw_all = [row for _, row in scale.rows(scale.RAW)]
    raw = {row["record_id"]: row for row in raw_all if row.get("record_id") in wanted}
    parent_ids = {str(row["parent_id"]) for row in packets.values()}
    parents = {row["record_id"]: row for _, row in scale.rows(scale.PARENTS) if row.get("record_id") in parent_ids}
    if set(packets) != wanted or set(raw) != wanted or set(parents) != parent_ids: raise ValueError("packet/raw/parent join is incomplete")
    sibling_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_all:
        if row.get("upstream_source"):
            sibling_groups[str(row["upstream_source"])].append({"record_id": str(row["record_id"]), "method": str(row.get("question_type", ""))})
    for rid in ordered:
        expected = "parent:" + str(packets[rid]["parent_id"])
        if str(raw[rid].get("upstream_source", "")) != expected: raise AssertionError(f"raw upstream mismatch for {rid}")
    receipt_rows = [row for _, row in scale.rows(RECEIPTS)]; receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipt_rows:
        if row.get("record_id") in wanted: receipts[str(row["record_id"])].append(row)
    if len(receipt_rows) != 500 or set(receipts) != wanted: raise ValueError(f"Pantograph batch incomplete: rows={len(receipt_rows)}, ids={len(receipts)}")
    equality_choice: dict[str, tuple[str, str | None]] = {}
    for packet in packets.values():
        if str(packet["method"]).startswith("eq_to_le_"):
            equality_choice.setdefault(str(packet["parent_id"]), scale.equality_class(packet, parents[str(packet["parent_id"])]))
    output_rows = []; decisions, tiers, methods, receipt_stats = Counter(), Counter(), Counter(), Counter()
    eligible: dict[str, list[str]] = defaultdict(list); passes: dict[str, list[str]] = defaultdict(list)
    for number, rid in enumerate(ordered, 1):
        packet, raw_row = packets[rid], raw[rid]; proof_hash = str(packet["candidate"]["proof_sha256"])
        statement_hash = str(packet["diff"]["statement"]["after_sha256"])
        success = scale.compatible_success(receipts[rid], proof_hash); receipt_stats["compatible_success" if success else "fail"] += 1
        decision, tier, basis = classify(packet, raw_row, parents[str(packet["parent_id"])], success, equality_choice)
        hypothetical, _, _ = classify(packet, raw_row, parents[str(packet["parent_id"])], True, equality_choice)
        upstream = str(raw_row["upstream_source"])
        if hypothetical == "pass": eligible[upstream].append(rid)
        if decision == "pass":
            if not success: raise AssertionError(f"pass without compatible success: {rid}")
            passes[upstream].append(rid)
        siblings = [item for item in sibling_groups[upstream] if item["record_id"] != rid]
        sibling_text = ", ".join(f"{x['method']}:{x['record_id']}" for x in siblings) or "none"
        receipt_text = "compatible Pantograph success" if success else "Pantograph fail: " + b5.error_excerpt(receipts[rid])
        reason = (f"Row {number}; frozen raw upstream_source={upstream}. Compared parent goal `{packet['parent']['goal']}` "
                  f"with candidate goal `{packet['candidate']['goal']}`, problem text, proof, and complete global siblings [{sibling_text}]. "
                  f"Decision basis: {basis}. Exact statement/proof receipt: {receipt_text}.")
        method = str(packet["method"])
        output_rows.append({"record_id": rid, "quality_decision": decision, "quality_tier": tier, "manual_reviewed": True,
                            "reviewer": "codex-quality-adjudicator", "review_reason": reason,
                            "reviewed_statement_sha256": statement_hash, "reviewed_proof_sha256": proof_hash,
                            "review_version": VERSION, "parent_id": upstream, "upstream_source": upstream,
                            "method": method, "siblings": siblings, "pantograph_compatible_success": success})
        decisions[decision] += 1; tiers[tier] += 1; methods[(method, decision)] += 1
    eligible_bad = {k: v for k, v in eligible.items() if len(v) > 1}; pass_bad = {k: v for k, v in passes.items() if len(v) > 1}
    if eligible_bad or pass_bad: raise AssertionError(f"multiple retained siblings: eligible={eligible_bad}, pass={pass_bad}")
    if len(output_rows) != 500 or len({r["record_id"] for r in output_rows}) != 500: raise AssertionError("coverage failure")
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in output_rows)
    report = {"schema_version": "numinamath_shard_review_report_v1", "review_version": VERSION,
              "batch_id": "numinamath_expand_s00_b00009", "rows": 500, "unique_record_ids": 500,
              "candidate_manifest_sha256": scale.sha_file(MANIFEST), "receipts_sha256": scale.sha_file(RECEIPTS),
              "packets_sha256": scale.sha_file(scale.PACKETS), "raw_sha256": scale.sha_file(scale.RAW),
              "receipt_status": dict(sorted(receipt_stats.items())), "final_decisions": dict(sorted(decisions.items())),
              "final_tiers": dict(sorted(tiers.items())),
              "by_method_decision": {f"{m}|{d}": n for (m, d), n in sorted(methods.items())},
              "raw_upstream_parent_groups": len({raw[rid]["upstream_source"] for rid in ordered}),
              "max_eligible_per_upstream": max(map(len, eligible.values()), default=0),
              "max_pass_per_upstream": max(map(len, passes.values()), default=0),
              "global_siblings_embedded": True, "output_sha256": hashlib.sha256(payload).hexdigest()}
    b5.safe_write(OUTPUT, payload); b5.safe_write(REPORT, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__": main()
