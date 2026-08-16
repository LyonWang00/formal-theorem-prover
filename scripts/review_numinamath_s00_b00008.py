#!/usr/bin/env python3
"""Strict raw-upstream review for NuminaMath shard s00/b00008."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00008"
MANIFEST = BATCH / "candidate_manifest.jsonl"
RECEIPTS = BATCH / "verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00008.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00008_report.json")
VERSION = "numinamath_s00_b00008_strict_upstream_review_v1"

helper_spec = importlib.util.spec_from_file_location(
    "numinamath_b7_review_helpers", ROOT / "scripts/review_numinamath_s00_b00007.py"
)
helper = importlib.util.module_from_spec(helper_spec)
sys.modules[helper_spec.name] = helper
helper_spec.loader.exec_module(helper)
scale = helper.scale
b5 = helper.base.base

# One standalone, non-scalar conjunction component per raw upstream.  The
# choices below were made after reading all global siblings, not merely the
# candidates that happened to land in this shard.
SELECTED_AND = {
    "98f39766-8a30-5216-96bd-bd0d7cf0cfc9::and_right_159522bb3528",
    "ce76a3b6-3c63-5ea3-b1e2-e96ddb6e5ddd::and_left_adc74773102c",
    "e9846d56-477e-54d9-b44a-7a785d9956f8::and_right_e21ee1730129",
    "2af6a33f-fff4-537f-9265-989a784e9985::and_right_06dea7177d91",
    "e0ee9d4b-ee89-54ee-a39b-db0f0bfe9327::and_left_fee4df11e8f8",
    "e9e71b25-d213-5c47-86cf-de985ecb5414::and_right_a5e776cc83d6",
    "b59cf019-16ce-5af3-a728-e92b1399a983::and_left_94c2c6714151",
    "d74e496f-58a0-5b2d-b115-ad105bdeea9c::and_right_a2ee4976d055",
    "c6eb92b1-c6ed-5361-991a-f0c2216c8cf5::and_left_0a60d166ba6d",
    "896761ff-6d47-5921-b869-58e66cd43e37::and_right_dbd3308d85bd",
    "f166556f-e08b-5410-a7c9-77edf1a81406::and_right_807500170a49",
}

TRIVIAL_EQUALITY_BOUNDS = {
    # Nat.mod <= 0 merely disguises the original equality; the remaining
    # three are closed numerical identities or a bare function iterate value.
    "7e4743b3-8130-5433-bab3-6e7a63031a31::eq_to_le_forward_c2a3c369ba69",
    "bbd9ddc4-20d0-5ffc-9429-15684491b1c5::eq_to_le_forward_7b257d4c7210",
    "00553ac4-c6e9-5c20-835f-f8cffa1dc92f::eq_to_le_forward_32a9d38015d2",
    "e7d886ae-bde3-549d-ad39-b4167ab22753::eq_to_le_forward_08e8321a9419",
    # Pure computed-answer exercises whose numerical equality was merely made
    # less informative; these are not natural independent extremal claims.
    "e0b62afe-909a-5932-82d9-8341c369459a::eq_to_le_forward_5c0accee2b88",
    "ade880ca-a98d-5e6a-925c-9aafd25d6e18::eq_to_le_forward_453430980cfd",
    "3dcb7972-b51b-5a34-8abc-f6386f3ab700::eq_to_le_forward_96b6a0cb6724",
    "4f524cb7-0ce8-56af-b6ca-63dc58bfe8c7::eq_to_le_forward_9788b9850847",
    "fa7395f4-65da-5d70-a17c-8bb9df1c6fc1::eq_to_le_forward_4662b8401aaf",
    "8fb97c32-fe36-56d5-9a60-150dbf4fa4c3::eq_to_le_forward_c0ada43deeae",
    "b7ce61eb-d2c0-5bff-b894-f730fb70ba43::eq_to_le_forward_70460f71816b",
    "c895eee2-8ed8-5bf4-8993-2cca02d1f63b::eq_to_le_forward_336ecc0b63b7",
    "9a842f3b-97bc-5cb0-8f48-648760aa3046::eq_to_le_forward_341699c49a95",
    "00b0d2de-7f95-5948-8d05-f71d9c9f4957::eq_to_le_forward_73ee2266763d",
    "f3b55eb8-167f-5fb4-a62a-f3b6907b6b50::eq_to_le_forward_95f2c98b85ac",
}

SELECTED_CONTRAPOSITIVE = {
    "b078972c-df98-56ab-8036-892584c66ce1::implication_contrapositive_e01600409ea4",
}

INVALID_ORDER_REFINEMENTS = {
    "0a989133-b202-5c0d-a1aa-a3f2ddf33068::le_to_lt_or_eq_34bac0ba9f33",
    "91c16c00-336c-5414-85e6-161389be3880::le_to_lt_or_eq_a8ccccd9f66f",
    "4be86fed-4df2-5667-80af-3f2e62cd78a8::le_to_lt_or_eq_703e82b9d7b1",
    "e1c733a1-e4b6-58b0-8390-5daad0ef6537::le_to_lt_or_eq_f1d42fbe6426",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "e08e8cba-442f-5d28-833b-2573ac939af5::lt_to_le_aef1ad4dce52",
    "7fac1964-4ea3-5aea-a134-2ca4659dbb36::lt_to_le_36a1b5f692f0",
    "c2db1f35-da18-58eb-a6e2-48f59f12131e::lt_to_le_0714fd5b0fa0",
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
        return "reject", "extremely_low", "pure reordering, symmetry, negation weakening, or proof-only variation"

    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject", "low", "closed/function-value weakening, Nat.mod <= 0 disguised equality, or pure computed answer made less informative"
        if chosen is None:
            labels = {
                "bare_value_to_numeric_bound": "bare variable/function value weakened from an exact numeral to a scalar bound",
                "arbitrary_expression_order": "arbitrary ordering of two expression-valued equality sides",
                "invalid_scope_or_prop_cut": "equality cut inside a quantifier, Prop, set description, conjunction, or declaration",
                "closed_numeric_relaxation": "closed numerical equality mechanically weakened",
                "not_top_level_equality": "parent is not a safe top-level equality",
            }
            tier = "extremely_low" if category == "invalid_scope_or_prop_cut" else "low"
            return "reject", tier, labels.get(category, category)
        if method != chosen:
            return "reject", "low", f"paired equality direction; sole natural direction is {chosen}"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "candidate has scope, declaration, or problem-text pollution"
        basis = "substantive set-cardinality containment" if category == "substantive_set_containment" else "natural nontrivial numerical bound"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            tier = "extremely_low" if contaminated or reasons else "low"
            return "reject", tier, "not the sole retained global sibling: scalar answer, partial comparison, trivial positivity, lost binder, paired component, or pollution"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "selected-looking component still has unresolved scope/declaration pollution"
        basis = "sole retained well-scoped substantive extrema, uniqueness, range, or inequality component after global sibling comparison"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "iff_forward":
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "generator split an iff under exists/forall or a polluted declaration"
        basis = "sole retained nontrivial solution-classification, parameter-range, or completeness direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "iff_reverse":
        tier = "extremely_low" if contaminated or reasons else "low"
        return "reject", tier, "reverse/substitution sibling is not retained; the global forward classification direction owns this upstream"

    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low", "quantifier-internal arrow or disjunction repackaging is not a safe substantive top-level implication"
        if contaminated or reasons or not problem_ok or scale.top_arrow(parent_goal) is None:
            return "reject", "extremely_low", "contrapositive has unresolved scope or declaration pollution"
        basis = "complete well-scoped top-level contrapositive with substantive divisibility content"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok or not b5.top_order(parent_goal, ("≤", "≥")):
            return "reject", "extremely_low", "order token was cut inside an exists, conjunction, Prop, or declaration"
        basis = "substantive top-level order refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "low", "closed numerical/trigonometric inequality or bare product positivity mechanically weakened"
        if contaminated or reasons or not problem_ok or not b5.top_order(parent_goal, ("<", ">")):
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
        receipt_text = "compatible Pantograph success" if success else "Pantograph fail: " + b5.error_excerpt(receipts[rid])
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
        decisions[decision] += 1
        tiers[tier] += 1
        methods[(method, decision)] += 1

    eligible_bad = {key: value for key, value in eligible.items() if len(value) > 1}
    pass_bad = {key: value for key, value in passes.items() if len(value) > 1}
    if eligible_bad or pass_bad:
        raise AssertionError(f"multiple retained siblings: eligible={eligible_bad}, pass={pass_bad}")
    if len(output_rows) != 500 or len({row["record_id"] for row in output_rows}) != 500:
        raise AssertionError("coverage failure")

    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in output_rows)
    report = {
        "schema_version": "numinamath_shard_review_report_v1", "review_version": VERSION,
        "batch_id": "numinamath_expand_s00_b00008", "rows": 500, "unique_record_ids": 500,
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
    b5.safe_write(OUTPUT, payload)
    b5.safe_write(REPORT, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
