#!/usr/bin/env python3
"""Strict raw-upstream review for NuminaMath shard s00/b00007."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00007"
MANIFEST = BATCH / "candidate_manifest.jsonl"
RECEIPTS = BATCH / "verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00007.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00007_report.json")
VERSION = "numinamath_s00_b00007_strict_upstream_review_v1"

base_spec = importlib.util.spec_from_file_location(
    "numinamath_b6_review_helpers", ROOT / "scripts/review_numinamath_s00_b00006.py"
)
base = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base
base_spec.loader.exec_module(base)
scale = base.scale

# Exactly one mathematically substantive component was retained after comparing
# every raw sibling sharing the same upstream source.  Atomic answers, plain
# hypothesis restatements, lost binders, and polluted declarations were omitted.
SELECTED_AND = {
    "a9f9d791-16a9-5258-9043-08cb8328c354::and_left_d53b12a9fb75",
    "fac9f090-4b23-5820-8ea6-38bf425b8e26::and_left_ec91a6ff5cc8",
    "2edcf756-2696-512e-b43d-8c5a417c2e00::and_right_827bc1a2197b",
    "1af669df-51ac-5db5-8082-24df44e33752::and_left_c92a491336be",
    "c2da08f1-4208-5adf-93f2-916310f937d2::and_right_7efd18668e56",
    "87f0568b-e6b3-5e37-8ef8-8fc8f8c24034::and_left_27c95013f7d7",
    "bedd2eef-a5fe-5b59-adcb-145a76a487d7::and_right_8f58c73a7701",
    "cb842f5f-6217-5146-8451-12afb48a975b::and_right_b37bcd6b9927",
    "ebb744a7-0064-5d0c-833f-a58f5bf97e1e::and_left_531055f7546e",
    "5f846f31-651a-562c-8650-352f5592ea21::and_right_f28f9a500f48",
    "b1a8a18f-1684-5d9d-b717-6ecfcc052c51::and_right_148e792400dc",
    "938c4ae3-f099-5c77-aaff-e77cac23075f::and_right_fd5e0fc3a99d",
    "cec7632c-b438-5ca8-9c54-99b20fbcb97c::and_right_3de0ab46647a",
    "685602bd-5714-5e28-8632-e6597cea7cce::and_left_fdf067e43885",
    "457516d3-0fdd-511d-9ce6-d56d696a2dc1::and_left_1238028e47e0",
    "710c6c31-1d41-5191-b1bf-59a4aa61b9ee::and_right_cb821470bbc4",
    "c5424692-7a1d-5d4f-a9c1-987187f146a0::and_right_8734918b2cb6",
    "ecb7564d-7cc8-58c8-b937-6b1db2fb5252::and_right_251aaedccafc",
    "5275e439-aeec-51db-b9d5-efb5e5776b52::and_left_4f2e540f021c",
}

TRIVIAL_EQUALITY_BOUNDS = {
    # Exact function iterate at a numeral -> one-sided numeral bound is the
    # calibrated mechanical weakening, not a new mathematical target.
    "4bc5bd6d-c3d7-5d6e-9390-48562574db3d::eq_to_le_forward_519ffd1d1421",
    "b4ba8603-a29b-5a53-b351-e3940c4149d0::eq_to_le_forward_c8a21a1f1c9e",
}

SELECTED_CONTRAPOSITIVE = {
    "cdfa0750-eea9-593b-b0ff-109d9a7c0eca::implication_contrapositive_1bf2e4fc91c6",
}

INVALID_ORDER_REFINEMENTS = {
    # These cut an order token out of a quantified/conjoined proposition.
    "8fc07119-7cea-59d5-b402-e85e65447a86::le_to_lt_or_eq_f9e4027c9874",
    "f017ceb1-7860-5257-a1f4-c964138e2773::le_to_lt_or_eq_8be6291a3785",
    # Bare parameter bounds rewritten as strict-or-equal add no substantive
    # mathematical content.
    "681e169c-bcc9-53e6-b507-caabb64dc651::le_to_lt_or_eq_e40eec56dd6e",
    "a7424e2b-c0f3-5210-b2ba-a9f1b725c1c4::le_to_lt_or_eq_4e12e24f5cdc",
    "06cc20a5-19a7-5e47-bac9-cb1e81185aa6::le_to_lt_or_eq_2cd05d4b4fb5",
}

TRIVIAL_STRICT_WEAKENINGS = {
    # Closed numeral inequalities or a bare sequence value bounded by a numeral.
    "f3181212-7ab1-598f-9791-628283705356::lt_to_le_be243bbd53f7",
    "6c9d0074-3343-5887-b9ef-7a1b347189de::lt_to_le_62c3938fe327",
    "0cefa9da-d2b8-57da-bc0c-0a99f860279f::lt_to_le_de56a867f206",
    "cd2817be-965c-5877-8e35-071f1d65026c::lt_to_le_d7934872613c",
}

HARD_REJECT = set(scale.HARD_REJECT) | {"lt_to_ne"}

DECLARATION_POLLUTION = {
    "d6f96937-c091-5c36-bed8-55d74e2c9ad6::and_left_251955d7724c",
    "282a637b-b972-5027-9408-7ba8de816e0d::and_left_e2b6cd77b7bb",
}


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

    if rid in DECLARATION_POLLUTION:
        return "reject", "extremely_low", "parent statement contains a leaked declaration terminator `:=`; the projected candidate is polluted"

    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject", "low", "bare function value at a numeral weakened from equality to a scalar bound"
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
        basis = "substantive set containment" if category == "substantive_set_containment" else "natural nontrivial numerical bound"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            tier = "extremely_low" if contaminated or reasons else "low"
            return "reject", tier, "not the sole retained global sibling: scalar answer, hypothesis restatement, lost binder, weaker paired component, or pollution"
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
        tier = "extremely_low" if contaminated or reasons else "low"
        return "reject", tier, "reverse/substitution sibling is not retained; the global forward classification direction owns this upstream"

    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low", "forall/exists/conjunction-internal arrow was mistaken for a safe top-level implication"
        if contaminated or reasons or not problem_ok or scale.top_arrow(parent_goal) is None:
            return "reject", "extremely_low", "contrapositive has unresolved scope or declaration pollution"
        basis = "complete well-scoped top-level contrapositive with substantive digit/divisibility content"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok or not base.base.top_order(parent_goal, ("≤", "≥")):
            return "reject", "extremely_low", "not a clean complete top-level non-strict order relation"
        basis = "substantive top-level order refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "low", "closed numerical or bare function-value inequality mechanically weakened"
        if contaminated or reasons or not problem_ok or not base.base.top_order(parent_goal, ("<", ">")):
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
        receipt_text = "compatible Pantograph success" if success else "Pantograph fail: " + base.base.error_excerpt(receipts[rid])
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
        "batch_id": "numinamath_expand_s00_b00007", "rows": 500, "unique_record_ids": 500,
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
    base.base.safe_write(OUTPUT, payload)
    base.base.safe_write(REPORT, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
