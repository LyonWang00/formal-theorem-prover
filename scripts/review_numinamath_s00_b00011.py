#!/usr/bin/env python3
"""Strict raw-upstream review for NuminaMath shard s00/b00011."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00011"
MANIFEST = BATCH / "candidate_manifest.jsonl"
RECEIPTS = BATCH / "verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00011.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00011_report.json")
VERSION = "numinamath_s00_b00011_strict_upstream_review_v1"

spec = importlib.util.spec_from_file_location(
    "numinamath_b10_helpers", ROOT / "scripts/review_numinamath_s00_b00010.py"
)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
scale = helper.scale
b5 = helper.b5

# Each item is the sole retained substantive component in its complete
# frozen-raw sibling group.
SELECTED_AND = {
    "facf5f75-59ed-5140-809d-1f13e9b7cfbc::and_right_aca0e575ffe9",
    "8261afbe-675d-5753-9aa7-3ed5408d5639::and_right_d9f060abbc76",
    "7fbdd44c-1c10-5e53-b8b7-4e3b40386268::and_left_cf1bba9b816b",
    "18546ab3-25eb-5876-972f-399472bb2e21::and_left_e2753928ddbc",
    "6fb64914-b351-564d-8bbc-da8dfc8f4ab2::and_left_913253e1ed48",
    "43c27aaa-9b6f-5ada-aab2-be69f60465dd::and_left_78e7d9de2a28",
    "06746875-9171-59a4-b13b-81cb8d8bb244::and_right_284f365bc18c",
    "458a5edf-838d-50e1-8fac-04dedb1177d9::and_left_10c9aefc2c69",
    "a5620a1c-e24c-504d-8043-ed0e3582554f::and_right_ac1adeb898a1",
    "c92de990-eb2c-5836-9911-2841b28a63fc::and_left_9366938e02b7",
    "f3abe591-1f19-5eab-aa63-0e9555078a83::and_right_60e413e1cd21",
    "42cdf500-c0f5-57f9-aaa3-d77dad2a2451::and_left_53ecb2a53472",
    "14b690dd-6b45-55f0-ac69-91a5aae6a8f8::and_left_58ea65d2d2de",
    "68c680fb-6de0-5b95-981c-0f1c3f6a66d5::and_left_c06883adae0d",
    "09adc718-4b49-5bc1-a945-4e89f384c34b::and_left_64f3ade4cd37",
}

# The automatic equality classifier found a superficially natural direction,
# but manual inspection found a computed/closed value, contradiction-driven
# model, unrelated curve variables, or Nat.mod <= 0.
TRIVIAL_EQUALITY_BOUNDS = {
    "dfc99645-5add-5ac9-a4ef-69bf2cc8c14b::eq_to_le_forward_3b3dd82edc48",
    "bad00187-b4ed-5546-b928-b8d7202c218a::eq_to_le_forward_86662c4fe405",
    "2f4e32fe-7415-5fad-9327-d52ec73b709e::eq_to_le_forward_80ec45fe714f",
    "e4db806b-7b39-5883-90e5-25ef9212c288::eq_to_le_forward_6aa2755a6079",
    "cd41b134-b573-5fc4-82f3-b33a3aa06a64::eq_to_le_forward_f52c550ed9b8",
    "4b4d2911-2bb4-5533-95fa-a43fad8d6265::eq_to_le_forward_a65b0bb3e203",
    "f7c5f06d-7c8c-5c0a-b250-bb7fbbce3b6b::eq_to_le_forward_a208a7fed5ca",
    "560ce173-96e6-588f-bba4-fd2c26914b4c::eq_to_le_forward_70b4f6d3162f",
}

INVALID_IFF_FORWARD = {
    "fcfffc25-e4dd-5185-a47b-d5000ec2fa13::iff_forward_622fd2471e6c",
    "e0e9ead9-117c-5044-a0c0-77e891815121::iff_forward_0992c75b084e",
    "745a336c-3d7c-55f1-9240-bbf3353460bc::iff_forward_80daa2deb7b8",
}

SELECTED_CONTRAPOSITIVE = {
    "26e0dd77-b6ae-5601-8908-d2d7a0d0d249::implication_contrapositive_47f53131efca",
    "44b9575d-5cc9-566d-bf5e-c771fd59354c::implication_contrapositive_5363749a85da",
    "60cee035-d9d7-5e8d-a1ae-937648d0d26f::implication_contrapositive_6c212f9ea2f8",
}

INVALID_ORDER_REFINEMENTS = {
    "8132445a-e315-5992-8aae-36571082d437::le_to_lt_or_eq_e3a4f60f3a2f",
    "9b95da8c-a373-55a2-ab96-cadd3eb52769::le_to_lt_or_eq_d65d3319b3f9",
    "c397f60d-516e-5f8c-8cd0-eaffbae40555::le_to_lt_or_eq_ac70ad3f30d3",
    "2834ce2e-2c3c-5e34-9e28-f4637d6fcd14::le_to_lt_or_eq_82b598cbbcc3",
    "67d4b5b8-3cc6-54d7-ab16-ad71010d4336::le_to_lt_or_eq_1d1bdd7c19e4",
    "6e474ac8-c285-5bb3-9d86-d007d2c32ed2::le_to_lt_or_eq_31896258735e",
    "9837c051-899a-5e16-a39a-1ce44606d4d3::le_to_lt_or_eq_5877a2c4e3ec",
    "dcb74846-e003-544f-9a4d-03a2580fd2a7::le_to_lt_or_eq_29a702c49125",
    "aa22de24-4724-5d84-af84-74139b162a70::le_to_lt_or_eq_bc6e14acde85",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "18b8ac61-f831-59b1-8a36-a20b51070d96::lt_to_le_697c63d5df2f",
    "e3be4db0-ff68-55ed-90cc-89b3d9a73c8e::lt_to_le_ab0ba429dc4d",
}

HARD_REJECT = set(scale.HARD_REJECT) | {"lt_to_ne"}


def classify(
    packet: Mapping[str, Any], raw: Mapping[str, Any], parent: Mapping[str, Any],
    success: bool, equality_choice: Mapping[str, tuple[str, str | None]],
) -> tuple[str, str, str]:
    del parent
    rid = str(packet["record_id"])
    method = str(packet["method"])
    pid = str(packet["parent_id"])
    goal = str(packet["parent"]["goal"])
    contaminated = scale.contaminated(packet, raw)
    problem_ok = scale.problem_ok(method, raw)
    reasons = bool(packet.get("reason_codes"))
    if method in HARD_REJECT:
        return "reject", "extremely_low", "pure reordering, symmetry, negation weakening, or proof-only variation"
    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject", "low", "computed/closed value, contradiction-driven model, unrelated variables, or Nat.mod <= 0"
        if chosen is None:
            labels = {
                "bare_value_to_numeric_bound": "bare variable/function value weakened to a scalar bound",
                "arbitrary_expression_order": "arbitrary ordering of equality sides",
                "invalid_scope_or_prop_cut": "equality cut inside quantifier/Prop/set/conjunction/declaration",
                "closed_numeric_relaxation": "closed numerical equality mechanically weakened",
                "not_top_level_equality": "parent is not a safe top-level equality",
            }
            tier = "extremely_low" if category == "invalid_scope_or_prop_cut" else "low"
            return "reject", tier, labels.get(category, category)
        if method != chosen:
            return "reject", "low", f"paired equality direction; sole natural direction is {chosen}"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "scope, declaration, or problem-text pollution"
        basis = "natural nontrivial numerical, algebraic, or set-containment bound"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            tier = "extremely_low" if contaminated or reasons else "low"
            return "reject", tier, "not the sole substantive global sibling: scalar component, lost binder, weaker side, or pollution"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "selected-looking component has unresolved scope or text pollution"
        basis = "sole retained substantive range, extremum, set, geometric, or inequality component"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method == "iff_forward":
        if rid in INVALID_IFF_FORWARD:
            return "reject", "extremely_low", "closed contradiction, problem-text pollution, or formal/prose semantic mismatch"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "iff split under quantifier/existential, declaration, or polluted syntax"
        basis = "sole retained nontrivial classification, completeness, construction, or parameter direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "iff_reverse":
        tier = "extremely_low" if contaminated or reasons else "low"
        return "reject", tier, "reverse sibling rejected; global forward direction owns this upstream"
    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low", "non-top-level arrow, escaped binder, or premise-negation repackaging"
        if contaminated or reasons or not problem_ok or scale.top_arrow(goal) is None:
            return "reject", "extremely_low", "contrapositive has scope or declaration pollution"
        basis = "complete top-level contrapositive with substantive set, congruence, or parity content"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "bare scalar, conjunction/quantifier cut, or problem-text pollution"
        basis = "substantive top-level order refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "low", "closed numerical weakening or existential-scope cut"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "strict relation cut inside quantifier/Prop/declaration"
        basis = "natural reusable non-strict consequence of a substantive strict relation"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    return "reject", "extremely_low", f"method {method} is outside calibrated acceptance scale"


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("review output already exists")
    ordered = [str(r["record_id"]) for _, r in scale.rows(MANIFEST)]
    wanted = set(ordered)
    if len(ordered) != 500 or len(wanted) != 500:
        raise ValueError("manifest must contain 500 unique record_ids")
    packets = {r["record_id"]: r for _, r in scale.rows(scale.PACKETS) if r.get("record_id") in wanted}
    raw_all = [r for _, r in scale.rows(scale.RAW)]
    raw = {r["record_id"]: r for r in raw_all if r.get("record_id") in wanted}
    pids = {str(r["parent_id"]) for r in packets.values()}
    parents = {r["record_id"]: r for _, r in scale.rows(scale.PARENTS) if r.get("record_id") in pids}
    if set(packets) != wanted or set(raw) != wanted or set(parents) != pids:
        raise ValueError("packet/raw/parent join incomplete")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_all:
        if row.get("upstream_source"):
            groups[str(row["upstream_source"])].append({
                "record_id": str(row["record_id"]), "method": str(row.get("question_type", "")),
            })
    for rid in ordered:
        if str(raw[rid].get("upstream_source", "")) != "parent:" + str(packets[rid]["parent_id"]):
            raise AssertionError(f"raw upstream mismatch {rid}")
    receipt_rows = [r for _, r in scale.rows(RECEIPTS)]
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
    out: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    methods: Counter[tuple[str, str]] = Counter()
    receipt_stats: Counter[str] = Counter()
    eligible: dict[str, list[str]] = defaultdict(list)
    passes: dict[str, list[str]] = defaultdict(list)
    for number, rid in enumerate(ordered, 1):
        packet = packets[rid]
        raw_row = raw[rid]
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
                raise AssertionError(f"pass without success {rid}")
            passes[upstream].append(rid)
        siblings = [x for x in groups[upstream] if x["record_id"] != rid]
        sibling_text = ", ".join(f"{x['method']}:{x['record_id']}" for x in siblings) or "none"
        receipt_text = "compatible Pantograph success" if success else "Pantograph fail: " + b5.error_excerpt(receipts[rid])
        reason = (
            f"Row {number}; frozen raw upstream_source={upstream}. Compared parent goal `{packet['parent']['goal']}` "
            f"with candidate goal `{packet['candidate']['goal']}`, problem text, proof, and complete global siblings "
            f"[{sibling_text}]. Decision basis: {basis}. Exact statement/proof receipt: {receipt_text}."
        )
        method = str(packet["method"])
        out.append({
            "record_id": rid, "quality_decision": decision, "quality_tier": tier,
            "manual_reviewed": True, "reviewer": "codex-quality-adjudicator", "review_reason": reason,
            "reviewed_statement_sha256": statement_hash, "reviewed_proof_sha256": proof_hash,
            "review_version": VERSION, "parent_id": upstream, "upstream_source": upstream,
            "method": method, "siblings": siblings, "pantograph_compatible_success": success,
        })
        decisions[decision] += 1
        tiers[tier] += 1
        methods[(method, decision)] += 1
    eligible_bad = {k: v for k, v in eligible.items() if len(v) > 1}
    pass_bad = {k: v for k, v in passes.items() if len(v) > 1}
    if eligible_bad or pass_bad:
        raise AssertionError(f"multiple retained siblings eligible={eligible_bad} pass={pass_bad}")
    if len(out) != 500 or len({r['record_id'] for r in out}) != 500:
        raise AssertionError("coverage failure")
    payload = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode() for r in out)
    report = {
        "schema_version": "numinamath_shard_review_report_v1", "review_version": VERSION,
        "batch_id": "numinamath_expand_s00_b00011", "rows": 500, "unique_record_ids": 500,
        "candidate_manifest_sha256": scale.sha_file(MANIFEST), "receipts_sha256": scale.sha_file(RECEIPTS),
        "packets_sha256": scale.sha_file(scale.PACKETS), "raw_sha256": scale.sha_file(scale.RAW),
        "receipt_status": dict(sorted(receipt_stats.items())), "final_decisions": dict(sorted(decisions.items())),
        "final_tiers": dict(sorted(tiers.items())),
        "by_method_decision": {f"{m}|{d}": n for (m, d), n in sorted(methods.items())},
        "raw_upstream_parent_groups": len({raw[r]["upstream_source"] for r in ordered}),
        "max_eligible_per_upstream": max(map(len, eligible.values()), default=0),
        "max_pass_per_upstream": max(map(len, passes.values()), default=0),
        "global_siblings_embedded": True, "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    b5.safe_write(OUTPUT, payload)
    b5.safe_write(REPORT, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
