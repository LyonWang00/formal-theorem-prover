#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00013.

This specialization deliberately reuses the frozen joins, hash binding, receipt
compatibility, global-sibling packet construction, and append-safe writes from
the b00012 reviewer.  Only the content adjudication for this shard differs.
"""

from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "numinamath_b12_review_base", ROOT / "scripts/review_numinamath_s00_b00012.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
scale = base.scale

base.BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00013"
base.BATCH_ID = base.BATCH.name
base.MANIFEST = base.BATCH / "candidate_manifest.jsonl"
base.RECEIPTS = base.BATCH / "verification_results.jsonl"
base.OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00013_resolved.jsonl"
base.REPORT = base.OUTPUT.with_name("review_s00_b00013_resolved_report.json")
base.VERSION = "numinamath_s00_b00013_strict_resolved_v1"

SELECTED_AND = {
    "e838ac16-8030-53ac-a500-9e7cdc43e688::and_right_bee7ba65a178",
    "ac04f6e7-86cf-57f9-83f3-d94aab154992::and_left_83d91d4c2d8d",
    "0e87f2ca-8d08-5ab1-a170-69175ca50502::and_right_08fb85dbb1d8",
    "25eb1de3-a9ad-56c3-8e9f-96a78cdcd176::and_left_bf8d5ef56b6b",
    "d0d4a5cc-ae23-56e3-ad6d-1d4563271b2f::and_left_65488fd4bfdc",
    "c9ab1fd2-94a1-58c7-a507-51c98841e0bb::and_left_dc02da5d5e98",
    "9989bd81-1cbb-5d40-bfe0-39748b9caec1::and_right_c282f9ce2b69",
    "e12ed591-cd8f-5e43-8a78-a773b8a9f116::and_left_5bab8cb70b04",
    "c8093637-ce24-5b1b-b6dc-0c10a0f90b8b::and_right_4e7b16251c23",
    "1bfe4b52-3361-5f7e-a057-1f678dc3de25::and_right_c910c019d3c6",
    "c57b6b27-7157-5d50-8ac2-29728ab10872::and_left_08125a7b02eb",
}

TRIVIAL_EQUALITY_BOUNDS = {
    "b646b9e4-d252-503c-a065-96f65adbb984::eq_to_le_forward_4df487d5d082",
    "02cd6e39-966b-55ac-9b50-0c1966999419::eq_to_le_forward_62e77cd336f2",
    "5d96aaa0-7ba7-5dff-af33-cccf122ef41c::eq_to_le_forward_a1e2d30e9a1d",
    "6f5f7d03-72c9-555e-857c-f5b9860e90d4::eq_to_le_forward_06811021afb7",
    "60fd1347-1bc5-560e-8193-67f09327cf46::eq_to_le_forward_c4f3ce96bab9",
    "c8cc32bd-e5ee-585f-92e3-2b6fddc119eb::eq_to_le_forward_b7884e8ce93d",
    "aab62767-1e80-5da4-8fcf-4c8a9b819cf3::eq_to_le_forward_b9ece07f718b",
    "9ce4fb26-e36e-52f0-89e3-9590642a5f9c::eq_to_le_forward_7718d60cee65",
    "d16be845-4227-5297-b77b-7c239663e616::eq_to_le_forward_bbd3f3970c63",
    "feab975a-4804-5b5a-98b7-7f880dcd2d8b::eq_to_le_forward_217dbe7b53a5",
    "c1f4a16a-db6c-5433-b970-d853e6f05417::eq_to_le_forward_451eb838aefc",
    "4cbe23bc-103b-5ec5-b720-1a3e6a26a2d1::eq_to_le_forward_3d8bfed92073",
}

SELECTED_CONTRAPOSITIVE: set[str] = set()

INVALID_ORDER_REFINEMENTS = {
    "13186c7d-dc23-5ae6-952e-3d4e6f87e8b7::le_to_lt_or_eq_456a4bde3aff",
    "088b653a-f406-5150-9622-6b96dcdc8115::le_to_lt_or_eq_ee33fc8decd4",
    "6559b520-fcf7-5f09-9354-7930b593367b::le_to_lt_or_eq_6463e41a6b49",
    "e60904ae-0b5e-514a-a389-43342e9aa64f::le_to_lt_or_eq_e22987712277",
    "af7dedee-a4b1-554c-89d6-0674140612f7::le_to_lt_or_eq_434fa7deef37",
    "03fc86c3-3830-5078-bc6b-39f7f9a432ab::le_to_lt_or_eq_9549471085dd",
    "d8a95442-53f4-5475-bb7d-2adb938fcf3b::le_to_lt_or_eq_4082ebd1ce54",
    "b2d441ad-a5ed-5166-931a-46b94f1b9b47::le_to_lt_or_eq_58353b54d73f",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "2654c760-c52c-5c76-aa46-81d942940c6f::lt_to_le_6299dedf6096",
}


def classify(packet: Mapping[str, Any], raw: Mapping[str, Any], parent: Mapping[str, Any],
             success: bool, equality_choice: Mapping[str, tuple[str, str | None]]) -> tuple[str, str, str]:
    del parent
    rid=str(packet["record_id"]); method=str(packet["method"]); pid=str(packet["parent_id"])
    goal=str(packet["parent"]["goal"]); contaminated=scale.contaminated(packet,raw)
    problem_ok=scale.problem_ok(method,raw); reasons=bool(packet.get("reason_codes"))
    if method in base.HARD_REJECT:
        return "reject","extremely_low","pure reordering, symmetry, negation weakening, or proof-only variation"
    if method.startswith("eq_to_le_"):
        category,chosen=equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject","low","closed computation, digit/mod fact, direct root evaluation, or bare scalar/function value mechanically weakened"
        if chosen is None:
            labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
            return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
        if method!=chosen: return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","scope, declaration, or problem-text pollution"
        basis="sole natural nontrivial numerical, algebraic, geometric, sequence, number-theoretic, or finite-set bound"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method in {"and_left","and_right"}:
        if rid not in SELECTED_AND:
            return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: scalar/weaker component, lost binder, or pollution"
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","selected-looking component has unresolved scope or text pollution"
        basis="sole retained substantive extremum, structural, congruence, continuity, nonuniqueness, or inequality component"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method=="iff_forward":
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","iff split under quantifier/existential, declaration, or polluted syntax"
        basis="sole retained nontrivial solution, classification, completeness, inconsistency, or parameter direction"
        return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
    if method=="iff_reverse":
        return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward direction owns this upstream"
    if method in {"implication_contrapositive","implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE: return "reject","extremely_low","non-top-level arrow, escaped binder, or premise-negation repackaging"
        if contaminated or reasons or not problem_ok or scale.top_arrow(goal) is None: return "reject","extremely_low","contrapositive has scope pollution"
        basis="complete top-level contrapositive with substantive finite-set prime-sum content"
        return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
    if method=="le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok: return "reject","extremely_low","bare scalar/function-value split or scope pollution"
        basis="substantive top-level order refined into strict and equality cases"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method=="lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS: return "reject","extremely_low","strict relation was cut inside a quantifier"
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
        basis="natural reusable non-strict consequence of a substantive strict relation"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"


if __name__ == "__main__":
    base.classify = classify
    base.main()
