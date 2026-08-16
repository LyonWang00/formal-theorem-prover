#!/usr/bin/env python3
"""Frozen strict review for NuminaMath shard s00/b00031.

This binds no decision to a stale or missing compile result.  The inherited
writer emits a resolved sidecar only after all 500 Pantograph receipts exist.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "numinamath_b30_review_base", ROOT / "scripts/review_numinamath_s00_b00030.py"
)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
base = previous.base
scale = previous.scale

base.BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00031"
base.BATCH_ID = base.BATCH.name
base.MANIFEST = base.BATCH / "candidate_manifest.jsonl"
base.RECEIPTS = base.BATCH / "verification_results.jsonl"
base.OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00031_resolved.jsonl"
base.REPORT = base.OUTPUT.with_name("review_s00_b00031_resolved_report.json")
base.VERSION = "numinamath_s00_b00031_strict_frozen_v1"

SELECTED_AND = {
    "41859089-a269-503c-ac3d-c3535c080aa0::and_right_2f5e0dcf9292",
    "2e726805-eb31-560b-bdff-b100bcd4b91b::and_right_2d613797eb0b",
    "c36484a5-e6dc-5a6d-98f9-731f817069cf::and_right_de33f64c2b20",
    "2265f62c-8518-5493-af97-ef7303323c57::and_right_86a8b6034a2a",
    "711130e3-d5ec-5f47-b61b-c86460652103::and_right_727f7771c0fe",
    "869bfaea-0a10-5e68-a5a9-428271f66f0c::and_left_a3f75e3fa294",
    "df8e6377-5b48-51f6-8198-9864a98089c3::and_left_cde01d30025c",
}

ACCEPTED_EQUALITY_BOUNDS: set[str] = set()

SELECTED_CONTRAPOSITIVE = {
    "248674e8-0e29-5818-af01-811d60472b99::implication_contrapositive_fe48d9a4bb04",
}

INVALID_IFF_FORWARD = {
    # Bare one-step multiple-choice arithmetic is not a high-quality
    # classification augmentation.
    "35fd53a4-396b-51ac-b589-552e10c8c608::iff_forward_f81c257800dd",
    # Translation instruction appended to the mathematical problem.
    "655603b3-39e8-570e-b947-5bb066bc92f0::iff_forward_a702cd885d06",
}

INVALID_ORDER_REFINEMENTS = {
    # Prop/conjunction/negation/existential scope cuts.
    "96e914ec-94af-5be5-9947-02872749acbf::le_to_lt_or_eq_aff9406cabfa",
    "3918255a-b443-5b69-a4e6-963999292fe0::le_to_lt_or_eq_38f153ebe519",
    "7d6e6140-37e8-5822-afb4-f88c1efe0ff6::le_to_lt_or_eq_ff064e9ee08e",
    "7d88fd89-2957-5fd8-83f5-e736e104d7c2::le_to_lt_or_eq_3c2012f647c2",
    # Bare sign/parameter/cardinality thresholds and numerical template
    # variants do not clear the medium-quality gate.
    "f885022a-0224-530f-a670-3d99be880f0a::le_to_lt_or_eq_fae4d3e0a8b9",
    "86a2f7d7-f88e-590e-9294-92f885afb85f::le_to_lt_or_eq_1a78084b913d",
    "6fb9d551-1793-5d79-997e-8ce3f887276b::le_to_lt_or_eq_3c5217eef645",
    "286585cb-f4cd-5a2e-9092-c250f0c6655d::le_to_lt_or_eq_61e95068baef",
    "1a3eb0be-768b-5390-9f51-785ef1f7a7a2::le_to_lt_or_eq_e27a4c55a0d9",
    "2b0ffef8-7cf6-590f-b8c4-bc8ff46d1e13::le_to_lt_or_eq_e4651ed4d5cd",
    "fdb77e54-0e67-5ef4-99f2-193479ff8dcf::le_to_lt_or_eq_d9587a01656c",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "b65b17f5-f276-5363-a9e8-12f79b7ee056::lt_to_le_6e519482e30f",
}

CROSS_SHARD_DEMOTIONS: set[str] = set()

NEAR_DUPLICATE_DEMOTIONS = {
    # Exact normalized order refinements already retained historically.
    "d8ca4dfb-3641-5f37-b6e0-241b699441ce::le_to_lt_or_eq_6a01d4f3632f",
    "6279f3a9-ded8-5592-a692-2669a6c47c88::le_to_lt_or_eq_dccfc6ec6629",
    "4b89374d-a005-5d0f-9aa3-2c221d4a1245::le_to_lt_or_eq_503f48f1066d",
    "61c24ed4-90a0-5045-a3bb-04b33fbda87a::le_to_lt_or_eq_23eab13d7e0e",
    "12c254d5-37b2-5224-884a-c4f399a79fa7::le_to_lt_or_eq_bc0aee94b120",
    # Variable-permuted form of the same retained (x+y)(y+z) bound.
    "186f8a78-8d74-5350-8d22-050cf4fd0515::le_to_lt_or_eq_401345a4b482",
    # This exact upper conjunction component was already retained.
    "beed2b26-42dc-53ed-bee9-b04146dfe62c::and_right_d3c2cad3c2f5",
}


def classify(
    packet: Mapping[str, Any],
    raw: Mapping[str, Any],
    parent: Mapping[str, Any],
    success: bool,
    equality_choice: Mapping[str, tuple[str, str | None]],
) -> tuple[str, str, str]:
    del parent
    rid = str(packet["record_id"])
    method = str(packet["method"])
    pid = str(packet["parent_id"])
    contaminated = scale.contaminated(packet, raw)
    problem_ok = scale.problem_ok(method, raw)
    reasons = bool(packet.get("reason_codes"))

    if method in base.HARD_REJECT:
        return "reject", "extremely_low", "logical repackaging, commutation, symmetry, squaring, negated-bound restatement, or proof-only variation"
    if rid in CROSS_SHARD_DEMOTIONS:
        return "reject", "low", "content-preferred global sibling retained after parent-level comparison"
    if rid in NEAR_DUPLICATE_DEMOTIONS:
        return "reject", "low", "exact or semantically near-exact normalized theorem already retained in an earlier reviewed shard"

    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        if chosen is None:
            labels = {
                "bare_value_to_numeric_bound": "bare variable/function value weakened to a scalar bound",
                "arbitrary_expression_order": "arbitrary ordering of equality sides",
                "invalid_scope_or_prop_cut": "equality cut inside quantifier/Prop/set/conjunction/declaration",
                "closed_numeric_relaxation": "closed numerical equality mechanically weakened",
                "not_top_level_equality": "parent is not a safe top-level equality",
            }
            return "reject", "extremely_low" if category == "invalid_scope_or_prop_cut" else "low", labels.get(category, category)
        if method != chosen:
            return "reject", "low", f"paired equality direction; sole natural direction is {chosen}"
        return "reject", "low", "exact scalar, expression, sequence, or function-value result mechanically weakened"

    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            return "reject", "extremely_low" if contaminated or reasons else "low", "not the sole substantive global sibling: scalar answer, incomplete endpoint, escaped binder, direct premise fragment, or prior duplicate"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "selected-looking component has unresolved scope/declaration/text pollution"
        basis = "sole retained substantive sharp range, discriminant, digit-set, or structural arithmetic component"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "iff_forward":
        if rid in INVALID_IFF_FORWARD or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "trivial arithmetic, escaped binder, or declaration/problem-text pollution"
        basis = "complete forward solution, classification, impossibility, parametrization, or sharp parameter direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "iff_reverse":
        return "reject", "extremely_low" if contaminated or reasons else "low", "reverse sibling rejected; global forward solution/classification direction owns this upstream"

    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low" if contaminated or reasons else "low", "escaped binder, scalar calculation, or low-value logical repackaging"
        if contaminated or reasons or not problem_ok or scale.top_arrow(str(packet["parent"]["goal"])) is None:
            return "reject", "extremely_low", "selected contrapositive has scope/declaration/text pollution"
        basis = "complete top-level contrapositive of a substantive divisibility implication"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "bare/numerical boundary, Prop cut, or scope/declaration pollution"
        basis = "substantive top-level sharp inequality refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "low", "bare comparison/multiple-choice weakening adds no reusable inequality content"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "strict relation cut inside quantifier/Prop/declaration"
        basis = "natural reusable non-strict consequence of a substantive strict inequality"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    return "reject", "extremely_low", f"method {method} is outside calibrated acceptance scale"


if __name__ == "__main__":
    base.classify = classify
    base.main()
