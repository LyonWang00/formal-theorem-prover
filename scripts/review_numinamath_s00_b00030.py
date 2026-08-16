#!/usr/bin/env python3
"""Frozen strict review for NuminaMath shard s00/b00030.

The semantic decisions in this file were made before Pantograph receipts were
available.  The inherited writer intentionally refuses to emit a resolved
sidecar until all 500 current receipts exist; a retained candidate then becomes
``pass`` only for a current success and otherwise remains ``pending``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "numinamath_b21_review_base", ROOT / "scripts/review_numinamath_s00_b00021.py"
)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
base = previous.base
scale = previous.scale

base.BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00030"
base.BATCH_ID = base.BATCH.name
base.MANIFEST = base.BATCH / "candidate_manifest.jsonl"
base.RECEIPTS = base.BATCH / "verification_results.jsonl"
base.OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00030_resolved.jsonl"
base.REPORT = base.OUTPUT.with_name("review_s00_b00030_resolved_report.json")
base.VERSION = "numinamath_s00_b00030_strict_frozen_v1"

# These are the only ordinary conjunction projections that remain useful after
# reading the complete parent and all global siblings.  They retain a variance,
# a nontrivial sharp range endpoint, and a quadratic-form endpoint.  Scalar
# coordinates, direct transitivity fragments, escaped existentials, and the
# weaker/incomplete endpoint siblings are deliberately excluded.
SELECTED_AND = {
    "4b850a75-d90f-58ae-a84b-04612cd67a33::and_right_a3b3b968ecec",
    "50f20e3d-da42-573f-bcda-97116c348909::and_right_46043576fb0e",
    "9b7ffc03-f54c-5521-a25e-54d7f1e3a0b3::and_right_03cca9210ead",
}

# Unlike scalar equalities, exact range equality has a natural, reusable
# containment consequence which is materially weaker but not a numerical
# template mutation.
ACCEPTED_EQUALITY_BOUNDS = {
    "d42dd21b-f567-587a-8e22-972c00ca5fc0::eq_to_le_forward_807a58f46e71",
}

SELECTED_CONTRAPOSITIVE: set[str] = set()

INVALID_IFF_FORWARD = {
    # Translation-instruction pollution survives in the problem text.
    "391bd4f5-2b8d-5774-ba7d-1cda16a8c181::iff_forward_966d6a9e0ca6",
}

INVALID_ORDER_REFINEMENTS = {
    # Quantifier/Prop-valued cuts rather than ordered mathematical terms.
    "350194d5-4f1a-524e-8ee9-fb4ebea1ef35::le_to_lt_or_eq_3d9845e4bf2e",
    "ea597f0a-f391-58cb-a457-74853e9b5a0e::le_to_lt_or_eq_ebd9f6d2c8a7",
    # Bare scalar thresholds or low-value numerical variants of already
    # reviewed min/max templates.
    "94fb445c-323a-59cf-9ced-680007db7bc5::le_to_lt_or_eq_9dd1b8901f0e",
    "441512cd-3ea3-5085-8996-799077ddd675::le_to_lt_or_eq_c13893401641",
    "3d9168dd-9f4f-5725-b059-57df7a28adf4::le_to_lt_or_eq_079a7baf7962",
    "f76b7756-4b64-55b3-a951-9aaba3e433f3::le_to_lt_or_eq_695ee02a2e43",
    "c3d3273c-46df-5b0c-ad0d-48d0ba2d4355::le_to_lt_or_eq_b62cd4c95750",
    "68153648-7527-524d-aede-4d06b46b14ad::le_to_lt_or_eq_ac91bb926c38",
    "456a7ad3-6ddf-5970-978e-5084268170e6::le_to_lt_or_eq_063495ab8514",
    "fc1c7929-4442-5cdf-ac30-363e47610dbb::le_to_lt_or_eq_706c95f3ef66",
    "13c07509-448d-5595-89db-701d6c4b2328::le_to_lt_or_eq_839e0259839b",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "c5a1e372-77b6-5e6d-a062-b71788ec6492::lt_to_le_e3a7f166169d",
}

CROSS_SHARD_DEMOTIONS: set[str] = set()

NEAR_DUPLICATE_DEMOTIONS = {
    # Exact normalized theorems already retained in reviewed shards.
    "ed418d7c-9559-54b3-9b49-24a271daffec::iff_forward_1a395a822c77",
    "077c83b8-4f25-5303-a872-c1de00d6c0ce::le_to_lt_or_eq_0bad0b7c35b2",
    "2357d12b-01bb-5bd4-9d09-cbd1e46bf6fb::le_to_lt_or_eq_f02661b0857d",
    # Same cyclic rational inequality after harmless factor ordering.
    "3f2a8905-44e6-5494-8427-5041e399902d::le_to_lt_or_eq_7101f6c3fcd3",
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
        if rid not in ACCEPTED_EQUALITY_BOUNDS:
            return "reject", "low", "exact scalar, expression, sequence, or function-value result mechanically weakened"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "scope, declaration, or problem-text pollution"
        basis = "natural image-containment consequence of an exact polynomial range classification"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            return "reject", "extremely_low" if contaminated or reasons else "low", "not the sole substantive global sibling: scalar answer, incomplete endpoint, transitivity fragment, escaped binder, or prior duplicate"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "selected-looking component has unresolved scope/declaration/text pollution"
        basis = "sole retained substantive variance or sharp range endpoint after global sibling comparison"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "iff_forward":
        if rid in INVALID_IFF_FORWARD or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "iff split escaped a binder or contains declaration/problem-text pollution"
        basis = "complete forward solution, classification, impossibility, parametrization, or sharp parameter direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "iff_reverse":
        return "reject", "extremely_low" if contaminated or reasons else "low", "reverse sibling rejected; global forward solution/classification direction owns this upstream"

    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low" if contaminated or reasons else "low", "escaped binder, vacuous premise, or low-value logical repackaging"
        basis = "complete top-level contrapositive of a substantive implication"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "bare/numerical boundary, Prop cut, or scope/declaration/text pollution"
        basis = "substantive top-level sharp inequality refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "low", "bare sign/multiple-choice weakening adds no reusable inequality content"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "strict relation cut inside quantifier/Prop/declaration"
        basis = "natural reusable non-strict consequence of a substantive strict inequality"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    return "reject", "extremely_low", f"method {method} is outside calibrated acceptance scale"


if __name__ == "__main__":
    base.classify = classify
    base.main()
