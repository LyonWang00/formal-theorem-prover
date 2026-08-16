#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00020."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "numinamath_b19_review_base", ROOT / "scripts/review_numinamath_s00_b00019.py"
)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
base = previous.base
scale = previous.scale

base.BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00020"
base.BATCH_ID = base.BATCH.name
base.MANIFEST = base.BATCH / "candidate_manifest.jsonl"
base.RECEIPTS = base.BATCH / "verification_results.jsonl"
base.OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00020_resolved.jsonl"
base.REPORT = base.OUTPUT.with_name("review_s00_b00020_resolved_report.json")
base.VERSION = "numinamath_s00_b00020_strict_resolved_v1"

# These are the sole useful projections after reading the parent theorem, the
# mathematical problem text, and every global sibling.  Each is a reusable
# structural fact or a nontrivial endpoint; scalar-answer and incomplete
# multiple-choice projections are deliberately absent.
SELECTED_AND = {
    "01a1ac93-cedb-56dd-85f0-6e08cf893ed6::and_left_f433b4b1513a",
    "28b5cab0-9a1f-5891-98ea-8abd6174288e::and_left_41eb5f4349c7",
    "5e576584-a6cb-5b7e-89b0-fe0c6c827bb2::and_left_7c4a7f0dcbfb",
    "704f9921-a2d8-50be-a38e-902af0d81bcb::and_right_7d4c5947d40a",
    "7f17841e-01d4-52f2-9cfa-1c5afbc1a7b5::and_left_a10b402aa89c",
    "85f43907-9fcd-582d-9730-490415b6e3ea::and_right_579baa9406fd",
    "8ad271d5-84e8-5e1d-8c7f-72e07c8a99a2::and_right_4a4f02f99db3",
}

ACCEPTED_EQUALITY_BOUNDS = {
    # Structural cardinality and set-containment consequences, not scalar
    # variable/function evaluations.
    "327638bb-3678-50ff-b0e3-af2282785504::eq_to_le_forward_0ce069ac351b",
    "d152d5be-6690-5336-a94d-1b7ef6e2d9c3::eq_to_le_forward_ddcfdf778872",
    "f590ef3e-32d2-51b0-ba48-c7405c8b700c::eq_to_le_forward_37e17fcc1096",
}

SELECTED_CONTRAPOSITIVE = {
    "be774be5-9f74-5fb6-b086-0a861f775970::implication_contrapositive_785b311739af",
}

INVALID_IFF_FORWARD = {
    # The target contains a leaked declaration terminator.
    "25bfa1cb-9381-53ac-8e49-56b88db8f5e3::iff_forward_de53e9daf864",
    # The appended translation instruction makes the problem text disagree
    # with the requested proof task even though the Lean target is parseable.
    "d66cb21c-3550-5631-8f6a-4fc95eea9f50::iff_forward_e3bc8e18f4c5",
}

INVALID_ORDER_REFINEMENTS = {
    # Bare multiple-choice conclusion, not a substantive inequality theorem.
    "02f8ef94-df53-5b80-8c8a-154d9ff5a0e8::le_to_lt_or_eq_1cd6a7ff767c",
    # The splitter cut a conjunction/Prop rather than an ordered expression.
    "0cf8ad5a-6798-56ad-8a63-1893531e0da7::le_to_lt_or_eq_0758c016862d",
    # The formal hypotheses make 4049 an unattained coarse lower boundary;
    # the strict-or-equality wrapper adds no equality-case content.
    "10ec0791-bbd5-544d-96e9-67b329e41d5d::le_to_lt_or_eq_201f2a9563be",
    # Existential and universal binders were cut, leaving Prop-valued terms.
    "69addd5e-5596-5348-922f-8a97d1619921::le_to_lt_or_eq_5535c2d53659",
    "da08771e-0ab8-5070-a6b4-a958acc5c2d5::le_to_lt_or_eq_96e99c564f46",
}

TRIVIAL_STRICT_WEAKENINGS = {
    # Mechanical weakening of a one-bit multiple-choice sign conclusion.
    "d76f19e6-9ef2-5c7c-86cf-c6c599cb3761::lt_to_le_7a8702733510",
}

CROSS_SHARD_DEMOTIONS = {
    # The paired 14 | y conclusion is slightly more substantive than the
    # prime-divisor projection 13 | x for the same proportionality argument.
    "28b5cab0-9a1f-5891-98ea-8abd6174288e::and_left_41eb5f4349c7",
    # The line factorization retained by s01 is more informative than merely
    # stating that the three components have no common intersection.
    "5c94afdb-34f3-590d-9bc0-608696789ba0::and_right_97d6dcfe8263",
    # The product congruence is the less routine closure component.
    "c1c191fa-6ce5-5188-b185-d647ce893bf7::and_left_3ecbfa8c9b0b",
}

NEAR_DUPLICATE_DEMOTIONS = {
    # Exact normalized theorems already retained in reviewed shards.
    "2cdd25c5-aacc-55ff-bcb1-09eab47af936::and_left_e3193535d23b",
    "2ca08421-8e2c-5858-974f-cc9757d7443e::le_to_lt_or_eq_978296b97c88",
    "3531e5a4-3779-5ca2-a5f3-c982f8923c0a::lt_to_le_c655c39e3fe7",
    "48ee7466-0f7a-5456-94ad-69a484292306::le_to_lt_or_eq_37631411c8bb",
    "4b77b7db-6485-5bde-83f2-d40286ad6581::le_to_lt_or_eq_c52250dbe072",
    "516f13dc-7280-5dae-9a0f-ec56e4b7633f::le_to_lt_or_eq_0eb0c143ac2f",
    "64f7c523-a183-58a4-87e4-4971695f2692::le_to_lt_or_eq_79e3489520bf",
    "82403e90-f4d8-5ab4-bf50-8cd70f53f198::le_to_lt_or_eq_04e359f6f251",
    "86225034-f2c0-5b19-9f47-b464d28004c1::le_to_lt_or_eq_f967ac7e16ea",
    "9a114d87-0de2-54ec-8779-77d39a9efc9d::le_to_lt_or_eq_99d9218fcc21",
    "9f4a5145-ba25-514a-b024-8edf617dcbf9::lt_to_le_0610d7764f52",
    "c0759ddd-08f1-5709-aaf9-3962180b26d4::le_to_lt_or_eq_a7dcbf14ba3f",
    # Algebraically identical formatting variant of an earlier theorem.
    "53853bd0-fd8f-5561-9448-b4771589804c::le_to_lt_or_eq_6b3de6a09eb0",
    "75f23c6a-bb20-524a-adf1-b21f88fbf544::le_to_lt_or_eq_de33c5415247",
    "e45b7e89-5cf3-5079-a56c-6e10211a80e1::le_to_lt_or_eq_8b04db61ebaa",
    # Same frozen upstream content already retained in the opposite endpoint
    # or monotonicity branch in an earlier shard.
    "707f00bc-71bb-5ffe-936c-996c0116f7b5::and_right_210fb4e5fa4b",
    "a9710535-6c20-564a-8c08-aa3edc3e1e4a::and_left_e3cf206a4812",
    "cf5239ed-dfc2-55ea-b86c-ceb5006b2920::and_left_cc9184c536db",
    # Numerical variants of the same prior range template are excluded.
    "b70fc829-8985-597e-8fbd-a2a23e5cedc3::and_left_5e06a167ee62",
    "f650fca6-46f9-54e4-992c-803e6b3699e5::and_right_8cc6a0c9904b",
    "3a5ba0db-51b9-59f7-9406-46634fb2e73b::le_to_lt_or_eq_5dc3ba230d82",
    # Exact normalized duplicate inside this batch; retain c81a... only.
    "fef50e8e-897f-561c-95af-c0ff1965bd1a::lt_to_le_94cabc68a656",
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
        return "reject", "extremely_low", "logical repackaging, commutation, symmetry, negated-bound restatement, or proof-only variation"
    if rid in CROSS_SHARD_DEMOTIONS:
        return "reject", "low", "content-preferred global sibling retained after parent-level comparison"
    if rid in NEAR_DUPLICATE_DEMOTIONS:
        return "reject", "low", "exact or semantically near-exact normalized theorem already retained, or sole in-batch duplicate demoted"

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
            return "reject", "low", "exact scalar answer, sequence/function value, numerical identity, or routine expression mechanically weakened"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "scope, declaration, or problem-text pollution"
        basis = "natural structural cardinality or set-containment consequence of a substantive exact classification"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            return "reject", "extremely_low" if contaminated or reasons else "low", "not the sole substantive global sibling: scalar answer, incomplete projection, weaker endpoint, escaped binder, text pollution, or prior near duplicate"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "selected-looking component has unresolved scope/declaration/text pollution"
        basis = "sole retained substantive extremum, range, floor, divisibility, gcd, or higher-order algebraic component"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "iff_forward":
        if rid in INVALID_IFF_FORWARD or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "iff split escaped a binder, leaked syntax/declaration text, or mismatched the problem text"
        basis = "complete forward solution, classification, impossibility, parametrization, or sharp parameter direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "iff_reverse":
        return "reject", "extremely_low" if contaminated or reasons else "low", "reverse sibling rejected; the global forward solution/classification direction owns this upstream"

    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low" if contaminated or reasons else "low", "escaped binder, vacuous premise, or low-value logical repackaging"
        if contaminated or reasons or not problem_ok or scale.top_arrow(str(packet["parent"]["goal"])) is None:
            return "reject", "extremely_low", "selected contrapositive has scope/declaration/text pollution"
        basis = "complete top-level contrapositive of a substantive divisibility implication"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "bare scalar boundary, unattained coarse boundary, Prop cut, or scope/declaration pollution"
        basis = "substantive top-level sharp inequality refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "extremely_low" if reasons else "low", "bare sign/multiple-choice weakening adds no reusable inequality content"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "strict relation cut inside quantifier/Prop/declaration"
        basis = "natural reusable non-strict consequence of a substantive strict inequality"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    return "reject", "extremely_low", f"method {method} is outside calibrated acceptance scale"


if __name__ == "__main__":
    base.classify = classify
    base.main()
