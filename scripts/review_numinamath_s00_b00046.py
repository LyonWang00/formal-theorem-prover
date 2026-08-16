#!/usr/bin/env python3
"""Frozen strict review for NuminaMath shard s00/b00046.

The 500 semantic decisions were frozen before Pantograph receipts existed.
The inherited writer therefore refuses to emit a resolved sidecar until all
500 current receipts are present.  A manually retained item becomes ``pass``
only for a hash-compatible success; otherwise it is emitted as ``pending``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "numinamath_b41_review_base", ROOT / "scripts/review_numinamath_s00_b00041.py"
)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
base = previous.base
scale = previous.scale

base.BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00046"
base.BATCH_ID = base.BATCH.name
base.MANIFEST = base.BATCH / "candidate_manifest.jsonl"
base.RECEIPTS = base.BATCH / "verification_results.jsonl"
base.OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00046_resolved.jsonl"
base.REPORT = base.OUTPUT.with_name("review_s00_b00046_resolved_report.json")
base.VERSION = "numinamath_s00_b00046_strict_frozen_v1"

# Complete forward solution/classification directions.  Other forward splits
# in this batch duplicate an already retained floor, functional-equation, or
# decimal-digit classification (including harmless notation changes).
SELECTED_IFF_FORWARD = {
    "501236fd-e406-5e7b-bb87-e785f3661d24::iff_forward_b79185b3e165",
    "896b2a6e-f00d-5fc8-a607-21ffc8ac7ff6::iff_forward_3e936ca4b9d5",
    "fc05f7a8-5adb-5e84-b601-9f2f53eb8ad6::iff_forward_c2046186eae2",
    "73bc0a1c-43d7-58dd-b682-b7d2bf2be0e3::iff_forward_77b11c15e872",
}

# Sole substantive projection per frozen raw upstream after reading every
# global sibling.  Scalar coordinates, symmetric identities, repeated range
# endpoints, and escaped existential conjuncts are deliberately absent.
SELECTED_AND = {
    "31757dee-df28-5553-839d-3b95a7221b52::and_right_766c81c2bcf6",
    "4d2aaf8e-efa8-5bba-aff0-ed8983c899a4::and_left_9ed0f548c714",
    "f3530b78-3fe9-5f39-8dcf-f868607b0dda::and_right_3f0480442ead",
    "f12bc2e6-096e-5c9b-ac5c-ad8d92de6fbd::and_right_383e69adea92",
    "c0d7bf32-5435-5f99-b12d-97553df71c98::and_right_7b1058f9f9d3",
    "4028da50-092f-5d9b-aec0-0bf9d421e2a3::and_left_54e9836a84b5",
}

# Natural reusable non-strict consequences of genuinely nontrivial strict
# inequalities.  The remaining clean-looking item is only the scalar
# sequence evaluation a 100 > 2^99, and four others cut quantifier scope.
SELECTED_LT_TO_LE = {
    "22699256-5705-5472-9c00-6b4cc753dbfd::lt_to_le_bc8c4bd627d2",
    "3a6e7be6-63cd-5f28-bb0a-a9c56cf2be22::lt_to_le_25d778c39045",
    "2ad5dd8f-f846-5ebf-bab3-88d0aefc89fa::lt_to_le_bd090559c588",
}

# Each item below is a distinct, nontrivial sharp inequality whose equality
# case has mathematical content.  The much larger rejected remainder is
# dominated by exact/notation-only duplicates in reviewed history and this
# batch.  Numeric-variable/function thresholds are never selected.
SELECTED_ORDER_REFINEMENTS = {
    "fca95ad2-e652-5da5-8704-ee8181692738::le_to_lt_or_eq_76de4c9ea0be",
    "fbb82395-9a00-562c-8aa6-d78a61f46291::le_to_lt_or_eq_a2fafb2ee979",
    "993ee9fd-50f8-58da-9cbe-1aaf163a789e::le_to_lt_or_eq_4adac671395b",
    "0aa9accd-05fb-5380-9c68-50114c7f55a1::le_to_lt_or_eq_e9756f4be01b",
    "af13bea1-68c2-580e-b952-95a418fb6407::le_to_lt_or_eq_56e3f5591656",
    "9ba3fcba-3477-5730-88fc-8c24bf7179ee::le_to_lt_or_eq_ecf01799eb42",
    "606594dc-d10e-52ba-af90-68db3ce4d7ae::le_to_lt_or_eq_186c0f506c1d",
    "44685178-5f94-5c52-b5aa-a88e1d34e76a::le_to_lt_or_eq_b9c2a4b9fc38",
    "3c475bbd-9ce4-536e-8445-b8f1863bad53::le_to_lt_or_eq_84e90b266708",
    "896aea98-eac6-5617-bc8d-d907e013cb08::le_to_lt_or_eq_78cf182242a4",
    "b448996f-2684-5012-aa81-2d686ae7f965::le_to_lt_or_eq_33941a57c0e2",
    "e909354b-e355-5239-a9de-34a469924401::le_to_lt_or_eq_2201c675ba5f",
    "7f3b13f6-19ea-5886-8871-ee4383eedb84::le_to_lt_or_eq_d897193d579a",
    "f150aafd-77e4-5bce-96d6-3b49ef7a88fd::le_to_lt_or_eq_7ae410d6af2c",
    "6f71d478-ca04-58f7-8c0f-2b0cc185dee7::le_to_lt_or_eq_03446393f337",
    "3b225e67-ca22-54e4-b425-a8bf628ae86c::le_to_lt_or_eq_11becff91c05",
    "7cfbfba8-2408-56c3-add8-913dd489537d::le_to_lt_or_eq_f0bf10e78a9c",
}

BARE_ORDER_REFINEMENTS = {
    "de4ba7fc-f18b-5832-81bd-64c86ced77b3::le_to_lt_or_eq_240ca3491d12",
    "a6a212a9-e708-5458-a022-2d97c0e5bbf5::le_to_lt_or_eq_47c39f407251",
    "285702d1-d2e7-52c2-82fc-3e696e76a979::le_to_lt_or_eq_12056deb4916",
    "3d983d9e-0c8c-5854-ac6b-7ef646de86ac::le_to_lt_or_eq_53e77a97031a",
}

ACCEPTED_EQUALITY_BOUNDS: set[str] = set()
SELECTED_CONTRAPOSITIVE: set[str] = set()


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

    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        labels = {
            "bare_value_to_numeric_bound": "bare variable/function value weakened to a scalar bound",
            "arbitrary_expression_order": "arbitrary ordering of equality sides",
            "invalid_scope_or_prop_cut": "equality cut inside a quantifier, Prop, set, conjunction, or declaration",
            "closed_numeric_relaxation": "closed numerical equality mechanically weakened",
            "not_top_level_equality": "parent is not a safe top-level equality",
        }
        if chosen is None:
            return "reject", "extremely_low" if category == "invalid_scope_or_prop_cut" else "low", labels.get(category, category)
        if method != chosen:
            return "reject", "low", f"paired equality direction; sole natural direction would be {chosen}"
        return "reject", "low", "exact scalar, sum, sequence, function-value, or numerical answer mechanically weakened"

    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            tier = "extremely_low" if contaminated or reasons else "low"
            return "reject", tier, "not the sole substantive global sibling: scalar coordinate, symmetric/repeated identity, escaped binder, weaker endpoint, or reviewed near duplicate"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "selected-looking component has unresolved scope, declaration, or problem-text pollution"
        basis = "sole retained substantive chain inequality, approximation bound, structural component, or nontrivial range endpoint after global sibling comparison"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "iff_forward":
        if rid not in SELECTED_IFF_FORWARD:
            tier = "extremely_low" if contaminated or reasons or not problem_ok else "low"
            return "reject", tier, "exact, notation-only, or numerical-template duplicate of a forward solution/classification already retained in reviewed history"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "forward split has scope, declaration, or problem-text pollution"
        basis = "complete forward solution, classification, parametrization, or sharp interval direction with distinct mathematical content"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")

    if method == "iff_reverse":
        return "reject", "extremely_low" if contaminated or reasons else "low", "reverse sibling rejected; the complete forward solution/classification owns this theorem family"

    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        return "reject", "extremely_low" if contaminated or reasons else "low", "escaped binder or low-value logical repackaging; no clean substantive top-level implication survives this shard"

    if method == "le_to_lt_or_eq":
        if rid in BARE_ORDER_REFINEMENTS:
            return "reject", "low", "bare scalar, sequence value, or single function-value boundary; strict-or-equality wrapping adds no reusable mathematical structure"
        if rid not in SELECTED_ORDER_REFINEMENTS:
            tier = "extremely_low" if contaminated or reasons or not problem_ok else "low"
            return "reject", tier, "exact/current/global or notation/numerical-template near duplicate of an inequality refinement already represented in reviewed history"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "ordered relation was cut inside scope or the declaration/problem text is polluted"
        basis = "distinct substantive top-level sharp inequality refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    if method == "lt_to_le":
        if rid not in SELECTED_LT_TO_LE:
            tier = "extremely_low" if contaminated or reasons else "low"
            return "reject", tier, "quantifier-scope cut or bare scalar sequence evaluation; mechanical weakening adds no reusable inequality content"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "strict relation was cut inside a quantifier, Prop, or declaration"
        basis = "natural reusable non-strict consequence of a distinct substantive strict inequality"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")

    return "reject", "extremely_low", f"method {method} is outside the calibrated acceptance scale"


if __name__ == "__main__":
    base.classify = classify
    base.main()
