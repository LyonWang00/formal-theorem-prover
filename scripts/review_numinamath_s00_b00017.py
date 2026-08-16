#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00017."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "numinamath_b16_review_base", ROOT / "scripts/review_numinamath_s00_b00016.py"
)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
base = previous.base
scale = previous.scale

base.BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00017"
base.BATCH_ID = base.BATCH.name
base.MANIFEST = base.BATCH / "candidate_manifest.jsonl"
base.RECEIPTS = base.BATCH / "verification_results.jsonl"
base.OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00017_resolved.jsonl"
base.REPORT = base.OUTPUT.with_name("review_s00_b00017_resolved_report.json")
base.VERSION = "numinamath_s00_b00017_strict_resolved_v1"

SELECTED_AND = {
    "b899f7b0-d6e2-5f42-8bed-1bd1cd87f0ff::and_left_269d1227215e",
    "7d53e48b-3873-5b97-b7b4-1ab992610ce2::and_left_7739d98ddaed",
    "f19ab417-2f17-5989-9d2e-71889ee46b16::and_left_bccfbe14036a",
    "a04a338a-9e86-579f-b4ef-66c3d2275412::and_right_37ef95db7610",
    "12e546ee-7de5-50bf-92fe-08b80c072955::and_left_31ae56dbf227",
    "28a82d60-8b5a-5a54-bd52-2c885e7b05c7::and_left_e4aaca359b34",
    "542b9dba-b30f-5102-a25d-6cea0ca776b9::and_left_dd9e894294df",
    "dbde434b-0a80-5cad-b8e8-0a208255c143::and_right_5c4d0889e3d3",
    "32e7b594-d5d7-5d13-8589-e7b6c6868939::and_left_16e6c2a4d5ba",
    "f83545a0-afa3-5589-8f91-34ef8d4e622f::and_left_de281b7269c7",
    "92f8c45b-29c3-54c8-8b55-91efc978b603::and_left_1ffd9c7406d8",
    "f4f426e3-d671-50eb-a57a-99143ee7cc7b::and_right_ddeb357ec08a",
    "87be8edd-e438-5a47-8311-464fca765e2f::and_left_ddeed5811175",
    "45351b8a-63ee-5a60-b29e-8446b1953687::and_right_55c0863ea966",
    "ffb3b1c6-f551-5bf5-bcbd-48e42b923eb5::and_right_e295c58b5c16",
    "913a2bb4-4b2f-5d93-b7f4-8ab24dc31b1c::and_left_91bc38578e4f",
    "4b5b6abf-6043-5642-bf18-9ffeced35cdf::and_left_46a8a10fa81d",
    "53da6c25-ded8-5fa2-81d8-b740eff446fa::and_right_8dcccec01a07",
    "f40e6a1e-7042-5365-bd7d-508e2ef206e0::and_left_3d72412173e4",
}

SAFE_WHOLE_BINDER_AND = {
    "b899f7b0-d6e2-5f42-8bed-1bd1cd87f0ff::and_left_269d1227215e",
    "7d53e48b-3873-5b97-b7b4-1ab992610ce2::and_left_7739d98ddaed",
}

# Equality weakening is accepted only when the one-sided target is independently
# natural and reusable.  Exact answer, digit, sequence-value, closed computation,
# and routine numeric-expression rows are rejected even if the generic detector
# labels them natural_nontrivial_upper_bound.
ACCEPTED_EQUALITY_BOUNDS = {
    "d3af9f4a-a972-594c-b99e-99e02ff19ff9::eq_to_le_forward_e0d5bf7c175a",
    "42a7962c-b7ff-5afa-8964-7b27d3d2f5ac::eq_to_le_forward_323f84ddb659",
    "2198e418-0a7f-5834-a437-3d3c60261f23::eq_to_le_forward_eee323f7da7c",
    "39603434-6d98-5d25-930a-5ea16943e812::eq_to_le_forward_4e39d433aa98",
    "8299202a-4ea7-54b2-bb5d-9a95a589aff8::eq_to_le_forward_1c81045eff2e",
    "773264b3-13e6-5f8c-908f-6044fde89d7f::eq_to_le_forward_fe8991f732da",
    "b913b10c-14ce-5764-8a30-fa9a466c2abd::eq_to_le_reverse_310893ebdd13",
}

INVALID_ORDER_REFINEMENTS = {
    "ba51dd03-694d-5436-b446-82d98c81660d::le_to_lt_or_eq_624212e07000",
    "974a0ad4-042d-5ea0-8ed9-a762da2d0117::le_to_lt_or_eq_3b3b52d392dc",
    "5aa898a0-8b3a-52ef-9848-d935184501eb::le_to_lt_or_eq_a407a753f373",
    "80d85b2d-0c7f-5fb1-8442-0ff934b93eb3::le_to_lt_or_eq_8bcdded62289",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "acab6ec8-ca00-5a8c-aa6f-50797139b557::lt_to_le_8a0b3a2bd862",
}

# Filled by the frozen all-history audit below; entries are explicit so every
# demotion remains inspectable and reproducible.
CROSS_SHARD_DEMOTIONS = {
    # The paired upper endpoint x <= n is already the retained reviewed
    # consequence for this exact parent; do not keep both interval projections.
    "28a82d60-8b5a-5a54-bd52-2c885e7b05c7::and_left_e4aaca359b34",
    # Cross-shard content adjudication prefers the sharp upper endpoint.
    "87be8edd-e438-5a47-8311-464fca765e2f::and_left_ddeed5811175",
    # The constructive representation of every prime is more substantial than
    # the paired non-primality projection and is already a compiled s01 pass.
    "ffb3b1c6-f551-5bf5-bcbd-48e42b923eb5::and_right_e295c58b5c16",
}
NEAR_DUPLICATE_DEMOTIONS = {
    "ef417704-2843-5ebc-a50c-d97f8a91b274::iff_forward_7dc32d78d1bf",
    "d9a9be6d-5a6c-54c2-ad29-4745e4909ae4::le_to_lt_or_eq_abc9fc99390d",
    "2198e418-0a7f-5834-a437-3d3c60261f23::eq_to_le_forward_eee323f7da7c",
    "1b34d218-0aad-5483-bfef-11f675a994b6::le_to_lt_or_eq_9a8d23c976ba",
    "12e546ee-7de5-50bf-92fe-08b80c072955::and_left_31ae56dbf227",
    "fb63a601-b0df-58f4-8214-495f969ec4be::le_to_lt_or_eq_23265b049e14",
    "a04a338a-9e86-579f-b4ef-66c3d2275412::and_right_37ef95db7610",
    "8fd641cb-bc18-54db-80d3-c6ca86646654::le_to_lt_or_eq_d67976164b4d",
    "773264b3-13e6-5f8c-908f-6044fde89d7f::eq_to_le_forward_fe8991f732da",
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
        return "reject", "low", "exact or near-exact normalized theorem already retained in an earlier reviewed shard"
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
            return "reject", "low", "exact numeric answer, digit, sequence value, closed evaluation, or routine scalar expression mechanically weakened"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "scope, declaration, or problem-text pollution"
        basis = "natural structural set/degree consequence or nontrivial reusable geometric, algebraic, or finite-series bound"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            return "reject", "extremely_low" if contaminated or reasons else "low", "not the sole substantive global sibling: incomplete projection, weaker/trivial endpoint, escaped binder, declaration pollution, or exact scalar answer"
        safe_binder = rid in SAFE_WHOLE_BINDER_AND
        if contaminated or (reasons and not safe_binder) or not problem_ok:
            return "reject", "extremely_low", "selected-looking component has unresolved scope/declaration/text pollution"
        basis = "sole retained substantive quantified recurrence, sharp bound, monotonicity, periodicity, extremum, or number-theoretic component"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method == "iff_forward":
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "iff split escaped an exists/forall/let binder or has declaration/text pollution"
        basis = "complete forward solution, classification, impossibility, parametrization, or sharp-constant necessity direction"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "iff_reverse":
        return "reject", "extremely_low" if contaminated or reasons else "low", "reverse sibling rejected; global forward completeness direction owns this upstream"
    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        return "reject", "extremely_low" if contaminated or reasons else "low", "non-top-level arrow, escaped binder, or low-value logical repackaging"
    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "bare numeric boundary, Prop/conjunction cut, or scope/declaration pollution"
        basis = "substantive top-level sharp inequality refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "low", "bare scalar parameter comparison mechanically weakened"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "strict relation cut inside quantifier/Prop/declaration"
        basis = "natural reusable non-strict consequence of a substantive cyclic strict inequality"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    return "reject", "extremely_low", f"method {method} is outside calibrated acceptance scale"


if __name__ == "__main__":
    base.classify = classify
    base.main()
