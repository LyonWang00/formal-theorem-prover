#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00016."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "numinamath_b15_review_base", ROOT / "scripts/review_numinamath_s00_b00015.py"
)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
base = previous.base
scale = previous.scale

base.BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00016"
base.BATCH_ID = base.BATCH.name
base.MANIFEST = base.BATCH / "candidate_manifest.jsonl"
base.RECEIPTS = base.BATCH / "verification_results.jsonl"
base.OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00016_resolved.jsonl"
base.REPORT = base.OUTPUT.with_name("review_s00_b00016_resolved_report.json")
base.VERSION = "numinamath_s00_b00016_strict_resolved_v1"

SELECTED_AND = {
    "a2f9b336-f66d-5db1-999b-bfb07404dc64::and_left_fc3d445f9d4a",
    "eb8f08de-74bc-53b8-b9e3-40e475cf6606::and_left_4cb2d3afa758",
    "abc3ca66-8803-5502-84c9-4a57774a7de1::and_right_a9fa13278576",
    "4900820d-f8ab-57d7-8118-8219cb5d385b::and_left_1fde5d14b580",
    "39b89b35-1817-5f1c-b090-e0b8716fc1cc::and_left_382eb3ea8702",
    "b5169a51-0b27-545b-90ab-bb7857a9086f::and_left_03315ab87f72",
    "6573b559-adcb-5c4b-9be4-dd8ab48b8166::and_left_985b418100d7",
    "d2cfd13c-f7c7-5213-8475-957c7e60ea9f::and_left_524743cf0300",
    "5f20e408-ca1e-5833-bc93-6bf87c308f21::and_left_18f73cf76cc9",
}

SAFE_WHOLE_BINDER_AND = {
    "4900820d-f8ab-57d7-8118-8219cb5d385b::and_left_1fde5d14b580",
    "39b89b35-1817-5f1c-b090-e0b8716fc1cc::and_left_382eb3ea8702",
}

TRIVIAL_EQUALITY_BOUNDS = {
    "2ab625a5-6c2e-557a-b0dc-4af21b4515a4::eq_to_le_forward_9670566d535e",
    "1f0b6c3b-a6c2-58f1-b43e-3e1659488dbf::eq_to_le_forward_ce11a1c41952",
    "79bed316-a310-5b6d-a37a-fbef36e0df30::eq_to_le_forward_98914f439901",
    "384e8226-7983-5cc0-97b6-3402c11215fd::eq_to_le_forward_934fda857c66",
    "4565c9e6-8886-5dc9-b5ce-fc9bc11dc302::eq_to_le_forward_2735da91064a",
    "8a3c9979-186a-5d1e-857f-0c73f89d0821::eq_to_le_forward_4aaf7cee2dbf",
    "9fd0a35f-5124-5a04-b919-8bf8a8dcd602::eq_to_le_forward_1a72f215be7a",
    "8ccc5ed6-c2e6-577a-ad1c-4f6bbe9c2d2e::eq_to_le_forward_af1a03031e74",
    "5528733f-dfa4-5106-9a32-097cc694b28e::eq_to_le_forward_6db8ba2bea34",
    "6cc1e9d8-019b-5ce6-81e5-0953240f2a3c::eq_to_le_forward_98b44b4fc258",
    "5b74d7a2-314d-519c-95ff-3a39c0224a46::eq_to_le_forward_f4616ec56183",
    "9618fff2-2a18-532a-8a5e-cd1b5ed1fd7c::eq_to_le_forward_a7aa566bdddd",
    # Further row-by-row calibration: these are merely exact numeric answers
    # (digits, sequence terms, parameter sums/products) turned into one-sided
    # scalar bounds, with no new extremal or structural content.
    "d8a810e6-916a-5ef4-b443-424f4e755f3a::eq_to_le_forward_2738bb592cf6",
    "c5d4dc48-a073-5763-9b15-a92968f606a3::eq_to_le_forward_79fb558a8da8",
    "b6a97812-98bd-5b6f-a26e-20cefa67c743::eq_to_le_forward_7a4559988504",
    "00675c1d-403d-587e-b470-b47deb0a9032::eq_to_le_forward_6078a8ffe8c8",
    "719d4144-9698-54e7-b241-8b96fc02f309::eq_to_le_forward_7204f29724ed",
    "fbb886f7-0a51-5e31-b81c-ef64c8142d63::eq_to_le_forward_d4e053cf62fc",
    "d5721b1d-ed71-5f35-8b89-75807096ebc2::eq_to_le_forward_0d80d67967f3",
    "b1a46e80-c1f0-5502-9591-c1f6f0088bbf::eq_to_le_forward_4c663c68cf4d",
    "05cc813f-1487-51ca-b3ae-9ef3c23838d0::eq_to_le_forward_0f330bf3982d",
    "2813ee14-2aa2-56ff-8a62-137e6ab9b70b::eq_to_le_forward_d173676545de",
    "c3eb5234-156a-5676-a3bc-60fcfc6ba222::eq_to_le_forward_29750f22f2c9",
    "b3f0d564-92d0-5725-91ff-07b7cc043412::eq_to_le_forward_3cb1e661e792",
    "f4401662-cd61-5608-b14e-a757bedf66cd::eq_to_le_forward_6c47e1aad9f4",
    "9da54d3b-6b88-5164-9ab8-95dee9ce491d::eq_to_le_forward_def6faee8db2",
}

# No scope-risk iff split is accepted in this batch.  In particular, the
# three superficially plausible exists-cases parse as `∃ x, (P x ↔ Q)`, not
# `(∃ x, P x) ↔ Q`; their generated implications change quantifier scope.
SAFE_WHOLE_BINDER_IFF: set[str] = set()

SELECTED_CONTRAPOSITIVE = {
    "979267e0-7baa-5ff7-9c50-16483397ebcc::implication_contrapositive_89ed965b59a3",
    "7ef82834-cf6f-52e3-9014-6d55b36be08f::implication_contrapositive_33999cc27830",
    "fe4a1001-8519-5ca8-b5e2-095dce6f3c3e::implication_contrapositive_564f81052597",
}

INVALID_ORDER_REFINEMENTS = {
    "5cda28b8-5d9d-529c-8fe1-9737fbe378bc::le_to_lt_or_eq_65dde5f8bf3f",
    "e13dc743-78dd-5c1a-ac22-35426901933d::le_to_lt_or_eq_92be32e19f55",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "6c1b199e-916f-547b-b634-dbeccad97dfd::lt_to_le_48fcfbe733ec",
    "732884f7-683e-5f81-8e2b-78d3b01a4c73::lt_to_le_f6e91c2ac4c3",
    "3501d02a-370c-5eac-b976-a858b3d9246b::lt_to_le_06a611842d81",
}

# Populated after the all-history exact/near-duplicate audit.  Keeping these
# explicit makes every demotion reviewable instead of hiding it in heuristics.
CROSS_SHARD_DEMOTIONS = set()
NEAR_DUPLICATE_DEMOTIONS = {
    "e6721a1c-302e-52c4-89ee-3e6fedb87ff0::eq_to_le_forward_f418e207b1d8",
    "fc17f14b-2027-563b-a21e-8f26f5463091::le_to_lt_or_eq_5acdc22afebf",
    "e22273d8-0bb7-5ea7-9ab9-8d12db25ced5::iff_forward_def7f745fbdd",
    "3946b6d8-951e-5462-a275-31ec204a2625::le_to_lt_or_eq_1f27f3050199",
    "1c64ada6-892b-5c24-9ae7-27bf9806b0a2::le_to_lt_or_eq_07a2d9d9d595",
    "2dcf20f8-5b0b-54cd-9b51-ca423f566b57::iff_forward_3bd31ad2ab85",
    "0625f81f-f822-5c11-a2f6-c530cb68b075::le_to_lt_or_eq_916c34c68ba1",
    "4d50fc84-9702-579d-9dc2-4ce2cd7cbf3f::le_to_lt_or_eq_f9faa816b7bf",
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
        return "reject", "extremely_low", "logical repackaging, symmetry, negated-bound restatement, commutation, or proof-only variation"
    if rid in CROSS_SHARD_DEMOTIONS:
        return "reject", "low", "content-preferred global sibling retained after parent-level comparison"
    if rid in NEAR_DUPLICATE_DEMOTIONS:
        return "reject", "low", "exact or near-exact formal theorem already retained in an earlier reviewed shard; removed to limit overfitting"
    if method.startswith("eq_to_le_"):
        category, chosen = equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject", "low", "closed computation, mod/digit fact, or explicit sequence/function value mechanically weakened"
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
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "scope, declaration, or problem-text pollution"
        basis = "sole natural nontrivial algebraic, geometric, combinatorial, recurrence, set-containment, or number-theoretic bound"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method in {"and_left", "and_right"}:
        if rid not in SELECTED_AND:
            return "reject", "extremely_low" if contaminated or reasons else "low", "not the sole substantive global sibling: incomplete projection, weaker endpoint, lost binder, declaration pollution, or trivial component"
        safe_binder = rid in SAFE_WHOLE_BINDER_AND
        if contaminated or (reasons and not safe_binder) or not problem_ok:
            return "reject", "extremely_low", "selected-looking component has unresolved scope/declaration/text pollution"
        basis = "sole retained substantive bound, divisibility law, complete quantified recurrence, inverse law, or Diophantine component with binders intact"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method == "iff_forward":
        unsafe_reason = reasons and rid not in SAFE_WHOLE_BINDER_IFF
        if contaminated or unsafe_reason or not problem_ok:
            return "reject", "extremely_low", "iff split escaped a forall/exists binder or has declaration/text pollution"
        basis = "complete forward solution, classification, divisibility, functional, or parameter necessity direction with all binders intact"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "iff_reverse":
        return "reject", "extremely_low" if contaminated or reasons else "low", "reverse sibling rejected; global forward completeness direction owns this upstream"
    if method in {"implication_contrapositive", "implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE:
            return "reject", "extremely_low" if contaminated or reasons else "low", "non-top-level arrow, escaped binder, tautological premise, or low-value premise-negation repackaging"
        if contaminated or reasons or not problem_ok or scale.top_arrow(str(packet["parent"]["goal"])) is None:
            return "reject", "extremely_low", "selected contrapositive has scope/declaration/text pollution"
        basis = "complete top-level contrapositive with substantive number-theoretic or range-classification content"
        return ("pass", "high", basis) if success else ("pending", "high", basis + "; exact proof needs repair")
    if method == "le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "bare scalar/function boundary split, Prop/conjunction cut, or scope pollution"
        basis = "substantive top-level sharp inequality refined into strict and equality cases"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    if method == "lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS:
            return "reject", "extremely_low", "strict relation was cut inside forall/exists scope"
        if contaminated or reasons or not problem_ok:
            return "reject", "extremely_low", "strict relation cut inside quantifier/Prop/declaration"
        basis = "natural reusable non-strict consequence of a substantive strict relation"
        return ("pass", "medium", basis) if success else ("pending", "medium", basis + "; exact proof needs repair")
    return "reject", "extremely_low", f"method {method} is outside calibrated acceptance scale"


if __name__ == "__main__":
    base.classify = classify
    base.main()
