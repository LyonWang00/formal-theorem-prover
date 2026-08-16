#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00014."""

from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from typing import Any, Mapping

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("numinamath_b12_review_base",ROOT/"scripts/review_numinamath_s00_b00012.py")
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
scale=base.scale
base.BATCH=ROOT/"outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00014"
base.BATCH_ID=base.BATCH.name;base.MANIFEST=base.BATCH/"candidate_manifest.jsonl";base.RECEIPTS=base.BATCH/"verification_results.jsonl"
base.OUTPUT=ROOT/"outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00014_resolved.jsonl"
base.REPORT=base.OUTPUT.with_name("review_s00_b00014_resolved_report.json")
base.VERSION="numinamath_s00_b00014_strict_resolved_v1"

SELECTED_AND={
 "b9bf5f76-1e40-52d1-b589-295f326c4805::and_right_757ee19e2f62",
 "138c5268-64a2-503a-963e-cdb6195bab44::and_right_1c9d0edfa94b",
 "8996846a-7351-57ff-9ec3-e048affea985::and_right_cf20b24aa834",
 "7938505f-4bbc-5ee3-bf1e-358b4113fde2::and_left_2e7408fc8a83",
 "bfc5fa78-42c8-5485-b207-1bf2b7b5f2e0::and_right_f3f7fc13b681",
 "104222a9-c19c-511c-9a53-7e31921761ae::and_left_1f1b4ef7eb04",
 "aee70739-9067-5990-9e12-935d08870ed0::and_left_7cae07349b35",
 "d11bb440-ddd3-5192-9f6c-96ec02fdd47d::and_right_77a2520baa28",
 "b8cb765c-5b95-5570-a013-1856e4ce88dd::and_left_da6acc28570e",
}

TRIVIAL_EQUALITY_BOUNDS={
 "ad5a9844-430a-58ab-8cfd-6e17e76c3de2::eq_to_le_forward_f81ea988fdcc",
 "a3cf361b-684b-57c7-b6f5-ddf909eb2dbd::eq_to_le_forward_f3ea5a88d604",
 "089d418a-ae25-574c-bba2-763a7bdcf2e7::eq_to_le_forward_73d6689ff74a",
 "87c97048-4750-508c-b495-1cb0088e010a::eq_to_le_forward_4658fd920e54",
 "2b76d69a-f7a8-552c-96e9-fe3a299c7b2a::eq_to_le_forward_571c9275583e",
 "bcfb636a-729d-5aa4-9621-ab381550a076::eq_to_le_forward_c6b3d5f3e947",
 "849f9ddc-7126-50c1-a236-e4aef3e9adaa::eq_to_le_forward_4aeecb23919b",
}

SELECTED_CONTRAPOSITIVE={
 "e5e79001-ae63-56c1-97cc-a767f855d3e7::implication_contrapositive_41dfdda19a1e",
}

INVALID_ORDER_REFINEMENTS={
 "079275e5-2961-5026-a665-c3bbd7fb4944::le_to_lt_or_eq_db991cb3069f",
 "ccf980f3-1330-5692-aa50-f40a224b1868::le_to_lt_or_eq_285f14121302",
 "eb91b060-db75-5b9d-85c7-c806d4fc7563::le_to_lt_or_eq_569623ccf6be",
 "b4a5a573-a57f-5cb0-84fe-16149e734b08::le_to_lt_or_eq_bb188ba88a31",
 "2af607ae-0cd8-5426-a8f8-569db77e1903::le_to_lt_or_eq_a3c13b60db41",
}

TRIVIAL_STRICT_WEAKENINGS={
 "2b8a4651-04a9-5beb-a87d-ab5dca879a72::lt_to_le_e0c8134e5a8c",
}

SAFE_WHOLE_BINDER_IFF={
 "8c7f3359-6107-5fff-8393-0343bd76f70b::iff_forward_f38d921e4072",
 "32f2f8a2-3905-5fb9-8a72-0af4134ad3d6::iff_forward_74c3a85746c5",
}

CROSS_SHARD_DEMOTIONS={
 "f993906f-a41f-591f-92eb-c32389c74158::and_right_01869ef41a77",
 "284442f8-5dee-5065-9743-673d79f3d3bb::and_left_b1c2cd96409d",
 "b80ac73c-c12f-51e1-8056-a54ca7bac420::eq_to_le_forward_931b8d7b8871",
}


def classify(packet:Mapping[str,Any],raw:Mapping[str,Any],parent:Mapping[str,Any],success:bool,
             equality_choice:Mapping[str,tuple[str,str|None]])->tuple[str,str,str]:
 del parent
 rid=str(packet["record_id"]);method=str(packet["method"]);pid=str(packet["parent_id"]);goal=str(packet["parent"]["goal"])
 contaminated=scale.contaminated(packet,raw);problem_ok=scale.problem_ok(method,raw);reasons=bool(packet.get("reason_codes"))
 if method in base.HARD_REJECT:return "reject","extremely_low","logical repackaging, symmetry, negated-bound restatement, commutation, or proof-only variation"
 if rid in CROSS_SHARD_DEMOTIONS:return "reject","low","content-preferred global sibling retained: stronger completeness/minimum law or more general exponential identity"
 if method.startswith("eq_to_le_"):
  category,chosen=equality_choice[pid]
  if rid in TRIVIAL_EQUALITY_BOUNDS:return "reject","low","closed computation, digit/mod result, explicit function evaluation, or elementary arithmetic value mechanically weakened"
  if chosen is None:
   labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
   return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
  if method!=chosen:return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","scope, declaration, or problem-text pollution"
  basis="sole natural nontrivial algebraic, geometric, combinatorial, probability, set-containment, or number-theoretic bound"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method in {"and_left","and_right"}:
  if rid not in SELECTED_AND:return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: incomplete scalar projection, weaker endpoint, trivial positivity, lost binder, or pollution"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","selected-looking component has unresolved scope or text pollution"
  basis="sole retained substantive sharp extremum, range endpoint, recurrence/function law, arithmetic classification, or nontrivial inequality component"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="iff_forward":
  unsafe_reason=reasons and rid not in SAFE_WHOLE_BINDER_IFF
  if contaminated or unsafe_reason or not problem_ok:return "reject","extremely_low","iff split escaped a forall/exists binder or has declaration/text pollution"
  basis="complete forward solution, classification, functional characterization, or parameter necessity direction with all binders intact"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="iff_reverse":return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward direction owns this upstream"
 if method in {"implication_contrapositive","implication_as_disjunction"}:
  if rid not in SELECTED_CONTRAPOSITIVE:return "reject","extremely_low","non-top-level arrow, escaped binder, or premise-negation repackaging"
  if contaminated or reasons or not problem_ok or scale.top_arrow(goal) is None:return "reject","extremely_low","contrapositive has scope or declaration pollution"
  basis="complete top-level contrapositive preserving all quadratic-residue witnesses"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="le_to_lt_or_eq":
  if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:return "reject","extremely_low","bare scalar boundary split, Prop/conjunction cut, or scope pollution"
  basis="substantive top-level sharp inequality refined into strict and equality cases"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="lt_to_le":
  if rid in TRIVIAL_STRICT_WEAKENINGS:return "reject","extremely_low","strict relation was cut inside a universal quantifier"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
  basis="natural reusable non-strict consequence of a substantive strict relation"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"


if __name__=="__main__":base.classify=classify;base.main()
