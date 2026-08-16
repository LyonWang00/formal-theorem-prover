#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00018."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
from typing import Any,Mapping

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("numinamath_b17_review_base",ROOT/"scripts/review_numinamath_s00_b00017.py");previous=importlib.util.module_from_spec(spec);sys.modules[spec.name]=previous;spec.loader.exec_module(previous);base=previous.base;scale=previous.scale
base.BATCH=ROOT/"outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00018";base.BATCH_ID=base.BATCH.name;base.MANIFEST=base.BATCH/"candidate_manifest.jsonl";base.RECEIPTS=base.BATCH/"verification_results.jsonl"
base.OUTPUT=ROOT/"outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00018_resolved.jsonl";base.REPORT=base.OUTPUT.with_name("review_s00_b00018_resolved_report.json");base.VERSION="numinamath_s00_b00018_strict_resolved_v1"

SELECTED_AND={
 "ca80ebfa-c4a7-50b0-bffe-6574a5119260::and_left_6b2addc0dd9e",
 "831277e3-c4d0-5dfb-a6bf-afe6f4e792cb::and_left_38d87ca99727",
 "c8687640-c93e-56ee-9a06-02e344967001::and_right_e875c0ad9107",
 "40fde53b-7902-5a65-bcb1-7a34208501d0::and_right_9ef02d223931",
 "ed57171e-ff3e-52cf-bad5-d4138c3ed03c::and_left_1dca829486d3",
 "2f8cf3fd-2e47-5e56-802f-a410ef004dcf::and_left_89a102429147",
 "e9df4dd9-f84c-5116-804f-0898be4f1a15::and_left_006c88b3ac03",
 "7b070c4e-282d-50b8-ab4a-381ebb66f87b::and_left_c2453e7b63b6",
 "9009d91d-c588-5a0e-a829-ba0c5f0391fc::and_right_9dc77c7dc930",
 "2baee8a5-a17d-5238-b064-ceb7d442bd9e::and_left_cfae6adcebf0",
 "2d2efb76-2bfa-5bc6-9f2c-6ca402abf519::and_left_b31604a3deb6",
 "01832505-e44f-50bf-bd85-7365135672f3::and_right_d7b37619de50",
 "c68513c4-47a5-5d84-805b-e3a5b87929ec::and_left_f1ba704f3eb3",
 "f04630aa-fae1-518f-9319-5a1d2573b6b4::and_right_15f2a8138051",
 "76bd896e-84ba-55fb-b491-f8bd80b52172::and_left_12c1fe229eb3",
 "c187d9f9-2c9d-5498-9c3f-9b4e2403deba::and_right_ebcdef235922",
}
ACCEPTED_EQUALITY_BOUNDS={
 "026966d8-4b31-5e20-9fe7-86e6839b6278::eq_to_le_forward_3bdb7bc14e4a",
 "2a26f66f-8fa9-5988-9818-9705e5c46773::eq_to_le_forward_519ee080c547",
 "40289473-a6c0-5582-9109-6d525b9deb4b::eq_to_le_forward_37ad1210d145",
 "8ee6606f-9c25-55d8-86c7-07b76c6644d5::eq_to_le_forward_67234d2d9c8e",
 "22d78ce5-e771-5a6b-8f9d-09b3699dc9c5::eq_to_le_forward_eba25bd39fb2",
 "fdf04176-7498-53e3-96e1-ef33cf2ea835::eq_to_le_forward_1aa4c2a0e0cd",
}
SELECTED_CONTRAPOSITIVE={"ca4e1175-7234-513c-8851-660cfecda55e::implication_contrapositive_e592aae60004"}
INVALID_ORDER_REFINEMENTS={
 "befbbaa3-600d-5dba-836c-08c0d8ecdd7d::le_to_lt_or_eq_9ce9949518af",
 "7bbaf566-fb2b-5db5-95e6-7d1a8e56a4be::le_to_lt_or_eq_70b4b9502dc5",
 "f578bbd9-a12a-5688-8f4b-e0ad80a756d8::le_to_lt_or_eq_e6057cacb416",
 "a6ffc742-89f6-52ad-bdf0-fd796cc84a4f::le_to_lt_or_eq_ac1390db6353",
 "72c4748b-06d8-5541-8ee4-29c4d8741a75::le_to_lt_or_eq_f355fc755904",
}
TRIVIAL_STRICT_WEAKENINGS={
 "7196dee6-6e81-51fa-bb67-bbac198d5717::lt_to_le_b1c8b848cf90",
 "852c3ed0-fa5f-5c1b-8268-8f72e1fcb49b::lt_to_le_22a4ebfd114f",
 "a1e5dcfc-771f-5dda-919e-0de27d8ab552::lt_to_le_c8caa8d9a108",
}
CROSS_SHARD_DEMOTIONS={
 # The paired upper endpoint ab <= 4/9 is the more informative retained range component.
 "2d2efb76-2bfa-5bc6-9f2c-6ca402abf519::and_left_b31604a3deb6",
 # The sharp minimum -1/4 is less routine than the paired maximum 2 for this feasible set.
 "01832505-e44f-50bf-bd85-7365135672f3::and_right_d7b37619de50",
 # Even x alone does not identify the multiple-choice answer; the paired Odd y projection is
 # likewise incomplete and is rejected by the cross-shard override rather than retained.
 "c68513c4-47a5-5d84-805b-e3a5b87929ec::and_left_f1ba704f3eb3",
 # Retain the sharper-looking lower endpoint 1 < S as the sole component of this two-sided bound.
 "f04630aa-fae1-518f-9319-5a1d2573b6b4::and_right_15f2a8138051",
}
NEAR_DUPLICATE_DEMOTIONS={
 # Exact normalized theorem already retained: same four-variable cyclic fraction inequality.
 "d94e389f-20c9-5b75-95d4-f187298d11ec::le_to_lt_or_eq_16bdce482a29",
 # Exact normalized theorem already retained: same a,b > 1 rational lower bound.
 "592c9a76-e807-5dce-91d9-4a895d705b97::le_to_lt_or_eq_e19191603829",
 # Exact normalized theorem already retained several times: same IMO 0 <= expression <= 7/27 component.
 "40fde53b-7902-5a65-bcb1-7a34208501d0::and_right_9ef02d223931",
 # Semantically equivalent to the retained positive-variable theorem: abcd = 1 makes the
 # nominally nonnegative variables strictly positive, so the changed hypotheses add no case.
 "da0f3e60-758b-5e25-8943-d8cd7d08912f::le_to_lt_or_eq_bae9696e1c42",
}

def classify(packet:Mapping[str,Any],raw:Mapping[str,Any],parent:Mapping[str,Any],success:bool,equality_choice:Mapping[str,tuple[str,str|None]])->tuple[str,str,str]:
 del parent
 rid=str(packet["record_id"]);method=str(packet["method"]);pid=str(packet["parent_id"]);contaminated=scale.contaminated(packet,raw);problem_ok=scale.problem_ok(method,raw);reasons=bool(packet.get("reason_codes"))
 if method in base.HARD_REJECT:return "reject","extremely_low","logical repackaging, commutation, symmetry, same-operation equality mutation, negated-bound restatement, or proof-only variation"
 if rid in CROSS_SHARD_DEMOTIONS:return "reject","low","content-preferred global sibling retained after parent-level comparison"
 if rid in NEAR_DUPLICATE_DEMOTIONS:return "reject","low","exact or near-exact normalized theorem already retained in an earlier reviewed shard"
 if method.startswith("eq_to_le_"):
  category,chosen=equality_choice[pid]
  if chosen is None:
   labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
   return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
  if method!=chosen:return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
  if rid not in ACCEPTED_EQUALITY_BOUNDS:return "reject","low","exact numeric answer, digit/mod value, sequence value, closed trigonometric evaluation, or routine scalar expression mechanically weakened"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","scope, declaration, or problem-text pollution"
  basis="natural structural root-count/set-containment consequence or reusable nontrivial algebraic/norm bound"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method in {"and_left","and_right"}:
  if rid not in SELECTED_AND:return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: incomplete/trivial projection, weaker endpoint, exact scalar answer, escaped binder, or pollution"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","selected-looking component has unresolved scope/declaration/text pollution"
  basis="sole retained substantive extremum, sharp bound, monotonicity, group identity, number-theoretic law, parity, or pigeonhole component"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="iff_forward":
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","iff split escaped an exists/forall binder or contains declaration/text pollution"
  basis="complete forward solution, classification, impossibility, parametrization, or sharp parameter direction"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="iff_reverse":return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward completeness direction owns this upstream"
 if method in {"implication_contrapositive","implication_as_disjunction"}:
  if rid not in SELECTED_CONTRAPOSITIVE:return "reject","extremely_low" if contaminated or reasons else "low","escaped binder, vacuous premise, or low-value logical repackaging"
  if contaminated or reasons or not problem_ok or scale.top_arrow(str(packet["parent"]["goal"])) is None:return "reject","extremely_low","selected contrapositive has scope/declaration/text pollution"
  basis="complete top-level contrapositive of a substantive coordinate-distance inequality"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="le_to_lt_or_eq":
  if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:return "reject","extremely_low","bare numeric boundary, Prop/conjunction cut, or scope/declaration pollution"
  basis="substantive top-level sharp inequality refined into strict and equality cases"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="lt_to_le":
  if rid in TRIVIAL_STRICT_WEAKENINGS:return "reject","extremely_low" if reasons else "low","scope/Prop cut or bare sequence/function-value weakening"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
  basis="natural reusable non-strict consequence of a substantive strict inequality"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"

if __name__=="__main__":base.classify=classify;base.main()
