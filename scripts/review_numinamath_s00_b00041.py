#!/usr/bin/env python3
"""Frozen strict review for NuminaMath shard s00/b00041."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
from typing import Any,Mapping

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("numinamath_b40_review_base",ROOT/"scripts/review_numinamath_s00_b00040.py");previous=importlib.util.module_from_spec(spec);sys.modules[spec.name]=previous;spec.loader.exec_module(previous);base=previous.base;scale=previous.scale
base.BATCH=ROOT/"outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00041";base.BATCH_ID=base.BATCH.name;base.MANIFEST=base.BATCH/"candidate_manifest.jsonl";base.RECEIPTS=base.BATCH/"verification_results.jsonl"
base.OUTPUT=ROOT/"outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00041_resolved.jsonl";base.REPORT=base.OUTPUT.with_name("review_s00_b00041_resolved_report.json");base.VERSION="numinamath_s00_b00041_strict_frozen_v1"

SELECTED_AND={
 "d9420cc5-391d-5f69-b073-698a44c1e463::and_right_0661ee482237",
 "9deda840-2cd9-5984-bec7-82188326476f::and_right_7259d6495532",
 "7f88edfc-9edc-58f4-8ebb-31527c8282cf::and_right_554c7b7281ab",
 "0014622f-ffa5-5afa-923f-99ac05f38dde::and_right_d2db2e5d854a",
 "2dc9133a-b0eb-5d0a-b951-eba636a3b1a6::and_right_97dd0cc957bc",
 "0ffc4d9b-8b1a-53c4-aaf5-3d668d9643ea::and_right_0bab3ae74b21",
 "e2ea80b4-2317-546c-8eb9-9ff36d715918::and_right_bd20418ec898",
 "93543694-a650-5221-8d0e-01a40dbe19c4::and_left_e313a129aca5",
 "6b47657c-b076-5af8-aa7a-bad3c4a3557a::and_right_cb0dd9326e46",
 "ef70c079-5b60-5809-9ce3-d46e68c4f856::and_right_ef40680a3441",
 "5d667434-bdc4-5298-8739-77101e089024::and_right_0392c1e85650",
 "cb1cb6a9-fc87-5693-9bc0-3cdde444a81f::and_right_0f615d5ea231",
 "36fff0d5-0a10-5aa6-9aa6-f5b16be312a3::and_right_076c84557f1a",
}
ACCEPTED_EQUALITY_BOUNDS:set[str]=set()
SELECTED_CONTRAPOSITIVE={"2bfc8de4-72c0-540c-9dab-f0542730ea26::implication_contrapositive_1312de872672"}
INVALID_IFF_FORWARD={
 # Three instances of the already-retained page-number digit-count template.
 "a84618af-cb3c-5740-99db-874c2fa00fd1::iff_forward_d82f4d5deb29",
 "92e17503-ab3a-554d-96c3-cb54b78022d6::iff_forward_24be707a0707",
 "ceb417eb-26bf-50a6-8dac-6cedd6b5abd4::iff_forward_7aef222c150f",
}
INVALID_ORDER_REFINEMENTS={
 "91e015ce-4245-5b1b-adfa-5154f2cb39cc::le_to_lt_or_eq_f7c0e76e2471",
 "31d62f90-3820-5a8b-9428-5be72a07d11b::le_to_lt_or_eq_6b0ba31c0d76",
}
TRIVIAL_STRICT_WEAKENINGS={
 "6632e45b-91e9-5160-9bc5-c68254cc05e9::lt_to_le_1cd319de0e20",
 "fb88192a-1c60-5be0-aa10-f18c7cdd7f34::lt_to_le_098a5e6dac18",
}
CROSS_SHARD_DEMOTIONS:set[str]=set()
NEAR_DUPLICATE_DEMOTIONS={
 "29e6d6f7-ae12-5aa6-8195-57544e71c6a5::le_to_lt_or_eq_e574b65d36a2",
 "3c5a5097-7628-5cb3-9d93-c280093e3915::le_to_lt_or_eq_9b61cf2c7c6e",
 "b1db787a-d621-584a-a9fc-150a225da9a3::le_to_lt_or_eq_bfcc9aa4e0d1",
}

def classify(packet:Mapping[str,Any],raw:Mapping[str,Any],parent:Mapping[str,Any],success:bool,equality_choice:Mapping[str,tuple[str,str|None]])->tuple[str,str,str]:
 del parent
 rid=str(packet["record_id"]);method=str(packet["method"]);pid=str(packet["parent_id"]);contaminated=scale.contaminated(packet,raw);problem_ok=scale.problem_ok(method,raw);reasons=bool(packet.get("reason_codes"))
 if method in base.HARD_REJECT:return "reject","extremely_low","logical repackaging, commutation, symmetry, squaring, negated-bound restatement, or proof-only variation"
 if rid in CROSS_SHARD_DEMOTIONS:return "reject","low","content-preferred global sibling retained after parent-level comparison"
 if rid in NEAR_DUPLICATE_DEMOTIONS:return "reject","low","exact or numerical-template normalized theorem already retained in reviewed history"
 if method.startswith("eq_to_le_"):
  category,chosen=equality_choice[pid];labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
  if chosen is None:return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
  if method!=chosen:return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
  return "reject","low","exact scalar, sum, sequence, or function-value result mechanically weakened"
 if method in {"and_left","and_right"}:
  if rid not in SELECTED_AND:return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: scalar coordinate, source premise, weaker formula, escaped binder, or redundant endpoint"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","selected-looking component has unresolved scope/declaration/text pollution"
  basis="sole retained substantive sequence conclusion, case branch, divisibility fact, classification, or nonlinear sum after global sibling comparison"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="iff_forward":
  if rid in INVALID_IFF_FORWARD or contaminated or reasons or not problem_ok:return "reject","low" if rid in INVALID_IFF_FORWARD else "extremely_low","numeric template duplicate, escaped binder, or declaration/problem-text pollution"
  basis="complete forward solution, classification, impossibility, parametrization, or sharp parameter direction"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="iff_reverse":return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward solution/classification direction owns this upstream"
 if method in {"implication_contrapositive","implication_as_disjunction"}:
  if rid not in SELECTED_CONTRAPOSITIVE:return "reject","extremely_low" if contaminated or reasons else "low","escaped binder or low-value logical repackaging"
  if contaminated or reasons or not problem_ok or scale.top_arrow(str(packet["parent"]["goal"])) is None:return "reject","extremely_low","selected contrapositive has scope/declaration/text pollution"
  basis="complete top-level contrapositive of a substantive discrete extremum implication"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="le_to_lt_or_eq":
  if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:return "reject","extremely_low","quantifier/Prop cut or scope/declaration pollution"
  basis="substantive top-level sharp inequality refined into strict and equality cases"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="lt_to_le":
  if rid in TRIVIAL_STRICT_WEAKENINGS:return "reject","low","bare comparison or numerical-template weakening adds no reusable content"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
  basis="natural reusable non-strict consequence of a substantive strict sequence or product inequality"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"

if __name__=="__main__":base.classify=classify;base.main()
