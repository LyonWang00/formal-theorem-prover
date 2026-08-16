#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00021."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
from typing import Any,Mapping

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("numinamath_b20_review_base",ROOT/"scripts/review_numinamath_s00_b00020.py");previous=importlib.util.module_from_spec(spec);sys.modules[spec.name]=previous;spec.loader.exec_module(previous);base=previous.base;scale=previous.scale
base.BATCH=ROOT/"outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00021";base.BATCH_ID=base.BATCH.name;base.MANIFEST=base.BATCH/"candidate_manifest.jsonl";base.RECEIPTS=base.BATCH/"verification_results.jsonl"
base.OUTPUT=ROOT/"outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00021_resolved.jsonl";base.REPORT=base.OUTPUT.with_name("review_s00_b00021_resolved_report.json");base.VERSION="numinamath_s00_b00021_strict_resolved_v1"

SELECTED_AND={
 "1f302d55-5b28-51a5-bf0c-ff107758d4e6::and_right_99fe5da4ee52",
 "6bedb168-46d8-52bb-8d38-e27ee1bb1e08::and_right_2a91ee0b894e",
 "9e379627-ee8d-5a22-9065-e9c78b4fe093::and_left_d11f0af278c6",
 "a0e9d181-990f-598e-9af9-099c58ffefd3::and_right_70268782a493",
}

# Every equality mutation in this shard is an exact scalar/expression result.
# Even the polynomial-degree example is only a numerical-exponent variant of
# an already retained degree template, so none clears the medium-quality gate.
ACCEPTED_EQUALITY_BOUNDS:set[str]=set()

SELECTED_CONTRAPOSITIVE={
 "2d7cb5f6-60eb-567e-98cb-dc71e9b4a10d::implication_contrapositive_c2a3abd5cb42",
 "301428de-efa4-54cb-add0-fb64e710d701::implication_contrapositive_6dc486016ac3",
 "eb3151d0-7559-5f4b-b966-8ec4da5b4ebb::implication_contrapositive_4cb9dbf349cf",
}

INVALID_IFF_FORWARD={
 "0e845cf8-6314-5615-8e5b-2d23b2d8cbac::iff_forward_02036bdd0c9c",
 "365c4153-f6c9-59d0-bbdc-629144659638::iff_forward_fc5f8d18898b",
 "74214bb5-e2a4-5dc6-ba59-514b64649aea::iff_forward_bdced1b63753",
 "973269b0-2869-54f8-a41d-f1994017a988::iff_forward_35395bea7f2d",
}

INVALID_ORDER_REFINEMENTS={
 "2bf5b2bc-ff65-52d1-9771-7dd973109542::le_to_lt_or_eq_6b2daa1b946d",
 "2fabef67-6a84-59fe-84c7-69c6e81bf967::le_to_lt_or_eq_2e4126cbd66f",
 "5b86261d-643f-58d9-8466-d7e5720c4a80::le_to_lt_or_eq_f3ed672bebd7",
 "db3c9548-8640-54f2-9ab6-f173673781c0::le_to_lt_or_eq_761d45f6ffd9",
}

TRIVIAL_STRICT_WEAKENINGS={
 "73bedd3f-6b8b-5fa5-a582-c10ed41a049b::lt_to_le_67b1207758c9",
 "8b0e66ed-ca83-541b-bc5f-c2622ed216e9::lt_to_le_9f97c4a54444",
}

CROSS_SHARD_DEMOTIONS:set[str]=set()

NEAR_DUPLICATE_DEMOTIONS={
 # Exact/near-exact forward classifications already retained.
 "35694f4a-b2d4-5885-b69f-f797655fe00d::iff_forward_3bc5e4bd3df7",
 "7d7f2ce8-aa79-54ac-a40b-dccb478b4385::iff_forward_f32f23cecd3d",
 "7def8f32-5707-5d37-b0be-bee9d7538199::iff_forward_34d87e2e6b37",
 # Same percentage-recovery template with only the discount changed, and an
 # algebraically identical functional equation written on the other side.
 "179ee49b-0d4c-5514-991f-c65e2fbd8606::iff_forward_729c2119196c",
 "96434e5c-80b0-5fde-813d-2001fd00ba35::iff_forward_cfc63c391071",
 # Repeated endpoint/projection theorems from duplicated source problems.
 "644c2430-cfac-5317-a503-c2ab7c0f382a::and_right_bea50ae86c70",
 "d70ff355-d2b3-5ed6-b8ff-2e1a12200c6d::and_left_81bc4b0b72b5",
 "80186de1-9bd2-5dba-9c41-0415f97576f8::and_right_4cd69323c980",
 "af76b845-f389-5a6d-89ab-f873a7e84bbe::and_right_d6d873aeb894",
 "e4f8f3fc-2c96-5137-ad54-632e433cf3c2::and_left_6dbf718d1b73",
 "8e8d1d47-fc6c-51e7-a68a-7e48f17a155d::and_left_b8d953002714",
 "a6286e5b-d660-5a11-870f-b29a48add61c::and_left_dfb2de049ac8",
 "d8de34dd-4371-526b-ad32-18264d4f9eef::and_right_5213f7b5d035",
 "d8fd26d8-1179-5c55-8bbb-8b006f81c41d::and_left_e01de4e35711",
 # Exact normalized inequalities already retained in earlier shards.
 "4bbf6620-b7c8-5c3d-b330-9493bad0c45f::le_to_lt_or_eq_e42a67644496",
 "4dcae655-1902-53d5-8dea-93ca18979f1d::le_to_lt_or_eq_7af15b2773f1",
 "5839c76c-1430-5fc7-a61a-d88bd2771df9::le_to_lt_or_eq_ff5ec699f6bf",
 "81cc4974-47ad-5a62-ad5a-dae42491b9ef::le_to_lt_or_eq_644d3ff0f991",
 "bb501fdc-3d79-5819-a534-b3c54c5b95e9::le_to_lt_or_eq_23895165d69e",
 "bc9c8161-4982-5277-8e2a-29e0eb78ced5::le_to_lt_or_eq_1837957b1189",
 "dbf6d4d9-1307-56b3-8fab-5801bf223f0d::le_to_lt_or_eq_b7c355fcd0f1",
 "f035d226-d688-585d-ba35-6854426f7be5::le_to_lt_or_eq_3fea2fbbe97e",
 # Pure coefficient/numeric variant of the same quadratic-discriminant task.
 "8d0b064b-a847-5bce-9df7-65e32cddd76c::le_to_lt_or_eq_65d6cc08af88",
 # Same four-variable adjacent-power maximum template with only exponents and
 # numerical constants changed.
 "d23d11ce-4d9a-5a44-b086-427187710269::le_to_lt_or_eq_dbb6d919de53",
}

def classify(packet:Mapping[str,Any],raw:Mapping[str,Any],parent:Mapping[str,Any],success:bool,equality_choice:Mapping[str,tuple[str,str|None]])->tuple[str,str,str]:
 del parent
 rid=str(packet["record_id"]);method=str(packet["method"]);pid=str(packet["parent_id"]);contaminated=scale.contaminated(packet,raw);problem_ok=scale.problem_ok(method,raw);reasons=bool(packet.get("reason_codes"))
 if method in base.HARD_REJECT:return "reject","extremely_low","logical repackaging, commutation, symmetry, squaring, negated-bound restatement, or proof-only variation"
 if rid in CROSS_SHARD_DEMOTIONS:return "reject","low","content-preferred global sibling retained after parent-level comparison"
 if rid in NEAR_DUPLICATE_DEMOTIONS:return "reject","low","exact or semantically near-exact normalized theorem already retained in an earlier reviewed shard"
 if method.startswith("eq_to_le_"):
  category,chosen=equality_choice[pid]
  if chosen is None:
   labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
   return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
  if method!=chosen:return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
  if rid not in ACCEPTED_EQUALITY_BOUNDS:return "reject","low","exact scalar, sum, product, root, trigonometric, modular, or numerical-degree answer mechanically weakened; the degree case is a prior numeric template variant"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","scope, declaration, or problem-text pollution"
  basis="natural structural consequence of a substantive exact classification"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method in {"and_left","and_right"}:
  if rid not in SELECTED_AND:return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: scalar answer, incomplete multiple-choice projection, weaker endpoint, escaped binder, or prior near duplicate"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","selected-looking component has unresolved scope/declaration/text pollution"
  if rid.startswith("1f302d55-"):
   basis="independent sharpness witness complementing the previously retained universal lower-bound half of the exact minimum classification"
  else:basis="sole retained substantive threshold, sharp extremum, or structural linear-map coefficient component"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="iff_forward":
  if rid in INVALID_IFF_FORWARD or contaminated or reasons or not problem_ok:return "reject","extremely_low","iff split escaped a binder, leaked declaration syntax, or contains substantial unrelated problem-text pollution"
  basis="complete forward solution, classification, impossibility, parametrization, or sharp parameter direction"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="iff_reverse":return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward solution/classification direction owns this upstream"
 if method in {"implication_contrapositive","implication_as_disjunction"}:
  if rid not in SELECTED_CONTRAPOSITIVE:return "reject","extremely_low" if contaminated or reasons else "low","escaped binder, exact scalar/function-value relation, incomplete multiple-choice rewrite, or low-value logical repackaging"
  if contaminated or reasons or not problem_ok or scale.top_arrow(str(packet["parent"]["goal"])) is None:return "reject","extremely_low","selected contrapositive has scope/declaration/text pollution"
  basis="complete top-level contrapositive of a substantive range or number-theoretic implication"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="le_to_lt_or_eq":
  if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:return "reject","extremely_low","Prop/conjunction cut, escaped binder, or incomplete/mismatched problem statement"
  basis="substantive top-level sharp inequality refined into strict and equality cases"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="lt_to_le":
  if rid in TRIVIAL_STRICT_WEAKENINGS:return "reject","extremely_low" if reasons else "low","closed numeric comparison or Prop-valued scope cut adds no reusable inequality content"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
  basis="natural reusable non-strict consequence of a substantive strict inequality"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"

if __name__=="__main__":base.classify=classify;base.main()
