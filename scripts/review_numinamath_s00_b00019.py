#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00019."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
from typing import Any,Mapping

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("numinamath_b18_review_base",ROOT/"scripts/review_numinamath_s00_b00018.py");previous=importlib.util.module_from_spec(spec);sys.modules[spec.name]=previous;spec.loader.exec_module(previous);base=previous.base;scale=previous.scale
base.BATCH=ROOT/"outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00019";base.BATCH_ID=base.BATCH.name;base.MANIFEST=base.BATCH/"candidate_manifest.jsonl";base.RECEIPTS=base.BATCH/"verification_results.jsonl"
base.OUTPUT=ROOT/"outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00019_resolved.jsonl";base.REPORT=base.OUTPUT.with_name("review_s00_b00019_resolved_report.json");base.VERSION="numinamath_s00_b00019_strict_resolved_v1"

SELECTED_AND={
 "f9b514b2-b126-50d1-99a5-07182bc81bd2::and_right_777f55933724",
 "f503c5ed-67cf-5ab1-a680-09b21d679317::and_left_a91b5babc27c",
 "b2c6b13f-0015-5404-af42-94bf656337b1::and_right_5b24c670b5f3",
 "01d387f8-acf6-5753-903e-9b9fe728e9d8::and_right_5c23253864b4",
 "5e6cdba2-8bab-59bc-98eb-7a796f7a49b4::and_right_7ebe77e4e0b3",
 "c155b258-4c17-5136-b8eb-2f02cf4b2ed5::and_left_6b3d8548457f",
 "f8803acc-2b55-5b13-9850-089aa714fc3f::and_right_7cf20177fb88",
 "923034a6-a1ca-588d-87d4-c87e2cb1ebc3::and_right_cd61455a3183",
 "32e56e4e-f5f1-5982-bf89-a0233a9c1028::and_right_d4d65c7abf37",
 "ce3b83b6-5f22-599c-be85-f11352f42a03::and_right_dbb66374f261",
}
ACCEPTED_EQUALITY_BOUNDS={
 "f0eb56f6-90da-5a4d-9569-592e5a14168a::eq_to_le_forward_29677d8d6076",
 "8d8a3623-892e-52c0-afc7-30e5f9a626e9::eq_to_le_forward_870054321262",
}
SAFE_SCOPED_IFF={
 "5c21a6d7-488a-5f8d-9d05-d8e42b810c75::iff_forward_8ae3be0d425c",
 "98041b00-2df0-5194-8069-5e977a5d2835::iff_forward_66a06e2d2e35",
}
INVALID_ORDER_REFINEMENTS={
 "17be599e-4578-5b36-bd10-5aed53957cb6::le_to_lt_or_eq_2c2db9699ba9",
 "bce9aaa8-5175-560a-8e1d-aff2e896ac60::le_to_lt_or_eq_04f453dabdc9",
 "703a4842-2c7a-51d1-9f94-c568abfd46a9::le_to_lt_or_eq_b4a2836b2c92",
 "d0a6a220-4547-5ac6-b226-8f06732b1ad9::le_to_lt_or_eq_2a2e8c799eb8",
 "2d64777e-05d3-5e41-91b6-f20e806211b4::le_to_lt_or_eq_61d08507a910",
 "8b152467-4dc3-5952-993f-dc3f0cc2b1e0::le_to_lt_or_eq_b53f5c00a3d5",
 "9af6f410-92b3-591b-9cfa-766784995f68::le_to_lt_or_eq_e07b6973b4e2",
 "747c932c-f066-5549-9d45-1ef483cfd7a2::le_to_lt_or_eq_1237d4daa4ab",
 "3409e024-8db7-542e-9b14-49c51369fcc9::le_to_lt_or_eq_230c887837ec",
 "e86b1d89-640b-5c24-a468-34582bb32bad::le_to_lt_or_eq_149db2209199",
 "914cf994-69dc-5612-b0cb-9a21e31db797::le_to_lt_or_eq_abb0ef5cbb8b",
}
TRIVIAL_STRICT_WEAKENINGS={
 "9c47b07c-e490-5bc0-baa3-d582fa65076f::lt_to_le_56954ab053b2",
 "d07ec627-1d77-5150-8dae-79fbd9fcf746::lt_to_le_ffb31df31bc8",
}
CROSS_SHARD_DEMOTIONS={
 # Retain the more relational lower estimate 3/(a^3+b^3+c^3) <= S from s01.
 "ce3b83b6-5f22-599c-be85-f11352f42a03::and_right_dbb66374f261",
}
NEAR_DUPLICATE_DEMOTIONS={
 "9b6cc1e4-55d8-5837-93eb-dbbb1ee92750::le_to_lt_or_eq_4e841314c65e",
 "edc911c3-31ef-571c-b2f2-57ebe817144f::iff_forward_1999f0274d25",
 "5a0702fd-95f9-584a-a9ce-8cf98db19d1e::le_to_lt_or_eq_139ecbcb4e0e",
 "3cf8a51c-58f1-5e1d-8271-be45121285ff::le_to_lt_or_eq_0dc98e9ef295",
 "f503c5ed-67cf-5ab1-a680-09b21d679317::and_left_a91b5babc27c",
 "11aaa7dc-e393-5d09-96ef-60b1ec3c403a::lt_to_le_c8f2a75a6879",
 "80e9e053-37b3-597f-9de9-fcd5bdf50309::le_to_lt_or_eq_c656c35d5c27",
 "66d2e237-2cf0-5ce5-b88d-73d4109efac6::lt_to_le_e324f6929772",
 "5894fb9c-da85-53b2-8296-67c734ef8726::le_to_lt_or_eq_f9c38ef98548",
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
  if rid not in ACCEPTED_EQUALITY_BOUNDS:return "reject","low","exact scalar answer, trigonometric evaluation, digit/mod value, sequence value, or routine expression mechanically weakened"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","scope, declaration, or problem-text pollution"
  basis="natural structural image/range containment consequence retaining the soundness half of a substantive set equality"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method in {"and_left","and_right"}:
  if rid not in SELECTED_AND:return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: exact scalar answer, incomplete multiple-choice projection, trivial gcd/order fact, escaped binder, or weaker endpoint"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","selected-looking component has unresolved scope/declaration/text pollution"
  basis="sole retained substantive sharp extremum/bound, sufficiency direction, recurrence inheritance, or number-theoretic representation component"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="iff_forward":
  if contaminated or not problem_ok or (reasons and rid not in SAFE_SCOPED_IFF):return "reject","extremely_low","iff split has escaped binder, free declaration suffix, or problem-text pollution"
  basis="complete forward solution, classification, necessity, parametrization, range realization, or structural periodicity direction"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="iff_reverse":return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward solution/classification direction owns this upstream"
 if method in {"implication_contrapositive","implication_as_disjunction"}:return "reject","extremely_low" if contaminated or reasons else "low","non-top-level arrow, escaped binder/free variable, or low-value logical repackaging"
 if method=="le_to_lt_or_eq":
  if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:return "reject","extremely_low","bare scalar/function boundary, non-sharp coarse estimate, vacuous formalization, Prop/conjunction cut, or scope/declaration pollution"
  basis="substantive top-level sharp inequality refined into strict and equality cases"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="lt_to_le":
  if rid in TRIVIAL_STRICT_WEAKENINGS:return "reject","extremely_low" if reasons else "low","scope cut or bare sequence/function-value weakening"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
  basis="natural reusable non-strict consequence of a substantive strict inequality"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"

if __name__=="__main__":base.classify=classify;base.main()
