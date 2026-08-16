#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00015."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
from typing import Any,Mapping

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("numinamath_b12_review_base",ROOT/"scripts/review_numinamath_s00_b00012.py");base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base);scale=base.scale
base.BATCH=ROOT/"outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00015";base.BATCH_ID=base.BATCH.name;base.MANIFEST=base.BATCH/"candidate_manifest.jsonl";base.RECEIPTS=base.BATCH/"verification_results.jsonl"
base.OUTPUT=ROOT/"outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00015_resolved.jsonl";base.REPORT=base.OUTPUT.with_name("review_s00_b00015_resolved_report.json");base.VERSION="numinamath_s00_b00015_strict_resolved_v1"

SELECTED_AND={
 "097935b2-2623-5546-aadb-71185f589caf::and_left_7651d23f1f28",
 "139f43bc-3dff-5ed4-b1f2-5c01abb880de::and_left_f4b0d90f8e42",
 "d7e97583-48a2-5ec7-b007-9790f4b8f269::and_left_5ec9b25399dc",
 "7e2e62f3-ef1c-53fe-838b-45598d89d516::and_right_eb00a2cff3c1",
 "4756acc3-bc15-5ab8-ac52-ee2125cb2df4::and_left_8ac7fb89f018",
 "42d00349-e58b-5cf0-837e-218520e6f098::and_left_a91daf29fe63",
 "55d7ff01-b6f3-5a55-b939-74e6799505bd::and_right_32e0b0064cef",
 "e19dfd1d-ee78-59e9-bc27-23cc842b35b6::and_left_84e37f8bd5dd",
 "c837798b-f7ef-5e42-944a-616842b273ea::and_right_ec52a63c1d5e",
 "ef42dd8d-fab1-5171-9a5d-314f6a287d40::and_left_5421a470ef9a",
 "16ded5a4-7cf9-5c7d-8ce0-35291ff8bf31::and_right_5a81c1ef8765",
 "ce1faa55-f86d-5511-a179-b415c5eb4c6f::and_left_56144bc1aa1c",
 "9f9f1318-84d0-50ab-87ad-7133c5f58356::and_right_cb7710b3d4a8",
}

TRIVIAL_EQUALITY_BOUNDS={
 "55fc9a84-f2e6-54f1-a116-fb2d555d3c9b::eq_to_le_forward_21f07cce95dd",
 "5835de75-0889-59f3-9cb3-61b80529c0d2::eq_to_le_forward_360e09cea813",
 "b98fa324-337e-5035-8159-32a56f64830c::eq_to_le_forward_c738d9ea03e1",
 "80edaa10-9402-590f-90c1-fccfcdbc58a8::eq_to_le_forward_bd4afd2a0663",
 "a92c54c5-8d8a-5c8f-9231-42dede1e0724::eq_to_le_forward_40b055806966",
 "e2233275-ad05-5357-87c5-616082d3a51b::eq_to_le_forward_af1c79db694a",
 "2693ade6-54af-5c17-97f3-fa3450cdc062::eq_to_le_forward_45709a8bcd53",
 "ee3fd8ca-33ce-5ffe-8921-fcebfb36dbdc::eq_to_le_forward_fda91a5f093b",
 "408ad71b-17d3-5eea-9f01-6e5656744f28::eq_to_le_forward_af3c0f1aab46",
 "eefa3e74-97dc-58eb-a512-7006d14e9bc3::eq_to_le_forward_cfdee5ffaba6",
 "30ecb559-69f8-56af-9e95-87b865054db1::eq_to_le_forward_38569d56c548",
 "e444dd74-a6f8-58cf-a557-b903e6b43505::eq_to_le_forward_9824def1095b",
}

INVALID_ORDER_REFINEMENTS={
 "bf06b780-03dd-516c-9b22-5c0642f76262::le_to_lt_or_eq_3f4f0be384ce",
 "1beb1acc-fd4b-5317-8bba-593dcba12676::le_to_lt_or_eq_05308e126a79",
 "fca34a49-de7c-58df-9b15-e5889a9ec730::le_to_lt_or_eq_0eaa7155cd82",
 "8fe64d70-7396-545b-9460-d6745cc34dae::le_to_lt_or_eq_c0ae658a4f1a",
 "0944f3e4-67ee-5a6e-8c42-f41a4ed5cf4b::le_to_lt_or_eq_e31ecee5cf36",
 "15caf027-f28a-5711-a322-e3343a27d34b::le_to_lt_or_eq_edf86e089b50",
 "a956fb0c-5d1d-5b58-ad01-4a027fa9b7e2::le_to_lt_or_eq_900069af62d9",
}

TRIVIAL_STRICT_WEAKENINGS={
 "6849a7af-f78c-5685-8905-f940bdd87123::lt_to_le_2ae52c7a6560",
 "a1b856ff-bdb5-5451-9844-06b46ee6ef45::lt_to_le_5957bd6331f6",
 "facce8ca-858f-5115-9f12-b8078b792180::lt_to_le_5f981534c9d2",
 "a055300a-7d55-55b3-9d63-f66eef7cf378::lt_to_le_ebbbdf2a86eb",
 "a15189b4-ff0c-5432-a493-61bc1564b36d::lt_to_le_a87d8771f2ec",
}

SAFE_WHOLE_BINDER_IFF={
 "67d31e65-ddc6-5fa0-8d5a-01fc488e6577::iff_forward_47b9224bc970",
 "e6da95f8-5344-51e0-9980-88be2c4f20b8::iff_forward_db8a4845edeb",
 "4318684b-52ce-5564-a835-cf2763791254::iff_forward_1d218eab9ed5",
}
CROSS_SHARD_DEMOTIONS={
 "597d8e93-4675-59bd-a421-62f07ce24ef3::and_right_8ec015d6aa45",
}
NEAR_DUPLICATE_DEMOTIONS={
 "0c557f58-8b15-5237-8840-09141cbcf755::le_to_lt_or_eq_9cea63978f98",
 "35b4aa5d-1585-5b26-be8a-c5c9704f48e5::le_to_lt_or_eq_3d8a4a76b607",
 "455df070-78a4-5c32-a039-bc42aea5c0c0::le_to_lt_or_eq_70934a4b1eab",
 "4670ae05-3cb1-542b-8379-36b51464d826::le_to_lt_or_eq_717ca9290084",
 "4e28c1c1-4ab0-5f9f-8885-abbeebed0da8::le_to_lt_or_eq_443760f5b115",
 "67fae42a-2c6e-56dc-b20c-07665fae7f8a::le_to_lt_or_eq_7488ab01980a",
 "6f221732-871b-567c-9a2c-fb728dd71ad9::le_to_lt_or_eq_82b37bc2e8af",
 "88bf7b13-1a4e-58c4-951d-f72556a2c8a9::eq_to_le_forward_d3d4a0474614",
 "a9c7401e-168b-56cb-a35c-4de0a7ab93d9::eq_to_le_forward_1788460ef7c8",
 "bfafda02-d40f-5dc8-80ee-5f611cf59ac8::le_to_lt_or_eq_40f4fea3f3e9",
 "c837798b-f7ef-5e42-944a-616842b273ea::and_right_ec52a63c1d5e",
 "ce995f29-7e16-574a-beb0-f7636f0679f1::iff_forward_7b27880c5649",
 "d5abe5de-a7f1-5a5d-a802-4137b936ec73::le_to_lt_or_eq_d1bcc7444bba",
 "3be50118-371a-5cc0-b7f7-2899b4d0597f::eq_to_le_forward_586f52157f41",
}

def classify(packet:Mapping[str,Any],raw:Mapping[str,Any],parent:Mapping[str,Any],success:bool,equality_choice:Mapping[str,tuple[str,str|None]])->tuple[str,str,str]:
 del parent
 rid=str(packet["record_id"]);method=str(packet["method"]);pid=str(packet["parent_id"]);contaminated=scale.contaminated(packet,raw);problem_ok=scale.problem_ok(method,raw);reasons=bool(packet.get("reason_codes"))
 if method in base.HARD_REJECT:return "reject","extremely_low","logical repackaging, symmetry, negated-bound restatement, commutation, or proof-only variation"
 if rid in CROSS_SHARD_DEMOTIONS:return "reject","low","content-preferred global sibling retained after parent-level comparison"
 if rid in NEAR_DUPLICATE_DEMOTIONS:return "reject","low","exact or near-exact formal theorem already retained in an earlier reviewed shard; removed to limit overfitting"
 if method.startswith("eq_to_le_"):
  category,chosen=equality_choice[pid]
  if rid in TRIVIAL_EQUALITY_BOUNDS:return "reject","low","closed computation, digit/mod fact, explicit function value, or elementary numerical result mechanically weakened"
  if chosen is None:
   labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
   return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
  if method!=chosen:return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","scope, declaration, or problem-text pollution"
  basis="sole natural nontrivial algebraic, geometric, combinatorial, recurrence, or number-theoretic bound"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method in {"and_left","and_right"}:
  if rid not in SELECTED_AND:return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: incomplete projection, weaker endpoint, lost binder, declaration pollution, or trivial component"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","selected-looking component has unresolved scope/declaration/text pollution"
  basis="sole retained substantive sharp bound, extremum, infinite-parity result, pseudoprime law, function decomposition, or general inequality component"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="iff_forward":
  unsafe_reason=reasons and rid not in SAFE_WHOLE_BINDER_IFF
  if contaminated or unsafe_reason or not problem_ok:return "reject","extremely_low","iff split escaped a forall/exists binder or has declaration/text pollution"
  basis="complete forward solution, classification, divisibility, functional, or parameter necessity direction with all binders intact"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="iff_reverse":return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward completeness direction owns this upstream"
 if method in {"implication_contrapositive","implication_as_disjunction"}:return "reject","extremely_low","non-top-level arrow, escaped binder, declaration pollution, or premise-negation repackaging"
 if method=="le_to_lt_or_eq":
  if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:return "reject","extremely_low","bare scalar/function boundary split, Prop/conjunction cut, or scope pollution"
  basis="substantive top-level sharp inequality refined into strict and equality cases"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="lt_to_le":
  if rid in TRIVIAL_STRICT_WEAKENINGS:return "reject","extremely_low","bare variable/function comparison, elementary arithmetic weakening, or escaped existential binder"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
  basis="natural reusable non-strict consequence of a substantive strict relation"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"

if __name__=="__main__":base.classify=classify;base.main()
