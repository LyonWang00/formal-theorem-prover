#!/usr/bin/env python3
"""Frozen strict review for NuminaMath shard s00/b00040."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
from typing import Any,Mapping

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("numinamath_b31_review_base",ROOT/"scripts/review_numinamath_s00_b00031.py");previous=importlib.util.module_from_spec(spec);sys.modules[spec.name]=previous;spec.loader.exec_module(previous);base=previous.base;scale=previous.scale
base.BATCH=ROOT/"outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00040";base.BATCH_ID=base.BATCH.name;base.MANIFEST=base.BATCH/"candidate_manifest.jsonl";base.RECEIPTS=base.BATCH/"verification_results.jsonl"
base.OUTPUT=ROOT/"outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00040_resolved.jsonl";base.REPORT=base.OUTPUT.with_name("review_s00_b00040_resolved_report.json");base.VERSION="numinamath_s00_b00040_strict_frozen_v1"

# Sole useful projections after comparing the complete parent and every global
# sibling: non-square/favourable counts, the natural divisibility direction,
# and one nontrivial endpoint from each genuine extremum classification.
SELECTED_AND={
 "c9e6b573-bfcd-5a01-9dae-ad487c079aa1::and_right_a1e4d9321fb4",
 "e13a4602-1d04-5ace-92d1-d5443aee1f41::and_left_c74315cc76c0",
 "b08e81f1-3afe-59f9-8505-f683f5a41a9c::and_right_ff40b3996203",
 "e12fa8b2-147f-55e3-be59-fb391afc8a82::and_right_9e409152c9cb",
 "9ad9dafb-eca6-5fc8-ac58-e59d668d55f9::and_right_17a2ace87ed1",
 "26faeff6-6ae3-50de-9840-0f7514196ba9::and_left_1e5f2fd9497c",
}
ACCEPTED_EQUALITY_BOUNDS:set[str]=set()
SELECTED_CONTRAPOSITIVE:set[str]=set()
INVALID_IFF_FORWARD={
 # One-step closed arithmetic and multiple-choice calculations.
 "e378fb79-7303-5858-a5da-c2fd7f0b8fa8::iff_forward_fd9b43eb1742",
 "42504776-a346-50b2-be0f-49d94bff7d61::iff_forward_f920ca9a0514",
 # The source text still contains a translation instruction.
 "0800bdeb-5a0d-5ac8-b658-78c70de805ad::iff_forward_64816477f76a",
}
INVALID_ORDER_REFINEMENTS={
 "9315d089-3e27-53d7-b57d-96b9f68ee8a7::le_to_lt_or_eq_1d0a6cc19e76",
 "a3db11c7-c56a-5a0e-9a38-01ba08ca749b::le_to_lt_or_eq_7345f77d4601",
 "4686fe24-769d-5714-a3a0-386bce496954::le_to_lt_or_eq_6f8fd81526ca",
 "7901fb0e-217c-532a-a3f1-d332ac112547::le_to_lt_or_eq_89afee249312",
}
TRIVIAL_STRICT_WEAKENINGS={
 "9e7f9cd6-5a45-570e-8a23-329cc62ff01f::lt_to_le_6846a57574c1",
 "f38f6514-f7a9-5d50-a9aa-0babf5f4e315::lt_to_le_4be20b39d38e",
 "e2974836-aaf0-5f3e-9f57-d4f02fd6aea8::lt_to_le_8ecb2825a3f7",
}
CROSS_SHARD_DEMOTIONS:set[str]=set()
NEAR_DUPLICATE_DEMOTIONS={
 # Same sqrt(x-2)+sqrt(x-7)=5 theorem occurs twice in this batch; the
 # cleaner e437... source is retained.
 "93f31035-addc-5193-ba96-1a7c972ef730::iff_forward_035024308ee7",
 # Same universal sqrt inequality as ce165..., but with a redundant ha>0.
 "76d97e7b-3134-58f6-9f01-86239d4f07e7::iff_forward_7000187f8e13",
}

def classify(packet:Mapping[str,Any],raw:Mapping[str,Any],parent:Mapping[str,Any],success:bool,equality_choice:Mapping[str,tuple[str,str|None]])->tuple[str,str,str]:
 del parent
 rid=str(packet["record_id"]);method=str(packet["method"]);pid=str(packet["parent_id"]);contaminated=scale.contaminated(packet,raw);problem_ok=scale.problem_ok(method,raw);reasons=bool(packet.get("reason_codes"))
 if method in base.HARD_REJECT:return "reject","extremely_low","logical repackaging, commutation, symmetry, squaring, negated-bound restatement, or proof-only variation"
 if rid in CROSS_SHARD_DEMOTIONS:return "reject","low","content-preferred global sibling retained after parent-level comparison"
 if rid in NEAR_DUPLICATE_DEMOTIONS:return "reject","low","exact or semantically near-exact normalized theorem already represented by the retained cleaner item"
 if method.startswith("eq_to_le_"):
  category,chosen=equality_choice[pid]
  labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
  if chosen is None:return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
  if method!=chosen:return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
  return "reject","low","exact scalar, count, sum, sequence, or function-value result mechanically weakened"
 if method in {"and_left","and_right"}:
  if rid not in SELECTED_AND:return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: scalar coordinate, premise restatement, escaped binder, weaker endpoint, or symmetric total"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","selected-looking component has unresolved scope/declaration/text pollution"
  basis="sole retained substantive count, divisibility direction, or extremum endpoint after global sibling comparison"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="iff_forward":
  if rid in INVALID_IFF_FORWARD or contaminated or reasons or not problem_ok:return "reject","extremely_low","trivial closed arithmetic, escaped binder, or declaration/problem-text pollution"
  basis="complete forward solution, classification, impossibility, parametrization, or sharp parameter direction"
  return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
 if method=="iff_reverse":return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward solution/classification direction owns this upstream"
 if method in {"implication_contrapositive","implication_as_disjunction"}:return "reject","extremely_low" if contaminated or reasons else "low","no clean substantive top-level implication survives this shard"
 if method=="le_to_lt_or_eq":
  if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok:return "reject","extremely_low","quantifier/Prop cut or scope/declaration pollution"
  basis="substantive top-level sharp inequality refined into strict and equality cases"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 if method=="lt_to_le":
  if rid in TRIVIAL_STRICT_WEAKENINGS:return "reject","low","closed numeric, Prop-valued, or numerical-template weakening adds no reusable content"
  if contaminated or reasons or not problem_ok:return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
  basis="natural reusable non-strict consequence of a substantive strict inequality or count comparison"
  return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
 return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"

if __name__=="__main__":base.classify=classify;base.main()
