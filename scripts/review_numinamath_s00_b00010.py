#!/usr/bin/env python3
"""Strict raw-upstream review for NuminaMath shard s00/b00010."""

from __future__ import annotations
import hashlib, importlib.util, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00010"
MANIFEST = BATCH / "candidate_manifest.jsonl"; RECEIPTS = BATCH / "verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00010.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00010_report.json")
VERSION = "numinamath_s00_b00010_strict_upstream_review_v1"

spec = importlib.util.spec_from_file_location("numinamath_b9_helpers", ROOT / "scripts/review_numinamath_s00_b00009.py")
helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
scale = helper.scale; b5 = helper.b5

SELECTED_AND = {
    "59148f26-5380-516f-b572-c52d2795bf9c::and_right_1d50c53df3f1",
    "30dd75b5-251b-5a62-9c39-458c2007e9d9::and_left_b76abbd8a900",
    "08b43931-79eb-53b1-9522-d8565b8d0fc9::and_right_603bfad08ef5",
    "38360797-f88f-5e30-befe-d88505023b4f::and_left_972b4a4b3ff3",
    "7b40d470-ff4a-57e8-93c5-7ca160efb2dd::and_right_698d67c750f9",
    "4780648d-5a10-5763-ad98-5aaf9ceddb6a::and_right_013807705129",
    "fa01a2ad-fe83-5af1-8c85-2f58eca8f2c3::and_right_b0e7ebeefb5e",
    "727e56d8-ea5f-59b3-8c0e-d69f2578348e::and_right_d9a5e5deb7aa",
    "59421d01-9f20-528a-9963-80441182ef74::and_right_8e28ffc6549b",
    "b17091da-e433-551a-b92c-5d6092c3367b::and_left_a7123aeb3f22",
}
TRIVIAL_EQUALITY_BOUNDS = {
    "71a77c93-49d2-5d3b-a4f0-c76b54b8f979::eq_to_le_forward_e0e99d30ccfd",
    "0d848332-2a82-5b36-ad5b-d6faa6a1e73b::eq_to_le_forward_9c8bc51f52a6",
    "52961566-698c-5e1d-8e6d-9eb99f66f809::eq_to_le_forward_76686c89a1d6",
    "26080125-7069-5652-a28a-fcec05d04ced::eq_to_le_forward_99b7984cd466",
    "8f5a1b8a-6390-5384-a4c6-3e81bd924dfd::eq_to_le_forward_8626d3002085",
    "af9eb625-2d2f-503f-a017-ead87ade6ab6::eq_to_le_forward_80ce085c7e91",
    "59914f9b-bf52-5a38-991c-722cd1ba0545::eq_to_le_forward_b508f9f31101",
    "e70f066f-deca-5c52-ada8-1abaeba0e18a::eq_to_le_forward_e9af7525ad0a",
    "9ed26723-da6f-5a6a-8fe2-4107b3bb4b5b::eq_to_le_forward_7228f54c7d59",
    "0de864f3-ead7-56e3-86a5-b99f93850c91::eq_to_le_forward_042104ac0b21",
    "94934947-8562-506c-a76f-320b4b63c4ad::eq_to_le_forward_d4453793e132",
    "39b708d0-4dce-5313-9110-898ef66c115e::eq_to_le_forward_32f74932ff39",
    "2d940420-444e-5544-8f85-e691f868dac6::eq_to_le_forward_2461ca793616",
    "50d6548d-c178-5768-88d3-3e6d283952f4::eq_to_le_forward_5b0fa617ba14",
    "c878fe77-6056-53cb-a1e3-58bf94b869a9::eq_to_le_forward_d0f8b389ffeb",
}
SELECTED_CONTRAPOSITIVE = {
    "476f4401-aa4d-5f29-be72-e9e079f56adc::implication_contrapositive_71977dde3263",
    "df2c677b-cb76-598f-abe2-d5856be0aef4::implication_contrapositive_49c26999c09e",
}
INVALID_ORDER_REFINEMENTS = {
    "33763b6d-d27d-5910-8837-7698e50bd3e0::le_to_lt_or_eq_32d9ec22be8d",
    "0957667a-f413-5f70-acdc-732b12c8db7f::le_to_lt_or_eq_cb2e8109ab3e",
    "28eb97ed-16f1-5816-ad20-753c2f4e0dd6::le_to_lt_or_eq_333214cb3789",
}
TRIVIAL_STRICT_WEAKENINGS = {
    "c875a954-c405-51af-af1b-56fea5eede6a::lt_to_le_b3649a74275c",
    "453bd63d-fe4c-5b67-aec5-a11b8bbee7f6::lt_to_le_cdaa776790c8",
    "9d202bd9-bfcc-53d0-9b7f-7f9afd6b14d0::lt_to_le_2ae119eb34a0",
    "71c0b7e7-8f8c-5fce-9307-7f798e5d5fde::lt_to_le_83c70f4e45d2",
    "c6e2db3c-ba10-5e15-9684-abf33197df92::lt_to_le_fffaf01a32eb",
    "ec082dca-4f4f-5801-9418-5fbf36b090f6::lt_to_le_1e835f30de15",
    "56c9caca-227d-5245-8b90-7e51fbe79cab::lt_to_le_b441aa37b9f2",
    "bbb5dcce-c699-5815-99db-4db3b0a4581c::lt_to_le_a73ba81d2c6b",
}
HARD_REJECT = set(scale.HARD_REJECT) | {"lt_to_ne"}

def classify(packet: Mapping[str, Any], raw: Mapping[str, Any], parent: Mapping[str, Any], success: bool,
             equality_choice: Mapping[str, tuple[str, str | None]]) -> tuple[str, str, str]:
    rid=str(packet["record_id"]); method=str(packet["method"]); pid=str(packet["parent_id"]); goal=str(packet["parent"]["goal"])
    contaminated=scale.contaminated(packet,raw); problem_ok=scale.problem_ok(method,raw); reasons=bool(packet.get("reason_codes"))
    if method in HARD_REJECT: return "reject","extremely_low","pure reordering, symmetry, negation weakening, or proof-only variation"
    if method.startswith("eq_to_le_"):
        category,chosen=equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS: return "reject","low","pure computed remainder/function/physical/closed value, or Nat.mod <= 0 disguised equality"
        if chosen is None:
            labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of expression equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
            return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
        if method!=chosen: return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","scope, declaration, or problem-text pollution"
        basis="natural nontrivial numerical or algebraic bound"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method in {"and_left","and_right"}:
        if rid not in SELECTED_AND: return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: scalar/weak component, lost binder, or pollution"
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","selected-looking component has unresolved pollution"
        basis="sole retained substantive divisibility, range, equivalence direction, or inequality component"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method=="iff_forward":
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","iff split under quantifier/negation or polluted declaration"
        basis="sole retained nontrivial classification, completeness, construction, or parameter direction"
        return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
    if method=="iff_reverse": return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward direction owns this upstream"
    if method in {"implication_contrapositive","implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE: return "reject","extremely_low","quantifier-internal arrow or disjunction repackaging is not substantive"
        if contaminated or reasons or not problem_ok or scale.top_arrow(goal) is None: return "reject","extremely_low","contrapositive has scope pollution"
        basis="complete top-level contrapositive with substantive nonexistence/divisibility content"
        return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
    if method=="le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok or not b5.top_order(goal,("≤","≥")): return "reject","extremely_low","order token cut inside quantifier/Prop, or bare scalar bound merely restated"
        basis="substantive top-level order refined into strict and equality cases"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method=="lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS: return "reject","low","scope/conjunction cut, bare function value, or closed numerical weakening"
        if contaminated or reasons or not problem_ok or not b5.top_order(goal,("<",">")): return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
        basis="natural reusable non-strict consequence of a substantive strict relation"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"

def main() -> None:
    if OUTPUT.exists() or REPORT.exists(): raise FileExistsError("review output already exists")
    ordered=[str(r["record_id"]) for _,r in scale.rows(MANIFEST)]; wanted=set(ordered)
    if len(ordered)!=500 or len(wanted)!=500: raise ValueError("manifest must contain 500 unique record_ids")
    packets={r["record_id"]:r for _,r in scale.rows(scale.PACKETS) if r.get("record_id") in wanted}; raw_all=[r for _,r in scale.rows(scale.RAW)]
    raw={r["record_id"]:r for r in raw_all if r.get("record_id") in wanted}; pids={str(r["parent_id"]) for r in packets.values()}
    parents={r["record_id"]:r for _,r in scale.rows(scale.PARENTS) if r.get("record_id") in pids}
    if set(packets)!=wanted or set(raw)!=wanted or set(parents)!=pids: raise ValueError("packet/raw/parent join incomplete")
    groups:dict[str,list[dict[str,str]]]=defaultdict(list)
    for r in raw_all:
        if r.get("upstream_source"): groups[str(r["upstream_source"])].append({"record_id":str(r["record_id"]),"method":str(r.get("question_type",""))})
    for rid in ordered:
        if str(raw[rid].get("upstream_source",""))!="parent:"+str(packets[rid]["parent_id"]): raise AssertionError(f"raw upstream mismatch {rid}")
    receipt_rows=[r for _,r in scale.rows(RECEIPTS)]; receipts:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for r in receipt_rows:
        if r.get("record_id") in wanted: receipts[str(r["record_id"])].append(r)
    if len(receipt_rows)!=500 or set(receipts)!=wanted: raise ValueError(f"Pantograph batch incomplete: rows={len(receipt_rows)}, ids={len(receipts)}")
    eq={}
    for p in packets.values():
        if str(p["method"]).startswith("eq_to_le_"): eq.setdefault(str(p["parent_id"]),scale.equality_class(p,parents[str(p["parent_id"])]))
    out=[]; decisions=Counter(); tiers=Counter(); methods=Counter(); receipt_stats=Counter(); eligible=defaultdict(list); passes=defaultdict(list)
    for number,rid in enumerate(ordered,1):
        p=packets[rid]; rr=raw[rid]; ph=str(p["candidate"]["proof_sha256"]); sh=str(p["diff"]["statement"]["after_sha256"])
        success=scale.compatible_success(receipts[rid],ph); receipt_stats["compatible_success" if success else "fail"]+=1
        d,t,basis=classify(p,rr,parents[str(p["parent_id"])],success,eq); hd,_,_=classify(p,rr,parents[str(p["parent_id"])],True,eq); up=str(rr["upstream_source"])
        if hd=="pass": eligible[up].append(rid)
        if d=="pass":
            if not success: raise AssertionError(f"pass without success {rid}")
            passes[up].append(rid)
        siblings=[x for x in groups[up] if x["record_id"]!=rid]; sibling_text=", ".join(f"{x['method']}:{x['record_id']}" for x in siblings) or "none"
        receipt_text="compatible Pantograph success" if success else "Pantograph fail: "+b5.error_excerpt(receipts[rid])
        reason=(f"Row {number}; frozen raw upstream_source={up}. Compared parent goal `{p['parent']['goal']}` with candidate goal `{p['candidate']['goal']}`, problem text, proof, and complete global siblings [{sibling_text}]. Decision basis: {basis}. Exact statement/proof receipt: {receipt_text}.")
        method=str(p["method"]); out.append({"record_id":rid,"quality_decision":d,"quality_tier":t,"manual_reviewed":True,"reviewer":"codex-quality-adjudicator","review_reason":reason,"reviewed_statement_sha256":sh,"reviewed_proof_sha256":ph,"review_version":VERSION,"parent_id":up,"upstream_source":up,"method":method,"siblings":siblings,"pantograph_compatible_success":success})
        decisions[d]+=1; tiers[t]+=1; methods[(method,d)]+=1
    eb={k:v for k,v in eligible.items() if len(v)>1}; pb={k:v for k,v in passes.items() if len(v)>1}
    if eb or pb: raise AssertionError(f"multiple retained siblings eligible={eb} pass={pb}")
    if len(out)!=500 or len({r['record_id'] for r in out})!=500: raise AssertionError("coverage failure")
    payload=b"".join((json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n").encode() for r in out)
    report={"schema_version":"numinamath_shard_review_report_v1","review_version":VERSION,"batch_id":"numinamath_expand_s00_b00010","rows":500,"unique_record_ids":500,"candidate_manifest_sha256":scale.sha_file(MANIFEST),"receipts_sha256":scale.sha_file(RECEIPTS),"packets_sha256":scale.sha_file(scale.PACKETS),"raw_sha256":scale.sha_file(scale.RAW),"receipt_status":dict(sorted(receipt_stats.items())),"final_decisions":dict(sorted(decisions.items())),"final_tiers":dict(sorted(tiers.items())),"by_method_decision":{f"{m}|{d}":n for (m,d),n in sorted(methods.items())},"raw_upstream_parent_groups":len({raw[r]["upstream_source"] for r in ordered}),"max_eligible_per_upstream":max(map(len,eligible.values()),default=0),"max_pass_per_upstream":max(map(len,passes.values()),default=0),"global_siblings_embedded":True,"output_sha256":hashlib.sha256(payload).hexdigest()}
    b5.safe_write(OUTPUT,payload); b5.safe_write(REPORT,(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()); print(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=="__main__": main()
