#!/usr/bin/env python3
"""Strict resolved review for NuminaMath shard s00/b00012."""

from __future__ import annotations
import hashlib, importlib.util, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00012"
BATCH_ID = BATCH.name
MANIFEST = BATCH / "candidate_manifest.jsonl"
RECEIPTS = BATCH / "verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00012_resolved.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00012_resolved_report.json")
VERSION = "numinamath_s00_b00012_strict_resolved_v1"

spec = importlib.util.spec_from_file_location("numinamath_b11_helpers", ROOT / "scripts/review_numinamath_s00_b00011.py")
helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
scale = helper.scale; b5 = helper.b5

SELECTED_AND = {
    "c803e0cb-74fe-5244-8d7c-87eed69f8bc8::and_left_789202027ba2",
    "1ea04885-bc57-5fd1-a2aa-c2e1380231c4::and_right_3ccc46d22e3c",
    "7f5ebd39-a32b-5e16-a403-c29b1b8e6aa1::and_right_509419d86733",
    "6d5257b1-f08b-5e30-bdc5-b950cc41361b::and_left_a3ac88d3444c",
    "4f62d529-dc0c-52a6-95a5-a818972c2113::and_right_2bacb1afeae9",
    "ce4b126a-3523-5b5b-b1d7-603c83d62da9::and_right_d25069ffbf03",
    "798eaad3-ae3b-561e-a53b-0b34b1344d15::and_right_f77df12e5529",
    "946bc5fe-4176-592b-9991-710f709066e7::and_left_a56eefda3f9f",
    "05f15281-693f-5ca5-b0b9-531936c35da0::and_right_8db00daf0543",
    "4b128ca7-7764-5608-9424-7aaf603be250::and_left_1569e1dea345",
}

TRIVIAL_EQUALITY_BOUNDS = {
    "cbc15768-8b79-5ba3-9a07-a1185f1e07cc::eq_to_le_forward_3094eb0c0433",
    "bdb8c0b3-277f-5e86-94af-ae9138d38fa1::eq_to_le_forward_0a39794f83a4",
    "7b4b6c05-cbc9-5544-b9cc-6883271a7034::eq_to_le_forward_724365a1271d",
    "7f2b1ccd-9bb3-5ddd-8b9f-1caf26366713::eq_to_le_forward_48e659aa4b4b",
    "037c1ca2-663e-53b5-a7cd-9e2c329a1a45::eq_to_le_forward_affc6aa0698d",
    "151b10a1-7a4f-56b6-9bdc-d55b8fd2c19a::eq_to_le_forward_86fcd8773490",
}

SELECTED_CONTRAPOSITIVE = {
    "59d62e2c-2ec3-5dd9-88f1-1c1d52cc2e5c::implication_contrapositive_041f6bc072f5",
}

INVALID_ORDER_REFINEMENTS = {
    "f343032a-b6c2-57ee-abc5-b2a17cbcdc2b::le_to_lt_or_eq_1da0c5d2c12f",
    "fd671d33-da96-5991-beea-e36aa0ec9964::le_to_lt_or_eq_001bf7581f91",
    "a039b0a4-daaa-528c-86df-dd5571646434::le_to_lt_or_eq_eb173466f8eb",
}

TRIVIAL_STRICT_WEAKENINGS = {
    "7425e5c9-3a1e-5c12-9823-944d7b6fdc2b::lt_to_le_21d0440e7566",
    "09753d82-3174-5bae-97f6-ec63385ca792::lt_to_le_03dc12f510f2",
    "b18cad92-3508-5fb1-a1d9-9e6a162677f7::lt_to_le_7f50d02f1187",
    "e44fdc82-5e32-563f-ada5-96f3a210a1f7::lt_to_le_3cd17df48ab5",
}

HARD_REJECT = set(scale.HARD_REJECT) | {"lt_to_ne"}


def classify(packet: Mapping[str, Any], raw: Mapping[str, Any], parent: Mapping[str, Any], success: bool,
             equality_choice: Mapping[str, tuple[str, str | None]]) -> tuple[str, str, str]:
    del parent
    rid=str(packet["record_id"]); method=str(packet["method"]); pid=str(packet["parent_id"])
    goal=str(packet["parent"]["goal"]); contaminated=scale.contaminated(packet,raw)
    problem_ok=scale.problem_ok(method,raw); reasons=bool(packet.get("reason_codes"))
    if method in HARD_REJECT:
        return "reject","extremely_low","pure reordering, symmetry, negation weakening, or proof-only variation"
    if method.startswith("eq_to_le_"):
        category,chosen=equality_choice[pid]
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject","low","closed trig/computed value, Nat.mod <= 0, tautological identity, or bare speed ratio"
        if chosen is None:
            labels={"bare_value_to_numeric_bound":"bare variable/function value weakened to a scalar bound","arbitrary_expression_order":"arbitrary ordering of equality sides","invalid_scope_or_prop_cut":"equality cut inside quantifier/Prop/set/conjunction/declaration","closed_numeric_relaxation":"closed numerical equality mechanically weakened","not_top_level_equality":"parent is not a safe top-level equality"}
            return "reject","extremely_low" if category=="invalid_scope_or_prop_cut" else "low",labels.get(category,category)
        if method!=chosen: return "reject","low",f"paired equality direction; sole natural direction is {chosen}"
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","scope, declaration, or problem-text pollution"
        basis="natural nontrivial numerical, algebraic, geometric, or finite-set bound"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method in {"and_left","and_right"}:
        if rid not in SELECTED_AND:
            return "reject","extremely_low" if contaminated or reasons else "low","not the sole substantive global sibling: scalar/weaker component, lost binder, or pollution"
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","selected-looking component has unresolved scope or text pollution"
        basis="sole retained substantive extremum, range, functional, monotonicity, or inequality component"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method=="iff_forward":
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","iff split under quantifier/existential, declaration, or polluted syntax"
        basis="sole retained nontrivial solution, classification, completeness, or parameter direction"
        return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
    if method=="iff_reverse":
        return "reject","extremely_low" if contaminated or reasons else "low","reverse sibling rejected; global forward direction owns this upstream"
    if method in {"implication_contrapositive","implication_as_disjunction"}:
        if rid not in SELECTED_CONTRAPOSITIVE: return "reject","extremely_low","non-top-level arrow, escaped binder, or premise-negation repackaging"
        if contaminated or reasons or not problem_ok or scale.top_arrow(goal) is None: return "reject","extremely_low","contrapositive has scope pollution"
        basis="complete top-level contrapositive with substantive parity and sum-of-squares content"
        return ("pass","high",basis) if success else ("pending","high",basis+"; exact proof needs repair")
    if method=="le_to_lt_or_eq":
        if rid in INVALID_ORDER_REFINEMENTS or contaminated or reasons or not problem_ok: return "reject","extremely_low","bare scalar/function-value case split or scope pollution"
        basis="substantive top-level order refined into strict and equality cases"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    if method=="lt_to_le":
        if rid in TRIVIAL_STRICT_WEAKENINGS: return "reject","low","existential/forall scope cut or bare function-value weakening"
        if contaminated or reasons or not problem_ok: return "reject","extremely_low","strict relation cut inside quantifier/Prop/declaration"
        basis="natural reusable non-strict consequence of a substantive strict relation"
        return ("pass","medium",basis) if success else ("pending","medium",basis+"; exact proof needs repair")
    return "reject","extremely_low",f"method {method} is outside calibrated acceptance scale"


def main() -> None:
    if OUTPUT.exists() or REPORT.exists(): raise FileExistsError("resolved review output already exists")
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
        reason=f"Row {number}; frozen raw upstream_source={up}. Compared parent goal `{p['parent']['goal']}` with candidate goal `{p['candidate']['goal']}`, problem text, proof, latest cross-shard adjudication, and complete global siblings [{sibling_text}]. Decision basis: {basis}. Exact statement/proof receipt: {receipt_text}."
        method=str(p["method"]); out.append({"record_id":rid,"quality_decision":d,"quality_tier":t,"manual_reviewed":True,"reviewer":"codex-quality-adjudicator","review_reason":reason,"reviewed_statement_sha256":sh,"reviewed_proof_sha256":ph,"review_version":VERSION,"parent_id":up,"upstream_source":up,"method":method,"siblings":siblings,"pantograph_compatible_success":success})
        decisions[d]+=1; tiers[t]+=1; methods[(method,d)]+=1
    eb={k:v for k,v in eligible.items() if len(v)>1}; pb={k:v for k,v in passes.items() if len(v)>1}
    if eb or pb: raise AssertionError(f"multiple retained siblings eligible={eb} pass={pb}")
    if len(out)!=500 or len({r['record_id'] for r in out})!=500: raise AssertionError("coverage failure")
    payload=b"".join((json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n").encode() for r in out)
    report={"schema_version":"numinamath_shard_review_report_v1","review_version":VERSION,"batch_id":BATCH_ID,"rows":500,"unique_record_ids":500,"candidate_manifest_sha256":scale.sha_file(MANIFEST),"receipts_sha256":scale.sha_file(RECEIPTS),"packets_sha256":scale.sha_file(scale.PACKETS),"raw_sha256":scale.sha_file(scale.RAW),"receipt_status":dict(sorted(receipt_stats.items())),"final_decisions":dict(sorted(decisions.items())),"final_tiers":dict(sorted(tiers.items())),"by_method_decision":{f"{m}|{d}":n for (m,d),n in sorted(methods.items())},"raw_upstream_parent_groups":len({raw[r]["upstream_source"] for r in ordered}),"max_eligible_per_upstream":max(map(len,eligible.values()),default=0),"max_pass_per_upstream":max(map(len,passes.values()),default=0),"global_siblings_embedded":True,"output_sha256":hashlib.sha256(payload).hexdigest()}
    b5.safe_write(OUTPUT,payload); b5.safe_write(REPORT,(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()); print(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=="__main__": main()
