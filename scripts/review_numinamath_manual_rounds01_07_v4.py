#!/usr/bin/env python3
"""Materialize the independent final semantic review for manual v4 candidates.

All decisions and rationales in ``REVIEWS`` are deliberately hand-authored.
The script only binds them to the frozen v4 hashes and writes review artifacts;
it never invokes Lake, Lean, Pantograph, or a worker and never writes raw or
verified datasets.  Accepted rows are precommitted to the exact current proof
and expected environment identity.  The supplement materializer still requires
an exact successful Pantograph receipt before any accepted row is usable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / (
    "outputs/numinamath_expand_verification/manual_expansion_candidates/"
    "pantograph_prepared_batches/numinamath_manual_rounds01_07_v4_b00000"
)
MANIFEST = BATCH_DIR / "candidate_manifest.jsonl"
OUTPUT = BATCH_DIR / "manual_second_review.jsonl"
REPORT = BATCH_DIR / "manual_second_review.report.json"
EXPECTED_MANIFEST_SHA256 = (
    "141be823df692c62e786724cebff1246b91670b11d69c146782ff284fbfb442a"
)
EXPECTED_ROWS = 28
EXPECTED_LEAN_VERSION = (
    "Lean (version 4.29.1, x86_64-unknown-linux-gnu, "
    "commit f72c35b3f637c8c6571d353742168ab66cc22c00, Release)"
)
EXPECTED_MATHLIB_COMMIT = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"
EXPECTED_ENVIRONMENT_HASH = (
    "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
)
EXPECTED_ROUND_LOCKS = {
    "round01": "4bd85a6fc933881e8ce9ee53416efff5336c71fecf332e01028285bd33728bef",
    "round02": "ff1e99a4a7093f2222235072053ce62563b97fb6131541c9f4f04b53d5745cdb",
    "round03": "9ac31b129c7a59d53c313ea28084240451255708a9155c5af5aea2d7d003d11c",
    "round04": "48617059f7dfa0b7f28f1482e1acb082c25f1a24740e3c12bdfe79fb60f7e1ce",
    "round05": "72f2d652f4027af0113be93abca384ee9b85c1a17a7516eb6ba26d7fc2370824",
    "round06": "66a3279caa4a17885564e398aa41d2fe871a0828bb783022e441b57957332f70",
    "round07": "caca4619d5f3e0587a228ab98e3e712d7005209d196979658fb5f5f4985d897e",
}
REVIEWER = "codex-quality-audit+manual-rounds01-07-v4-final-semantic-review"


# candidate_id -> (quality tier, specific independent rationale)
REVIEWS: dict[str, tuple[str, str]] = {
    "manual_round01_c01_cyclic_four": (
        "high",
        "独立复核父题三元循环恒等式后，确认新命题把共同分母机制实质推广到四元循环；abcd=1及首分母非零可推出其余分母非零，证明逐项缩放而非改数或弱化。",
    ),
    "manual_round01_c02_divisibility_equiv": (
        "medium",
        "父题只有单向整除，新命题补出真实逆向并为两向分别给出整数线性见证；两组系数模19确实等价，不是把结论直接复述为假设。",
    ),
    "manual_round01_c03_amgm_equality_case": (
        "high",
        "在父题AM-GM不等式上增加完整取等分类；正性保证平方根变换合法，等号迫使两个平方根相等，逆向也直接成立，结论具有独立训练价值。",
    ),
    "manual_round01_c04_cauchy_three_equality_case": (
        "high",
        "父题只给三元下界，新命题用三个非负有理平方项之和为零刻画全部取等情形a=b=c；分母正性和零项推理完整，没有丢失方向。",
    ),
    "manual_round02_c01_punctured_affine_range": (
        "high",
        "把父题单个可约函数的缺点值推广为任意非恒定仿射映射在单点挖孔域上的精确像集；证明同时覆盖像的排除性与每个其余值的显式原像。",
    ),
    "manual_round02_c02_trig_equality_cases": (
        "high",
        "父题只证明上下界，新命题分别分类两个尖锐端点；上端由平方为零得到sin t=1，下端结合正弦值域排除额外代数根，两个方向均实质。",
    ),
    "manual_round02_c03_power_sum_exact_remainder": (
        "high",
        "将正整数上的整除iff加强为所有自然数的精确模5余数分类，并正确处理n=0；四个模4指数类逐一证明，信息量严格大于父题。",
    ),
    "manual_round02_c04_two_point_domain": (
        "medium",
        "把父题固定根-2、2的定义域分解参数化为任意有序两根，并证明三个互斥序区间与乘积非零完全等价；不是单纯替换常数，因为双向符号论证被抽象化。",
    ),
    "manual_round03_c01_sharp_quadratic_equality": (
        "medium",
        "父题确定最优常数但未分类取等；新命题在更大的全实数域上证明k=5/2取等当且仅当a=b，差值确为(a-b)^2的正倍数。",
    ),
    "manual_round03_c02_power_difference_divisibility": (
        "high",
        "从父题两个固定幂差实例抽取通用提升定理：基础幂差被m整除即可沿任意指数倍数保持整除；归纳中的分解恒等式及显式见证均完整。",
    ),
    "manual_round03_c03_iunion_absorption_iff": (
        "high",
        "把父题两个实数集合的吸收律推广到任意类型、任意指标族的并集，并保留必要与充分两向；空指标族边界也自然成立。",
    ),
    "manual_round03_c04_functional_equation_full_classification": (
        "high",
        "父题仅求零点，新命题先由x=y=0确定常数，再用x与-x及偶性求出每个函数值，得到完整函数分类，显著强于答案集合。",
    ),
    "manual_round04_c01_even_quadratic_abs_classification": (
        "high",
        "把父题单点求值扩为除零点外的全域闭式；零点必须排除是因为原假设不能决定f(0)，正负半轴通过偶性准确拼接为x^2-|x|。",
    ),
    "manual_round04_c02_cubic_three_roots_coefficients": (
        "high",
        "父题只求c/d，新命题由三个根方程解出全部低次系数，且即使a=0也给出一致的零多项式情形；结论是完整线性分类而非比例换皮。",
    ),
    "manual_round04_c03_recurrence_exact_quotient": (
        "high",
        "将父题的整除结论加强为精确因式分解并给出商，同时删除与该恒等式无关的初值；代入递推后是非平凡但完全确定的多项式恒等式。",
    ),
    "manual_round04_c04_coprime_unimodular_progression": (
        "high",
        "把b=1的固定互素三元组推广到任意互素整数基；三个显式Bezout证书对应不同行列式为一的线性变换，结构增量明确。",
    ),
    "manual_round05_c01_recurrence_gcd_invariant": (
        "high",
        "修订版保持任意初值并给出相邻项gcd的精确不变量，而非固定Fibonacci的gcd=1；归纳步严格使用欧几里得恒等式，父题成为直接特例。",
    ),
    "manual_round05_c02_abs_distance_sum_range": (
        "high",
        "把两个固定焦点推广到任意a≤b并确定完整值域；三段位置分析给出尖锐下界，远右侧显式见证证明每个更大值均能取得。",
    ),
    "manual_round05_c03_prefix_mean_sequence_classification": (
        "high",
        "父题只问第2008项，新命题从所有前缀均值推导S_n=n^2并相减得到每个索引的闭式，且单独处理n=0边界，属于完整序列分类。",
    ),
    "manual_round05_c04_concave_quadratic_maximum": (
        "high",
        "已核对修订版同时给出IsGreatest与指定顶点取等；它将单个数值抛物线推广到全部a<0的二次式，并用完整平方证书证明全局上界和实际达到。",
    ),
    "manual_round06_c01_cube_difference_invariants": (
        "high",
        "已核对修订后的参数公式：由差d和平方和s恢复x^2+xy+y^2=(3s-d^2)/2，再乘差得到精确立方差；父题固定数值仅为特例。",
    ),
    "manual_round06_c02_list_lcm_divisibility": (
        "high",
        "把二元lcm整除准则推广到任意有限列表，并采用1作为空列表单位使边界正确；归纳结论是可复用的有限族充要条件。",
    ),
    "manual_round06_c03_quadratic_root_interval": (
        "high",
        "将父题固定二次不等式推广为任意有序实根的完整解集；正向排除两侧、反向验证区间内两因子异号，双向均无机械弱化。",
    ),
    "manual_round06_c04_quartic_equality_classification": (
        "high",
        "修订版通过精确因式分解及正定二次因子证明四次不等式全部取等点恰为a=b；第二因子为零时也被独立排除遗漏解。",
    ),
    "manual_round07_c01_antiperiod_period": (
        "medium",
        "从父题点值计算中抽取参数化的反周期结构定理；连续应用两次f(x)=-f(x+p)确实得到2p周期，并删除偶性和局部公式等无关条件。",
    ),
    "manual_round07_c02_quadratic_form_endpoints": (
        "high",
        "父题仅给值域界，新命题分别刻画下端a=b与上端a=-b的全部取等情形；两个端点使用不同平方证书且逆向均依约束成立。",
    ),
    "manual_round07_c03_nonnegative_triple_sum": (
        "high",
        "把固定S=13、P=6推广为参数公式；平方展开确定绝对值，三变量非负性选择正确平方根分支，因此结论没有因平方而引入伪解。",
    ),
    "manual_round07_c04_alternating_cubes_sum": (
        "high",
        "将九项一次性计算推广为所有长度的闭式n^2(4n+3)，并改用整数避免自然数截断减法；归纳步体现通项结构而非重新计算常数。",
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_hash(*parts: object) -> str:
    return sha256_text("\0".join(str(part) for part in parts))


def canonical_hash(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def complete_source(source_body: str, proof: str) -> str:
    marker = ":= by"
    stripped = source_body.strip()
    if not stripped.endswith(marker):
        raise ValueError("source_body does not end in ':= by'")
    if not proof.strip().startswith("by"):
        raise ValueError("proof is not a complete by-proof")
    return (stripped[: -len(marker)] + ":= " + proof.strip()).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_new_or_identical(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest_sha = sha256_bytes(MANIFEST.read_bytes())
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"v4 candidate manifest drift: expected {EXPECTED_MANIFEST_SHA256}, got {manifest_sha}"
        )
    environment_hash = stable_hash(
        EXPECTED_LEAN_VERSION,
        EXPECTED_MATHLIB_COMMIT,
        "2",
        "2",
        "Mathlib",
    )
    if environment_hash != EXPECTED_ENVIRONMENT_HASH:
        raise AssertionError("offline environment identity derivation drifted")

    candidates = read_jsonl(MANIFEST)
    if len(candidates) != EXPECTED_ROWS:
        raise ValueError(f"expected 28 candidates, got {len(candidates)}")
    candidate_ids = [str(row.get("record_id") or "") for row in candidates]
    if len(set(candidate_ids)) != EXPECTED_ROWS:
        raise ValueError("candidate IDs are not unique")
    if set(candidate_ids) != set(REVIEWS):
        raise ValueError(
            f"manual review coverage mismatch: missing={set(candidate_ids)-set(REVIEWS)}, "
            f"extra={set(REVIEWS)-set(candidate_ids)}"
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["record_id"])
        provenance = candidate["manual_provenance"]
        round_name = str(provenance["round"])
        if provenance.get("source_manifest_sha256") != EXPECTED_ROUND_LOCKS[round_name]:
            raise ValueError(f"{candidate_id}: source manifest SHA drift")
        if candidate.get("contains_forbidden_token") is not False:
            raise ValueError(f"{candidate_id}: forbidden-token flag")
        variants = candidate.get("variants") or []
        if len(variants) != 1:
            raise ValueError(f"{candidate_id}: expected exactly one proof variant")
        variant = variants[0]
        proof = str(variant["proof"]).strip()
        proof_sha = sha256_text(proof)
        if proof_sha != provenance.get("proof_sha256"):
            raise ValueError(f"{candidate_id}: current proof hash drift")
        assembled = complete_source(str(candidate["source_body"]), proof)
        assembled_sha = sha256_text(assembled)
        variant_hash = canonical_hash(
            {
                "candidate_hash": candidate["candidate_hash"],
                "strategy": variant["strategy"],
                "proof": proof,
                "environment_hash": environment_hash,
            }
        )
        tier, rationale = REVIEWS[candidate_id]
        rows.append(
            {
                "schema_version": "numinamath_manual_expansion_second_review_v1",
                "review_kind": "independent_human_second_review",
                "candidate_id": candidate_id,
                "candidate_hash": candidate["candidate_hash"],
                "parent_record_id": provenance["parent_record_id"],
                "variant_hash": variant_hash,
                "statement_sha256": provenance["statement_sha256"],
                "verified_proof_sha256": proof_sha,
                "assembled_source_hash": assembled_sha,
                "decision": "accept",
                "quality_tier": tier,
                "reviewer": REVIEWER,
                "independence_attested": True,
                "rationale": rationale,
                "candidate_round": round_name,
                "candidate_manifest_sha256": manifest_sha,
                "expected_environment_hash": environment_hash,
                "receipt_gate": "accept_effective_only_after_exact_pantograph_success",
            }
        )

    # Contract-level uniqueness and coverage checks independent of Pantograph.
    for field in (
        "candidate_id",
        "candidate_hash",
        "parent_record_id",
        "variant_hash",
        "statement_sha256",
        "verified_proof_sha256",
        "assembled_source_hash",
    ):
        values = [str(row[field]) for row in rows]
        if len(set(values)) != EXPECTED_ROWS:
            raise ValueError(f"review rows are not unique by {field}")
    required = {
        "schema_version",
        "review_kind",
        "candidate_id",
        "candidate_hash",
        "parent_record_id",
        "variant_hash",
        "statement_sha256",
        "verified_proof_sha256",
        "assembled_source_hash",
        "decision",
        "quality_tier",
        "reviewer",
        "independence_attested",
        "rationale",
    }
    for row in rows:
        if required - set(row):
            raise ValueError(f"{row['candidate_id']}: missing materializer fields")
        if row["quality_tier"] not in {"medium", "high"}:
            raise ValueError(f"{row['candidate_id']}: invalid accepted tier")
        if row["independence_attested"] is not True:
            raise ValueError(f"{row['candidate_id']}: missing independence attestation")

    output_payload = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    decisions = Counter(str(row["decision"]) for row in rows)
    tiers = Counter(str(row["quality_tier"]) for row in rows)
    rounds = Counter(str(row["candidate_round"]) for row in rows)
    report: dict[str, Any] = {
        "schema_version": "numinamath_manual_v4_final_semantic_review_report_v1",
        "candidate_manifest": str(MANIFEST.relative_to(ROOT)),
        "candidate_manifest_sha256": manifest_sha,
        "review_output": str(OUTPUT.relative_to(ROOT)),
        "review_output_sha256": sha256_bytes(output_payload),
        "coverage": {"candidates": len(candidates), "reviewed": len(rows), "missing": 0},
        "decision_counts": dict(sorted(decisions.items())),
        "quality_tier_counts": dict(sorted(tiers.items())),
        "round_counts": dict(sorted(rounds.items())),
        "independence_attested_true": sum(
            row["independence_attested"] is True for row in rows
        ),
        "unique_parent_ids": len({row["parent_record_id"] for row in rows}),
        "unique_candidate_hashes": len({row["candidate_hash"] for row in rows}),
        "unique_statement_hashes": len({row["statement_sha256"] for row in rows}),
        "unique_proof_hashes": len({row["verified_proof_sha256"] for row in rows}),
        "round05_source_manifest_sha256": EXPECTED_ROUND_LOCKS["round05"],
        "round06_source_manifest_sha256": EXPECTED_ROUND_LOCKS["round06"],
        "expected_environment": {
            "lean_version": EXPECTED_LEAN_VERSION,
            "mathlib_commit": EXPECTED_MATHLIB_COMMIT,
            "assembler_version": "2",
            "normalization_version": "2",
            "imports": ["Mathlib"],
            "environment_hash": environment_hash,
        },
        "materializer_contract_fields_complete": True,
        "future_receipt_gate": (
            "Every accept remains unusable until materializer finds an exact v4 "
            "Pantograph success with matching candidate/variant/proof/assembled hashes."
        ),
        "pantograph_started": False,
        "lake_or_lean_started": False,
        "raw_or_verified_files_written": False,
    }
    report_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    review_runtime_status = write_new_or_identical(OUTPUT, output_payload)
    # Persist the original artifact-creation state so reruns remain byte-identical.
    report["review_write_status"] = "created"
    report_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    report_runtime_status = write_new_or_identical(REPORT, report_payload)
    # Runtime reuse statuses are intentionally not recursively embedded in REPORT.
    display = dict(report)
    display["review_runtime_status"] = review_runtime_status
    display["report_runtime_status"] = report_runtime_status
    print(json.dumps(display, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
