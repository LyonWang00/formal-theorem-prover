from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


K_VALUES = (1, 4, 8, 16, 32)
STRICT_API_PATTERNS = {
    "unknown_identifier": re.compile(r"\bunknown identifier\b", re.I),
    "unknown_constant_or_declaration": re.compile(
        r"\bunknown (?:constant|declaration|namespace|tactic)\b", re.I
    ),
    "missing_field": re.compile(r"\b(?:no|unknown) field (?:named )?\b", re.I),
    "invalid_field_name": re.compile(r"\binvalid field (?:name|notation)\b", re.I),
}
INVALID_PROJECTION = re.compile(r"\binvalid projection\b", re.I)
SYNTAX_PATTERN = re.compile(
    r"unexpected (?:token|identifier)|expected (?:token|command|term)|parser|invalid syntax",
    re.I,
)
TYPE_PATTERN = re.compile(
    r"application type mismatch|type mismatch|failed to synthesize|function expected|"
    r"invalid argument|typeclass instance problem|has type .* but is expected",
    re.I | re.S,
)
TACTIC_PATTERN = re.compile(
    r"tactic .*failed|linarith failed|omega could not|simp made no progress|"
    r"no goals to be solved|unsolved goals",
    re.I,
)
FORBIDDEN_PATTERN = re.compile(r"\b(?:sorry|admit)\b", re.I)
API_SYMBOL_PATTERN = re.compile(
    r"\bunknown (?:identifier|constant|declaration|namespace|tactic)\b\s*[`'\"]?"
    r"([A-Za-z_][A-Za-z0-9_'.]*)",
    re.I,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_generations(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    # GRPO generation is sharded (shard_0..shard_3/chunks); SFT uses one
    # flat chunks directory.  Recursing keeps both layouts on the same path
    # while retaining the duplicate-key guard.
    for path in sorted(root.rglob("problem_*.jsonl")):
        for row in iter_jsonl(path):
            key = (str(row["problem_id"]), int(row["attempt_index"]))
            if key in rows:
                raise ValueError(f"duplicate generation key {key}")
            rows[key] = row
    if len(rows) != 244 * 32:
        raise ValueError(f"generation rows={len(rows)}, expected 7808")
    by_problem: dict[str, set[int]] = defaultdict(set)
    for problem_id, attempt_index in rows:
        by_problem[problem_id].add(attempt_index)
    if len(by_problem) != 244 or any(v != set(range(32)) for v in by_problem.values()):
        raise ValueError("generation is not exact 244 x attempt indices 0..31")
    return rows


def load_sft_receipts(
    bindings_path: Path, receipt_root: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    bindings = list(iter_jsonl(bindings_path))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bindings:
        grouped[str(row["source_receipts_relative_path"])].append(row)
    receipts: dict[tuple[str, int], dict[str, Any]] = {}
    for relative, rows in grouped.items():
        path = receipt_root / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        for binding in rows:
            line_number = int(binding["source_line_number"])
            receipt = json.loads(lines[line_number - 1])
            key = (str(binding["problem_id"]), int(binding["attempt_index"]))
            if key in receipts:
                raise ValueError(f"duplicate SFT receipt key {key}")
            if str(receipt["problem_id"]) != key[0] or int(receipt["attempt_index"]) != key[1]:
                raise ValueError(f"SFT binding key mismatch {key}")
            if bool(receipt.get("success")) != bool(binding.get("success")):
                raise ValueError(f"SFT binding success mismatch {key}")
            if str(receipt.get("generation_payload_sha256")) != str(
                binding.get("generation_payload_sha256")
            ):
                raise ValueError(f"SFT generation hash mismatch {key}")
            receipts[key] = receipt
    if len(receipts) != 244 * 32:
        raise ValueError(f"SFT receipts={len(receipts)}, expected 7808")
    return receipts


def load_grpo_receipts(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    receipts = {}
    for row in iter_jsonl(path):
        key = (str(row["problem_id"]), int(row["attempt_index"]))
        if key in receipts:
            raise ValueError(f"duplicate GRPO receipt key {key}")
        receipts[key] = row
    if len(receipts) != 244 * 32:
        raise ValueError(f"GRPO receipts={len(receipts)}, expected 7808")
    return receipts


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def classify_error(row: dict[str, Any]) -> tuple[str, list[str]]:
    if row.get("success"):
        return "success", []
    if row.get("timed_out"):
        return "timeout", []
    rejected = str(row.get("rejected_reason") or "")
    text = "\n".join(
        [
            str(row.get("diagnostics") or ""),
            rejected,
            *[str(item) for item in (row.get("compile_errors") or [])],
        ]
    )
    api = [name for name, pattern in STRICT_API_PATTERNS.items() if pattern.search(text)]
    if rejected or FORBIDDEN_PATTERN.search(str(row.get("generated_proof") or "")):
        return "rejected_or_forbidden", api
    if api:
        return "hallucinated_api_strict", api
    if INVALID_PROJECTION.search(text):
        return "invalid_projection", api
    if SYNTAX_PATTERN.search(text):
        return "syntax_or_parser", api
    if TYPE_PATTERN.search(text):
        return "type_or_elaboration", api
    if TACTIC_PATTERN.search(text):
        return "tactic_or_unsolved_goal", api
    return "other_lean_failure", api


def longest_identical_line_run(text: str) -> int:
    longest = current = 0
    previous = None
    for line in (item.strip() for item in text.splitlines() if item.strip()):
        current = current + 1 if line == previous else 1
        previous = line
        longest = max(longest, current)
    return longest


def max_character_run(text: str) -> int:
    longest = current = 0
    previous = None
    for char in text:
        current = current + 1 if char == previous else 1
        previous = char
        longest = max(longest, current)
    return longest


def summarize_model(
    label: str,
    generations: dict[tuple[str, int], dict[str, Any]],
    receipts: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    if set(generations) != set(receipts):
        raise ValueError(f"{label}: generation/receipt keys differ")
    for key in generations:
        generation_hash = generations[key].get("generation_payload_sha256")
        receipt_hash = receipts[key].get("generation_payload_sha256")
        if generation_hash != receipt_hash:
            raise ValueError(f"{label}: receipt provenance mismatch {key}")

    by_problem: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key in sorted(generations):
        by_problem[key[0]].append((generations[key], receipts[key]))
    successes_per_problem = {
        problem: sum(bool(receipt.get("success")) for _, receipt in sorted(rows, key=lambda x: x[0]["attempt_index"]))
        for problem, rows in by_problem.items()
    }
    prefix_pass = {}
    estimator_pass = {}
    solved_sets: dict[str, list[str]] = {}
    for k in K_VALUES:
        solved = []
        for problem, rows in by_problem.items():
            ordered = sorted(rows, key=lambda x: int(x[0]["attempt_index"]))
            if any(receipt.get("success") for _, receipt in ordered[:k]):
                solved.append(problem)
        solved_sets[str(k)] = sorted(solved)
        prefix_pass[str(k)] = {"count": len(solved), "rate": len(solved) / 244}
        estimate = sum(unbiased_pass_at_k(32, c, k) for c in successes_per_problem.values()) / 244
        estimator_pass[str(k)] = estimate

    normalized = {
        key: str(row.get("normalized_proof") or row.get("generated_proof") or "").strip()
        for key, row in generations.items()
    }
    within_duplicate_attempts = 0
    collision_pairs = 0
    affected = 0
    all_identical = 0
    unique_counts = []
    for problem in by_problem:
        proofs = [normalized[(problem, i)] for i in range(32)]
        counts = Counter(proofs)
        unique = len(counts)
        unique_counts.append(unique)
        duplicates = 32 - unique
        within_duplicate_attempts += duplicates
        collision_pairs += sum(math.comb(count, 2) for count in counts.values())
        affected += int(duplicates > 0)
        all_identical += int(unique == 1)

    error_counts: Counter[str] = Counter()
    api_subtypes: Counter[str] = Counter()
    api_symbols: Counter[str] = Counter()
    api_problems = set()
    for key, receipt in receipts.items():
        category, api = classify_error(receipt)
        error_counts[category] += 1
        if api:
            api_problems.add(key[0])
            api_subtypes.update(api)
            diagnostic_text = "\n".join(
                [
                    str(receipt.get("diagnostics") or ""),
                    *[str(item) for item in (receipt.get("compile_errors") or [])],
                ]
            )
            api_symbols.update(match.group(1) for match in API_SYMBOL_PATTERN.finditer(diagnostic_text))

    lengths = [int(row.get("completion_tokens") or 0) for row in generations.values()]
    verification = [float(row.get("verification_seconds") or 0.0) for row in receipts.values()]
    line_repeat = sum(
        longest_identical_line_run(normalized[key]) >= 4 for key in normalized
    )
    character_repeat = sum(max_character_run(normalized[key]) >= 64 for key in normalized)
    prefix_contamination = sum(
        bool(re.search(r"(?m)^\s*(?:import|namespace|end)\b", proof))
        for proof in normalized.values()
    )
    success_attempts = error_counts["success"]
    return {
        "label": label,
        "problem_count": 244,
        "attempt_count": 7808,
        "successful_attempt_count": success_attempts,
        "attempt_success_rate": success_attempts / 7808,
        "pass_at_k_unbiased_estimator": estimator_pass,
        "pass_at_k_actual_prefix": prefix_pass,
        "solved_problem_ids_by_prefix_k": solved_sets,
        "success_count_histogram": dict(sorted(Counter(successes_per_problem.values()).items())),
        "generation_quality": {
            "within_problem_exact_duplicate_attempt_count": within_duplicate_attempts,
            "within_problem_exact_duplicate_attempt_rate": within_duplicate_attempts / 7808,
            "within_problem_pair_collision_rate": collision_pairs / (244 * math.comb(32, 2)),
            "problems_with_any_exact_duplicate_count": affected,
            "problems_all_32_identical_count": all_identical,
            "mean_unique_proofs_per_problem": statistics.mean(unique_counts),
            "global_unique_normalized_proof_count": len(set(normalized.values())),
            "consecutive_identical_line_run_ge4_attempt_count": line_repeat,
            "same_character_run_ge64_attempt_count": character_repeat,
            "import_namespace_prefix_contamination_count": prefix_contamination,
            "finish_reason_counts": dict(Counter(str(r.get("finish_reason") or "unknown") for r in generations.values())),
            "hit_max_new_tokens_count": sum(bool(r.get("hit_max_new_tokens")) for r in generations.values()),
            "completion_tokens": {
                "mean": statistics.mean(lengths),
                "p50": percentile(lengths, 0.50),
                "p95": percentile(lengths, 0.95),
                "max": max(lengths),
            },
        },
        "compile_quality": {
            "error_category_counts": dict(sorted(error_counts.items())),
            "strict_hallucinated_api_attempt_count": error_counts["hallucinated_api_strict"],
            "strict_hallucinated_api_attempt_rate": error_counts["hallucinated_api_strict"] / 7808,
            "strict_hallucinated_api_problem_count": len(api_problems),
            "strict_hallucinated_api_subtypes": dict(sorted(api_subtypes.items())),
            "strict_hallucinated_api_top_symbols": api_symbols.most_common(30),
            "verification_seconds": {
                "mean": statistics.mean(verification),
                "p50": percentile(verification, 0.50),
                "p95": percentile(verification, 0.95),
                "max": max(verification),
            },
        },
    }


def paired_bootstrap(
    grpo: dict[str, Any], sft: dict[str, Any], k: int, draws: int = 10000
) -> dict[str, float]:
    g = set(grpo["solved_problem_ids_by_prefix_k"][str(k)])
    s = set(sft["solved_problem_ids_by_prefix_k"][str(k)])
    problems = sorted(set(grpo["solved_problem_ids_by_prefix_k"]["32"]) | set(sft["solved_problem_ids_by_prefix_k"]["32"]))
    # Include the common unsolved complement using the exact 244 problem universe.
    all_ids = sorted(set(grpo["_problem_ids"]) | set(sft["_problem_ids"]))
    values = [(int(p in g), int(p in s)) for p in all_ids]
    rng = random.Random(20260829 + k)
    deltas = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        deltas.append(sum(a - b for a, b in sample) / len(sample))
    return {
        "delta": sum(a - b for a, b in values) / len(values),
        "bootstrap_95_low": percentile(deltas, 0.025),
        "bootstrap_95_high": percentile(deltas, 0.975),
    }


def gpu_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = []
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        stamp, gpu, used, total, util = line.split(",")
        rows.append((int(gpu), float(used), float(total), float(util)))
    by_gpu: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for gpu, used, _, util in rows:
        by_gpu[gpu].append((used, util))
    return {
        "samples": len(rows),
        "per_gpu": {
            str(gpu): {
                "memory_used_mib_mean": statistics.mean(v[0] for v in values),
                "memory_used_mib_peak": max(v[0] for v in values),
                "utilization_mean_pct": statistics.mean(v[1] for v in values),
                "utilization_peak_pct": max(v[1] for v in values),
            }
            for gpu, values in sorted(by_gpu.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo-root", required=True)
    parser.add_argument("--sft-eval-root", required=True)
    parser.add_argument("--sft-receipt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    grpo_root = Path(args.grpo_root)
    sft_eval = Path(args.sft_eval_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    grpo_generations = load_generations(grpo_root / "generation")
    grpo_receipts = load_grpo_receipts(grpo_root / "compile_persistent_v1/receipts.jsonl")
    # The SFT run later gained a byte-identical shard_0 mirror.  Bind the
    # original flat chunks view to avoid double-reading the same 244 x 32 rows.
    sft_generations = load_generations(sft_eval / "full_generation/chunks")
    sft_receipts = load_sft_receipts(
        sft_eval / "final_eval/ATTEMPT_RECEIPT_BINDINGS.jsonl",
        Path(args.sft_receipt_root),
    )
    grpo = summarize_model("GRPO-checkpoint9500", grpo_generations, grpo_receipts)
    sft = summarize_model("SFT-epoch2-checkpoint18688", sft_generations, sft_receipts)
    grpo["_problem_ids"] = sorted({key[0] for key in grpo_generations})
    sft["_problem_ids"] = sorted({key[0] for key in sft_generations})
    if grpo["_problem_ids"] != sft["_problem_ids"]:
        raise ValueError("GRPO and SFT problem sets differ")
    paired = {str(k): paired_bootstrap(grpo, sft, k) for k in K_VALUES}
    for item in (grpo, sft):
        item.pop("_problem_ids", None)
    gpu_files = sorted((grpo_root / "audit").glob("gpu-*.csv"))
    report = {
        "schema": "minif2f_test244_grpo_checkpoint9500_vs_sft_epoch2_pass32_analysis_v1_persistent_pantograph",
        "status": "PASS",
        "evaluation_scope": {"split": "test", "problem_count": 244, "samples_per_problem": 32},
        "sampling_contract": {
            "backend": "vllm",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_new_tokens": 2048,
            "max_model_len": 4096,
            "seed_base": 20260711,
        },
        "grpo": grpo,
        "sft": sft,
        "paired_prefix_pass_delta_bootstrap": paired,
        # Retry jobs may only replay already-complete shards and therefore leave
        # short, near-idle GPU traces.  Bind the report to the longest trace,
        # which is the actual four-shard generation run.
        "grpo_gpu": gpu_summary(max(gpu_files, key=lambda path: path.stat().st_size)) if gpu_files else {},
        "authority": {
            "grpo_execution_complete_sha256": sha256(grpo_root / "PERSISTENT_COMPILE_COMPLETE.json"),
            "grpo_receipts_sha256": sha256(grpo_root / "compile_persistent_v1/receipts.jsonl"),
            "sft_final_report_sha256": sha256(sft_eval / "final_eval/FINAL_REPORT.json"),
            "sft_bindings_sha256": sha256(sft_eval / "final_eval/ATTEMPT_RECEIPT_BINDINGS.jsonl"),
        },
    }
    json_path = output / "analysis.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def pct(x: float) -> str:
        return f"{100*x:.2f}%"

    lines = [
        "# GRPO vs SFT epoch-2：miniF2F-test pass@32 评估报告",
        "",
        "## 结论摘要",
        "",
        "本报告仅覆盖 miniF2F-test 的244题；每题固定生成32次。GRPO与SFT使用相同的vLLM采样合同；GRPO checkpoint-9500证明由4个常驻Pantograph worker逐采样校验，SFT沿用此前冻结的Pantograph收据。",
        "",
        "## Pass@k",
        "",
        "| k | GRPO（无偏估计） | SFT（无偏估计） | GRPO实际前缀 | SFT实际前缀 | 前缀差值 | 配对bootstrap 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in K_VALUES:
        key = str(k)
        gp = grpo["pass_at_k_actual_prefix"][key]
        sp = sft["pass_at_k_actual_prefix"][key]
        ci = paired[key]
        lines.append(
            f"| {k} | {pct(grpo['pass_at_k_unbiased_estimator'][key])} | "
            f"{pct(sft['pass_at_k_unbiased_estimator'][key])} | "
            f"{gp['count']}/244 ({pct(gp['rate'])}) | {sp['count']}/244 ({pct(sp['rate'])}) | "
            f"{100*ci['delta']:+.2f} pp | [{100*ci['bootstrap_95_low']:+.2f}, {100*ci['bootstrap_95_high']:+.2f}] pp |"
        )
    gq, sq = grpo["generation_quality"], sft["generation_quality"]
    gc, sc = grpo["compile_quality"], sft["compile_quality"]
    lines += [
        "",
        "## 生成与编译质量",
        "",
        "| 指标 | GRPO | SFT epoch-2 |",
        "|---|---:|---:|",
        f"| 编译成功尝试 | {grpo['successful_attempt_count']}/7808 ({pct(grpo['attempt_success_rate'])}) | {sft['successful_attempt_count']}/7808 ({pct(sft['attempt_success_rate'])}) |",
        f"| 题内精确重复尝试率 | {pct(gq['within_problem_exact_duplicate_attempt_rate'])} | {pct(sq['within_problem_exact_duplicate_attempt_rate'])} |",
        f"| 题内成对碰撞率 | {pct(gq['within_problem_pair_collision_rate'])} | {pct(sq['within_problem_pair_collision_rate'])} |",
        f"| 出现重复的题数 | {gq['problems_with_any_exact_duplicate_count']}/244 | {sq['problems_with_any_exact_duplicate_count']}/244 |",
        f"| 严格幻觉API尝试率 | {gc['strict_hallucinated_api_attempt_count']}/7808 ({pct(gc['strict_hallucinated_api_attempt_rate'])}) | {sc['strict_hallucinated_api_attempt_count']}/7808 ({pct(sc['strict_hallucinated_api_attempt_rate'])}) |",
        f"| import/namespace前缀污染 | {gq['import_namespace_prefix_contamination_count']} | {sq['import_namespace_prefix_contamination_count']} |",
        f"| 达到2048 token上限 | {gq['hit_max_new_tokens_count']} | {sq['hit_max_new_tokens_count']} |",
        "",
        "严格幻觉API只计Lean明确报告的unknown identifier/constant/declaration/tactic/field；错误投影、类型不匹配、策略失败等另列，不混入该指标。",
        "",
        "## 错误类型（尝试数）",
        "",
        "```json",
        json.dumps({"GRPO": gc["error_category_counts"], "SFT": sc["error_category_counts"]}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 资源",
        "",
        "```json",
        json.dumps(report["grpo_gpu"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 口径说明",
        "",
        "- 主要 pass@k 同时给出标准无偏估计和冻结样本顺序的实际前缀命中率；pass@32两者相同。",
        "- 重复生成率按同一道题32个规范化proof中的精确重复计算；另给成对碰撞率，避免单一高频模板被低估。",
        "- 所有含sorry/admit的结果均按失败处理。",
        "- 本版修复了旧直接编译回退遗漏imports/open前缀的问题；按题分组仅复用Mathlib加载，每个采样由namespace与到达哨兵隔离，哨兵缺失时自动单条复核。旧v1统计已作废。",
    ]
    md_path = output / "REPORT.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sums = output / "SHA256SUMS"
    sums.write_text(
        f"{sha256(json_path)}  analysis.json\n{sha256(md_path)}  REPORT.md\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "analysis": str(json_path), "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
