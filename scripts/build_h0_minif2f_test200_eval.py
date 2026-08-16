#!/usr/bin/env python3
"""Build and summarize the fixed H0 miniF2F-test200 pass@8 evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id") or row.get("statement_id") or "")


def normalized_statement(row: dict[str, Any]) -> str:
    statement = str(row.get("lean_statement") or row.get("formal_statement") or "")
    return " ".join(statement.replace(":= by sorry", "").replace(":= sorry", "").split())


def normalized_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(normalized_statement(row).encode("utf-8")).hexdigest()


def proof_free(row: dict[str, Any]) -> bool:
    prompt = str(row.get("prompt") or "")
    marker = "### Lean proof"
    if marker not in prompt:
        return False
    return not prompt.split(marker, 1)[1].strip()


def select_supplement(args: argparse.Namespace) -> None:
    all_rows = read_jsonl(args.all_normalized)
    selected = read_jsonl(args.initial_sample)
    selected_ids = {record_id(row) for row in selected}
    candidates = [row for row in all_rows if record_id(row) not in selected_ids]
    if len(candidates) < args.count:
        raise ValueError(f"only {len(candidates)} unused normalized candidates")
    rows = candidates[: args.count]
    write_jsonl(args.output, rows)
    write_json(
        args.output.with_suffix(".audit.json"),
        {
            "status": "SUPPLEMENT_SELECTED",
            "source_rows": len(all_rows),
            "initial_rows": len(selected),
            "unused_candidates": len(candidates),
            "selected_rows": len(rows),
            "selected_ids": [record_id(row) for row in rows],
            "output_sha256": file_sha256(args.output),
        },
    )


def select_subset(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    if len(rows) < args.count:
        raise ValueError(f"requested {args.count} rows from only {len(rows)}")
    write_jsonl(args.output, rows[: args.count])


def finalize_manifest(args: argparse.Namespace) -> None:
    initial_requested = read_jsonl(args.initial_sample)
    initial_verified = read_jsonl(args.initial_verified)
    quarantined = read_jsonl(args.initial_quarantined)
    supplement_verified = read_jsonl(args.supplement_verified)
    verified_by_id = {record_id(row): row for row in initial_verified}
    ordered = [
        verified_by_id[record_id(row)]
        for row in initial_requested
        if record_id(row) in verified_by_id
    ]
    needed = args.expected - len(ordered)
    if needed < 0 or len(supplement_verified) < needed:
        raise ValueError(
            f"cannot finalize {args.expected}: initial={len(ordered)}, "
            f"supplement={len(supplement_verified)}, needed={needed}"
        )
    replacements = supplement_verified[:needed]
    final_rows = [
        {**row, "selection_origin": "initial_fixed_seed_sample"} for row in ordered
    ] + [
        {**row, "selection_origin": "statement_gate_replacement"}
        for row in replacements
    ]
    ids = [record_id(row) for row in final_rows]
    hashes = [normalized_hash(row) for row in final_rows]
    if len(final_rows) != args.expected:
        raise ValueError(f"expected {args.expected}, found {len(final_rows)}")
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("record or normalized-statement duplicate in final sample")
    if any(row.get("statement_verified") is not True for row in final_rows):
        raise ValueError("final sample contains an unattested statement")
    if any(not proof_free(row) for row in final_rows):
        raise ValueError("final sample contains a non-proof-free prompt")

    prior = read_jsonl(args.prior_benchmark)
    prior_ids = {record_id(row) for row in prior}
    prior_hashes = {normalized_hash(row) for row in prior}
    h0_train = read_jsonl(args.h0_training)
    h0_ids = {record_id(row) for row in h0_train}
    h0_hashes = {normalized_hash(row) for row in h0_train}
    output = args.output
    write_jsonl(output, final_rows)
    audit = {
        "schema_version": "h0-minif2f-test200-selection-v1",
        "status": "READY_FOR_EVALUATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "miniF2F test split",
        "selection_seed": 20260805,
        "requested_rows": args.expected,
        "initial_requested_rows": len(initial_requested),
        "initial_statement_verified": len(initial_verified),
        "initial_quarantined": len(quarantined),
        "replacement_rows": len(replacements),
        "replacement_ids": [record_id(row) for row in replacements],
        "quarantined_ids": [record_id(row) for row in quarantined],
        "final_rows": len(final_rows),
        "record_duplicates": len(ids) - len(set(ids)),
        "normalized_statement_duplicates": len(hashes) - len(set(hashes)),
        "statement_verified_rows": sum(row.get("statement_verified") is True for row in final_rows),
        "proof_free_prompt_rows": sum(proof_free(row) for row in final_rows),
        "h0_training_overlap": {
            "record_ids": sorted(set(ids) & h0_ids),
            "normalized_statement_hashes": sorted(set(hashes) & h0_hashes),
        },
        "prior_minif2f_test96_overlap": {
            "record_id_count": len(set(ids) & prior_ids),
            "normalized_statement_count": len(set(hashes) & prior_hashes),
            "record_ids": sorted(set(ids) & prior_ids),
        },
        "environment": {
            "lean_commit": "f72c35b3f637c8c6571d353742168ab66cc22c00",
            "mathlib_commit": "5e932f97dd25535344f80f9dd8da3aab83df0fe6",
            "environment_hash": "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a",
        },
        "manifest_sha256": file_sha256(output),
    }
    if audit["h0_training_overlap"]["record_ids"] or audit["h0_training_overlap"]["normalized_statement_hashes"]:
        raise ValueError("miniF2F sample overlaps H0 training data")
    write_json(args.audit, audit)


def pass_at_k(attempts: list[dict[str, Any]], k: int) -> tuple[int, float]:
    by_problem: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        by_problem[str(row["problem_id"])].append(row)
    solved = sum(
        any(bool(row.get("success")) for row in rows if int(row["attempt_index"]) < k)
        for rows in by_problem.values()
    )
    return solved, solved / max(1, len(by_problem))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (rate + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * (
            rate * (1.0 - rate) / total
            + z_squared / (4.0 * total * total)
        )
        ** 0.5
        / denominator
    )
    return [center - half_width, center + half_width]


def finalize_report(args: argparse.Namespace) -> None:
    manifest = read_jsonl(args.manifest)
    attempts = read_jsonl(args.attempts)
    verifications = read_jsonl(args.verifications)
    counts = Counter(str(row["problem_id"]) for row in attempts)
    if len(manifest) != 200 or len(attempts) != 1600 or set(counts.values()) != {8}:
        raise ValueError(
            f"incomplete pass@8 evaluation: manifest={len(manifest)}, "
            f"attempts={len(attempts)}, per_problem={sorted(set(counts.values()))}"
        )
    success_candidates = sum(bool(row.get("success")) for row in attempts)
    pass_values = {k: pass_at_k(attempts, k) for k in (1, 2, 4, 8)}
    failure_distribution = Counter(
        str(row.get("error_type") or row.get("status") or "unknown")
        for row in verifications
        if not bool(row.get("verified"))
    )
    success_count_by_problem = Counter()
    for row in attempts:
        success_count_by_problem[str(row["problem_id"])] += int(bool(row.get("success")))
    success_histogram = Counter(success_count_by_problem.values())
    summary = {
        "schema_version": "h0-minif2f-test200-pass8-v1",
        "status": "COMPLETED",
        "model": "H0-No-Hard-MERGED",
        "statement_count": 200,
        "candidate_count": 1600,
        "verified_candidate_count": success_candidates,
        "candidate_success_rate": success_candidates / 1600,
        "pass_at": {
            f"pass@{k}": {
                "solved": solved,
                "rate": rate,
                "wilson_95_ci": wilson_interval(solved, 200),
            }
            for k, (solved, rate) in pass_values.items()
        },
        "success_count_histogram": {
            f"{index}_of_8": success_histogram.get(index, 0) for index in range(9)
        },
        "failure_distribution": dict(sorted(failure_distribution.items())),
        "manifest_sha256": file_sha256(args.manifest),
        "attempts_sha256": file_sha256(args.attempts),
        "verifications_sha256": file_sha256(args.verifications),
        "generation": {
            "backend": "transformers",
            "checkpoint_type": "merged",
            "dtype": "bfloat16",
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 20,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "max_new_tokens": 256,
            "candidates_per_statement": 8,
            "seed": 20260805,
        },
        "no_training_started": True,
    }
    write_json(args.summary, summary)
    lines = [
        "# H0 miniF2F-test200 pass@8 评估报告",
        "",
        "状态：**COMPLETED**",
        "",
        "- 数据：miniF2F test split 固定抽样 200 题",
        "- 模型：H0-No-Hard merged checkpoint",
        "- 候选：每题 8 个，共 1,600 个",
        f"- 编译成功候选：{success_candidates} / 1600（{success_candidates / 1600:.2%}）",
        "",
        "## Pass@k",
        "",
        "| 指标 | solved | rate | Wilson 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for k in (1, 2, 4, 8):
        solved, rate = pass_values[k]
        low, high = wilson_interval(solved, 200)
        lines.append(
            f"| pass@{k} | {solved}/200 | {rate:.2%} | {low:.2%}–{high:.2%} |"
        )
    lines += [
        "",
        "## 运行合同",
        "",
        "`H0-No-Hard-MERGED + Transformers + BF16 + frozen generation parameters`；仅将候选数扩展为 8。",
        "",
        "本次未启动 Trainer，也未修改模型或训练数据。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    supplement = sub.add_parser("select-supplement")
    supplement.add_argument("--all-normalized", type=Path, required=True)
    supplement.add_argument("--initial-sample", type=Path, required=True)
    supplement.add_argument("--output", type=Path, required=True)
    supplement.add_argument("--count", type=int, default=10)
    supplement.set_defaults(func=select_supplement)

    subset = sub.add_parser("select-subset")
    subset.add_argument("--input", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--count", type=int, required=True)
    subset.set_defaults(func=select_subset)

    finalize = sub.add_parser("finalize-manifest")
    finalize.add_argument("--initial-sample", type=Path, required=True)
    finalize.add_argument("--initial-verified", type=Path, required=True)
    finalize.add_argument("--initial-quarantined", type=Path, required=True)
    finalize.add_argument("--supplement-verified", type=Path, required=True)
    finalize.add_argument("--prior-benchmark", type=Path, required=True)
    finalize.add_argument("--h0-training", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--audit", type=Path, required=True)
    finalize.add_argument("--expected", type=int, default=200)
    finalize.set_defaults(func=finalize_manifest)

    report = sub.add_parser("finalize-report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--attempts", type=Path, required=True)
    report.add_argument("--verifications", type=Path, required=True)
    report.add_argument("--summary", type=Path, required=True)
    report.add_argument("--report", type=Path, required=True)
    report.set_defaults(func=finalize_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
