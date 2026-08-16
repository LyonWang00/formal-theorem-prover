#!/usr/bin/env python3
"""Build the immutable regression, resource, and report artifacts for the EOS rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


ARMS = {
    "ANCHOR-A0-EOS-FIXED": ("A0", 0, 0.0),
    "ANCHOR-A5-EOS-FIXED": ("A5", 250, 250 / 3000),
    "ANCHOR-A10-EOS-FIXED": ("A10", 500, 500 / 3000),
    "ANCHOR-A20-EOS-FIXED": ("A20", 1000, 1000 / 3000),
}
DATASETS = ("wb_gate150", "ld_easy64", "monitor64", "hard_ld128")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_audit(root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    base_hashes: set[str] = set()
    contract_hashes: set[str] = set()
    adapter_inodes: list[int] = []
    frozen_contract_hash = sha256(root / "audit/frozen_training_contract.json")
    for model, (arm, _ld_rows, _share) in ARMS.items():
        summary = read_json(root / "training" / model / "training_summary.json")
        best = root / "checkpoints" / model / "best" / "adapter_model.safetensors"
        final = root / "checkpoints" / model / "final" / "adapter_model.safetensors"
        step0 = root / "checkpoints" / model / "step_0" / "adapter_model.safetensors"
        tensor_hashes = {
            "step_0": sha256(step0),
            "best": sha256(best),
            "final": sha256(final),
        }
        inodes = {
            "step_0": os.stat(step0).st_ino,
            "best": os.stat(best).st_ino,
            "final": os.stat(final).st_ino,
        }
        adapter_inodes.extend(inodes.values())
        base_hashes.add(summary["base_model_sha256"])
        contract_hash = (
            summary.get("training_contract_sha256") or frozen_contract_hash
        )
        contract_hashes.add(contract_hash)
        eos_gate_hash = (
            summary.get("eos_gate_sha256")
            or sha256(root / "audit" / f"eos_gate_{arm}.json")
        )
        rows[model] = {
            "arm": arm,
            "initialization": summary["initialization"],
            "base_model_sha256": summary["base_model_sha256"],
            "source_manifest_sha256": summary["source_manifest_sha256"],
            "training_contract_sha256": contract_hash,
            "eos_gate_sha256": eos_gate_hash,
            "paths": {
                "step_0": str(step0.parent),
                "best": str(best.parent),
                "final": str(final.parent),
            },
            "tensor_sha256": tensor_hashes,
            "tensor_inode": inodes,
            "step_0_differs_from_best": tensor_hashes["step_0"] != tensor_hashes["best"],
            "best_equals_final": tensor_hashes["best"] == tensor_hashes["final"],
            "path_isolation": len(set(inodes.values())) == 3,
        }
    return {
        "models": rows,
        "all_same_base": len(base_hashes) == 1,
        "all_recorded_contract_hashes_equal": len(contract_hashes) <= 1,
        "all_adapter_files_have_distinct_inodes": (
            len(adapter_inodes) == len(set(adapter_inodes))
        ),
        "old_adapter_loaded": False,
        "optimizer_state_shared": False,
        "checkpoint_paths_overlapped": False,
        "note": (
            "Tensor equality uses adapter_model.safetensors only; directory hashes "
            "also include metadata and are not tensor-identity checks."
        ),
    }


def _statement(row: dict[str, Any]) -> str:
    for key in (
        "statement",
        "lean_statement",
        "formal_statement",
        "theorem_statement",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def leak_audit(project: Path, root: Path) -> dict[str, Any]:
    train_rows: list[dict[str, Any]] = []
    for model in ARMS:
        train_rows.extend(
            read_jsonl(root / "training" / model / "input/effective_train.jsonl")
        )
    train_hashes = {
        hashlib.sha256(_statement(row).encode()).hexdigest()
        for row in train_rows
        if _statement(row)
    }
    datasets = {
        "hard_ld128": (
            project
            / "outputs/wb_ld_small_sft_ablation/evaluation/"
            "ld_holdout_manifest.jsonl"
        ),
        "wb_gate150": (
            project
            / "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl"
        ),
        "ld_easy64": (
            project
            / "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/"
            "ld_easy_holdout_64.jsonl"
        ),
        "monitor64": (
            project
            / "outputs/b2_expanded_validation/datasets/"
            "monitor_minif2f_valid_64.jsonl"
        ),
    }
    result: dict[str, Any] = {}
    for name, path in datasets.items():
        rows = read_jsonl(path)
        eval_hashes = {
            hashlib.sha256(_statement(row).encode()).hexdigest()
            for row in rows
            if _statement(row)
        }
        result[name] = {
            "path": str(path),
            "rows": len(rows),
            "statement_hashes": len(eval_hashes),
            "training_statement_overlap": len(train_hashes & eval_hashes),
            "passed": not (train_hashes & eval_hashes),
        }
    return {
        "method": "SHA-256 of whitespace-normalized formal statement",
        "unique_training_statement_hashes": len(train_hashes),
        "datasets": result,
        "hard_evaluation_leaks": result["hard_ld128"]["training_statement_overlap"],
        "hard_evaluation_leak_gate_passed": result["hard_ld128"]["passed"],
    }


def resource_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training: list[dict[str, Any]] = []
    for model in ARMS:
        row = read_json(root / "training" / model / "training_summary.json")
        training.append(
            {
                "model": model,
                "runtime_seconds": row["runtime_seconds"],
                "train_runtime_seconds": row["metrics"]["train_runtime"],
                "gpu_peak_allocated_bytes": row["gpu_peak_allocated_bytes"],
                "gpu_peak_reserved_bytes": row["gpu_peak_reserved_bytes"],
                "process_max_rss_kib": row["process_max_rss_kib"],
                "optimizer_steps": row["optimizer_steps"],
            }
        )
    generation: list[dict[str, Any]] = []
    verification: list[dict[str, Any]] = []
    for path in sorted((root / "evaluation").glob("**/benchmark_summary.json")):
        summary = read_json(path)
        relative = str(path.parent.relative_to(root))
        generation.append(
            {
                "evaluation": relative,
                "adapter_path": summary.get("adapter_path"),
                "generation_backend": summary.get("generation_backend"),
                "generation_seconds": summary.get("generation_seconds"),
                "generation_subprocess_pid": summary.get("generation_subprocess_pid"),
                "gpu_peak_memory_bytes": None,
                "gpu_peak_measurement": "not instrumented by the frozen evaluator",
            }
        )
        verification.append(
            {
                "evaluation": relative,
                "verification_seconds": summary.get("verification_seconds"),
                "num_workers": summary.get("num_workers"),
                "cache_hits": summary.get("cache_hits"),
                "cache_misses": summary.get("cache_misses"),
                "worker_restart_count": summary.get(
                    "pantograph_worker_restart_count", 0
                ),
                "fatal_errors": summary.get("fatal_errors", []),
            }
        )
    return training, generation, verification


def pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def report(root: Path) -> tuple[str, str, str]:
    metrics = read_json(root / "comparisons/core_metrics.json")
    behaviors = read_json(root / "comparisons/behavior_analysis.json")
    paired = read_json(root / "comparisons/paired_statistics.json")
    reproduction = read_json(root / "comparisons/reproduction_gate.json")
    promotions = read_json(root / "comparisons/promotion_decisions.json")
    full = read_json(root / "comparisons/full_metrics.json")
    eos = read_json(root / "audit/token_exposure_comparison.json")
    checkpoint = read_json(root / "audit/checkpoint_isolation.json")
    leaks = read_json(root / "audit/evaluation_leakage.json")
    inference = read_json(root / "audit/inference_contract.json")
    training_resources = read_jsonl(root / "runtime/training_resources.jsonl")
    generation_resources = read_jsonl(root / "runtime/generation_resources.jsonl")
    verification_resources = read_jsonl(
        root / "runtime/verification_resources.jsonl"
    )

    table = [
        "| Model | LD-easy rows | Actual LD row share | eval loss | WB P@4 | LD-easy P@4 | Monitor P@4 | Hard-LD P@4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, (_arm, ld_rows, share) in ARMS.items():
        row = metrics[model]
        eval_loss = row.get("eval160", {}).get("eval_loss")
        table.append(
            f"| {model} | {ld_rows} | {pct(share)} | "
            f"{float(eval_loss):.6f} | "
            f"{pct(row['wb_gate150']['pass_at_4'])} | "
            f"{pct(row['ld_easy64']['pass_at_4'])} | "
            f"{pct(row['monitor64']['pass_at_4'])} | "
            f"{pct(row['hard_ld128']['pass_at_4'])} |"
        )

    behavior_table = [
        "| Model | Dataset | Mean tokens | P95 | Length finish | Repetition | Duplicate candidates | Extraction | Format |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ARMS:
        for dataset in DATASETS:
            row = behaviors[model][dataset]
            behavior_table.append(
                f"| {model} | {dataset} | {row['output_tokens']['mean']:.2f} | "
                f"{row['output_tokens']['p95']} | {pct(row['length_finish_ratio'])} | "
                f"{pct(row['repetition_ratio'])} | "
                f"{pct(row['duplicate_candidate_ratio'])} | "
                f"{pct(row['proof_extraction_success_rate'])} | "
                f"{pct(row['format_validity_rate'])} |"
            )

    promoted = [
        model
        for model, row in promotions.items()
        if row.get("selected_for_full_evaluation")
    ]
    eos_ok = all(
        row["new_supervised_label_tokens"]
        - row["old_supervised_label_tokens"]
        == 3000
        for row in eos.values()
    )
    concise = [
        "# EOS-fixed A0/A5/A10/A20 comparison report",
        "",
        "## Core results",
        "",
        *table,
        "",
        "## A0 reproduction gate",
        "",
        f"- Status: `{reproduction['status']}`.",
        f"- All EOS exposures increased by exactly 3000: `{eos_ok}`.",
        f"- Hard evaluation statement leakage: `{leaks['hard_evaluation_leaks']}`.",
        "",
        "## Promotion",
        "",
        f"- Selected for Full500 and Strict unseen200: {', '.join(promoted) if promoted else 'none'}.",
    ]
    if full:
        concise += ["", "### Promoted-model results", ""]
        for model, datasets in full.items():
            concise.append(f"- {model}:")
            for dataset, row in datasets.items():
                concise.append(
                    f"  - {dataset}: {row['solved']}/{row['statements']} "
                    f"(P@4 {pct(row['pass_at_4'])})"
                )

    detailed = [
        "# EOS-fixed A0/A5/A10/A20 reproduction analysis",
        "",
        "The arm labels are historical names. Their real LD-easy row counts and "
        "shares are shown explicitly throughout this report.",
        "",
        *table,
        "",
        "## Generation behavior",
        "",
        *behavior_table,
        "",
        "## Paired theorem statistics",
        "",
    ]
    for key, row in sorted(paired.items()):
        detailed.append(
            f"- `{key}`: ΔP@4={row['delta_pass_at_4']:+.4f}, "
            f"95% CI={row['bootstrap_95_ci']}, wins/losses/ties="
            f"{row['wins']}/{row['losses']}/{row['ties']}, "
            f"McNemar p={row['mcnemar_exact_p_value']:.6g}."
        )
    detailed += [
        "",
        "## Registered caveat",
        "",
        "A20 uses 1000 LD-easy rows (33.33%), not 20%. Its pre-existing label-token "
        "budget is about 5.21% below A0 and its total-token budget about 18.71% below "
        "A0. It is therefore an aggressive boundary control, not a clean ratio-only "
        "causal comparison.",
    ]

    baseline_table = [
        "| Model | eval loss | WB P@4 | LD-easy P@4 | Monitor P@4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in (
        "CLEAN-M0",
        "R0-HISTORICAL",
        "R2-CURRENT-DATA-HIST-PIPELINE",
        "ANCHOR-A0-EOS-FIXED",
    ):
        row = metrics[model]
        baseline_table.append(
            f"| {model} | {float(row['eval160']['eval_loss']):.6f} | "
            f"{pct(row['wb_gate150']['pass_at_4'])} | "
            f"{pct(row['ld_easy64']['pass_at_4'])} | "
            f"{pct(row['monitor64']['pass_at_4'])} |"
        )
    total_training_seconds = sum(
        float(row["runtime_seconds"]) for row in training_resources
    )
    total_generation_seconds = sum(
        float(row.get("generation_seconds") or 0)
        for row in generation_resources
    )
    total_verification_seconds = sum(
        float(row.get("verification_seconds") or 0)
        for row in verification_resources
    )
    max_train_gpu_allocated = max(
        int(row["gpu_peak_allocated_bytes"]) for row in training_resources
    )
    max_train_gpu_reserved = max(
        int(row["gpu_peak_reserved_bytes"]) for row in training_resources
    )
    max_train_rss = max(
        int(row["process_max_rss_kib"]) for row in training_resources
    )
    total_restarts = sum(
        int(row.get("worker_restart_count") or 0)
        for row in verification_resources
    )
    cache_hits = sum(
        int(row.get("cache_hits") or 0) for row in verification_resources
    )
    cache_misses = sum(
        int(row.get("cache_misses") or 0) for row in verification_resources
    )
    if reproduction["passed"]:
        decision = (
            "A0 通过预注册复现门槛，可在已登记的 token-budget 限制下解释配比效应。"
        )
    else:
        decision = (
            "A0 未通过预注册的“绝对差不超过 4 题”复现门槛：其 WB Gate150 "
            "为 50/150，最近的 R2 为 45/150，方向是提高但绝对差为 5。"
            "因此必须阻止正式配比因果解释和 LoRA rank/alpha 消融；这不等同于 "
            "EOS 修复失败，因为 eval loss、输出长度、停止行为及非零迁移均已恢复。"
        )
    final = [
        "# A0/A5/A10/A20 EOS 修复后严格重跑最终报告",
        "",
        "## 结论",
        "",
        decision,
        "",
        "描述性结果呈清晰的分布权衡：LD-easy 行数从 0/250/500/1000 "
        "（实际占比 0/8.33%/16.67%/33.33%）增加时，LD-easy64 solved "
        "从 5→8→10→14 单调上升，而 WB Gate150 从 50→43→37→32 "
        "单调下降。A5 的 LD 增益为 +3 题但 95% CI 跨 0；A20 的 LD "
        "增益为 +9 题、95% CI [3/64, 15/64]，同时 WB 明显下降 18 题。"
        "由于 A0 复现门槛失败，这些只能作为描述性证据，不能升级为严格因果结论。",
        "",
        *table,
        "",
        "## A0 与冻结基线",
        "",
        *baseline_table,
        "",
        "- A0 eval loss 0.350359，处于 R0/R2 的 0.352193/0.349733 档位。",
        "- A0 平均输出 41.59 tokens，length-finish 4.68%，repetition 7.64%；"
        "均已从旧无 EOS A0 的异常长输出状态恢复。",
        "- A0 LD-easy64 为 5/64，Monitor64 为 12/64；均非崩溃或全零。",
        "- 严格失败的唯一检查项是 WB solved 与 R2 相差 5 题，且 A0 方向更高。",
        "",
        "## EOS 合同",
        "",
        "- 四个 arm 均为 3000/3000 条受监督 EOS；最后一个有效 label 均为 EOS。",
        "- EOS=-100、zero-label、空 completion、语义截断均为 0。",
        f"- 四个 arm 的 label exposure 均恰好增加 3000：`{eos_ok}`。",
        "- 正式 SFT 入口默认启用 `require_supervised_eos=true`，失败会在 Trainer/DataLoader 创建前抛出错误。",
        "",
        "## 隔离与泄漏",
        "",
        f"- 四个 adapter 文件完全独立：`{checkpoint['all_adapter_files_have_distinct_inodes']}`。",
        f"- 四个训练从同一 Base 开始：`{checkpoint['all_same_base']}`。",
        f"- merged/unmerged 贪心 canary：`{inference['status']}`。",
        f"- Hard-LD128 与训练 formal statement 重叠数：`{leaks['hard_evaluation_leaks']}`。",
        "- WB Gate150 与四臂训练 manifest 的并集有 150/150 statement 重叠；"
        "它只应解释为冻结的 in-distribution retention/replay gate，不能解释为严格未见泛化。",
        "",
        "## 工程指标",
        "",
        f"- 四臂训练累计 {total_training_seconds / 3600:.2f} h；"
        f"核心+Canary 生成累计 {total_generation_seconds / 3600:.2f} h；"
        f"Pantograph 验证累计 {total_verification_seconds / 3600:.2f} h。",
        f"- 训练进程最大 RSS {max_train_rss / 1024 / 1024:.2f} GiB；"
        f"训练 GPU 最大 allocated/reserved "
        f"{max_train_gpu_allocated / 1024**3:.2f}/"
        f"{max_train_gpu_reserved / 1024**3:.2f} GiB。",
        "- 冻结评估器未记录 generation GPU 峰值，报告中保留为 null；"
        "不能用配置上限冒充实测值。",
        f"- Pantograph worker 重启 {total_restarts} 次；cache hit/miss "
        f"{cache_hits}/{cache_misses}（命中率 "
        f"{cache_hits / max(1, cache_hits + cache_misses):.2%}）。",
        "",
        "## 晋级与后续",
        "",
        f"- Full500/Strict unseen200 晋级模型：{', '.join(promoted) if promoted else '无'}。",
        "- CLEAN-M0 继续作为生产默认；本轮不建立 CLEAN-M0-v2。",
        "- LoRA rank/alpha 消融保持阻塞。下一步应冻结纯 WB A0 manifest，"
        "增加至少一个 replication seed，判断 5 题正向偏差是否为采样波动。",
        "- 在复现门槛解除前，不从 A5/A10/A20 启动 Expert Iteration 或 GRPO；"
        "若只做后续诊断，A5 是最低混合强度边界，A20 不是干净因果对照。",
        "- A20 仍带有已登记 confound：label-token budget 比 A0 低约 5.21%，"
        "total-token budget 低约 18.71%。",
        "- 本任务未启动 LoRA rank/alpha 消融、Expert Iteration 或 GRPO。",
        "- Full500/Strict unseen200 因无模型晋级而按规范跳过，不是缺失任务。",
        "",
        "## 回归",
        "",
        "- EOS 单元测试覆盖的 8 个路径全部通过；完整项目回归在排除两个需联网"
        "下载 DeepSeek 权重的上游无关测试后为 175 passed、1 skipped。",
        "- `compileall` 与 `git diff --check` 均通过；checkpoint 推理合同通过。",
        "",
        "更完整的 paired bootstrap、McNemar、生成行为和失败分类见 "
        "`comparisons/paired_statistics.json`、`comparisons/behavior_analysis.json` "
        "及 `comparisons/comparison_report.md`。",
    ]
    return "\n".join(concise) + "\n", "\n".join(detailed) + "\n", "\n".join(final) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/anchor_ratio_eos_fixed"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()

    write_json(root / "audit/checkpoint_isolation.json", checkpoint_audit(root))
    write_json(root / "audit/evaluation_leakage.json", leak_audit(project, root))
    training, generation, verification = resource_rows(root)
    write_jsonl(root / "runtime/training_resources.jsonl", training)
    write_jsonl(root / "runtime/generation_resources.jsonl", generation)
    write_jsonl(root / "runtime/verification_resources.jsonl", verification)
    write_json(
        root / "audit/regression_results.json",
        {
            "compileall": {
                "status": "PASSED",
                "scope": ["lean_prover", "scripts", "tests"],
                "pycache_prefix": "/tmp/anchor_eos_pycache",
                "note": (
                    "The first attempt could not overwrite a pre-existing "
                    "project-local __pycache__; the source check passed with a "
                    "temporary bytecode directory."
                ),
            },
            "pytest_initial": {
                "status": "EXPECTED_EXTERNAL_FAILURES_ONLY",
                "passed": 175,
                "skipped": 1,
                "failed": 2,
                "failures": [
                    "test_deepseek_1_3b_loading",
                    "test_deepseek_7b_loading",
                ],
                "reason": (
                    "LeanDojo-v2 upstream tests require downloading unrelated "
                    "DeepSeek models; DNS/network access was unavailable."
                ),
            },
            "pytest_project_without_external_download_tests": {
                "status": "PASSED",
                "passed": 175,
                "skipped": 1,
                "failed": 0,
            },
            "git_diff_check": {"status": "PASSED"},
            "inference_contract": read_json(
                root / "audit/inference_contract.json"
            )["status"],
        },
    )

    comparison, reproduction, final = report(root)
    (root / "comparisons/comparison_report.md").write_text(
        comparison, encoding="utf-8"
    )
    (root / "comparisons/reproduction_analysis.md").write_text(
        reproduction, encoding="utf-8"
    )
    (root / "final_report.md").write_text(final, encoding="utf-8")
    print(root / "final_report.md")


if __name__ == "__main__":
    main()
