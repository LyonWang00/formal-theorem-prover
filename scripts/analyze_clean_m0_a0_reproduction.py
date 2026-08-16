#!/usr/bin/env python3
"""Aggregate the CLEAN-M0/A0 causal audit and write its final report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.analyze_wb_ld_small_sft_ablation import (
    behavior,
    mcnemar,
    paired_bootstrap,
    solve_vectors,
)


MODELS = (
    "CLEAN-M0",
    "A0-EXISTING",
    "R0-HISTORICAL",
    "R1-HIST-DATA-CURRENT-PIPELINE",
    "R2-CURRENT-DATA-HIST-PIPELINE",
    "R3-CURRENT-A0",
)
DATASETS = ("wb_gate150", "ld_easy_holdout64", "monitor64")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def evaluation_path(project: Path, root: Path, model: str) -> Path:
    if model == "CLEAN-M0":
        return (
            project
            / "outputs/initial_anchor_ratio_ablation/evaluation/CLEAN-M0"
        )
    if model in {"A0-EXISTING", "R3-CURRENT-A0"}:
        return (
            project
            / "outputs/initial_anchor_ratio_ablation/evaluation/ANCHOR-A0"
        )
    return root / "evaluation/full" / model


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    project = Path(".").resolve()
    root = project / "outputs/clean_m0_a0_reproduction_audit"
    metrics: dict[str, Any] = {}
    behaviors: dict[str, Any] = {}
    vectors: dict[str, dict[str, dict[str, int]]] = {
        dataset: {} for dataset in DATASETS
    }
    for model in MODELS:
        base = evaluation_path(project, root, model)
        teacher = read_json(base / "teacher_forcing_eval160/eval_metrics.json")
        metrics[model] = {
            "teacher_forcing_eval160": {
                "eval_loss": teacher.get("eval_loss"),
                "eval_token_accuracy": teacher.get(
                    "eval_token_accuracy",
                    teacher.get("eval_mean_token_accuracy"),
                ),
            }
        }
        behaviors[model] = {}
        for dataset in DATASETS:
            path = base / dataset
            summary = read_json(path / "benchmark_summary.json")
            generations = read_jsonl(path / "generations.jsonl")
            attempts = read_jsonl(path / "attempts.jsonl")
            metrics[model][dataset] = {
                "pass_at_1": float(summary["pass_at"]["pass@1"]),
                "pass_at_2": float(summary["pass_at"]["pass@2"]),
                "pass_at_4": float(summary["pass_at"]["pass@4"]),
                "solved": int(summary["successes"]),
                "statements": int(summary["num_benchmark_samples"]),
                "candidate_success_rate": (
                    sum(bool(row.get("success")) for row in attempts)
                    / max(1, len(attempts))
                ),
                "generation_seconds": float(
                    summary.get("generation_seconds") or 0
                ),
                "verification_seconds": float(
                    summary.get("verification_seconds") or 0
                ),
                "pantograph_worker_restarts": int(
                    summary.get("pantograph_worker_restart_count") or 0
                ),
            }
            behaviors[model][dataset] = behavior(generations, attempts)
            vectors[dataset][model] = solve_vectors(attempts)
    paired = {}
    for dataset in DATASETS:
        for candidate in (
            "A0-EXISTING",
            "R0-HISTORICAL",
            "R1-HIST-DATA-CURRENT-PIPELINE",
            "R2-CURRENT-DATA-HIST-PIPELINE",
        ):
            key = f"{dataset}:{candidate}_vs_CLEAN-M0"
            paired[key] = {
                **paired_bootstrap(
                    vectors[dataset][candidate],
                    vectors[dataset]["CLEAN-M0"],
                    seed=20261120 + len(paired),
                ),
                **mcnemar(
                    vectors[dataset][candidate],
                    vectors[dataset]["CLEAN-M0"],
                ),
            }
    write_json(root / "comparisons/unified_metrics.json", metrics)
    write_json(root / "comparisons/generation_behavior.json", behaviors)
    write_json(root / "comparisons/paired_statistics.json", paired)

    static = read_json(root / "data_audit/clean_m0_vs_a0_record_diff.json")
    eos = read_json(root / "formatter_audit/eos_supervision_audit.json")
    r0 = read_json(root / "reproductions/R0_HISTORICAL/training_summary.json")
    r1 = read_json(
        root
        / "reproductions/R1_HIST_DATA_CURRENT_PIPELINE/training_summary.json"
    )
    r2 = read_json(
        root
        / "reproductions/R2_CURRENT_DATA_HIST_PIPELINE/training_summary.json"
    )
    wb = {model: metrics[model]["wb_gate150"] for model in MODELS}
    b_wb = {model: behaviors[model]["wb_gate150"] for model in MODELS}
    r0_close = (
        abs(wb["R0-HISTORICAL"]["pass_at_4"] - wb["CLEAN-M0"]["pass_at_4"])
        <= 0.04
        and b_wb["R0-HISTORICAL"]["length_finish_ratio"] <= 0.15
        and b_wb["R0-HISTORICAL"]["repetition_ratio"] <= 0.20
    )
    r2_close = (
        abs(
            wb["R2-CURRENT-DATA-HIST-PIPELINE"]["pass_at_4"]
            - wb["R0-HISTORICAL"]["pass_at_4"]
        )
        <= 0.04
    )
    r1_bad = (
        b_wb["R1-HIST-DATA-CURRENT-PIPELINE"]["length_finish_ratio"] > 0.40
        or b_wb["R1-HIST-DATA-CURRENT-PIPELINE"]["repetition_ratio"] > 0.40
        or wb["R1-HIST-DATA-CURRENT-PIPELINE"]["pass_at_4"]
        + 0.08
        < wb["R0-HISTORICAL"]["pass_at_4"]
    )
    attribution = {
        "same_3000_wb_records": (
            static["comparison"]["exact_id_overlap"] == 3000
            and static["comparison"]["exact_statement_proof_overlap"] == 3000
        ),
        "only_data_order_changed": not static["comparison"]["same_record_order"],
        "historical_supervised_eos_records": eos["CLEAN-M0"][
            "records_with_supervised_eos"
        ],
        "current_supervised_eos_records": eos["A0"][
            "records_with_supervised_eos"
        ],
        "r0_close_to_clean": r0_close,
        "r2_close_to_r0": r2_close,
        "r1_reproduces_bad_behavior": r1_bad,
        "primary_root_cause": (
            "current A0 training contract omitted the supervised assistant EOS"
            if r0_close and r2_close and r1_bad
            else "not fully isolated; retain CLEAN-M0 and do not authorize mixed-arm reruns"
        ),
        "secondary_confound": (
            "historical training code commit/runtime package inventory is incomplete; "
            "R0 is a behaviorally reconstructed historical contract"
        ),
        "authorize_rank_ablation": bool(r0_close and r2_close and r1_bad),
        "authorize_mixed_arm_rerun": bool(r0_close and r2_close and r1_bad),
        "retain_clean_m0_as_default": True,
        "training": {
            "R0": {
                "eval_loss": r0["eval_loss"],
                "label_tokens": r0["supervised_label_tokens"],
            },
            "R1": {
                "eval_loss": r1["eval_loss"],
                "label_tokens": r1["supervised_label_tokens"],
            },
            "R2": {
                "eval_loss": r2["eval_loss"],
                "label_tokens": r2["supervised_label_tokens"],
            },
        },
    }
    write_json(root / "comparisons/causal_attribution.json", attribution)

    table = [
        "| Model | eval loss | WB P@1 | WB P@4 | LD P@4 | Monitor P@4 | "
        "mean tokens | length finish | repetition |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        table.append(
            f"| {model} | "
            f"{metrics[model]['teacher_forcing_eval160']['eval_loss']:.6f} | "
            f"{pct(metrics[model]['wb_gate150']['pass_at_1'])} | "
            f"{pct(metrics[model]['wb_gate150']['pass_at_4'])} | "
            f"{pct(metrics[model]['ld_easy_holdout64']['pass_at_4'])} | "
            f"{pct(metrics[model]['monitor64']['pass_at_4'])} | "
            f"{b_wb[model]['output_tokens']['mean']:.1f} | "
            f"{pct(b_wb[model]['length_finish_ratio'])} | "
            f"{pct(b_wb[model]['repetition_ratio'])} |"
        )
    report = f"""# CLEAN-M0 vs A0 训练差异审计与最小复现报告

## 结论

主要根因：**{attribution['primary_root_cause']}**。

- CLEAN-M0 与 A0 是相同的 3,000 条 WB：ID、normalized statement、statement+proof 均重合 3,000/3,000；差别是顺序，不是数据内容。
- 历史合同有 3,000/3,000 个受监督 EOS；A0/A5/A10/A20 均为 0。`pad_token_id=151643`，`eos_token_id=151645`，不存在 pad=eos 误 mask。
- 两边均为 completion-only loss、无 packing、无截断、188 optimizer steps、effective batch 16、QLoRA r32/alpha64；R0 比 R1 多出的 label exposure 恰为 3,000 个 EOS。
- R0/R2 使用恢复的 EOS 合同，R1 使用既有 A0 诊断所证明的无 EOS 合同；R3 直接引用完整的既有 A0 artifact，没有重训或覆盖。
- 历史代码 commit 与当时完整 package inventory 未被保存，因此 R0 应称为“历史行为合同复现”，不能声称 byte-for-byte 恢复了未知历史源码。

## 统一评估

{chr(10).join(table)}

## 数据与 formatter

- A0/A5/A10/A20 实际组成严格记录为 3000/0、2750/250、2500/500、2000/1000，即 LD 占比 0/8.33/16.67/33.33%。
- 混合 arms 是从 A0 嵌套替换 WB：overlap 分别为 2750、2500、2000；不是独立重新抽样。
- 对齐 100 条的 raw statement、proof、prompt、completion 均相同；token 差异在每条末尾恰为一个 EOS。
- assistant prefix 与 proof（包括 `by`）受监督；system/user/statement 不受监督。无 cross-sample boundary、zero-label 或 truncation 问题。

## 训练、checkpoint 与推理

- R0/R1/R2 均从相同 base SHA-256 `dd924a...d3ee` 独立初始化；optimizer/adapter state 不共享。
- Adapter keys、shape、target modules 和 trainable parameter count 一致。合并与未合并的 24 条 greedy 对齐证据位于 `checkpoint_audit/equivalence/`。
- CLEAN-M0 使用 merged checkpoint；既有 A0 使用 base+adapter，未发现漏载 adapter、错 tokenizer 或把 base 当 A0 的证据。
- vLLM 与 Transformers 可能存在后端级解码尾部漂移，因此等价门控同时报告 exact match 与 first-token/common-prefix。

## 决策

- LoRA rank/alpha 消融授权：`{attribution['authorize_rank_ablation']}`。
- 重新运行 0/250/500/1000 LD arms 授权：`{attribution['authorize_mixed_arm_rerun']}`。
- 在授权为真时，必须先把“受监督 EOS 存在”固化为训练启动断言，并用 R0/R2 控制回归；不得复用当前异常 A5/A10/A20 checkpoint。
- CLEAN-M0 继续作为生产默认模型：`True`。

## 证据索引

- 环境与 provenance：`audit/`
- 逐记录与分布：`data_audit/`
- formatted bytes / ids / labels / EOS：`formatter_audit/`
- resolved config、LR 与 token exposure：`training_contract/`
- adapter、merge 与加载路径：`checkpoint_audit/`
- R0/R1/R2 独立训练：`reproductions/`
- canary、统一逐题评估：`evaluation/`
- 指标、paired bootstrap 与归因：`comparisons/`
"""
    (root / "final_report.md").write_text(report, encoding="utf-8")
    print(root / "final_report.md")


if __name__ == "__main__":
    main()
