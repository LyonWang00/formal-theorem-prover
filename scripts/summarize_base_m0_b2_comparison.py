"""Summarize the real Base/M0/B2 comparison on fixed datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lean_prover.lean_training.evaluation.expanded_validation import (
    metric_summary,
    paired_analysis,
    read_jsonl,
    write_json_atomic,
)
if __package__:
    from scripts.summarize_b2_expanded_validation import generation_behavior
else:
    from summarize_b2_expanded_validation import generation_behavior


MODELS = ("Base", "M0", "B2")
STAGES = (
    ("Full500 Discovery", "full500"),
    ("Strict unseen 200", "strict_unseen_primary"),
    ("miniF2F-valid 64", "monitor_primary"),
    ("miniF2F-test 96", "benchmark"),
)


def evaluation_root(
    stage: str,
    model: str,
    output_root: Path,
    source_root: Path,
) -> Path:
    if stage != "benchmark" and model in ("M0", "B2"):
        return source_root / "evaluations" / stage / model
    return output_root / "evaluations" / stage / model


def load_model_result(path: Path) -> dict[str, Any]:
    attempts = read_jsonl(path / "attempts.jsonl")
    generations = read_jsonl(path / "generations.jsonl")
    runtime = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return {
        "metrics": metric_summary(attempts),
        "generation_behavior": generation_behavior(generations),
        "runtime": runtime,
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/base_m0_b2_comparison")
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("outputs/b2_expanded_validation")
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}
    private_attempts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label, stage in STAGES:
        results[stage] = {"label": label, "models": {}, "paired": {}}
        private_attempts[stage] = {}
        for model in MODELS:
            path = evaluation_root(stage, model, args.root, args.source_root)
            if not (path / "attempts.jsonl").is_file():
                raise FileNotFoundError(path / "attempts.jsonl")
            loaded = load_model_result(path)
            private_attempts[stage][model] = loaded.pop("attempts")
            loaded["source_path"] = str(path.resolve())
            loaded["reused_existing_result"] = (
                stage != "benchmark" and model in ("M0", "B2")
            )
            results[stage]["models"][model] = loaded
        for left, right in (("Base", "M0"), ("Base", "B2"), ("M0", "B2")):
            results[stage]["paired"][f"{left}_vs_{right}"] = paired_analysis(
                private_attempts[stage][left],
                private_attempts[stage][right],
            )

    summary = {
        "experiment": "base_m0_b2_comparison",
        "models": list(MODELS),
        "results": results,
        "comparison_contract": {
            "same_statements": True,
            "same_order": True,
            "same_prompt_and_tokenizer_contract": True,
            "same_generation_parameters": True,
            "same_seed": 20260730,
            "same_samples_per_statement": 4,
            "same_Lean_Pantograph_environment": True,
        },
        "training_performed": False,
    }
    write_json_atomic(args.root / "metrics.json", summary)

    table_rows = []
    for label, stage in STAGES:
        for model in MODELS:
            row = results[stage]["models"][model]["metrics"]
            table_rows.append(
                f"| {label} | {model} | {row['statement_count']} | "
                f"{row['pass_at_1']:.2%} | {row['pass_at_2']:.2%} | "
                f"{row['pass_at_4']:.2%} | {row['candidate_success_rate']:.2%} | "
                f"{row['solved_statement_count']} |"
            )

    delta_rows = []
    for label, stage in STAGES:
        for comparison in ("Base_vs_M0", "Base_vs_B2", "M0_vs_B2"):
            row = results[stage]["paired"][comparison]
            delta_rows.append(
                f"| {label} | {comparison.replace('_vs_', ' → ')} | "
                f"{row['delta_pass_at_4']:+.2%} | "
                f"[{row['pass_at_4_bootstrap_95_ci'][0]:+.2%}, "
                f"{row['pass_at_4_bootstrap_95_ci'][1]:+.2%}] | "
                f"{row['mcnemar_exact_two_sided_p']:.4f} |"
            )

    report = [
        "# Base、M0 与 B2 同协议测试报告",
        "",
        "本报告中的 Base 是 M0 训练前的原始 `Qwen/Qwen2.5-1.5B-Instruct`。",
        "所有新评估均使用 temperature=0.8、top_p=0.95、max_new_tokens=256、"
        "每题 4 个候选、seed=20260730，并由 Pantograph/Lean 真实验证。",
        "",
        "Full500、Strict unseen 和 miniF2F-valid 的 M0/B2 数字复用已完成的同协议结果；"
        "Base 为本次新增。miniF2F-test 过去没有结果，本次对 Base、M0、B2 三者全部重新运行。",
        "",
        "## 通过率对比",
        "",
        "| 数据集 | 模型 | 题数 | Pass@1 | Pass@2 | Pass@4 | 候选成功率 | 解题数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "## Pass@4 配对差异",
        "",
        "| 数据集 | 比较 | ΔPass@4 | Bootstrap 95% CI | McNemar p |",
        "|---|---|---:|---:|---:|",
        *delta_rows,
        "",
        "## 说明",
        "",
        "- Pass@k 的 success 必须同时满足 Lean 编译成功且不含 sorry/admit/axiom/timeout。",
        "- 不同数据集难度不同，不能横向用一个数据集的百分比替代另一个数据集。",
        "- 本次没有训练、没有更新 Proof Bank、没有创建 M1/M2，也没有进入第二轮。",
    ]
    (args.root / "comparison_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
