#!/usr/bin/env python3
"""Render the frozen length-ablation results and a conservative policy decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = (
    "M0-ZERO",
    "MIX-A-WB100",
    "MIX-B-LD25",
    "MIX-C-LD50",
    "MIX-E-LD100",
)
LENGTHS = (256, 512, 1024)
DATASETS = ("wb", "ld")
COMPARISONS = ("L512_vs_L256", "L1024_vs_L256", "L1024_vs_L512")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def mib(value: int | float) -> str:
    return f"{float(value) / 1024**2:.1f}"


def build_policy(
    metrics: dict[str, Any],
    expansion: dict[str, Any],
) -> dict[str, Any]:
    positive_pairs = expansion["positive_model_length_pairs"]
    if not positive_pairs:
        return {
            "status": "final_from_canary_gate",
            "default_max_new_tokens": 256,
            "retry_policy": None,
            "diagnostic_max_new_tokens": [512, 1024],
            "production_max_new_tokens": [256],
            "rationale": [
                "No model/longer-length pair crossed any pre-registered LD canary expansion gate.",
                "Longer outputs therefore do not justify their additional latency and divergence risk.",
            ],
            "scope": (
                "Use one common policy for all checkpoints and later evaluations; "
                "do not give LeanDojo-trained models a separate length allowance."
            ),
        }
    return {
        "status": "provisional_pending_gated_full_evaluation",
        "default_max_new_tokens": 256,
        "retry_policy": None,
        "diagnostic_max_new_tokens": sorted(
            {int(pair["length"][1:]) for pair in positive_pairs}
        ),
        "production_max_new_tokens": [256],
        "positive_canary_pairs": positive_pairs,
        "rationale": [
            "At least one pre-registered LD canary gate is positive.",
            "The default remains 256 until the required full LD holdout and gated WB retention checks finish.",
        ],
        "scope": (
            "This is deliberately provisional and must be replaced after the gated "
            "full evaluation; canary results alone do not authorize a production change."
        ),
    }


def build_full_policy(
    metrics: dict[str, Any],
    transitions: dict[str, Any],
    expansion: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    positive_models = sorted(
        {pair["model"] for pair in expansion["positive_model_length_pairs"]}
    )
    evidence: dict[str, Any] = {}
    effective_lengths: list[int] = []
    for length in (512, 1024):
        model_deltas = {}
        theorem_wins = 0
        theorem_losses = 0
        truncated_successes = 0
        repetition_deltas = []
        for model in positive_models:
            baseline = metrics[f"{model}:ld:L256"]
            longer = metrics[f"{model}:ld:L{length}"]
            model_deltas[model] = {
                "solved_delta": (
                    int(longer["solved_count"]) - int(baseline["solved_count"])
                ),
                "candidate_success_delta": (
                    float(longer["candidate_success_rate"])
                    - float(baseline["candidate_success_rate"])
                ),
            }
            transition = transitions[
                f"{model}:ld:L{length}_vs_L256"
            ]
            theorem_wins += int(transition["theorem_fail_to_success"])
            theorem_losses += int(transition["theorem_success_to_fail"])
            truncated_successes += int(
                transition["baseline_length_finish_to_success"]
            )
            repetition_deltas.append(
                float(longer["repetition_ratio"])
                - float(baseline["repetition_ratio"])
            )
        wb_drops = []
        for pair in expansion["positive_model_length_pairs"]:
            pair_length = int(str(pair["length"]).removeprefix("L"))
            if pair_length != length:
                continue
            model = pair["model"]
            baseline = metrics[f"{model}:wb:L256"]
            longer = metrics[f"{model}:wb:L{length}"]
            wb_drops.append(
                {
                    "model": model,
                    "solved_drop": (
                        int(baseline["solved_count"])
                        - int(longer["solved_count"])
                    ),
                }
            )
        net_solved = sum(
            value["solved_delta"] for value in model_deltas.values()
        )
        retention_ok = bool(wb_drops) and all(
            value["solved_drop"] <= 2 for value in wb_drops
        )
        effective = (
            net_solved >= 2
            and theorem_wins >= 2
            and truncated_successes >= 2
            and retention_ok
        )
        if effective:
            effective_lengths.append(length)
        evidence[f"L{length}"] = {
            "model_ld_deltas": model_deltas,
            "aggregate_ld_solved_delta": net_solved,
            "theorem_wins": theorem_wins,
            "theorem_losses": theorem_losses,
            "l256_length_finish_to_success": truncated_successes,
            "wb_retention": wb_drops,
            "wb_retention_ok": retention_ok,
            "mean_repetition_delta": (
                sum(repetition_deltas) / max(1, len(repetition_deltas))
            ),
            "operational_length_bottleneck_evidence": effective,
        }
    if 512 in effective_lengths:
        improving_models = sum(
            value["solved_delta"] > 0
            for value in evidence["L512"]["model_ld_deltas"].values()
        )
        broad = improving_models * 2 >= max(1, len(positive_models))
        no_wb_drop = all(
            value["solved_drop"] <= 0
            for value in evidence["L512"]["wb_retention"]
        )
        repetition_ok = evidence["L512"]["mean_repetition_delta"] <= 0.03
        if broad and no_wb_drop and repetition_ok:
            default = 512
            retry = (
                "Use 1024 only as a difficult-case retry."
                if 1024 in effective_lengths
                else None
            )
            production = [512] + ([1024] if retry else [])
        else:
            default = 256
            retry = "Retry once at 512 only after an L256 verification failure."
            production = [256, 512]
        rationale = [
            "The full source-disjoint LD holdout sustained at least two net solved gains.",
            "At least two theorem-level gains and two L256-truncated candidate recoveries were observed.",
            "The gated full WB retention criterion was satisfied.",
        ]
        status = "final_full_evaluation"
    else:
        default = 256
        retry = None
        production = [256]
        rationale = [
            "The gated full evaluation did not satisfy the joint success, truncation-correlation, and retention criteria.",
            "Longer generation is therefore retained for diagnosis only.",
        ]
        status = "final_full_evaluation"
    policy = {
        "status": status,
        "default_max_new_tokens": default,
        "retry_policy": retry,
        "production_max_new_tokens": production,
        "diagnostic_max_new_tokens": [
            length for length in (512, 1024) if length not in production
        ],
        "operational_criteria": {
            "aggregate_net_ld_solved_gain": ">=2",
            "theorem_level_wins": ">=2",
            "l256_length_finish_candidates_recovered": ">=2",
            "full_wb_solved_drop_per_gated_pair": "<=2",
        },
        "evidence": evidence,
        "rationale": rationale,
        "scope": (
            "One common generation policy applies to all later model evaluations; "
            "no LeanDojo-trained checkpoint receives a private length advantage."
        ),
    }
    return policy, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/length_ablation"),
    )
    args = parser.parse_args()
    root = args.root
    comparisons = root / "comparisons"
    metrics = read_json(comparisons / "canary_metrics.json")
    paired = read_json(comparisons / "paired_statistics.json")
    transitions = read_json(comparisons / "transitions.json")
    gates = read_json(comparisons / "gate_decisions.json")
    expansion = read_json(comparisons / "expansion_decision.json")
    full_metrics_path = comparisons / "full_expansion_metrics.json"
    if full_metrics_path.is_file():
        full_metrics = read_json(full_metrics_path)
        full_transitions = read_json(
            comparisons / "full_expansion_transitions.json"
        )
        policy, full_evidence = build_full_policy(
            full_metrics, full_transitions, expansion
        )
    else:
        full_metrics = {}
        full_evidence = {}
        policy = build_policy(metrics, expansion)
    write_json(root / "recommended_generation_policy.json", policy)

    lines = [
        "# LeanDojo generation-length ablation",
        "",
        "## Frozen experiment",
        "",
        "- Checkpoints: the five pre-existing audited identities only.",
        "- Datasets: frozen WB length canary 50 and source-disjoint LD length canary 64.",
        "- Candidates: 4 per theorem with paired theorem/candidate seeds.",
        "- Lengths: 256, 512, and 1024; `max_new_tokens` is the only changed generation field.",
        "- Verification: source-faithful Pantograph, unchanged timeout and acceptance policy.",
        "",
        "All 15 model-by-length units passed the prompt/seed/cache-pairing audit.",
        "",
        "## Canary metrics",
        "",
        "| Model | Set | L | P@1 | P@2 | P@4 | Candidate success | Solved | Mean tokens | Length finish | Repetition | Unique proof | Gen s | Verify s | GPU MiB | RAM MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for dataset in DATASETS:
            for length in LENGTHS:
                value = metrics[f"{model}:{dataset}:L{length}"]
                pass_at = value["pass_at"]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            model,
                            dataset.upper(),
                            str(length),
                            pct(pass_at["pass@1"]),
                            pct(pass_at["pass@2"]),
                            pct(pass_at["pass@4"]),
                            pct(value["candidate_success_rate"]),
                            str(value["solved_count"]),
                            f"{value['output_tokens']['mean']:.2f}",
                            (
                                f"{value['length_finish']['count']} "
                                f"({pct(value['length_finish']['rate'])})"
                            ),
                            pct(value["repetition_ratio"]),
                            pct(value["unique_proof_ratio"]),
                            f"{value['generation_wall_seconds']:.1f}",
                            f"{value['verification_wall_seconds']:.1f}",
                            mib(value["gpu_peak_bytes"]),
                            mib(value["ram_peak_bytes"]),
                        ]
                    )
                    + " |"
                )

    lines.extend(
        [
            "",
            "The runtime did not expose EOS-versus-stop-string detail through the installed "
            "vLLM output object: `finish_reason` is retained, while `stop_reason` is `none`. "
            "The report therefore does not fabricate that unavailable subdivision.",
            "",
            "## Paired comparisons",
            "",
            "| Model | Set | Comparison | ΔP@4 | Bootstrap 95% CI | Wins/Losses/Ties | McNemar exact p |",
            "|---|---|---|---:|---|---|---:|",
        ]
    )
    for model in MODELS:
        for dataset in DATASETS:
            for comparison in COMPARISONS:
                value = paired[f"{model}:{dataset}:{comparison}"]
                ci = value["bootstrap_95_ci"]
                lines.append(
                    f"| {model} | {dataset.upper()} | {comparison} | "
                    f"{pct(value['delta_pass_at_4'])} | "
                    f"[{pct(ci[0])}, {pct(ci[1])}] | "
                    f"{value['wins']}/{value['losses']}/{value['ties']} | "
                    f"{value['mcnemar_exact_p']:.4g} |"
                )

    lines.extend(
        [
            "",
            "## Pre-registered expansion gate",
            "",
            "| Model | Longer L | +2 solved | +1 pp candidate success | 2 truncated failures become success | Positive |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for length in ("L512", "L1024"):
            decision = gates[model][length]
            conditions = decision["conditions"]
            lines.append(
                f"| {model} | {length[1:]} | "
                f"{conditions['at_least_two_more_ld_solved']} | "
                f"{conditions['candidate_success_gain_at_least_one_point']} | "
                f"{conditions['two_l256_length_failures_become_success']} | "
                f"{decision['positive_signal']} |"
            )

    lines.extend(
        [
            "",
            "## Paired transitions",
            "",
            "The first four transition columns are theorem-level; the final column is "
            "candidate-level because the truncation gate is defined on individual sampled outputs.",
            "",
            "| Model | Set | Comparison | Theorem fail→success | Theorem success→fail | Theorem both fail | Theorem both succeed | Truncated candidate→success |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for dataset in DATASETS:
            for comparison in COMPARISONS:
                value = transitions[f"{model}:{dataset}:{comparison}"]
                lines.append(
                    f"| {model} | {dataset.upper()} | {comparison} | "
                    f"{value['theorem_fail_to_success']} | "
                    f"{value['theorem_success_to_fail']} | "
                    f"{value['theorem_both_fail']} | "
                    f"{value['theorem_both_success']} | "
                    f"{value['baseline_length_finish_fail_to_success']} |"
                )
    lines.extend(
        [
            "",
            "## Three-length theorem outcomes",
            "",
            "| Model | Set | All lengths fail | All lengths succeed | L256/L512/L1024 solve patterns |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model in MODELS:
        for dataset in DATASETS:
            value = transitions[f"{model}:{dataset}:all_lengths"]
            patterns = ", ".join(
                f"{key}:{count}"
                for key, count in value["solve_pattern_counts"].items()
            )
            lines.append(
                f"| {model} | {dataset.upper()} | "
                f"{value['all_lengths_fail']} | {value['all_lengths_success']} | "
                f"`{patterns}` |"
            )

    lines.extend(
        [
            "",
            "## Decision and generation policy",
            "",
            f"- Policy status: `{policy['status']}`.",
            f"- Default: `{policy['default_max_new_tokens']}` tokens.",
            (
                "- Retry policy: none."
                if policy["retry_policy"] is None
                else f"- Retry policy: `{policy['retry_policy']}`."
            ),
        ]
    )
    for reason in policy["rationale"]:
        lines.append(f"- {reason}")
    if expansion["positive_model_length_pairs"]:
        if full_metrics:
            lines.extend(
                [
                    "",
                    "## Gated full-evaluation evidence",
                    "",
                    "| Length | Aggregate LD solved Δ | Theorem wins/losses | L256 truncations recovered | WB retention OK | Mean repetition Δ | Bottleneck evidence |",
                    "|---:|---:|---|---:|---:|---:|---:|",
                ]
            )
            for length in ("L512", "L1024"):
                value = full_evidence[length]
                lines.append(
                    f"| {length[1:]} | "
                    f"{value['aggregate_ld_solved_delta']} | "
                    f"{value['theorem_wins']}/{value['theorem_losses']} | "
                    f"{value['l256_length_finish_to_success']} | "
                    f"{value['wb_retention_ok']} | "
                    f"{value['mean_repetition_delta']:.4f} | "
                    f"{value['operational_length_bottleneck_evidence']} |"
                )
            lines.extend(
                [
                    "",
                    "The operational decision requires success, truncation correlation, "
                    "and WB retention simultaneously; a lower length-finish rate alone is "
                    "not treated as evidence.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "The pre-registered gate requires full LD holdout evaluation at all three "
                    "lengths for every model with any signal, plus full WB Gate150 only for the "
                    "positive model-by-length pairs. A final bottleneck conclusion is deferred "
                    "until those gated runs complete.",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "No full expansion is authorized by the frozen gate. The observed reduction "
                "in length finishes is not, by itself, evidence of improved theorem proving.",
            ]
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(policy, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
