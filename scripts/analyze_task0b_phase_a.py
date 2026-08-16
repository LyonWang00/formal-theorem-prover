"""Phase A analysis for Task0B-Light filtered-hard examples.

This module is deliberately analysis-only: it loads frozen manifests and the
tokenizer, but never imports or constructs a Trainer or a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ANALYSIS_VERSION = "task0b_phase_a_v1"

KNOWN_TACTICS = (
    "aesop",
    "apply",
    "assumption",
    "by_cases",
    "cases",
    "cases'",
    "change",
    "constructor",
    "contrapose",
    "decide",
    "dsimp",
    "exact",
    "ext",
    "field_simp",
    "fun_prop",
    "have",
    "induction",
    "induction'",
    "infer_instance",
    "interval_cases",
    "intro",
    "linarith",
    "linear_combination",
    "native_decide",
    "nlinarith",
    "nontriviality",
    "norm_num",
    "obtain",
    "omega",
    "positivity",
    "rfl",
    "refine",
    "refine'",
    "rename_i",
    "repeat",
    "rintro",
    "ring",
    "ring_nf",
    "rw",
    "rwa",
    "simp",
    "simp_all",
    "simpa",
    "specialize",
    "subst",
    "tauto",
    "use",
)

CONTEXT_PATTERNS = (
    "unknown identifier",
    "invalid field notation",
    "unknown constant",
    "unknown namespace",
    "declaration has metavariables",
)

ENVIRONMENT_PATTERNS = (
    "out of memory",
    "worker",
    "server",
    "environment mismatch",
    "heartbeat",
    "connection",
    "internal error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path("outputs/task0b_light/classification_manifest.jsonl"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("outputs/task0b_light/candidate_results.jsonl"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("outputs/task0b_light/replay_manifest.jsonl"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path(
            "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/"
            "ADDON-B-WB2000-LD1000.jsonl"
        ),
    )
    parser.add_argument(
        "--generation-contract",
        type=Path,
        default=Path("outputs/task0b_light/generation_contract.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage2_data_ablation/phaseA_analysis"),
    )
    return parser.parse_args()


def resolve(project: Path, value: Path) -> Path:
    return value if value.is_absolute() else project / value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def percentile(values: Iterable[int | float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    result = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(result, 4)


def numeric_stats(values: Iterable[int | float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": round(sum(data) / len(data), 4) if data else None,
        "p50": percentile(data, 0.50),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "max": round(max(data), 4) if data else None,
    }


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id") or "")


def proof(row: dict[str, Any]) -> str:
    return str(row.get("proof") or row.get("completion") or "").strip()


def annotation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return ((row.get("difficulty_annotation") or {}).get("metrics") or {})


def source_kind(row: dict[str, Any]) -> str:
    value = str(row.get("sampling_source") or row.get("source") or "")
    if value.upper().startswith("WB") or "workbook" in value.lower():
        return "WB"
    if value.upper().startswith("LD") or "leandojo" in value.lower():
        return "LD"
    return value or "unknown"


def domain(row: dict[str, Any]) -> str:
    metrics = annotation_metrics(row)
    if metrics.get("mathlib_domain"):
        return str(metrics["mathlib_domain"])
    if source_kind(row) == "WB":
        return "Lean-Workbook"
    parts = str(row.get("source_file") or "").replace("\\", "/").split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "Mathlib" else "unknown"


def proof_token_count(row: dict[str, Any]) -> int:
    return int(annotation_metrics(row).get("proof_tokens") or row.get("label_tokens") or 0)


def lean_lexeme_count(value: str) -> int:
    """Project-standard model-free Lean token estimate for audit rules."""

    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_']*|\d+|[^\s]", value))


def tactic_step_count(value: str, row: dict[str, Any]) -> tuple[int, str]:
    annotated = annotation_metrics(row).get("tactic_steps")
    if annotated is not None:
        return int(annotated), "ld_review_trace"
    body = re.sub(r"^\s*by\b", "", value, count=1).strip()
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    syntactic = sum(
        max(1, len(re.findall(r"(?:<;>|;)", line)) + 1) for line in lines
    )
    return max(1, syntactic), "wb_syntactic_proxy"


def tactic_types(value: str) -> list[str]:
    found: list[str] = []
    for tactic in KNOWN_TACTICS:
        if re.search(rf"(?<![A-Za-z0-9_']){re.escape(tactic)}(?![A-Za-z0-9_'])", value):
            found.append(tactic)
    return found or ["other"]


def lemma_reference_proxy(value: str) -> int:
    bracket_references = re.findall(r"\[([^\]]+)\]", value)
    bracket_names = sum(
        len(re.findall(r"[A-Za-z_][\w.']*", item)) for item in bracket_references
    )
    commands = re.findall(
        r"\b(?:exact|apply|refine|using|rw|rwa|simpa\s+using|have)\s+\(?"
        r"([A-Za-z_][\w.']*)",
        value,
    )
    return bracket_names + len(commands)


def premise_measure(row: dict[str, Any], value: str) -> tuple[int, str]:
    annotated = annotation_metrics(row).get("premise_count")
    if annotated is not None:
        return int(annotated), "ld_review_premises"
    return lemma_reference_proxy(value), "wb_static_lemma_reference_proxy"


def primary_tactic(value: str) -> str:
    body = re.sub(r"^\s*(?:by|term)\b", "", value, count=1).strip()
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_!?']*)", body)
    return match.group(1) if match else "other"


def normalize_failure(candidate: dict[str, Any]) -> str:
    if candidate.get("compile_success") is True:
        return "success"
    if candidate.get("timed_out") is True:
        return "timeout"
    raw = str(
        candidate.get("failure_taxonomy")
        or candidate.get("error_type")
        or candidate.get("compile_status")
        or "other"
    ).lower()
    if "syntax" in raw or "parse" in raw:
        return "syntax_error"
    if "unsolved" in raw:
        return "unsolved_goals"
    if "tactic" in raw:
        return "tactic_error"
    if "elabor" in raw:
        return "elaboration_error"
    if "timeout" in raw:
        return "timeout"
    return "other"


def failure_origin(candidate: dict[str, Any], taxonomy: str) -> str:
    message = str(candidate.get("error_message") or "").lower()
    if taxonomy in {"unsolved_goals", "tactic_error"}:
        return "near_miss"
    if taxonomy == "syntax_error":
        return "syntax"
    if taxonomy == "timeout":
        return "timeout"
    if any(pattern in message for pattern in ENVIRONMENT_PATTERNS):
        return "environment"
    if any(pattern in message for pattern in CONTEXT_PATTERNS):
        return "context_or_missing_api"
    if taxonomy == "elaboration_error":
        return "semantic_elaboration"
    return "other"


def summarize_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "reference_proof_tokens": numeric_stats(
            item["reference_proof_tokens"] for item in rows
        ),
        "tactic_steps": numeric_stats(item["tactic_steps"] for item in rows),
        "lemma_or_premise_count": numeric_stats(
            item["lemma_or_premise_count"] for item in rows
        ),
        "source_distribution": dict(Counter(item["source"] for item in rows)),
        "domain_distribution": dict(
            Counter(item["domain"] for item in rows).most_common()
        ),
        "primary_tactic_distribution": dict(
            Counter(item["primary_tactic"] for item in rows).most_common()
        ),
        "tactic_type_distribution": dict(
            Counter(
                tactic for item in rows for tactic in item["reference_tactic_types"]
            ).most_common()
        ),
        "source_file_distribution": dict(
            Counter(item["source_file"] for item in rows).most_common()
        ),
        "measurement_provenance": {
            "tactic_steps": dict(Counter(item["tactic_step_kind"] for item in rows)),
            "lemma_or_premise": dict(
                Counter(item["lemma_or_premise_kind"] for item in rows)
            ),
        },
    }


def md_table(mapping: dict[str, Any], *, limit: int | None = None) -> str:
    items = list(mapping.items())
    if limit is not None:
        items = items[:limit]
    lines = ["| Value | Count |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value} |" for key, value in items)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    paths = {
        "classification": resolve(project, args.classification),
        "candidates": resolve(project, args.candidates),
        "replay": resolve(project, args.replay),
        "training_manifest": resolve(project, args.training_manifest),
        "generation_contract": resolve(project, args.generation_contract),
    }
    output = resolve(project, args.output)
    output.mkdir(parents=True, exist_ok=True)

    classifications = load_jsonl(paths["classification"])
    candidates = load_jsonl(paths["candidates"])
    replay = load_jsonl(paths["replay"])
    training_rows = load_jsonl(paths["training_manifest"])
    contract = json.loads(paths["generation_contract"].read_text(encoding="utf-8"))

    if len(classifications) != 3000 or len(training_rows) != 3000:
        raise RuntimeError(
            f"Expected 3000 classifications/training rows, got "
            f"{len(classifications)}/{len(training_rows)}"
        )
    if len(replay) != 800:
        raise RuntimeError(f"Expected frozen replay rows=800, got {len(replay)}")

    class_by_id = {record_id(row): row for row in classifications}
    train_by_id = {record_id(row): row for row in training_rows}
    if len(class_by_id) != 3000 or len(train_by_id) != 3000:
        raise RuntimeError("Duplicate or empty record_id in frozen 3000-row inputs")
    if set(class_by_id) != set(train_by_id):
        raise RuntimeError("Classification and first-round manifest record IDs differ")

    candidates_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        candidates_by_id[record_id(item)].append(item)

    filtered_ids = [
        record_id(row) for row in classifications if row.get("final_bucket") == "filtered_hard"
    ]
    if len(filtered_ids) != 833 or len(set(filtered_ids)) != 833:
        raise RuntimeError(
            f"Filtered Hard identity gate failed: rows={len(filtered_ids)}, "
            f"unique={len(set(filtered_ids))}"
        )
    missing_candidates = [key for key in filtered_ids if len(candidates_by_id[key]) != 2]
    if missing_candidates:
        raise RuntimeError(
            f"Each Filtered Hard row must retain exactly two candidates; failures="
            f"{missing_candidates[:10]}"
        )

    all_features: list[dict[str, Any]] = []
    for rid, row in train_by_id.items():
        reference = proof(row)
        steps, step_kind = tactic_step_count(reference, row)
        dependencies, dependency_kind = premise_measure(row, reference)
        all_features.append(
            {
                "record_id": rid,
                "source": source_kind(row),
                "domain": domain(row),
                "source_file": str(row.get("source_file") or "unknown"),
                "reference_proof_tokens": proof_token_count(row),
                "reference_lexeme_tokens": lean_lexeme_count(reference),
                "tactic_steps": steps,
                "tactic_step_kind": step_kind,
                "lemma_or_premise_count": dependencies,
                "lemma_or_premise_kind": dependency_kind,
                "primary_tactic": primary_tactic(reference),
                "reference_tactic_types": tactic_types(reference),
                "reference_proof_sha256": sha256_text(reference),
            }
        )
    all_feature_by_id = {item["record_id"]: item for item in all_features}
    hard_features = [all_feature_by_id[rid] for rid in filtered_ids]

    proof_values = [item["reference_proof_tokens"] for item in hard_features]
    step_values = [item["tactic_steps"] for item in hard_features]
    dependency_values = [item["lemma_or_premise_count"] for item in hard_features]
    thresholds = {
        "proof_tokens_p90": percentile(proof_values, 0.90),
        "proof_tokens_p95": percentile(proof_values, 0.95),
        "proof_tokens_extreme": max(128, math.ceil(percentile(proof_values, 0.95) or 0)),
        "tactic_steps_p90": percentile(step_values, 0.90),
        "tactic_steps_p95": percentile(step_values, 0.95),
        "tactic_steps_extreme": max(8, math.ceil(percentile(step_values, 0.95) or 0)),
        "dependencies_p90": percentile(dependency_values, 0.90),
        "dependencies_p95": percentile(dependency_values, 0.95),
        "dependencies_extreme": max(
            8, math.ceil(percentile(dependency_values, 0.95) or 0)
        ),
        "candidate_max_tokens": int(contract["max_new_tokens"]),
        "candidate_reference_ratio_max": 2.5,
        "candidate_reference_additive_slack": 64,
    }

    taxonomy_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    candidate_source_counts: Counter[str] = Counter()
    record_failure_mix: Counter[str] = Counter()
    reclassified: list[dict[str, Any]] = []

    for rid in filtered_ids:
        base = dict(all_feature_by_id[rid])
        row_candidates: list[dict[str, Any]] = []
        candidate_taxonomies: list[str] = []
        strong_near_miss_count = 0
        low_value_count = 0
        reference_tactics = set(base["reference_tactic_types"])
        for item in sorted(
            candidates_by_id[rid], key=lambda value: int(value.get("candidate_index") or 0)
        ):
            taxonomy = normalize_failure(item)
            origin = failure_origin(item, taxonomy)
            candidate_taxonomies.append(taxonomy)
            taxonomy_counts[taxonomy] += 1
            origin_counts[origin] += 1
            candidate_source_counts[str(item.get("candidate_source") or "unknown")] += 1
            candidate_proof = str(item.get("candidate_proof") or "").strip()
            candidate_tokens = lean_lexeme_count(candidate_proof) if candidate_proof else 0
            candidate_tactics = tactic_types(candidate_proof) if candidate_proof else []
            tactic_overlap = sorted(reference_tactics.intersection(candidate_tactics))
            candidate_limit = min(
                int(thresholds["candidate_max_tokens"]),
                max(
                    int(base["reference_lexeme_tokens"] * thresholds["candidate_reference_ratio_max"]),
                    int(base["reference_lexeme_tokens"] + thresholds["candidate_reference_additive_slack"]),
                ),
            )
            valid_prefix = bool(
                item.get("extraction_success")
                and candidate_proof
                and taxonomy in {"unsolved_goals", "tactic_error"}
                and not item.get("timed_out")
            )
            reasonable_length = 0 < candidate_tokens <= candidate_limit
            clear_progress = taxonomy == "unsolved_goals" or bool(tactic_overlap)
            strong_near_miss = valid_prefix and reasonable_length and clear_progress
            low_value = bool(
                not item.get("extraction_success")
                or origin in {"syntax", "context_or_missing_api", "environment", "timeout"}
            )
            strong_near_miss_count += int(strong_near_miss)
            low_value_count += int(low_value)
            row_candidates.append(
                {
                    "candidate_index": int(item.get("candidate_index") or 0),
                    "candidate_source": item.get("candidate_source"),
                    "failure_taxonomy": taxonomy,
                    "failure_origin": origin,
                    "extraction_success": bool(item.get("extraction_success")),
                    "finish_reason": item.get("finish_reason"),
                    "candidate_lexeme_tokens": candidate_tokens,
                    "reasonable_length": reasonable_length,
                    "valid_prefix": valid_prefix,
                    "reference_tactic_overlap": tactic_overlap,
                    "strong_near_miss": strong_near_miss,
                    "low_value_signal": low_value,
                }
            )
        record_failure_mix["+".join(sorted(candidate_taxonomies))] += 1

        complexity_flags = {
            "extreme_reference_proof": base["reference_proof_tokens"]
            > thresholds["proof_tokens_extreme"],
            "extreme_tactic_steps": base["tactic_steps"]
            > thresholds["tactic_steps_extreme"],
            "extreme_dependency": base["lemma_or_premise_count"]
            > thresholds["dependencies_extreme"],
            "joint_high_complexity": (
                base["reference_proof_tokens"] >= thresholds["proof_tokens_p90"]
                and (
                    base["tactic_steps"] >= thresholds["tactic_steps_p90"]
                    or base["lemma_or_premise_count"] >= thresholds["dependencies_p90"]
                )
            ),
        }
        hard_c = any(complexity_flags.values())
        if hard_c:
            bucket = "Hard-C"
            rationale = "extreme_or_joint_high_reference_complexity"
        elif strong_near_miss_count:
            bucket = "Hard-A"
            rationale = "at_least_one_valid_reasonable_near_miss"
        else:
            bucket = "Hard-B"
            rationale = "tractable_reference_but_no_clear_near_miss"
        reclassified.append(
            {
                **base,
                "hard_bucket": bucket,
                "rationale": rationale,
                "complexity_flags": complexity_flags,
                "strong_near_miss_candidates": strong_near_miss_count,
                "low_value_candidates": low_value_count,
                "candidates": row_candidates,
            }
        )

    bucket_counts = Counter(item["hard_bucket"] for item in reclassified)
    bucket_source = {
        bucket: dict(
            Counter(
                item["source"] for item in reclassified if item["hard_bucket"] == bucket
            )
        )
        for bucket in ("Hard-A", "Hard-B", "Hard-C")
    }
    class_raw = Counter(str(item.get("classification")) for item in classifications)
    class_final = Counter(str(item.get("final_bucket")) for item in classifications)
    taxonomy_complete = {
        key: taxonomy_counts[key]
        for key in (
            "syntax_error",
            "elaboration_error",
            "tactic_error",
            "unsolved_goals",
            "timeout",
            "other",
        )
    }
    origin_complete = {
        key: origin_counts[key]
        for key in (
            "near_miss",
            "semantic_elaboration",
            "syntax",
            "context_or_missing_api",
            "environment",
            "timeout",
            "other",
        )
    }
    phase_b_preflight = {
        "target_rows": 800,
        "hard_a_available": bucket_counts["Hard-A"],
        "hard_b_available": bucket_counts["Hard-B"],
        "hard_c_forbidden": bucket_counts["Hard-C"],
        "stable_core_available": class_final["stable_core"],
        "max_unique_rows_without_hard_c": (
            bucket_counts["Hard-A"]
            + bucket_counts["Hard-B"]
            + min(100, class_final["stable_core"])
        ),
    }
    phase_b_preflight["row_shortfall"] = max(
        0,
        phase_b_preflight["target_rows"]
        - phase_b_preflight["max_unique_rows_without_hard_c"],
    )
    phase_b_preflight["ready_under_current_constraints"] = (
        phase_b_preflight["row_shortfall"] == 0
    )

    complexity = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase A analysis only; no generation, Trainer, or training",
        "input_identity": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "hard_scale": {
            "stable_core": class_final["stable_core"],
            "frontier": class_final["frontier"],
            "raw_hard": class_raw["raw_hard"],
            "filtered_hard": class_final["filtered_hard"],
            "invalid_or_excluded": class_final["invalid_or_excluded"],
            "unclassified_not_sampled": class_final["unclassified_not_sampled"],
        },
        "filtered_hard": summarize_features(reclassified),
        "filtered_hard_by_source": {
            source: summarize_features(
                [item for item in reclassified if item["source"] == source]
            )
            for source in ("WB", "LD")
        },
        "first_round_training_baseline": summarize_features(all_features),
        "failure_analysis": {
            "candidate_rows": sum(taxonomy_counts.values()),
            "taxonomy": taxonomy_complete,
            "origin": origin_complete,
            "record_failure_mix": dict(record_failure_mix.most_common()),
            "candidate_source": dict(candidate_source_counts.most_common()),
            "high_value_near_miss_candidates": sum(
                item["strong_near_miss_candidates"] for item in reclassified
            ),
            "records_with_high_value_near_miss": sum(
                item["strong_near_miss_candidates"] > 0 for item in reclassified
            ),
            "low_value_candidates": sum(
                item["low_value_candidates"] for item in reclassified
            ),
            "records_with_only_low_value_candidates": sum(
                item["low_value_candidates"] == 2 for item in reclassified
            ),
        },
        "measurement_notes": {
            "reference_proof_tokens": "Frozen tokenizer-derived label_tokens; excludes the supervised EOS token.",
            "tactic_steps": "LD uses reviewed trace steps; WB uses a deterministic line/semicolon syntactic proxy.",
            "lemma_or_premise_count": "LD uses reviewed extracted premises; WB uses a conservative static lemma-reference proxy. These are reported with separate provenance and must not be interpreted as equally precise.",
            "candidate_near_miss": "Requires successful proof extraction, Pantograph tactic/unsolved-goal failure, no timeout, reasonable lean-lexeme-regex-v1 length, and either an unsolved goal or tactic overlap with the reference.",
        },
    }

    reclassification_payload = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "scope": "Existing Task0B-Light results only; no regeneration",
        "thresholds": thresholds,
        "rule_priority": [
            "Hard-C if any extreme/joint-high reference complexity flag is true",
            "Hard-A if non-Hard-C and at least one candidate is a valid, reasonable strong near-miss",
            "Hard-B otherwise (tractable reference but uncertain/weak candidate signal)",
        ],
        "counts": dict(bucket_counts),
        "counts_by_source": bucket_source,
        "phase_b_preflight": phase_b_preflight,
        "records": reclassified,
    }

    filtered_summary = complexity["filtered_hard"]
    failure = complexity["failure_analysis"]
    complexity_md = f"""# Phase A — Filtered Hard complexity analysis

## Scope and identity

- Phase: analysis only; no generation, Trainer, GPU training, or model training.
- First-round frozen rows: {len(training_rows)}.
- Filtered Hard rows: {len(reclassified)}.
- Candidate results analyzed: {failure['candidate_rows']} (exactly two per Filtered Hard row).
- Analysis version: `{ANALYSIS_VERSION}`.

## Hard scale

| Bucket | Rows |
|---|---:|
| Stable/Core | {complexity['hard_scale']['stable_core']} |
| Frontier | {complexity['hard_scale']['frontier']} |
| Raw Hard | {complexity['hard_scale']['raw_hard']} |
| Filtered Hard | {complexity['hard_scale']['filtered_hard']} |
| invalid_or_excluded | {complexity['hard_scale']['invalid_or_excluded']} |
| unclassified_not_sampled | {complexity['hard_scale']['unclassified_not_sampled']} |

Filtered Hard originates from Raw Hard rows whose reference proof currently verifies,
context is complete, no environment/syntax corruption or timeout-only failure is present,
and no semantic truncation is recorded.

## Reference-proof complexity

| Metric | Count | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Proof tokens | {filtered_summary['reference_proof_tokens']['count']} | {filtered_summary['reference_proof_tokens']['mean']} | {filtered_summary['reference_proof_tokens']['p50']} | {filtered_summary['reference_proof_tokens']['p90']} | {filtered_summary['reference_proof_tokens']['p95']} | {filtered_summary['reference_proof_tokens']['max']} |
| Tactic steps | {filtered_summary['tactic_steps']['count']} | {filtered_summary['tactic_steps']['mean']} | {filtered_summary['tactic_steps']['p50']} | {filtered_summary['tactic_steps']['p90']} | {filtered_summary['tactic_steps']['p95']} | {filtered_summary['tactic_steps']['max']} |
| Lemma/premise count | {filtered_summary['lemma_or_premise_count']['count']} | {filtered_summary['lemma_or_premise_count']['mean']} | {filtered_summary['lemma_or_premise_count']['p50']} | {filtered_summary['lemma_or_premise_count']['p90']} | {filtered_summary['lemma_or_premise_count']['p95']} | {filtered_summary['lemma_or_premise_count']['max']} |

### Comparison with the complete 3000-row first-round pool

| Metric | Filtered Hard mean | Full-pool mean | Filtered Hard P95 | Full-pool P95 |
|---|---:|---:|---:|---:|
| Proof tokens | {filtered_summary['reference_proof_tokens']['mean']} | {complexity['first_round_training_baseline']['reference_proof_tokens']['mean']} | {filtered_summary['reference_proof_tokens']['p95']} | {complexity['first_round_training_baseline']['reference_proof_tokens']['p95']} |
| Tactic steps | {filtered_summary['tactic_steps']['mean']} | {complexity['first_round_training_baseline']['tactic_steps']['mean']} | {filtered_summary['tactic_steps']['p95']} | {complexity['first_round_training_baseline']['tactic_steps']['p95']} |
| Lemma/premise count | {filtered_summary['lemma_or_premise_count']['mean']} | {complexity['first_round_training_baseline']['lemma_or_premise_count']['mean']} | {filtered_summary['lemma_or_premise_count']['p95']} | {complexity['first_round_training_baseline']['lemma_or_premise_count']['p95']} |

### Complexity by source

| Source | Rows | Proof mean/P50/P90/P95/max | Steps mean/P50/P90/P95/max | Lemma/premise mean/P50/P90/P95/max |
|---|---:|---|---|---|
| WB | {complexity['filtered_hard_by_source']['WB']['rows']} | {complexity['filtered_hard_by_source']['WB']['reference_proof_tokens']['mean']}/{complexity['filtered_hard_by_source']['WB']['reference_proof_tokens']['p50']}/{complexity['filtered_hard_by_source']['WB']['reference_proof_tokens']['p90']}/{complexity['filtered_hard_by_source']['WB']['reference_proof_tokens']['p95']}/{complexity['filtered_hard_by_source']['WB']['reference_proof_tokens']['max']} | {complexity['filtered_hard_by_source']['WB']['tactic_steps']['mean']}/{complexity['filtered_hard_by_source']['WB']['tactic_steps']['p50']}/{complexity['filtered_hard_by_source']['WB']['tactic_steps']['p90']}/{complexity['filtered_hard_by_source']['WB']['tactic_steps']['p95']}/{complexity['filtered_hard_by_source']['WB']['tactic_steps']['max']} | {complexity['filtered_hard_by_source']['WB']['lemma_or_premise_count']['mean']}/{complexity['filtered_hard_by_source']['WB']['lemma_or_premise_count']['p50']}/{complexity['filtered_hard_by_source']['WB']['lemma_or_premise_count']['p90']}/{complexity['filtered_hard_by_source']['WB']['lemma_or_premise_count']['p95']}/{complexity['filtered_hard_by_source']['WB']['lemma_or_premise_count']['max']} |
| LD | {complexity['filtered_hard_by_source']['LD']['rows']} | {complexity['filtered_hard_by_source']['LD']['reference_proof_tokens']['mean']}/{complexity['filtered_hard_by_source']['LD']['reference_proof_tokens']['p50']}/{complexity['filtered_hard_by_source']['LD']['reference_proof_tokens']['p90']}/{complexity['filtered_hard_by_source']['LD']['reference_proof_tokens']['p95']}/{complexity['filtered_hard_by_source']['LD']['reference_proof_tokens']['max']} | {complexity['filtered_hard_by_source']['LD']['tactic_steps']['mean']}/{complexity['filtered_hard_by_source']['LD']['tactic_steps']['p50']}/{complexity['filtered_hard_by_source']['LD']['tactic_steps']['p90']}/{complexity['filtered_hard_by_source']['LD']['tactic_steps']['p95']}/{complexity['filtered_hard_by_source']['LD']['tactic_steps']['max']} | {complexity['filtered_hard_by_source']['LD']['lemma_or_premise_count']['mean']}/{complexity['filtered_hard_by_source']['LD']['lemma_or_premise_count']['p50']}/{complexity['filtered_hard_by_source']['LD']['lemma_or_premise_count']['p90']}/{complexity['filtered_hard_by_source']['LD']['lemma_or_premise_count']['p95']}/{complexity['filtered_hard_by_source']['LD']['lemma_or_premise_count']['max']} |

### Source

{md_table(filtered_summary['source_distribution'])}

### Domain

{md_table(filtered_summary['domain_distribution'])}

### Primary tactic

{md_table(filtered_summary['primary_tactic_distribution'], limit=30)}

### Tactic types used anywhere in the reference proof

{md_table(filtered_summary['tactic_type_distribution'], limit=40)}

### Source files (top 30)

{md_table(filtered_summary['source_file_distribution'], limit=30)}

## Failure taxonomy

### Candidate-level taxonomy

{md_table(failure['taxonomy'])}

### Interpreted source

{md_table(failure['origin'])}

- High-value near-miss candidates: {failure['high_value_near_miss_candidates']}.
- Filtered Hard records with at least one high-value near miss: {failure['records_with_high_value_near_miss']}.
- Low-value candidates (syntax/context/environment/timeout/extraction failure): {failure['low_value_candidates']}.
- Records whose two candidates are both low-value: {failure['records_with_only_low_value_candidates']}.

## Measurement limits

- LD tactic-step and premise counts come from the reviewed trace metadata.
- WB has no premise trace. Its tactic steps and lemma references are deterministic
  static proxies and are reported separately in the JSON provenance fields.
- `elaboration_error` is retained as its own taxonomy. It is not automatically
  called near-miss or low-value unless its message supplies context/environment evidence.
- No candidate was regenerated and no reference theorem or proof was edited.
"""

    reclass_md = f"""# Phase A — Filtered Hard reclassification

## Frozen decision rule

1. **Hard-C** first: reference proof has an extreme length/step/dependency flag,
   or jointly exceeds the P90 proof threshold and a P90 step/dependency threshold.
2. **Hard-A** next: non-Hard-C row with at least one extracted, non-timeout
   `unsolved_goals`/`tactic_error` candidate of reasonable static lexeme length and
   clear progress (unsolved goal or reference-tactic overlap).
3. **Hard-B** otherwise: the verified reference remains tractable, but the existing
   candidates do not provide a sufficiently strong near-miss signal.

Thresholds are derived once from the frozen 833-row set and recorded in
`hard_reclassification.json`; no quota fitting or result-dependent resampling occurs.

## Result

| Bucket | Rows | WB | LD | Phase B interpretation |
|---|---:|---:|---:|---|
| Hard-A | {bucket_counts['Hard-A']} | {bucket_source['Hard-A'].get('WB', 0)} | {bucket_source['Hard-A'].get('LD', 0)} | Highest-priority SFT candidates |
| Hard-B | {bucket_counts['Hard-B']} | {bucket_source['Hard-B'].get('WB', 0)} | {bucket_source['Hard-B'].get('LD', 0)} | Small, stratified use only |
| Hard-C | {bucket_counts['Hard-C']} | {bucket_source['Hard-C'].get('WB', 0)} | {bucket_source['Hard-C'].get('LD', 0)} | Exclude from Phase B SFT; prefer Planner/Subgoal work |
| Total | {len(reclassified)} | {sum(1 for item in reclassified if item['source'] == 'WB')} | {sum(1 for item in reclassified if item['source'] == 'LD')} | — |

## Interpretation

Filtered Hard is a valid quality-gated pool, but it is not homogeneous training
data. Hard-A has direct evidence that the frozen M0 reached a valid partial proof
path and is the strongest SFT subset. Hard-B has verified targets but weaker
diagnostic evidence and should be sampled conservatively. Hard-C crosses the
frozen complexity boundary and should not be used in the Hard-heavy stress test.

This classification changes no proof, theorem, candidate, or frozen manifest.

## Phase B feasibility gate

- Required rows: {phase_b_preflight['target_rows']}.
- Unique Hard-A + Hard-B rows: {phase_b_preflight['hard_a_available'] + phase_b_preflight['hard_b_available']}.
- Existing Stable/Core rows available for the permitted small supplement: {phase_b_preflight['stable_core_available']}.
- Maximum unique rows without forbidden Hard-C: {phase_b_preflight['max_unique_rows_without_hard_c']}.
- Shortfall: {phase_b_preflight['row_shortfall']} rows.

Therefore Phase B is **not executable under the current exact constraints**. Filling
the shortfall would require a separately approved data-source or rule change; replacement,
duplicate theorems, or Hard-C must not be used implicitly.
"""

    phase_report = f"""# Stage 2 data ablation — Phase A report

## Decision

The 833 Filtered Hard rows are **quality-valid but not uniformly suitable** for
second-round SFT. Use Hard-A as the primary learnable hard pool, add a limited and
stratified amount of Hard-B only if needed, and exclude Hard-C from whole-proof SFT.

## Evidence

- Hard-A: {bucket_counts['Hard-A']} rows.
- Hard-B: {bucket_counts['Hard-B']} rows.
- Hard-C: {bucket_counts['Hard-C']} rows.
- Candidate taxonomy: `{json.dumps(failure['taxonomy'], ensure_ascii=False)}`.
- Records with a strong near-miss: {failure['records_with_high_value_near_miss']}.
- No new generation or compilation was performed; only frozen Task0B-Light results
  and the verified first-round reference rows were analyzed.
- Phase B preflight: **BLOCKED_DATA_SHORTFALL_{phase_b_preflight['row_shortfall']}**;
  the maximum unique eligible rows are {phase_b_preflight['max_unique_rows_without_hard_c']}
  versus the required 800.

## Phase boundary

Phase A completed.

No Trainer started.

Waiting for approval before Phase B.
"""

    write_json(output / "hard_complexity_analysis.json", complexity)
    write_text(output / "hard_complexity_analysis.md", complexity_md)
    write_json(output / "hard_reclassification.json", reclassification_payload)
    write_text(output / "hard_reclassification.md", reclass_md)
    write_text(output / "phaseA_report.md", phase_report)
    write_json(
        output / "phaseA_audit.json",
        {
            "passed": True,
            "analysis_version": ANALYSIS_VERSION,
            "training_rows": len(training_rows),
            "classification_rows": len(classifications),
            "filtered_hard_rows": len(reclassified),
            "filtered_hard_unique_record_ids": len(set(filtered_ids)),
            "candidate_rows_for_filtered_hard": sum(taxonomy_counts.values()),
            "candidates_per_filtered_hard": 2,
            "replay_rows_read_only": len(replay),
            "proofs_modified": 0,
            "theorems_modified": 0,
            "new_generations": 0,
            "trainer_started": False,
            "gpu_training_started": False,
            "phase_b_started": False,
            "phase_b_preflight": phase_b_preflight,
            "output_files": [
                "hard_complexity_analysis.json",
                "hard_complexity_analysis.md",
                "hard_reclassification.json",
                "hard_reclassification.md",
                "phaseA_report.md",
            ],
        },
    )
    print(
        json.dumps(
            {
                "phase": "A",
                "filtered_hard": len(reclassified),
                "hard_buckets": dict(bucket_counts),
                "candidate_taxonomy": taxonomy_complete,
                "phase_b_preflight": phase_b_preflight,
                "records_with_near_miss": failure[
                    "records_with_high_value_near_miss"
                ],
                "trainer_started": False,
                "phase_b_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
