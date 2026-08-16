"""Audit persisted expert-iteration generations without invoking a model."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TACTIC_NAMES = {
    "aesop",
    "exact",
    "field_simp",
    "linarith",
    "nlinarith",
    "norm_num",
    "omega",
    "rfl",
    "ring",
    "ring_nf",
    "simp",
    "simpa",
    "tauto",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def first_identifier(text: str) -> str:
    match = re.search(r"[A-Za-z_][A-Za-z0-9_']*", text)
    return match.group(0) if match else ""


def extraction_corruption(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "```",
            "### lean statement",
            "### informal statement",
            "here is",
            "the proof is",
        )
    ) or bool(re.search(r"\b(?:theorem|lemma)\b", text))


def unknown_root_cause(row: dict[str, Any], proof: str, imports: list[str]) -> str:
    error = str(row.get("error_message") or row.get("diagnostics") or "")
    if "unknown identifier" not in error.lower() and "unknown constant" not in error.lower():
        return "not_unknown_identifier"
    if extraction_corruption(proof):
        return "extraction_corruption"
    stripped = proof.lstrip()
    if not stripped.startswith("by") and first_identifier(stripped) in TACTIC_NAMES:
        return "term_mode"
    if "Mathlib" not in imports and stripped.startswith("by"):
        return "missing_import"
    if stripped.startswith("by") and "Mathlib" in imports:
        return "environment_mismatch"
    return "other"


def length_classification(text: str) -> str:
    lowered = text.lower()
    if text.count("```") % 2 == 1:
        return "unterminated_code_fence"
    if re.search(r"\b(?:theorem|lemma)\b", text):
        return "repeated_statement"
    if any(marker in lowered for marker in ("here is", "we need", "the proof", "therefore")):
        return "natural_language_continuation"
    if len(re.findall(r"\bby\b", text)) > 1:
        return "multiple_proofs"
    tokens = re.findall(r"\S+", text)
    if len(tokens) >= 24:
        trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
        if trigrams and len(set(trigrams)) / len(trigrams) < 0.55:
            return "repetitive_output"
    if text.rstrip().endswith(("<;>", "<", "[", "(", ",", ":=", "by")):
        return "other"
    return "long_but_plausible_proof"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    statements = {
        row["statement_id"]: row for row in read_jsonl(run_dir / "statements.jsonl")
    }
    generations = read_jsonl(run_dir / "iteration_000/discovery/generations.jsonl")
    verifications = read_jsonl(run_dir / "iteration_000/discovery/verifications.jsonl")
    verification_by_generation = {row["generation_id"]: row for row in verifications}
    benchmark_attempts = read_jsonl(run_dir / "benchmark/attempts.jsonl")

    finish_discovery = Counter(
        str((row.get("metadata") or {}).get("finish_reason") or "unknown")
        for row in generations
    )
    finish_benchmark = Counter(
        str(row.get("finish_reason") or "unknown") for row in benchmark_attempts
    )

    root_causes = Counter()
    reconstructed: dict[str, dict[str, Any]] = {}
    for generation in generations:
        verification = verification_by_generation.get(generation["generation_id"], {})
        statement = statements[generation["statement_id"]]
        proof = str(generation.get("extracted_proof") or "")
        imports = list(statement.get("imports") or [])
        assembled = f"{statement['statement'].rstrip()} := {proof.strip()}"
        cause = unknown_root_cause(verification, proof, imports)
        if cause != "not_unknown_identifier":
            root_causes[cause] += 1
        reconstructed[generation["generation_id"]] = {
            "statement_id": generation["statement_id"],
            "prompt": generation.get("prompt"),
            "raw_output": generation.get("raw_output"),
            "extracted_proof": proof,
            "normalized_proof": proof.strip(),
            "assembled_source": assembled,
            "imports": imports,
            "namespace": statement.get("namespace"),
            "context": statement.get("context"),
            "context_lines": statement.get("context_lines") or [],
            "error": verification.get("error_message"),
            "status": verification.get("status"),
            "finish_reason": (generation.get("metadata") or {}).get("finish_reason"),
            "output_tokens": (generation.get("metadata") or {}).get("completion_tokens"),
            "checkpoint": generation.get("checkpoint"),
            "unknown_identifier_root_cause": cause,
        }

    rng = random.Random(args.seed)
    audit_plan = {
        "elaboration_error": 10,
        "syntax_error": 3,
        "environment_error": 3,
        "tactic_error": 2,
    }
    audit_rows: list[dict[str, Any]] = []
    for status, count in audit_plan.items():
        candidates = [
            reconstructed[row["generation_id"]]
            for row in verifications
            if row.get("status") == status and row["generation_id"] in reconstructed
        ]
        audit_rows.extend(rng.sample(candidates, min(count, len(candidates))))
    length_discovery = [
        reconstructed[row["generation_id"]]
        for row in generations
        if (row.get("metadata") or {}).get("finish_reason") == "length"
    ]
    audit_rows.extend(rng.sample(length_discovery, min(2, len(length_discovery))))

    length_candidates: list[dict[str, Any]] = []
    for row in benchmark_attempts:
        if row.get("finish_reason") == "length" or row.get("hit_max_new_tokens"):
            text = str(row.get("raw_completion") or "")
            length_candidates.append(
                {
                    "source": "benchmark",
                    "problem_id": row.get("problem_id"),
                    "attempt_id": row.get("attempt_id"),
                    "classification": length_classification(text),
                    "raw_output": text,
                    "completion_tokens": row.get("completion_tokens"),
                }
            )
    selected_lengths = rng.sample(length_candidates, min(30, len(length_candidates)))
    length_distribution = Counter(row["classification"] for row in selected_lengths)

    report = {
        "run_dir": str(run_dir.resolve()),
        "reused_generation_counts": {
            "discovery": len(generations),
            "benchmark": len(benchmark_attempts),
        },
        "finish_reason_distribution": {
            "discovery": dict(finish_discovery),
            "benchmark": dict(finish_benchmark),
        },
        "unknown_identifier_by_root_cause": dict(root_causes),
        "failure_audit": audit_rows,
        "length_sample_size": len(selected_lengths),
        "length_classification_distribution": dict(length_distribution),
        "length_samples": selected_lengths,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"failure_audit", "length_samples"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
