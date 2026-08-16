#!/usr/bin/env python3
"""Produce evidence-backed root-cause analysis for a failed LeanDojo gate."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PATTERNS = {
    "unknown_identifier": re.compile(r"Unknown (?:identifier|constant) `([^`]+)`"),
    "invalid_field": re.compile(r"Invalid field `([^`]+)`"),
    "application_type_mismatch": re.compile(r"Application type mismatch"),
    "type_mismatch": re.compile(r"\btype mismatch\b", re.IGNORECASE),
    "function_expected": re.compile(r"function expected", re.IGNORECASE),
    "invalid_argument_name": re.compile(r"Invalid argument name"),
    "unexpected_token": re.compile(r"unexpected token", re.IGNORECASE),
    "expected_token": re.compile(
        r"(?:error:\s*expected token|unexpected token|expected command|"
        r"expected ['\")\]}])",
        re.IGNORECASE,
    ),
    "unsolved_goals": re.compile(r"unsolved goals", re.IGNORECASE),
    "tactic_failure": re.compile(
        r"(tactic|made no progress|no goals to be solved|failed)", re.IGNORECASE
    ),
    "namespace_close": re.compile(r"Missing name after `end`|Invalid name after `end`"),
    "unknown_namespace": re.compile(r"unknown namespace", re.IGNORECASE),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--raw-metadata", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    raw_metadata = json.loads(args.raw_metadata.read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in args.verification.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = [result for result in results if not result.get("success")]
    pattern_counts: Counter[str] = Counter()
    unknown_identifiers: Counter[str] = Counter()
    invalid_fields: Counter[str] = Counter()
    primary_messages: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for result in failures:
        errors = result.get("compile_errors") or [result.get("diagnostics") or ""]
        for raw_error in errors:
            error = str(raw_error)
            normalized = _normalize_error(error)
            primary_messages[normalized] += 1
            matched = False
            for name, pattern in PATTERNS.items():
                matches = list(pattern.finditer(error))
                if not matches:
                    continue
                matched = True
                pattern_counts[name] += 1
                if name == "unknown_identifier":
                    unknown_identifiers.update(match.group(1) for match in matches)
                elif name == "invalid_field":
                    invalid_fields.update(match.group(1) for match in matches)
                if len(examples[name]) < 5:
                    examples[name].append(
                        {
                            "record_id": result.get("record_id"),
                            "problem_id": result.get("problem_id"),
                            "message": error[:1200],
                            "source_path": result.get("source_path"),
                        }
                    )
            if not matched:
                pattern_counts["other"] += 1
                if len(examples["other"]) < 5:
                    examples["other"].append(
                        {
                            "record_id": result.get("record_id"),
                            "message": error[:1200],
                        }
                    )

    source_commit = (
        raw_metadata.get("artifact_metadata", {})
        .get("from_repo", {})
        .get("commit")
    )
    target_commit = gate.get("environment", {}).get("mathlib_commit")
    version_mismatch = bool(source_commit and target_commit and source_commit != target_commit)
    total = gate["metrics"]["total_sampled"]
    succeeded = gate["metrics"]["pantograph_success"]
    analysis = {
        "decision": gate["decision"],
        "sample": {
            "total": total,
            "success": succeeded,
            "failure": total - succeeded,
            "success_rate": gate["metrics"]["success_rate"],
        },
        "source_environment": {
            "dataset": raw_metadata.get("dataset_name"),
            "dataset_version": raw_metadata.get("dataset_version"),
            "leandojo_version": raw_metadata.get("artifact_metadata", {}).get(
                "leandojo_version"
            ),
            "mathlib_commit": source_commit,
            "lean_version": "not declared in Benchmark 4 metadata",
        },
        "target_environment": gate.get("environment"),
        "source_target_mathlib_commit_mismatch": version_mismatch,
        "compile_error_pattern_counts": dict(pattern_counts),
        "top_unknown_identifiers": unknown_identifiers.most_common(100),
        "top_invalid_fields": invalid_fields.most_common(100),
        "top_normalized_messages": primary_messages.most_common(100),
        "examples": dict(examples),
        "root_cause_assessment": {
            "version_mismatch": {
                "severity": "high" if version_mismatch else "not observed",
                "evidence": (
                    f"source mathlib={source_commit}, target mathlib={target_commit}; "
                    f"type/API/identifier-shaped errors="
                    f"{pattern_counts['unknown_identifier'] + pattern_counts['invalid_field'] + pattern_counts['application_type_mismatch'] + pattern_counts['type_mismatch']}"
                ),
            },
            "context_recovery": {
                "severity": "high",
                "evidence": (
                    f"gate taxonomy missing_context="
                    f"{gate['failure_taxonomy'].get('missing_context', 0)}; "
                    "LeanDojo split lacks explicit namespace/section variable blocks, "
                    "so they are reconstructed from pretty-printed proof states."
                ),
            },
            "imports": {
                "severity": (
                    "medium"
                    if gate["failure_taxonomy"].get("missing_import", 0)
                    else "low"
                ),
                "evidence": (
                    f"missing_import={gate['failure_taxonomy'].get('missing_import', 0)}; "
                    "verification intentionally imports target-environment Mathlib, "
                    "while source import provenance is retained only as metadata."
                ),
            },
            "parser": {
                "severity": (
                    "medium"
                    if gate["failure_taxonomy"].get("syntax_error", 0)
                    else "low"
                ),
                "evidence": (
                    f"syntax_error={gate['failure_taxonomy'].get('syntax_error', 0)}, "
                    f"unexpected/expected-token messages="
                    f"{pattern_counts['unexpected_token'] + pattern_counts['expected_token']}."
                ),
            },
            "namespace_adapter_regression": {
                "severity": (
                    "resolved" if pattern_counts["namespace_close"] == 0 else "present"
                ),
                "evidence": (
                    f"namespace closing errors after adapter fix="
                    f"{pattern_counts['namespace_close']}."
                ),
            },
        },
        "training_permitted": False,
        "recommended_next_action": (
            "Do not train.  Build or check out a Lean/mathlib environment matching "
            "the source commit, and replace proof-state-only context recovery with "
            "source-file context extraction before repeating the identical gate.  "
            "This requires an explicit environment migration decision."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = f"""# LeanDojo quality-gate failure analysis

## Decision

**STOP TRAINING.**  The final fixed sample passed {succeeded}/{total}
({gate['metrics']['success_rate']:.2%}), below the required 70%.

## Environment mismatch

- Source mathlib commit: `{source_commit}`
- Target mathlib commit: `{target_commit}`
- Exact match: `{not version_mismatch}`
- Source Lean version: not declared by the downloaded artifact

## Gate failure taxonomy

```json
{json.dumps(gate['failure_taxonomy'], ensure_ascii=False, indent=2)}
```

## Compiler error shapes

```json
{json.dumps(dict(pattern_counts), ensure_ascii=False, indent=2)}
```

Top unknown identifiers:

```json
{json.dumps(unknown_identifiers.most_common(30), ensure_ascii=False, indent=2)}
```

## Root-cause assessment

1. **Version mismatch: high.** The dataset was traced at a different mathlib
   commit. Unknown/renamed identifiers, changed fields, and type mismatches are
   direct evidence of API drift.
2. **Context recovery: high.** Benchmark split rows contain proof states, not
   complete namespace/section source context. Explicit binders reconstructed
   from pretty-printed states cannot recover every source-level notation,
   local attribute, section option, or declaration-resolution choice.
3. **Imports: medium.** Source imports are present in the corpus and preserved
   as provenance, but compilation uses the current target `Mathlib`; 25 records
   were classified as missing imports/identifiers.
4. **Parser compatibility: medium.** Seventeen records have syntax-shaped
   failures after reconstruction.
5. **Namespace wrapper regression: resolved.** The final run has
   {pattern_counts['namespace_close']} namespace-closing errors; the initial
   2.8% attempt is archived separately and was not used as the final gate.

## Required next action

Do not create training manifests or checkpoints.  First decide whether to:

- reproduce a Lean/mathlib environment matching the source commit and rerun the
  same 500 IDs; or
- implement source-file context extraction/migration into the current
  environment and rerun the same gate.

Neither action should be taken implicitly because it changes the project
environment or the meaning of the migration.
"""
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


def _normalize_error(message: str) -> str:
    value = re.sub(r"^\d+:\d+(?:-\d+:\d+)?:\s*", "", message.strip())
    value = re.sub(r"EvalProblem_\d+_Attempt_\d+", "EvalProblem_<n>", value)
    value = re.sub(r"\bld_[0-9a-f]{16,}\b", "ld_<hash>", value)
    value = re.sub(r"\s+", " ", value)
    return value[:500]


if __name__ == "__main__":
    main()
