#!/usr/bin/env python3
"""Render a complete but non-redundant manual-review packet; never label it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.batch.open(encoding="utf-8-sig")
        if line.strip()
    ]
    end = args.end or len(rows)
    for index, row in enumerate(rows, 1):
        if index < args.start or index > end:
            continue
        recovery = row.get("context_recovery") or {}
        verification = row.get("verification") or {}
        print(f"## {index}. {row['sample_id']} — {row.get('qualified_name')}")
        print(f"file/domain: {row.get('source_file')} / {row.get('mathlib_domain')}")
        print(
            "metrics: "
            f"statement={row.get('statement_tokens')}, proof={row.get('proof_tokens')}, "
            f"steps={row.get('tactic_steps')}, premises={len(row.get('premises') or [])}, "
            f"same_file={row.get('same_file_premise_count')}, "
            f"style={row.get('proof_style')}"
        )
        print("statement:\n```lean")
        print(row.get("statement") or "")
        print("```\nproof:\n```lean")
        print(row.get("proof") or "")
        print("```")
        print(f"imports: {row.get('imports')}")
        print(f"namespace: {row.get('namespace_stack')}; opens: {row.get('open_namespaces')}")
        print(
            f"scopes: {row.get('open_scopes')}; local_notations: "
            f"{row.get('local_notations')}; local_attributes: {row.get('local_attributes')}"
        )
        print(f"local_instances: {row.get('local_instances')}")
        print(f"variables: {row.get('variables')}; hypotheses: {row.get('hypotheses')}")
        print(f"active_context: {recovery.get('active_commands')}")
        print("premises:")
        premises = row.get("premises") or []
        if not premises:
            print("- none")
        for premise in premises:
            print(
                f"- {premise.get('qualified_name')} | same_file="
                f"{bool(premise.get('is_same_file'))} | file={premise.get('source_file')} "
                f"| type={premise.get('statement') or premise.get('type') or 'unavailable'}"
            )
        print("tactic trace:")
        trace = row.get("tactic_trace") or []
        if not trace:
            print("- none (term proof)")
        for step in trace:
            print(f"- tactic: {step.get('tactic')}")
            print(f"  before: {step.get('state_before')}")
            print(f"  after: {step.get('state_after')}")
        print(
            "verification: "
            f"status={row.get('verification_status')}, "
            f"compile={verification.get('compile_success')}, "
            f"timeout={verification.get('timed_out')}, "
            f"elapsed={verification.get('elapsed_seconds')}, "
            f"error={verification.get('error_category') or ''}"
        )
        print()


if __name__ == "__main__":
    main()
