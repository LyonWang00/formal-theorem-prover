#!/usr/bin/env python3
"""Audit official LeanDojo Benchmark 4 schema and proof distributions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.leandojo import iter_json_array
from lean_prover.lean_training.data.preparation import lean_code_tokens
from lean_prover.lean_training.expert_iteration.utils import file_sha256


def percentile(values: list[int], ratio: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def length_summary(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": sum(values) / max(1, len(values)),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values, default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    row_keys: Counter[str] = Counter()
    trace_keys: Counter[str] = Counter()
    total = 0
    empty_traces = 0
    complete_traces = 0
    contiguous_traces = 0
    proof_lengths: list[int] = []
    statement_lengths: list[int] = []
    tactic_steps: list[int] = []
    tactic_usage: Counter[str] = Counter()
    theorem_categories: Counter[str] = Counter()
    file_sources: Counter[str] = Counter()
    commits: Counter[str] = Counter()
    urls: Counter[str] = Counter()

    for row in iter_json_array(args.split):
        total += 1
        row_keys.update(row.keys())
        file_path = str(row.get("file_path") or "")
        full_name = str(row.get("full_name") or "")
        file_sources[file_path] += 1
        theorem_categories[full_name.split(".", 1)[0] if full_name else "unknown"] += 1
        commits[str(row.get("commit") or "missing")] += 1
        urls[str(row.get("url") or "missing")] += 1
        trace = list(row.get("traced_tactics") or ())
        if not trace:
            empty_traces += 1
            continue
        trace_keys.update(key for step in trace for key in step.keys())
        tactic_steps.append(len(trace))
        complete_traces += str(trace[-1].get("state_after") or "").strip() == "no goals"
        contiguous_traces += all(
            str(trace[index - 1].get("state_after") or "").strip()
            == str(trace[index].get("state_before") or "").strip()
            for index in range(1, len(trace))
        )
        tactics = "\n".join(str(step.get("tactic") or "") for step in trace)
        proof_lengths.append(sum(1 for _ in lean_code_tokens(tactics)))
        for step in trace:
            tactic = str(step.get("tactic") or "").strip()
            head = re.match(r"[A-Za-z_][A-Za-z0-9_']*", tactic)
            tactic_usage[head.group(0) if head else "other"] += 1
        state = str(trace[0].get("state_before") or "")
        target = state.rsplit("⊢", 1)[-1] if "⊢" in state else ""
        statement_lengths.append(sum(1 for _ in lean_code_tokens(target)))

    corpus_rows = 0
    corpus_premises = 0
    corpus_row_keys: Counter[str] = Counter()
    premise_keys: Counter[str] = Counter()
    import_counts: Counter[str] = Counter()
    declaration_kinds: Counter[str] = Counter()
    with_imports = 0
    with_premises = 0
    with open(args.corpus, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            corpus_rows += 1
            row = json.loads(line)
            corpus_row_keys.update(row.keys())
            imports = list(row.get("imports") or ())
            with_imports += bool(imports)
            import_counts.update(str(value) for value in imports)
            premises = list(row.get("premises") or ())
            with_premises += bool(premises)
            corpus_premises += len(premises)
            for premise in premises:
                premise_keys.update(premise.keys())
                declaration_kinds[str(premise.get("kind") or "missing")] += 1

    schema = {
        "dataset": "LeanDojo Benchmark 4",
        "split_format": "top-level JSON array",
        "corpus_format": "JSONL, one source file per row",
        "split_fields": dict(row_keys),
        "traced_tactic_fields": dict(trace_keys),
        "corpus_fields": dict(corpus_row_keys),
        "premise_fields": dict(premise_keys),
        "field_locations": {
            "statement": "corpus.premises[].code plus initial proof-state target",
            "theorem_name": "split.full_name / corpus.premises[].full_name",
            "proof": "split.traced_tactics[].tactic",
            "file_context": "initial state_before plus corpus declaration",
            "imports": "corpus.imports",
            "namespace": "qualified split.full_name (recovered)",
            "environment": "split url/commit and root metadata.json",
            "tactic_trace": "split.traced_tactics",
            "premises": "corpus.premises",
        },
    }
    statistics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_metadata": metadata,
        "input": {
            "split": str(args.split.resolve()),
            "split_sha256": file_sha256(args.split),
            "corpus": str(args.corpus.resolve()),
            "corpus_sha256": file_sha256(args.corpus),
        },
        "split": {
            "records": total,
            "empty_tactic_traces": empty_traces,
            "complete_tactic_traces": complete_traces,
            "contiguous_tactic_traces": contiguous_traces,
            "single_tactic_records": sum(value == 1 for value in tactic_steps),
            "single_tactic_ratio": sum(value == 1 for value in tactic_steps)
            / max(1, len(tactic_steps)),
            "multi_step_records": sum(value > 1 for value in tactic_steps),
            "proof_token_length": length_summary(proof_lengths),
            "statement_token_length": length_summary(statement_lengths),
            "tactic_step_count": length_summary(tactic_steps),
            "used_tactics": dict(tactic_usage.most_common()),
            "theorem_categories_top100": theorem_categories.most_common(100),
            "file_source_count": len(file_sources),
            "top_file_sources": file_sources.most_common(100),
            "source_commits": dict(commits),
            "source_urls": dict(urls),
        },
        "corpus": {
            "source_files": corpus_rows,
            "declarations": corpus_premises,
            "files_with_imports": with_imports,
            "files_with_declarations": with_premises,
            "unique_imports": len(import_counts),
            "top_imports": import_counts.most_common(100),
            "declaration_kinds": dict(declaration_kinds),
        },
        "environment": {
            "lean_version": metadata.get("lean_version", "not declared by artifact"),
            "mathlib_commit": (metadata.get("from_repo") or {}).get("commit"),
            "leandojo_version": metadata.get("leandojo_version"),
        },
    }
    (output / "leandojo_raw_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "leandojo_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = f"""# LeanDojo Benchmark 4 raw audit

- Records: {total}
- Source files in corpus: {corpus_rows}
- Declarations in corpus: {corpus_premises}
- Complete tactic traces: {complete_traces} ({complete_traces / max(1, total):.2%})
- Contiguous tactic traces: {contiguous_traces} ({contiguous_traces / max(1, total):.2%})
- Empty tactic traces: {empty_traces}
- Single-tactic ratio: {statistics['split']['single_tactic_ratio']:.2%}
- Proof token length: {json.dumps(statistics['split']['proof_token_length'], ensure_ascii=False)}
- Statement token length: {json.dumps(statistics['split']['statement_token_length'], ensure_ascii=False)}
- Tactic step count: {json.dumps(statistics['split']['tactic_step_count'], ensure_ascii=False)}

## Environment provenance

- LeanDojo version: {metadata.get('leandojo_version')}
- Source repository: {(metadata.get('from_repo') or {}).get('url')}
- Source mathlib commit: {(metadata.get('from_repo') or {}).get('commit')}
- Lean version: not explicitly declared in the downloaded metadata

The split and corpus must be joined.  Tactics alone are not a valid training
record because imports, namespace, universes, variables, and hypotheses would
be lost.
"""
    (output / "leandojo_statistics.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
