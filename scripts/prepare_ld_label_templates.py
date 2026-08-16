#!/usr/bin/env python3
"""Create blank, metric-locked forms for actual Codex difficulty review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty-dir",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/difficulty"),
    )
    args = parser.parse_args()
    root = args.difficulty_dir.resolve()
    metrics_by_id = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "review_metrics.jsonl")
    }
    batch_paths = sorted((root / "review_batches").glob("batch_*.jsonl"))
    if not batch_paths:
        raise FileNotFoundError("no frozen review batches found")
    for batch_path in batch_paths:
        templates: list[dict[str, Any]] = []
        for source in read_jsonl(batch_path):
            sample_id = str(source["sample_id"])
            metrics = metrics_by_id[sample_id]
            templates.append(
                {
                    "sample_id": sample_id,
                    "qualified_name": source.get("qualified_name"),
                    "difficulty": "",
                    "confidence": "",
                    "reason_summary": "",
                    "evidence": {
                        "statement_complexity": "",
                        "proof_structure": "",
                        "tactic_complexity": "",
                        "premise_dependency": "",
                        "context_dependency": "",
                        "fit_for_subgoal_whole_proof": "",
                    },
                    "metrics": {
                        "statement_tokens": metrics["statement_tokens"],
                        "proof_tokens": metrics["proof_tokens"],
                        "tactic_steps": metrics["tactic_steps"],
                        "premise_count": metrics["premise_count"],
                        "same_file_premise_count": metrics[
                            "same_file_premise_count"
                        ],
                        "proof_style": metrics["proof_style"],
                    },
                    "trainable_for_short_whole_proof": None,
                    "reviewed_by": "codex",
                    "classification_version": "",
                    "classified_at": "",
                }
            )
        write_jsonl(root / "label_templates" / batch_path.name, templates)
    print(
        json.dumps(
            {
                "template_batches": len(batch_paths),
                "labels_assigned": 0,
                "purpose": "blank forms for actual Codex review",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
