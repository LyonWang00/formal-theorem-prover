#!/usr/bin/env python3
"""Produce source-aware composition and format audits for frozen anchor manifests."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer


ARMS = (
    "A0_WB3000_LD0",
    "A5_WB2750_LD250",
    "A10_WB2500_LD500",
    "A20_WB2000_LD1000",
)
TACTICS = (
    "simp",
    "simpa",
    "norm_num",
    "linarith",
    "nlinarith",
    "aesop",
    "omega",
    "ring",
    "ring_nf",
    "exact",
    "apply",
    "rw",
    "constructor",
    "cases",
    "induction",
)


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


def distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(ratio: float) -> int:
        return ordered[
            min(len(ordered) - 1, round((len(ordered) - 1) * ratio))
        ]

    return {
        "count": len(ordered),
        "mean": mean(ordered) if ordered else 0,
        "p50": percentile(0.50) if ordered else 0,
        "p90": percentile(0.90) if ordered else 0,
        "p95": percentile(0.95) if ordered else 0,
        "max": max(ordered, default=0),
    }


def tactic_usage(proofs: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for proof in proofs:
        matched = False
        for tactic in TACTICS:
            found = len(re.findall(rf"\b{re.escape(tactic)}\b", proof))
            if found:
                counts[tactic] += found
                matched = True
        if proof.strip() and not matched:
            counts["other"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/initial_anchor_ratio_ablation"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("models/Qwen2.5-1.5B-Instruct"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer.resolve(), local_files_only=True
    )
    audit: dict[str, Any] = {}
    sample_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        rows = read_jsonl(root / "manifests" / f"{arm}.jsonl")
        by_source: dict[str, list[dict[str, Any]]] = {
            source: [row for row in rows if row["sampling_source"] == source]
            for source in ("WB", "LD")
        }
        source_audit: dict[str, Any] = {}
        for source, selected in by_source.items():
            if not selected:
                continue
            statement_tokens = [
                len(
                    tokenizer(
                        str(row["statement"]),
                        add_special_tokens=False,
                    )["input_ids"]
                )
                for row in selected
            ]
            annotations = [
                row.get("difficulty_annotation") or {} for row in selected
            ]
            metrics = [annotation.get("metrics") or {} for annotation in annotations]
            premise_counts = [
                int(metric["premise_count"])
                for metric in metrics
                if metric.get("premise_count") is not None
            ]
            same_file_counts = [
                int(metric["same_file_premise_count"])
                for metric in metrics
                if metric.get("same_file_premise_count") is not None
            ]
            tactic_steps = [
                int(metric["tactic_steps"])
                for metric in metrics
                if metric.get("tactic_steps") is not None
            ]
            source_audit[source] = {
                "rows": len(selected),
                "ratio": len(selected) / len(rows),
                "manual_difficulty": dict(
                    Counter(
                        str(annotation.get("difficulty") or "not_available")
                        for annotation in annotations
                    )
                ),
                "manual_confidence": dict(
                    Counter(
                        str(annotation.get("confidence") or "not_available")
                        for annotation in annotations
                    )
                ),
                "proof_style": dict(
                    Counter(str(row.get("proof_style") or "unknown") for row in selected)
                ),
                "proof_tokens": distribution(
                    [int(row["label_tokens"]) for row in selected]
                ),
                "statement_tokens": distribution(statement_tokens),
                "tactic_steps": (
                    distribution(tactic_steps) if tactic_steps else "not_available"
                ),
                "premise_count": (
                    distribution(premise_counts)
                    if premise_counts
                    else "not_available"
                ),
                "same_file_premise_count": (
                    distribution(same_file_counts)
                    if same_file_counts
                    else "not_available"
                ),
                "source_file_count": len(
                    {str(row.get("source_file") or "") for row in selected}
                ),
                "domain_count": len(
                    {
                        str(row.get("source_file") or "").replace("\\", "/").split("/")[1]
                        for row in selected
                        if "/" in str(row.get("source_file") or "").replace("\\", "/")
                    }
                ),
                "tactic_usage": tactic_usage(
                    [str(row.get("proof") or "") for row in selected]
                ),
            }
        audit[arm] = {
            "rows": len(rows),
            "sources": source_audit,
            "format_contract": {
                "prompt_contains_reference_proof": sum(
                    bool(
                        str(row.get("prompt") or "").partition(
                            "### Lean proof\n"
                        )[2].strip()
                    )
                    for row in rows
                ),
                "completion_repeats_assignment": sum(
                    str(row.get("completion") or "").lstrip().startswith(":=")
                    for row in rows
                ),
                "completion_repeats_by": sum(
                    bool(
                        re.match(
                            r"^by\s+by\b",
                            str(row.get("completion") or "").strip(),
                            flags=re.DOTALL,
                        )
                    )
                    for row in rows
                ),
                "zero_label": sum(bool(row.get("zero_label")) for row in rows),
                "truncated": sum(bool(row.get("truncated")) for row in rows),
                "label_scope": "completion/proof only; exact count asserted by trainer",
            },
        }
        if arm == "A20_WB2000_LD1000":
            sample_rows["WB_50"] = by_source["WB"][:50]
            ld = by_source["LD"]
            sample_rows["LD_tactic_by_50"] = [
                row for row in ld if row.get("proof_style") == "tactic_by"
            ][:50]
            sample_rows["LD_term_20"] = [
                row for row in ld if row.get("proof_style") == "term"
            ][:20]
            sample_rows["LD_mixed_20"] = [
                row for row in ld if row.get("proof_style") == "mixed"
            ][:20]

    write_json(root / "audit/data_composition.json", audit)
    write_json(
        root / "audit/format_sample_audit.json",
        {
            name: [
                {
                    "record_id": row["record_id"],
                    "sampling_source": row["sampling_source"],
                    "proof_style": row.get("proof_style"),
                    "prompt": row.get("prompt"),
                    "completion": row.get("completion"),
                    "difficulty_annotation": row.get("difficulty_annotation"),
                }
                for row in rows
            ]
            for name, rows in sample_rows.items()
        },
    )
    print(json.dumps({"arms": list(audit), "sample_sizes": {
        key: len(value) for key, value in sample_rows.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
