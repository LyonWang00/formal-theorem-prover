#!/usr/bin/env python3
"""Compare aligned greedy outputs for one reproduction checkpoint."""

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


def common_prefix(left: list[int], right: list[int]) -> int:
    matched = 0
    for first, second in zip(left, right):
        if first != second:
            break
        matched += 1
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    for label in ("base", "adapter", "merged", "vllm"):
        parser.add_argument(f"--{label}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = ("base", "adapter", "merged", "vllm")
    tables = {label: read_jsonl(getattr(args, label)) for label in labels}
    counts = {label: len(rows) for label, rows in tables.items()}
    if len(set(counts.values())) != 1 or not counts["base"]:
        raise ValueError(f"unaligned equivalence row counts: {counts}")
    details = []
    for index in range(counts["base"]):
        rows = {label: tables[label][index] for label in labels}
        ids = {str(row.get("record_id")) for row in rows.values()}
        if len(ids) != 1:
            raise ValueError(f"prompt mismatch at row {index}: {ids}")
        tokens = {label: list(rows[label]["token_ids"]) for label in labels}
        details.append(
            {
                "record_id": next(iter(ids)),
                "base_adapter_exact": tokens["base"] == tokens["adapter"],
                "adapter_merged_exact": tokens["adapter"] == tokens["merged"],
                "merged_vllm_exact": tokens["merged"] == tokens["vllm"],
                "adapter_merged_prefix": common_prefix(
                    tokens["adapter"], tokens["merged"]
                ),
                "merged_vllm_prefix": common_prefix(
                    tokens["merged"], tokens["vllm"]
                ),
                "token_counts": {
                    label: len(values) for label, values in tokens.items()
                },
                "outputs": {
                    label: rows[label]["output_text"] for label in labels
                },
            }
        )
    count = len(details)
    report = {
        "success": (
            sum(not row["base_adapter_exact"] for row in details) > 0
            and sum(row["adapter_merged_prefix"] >= 1 for row in details) == count
            and sum(row["merged_vllm_prefix"] >= 1 for row in details) == count
        ),
        "generation_contract": {
            "prompts": count,
            "do_sample": False,
            "temperature": 0.0,
        },
        "base_adapter_different": sum(
            not row["base_adapter_exact"] for row in details
        ),
        "adapter_merged_exact_matches": sum(
            row["adapter_merged_exact"] for row in details
        ),
        "adapter_merged_first_token_matches": sum(
            row["adapter_merged_prefix"] >= 1 for row in details
        ),
        "merged_vllm_exact_matches": sum(
            row["merged_vllm_exact"] for row in details
        ),
        "merged_vllm_first_token_matches": sum(
            row["merged_vllm_prefix"] >= 1 for row in details
        ),
        "actual_paths": {
            label: tables[label][0].get("model_path") for label in labels
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Checkpoint equivalence",
        "",
        f"- Prompts: `{count}`",
        f"- Base differs from adapter: `{report['base_adapter_different']}/{count}`",
        f"- Adapter/merged exact: `{report['adapter_merged_exact_matches']}/{count}`",
        (
            "- Adapter/merged first token: "
            f"`{report['adapter_merged_first_token_matches']}/{count}`"
        ),
        f"- Merged/vLLM exact: `{report['merged_vllm_exact_matches']}/{count}`",
        (
            "- Merged/vLLM first token: "
            f"`{report['merged_vllm_first_token_matches']}/{count}`"
        ),
        f"- Contract success: `{report['success']}`",
        "",
    ]
    args.output.with_suffix(".md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
