"""Read-only audit of incomplete EI Round 0 verification tasks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    manifest = read_jsonl(root / "discovery_manifest.jsonl")
    generations = read_jsonl(root / "candidate_generations.jsonl")
    completed = {
        row["generation_id"]
        for row in read_jsonl(root / "candidate_verifications.jsonl")
    }
    by_prompt = {row["prompt"]: row for row in manifest}
    missing = [row for row in generations if row["generation_id"] not in completed]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in missing:
        grouped[str(row["statement_id"])].append(row)
    statements: list[dict[str, Any]] = []
    for statement_id, rows in grouped.items():
        source = by_prompt.get(str(rows[0].get("prompt") or ""), {})
        statements.append(
            {
                "statement_id": statement_id,
                "source_id": source.get("id"),
                "source": source.get("source"),
                "category": source.get("category"),
                "proof_length_bin": source.get("proof_length_bin"),
                "source_template": bool(source.get("preassembled_source_template")),
                "source_template_chars": len(
                    str(source.get("preassembled_source_template") or "")
                ),
                "candidates": len(rows),
                "extracted": sum(bool(row.get("extracted_proof")) for row in rows),
                "length_finishes": sum(row.get("finish_reason") == "length" for row in rows),
                "completion_tokens": [
                    (row.get("metadata") or {}).get("completion_tokens") for row in rows
                ],
                "generation_ids": [row["generation_id"] for row in rows],
            }
        )
    payload = {
        "missing_candidates": len(missing),
        "missing_statements": len(grouped),
        "source_counts": dict(
            Counter(str(row.get("source") or "unknown") for row in statements)
        ),
        "source_template_statements": sum(row["source_template"] for row in statements),
        "finish_reasons": dict(Counter(row.get("finish_reason") for row in missing)),
        "shortest_candidates": [
            {
                "generation_id": row["generation_id"],
                "statement_id": row["statement_id"],
                "completion_tokens": (row.get("metadata") or {}).get(
                    "completion_tokens"
                ),
                "finish_reason": row.get("finish_reason"),
                "extracted_proof": row.get("extracted_proof"),
            }
            for row in sorted(
                missing,
                key=lambda item: (
                    int((item.get("metadata") or {}).get("completion_tokens") or 10**9),
                    str(item.get("statement_id") or ""),
                    int(item.get("sample_index") or 0),
                ),
            )[:10]
        ],
        "statements": statements,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
