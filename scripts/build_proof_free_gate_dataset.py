#!/usr/bin/env python3
"""Build a fixed proof-free subset for model-generation hard gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


TARGET_FIELDS = {"proof", "completion", "text", "reference_proof"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proof_free_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    statement = str(row.get("lean_statement") or row.get("statement") or "")
    record_id = row.get("record_id") or row.get("id") or row.get("statement_id")
    if not prompt or not statement or not record_id:
        raise ValueError(f"record {record_id} lacks id, prompt, or statement")
    result = {
        "id": str(record_id),
        "statement_id": row.get("statement_id"),
        "prompt": prompt,
        "lean_statement": statement,
        "imports": list(row.get("imports") or ["Mathlib"]),
        "context_lines": list(
            row.get("context_lines")
            or (row.get("preamble") or {}).get("context_lines")
            or []
        ),
        "source_dataset": row.get("source_dataset"),
        "source_record_id": row.get("record_id") or row.get("id"),
        "statement_verified": bool(row.get("statement_verified")),
        "proof_verified_at_selection": bool(row.get("proof_verified")),
        "pantograph_verified_at_selection": bool(row.get("pantograph_verified")),
        "attestation_id": row.get("attestation_id"),
    }
    leaked = TARGET_FIELDS.intersection(result)
    if leaked:
        raise AssertionError(f"target fields leaked into gate row: {sorted(leaked)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=("fixed_random", "shortest_prompt"),
        default="fixed_random",
    )
    parser.add_argument("--require-reference-proof-verified", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    rows = read_jsonl(source)
    eligible = [
        (index, row)
        for index, row in enumerate(rows)
        if not args.require_reference_proof_verified
        or (
            row.get("reference_proof_verified") is True
            and bool(str(row.get("reference_proof") or "").strip())
        )
    ]
    if args.count <= 0 or len(eligible) < args.count:
        raise ValueError(
            f"requested {args.count} rows from {len(eligible)} eligible records "
            f"in source containing {len(rows)}"
        )
    if args.selection == "shortest_prompt":
        chosen = sorted(
            eligible,
            key=lambda item: (
                len(str(item[1].get("prompt") or "")),
                str(item[1].get("record_id") or item[1].get("id") or ""),
            ),
        )[: args.count]
        indices = sorted(index for index, _ in chosen)
    else:
        indices = sorted(
            random.Random(args.seed).sample(
                [index for index, _ in eligible], args.count
            )
        )
    selected = [proof_free_row(rows[index]) for index in indices]
    ids = [row["id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("selected gate records contain duplicate IDs")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "selection": args.selection,
        "seed": args.seed,
        "count": len(selected),
        "eligible_count": len(eligible),
        "require_reference_proof_verified": args.require_reference_proof_verified,
        "source_indices": indices,
        "target_fields_removed": sorted(TARGET_FIELDS),
        "proof_free": True,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
