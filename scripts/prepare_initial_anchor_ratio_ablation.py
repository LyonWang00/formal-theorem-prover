#!/usr/bin/env python3
"""Freeze audited 3000-row WB/LD-easy initial-anchor ablation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from scripts.prepare_initial_anchor_ld_expansion import (
    PROTECTED_PATHS,
    identities,
    normalized_statement,
    protected_index,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
)
from scripts.prepare_ld_easy_pilot import (
    annotation,
    default_timeout_verified,
    domain,
    replace_matched_wb,
    select_diverse,
)
from scripts.prepare_wb_ld_small_sft_ablation import (
    format_row,
    record_id,
    theorem_group,
    unique_by_theorem_group,
)


SEED = 42
ARMS = {
    "A0_WB3000_LD0": (3000, 0),
    "A5_WB2750_LD250": (2750, 250),
    "A10_WB2500_LD500": (2500, 500),
    "A20_WB2000_LD1000": (2000, 1000),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows: list[dict[str, Any]], target_labels: int) -> dict[str, Any]:
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    labels = sum(int(row["label_tokens"]) for row in rows)
    totals = sum(int(row["total_tokens"]) for row in rows)
    return {
        "rows": len(rows),
        "sources": dict(Counter(str(row["sampling_source"]) for row in rows)),
        "label_tokens": labels,
        "total_tokens": totals,
        "label_token_error_ratio": abs(labels - target_labels) / max(1, target_labels),
        "duplicate_rows": len(ids) - len(set(ids)),
        "theorem_group_duplicates": len(groups) - len(set(groups)),
        "max_repeat": max(Counter(ids).values(), default=0),
        "truncated_rows": sum(bool(row.get("truncated")) for row in rows),
        "zero_label_rows": sum(bool(row.get("zero_label")) for row in rows),
        "proof_styles": dict(Counter(str(row.get("proof_style")) for row in rows)),
        "source_files": len({str(row.get("source_file") or "") for row in rows}),
        "domains": len({domain(row) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/initial_anchor_ratio_ablation"),
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/Qwen2.5-1.5B-Instruct"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    output = (project / args.output).resolve()
    base_model = (project / args.base_model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, local_files_only=True, trust_remote_code=False
    )

    protected_ids, protected_statements, protected_inventory = protected_index(project)
    protected_rows: list[dict[str, Any]] = []
    for relative in PROTECTED_PATHS:
        for row in read_jsonl(project / relative):
            protected_rows.append(
                {
                    "dataset": relative,
                    "identities": sorted(identities(row)),
                    "normalized_statement": normalized_statement(row),
                    "theorem_group_id": str(row.get("theorem_group_id") or ""),
                    "qualified_name": str(row.get("qualified_name") or ""),
                    "source_file": str(row.get("source_file") or ""),
                    "source_span": row.get("source_span") or row.get("declaration_span"),
                }
            )
    write_json(
        output / "audit/protected_eval_index.json",
        {"datasets": protected_inventory, "rows": protected_rows},
    )

    wb_raw = read_jsonl(
        project / "data/processed/lean_workbook_verified_v2/train.jsonl"
    )
    for row in wb_raw:
        row["source"] = "lean_workbook"
    wb = [format_row(row, tokenizer) for row in wb_raw]
    for formatted, raw in zip(wb, wb_raw, strict=True):
        formatted["theorem_group_id"] = str(
            raw.get("theorem_group_id")
            or raw.get("statement_id")
            or raw.get("record_id")
        )
    if len(wb) != 3000:
        raise RuntimeError(f"WB baseline must contain 3000 theorem groups, got {len(wb)}")
    easy_path = output / "audit/ld_easy_expansion/trainable_easy_pool.jsonl"
    easy_raw = unique_by_theorem_group(
        [
            row
            for row in read_jsonl(easy_path)
            if annotation(row).get("difficulty") == "easy"
            and annotation(row).get("confidence") in {"high", "medium"}
            and annotation(row).get("trainable_for_short_whole_proof") is True
            and default_timeout_verified(row)
            and not (identities(row) & protected_ids)
            and normalized_statement(row) not in protected_statements
        ]
    )
    wb_groups = {theorem_group(row) for row in wb}
    easy_raw = [row for row in easy_raw if theorem_group(row) not in wb_groups]
    if len(easy_raw) < 1000:
        raise RuntimeError(f"A20 requires 1000 eligible LD-easy rows, got {len(easy_raw)}")
    selected_raw = select_diverse(easy_raw, count=1000, seed=SEED + 1)
    selected_ld = [format_row(row, tokenizer) for row in selected_raw]
    for formatted, raw in zip(selected_ld, selected_raw, strict=True):
        formatted["difficulty_annotation"] = annotation(raw)

    target_labels = sum(int(row["label_tokens"]) for row in wb)
    manifests: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, Any] = {}
    for offset, (name, (wb_count, ld_count)) in enumerate(ARMS.items()):
        ld_rows = selected_ld[:ld_count]
        rows = list(wb) if not ld_rows else replace_matched_wb(wb, ld_rows) + ld_rows
        random.Random(SEED + 10 + offset).shuffle(rows)
        stats = summarize(rows, target_labels)
        if stats["sources"] != {"WB": wb_count, **({"LD": ld_count} if ld_count else {})}:
            raise RuntimeError(f"{name} source contract failed: {stats['sources']}")
        if (
            stats["rows"] != 3000
            or stats["duplicate_rows"]
            or stats["theorem_group_duplicates"]
            or stats["max_repeat"] != 1
            or stats["truncated_rows"]
            or stats["zero_label_rows"]
        ):
            raise RuntimeError(f"{name} violates fixed manifest contract: {stats}")
        manifests[name] = rows
        audits[name] = stats
        write_jsonl(output / f"manifests/{name}.jsonl", rows)

    manifest_hashes = {
        name: file_sha256(output / f"manifests/{name}.jsonl") for name in manifests
    }
    base_hashes = {
        name: file_sha256(base_model / name)
        for name in ("config.json", "model.safetensors", "tokenizer.json")
    }
    audit = {
        "seed": SEED,
        "base_model": str(base_model),
        "base_hashes": base_hashes,
        "environment_hash": "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a",
        "lean_commit": "f72c35b3f637c8c6571d353742168ab66cc22c00",
        "mathlib_commit": "5e932f97dd25535344f80f9dd8da3aab83df0fe6",
        "pantograph": "0.3.15",
        "eligible_ld_easy": len(easy_raw),
        "selected_ld_easy": len(selected_ld),
        "automatic_difficulty_labels": 0,
        "protected_datasets": protected_inventory,
        "hard_evaluation_leaks": 0,
        "arms": audits,
        "manifest_hashes": manifest_hashes,
    }
    write_json(output / "audit/manifest_audit.json", audit)
    write_json(output / "manifests/manifest_hashes.json", manifest_hashes)
    leakage_lines = [
        "# Initial Anchor Leakage Audit",
        "",
        f"- Protected datasets: {len(protected_inventory)}",
        f"- Protected rows indexed: {len(protected_rows)}",
        "- Hard evaluation leaks: 0",
        "- Theorem-group duplicates within every arm: 0",
        "- Duplicate draws in every frozen manifest: 0",
        "- Automatic difficulty labels: 0",
        "",
    ]
    (output / "audit/leakage_audit.md").write_text(
        "\n".join(leakage_lines), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
