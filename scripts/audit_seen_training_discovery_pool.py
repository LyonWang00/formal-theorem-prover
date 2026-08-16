#!/usr/bin/env python3
"""Audit the M0+H0 seen-training pool for the revised EI Discovery contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def statement(row: dict[str, Any]) -> str:
    return str(
        row.get("statement")
        or row.get("lean_statement")
        or row.get("training_statement")
        or ""
    ).strip()


def proof(row: dict[str, Any]) -> str:
    return str(row.get("proof") or row.get("completion") or "").strip()


def source_kind(row: dict[str, Any]) -> str:
    explicit = str(row.get("sampling_source") or "").upper()
    text = " ".join(
        str(row.get(key) or "")
        for key in ("source", "source_dataset", "source_name", "record_id", "id")
    ).lower()
    if explicit == "LD" or "leandojo" in text or "ldv2_" in text:
        return "LD"
    if explicit == "WB" or "workbook" in text:
        return "WB"
    return "unknown"


def axes(row: dict[str, Any]) -> dict[str, str]:
    record = str(row.get("record_id") or row.get("id") or "").strip()
    group = str(row.get("theorem_group_id") or row.get("statement_id") or "").strip()
    qualified = str(
        row.get("qualified_name")
        or row.get("theorem_name")
        or row.get("source_declaration")
        or ""
    ).strip()
    stmt = statement(row)
    pf = norm(proof(row))
    return {
        "record_id": record,
        "theorem_group": group,
        "qualified_theorem": qualified,
        "exact_statement": stmt,
        "normalized_statement": norm(stmt),
        "proof_variant": hashlib.sha256(pf.encode("utf-8")).hexdigest() if pf else "",
    }


def axis_sets(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for key, value in axes(row).items():
            if value:
                result[key].add(value)
    return dict(result)


def overlaps(row: dict[str, Any], forbidden: dict[str, set[str]]) -> list[str]:
    return [
        key
        for key, value in axes(row).items()
        if value and value in forbidden.get(key, set())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/expert_iteration/difficulty_ablation_500rows/seen_training_pool"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    sources = {
        "M0": project
        / "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/ADDON-B-WB2000-LD1000.jsonl",
        "H0": project
        / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/source_manifest.jsonl",
    }
    protected_index_path = (
        project / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json"
    )
    protected_index = read_json(protected_index_path)
    protected_paths: dict[str, Path] = {}
    for name, item in protected_index.items():
        value = item.get("path") if isinstance(item, dict) else None
        if value and Path(str(value)).is_file():
            protected_paths[name] = Path(str(value))
    protected_paths["ld_medium_holdout64"] = (
        project
        / "outputs/stage2_data_ratio_ablation/phaseC3_ld_medium/evaluation/datasets/ld_medium_holdout64.jsonl"
    )
    protected_rows = {
        name: read_jsonl(path) for name, path in protected_paths.items() if path.is_file()
    }
    forbidden = axis_sets(row for rows in protected_rows.values() for row in rows)

    provenance: dict[str, set[str]] = defaultdict(set)
    candidates_by_group: dict[str, dict[str, Any]] = {}
    raw_counts: dict[str, Any] = {}
    excluded = Counter()
    protected_overlap_by_set: dict[str, Counter[str]] = {
        name: Counter() for name in protected_rows
    }
    for checkpoint, path in sources.items():
        rows = read_jsonl(path)
        raw_counts[checkpoint] = {
            "rows": len(rows),
            "source": dict(Counter(source_kind(row) for row in rows)),
            "sha256": sha256(path),
        }
        for row in rows:
            row_axes = axes(row)
            group = row_axes["theorem_group"] or row_axes["record_id"] or row_axes["normalized_statement"]
            provenance[group].add(checkpoint)
            if source_kind(row) not in {"WB", "LD"}:
                excluded["unknown_source"] += 1
                continue
            if not statement(row) or not proof(row):
                excluded["missing_statement_or_proof"] += 1
                continue
            if row.get("pantograph_verified") is not True:
                excluded["not_pantograph_verified"] += 1
                continue
            hit = overlaps(row, forbidden)
            if hit:
                excluded["protected_evaluation_overlap"] += 1
                for name, eval_rows in protected_rows.items():
                    eval_sets = axis_sets(eval_rows)
                    for axis in overlaps(row, eval_sets):
                        protected_overlap_by_set[name][axis] += 1
                continue
            current = candidates_by_group.get(group)
            if current is None or checkpoint == "H0":
                candidates_by_group[group] = dict(row)

    candidates: list[dict[str, Any]] = []
    for group, row in candidates_by_group.items():
        enriched = dict(row)
        enriched["seen_in_checkpoints"] = sorted(provenance[group])
        enriched["discovery_source_contract"] = "previously_used_training_data_only_v1"
        candidates.append(enriched)
    candidates.sort(key=lambda row: (source_kind(row), axes(row)["theorem_group"], axes(row)["record_id"]))

    duplicate_records = len(candidates) - len({axes(row)["record_id"] for row in candidates})
    duplicate_groups = len(candidates) - len({axes(row)["theorem_group"] for row in candidates})
    audit = {
        "status": "SEEN_TRAINING_DISCOVERY_POOL_READY",
        "contract": {
            "allowed": ["M0 training manifest", "H0 training manifest"],
            "forbidden": [
                "new unused WB",
                "new LD three-tier pools",
                "all protected evaluation identities",
            ],
            "reference_proof_policy": "retained only in private source pool; removed from public discovery prompts",
        },
        "source_manifests": {name: str(path) for name, path in sources.items()},
        "raw": raw_counts,
        "unique_eligible_rows": len(candidates),
        "eligible_source_counts": dict(Counter(source_kind(row) for row in candidates)),
        "seen_provenance_counts": dict(
            Counter("+".join(row["seen_in_checkpoints"]) for row in candidates)
        ),
        "excluded": dict(excluded),
        "duplicate_record_ids": duplicate_records,
        "theorem_group_duplicates": duplicate_groups,
        "protected_overlap_by_set_before_exclusion": {
            name: dict(counts) for name, counts in protected_overlap_by_set.items()
        },
        "protected_overlap_after_exclusion": 0,
        "output_manifest": str(output / "seen_training_source_pool.jsonl"),
    }
    if duplicate_records or duplicate_groups or any(source_kind(row) == "unknown" for row in candidates):
        raise RuntimeError("seen-training pool duplicate/source gate failed")
    write_jsonl(output / "seen_training_source_pool.jsonl", candidates)
    audit["output_sha256"] = sha256(output / "seen_training_source_pool.jsonl")
    write_json(output / "seen_training_pool_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
