#!/usr/bin/env python3
"""Freeze a manually reviewed LD-easy expansion pool for anchor ablation.

Auxiliary metrics are used only to prioritize which records Codex reads.  This
script never assigns or predicts a difficulty label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.prepare_ld_difficulty_review import (
    default_timeout_verified,
    render_dossier,
    review_metrics,
    source_domain,
)


SEED = 20260820
PROTECTED_PATHS = (
    "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
    "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/ld_easy_holdout_64.jsonl",
    "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
    "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl",
    "outputs/b2_expanded_validation/datasets/full500.jsonl",
    "outputs/b2_expanded_validation/datasets/strict_unseen_discovery.jsonl",
    "data/processed/lean_workbook_verified_v2/eval.jsonl",
)


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_statement(row: dict[str, Any]) -> str:
    value = str(
        row.get("lean_statement")
        or row.get("statement")
        or row.get("training_statement")
        or ""
    )
    return re.sub(r"\s+", " ", value).strip()


def identities(row: dict[str, Any]) -> set[str]:
    result = {
        str(row[key])
        for key in (
            "id",
            "record_id",
            "statement_id",
            "theorem_group_id",
            "qualified_name",
        )
        if row.get(key)
    }
    source_file = str(row.get("source_file") or "")
    source_span = row.get("source_span") or row.get("declaration_span")
    if source_file and source_span:
        result.add(f"source:{source_file}:{source_span}")
    return result


def protected_index(project: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    protected_ids: set[str] = set()
    protected_statements: set[str] = set()
    inventory: dict[str, Any] = {}
    for relative in PROTECTED_PATHS:
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing protected evaluation set: {path}")
        rows = read_jsonl(path)
        for row in rows:
            protected_ids.update(identities(row))
            statement = normalized_statement(row)
            if statement:
                protected_statements.add(statement)
        inventory[relative] = {"rows": len(rows), "sha256": sha256(path)}
    return protected_ids, protected_statements, inventory


def review_priority_candidate(metrics: dict[str, Any]) -> bool:
    """Select review order only; this is not a difficulty decision."""

    return bool(
        metrics["proof_tokens"] <= 30
        and metrics["premise_count"] <= 8
        and metrics["same_file_premise_count"] <= 1
        and metrics["tactic_steps"] <= 3
    )


def choose_diverse(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"need {count} review candidates, only {len(rows)} available")
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metrics = review_metrics(row)
        buckets[
            (
                source_domain(row),
                str(row.get("source_file") or ""),
                str(row.get("proof_style") or "unknown"),
            )
        ].append(row)
    for values in buckets.values():
        values.sort(
            key=lambda row: (
                review_metrics(row)["proof_tokens"],
                review_metrics(row)["premise_count"],
                str(row["id"]),
            )
        )
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < count:
        next_keys: list[tuple[str, str, str]] = []
        for key in keys:
            if len(selected) >= count:
                break
            selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
        rng.shuffle(keys)
    if len(selected) != count:
        raise RuntimeError("diverse review selection did not fill requested count")
    return selected


def template(row: dict[str, Any]) -> dict[str, Any]:
    metrics = review_metrics(row)
    return {
        "sample_id": str(row["id"]),
        "qualified_name": str(row.get("qualified_name") or ""),
        "source_file": str(row.get("source_file") or ""),
        "statement": str(row.get("statement") or row.get("training_statement") or ""),
        "training_statement": str(row.get("training_statement") or ""),
        "proof": str(row.get("proof") or row.get("training_proof") or ""),
        "training_proof": str(row.get("training_proof") or ""),
        "variables": list(row.get("variables") or []),
        "hypotheses": list(row.get("hypotheses") or []),
        "section_context": list(row.get("section_context") or []),
        "local_instances": list(row.get("local_instances") or []),
        "premises": list(row.get("premises") or []),
        "tactic_trace": list(row.get("tactic_trace") or []),
        "imports": list(row.get("imports") or []),
        "namespace_stack": list(row.get("namespace_stack") or []),
        "open_namespaces": list(row.get("open_namespaces") or []),
        "open_scopes": list(row.get("open_scopes") or []),
        "local_attributes": list(row.get("local_attributes") or []),
        "local_notations": list(row.get("local_notations") or []),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--refresh-templates", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion"
        ),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    output = (project / args.output).resolve()
    if output.exists() and not args.refresh_templates:
        raise FileExistsError(f"refusing to overwrite frozen expansion pool: {output}")
    if output.exists() and args.refresh_templates:
        pool_path = output / "frozen_expansion_pool.jsonl"
        selected = read_jsonl(pool_path)
        templates = [template(row) for row in selected]
        for start in range(0, len(templates), args.batch_size):
            batch = templates[start : start + args.batch_size]
            number = start // args.batch_size + 1
            write_jsonl(output / f"label_templates/batch_{number:03d}.jsonl", batch)
        print(
            json.dumps(
                {
                    "refreshed_templates": len(templates),
                    "frozen_pool_sha256": sha256(pool_path),
                    "selection_changed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    full_path = (
        project
        / "outputs/leandojo_v2_dataset_build/final_pool/"
        "leandojo_v2_final_train_candidates.jsonl"
    )
    labels_path = (
        project
        / "outputs/ld_length_difficulty_pipeline/difficulty/difficulty_labels.jsonl"
    )
    full = read_jsonl(full_path)
    reviewed = {str(row["sample_id"]) for row in read_jsonl(labels_path)}
    protected_ids, protected_statements, protected_inventory = protected_index(project)
    candidates: list[dict[str, Any]] = []
    excluded = Counter()
    for row in full:
        if str(row["id"]) in reviewed:
            excluded["previously_reviewed"] += 1
            continue
        if not default_timeout_verified(row):
            excluded["not_default_timeout_verified"] += 1
            continue
        if identities(row) & protected_ids or normalized_statement(row) in protected_statements:
            excluded["protected_evaluation_overlap"] += 1
            continue
        metrics = review_metrics(row)
        if not review_priority_candidate(metrics):
            excluded["outside_review_priority"] += 1
            continue
        candidates.append(row)
    selected = choose_diverse(candidates, count=args.count, seed=SEED)
    pool_path = output / "frozen_expansion_pool.jsonl"
    write_jsonl(pool_path, selected)
    templates = [template(row) for row in selected]
    for start in range(0, len(templates), args.batch_size):
        batch = templates[start : start + args.batch_size]
        number = start // args.batch_size + 1
        write_jsonl(output / f"label_templates/batch_{number:03d}.jsonl", batch)
        dossier_dir = output / "dossiers"
        dossier_dir.mkdir(parents=True, exist_ok=True)
        for raw, prepared in zip(
            selected[start : start + args.batch_size], batch, strict=True
        ):
            (dossier_dir / f"{raw['id']}.md").write_text(
                render_dossier(raw, prepared["metrics"]), encoding="utf-8"
            )
    write_json(
        output / "selection_audit.json",
        {
            "seed": SEED,
            "selection_purpose": "manual_review_order_only",
            "automatic_difficulty_labels": 0,
            "source_pool": str(full_path),
            "source_pool_sha256": sha256(full_path),
            "previous_labels_sha256": sha256(labels_path),
            "source_rows": len(full),
            "previously_reviewed": len(reviewed),
            "priority_candidates": len(candidates),
            "selected_rows": len(selected),
            "selected_source_files": len(
                {str(row.get("source_file") or "") for row in selected}
            ),
            "selected_domains": len({source_domain(row) for row in selected}),
            "proof_styles": dict(
                Counter(str(row.get("proof_style") or "unknown") for row in selected)
            ),
            "excluded": dict(excluded),
            "protected_sets": protected_inventory,
            "frozen_pool_sha256": sha256(pool_path),
            "batch_size": args.batch_size,
            "batches": (len(selected) + args.batch_size - 1) // args.batch_size,
        },
    )
    print((output / "selection_audit.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
