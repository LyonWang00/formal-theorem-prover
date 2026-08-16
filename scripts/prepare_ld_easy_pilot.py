#!/usr/bin/env python3
"""Build the supported audited pilot manifests from reviewed LD-easy samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from scripts.prepare_wb_ld_small_sft_ablation import (
    format_row,
    read_jsonl,
    record_id,
    theorem_group,
    unique_by_theorem_group,
    write_json,
    write_jsonl,
)


SEED = 20260803
ARMS = {
    "A2_WB1000": (1000, 0),
    "B2_WB900_LDE100": (900, 100),
    "C2_WB800_LDE200": (800, 200),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_statement(row: dict[str, Any]) -> str:
    return " ".join(
        str(
            row.get("statement")
            or row.get("lean_statement")
            or row.get("training_statement")
            or ""
        ).split()
    )


def domain(row: dict[str, Any]) -> str:
    parts = str(row.get("source_file") or "").split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "Mathlib" else "unknown"


def annotation(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("difficulty_annotation")
    return value if isinstance(value, dict) else {}


def default_timeout_verified(row: dict[str, Any]) -> bool:
    verification = row.get("verification")
    return bool(
        row.get("verification_status") == "verified_default_timeout"
        and isinstance(verification, dict)
        and verification.get("compile_success") is True
        and verification.get("timed_out") is False
        and not verification.get("error_category")
    )


def select_diverse(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Round-robin across domain/file/style; this never assigns difficulty."""

    if len(rows) < count:
        raise ValueError(f"need {count} rows, only {len(rows)} available")
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                domain(row),
                str(row.get("source_file") or ""),
                str(row.get("proof_style") or ""),
            )
        ].append(row)
    for values in buckets.values():
        values.sort(key=record_id)
        rng.shuffle(values)
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < count:
        next_keys: list[tuple[str, str, str]] = []
        for key in keys:
            if len(selected) >= count:
                break
            selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    if len(selected) != count:
        raise RuntimeError("diverse selection did not fill requested count")
    return selected


def choose_source_disjoint_holdout(
    rows: list[dict[str, Any]], *, seed: int, retain_train_rows: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Prefer whole source files while retaining every supported training row."""

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[str(row.get("source_file") or "")].append(row)
    rng = random.Random(seed)
    files = sorted(by_file)
    rng.shuffle(files)
    files.sort(key=lambda value: (len(by_file[value]), value))
    chosen: set[str] = set()
    available = len(rows)
    for source_file in files:
        size = len(by_file[source_file])
        if available - size < retain_train_rows:
            continue
        chosen.add(source_file)
        available -= size
        if sum(len(by_file[value]) for value in chosen) >= 64:
            candidates = [
                row
                for value in chosen
                for row in by_file[value]
            ]
            holdout = select_diverse(candidates, count=64, seed=seed + 1)
            train = [
                row
                for row in rows
                if str(row.get("source_file") or "") not in chosen
            ]
            return holdout, train, True
    holdout = select_diverse(rows, count=64, seed=seed + 2)
    holdout_groups = {theorem_group(row) for row in holdout}
    train = [row for row in rows if theorem_group(row) not in holdout_groups]
    return holdout, train, False


def add_source_template(formatted: dict[str, Any], raw: dict[str, Any]) -> None:
    assembled = str((raw.get("metadata") or {}).get("assembled_source") or "")
    declaration = str(
        raw.get("declaration_source")
        or raw.get("raw_declaration_source")
        or raw.get("training_declaration")
        or ""
    )
    replacement = (
        f"{raw.get('training_statement')} :=\n"
        "__CODEX_GENERATED_PROOF__"
    )
    if not assembled or not declaration or assembled.count(declaration) != 1:
        raise RuntimeError(
            f"cannot build source-faithful template for {record_id(raw)}"
        )
    formatted["preassembled_source_template"] = assembled.replace(
        declaration, replacement, 1
    )
    formatted["preassembled_source_placeholder"] = "__CODEX_GENERATED_PROOF__"
    formatted["reference_assembled_source_hash"] = str(
        raw.get("assembled_source_hash") or ""
    )


def replace_matched_wb(
    baseline: list[dict[str, Any]],
    ld_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove one closest WB row per LD row to preserve label-token budget."""

    remaining = list(baseline)
    for ld_row in ld_rows:
        ld_tokens = int(ld_row["label_tokens"])
        ld_style = str(ld_row.get("proof_style") or "")
        chosen = min(
            remaining,
            key=lambda wb: (
                str(wb.get("proof_style") or "") != ld_style,
                abs(int(wb["label_tokens"]) - ld_tokens),
                record_id(wb),
            ),
        )
        remaining.remove(chosen)
    return remaining


def summarize(rows: list[dict[str, Any]], *, target: int) -> dict[str, Any]:
    labels = sum(int(row["label_tokens"]) for row in rows)
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    return {
        "rows": len(rows),
        "source_rows": dict(Counter(str(row["sampling_source"]) for row in rows)),
        "label_tokens": labels,
        "target_label_tokens": target,
        "label_token_budget_error_ratio": abs(labels - target) / max(1, target),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "duplicate_rows": len(ids) - len(set(ids)),
        "theorem_group_duplicates": len(groups) - len(set(groups)),
        "max_repeat": max(Counter(ids).values(), default=0),
        "proof_styles": dict(Counter(str(row.get("proof_style")) for row in rows)),
        "source_files": len({str(row.get("source_file") or "") for row in rows}),
        "domains": len({domain(row) for row in rows}),
        "truncated": sum(bool(row.get("truncated")) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--difficulty-dir",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/difficulty"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/pilot_sft"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    difficulty = (root / args.difficulty_dir).resolve()
    output = (root / args.output).resolve()
    gate = json.loads((difficulty / "pilot_gate.json").read_text(encoding="utf-8"))
    if not gate.get("pilot_gate_passed"):
        raise RuntimeError("manual difficulty gate did not pass")
    model = (root / args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )

    easy_raw = read_jsonl(difficulty / "easy_pool.jsonl")
    eligible = unique_by_theorem_group(
        [
            row
            for row in easy_raw
            if annotation(row).get("difficulty") == "easy"
            and annotation(row).get("confidence") in {"high", "medium"}
            and annotation(row).get("trainable_for_short_whole_proof") is True
            and default_timeout_verified(row)
        ]
    )
    if len(eligible) < 64:
        raise RuntimeError(
            "fewer than 64 eligible LD-easy rows; cannot freeze the holdout"
        )
    maximum_train_rows = len(eligible) - 64
    supported_arms = {
        name: counts
        for name, counts in ARMS.items()
        if counts[1] <= maximum_train_rows
    }
    skipped_arms = {
        name: (
            f"requires {counts[1]} LD-easy training rows plus 64 holdout rows, "
            f"but only {len(eligible)} eligible rows are frozen"
        )
        for name, counts in ARMS.items()
        if name not in supported_arms
    }
    required_train_rows = max((counts[1] for counts in supported_arms.values()), default=0)
    holdout_raw, train_raw, source_disjoint = choose_source_disjoint_holdout(
        eligible,
        seed=SEED,
        retain_train_rows=required_train_rows,
    )
    if len(train_raw) < required_train_rows:
        raise RuntimeError(
            f"only {len(train_raw)} LD-easy rows remain after holdout, "
            f"but supported arms require {required_train_rows}"
        )
    ld_train_raw = select_diverse(
        train_raw, count=required_train_rows, seed=SEED + 10
    )
    holdout = [format_row(row, tokenizer) for row in holdout_raw]
    for formatted, raw in zip(holdout, holdout_raw, strict=True):
        add_source_template(formatted, raw)
        formatted["difficulty_annotation"] = annotation(raw)
    ld_train = [format_row(row, tokenizer) for row in ld_train_raw]
    for formatted, raw in zip(ld_train, ld_train_raw, strict=True):
        formatted["difficulty_annotation"] = annotation(raw)

    baseline_path = (
        root
        / "outputs/wb_ld_small_sft_ablation/manifests/A_WB1000_LD0.jsonl"
    )
    baseline = read_jsonl(baseline_path)
    if len(baseline) != 1000:
        raise RuntimeError("frozen WB baseline is not exactly 1000 rows")
    target_budget = sum(int(row["label_tokens"]) for row in baseline)
    manifests: dict[str, list[dict[str, Any]]] = {}
    for name, (_wb_count, ld_count) in supported_arms.items():
        selected_ld = ld_train[:ld_count]
        manifests[name] = (
            list(baseline)
            if not selected_ld
            else replace_matched_wb(baseline, selected_ld) + selected_ld
        )
    for offset, rows in enumerate(manifests.values()):
        random.Random(SEED + 20 + offset).shuffle(rows)

    holdout_ids = {record_id(row) for row in holdout}
    holdout_groups = {theorem_group(row) for row in holdout}
    holdout_statements = {normalized_statement(row) for row in holdout}
    summaries: dict[str, Any] = {}
    for name, rows in manifests.items():
        expected_wb, expected_ld = ARMS[name]
        stats = summarize(rows, target=target_budget)
        counts = Counter(str(row["sampling_source"]) for row in rows)
        if len(rows) != 1000 or counts["WB"] != expected_wb or counts["LD"] != expected_ld:
            raise RuntimeError(f"{name} source/row contract failed: {counts}")
        if (
            stats["duplicate_rows"]
            or stats["theorem_group_duplicates"]
            or stats["max_repeat"] != 1
            or stats["truncated"]
        ):
            raise RuntimeError(f"{name} violates no-replacement/length contract")
        if stats["label_token_budget_error_ratio"] > 0.05:
            raise RuntimeError(f"{name} exceeds 5% label-token budget error")
        for row in rows:
            if (
                record_id(row) in holdout_ids
                or theorem_group(row) in holdout_groups
                or normalized_statement(row) in holdout_statements
            ):
                raise RuntimeError(f"{name} leaks LD_EASY_HOLDOUT_64")
        summaries[name] = stats
        write_jsonl(output / f"manifests/{name}.jsonl", rows)

    write_jsonl(output / "evaluation/ld_easy_holdout_64.jsonl", holdout)
    audit = {
        "seed": SEED,
        "clean_m0_path": str(model),
        "clean_m0_model_sha256": sha256(model / "model.safetensors"),
        "difficulty_labels_sha256": sha256(difficulty / "difficulty_labels.jsonl"),
        "easy_pool_sha256": sha256(difficulty / "easy_pool.jsonl"),
        "baseline_manifest_sha256": sha256(baseline_path),
        "manual_gate": gate,
        "eligible_trainable_easy": len(eligible),
        "supported_arms": list(supported_arms),
        "skipped_arms": skipped_arms,
        "holdout_rows": len(holdout),
        "holdout_source_file_disjoint": source_disjoint,
        "holdout_source_files": sorted(
            {str(row.get("source_file") or "") for row in holdout}
        ),
        "holdout_domains": sorted({domain(row) for row in holdout}),
        "training_easy_rows": len(ld_train),
        "hard_evaluation_leaks": 0,
        "target_label_tokens": target_budget,
        "manifests": summaries,
    }
    write_json(output / "audit/pilot_manifest_audit.json", audit)
    hashes = {
        name: sha256(output / f"manifests/{name}.jsonl")
        for name in manifests
    }
    hashes["LD_EASY_HOLDOUT_64"] = sha256(
        output / "evaluation/ld_easy_holdout_64.jsonl"
    )
    write_json(output / "audit/input_inventory.json", hashes)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
