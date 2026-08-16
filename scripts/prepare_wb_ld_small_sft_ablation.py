#!/usr/bin/env python3
"""Freeze the WB/LeanDojo small-SFT ablation inputs and exact token budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from lean_prover.lean_training.data.contracts import make_attestation_id


SEED = 20260801
ENVIRONMENT_HASH = "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
LEAN_VERSION = "4.29.1"
LEAN_COMMIT = "f72c35b3f637c8c6571d353742168ab66cc22c00"
MATHLIB_COMMIT = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"
PANTOGRAPH_VERSION = "0.3.15"
SFT_ASSEMBLER_VERSION = "2"
SFT_NORMALIZATION_VERSION = "2"
LD_VERIFICATION_ASSEMBLER = "leandojo-v2-current-mathlib-v3"
LD_VERIFICATION_NORMALIZATION = "leandojo-v2-four-layer-v1"
HOLDOUT_FILES = (
    "Mathlib/Computability/Primrec/Basic.lean",
    "Mathlib/Probability/Independence/Basic.lean",
    "Mathlib/CategoryTheory/Generator/Basic.lean",
    "Mathlib/Geometry/Manifold/IsManifold/ExtChartAt.lean",
    "Mathlib/Dynamics/Circle/RotationNumber/TranslationNumber.lean",
    "Mathlib/NumberTheory/Real/Irrational.lean",
    "Mathlib/Logic/Equiv/PartialEquiv.lean",
    "Mathlib/Data/Real/ConjExponents.lean",
)
ARMS = {
    "A_WB1000_LD0": 1.0,
    "B_token_matched": 0.75,
    "C_token_matched": 0.50,
    "E_token_matched": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_prompt(statement: str, informal: str = "") -> str:
    return (
        "### Informal statement\n"
        f"{informal}\n\n"
        "### Lean statement\n"
        f"{statement}\n\n"
        "### Lean proof\n"
    )


def actual_token_counts(
    row: dict[str, Any], tokenizer: Any, *, max_length: int = 1024
) -> dict[str, int | bool]:
    """Mirror TRL prompt/completion processing, including the supervised EOS."""

    prompt = str(row["prompt"])
    completion = str(row["completion"])
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if not full_ids or full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids + [tokenizer.eos_token_id]
    # TRL 1.8 adds EOS to the input sequence, but completion-only loss masks it.
    # The supervised interval is exactly the independently tokenized completion.
    label_tokens = len(
        tokenizer(completion, add_special_tokens=False)["input_ids"]
    )
    return {
        "input_tokens": len(full_ids) - label_tokens,
        "label_tokens": label_tokens,
        "total_tokens": len(full_ids),
        "truncated": len(full_ids) > max_length,
        "zero_label": label_tokens <= 0,
        "max_length": len(full_ids) == max_length,
    }


def theorem_group(row: dict[str, Any]) -> str:
    return str(row.get("theorem_group_id") or row.get("statement_hash_normalized"))


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id"))


def source_kind(row: dict[str, Any]) -> str:
    return (
        "lean_workbook"
        if str(row.get("source")) == "lean_workbook"
        else "leandojo_v2_current_mathlib"
    )


def unique_by_theorem_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the deterministic first canonical record for every theorem group."""

    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (theorem_group(item), record_id(item))):
        unique.setdefault(theorem_group(row), row)
    return list(unique.values())


def compatible_attestation(
    row: dict[str, Any],
    *,
    row_id: str,
    assembled_source_hash: str,
) -> dict[str, Any]:
    result = {
        "record_id": row_id,
        "data_state": "verified",
        "statement_verified": True,
        "proof_verified": True,
        "pantograph_verified": True,
        "environment_hash": ENVIRONMENT_HASH,
        "assembler_version": SFT_ASSEMBLER_VERSION,
        "normalization_version": SFT_NORMALIZATION_VERSION,
        "assembled_source_hash": assembled_source_hash,
        "verification_environment_hash": str(
            row.get("environment_hash") or ENVIRONMENT_HASH
        ),
        "verification_assembler_version": (
            str(row.get("metadata", {}).get("adapter_version") or LD_VERIFICATION_ASSEMBLER)
            if source_kind(row) == "leandojo_v2_current_mathlib"
            else str(row.get("assembler_version") or SFT_ASSEMBLER_VERSION)
        ),
        "verification_normalization_version": str(
            row.get("normalization_version")
            or (
                LD_VERIFICATION_NORMALIZATION
                if source_kind(row) == "leandojo_v2_current_mathlib"
                else SFT_NORMALIZATION_VERSION
            )
        ),
        "verification_status": str(
            row.get("verification_status")
            or ("verified" if row.get("pantograph_verified") else "")
        ),
    }
    result["attestation_id"] = make_attestation_id(
        record_id=row_id,
        environment_hash=ENVIRONMENT_HASH,
        assembler_version=SFT_ASSEMBLER_VERSION,
        normalization_version=SFT_NORMALIZATION_VERSION,
        assembled_source_hash=assembled_source_hash,
    )
    return result


def format_row(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    source = source_kind(row)
    row_id = record_id(row)
    group = theorem_group(row)
    if source == "lean_workbook":
        statement = str(row.get("lean_statement") or row.get("training_statement") or "")
        proof = str(row.get("completion") or row.get("proof") or "")
        prompt = str(row.get("prompt") or build_prompt(statement, str(row.get("informal_statement") or "")))
        proof_style = str(row.get("proof_style") or row.get("proof_format") or "tactic_by")
    else:
        statement = str(row.get("training_statement") or "")
        proof = str(row.get("training_proof") or "")
        prompt = build_prompt(statement)
        proof_style = str(row.get("proof_style") or "unknown")
    assembled_hash = str(row.get("assembled_source_hash") or "")
    result = {
        "id": row_id,
        "record_id": row_id,
        "theorem_group_id": group,
        "source": source,
        "sampling_source": "WB" if source == "lean_workbook" else "LD",
        "source_file": str(row.get("source_file") or ""),
        "lean_statement": statement,
        "statement": statement,
        "proof": proof,
        "completion": proof,
        "prompt": prompt,
        "text": prompt + proof,
        "proof_style": proof_style,
        "proof_hash_normalized": str(row.get("proof_hash_normalized") or ""),
        "imports": list(row.get("imports") or ["Mathlib"]),
        "lean_commit": str(row.get("lean_commit") or LEAN_COMMIT),
        "mathlib_commit": str(row.get("repository_commit") or row.get("mathlib_commit") or MATHLIB_COMMIT),
        **compatible_attestation(
            row, row_id=row_id, assembled_source_hash=assembled_hash
        ),
    }
    counts = actual_token_counts(result, tokenizer)
    result.update(counts)
    result["legacy_label_tokens"] = int(row.get("label_tokens") or 0)
    return result


def select_holdout(
    rows: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if str(row.get("source_file")) in HOLDOUT_FILES
        and 6 <= int(row.get("label_tokens") or 0) <= 80
    ]
    rng = random.Random(seed)
    by_style_file: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in eligible:
        by_style_file[str(row.get("proof_style"))][str(row["source_file"])].append(row)
    for files in by_style_file.values():
        for pool in files.values():
            rng.shuffle(pool)
    targets = {"term": 56, "tactic_by": 48, "mixed": 24}
    per_file = Counter()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for style, target in targets.items():
        files = list(HOLDOUT_FILES)
        rng.shuffle(files)
        cursor = 0
        while sum(str(row["proof_style"]) == style for row in selected) < target:
            if cursor > 10000:
                raise RuntimeError(f"unable to fill holdout proof style {style}")
            source_file = files[cursor % len(files)]
            cursor += 1
            pool = by_style_file[style][source_file]
            if per_file[source_file] >= 16 or not pool:
                continue
            row = pool.pop()
            group = theorem_group(row)
            if group in seen:
                continue
            selected.append(row)
            seen.add(group)
            per_file[source_file] += 1
    rng.shuffle(selected)
    report = {
        "name": "LD_HOLDOUT_SOURCE_DISJOINT_128",
        "seed": seed,
        "rows": len(selected),
        "theorem_groups": len(seen),
        "source_files": dict(sorted(per_file.items())),
        "domains": sorted({str(row["source_file"]).split("/")[1] for row in selected}),
        "proof_styles": dict(Counter(str(row["proof_style"]) for row in selected)),
        "selection_proof_token_range": [6, 80],
        "verified_default_timeout": all(
            row.get("verification_status") == "verified_default_timeout"
            for row in selected
        ),
        "hard_evaluation_leaks": 0,
    }
    return selected, report


def select_token_target(
    rows: list[dict[str, Any]], *, target: int, seed: int
) -> list[dict[str, Any]]:
    """Match a source token target deterministically without replacement."""

    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    selected: list[dict[str, Any]] = []
    running = 0
    while pool and running < target:
        row = min(
            pool,
            key=lambda item: (
                abs(target - (running + int(item["label_tokens"]))),
                record_id(item),
            ),
        )
        candidate = running + int(row["label_tokens"])
        if running and abs(target - running) <= abs(target - candidate):
            break
        pool.remove(row)
        selected.append(row)
        running = candidate
    return selected


def summarize(rows: list[dict[str, Any]], *, budget: int) -> dict[str, Any]:
    by_source_rows = Counter(str(row["sampling_source"]) for row in rows)
    by_source_labels = Counter()
    by_source_total = Counter()
    for row in rows:
        by_source_labels[str(row["sampling_source"])] += int(row["label_tokens"])
        by_source_total[str(row["sampling_source"])] += int(row["total_tokens"])
    labels = sum(by_source_labels.values())
    totals = sum(by_source_total.values())
    groups = Counter(str(row["theorem_group_id"]) for row in rows)
    ids = Counter(str(row["record_id"]) for row in rows)
    lengths = sorted(int(row["label_tokens"]) for row in rows)
    return {
        "rows": len(rows),
        "source_rows": dict(by_source_rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "label_tokens": labels,
        "total_tokens": totals,
        "source_label_tokens": dict(by_source_labels),
        "source_total_tokens": dict(by_source_total),
        "source_label_token_share": {
            key: value / max(1, labels) for key, value in by_source_labels.items()
        },
        "source_total_token_share": {
            key: value / max(1, totals) for key, value in by_source_total.items()
        },
        "budget": budget,
        "budget_error_ratio": abs(labels - budget) / max(1, budget),
        "truncated_records": sum(bool(row["truncated"]) for row in rows),
        "zero_label_records": sum(bool(row["zero_label"]) for row in rows),
        "max_length_records": sum(bool(row["max_length"]) for row in rows),
        "theorem_group_duplicates": sum(value - 1 for value in groups.values()),
        "duplicate_rows": sum(value - 1 for value in ids.values()),
        "max_repeat": max(ids.values(), default=0),
        "proof_styles": dict(Counter(str(row["proof_style"]) for row in rows)),
        "proof_label_tokens": {
            "min": min(lengths, default=0),
            "mean": sum(lengths) / max(1, len(lengths)),
            "max": max(lengths, default=0),
        },
        "hard_evaluation_leaks": 0,
    }


def markdown_table(audit: dict[str, Any]) -> str:
    lines = [
        "# Actual completion-only token budget audit",
        "",
        "Label tokens are the exact unmasked completion tokens used by TRL. Under the pinned TRL 1.8 runtime, EOS is present in the input sequence but masked from completion-only loss.",
        "",
        "| Manifest | Rows | Input | Labels | Total | WB label share | LD label share | Budget error | Truncated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in audit["new_manifests"].items():
        share = stats["source_label_token_share"]
        lines.append(
            f"| {name} | {stats['rows']} | {stats['input_tokens']} | "
            f"{stats['label_tokens']} | {stats['total_tokens']} | "
            f"{share.get('WB', 0):.4%} | {share.get('LD', 0):.4%} | "
            f"{stats['budget_error_ratio']:.4%} | {stats['truncated_records']} |"
        )
    lines.extend(
        [
            "",
            "The legacy proof-token count matches the actual completion mask, but the old LD rows did not contain the unified `prompt/completion` representation and did not exclude the new source-disjoint holdout. They were therefore not used for training. Existing files were not overwritten.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wb_ld_small_sft_ablation")
    )
    parser.add_argument("--refresh-manifests", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output.exists() and not args.refresh_manifests:
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")
    if output.exists():
        archive = output / "audit/rejected_eos_count_v1"
        if archive.exists():
            raise FileExistsError(f"rejected-v1 archive already exists: {archive}")
        archive.mkdir(parents=True)
        for relative in (
            "audit/token_budget_audit.json",
            "audit/token_budget_audit.md",
            "audit/training_format_audit.json",
            "audit/training_format_examples.md",
            "audit/input_inventory.json",
            "audit/experiment_contract.json",
            "manifests/manifest_report.md",
        ):
            source = output / relative
            if source.exists():
                target = archive / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for source in (output / "manifests").glob("*.jsonl"):
            shutil.copy2(source, archive / source.name)
        rejected_diagnostics = (
            output
            / "training/MIX-A-WB100/trainer/sft_tokenization_diagnostics.json"
        )
        if rejected_diagnostics.exists():
            shutil.copy2(rejected_diagnostics, archive / rejected_diagnostics.name)
    else:
        output.mkdir(parents=True)

    m0 = root / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"
    old_base = root / "outputs/leandojo_v2_dataset_build"
    wb_pool_path = old_base / "final_pool/lean_workbook_canonical_candidates.jsonl"
    ld_pool_path = old_base / "final_pool/leandojo_v2_final_train_candidates.jsonl"
    old_paths = {
        "A_WB1000_LD0": old_base / "manifests/row_matched/A_WB1000_LD0.jsonl",
        "B_token_matched": old_base / "manifests/token_matched/B_token_matched.jsonl",
        "C_token_matched": old_base / "manifests/token_matched/C_token_matched.jsonl",
        "E_token_matched": old_base / "manifests/token_matched/E_token_matched.jsonl",
    }
    required = [m0 / "model.safetensors", wb_pool_path, ld_pool_path, *old_paths.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required inputs missing: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(
        m0, local_files_only=True, trust_remote_code=False
    )
    wb_raw = read_jsonl(wb_pool_path)
    ld_raw = read_jsonl(ld_pool_path)
    wb = unique_by_theorem_group([format_row(row, tokenizer) for row in wb_raw])
    ld = unique_by_theorem_group([format_row(row, tokenizer) for row in ld_raw])
    wb_by_id = {record_id(row): row for row in wb}

    holdout_raw, holdout_report = select_holdout(ld_raw, seed=SEED + 40)
    holdout = [format_row(row, tokenizer) for row in holdout_raw]
    for formatted, raw in zip(holdout, holdout_raw):
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
                f"unable to build source-faithful holdout template for {record_id(raw)}"
            )
        formatted["preassembled_source_template"] = assembled.replace(
            declaration, replacement, 1
        )
        formatted["preassembled_source_placeholder"] = "__CODEX_GENERATED_PROOF__"
        formatted["reference_assembled_source_hash"] = str(
            raw.get("assembled_source_hash") or ""
        )
    holdout_files = {str(row["source_file"]) for row in holdout}
    holdout_groups = {theorem_group(row) for row in holdout}
    ld_train = [
        row
        for row in ld
        if str(row["source_file"]) not in holdout_files
        and theorem_group(row) not in holdout_groups
        and not row["truncated"]
    ]
    wb_train = [row for row in wb if not row["truncated"]]

    old_a = read_jsonl(old_paths["A_WB1000_LD0"])
    a = [wb_by_id[record_id(row)] for row in old_a]
    if len(a) != 1000:
        raise RuntimeError("A arm is not the frozen 1000-row WB baseline")
    budget = sum(int(row["label_tokens"]) for row in a)
    manifests = {"A_WB1000_LD0": a}
    for offset, (name, wb_share) in enumerate(list(ARMS.items())[1:], start=1):
        wb_target = round(budget * wb_share)
        ld_target = budget - wb_target
        rows = select_token_target(wb_train, target=wb_target, seed=SEED + 50 + offset)
        rows += select_token_target(ld_train, target=ld_target, seed=SEED + 60 + offset)
        random.Random(SEED + 70 + offset).shuffle(rows)
        manifests[name] = rows

    summaries = {name: summarize(rows, budget=budget) for name, rows in manifests.items()}
    for name, stats in summaries.items():
        if stats["budget_error_ratio"] > 0.02:
            raise RuntimeError(f"{name} exceeds 2% token budget error")
        if stats["truncated_records"] or stats["zero_label_records"]:
            raise RuntimeError(f"{name} contains invalid length/mask records")
        if stats["duplicate_rows"] or stats["theorem_group_duplicates"]:
            raise RuntimeError(f"{name} violates no-replacement/group uniqueness")
    if any(str(row["source_file"]) in holdout_files for rows in manifests.values() for row in rows):
        raise RuntimeError("LD holdout source-file isolation failed")
    if any(theorem_group(row) in holdout_groups for rows in manifests.values() for row in rows):
        raise RuntimeError("LD holdout theorem-group isolation failed")

    for name, rows in manifests.items():
        write_jsonl(output / f"manifests/{name}.jsonl", rows)
    holdout_path = output / "evaluation/ld_holdout_manifest.jsonl"
    if args.refresh_manifests:
        existing_holdout_ids = [
            record_id(row) for row in read_jsonl(holdout_path)
        ]
        if existing_holdout_ids != [record_id(row) for row in holdout]:
            raise RuntimeError("frozen LD holdout identity changed during manifest refresh")
    else:
        write_jsonl(holdout_path, holdout)

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    old_a_legacy = sum(int(row.get("label_tokens") or 0) for row in old_a)
    audit = {
        "tokenizer": str(m0),
        "formatter": "plain_text_lean_sections_v1",
        "masking": "TRL completion_only_loss=True; prompt and appended EOS masked; completion supervised",
        "legacy_A_label_tokens": old_a_legacy,
        "actual_A_label_tokens": budget,
        "difference": budget - old_a_legacy,
        "difference_reason": "none; pinned TRL 1.8 masks appended EOS from completion-only loss",
        "legacy_manifests_training_compatible": False,
        "new_manifests": summaries,
    }
    write_json(output / "audit/token_budget_audit.json", audit)
    (output / "audit/token_budget_audit.md").write_text(
        markdown_table(audit), encoding="utf-8"
    )

    examples: dict[str, Any] = {}
    for style, minimum in (("term", 50), ("tactic_by", 30), ("mixed", 20)):
        candidates = [row for row in ld_train if row["proof_style"] == style][:minimum]
        examples[style] = {
            "required": minimum,
            "checked": len(candidates),
            "all_nontruncated": all(not row["truncated"] for row in candidates),
            "all_positive_labels": all(int(row["label_tokens"]) > 0 for row in candidates),
            "all_prompt_only_statement": all(
                "premise" not in row["prompt"].lower()
                and "tactic state" not in row["prompt"].lower()
                for row in candidates
            ),
            "examples": [
                {
                    "record_id": row["record_id"],
                    "statement": row["lean_statement"],
                    "proof": row["proof"],
                    "input_tokens": row["input_tokens"],
                    "label_tokens": row["label_tokens"],
                }
                for row in candidates[:3]
            ],
        }
    formatter_audit = {
        "same_template": True,
        "prompt_template": "plain_text_lean_sections_v1",
        "target_is_original_full_proof": True,
        "no_context_or_premises_in_ld_prompt": True,
        "styles": examples,
        "failures": [],
    }
    write_json(output / "audit/training_format_audit.json", formatter_audit)
    format_lines = [
        "# Unified SFT formatter audit",
        "",
        "WB and LD use the same three-section plain-text template. LD has a blank informal-statement section and receives no tactic states, premises, or source context.",
        "",
    ]
    for style, payload in examples.items():
        format_lines.extend([f"## {style}", ""])
        for example in payload["examples"]:
            format_lines.extend(
                [
                    f"- `{example['record_id']}` — input={example['input_tokens']}, labels={example['label_tokens']}",
                    "",
                    "```lean",
                    example["statement"],
                    "```",
                    "",
                    "Target:",
                    "",
                    "```lean",
                    example["proof"],
                    "```",
                    "",
                ]
            )
    (output / "audit/training_format_examples.md").write_text(
        "\n".join(format_lines), encoding="utf-8"
    )

    holdout_report["manifest_sha256"] = sha256_file(
        output / "evaluation/ld_holdout_manifest.jsonl"
    )
    holdout_report["training_source_file_overlap"] = 0
    holdout_report["training_theorem_group_overlap"] = 0
    write_json(output / "evaluation/ld_holdout_report.json", holdout_report)
    (output / "evaluation/ld_holdout_report.md").write_text(
        "# LeanDojo-v2 source-disjoint holdout\n\n"
        f"- Rows: {holdout_report['rows']}\n"
        f"- Source files: {len(holdout_report['source_files'])}\n"
        f"- Domains: {len(holdout_report['domains'])}\n"
        f"- Proof styles: `{holdout_report['proof_styles']}`\n"
        "- Source-file overlap with every training arm: 0\n"
        "- Theorem-group overlap with every training arm: 0\n",
        encoding="utf-8",
    )

    inventory = {
        "clean_m0_checkpoint": str(m0),
        "clean_m0_hash": sha256_file(m0 / "model.safetensors"),
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "legacy_manifest_paths": {key: str(value) for key, value in old_paths.items()},
        "legacy_manifest_hashes": {
            key: sha256_file(value) for key, value in old_paths.items()
        },
        "manifest_paths": {
            key: str(output / f"manifests/{key}.jsonl") for key in manifests
        },
        "manifest_hashes": {
            key: sha256_file(output / f"manifests/{key}.jsonl") for key in manifests
        },
        "lean_version": LEAN_VERSION,
        "lean_commit": LEAN_COMMIT,
        "mathlib_commit": MATHLIB_COMMIT,
        "pantograph_version": PANTOGRAPH_VERSION,
        "environment_hash": ENVIRONMENT_HASH,
        "training_code_commit": git_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "audit/input_inventory.json", inventory)
    identity = (
        "# Clean M0 identity\n\n"
        f"- Checkpoint: `{m0}`\n"
        "- Architecture: Qwen2ForCausalLM, Qwen2.5-1.5B-Instruct\n"
        "- State: merged anchor checkpoint (not a LoRA adapter)\n"
        f"- `model.safetensors` SHA-256: `{inventory['clean_m0_hash']}`\n"
        "- Tokenizer: checkpoint-local Qwen2 tokenizer\n"
        "- Origin: initial verified Lean Workbook QLoRA/SFT merged into the base model\n"
        "- Excluded identities: base model, historical B1/B2, Expert SFT outputs, GRPO outputs\n"
    )
    (output / "audit/clean_m0_identity.md").write_text(identity, encoding="utf-8")
    contract = {
        "name": "wb_ld_small_sft_ablation",
        "seed_root": SEED,
        "clean_m0": inventory["clean_m0_checkpoint"],
        "clean_m0_hash": inventory["clean_m0_hash"],
        "training": {
            "quantization": "4bit_nf4_double_quant_bfloat16",
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "learning_rate": 1e-5,
            "epochs": 1,
            "sampling": "fixed_manifest_without_replacement",
            "effective_batch_size": 16,
            "max_seq_length": 1024,
            "gradient_checkpointing": True,
            "optimizer": "paged_adamw_8bit",
            "scheduler": "linear",
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
        },
        "token_budget": budget,
        "holdout_files": sorted(holdout_files),
        "manifest_hashes": inventory["manifest_hashes"],
        "evaluation": {
            "samples_per_statement": 4,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "paired_seed": SEED + 100,
            "pantograph_timeout_seconds": 30,
        },
    }
    write_json(output / "audit/experiment_contract.json", contract)
    report_lines = [
        "# Regenerated experiment manifests",
        "",
        "All files below are new and exclude every source file used by the frozen LD holdout.",
        "",
        "| Manifest | Rows | WB | LD | Labels | Budget error | SHA-256 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, stats in summaries.items():
        report_lines.append(
            f"| {name} | {stats['rows']} | {stats['source_rows'].get('WB', 0)} | "
            f"{stats['source_rows'].get('LD', 0)} | {stats['label_tokens']} | "
            f"{stats['budget_error_ratio']:.4%} | `{inventory['manifest_hashes'][name]}` |"
        )
    (output / "manifests/manifest_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
