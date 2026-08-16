#!/usr/bin/env python3
"""Freeze the adapted WB/LD budget-support ablation training manifests."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from lean_prover.lean_training.sft_pipeline.trainer import (
    assert_supervised_eos_contract,
)

from scripts.audit_wb_ld_budget_support_preflight import (
    SOURCE_MANIFESTS,
    normalized_statement,
    overlap,
    read_jsonl,
    record_id,
    sha256,
    theorem_group,
    write_json,
    write_jsonl,
)


SEED = 42
EXPECTED_BASE_HASH = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
EXPECTED_ENVIRONMENT_HASH = (
    "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
)
OUTPUT = Path("outputs/wb_ld_budget_support_replay_ablation")
ARMS = {
    "BUDGET-A-WB2000": "manifests/fixed_budget/BUDGET-A-WB2000.jsonl",
    "BUDGET-A-WB1500-LD500": (
        "manifests/fixed_budget/BUDGET-A-WB1500-LD500.jsonl"
    ),
    "BUDGET-A-WB1000-LD1000": (
        "manifests/fixed_budget/BUDGET-A-WB1000-LD1000.jsonl"
    ),
    "ADDON-B-WB2000-LD1000": (
        "manifests/fixed_wb/ADDON-B-WB2000-LD1000.jsonl"
    ),
    "SUPPORT-C-LD1000": "manifests/fixed_ld/SUPPORT-C-LD1000.jsonl",
}


def hash_values(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def frozen_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        existing_ids = [record_id(row) for row in read_jsonl(path)]
        proposed_ids = [record_id(row) for row in rows]
        if existing_ids != proposed_ids:
            raise RuntimeError(f"refusing to replace changed frozen manifest: {path}")
        return
    write_jsonl(path, rows)


def source_rows(project: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for arm, (relative, expected) in SOURCE_MANIFESTS.items():
        path = project / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{arm} source hash changed: {actual} != {expected}")
        result[arm] = read_jsonl(path)
    return result


def local_search_subset(
    pool: list[dict[str, Any]],
    *,
    count: int,
    target_labels: int,
    target_total: int,
    seed: int,
    iterations: int = 120_000,
) -> list[dict[str, Any]]:
    """Choose a deterministic no-replacement length-targeted subset."""
    if count > len(pool):
        raise ValueError("subset count exceeds pool")
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)

    label_scale = max(1, target_labels)
    total_scale = max(1, target_total)

    def row_score(row: dict[str, Any]) -> float:
        return (
            int(row["label_tokens"]) / label_scale
            + int(row["total_tokens"]) / total_scale
        )

    # Start from long examples because LD replacements are substantially
    # shorter than WB. This is the best feasible direction for token matching.
    selected = sorted(
        shuffled,
        key=lambda row: (row_score(row), record_id(row)),
        reverse=True,
    )[:count]
    selected_ids = {record_id(row) for row in selected}
    unselected = [row for row in shuffled if record_id(row) not in selected_ids]
    labels = sum(int(row["label_tokens"]) for row in selected)
    totals = sum(int(row["total_tokens"]) for row in selected)

    def objective(label_sum: int, total_sum: int) -> float:
        return (
            abs(label_sum - target_labels) / label_scale
            + abs(total_sum - target_total) / total_scale
        )

    current = objective(labels, totals)
    for _ in range(iterations):
        if not selected or not unselected:
            break
        old_index = rng.randrange(len(selected))
        new_index = rng.randrange(len(unselected))
        old = selected[old_index]
        new = unselected[new_index]
        candidate_labels = (
            labels - int(old["label_tokens"]) + int(new["label_tokens"])
        )
        candidate_totals = (
            totals - int(old["total_tokens"]) + int(new["total_tokens"])
        )
        candidate = objective(candidate_labels, candidate_totals)
        if candidate < current:
            selected[old_index], unselected[new_index] = new, old
            labels, totals, current = candidate_labels, candidate_totals, candidate
    order = {record_id(row): index for index, row in enumerate(pool)}
    return sorted(selected, key=lambda row: order[record_id(row)])


def tactic_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    tactics = ("simp", "norm_num", "linarith", "nlinarith", "aesop", "omega", "ring")
    counts: Counter[str] = Counter()
    for row in rows:
        proof = str(row.get("proof") or row.get("completion") or "")
        matched = False
        for tactic in tactics:
            if re.search(rf"\b{re.escape(tactic)}\b", proof):
                counts[tactic] += 1
                matched = True
        if not matched:
            counts["other"] += 1
    return dict(sorted(counts.items()))


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(
    rows: list[dict[str, Any]],
    *,
    reference_labels: int,
    reference_total: int,
) -> dict[str, Any]:
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    labels = sum(int(row["label_tokens"]) for row in rows)
    inputs = sum(int(row["input_tokens"]) for row in rows)
    totals = sum(int(row["total_tokens"]) for row in rows)
    proof_lengths = [int(row["label_tokens"]) for row in rows]
    statement_lengths = [
        max(0, int(row["input_tokens"])) for row in rows
    ]
    sources = Counter(str(row.get("sampling_source") or "") for row in rows)
    return {
        "rows": len(rows),
        "wb_rows": sources.get("WB", 0),
        "ld_rows": sources.get("LD", 0),
        "optimizer_steps_at_effective_batch_16": (len(rows) + 15) // 16,
        "input_tokens": inputs,
        "label_tokens": labels,
        "total_tokens": totals,
        "label_token_error_vs_A_WB2000": (
            abs(labels - reference_labels) / max(1, reference_labels)
        ),
        "total_token_error_vs_A_WB2000": (
            abs(totals - reference_total) / max(1, reference_total)
        ),
        "proof_tokens": {
            "mean": sum(proof_lengths) / max(1, len(proof_lengths)),
            "p50": percentile(proof_lengths, 0.50),
            "p90": percentile(proof_lengths, 0.90),
            "p95": percentile(proof_lengths, 0.95),
            "max": max(proof_lengths, default=0),
        },
        "statement_tokens": {
            "mean": sum(statement_lengths) / max(1, len(statement_lengths)),
            "p50": percentile(statement_lengths, 0.50),
            "p90": percentile(statement_lengths, 0.90),
            "p95": percentile(statement_lengths, 0.95),
            "max": max(statement_lengths, default=0),
        },
        "tactic_distribution": tactic_distribution(rows),
        "source_files": len({str(row.get("source_file") or "") for row in rows}),
        "premise_count": sum(
            int((row.get("difficulty_annotation") or {}).get("metrics", {}).get(
                "premise_count", 0
            ))
            for row in rows
        ),
        "duplicate_rows": len(ids) - len(set(ids)),
        "theorem_group_duplicates": len(groups) - len(set(groups)),
        "max_repeat": max(Counter(ids).values(), default=0),
        "zero_label_rows": sum(bool(row.get("zero_label")) for row in rows),
        "semantic_truncation_rows": sum(bool(row.get("truncated")) for row in rows),
        "all_pantograph_verified": all(
            row.get("pantograph_verified") is True for row in rows
        ),
    }


def main() -> None:
    project = Path(".").resolve()
    output = project / OUTPUT
    base = project / "models/Qwen2.5-1.5B-Instruct"
    if sha256(base / "model.safetensors") != EXPECTED_BASE_HASH:
        raise RuntimeError("base model identity changed")

    preflight_path = output / "audit/preflight_split_audit.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASSED" or preflight.get("blockers"):
        raise RuntimeError("adapted protected-split preflight has not passed")

    sources = source_rows(project)
    core_wb = [
        row for row in sources["A20"] if row.get("sampling_source") == "WB"
    ]
    core_ld1000 = [
        row for row in sources["A20"] if row.get("sampling_source") == "LD"
    ]
    core_ld500 = [
        row for row in sources["A10"] if row.get("sampling_source") == "LD"
    ]
    core_ld250 = [
        row for row in sources["A5"] if row.get("sampling_source") == "LD"
    ]
    core_wb_ids = {record_id(row) for row in core_wb}
    a0_wb = [
        row for row in sources["A0"] if row.get("sampling_source") == "WB"
    ]
    extra_wb = [row for row in a0_wb if record_id(row) not in core_wb_ids]
    if tuple(map(len, (core_wb, extra_wb, core_ld250, core_ld500, core_ld1000))) != (
        2000,
        1000,
        250,
        500,
        1000,
    ):
        raise RuntimeError("frozen core counts changed")

    reference_labels = sum(int(row["label_tokens"]) for row in core_wb)
    reference_total = sum(int(row["total_tokens"]) for row in core_wb)
    ld1000_labels = sum(int(row["label_tokens"]) for row in core_ld1000)
    ld1000_total = sum(int(row["total_tokens"]) for row in core_ld1000)
    wb1000 = local_search_subset(
        core_wb,
        count=1000,
        target_labels=max(0, reference_labels - ld1000_labels),
        target_total=max(0, reference_total - ld1000_total),
        seed=SEED + 1000,
    )
    wb1000_ids = {record_id(row) for row in wb1000}
    remaining_wb = [row for row in core_wb if record_id(row) not in wb1000_ids]
    ld500_labels = sum(int(row["label_tokens"]) for row in core_ld500)
    ld500_total = sum(int(row["total_tokens"]) for row in core_ld500)
    add500 = local_search_subset(
        remaining_wb,
        count=500,
        target_labels=max(
            0,
            reference_labels
            - ld500_labels
            - sum(int(row["label_tokens"]) for row in wb1000),
        ),
        target_total=max(
            0,
            reference_total
            - ld500_total
            - sum(int(row["total_tokens"]) for row in wb1000),
        ),
        seed=SEED + 1500,
    )
    wb1500_ids = wb1000_ids | {record_id(row) for row in add500}
    wb1500 = [row for row in core_wb if record_id(row) in wb1500_ids]
    if not wb1000_ids < wb1500_ids < core_wb_ids:
        raise RuntimeError("WB nested subset contract failed")

    arms = {
        "BUDGET-A-WB2000": list(core_wb),
        "BUDGET-A-WB1500-LD500": wb1500 + list(core_ld500),
        "BUDGET-A-WB1000-LD1000": wb1000 + list(core_ld1000),
        # Preserve the exact prior A20 row order for a byte/audit-comparable
        # add-on arm while still training an independent adapter.
        "ADDON-B-WB2000-LD1000": list(sources["A20"]),
        "SUPPORT-C-LD1000": list(core_ld1000),
    }
    for offset, (name, rows) in enumerate(arms.items()):
        if name != "ADDON-B-WB2000-LD1000":
            random.Random(SEED + 2000 + offset).shuffle(rows)

    contract_source = (
        project / "outputs/anchor_ratio_eos_fixed/audit/frozen_training_contract.json"
    )
    contract_target = output / "audit/frozen_training_contract.json"
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    if contract_target.exists() and sha256(contract_target) != sha256(contract_source):
        raise RuntimeError("frozen training contract copy changed")
    if not contract_target.exists():
        shutil.copyfile(contract_source, contract_target)

    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=False
    )
    if (
        tokenizer.eos_token_id != 151645
        or tokenizer.pad_token_id != 151643
        or tokenizer.eos_token_id == tokenizer.pad_token_id
    ):
        raise RuntimeError("tokenizer EOS/PAD identity changed")

    core_identity: dict[str, Any] = {}
    for name, rows in {
        "Core-WB2000": core_wb,
        "Extra-WB1000": extra_wb,
        "Core-LD250": core_ld250,
        "Core-LD500": core_ld500,
        "Core-LD1000": core_ld1000,
        "Frozen-WB1000": wb1000,
        "Frozen-WB1500": wb1500,
    }.items():
        path = output / "manifests/core" / f"{name}.jsonl"
        frozen_write_jsonl(path, rows)
        core_identity[name] = {
            "rows": len(rows),
            "path": str(path),
            "sha256": sha256(path),
            "ordered_record_ids_sha256": hash_values(
                [record_id(row) for row in rows]
            ),
            "theorem_group_ids_sha256": hash_values(
                [theorem_group(row) for row in rows]
            ),
            "statement_hashes_sha256": hash_values(
                [
                    hashlib.sha256(normalized_statement(row).encode()).hexdigest()
                    for row in rows
                ]
            ),
            "proof_hashes_sha256": hash_values(
                [
                    hashlib.sha256(
                        str(row.get("proof") or row.get("completion") or "").encode()
                    ).hexdigest()
                    for row in rows
                ]
            ),
        }

    gates: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    true_holdout_rows: list[dict[str, Any]] = []
    for identity in preflight["protected_identity"].values():
        true_holdout_rows.extend(read_jsonl(Path(identity["path"])))
    for name, rows in arms.items():
        source_path = output / ARMS[name]
        frozen_write_jsonl(source_path, rows)
        arm_stats = summarize(
            rows,
            reference_labels=reference_labels,
            reference_total=reference_total,
        )
        leak = overlap(rows, true_holdout_rows)
        if (
            arm_stats["duplicate_rows"]
            or arm_stats["theorem_group_duplicates"]
            or arm_stats["max_repeat"] != 1
            or arm_stats["zero_label_rows"]
            or arm_stats["semantic_truncation_rows"]
            or not arm_stats["all_pantograph_verified"]
            or leak["theorem_group_overlap"]
            or leak["normalized_statement_overlap"]
        ):
            raise RuntimeError(f"{name} failed the data gate: {arm_stats}; {leak}")

        effective_rows: list[dict[str, Any]] = []
        for row in rows:
            completion = str(row.get("completion") or "")
            if not completion.strip() or tokenizer.eos_token in completion:
                raise RuntimeError(f"{name} has an invalid source completion")
            effective = dict(row)
            effective["completion"] = completion + tokenizer.eos_token
            effective_rows.append(effective)
        effective_path = output / "training" / name / "input/effective_train.jsonl"
        frozen_write_jsonl(effective_path, effective_rows)
        gate = assert_supervised_eos_contract(
            Dataset.from_list(effective_rows),
            tokenizer,
            max_seq_length=1024,
        )
        gate.update(
            {
                "status": "EOS_GATE_PASSED",
                "arm": name,
                "source_manifest": str(source_path),
                "source_manifest_sha256": sha256(source_path),
                "effective_manifest": str(effective_path),
                "effective_manifest_sha256": sha256(effective_path),
                "hard_holdout_overlap": leak,
            }
        )
        write_json(output / "audit" / f"eos_gate_{name}.json", gate)
        gates[name] = gate
        stats[name] = {
            **arm_stats,
            "source_manifest_sha256": sha256(source_path),
            "effective_manifest_sha256": sha256(effective_path),
            "label_budget_within_5_percent": (
                arm_stats["label_token_error_vs_A_WB2000"] <= 0.05
            ),
            "total_budget_within_10_percent": (
                arm_stats["total_token_error_vs_A_WB2000"] <= 0.10
            ),
        }

    easy_pool = (
        project
        / "outputs/initial_anchor_ratio_ablation/audit/"
        "ld_easy_expansion/trainable_easy_pool.jsonl"
    )
    conditional_ld2000 = {
        "status": "blocked_insufficient_data",
        "eligible_rows": sum(1 for line in easy_pool.open(encoding="utf-8") if line.strip()),
        "required_rows": 2000,
        "repetition_forbidden": True,
    }
    if conditional_ld2000["eligible_rows"] >= 2000:
        conditional_ld2000["status"] = "eligible_not_materialized"

    environment = {
        "base_model": str(base),
        "base_model_sha256": EXPECTED_BASE_HASH,
        "environment_hash": EXPECTED_ENVIRONMENT_HASH,
        "lean_commit": "f72c35b3f637c8c6571d353742168ab66cc22c00",
        "mathlib_commit": "5e932f97dd25535344f80f9dd8da3aab83df0fe6",
        "pantograph": "0.3.15",
        "seed": SEED,
        "data_seed": SEED,
    }
    write_json(output / "audit/environment.json", environment)
    write_json(output / "audit/core_dataset_identity.json", core_identity)
    write_json(output / "audit/training_manifest_audit.json", stats)
    write_json(output / "audit/conditional_ld2000.json", conditional_ld2000)
    write_json(
        output / "audit/experiment_contract.json",
        {
            "adaptation": {
                "WB-Train-Retention150": {
                    "role": "in_distribution_retention_replay",
                    "training_overlap_allowed": True,
                    "protected_hard_gate": False,
                },
                "WB-Unseen-Holdout150": {
                    "role": "unseen_wb_generalization",
                    "training_overlap_allowed": False,
                    "protected_hard_gate": True,
                },
            },
            "training_contract_sha256": sha256(contract_target),
            "arms": ARMS,
            "generation_contract": {
                "max_new_tokens": 256,
                "num_candidates": 4,
                "paired_candidate_seeds": True,
                "retention_seed": 20261001,
                "ld_easy_seed": 20261002,
                "monitor_seed": 20261003,
                "hard_ld_seed": 20261004,
                "wb_unseen_seed": 20261007,
            },
            "evaluation": {
                "wb_train_retention150": (
                    "outputs/expert_sft_anchor_ablation/gates/"
                    "anchor_gate_150.jsonl"
                ),
                "wb_unseen_holdout150": (
                    "outputs/wb_ld_budget_support_replay_ablation/datasets/"
                    "wb_unseen_holdout_150.jsonl"
                ),
                "ld_easy64": (
                    "outputs/ld_length_difficulty_pipeline/pilot_sft/"
                    "evaluation/ld_easy_holdout_64.jsonl"
                ),
                "hard_ld128": (
                    "outputs/wb_ld_small_sft_ablation/evaluation/"
                    "ld_holdout_manifest.jsonl"
                ),
                "monitor64": (
                    "outputs/b2_expanded_validation/datasets/"
                    "monitor_minif2f_valid_64.jsonl"
                ),
                "strict_unseen200": (
                    "outputs/b2_expanded_validation/datasets/"
                    "strict_unseen_discovery.jsonl"
                ),
            },
        },
    )

    gate_lines = ["# EOS and data launch gates", ""]
    for name, gate in gates.items():
        gate_lines.append(
            f"- {name}: {gate['status']}; rows={gate['total_records']}; "
            f"supervised EOS={gate['records_with_supervised_eos']}; "
            "hard holdout leaks=0"
        )
    (output / "audit/eos_gate_summary.md").write_text(
        "\n".join(gate_lines) + "\n", encoding="utf-8"
    )
    token_lines = [
        "# Fixed-budget token audit",
        "",
        "| Arm | Rows | WB | LD | Label tokens | Label error | Total tokens | Total error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in stats.items():
        token_lines.append(
            f"| {name} | {row['rows']} | {row['wb_rows']} | {row['ld_rows']} | "
            f"{row['label_tokens']} | {row['label_token_error_vs_A_WB2000']:.2%} | "
            f"{row['total_tokens']} | {row['total_token_error_vs_A_WB2000']:.2%} |"
        )
    token_lines += [
        "",
        "Where the requested 5%/10% token tolerance is infeasible, row count, "
        "one-epoch optimizer steps, uniqueness, and intact proofs take priority. "
        "No proof was truncated, altered, or repeated to force token matching.",
        "",
    ]
    (output / "audit/token_budget_audit.md").write_text(
        "\n".join(token_lines), encoding="utf-8"
    )
    print(json.dumps(
        {
            "status": "TRAINING_MANIFEST_GATES_PASSED",
            "arms": stats,
            "conditional_ld2000": conditional_ld2000,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
