#!/usr/bin/env python3
"""Freeze identities and build the pre-registered S2/S3 random replay manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from lean_prover.lean_training.sft_pipeline.trainer import (
    assert_supervised_eos_contract,
)
from scripts.audit_wb_ld_budget_support_preflight import (
    TRUE_HOLDOUTS,
    normalized_statement,
    proof_hash,
    read_jsonl,
    record_id,
    sha256,
    theorem_group,
    write_json,
    write_jsonl,
)
from scripts.prepare_wb_ld_budget_support_ablation import (
    EXPECTED_BASE_HASH,
    EXPECTED_ENVIRONMENT_HASH,
    frozen_write_jsonl,
    hash_values,
    source_rows,
)
from scripts.prepare_wb_ld_replay_ablation import (
    domain_family,
    length_bin,
    tactic_family,
)


OUTPUT = Path("outputs/r_random_replication_strict_eval")
PRIOR_ROOT = Path("outputs/wb_ld_budget_support_replay_ablation")
BASE = Path("models/Qwen2.5-1.5B-Instruct")
S1_NAME = "REPLAY-R-RANDOM-WB2000-LD1000"
ADDON_NAME = "ADDON-B-WB2000-LD1000"
NEW_MODELS = {
    "R-RANDOM-S2-SEED-20261502": (
        20261502,
        "manifests/r_random_s2_seed_20261502.jsonl",
    ),
    "R-RANDOM-S3-SEED-20261503": (
        20261503,
        "manifests/r_random_s3_seed_20261503.jsonl",
    ),
}
S1_SELECTION_SEED = 20261501
# This is S1's original final manifest shuffle seed. It stays fixed so that
# changing the WB subset seed remains the sole experimental variable.
TRAINING_ORDER_SEED = 20261601
EXPECTED_A0_HASH = (
    "9c4d2f5aee256c70937353a5f14e9d275d5d869d4feb2fb11c03771357f11686"
)
EXPECTED_S1_HASH = (
    "ae249938e4fa58165b8a27eeec45ee78b72aad75061b2d5ce71c093d40bb3959"
)
LEAN_COMMIT = "f72c35b3f637c8c6571d353742168ab66cc22c00"
MATHLIB_COMMIT = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def exact_statement(row: dict[str, Any]) -> str:
    return str(
        row.get("lean_statement")
        or row.get("statement")
        or row.get("training_statement")
        or ""
    ).strip()


def qualified_name(row: dict[str, Any]) -> str:
    annotation = row.get("difficulty_annotation") or {}
    value = (
        row.get("source_declaration")
        or annotation.get("qualified_name")
        or row.get("qualified_name")
    )
    if value:
        return str(value)
    match = re.search(
        r"\b(?:theorem|lemma|example|def)\s+([^\s(:{]+)",
        exact_statement(row),
    )
    return match.group(1) if match else ""


def normalized_proof(row: dict[str, Any]) -> str:
    value = str(
        row.get("proof")
        or row.get("completion")
        or row.get("reference_proof")
        or row.get("training_proof")
        or ""
    )
    return re.sub(r"\s+", " ", value).strip()


def detailed_overlap(
    training: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
) -> dict[str, Any]:
    def values(rows: list[dict[str, Any]], function: Any) -> set[str]:
        return {value for row in rows if (value := function(row))}

    dimensions = {
        "record_id": (record_id, record_id),
        "theorem_group": (theorem_group, theorem_group),
        "qualified_theorem": (qualified_name, qualified_name),
        "exact_statement": (exact_statement, exact_statement),
        "normalized_statement": (normalized_statement, normalized_statement),
    }
    result: dict[str, Any] = {}
    shared_statement_identity: set[str] = set()
    for name, (left_function, right_function) in dimensions.items():
        shared = sorted(
            values(training, left_function) & values(evaluation, right_function)
        )
        result[f"{name}_overlap"] = len(shared)
        result[f"{name}_samples"] = shared[:10]
        if name in {"theorem_group", "qualified_theorem", "normalized_statement"}:
            shared_statement_identity.update(shared)

    training_variants = {
        (theorem_group(row), normalized_proof(row))
        for row in training
        if theorem_group(row) and normalized_proof(row)
    }
    evaluation_variants = {
        (theorem_group(row), normalized_proof(row))
        for row in evaluation
        if theorem_group(row) and normalized_proof(row)
    }
    shared_variants = sorted(training_variants & evaluation_variants)
    result["proof_variant_overlap"] = len(shared_variants)
    result["proof_variant_samples"] = [
        {"theorem_group": group, "proof_sha256": text_hash(proof)}
        for group, proof in shared_variants[:10]
    ]
    result["semantic_identity_overlap"] = bool(shared_statement_identity)
    return result


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[round((len(ordered) - 1) * fraction)]


def identity_payload(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    include_source_distribution: bool = False,
) -> dict[str, Any]:
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    statements = [text_hash(normalized_statement(row)) for row in rows]
    proofs = [proof_hash(row) for row in rows]
    payload: dict[str, Any] = {
        "manifest_path": str(path),
        "manifest_sha256": sha256(path),
        "total_rows": len(rows),
        "unique_record_count": len(set(ids)),
        "unique_theorem_group_count": len(set(groups)),
        "duplicate_rows": len(ids) - len(set(ids)),
        "theorem_group_duplicates": len(groups) - len(set(groups)),
        "max_repeat": max(Counter(ids).values(), default=0),
        "record_ids": ids,
        "theorem_group_ids": groups,
        "normalized_statement_hashes": statements,
        "proof_hashes": proofs,
        "ordered_record_ids_sha256": hash_values(ids),
        "ordered_theorem_group_ids_sha256": hash_values(groups),
        "ordered_statement_hashes_sha256": hash_values(statements),
        "ordered_proof_hashes_sha256": hash_values(proofs),
        "sample_ids": ids[:20],
        "all_verified": all(
            row.get("statement_verified") is True
            and row.get("proof_verified") is True
            and row.get("pantograph_verified") is True
            for row in rows
        ),
        "source_data_versions": {
            "sources": sorted({str(row.get("source") or "") for row in rows}),
            "lean_commits": sorted(
                {str(row.get("lean_commit") or "") for row in rows}
            ),
            "mathlib_commits": sorted(
                {str(row.get("mathlib_commit") or "") for row in rows}
            ),
            "environment_hashes": sorted(
                {str(row.get("environment_hash") or "") for row in rows}
            ),
        },
    }
    if include_source_distribution:
        proof_lengths = [int(row["label_tokens"]) for row in rows]
        payload["proof_token_distribution"] = {
            "mean": sum(proof_lengths) / max(1, len(proof_lengths)),
            "p50": percentile(proof_lengths, 0.50),
            "p90": percentile(proof_lengths, 0.90),
            "p95": percentile(proof_lengths, 0.95),
            "max": max(proof_lengths, default=0),
        }
        payload["source_file_distribution"] = dict(
            sorted(Counter(str(row.get("source_file") or "") for row in rows).items())
        )
    return payload


def write_identity_markdown(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Manifest: `{payload['manifest_path']}`",
        f"- SHA-256: `{payload['manifest_sha256']}`",
        f"- Rows: {payload['total_rows']}",
        f"- Unique records: {payload['unique_record_count']}",
        f"- Unique theorem groups: {payload['unique_theorem_group_count']}",
        f"- Duplicate records: {payload['duplicate_rows']}",
        f"- Theorem-group duplicates: {payload['theorem_group_duplicates']}",
        f"- Maximum repeat: {payload['max_repeat']}",
        f"- All statement/proof/Pantograph attestations valid: {payload['all_verified']}",
        f"- Ordered record identity: `{payload['ordered_record_ids_sha256']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    result: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        result["rows"] = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    return result


def evaluation_artifacts(root: Path, model: str, *, strict: bool) -> dict[str, Any]:
    directories = {
        "eval160": root / "evaluation/eval160" / model,
        "wb_retention150": root / "evaluation/core" / model / "wb_train_retention150",
        "wb_unseen150": root / "evaluation/core" / model / "wb_unseen_holdout150",
        "ld_easy64": root / "evaluation/core" / model / "ld_easy64",
        "monitor64": root / "evaluation/core" / model / "monitor64",
        "hard_ld128": root / "evaluation/core" / model / "hard_ld128",
    }
    if strict:
        directories.update(
            {
                "full500": root / "evaluation/promoted" / model / "full500",
                "strict_unseen200": (
                    root / "evaluation/promoted" / model / "strict_unseen200"
                ),
            }
        )
    return {
        name: {
            filename: artifact_identity(directory / filename)
            for filename in (
                "benchmark_summary.json",
                "eval_metrics.json",
                "generations.jsonl",
                "verifications.jsonl",
                "attempts.jsonl",
            )
            if (directory / filename).is_file()
        }
        for name, directory in directories.items()
    }


def freeze_prior_model(
    project: Path,
    prior: Path,
    model: str,
    *,
    strict: bool,
) -> dict[str, Any]:
    checkpoint = prior / "checkpoints" / model / "best"
    training = prior / "training" / model
    source_manifest = (
        prior / "manifests/replay/REPLAY-R-RANDOM-WB2000-LD1000.jsonl"
        if model == S1_NAME
        else prior / "manifests/fixed_wb/ADDON-B-WB2000-LD1000.jsonl"
    )
    required = [
        checkpoint / "adapter_model.safetensors",
        checkpoint / "adapter_config.json",
        training / "training_identity.json",
        training / "training_summary.json",
        training / "sampling_trace.jsonl",
        source_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen {model} artifacts: {missing}")
    summary = json.loads(
        (training / "training_summary.json").read_text(encoding="utf-8")
    )
    return {
        "model": model,
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": tree_hash(checkpoint),
        "adapter_model_sha256": sha256(checkpoint / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(checkpoint / "adapter_config.json"),
        "training_manifest_path": str(source_manifest),
        "training_manifest_sha256": sha256(source_manifest),
        "effective_manifest_sha256": summary["effective_manifest_sha256"],
        "training_contract_sha256": summary["training_contract_sha256"],
        "resolved_training_config": summary["resolved_config"],
        "training_summary": artifact_identity(training / "training_summary.json"),
        "training_identity": artifact_identity(training / "training_identity.json"),
        "sampling_trace": artifact_identity(training / "sampling_trace.jsonl"),
        "training_log": artifact_identity(
            prior / "runtime/logs" / f"train_{model}.log"
        ),
        "evaluation_artifacts": evaluation_artifacts(
            prior, model, strict=strict
        ),
    }


def manifest_stats(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    wb_rows = [row for row in rows if row.get("sampling_source") == "WB"]
    statement_tokens = [int(row["input_tokens"]) for row in rows]
    proof_tokens = [int(row["label_tokens"]) for row in rows]
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    return {
        "wb_rows": len(wb_rows),
        "ld_rows": len(rows) - len(wb_rows),
        "total_rows": len(rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "label_tokens_excluding_added_eos": sum(proof_tokens),
        "actual_supervised_label_tokens_including_eos": gate[
            "valid_label_tokens_total"
        ],
        "total_tokens_excluding_added_eos": sum(
            int(row["total_tokens"]) for row in rows
        ),
        "statement_tokens": {
            "mean": sum(statement_tokens) / max(1, len(statement_tokens)),
            "p50": percentile(statement_tokens, 0.50),
            "p95": percentile(statement_tokens, 0.95),
        },
        "proof_tokens": {
            "mean": sum(proof_tokens) / max(1, len(proof_tokens)),
            "p50": percentile(proof_tokens, 0.50),
            "p95": percentile(proof_tokens, 0.95),
        },
        "wb_length_bins": dict(
            sorted(Counter(length_bin(row) for row in wb_rows).items())
        ),
        "primary_tactic_distribution": dict(
            sorted(Counter(tactic_family(row) for row in rows).items())
        ),
        "domain_distribution": dict(
            sorted(Counter(domain_family(row) for row in rows).items())
        ),
        "source_file_count": len(
            {str(row.get("source_file") or "") for row in rows}
        ),
        "theorem_group_count": len(set(groups)),
        "duplicate_rows": len(ids) - len(set(ids)),
        "theorem_duplicates": len(groups) - len(set(groups)),
        "max_repeat": max(Counter(ids).values(), default=0),
    }


def package_inventory() -> str:
    rows = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return "\n".join(rows) + "\n"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    project = Path(".").resolve()
    output = project / OUTPUT
    prior = project / PRIOR_ROOT
    base = project / BASE
    if sha256(base / "model.safetensors") != EXPECTED_BASE_HASH:
        raise RuntimeError("base model identity changed")

    sources = source_rows(project)
    a0_wb = [
        row for row in sources["A0"] if row.get("sampling_source") == "WB"
    ]
    if len(a0_wb) != 3000:
        raise RuntimeError(f"A0 WB pool changed: {len(a0_wb)} != 3000")
    a0_path = project / (
        "outputs/anchor_ratio_eos_fixed/manifests/A0_source_manifest.jsonl"
    )
    if sha256(a0_path) != EXPECTED_A0_HASH:
        raise RuntimeError("A0 source manifest hash changed")

    ld_path = prior / "manifests/core/Core-LD1000.jsonl"
    ld_rows = read_jsonl(ld_path)
    a20_ld = [
        row for row in sources["A20"] if row.get("sampling_source") == "LD"
    ]
    if [record_id(row) for row in ld_rows] != [
        record_id(row) for row in a20_ld
    ]:
        raise RuntimeError("Core-LD1000 no longer matches frozen A20 identity/order")

    s1_source = prior / "manifests/replay" / f"{S1_NAME}.jsonl"
    if sha256(s1_source) != EXPECTED_S1_HASH:
        raise RuntimeError("S1 manifest identity changed")
    s1_rows = read_jsonl(s1_source)
    reconstructed_wb = list(a0_wb)
    random.Random(S1_SELECTION_SEED).shuffle(reconstructed_wb)
    reconstructed = reconstructed_wb[:2000] + list(ld_rows)
    random.Random(TRAINING_ORDER_SEED).shuffle(reconstructed)
    if [record_id(row) for row in reconstructed] != [
        record_id(row) for row in s1_rows
    ]:
        raise RuntimeError("S1 could not be reconstructed from frozen inputs")
    s1_ld_ids = [
        record_id(row)
        for row in reconstructed
        if row.get("sampling_source") == "LD"
    ]

    output.mkdir(parents=True, exist_ok=True)
    frozen_write_jsonl(output / "manifests/r_random_s1.jsonl", s1_rows)
    contract_source = prior / "audit/frozen_training_contract.json"
    contract_target = output / "audit/frozen_training_contract.json"
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    if contract_target.exists() and sha256(contract_target) != sha256(contract_source):
        raise RuntimeError("frozen training contract target changed")
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
        raise RuntimeError("tokenizer identity changed")

    wb_identity = identity_payload(a0_path, a0_wb)
    ld_identity = identity_payload(
        ld_path, ld_rows, include_source_distribution=True
    )
    if (
        wb_identity["duplicate_rows"]
        or wb_identity["theorem_group_duplicates"]
        or wb_identity["max_repeat"] != 1
        or not wb_identity["all_verified"]
        or ld_identity["total_rows"] != 1000
        or ld_identity["duplicate_rows"]
        or ld_identity["theorem_group_duplicates"]
        or not ld_identity["all_verified"]
    ):
        raise RuntimeError("frozen source identity gate failed")
    write_json(output / "audit/frozen_wb3000_identity.json", wb_identity)
    write_identity_markdown(
        output / "audit/frozen_wb3000_identity.md",
        "Frozen legal A0 WB3000 identity",
        wb_identity,
    )
    write_json(output / "audit/frozen_ld1000_identity.json", ld_identity)
    write_identity_markdown(
        output / "audit/frozen_ld1000_identity.md",
        "Frozen Core-LD1000 identity",
        ld_identity,
    )

    write_json(
        output / "audit/r_random_s1_identity.json",
        {
            **freeze_prior_model(project, prior, S1_NAME, strict=False),
            "selection_seed": S1_SELECTION_SEED,
            "training_order_seed": TRAINING_ORDER_SEED,
            "reconstructed_exactly": True,
        },
    )
    write_json(
        output / "audit/addon_b_identity.json",
        freeze_prior_model(project, prior, ADDON_NAME, strict=True),
    )

    protected_paths = {
        "wb_unseen150": project / TRUE_HOLDOUTS["wb_unseen_holdout150"],
        "ld_easy64": project / TRUE_HOLDOUTS["ld_easy_holdout64"],
        "monitor64": project / TRUE_HOLDOUTS["monitor64"],
        "hard_ld128": project / TRUE_HOLDOUTS["hard_ld128"],
        "full500": project / TRUE_HOLDOUTS["full500"],
        "strict_unseen200": project / TRUE_HOLDOUTS["strict_unseen200"],
    }
    retention_path = project / (
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl"
    )
    protected = {
        name: read_jsonl(path) for name, path in protected_paths.items()
    }
    retention = read_jsonl(retention_path)
    ld_overlap = {
        name: detailed_overlap(ld_rows, rows)
        for name, rows in protected.items()
    }
    if any(row["semantic_identity_overlap"] for row in ld_overlap.values()):
        raise RuntimeError("Core-LD1000 overlaps a protected evaluation manifest")
    ld_identity["protected_overlap"] = ld_overlap
    ld_identity["matches_s1_ld_record_set"] = (
        set(s1_ld_ids) == {record_id(row) for row in ld_rows}
    )
    ld_identity["s1_filtered_ld_order_sha256"] = hash_values(s1_ld_ids)
    write_json(output / "audit/frozen_ld1000_identity.json", ld_identity)

    all_manifests: dict[str, list[dict[str, Any]]] = {"S1": s1_rows}
    gates: dict[str, Any] = {}
    leakage: dict[str, Any] = {}
    audit_stats: dict[str, Any] = {}
    build_provenance: dict[str, Any] = {}
    for model, (selection_seed, relative) in NEW_MODELS.items():
        selected_wb = list(a0_wb)
        random.Random(selection_seed).shuffle(selected_wb)
        selected_wb = selected_wb[:2000]
        rows = selected_wb + list(ld_rows)
        random.Random(TRAINING_ORDER_SEED).shuffle(rows)
        filtered_ld_ids = [
            record_id(row) for row in rows if row.get("sampling_source") == "LD"
        ]
        if filtered_ld_ids != s1_ld_ids:
            raise RuntimeError(f"{model} changed Core-LD1000 relative order")
        source_path = output / relative
        frozen_write_jsonl(source_path, rows)
        all_manifests[model] = rows

        per_dataset = {
            name: detailed_overlap(rows, eval_rows)
            for name, eval_rows in protected.items()
        }
        per_dataset["wb_train_retention150"] = detailed_overlap(rows, retention)
        blockers = [
            name
            for name in protected
            if per_dataset[name]["semantic_identity_overlap"]
            or per_dataset[name]["proof_variant_overlap"]
        ]
        leakage[model] = {
            "status": "LEAKAGE_GATE_FAILED" if blockers else "LEAKAGE_GATE_PASSED",
            "blockers": blockers,
            "datasets": per_dataset,
            "retention_overlap_allowed": True,
        }
        if blockers:
            write_json(output / "audit/leakage_audit.json", leakage)
            raise RuntimeError(f"{model} protected leakage: {blockers}")

        effective_rows: list[dict[str, Any]] = []
        for row in rows:
            completion = str(row.get("completion") or "")
            if not completion.strip() or tokenizer.eos_token in completion:
                raise RuntimeError(f"{model} contains invalid completion")
            effective_rows.append(
                {**row, "completion": completion + tokenizer.eos_token}
            )
        effective_path = output / "training" / model / "input/effective_train.jsonl"
        frozen_write_jsonl(effective_path, effective_rows)
        gate = assert_supervised_eos_contract(
            Dataset.from_list(effective_rows),
            tokenizer,
            max_seq_length=1024,
        )
        gate.update(
            {
                "status": "EOS_GATE_PASSED",
                "arm": model,
                "source_manifest": str(source_path),
                "source_manifest_sha256": sha256(source_path),
                "effective_manifest": str(effective_path),
                "effective_manifest_sha256": sha256(effective_path),
                "leakage_gate_status": leakage[model]["status"],
            }
        )
        required_gate = (
            gate.get("total_records") == 3000
            and gate.get("records_with_supervised_eos") == 3000
            and gate.get("records_whose_last_valid_label_is_eos") == 3000
            and gate.get("records_without_supervised_eos") == 0
            and gate.get("zero_label_records") == 0
            and gate.get("records_with_semantic_truncation") == 0
        )
        if not required_gate:
            gate["status"] = "EOS_GATE_FAILED"
            write_json(
                output / "audit" / f"eos_gate_{'s2' if 'S2' in model else 's3'}.json",
                gate,
            )
            raise RuntimeError(f"{model} failed EOS gate")
        gate_name = "s2" if "S2" in model else "s3"
        write_json(output / "audit" / f"eos_gate_{gate_name}.json", gate)
        gates[model] = gate
        audit_stats[model] = manifest_stats(rows, gate)
        build_provenance[model] = {
            "python_version": platform.python_version(),
            "random_library": "Python stdlib random",
            "random_generator": "random.Random / Mersenne Twister",
            "selection_seed": selection_seed,
            "seed_application_point": (
                "shuffle an exact in-memory copy of frozen A0 WB3000 once, "
                "then take the first 2000 records"
            ),
            "input_order_rule": "exact frozen A0 manifest order",
            "sampling_function": "random.Random(seed).shuffle; slice [:2000]",
            "replacement": False,
            "training_order_seed": TRAINING_ORDER_SEED,
            "training_order_rule": (
                "selected WB2000 + frozen Core-LD1000, followed by the exact "
                "fixed S1 final-shuffle seed"
            ),
            "core_ld_filtered_order_sha256": hash_values(filtered_ld_ids),
        }

    s1_gate = json.loads(
        (prior / f"audit/eos_gate_{S1_NAME}.json").read_text(encoding="utf-8")
    )
    audit_stats = {
        "S1": manifest_stats(s1_rows, s1_gate),
        **audit_stats,
    }
    wb_sets = {
        name: {
            record_id(row)
            for row in rows
            if row.get("sampling_source") == "WB"
        }
        for name, rows in all_manifests.items()
    }
    common = set.intersection(*wb_sets.values())
    overlap_audit = {
        "manifests": audit_stats,
        "pairwise_wb_overlap": {
            f"{left}_vs_{right}": len(wb_sets[left] & wb_sets[right])
            for index, left in enumerate(wb_sets)
            for right in list(wb_sets)[index + 1 :]
        },
        "three_seed_common_wb_records": len(common),
        "unique_wb_records_per_seed": {
            name: len(values - set.union(*(other for key, other in wb_sets.items() if key != name)))
            for name, values in wb_sets.items()
        },
        "token_exposure_confound": {
            "threshold": 0.10,
            "supervised_label_tokens": {
                name: row["actual_supervised_label_tokens_including_eos"]
                for name, row in audit_stats.items()
            },
        },
        "build_provenance": build_provenance,
    }
    token_values = list(
        overlap_audit["token_exposure_confound"]["supervised_label_tokens"].values()
    )
    token_spread = (max(token_values) - min(token_values)) / max(1, min(token_values))
    overlap_audit["token_exposure_confound"].update(
        {
            "max_relative_spread": token_spread,
            "flagged": token_spread > 0.10,
            "manifest_resampled": False,
        }
    )
    write_json(output / "audit/manifest_overlap_audit.json", overlap_audit)
    write_json(output / "audit/leakage_audit.json", leakage)
    leakage_lines = [
        "# Protected-split leakage audit",
        "",
        "- WB-Train-Retention150 is an in-distribution retention set; overlap is allowed.",
        "- Every other listed evaluation manifest is a protected holdout.",
        "",
    ]
    for model, row in leakage.items():
        leakage_lines.append(f"## {model}")
        leakage_lines.append("")
        leakage_lines.append(f"- Status: `{row['status']}`")
        for name, counts in row["datasets"].items():
            leakage_lines.append(
                f"- {name}: records={counts['record_id_overlap']}, "
                f"groups={counts['theorem_group_overlap']}, "
                f"qualified={counts['qualified_theorem_overlap']}, "
                f"exact={counts['exact_statement_overlap']}, "
                f"normalized={counts['normalized_statement_overlap']}, "
                f"proof variants={counts['proof_variant_overlap']}"
            )
        leakage_lines.append("")
    (output / "audit/leakage_audit.md").write_text(
        "\n".join(leakage_lines), encoding="utf-8"
    )
    eos_lines = ["# Supervised EOS launch gates", ""]
    for model, gate in gates.items():
        eos_lines.append(
            f"- {model}: `{gate['status']}`; total=3000; supervised EOS=3000; "
            "last label EOS=3000; zero-label=0; semantic truncation=0"
        )
    (output / "audit/eos_gate_summary.md").write_text(
        "\n".join(eos_lines) + "\n", encoding="utf-8"
    )

    environment = {
        "status": "FROZEN_ENVIRONMENT_VERIFIED",
        "lean_version": "4.29.1",
        "lean_commit": LEAN_COMMIT,
        "mathlib_commit": MATHLIB_COMMIT,
        "actual_mathlib_commit": git_head(project / "lean_project/.lake/packages/mathlib"),
        "pantograph_version": importlib.metadata.version("pantograph"),
        "environment_hash": EXPECTED_ENVIRONMENT_HASH,
        "base_model": str(base),
        "base_model_sha256": EXPECTED_BASE_HASH,
        "tokenizer_json_sha256": sha256(base / "tokenizer.json"),
        "tokenizer_config_sha256": sha256(base / "tokenizer_config.json"),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "python": sys.version,
        "platform": platform.platform(),
    }
    if (
        environment["actual_mathlib_commit"] != MATHLIB_COMMIT
        or environment["pantograph_version"] != "0.3.15"
    ):
        raise RuntimeError("Lean/mathlib/Pantograph environment identity changed")
    write_json(output / "audit/environment.json", environment)
    (output / "audit/packages.txt").write_text(
        package_inventory(), encoding="utf-8"
    )
    write_json(
        output / "audit/experiment_contract.json",
        {
            "status": "PRETRAINING_GATES_PASSED",
            "sole_variable": "WB2000 manifest sampling seed",
            "models": NEW_MODELS,
            "s1_selection_seed": S1_SELECTION_SEED,
            "fixed_training_order_seed": TRAINING_ORDER_SEED,
            "fixed_model_training_seed": 42,
            "fixed_data_loader_seed": 42,
            "base_model_sha256": EXPECTED_BASE_HASH,
            "environment_hash": EXPECTED_ENVIRONMENT_HASH,
            "training_contract_sha256": sha256(contract_target),
            "generation_contract": {
                "max_new_tokens": 256,
                "samples_per_statement": 4,
                "temperature": 0.8,
                "top_p": 0.95,
                "candidate_seeds": {
                    "wb_retention150": 20261001,
                    "wb_unseen150": 20261007,
                    "ld_easy64": 20261002,
                    "monitor64": 20261003,
                    "hard_ld128": 20261004,
                    "full500": 20261005,
                    "strict_unseen200": 20261006,
                },
            },
            "protected_manifests": {
                name: {
                    "path": str(path),
                    "sha256": sha256(path),
                    "rows": len(protected[name]),
                }
                for name, path in protected_paths.items()
            },
            "retention_manifest": {
                "path": str(retention_path),
                "sha256": sha256(retention_path),
                "rows": len(retention),
                "training_overlap_allowed": True,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "PRETRAINING_GATES_PASSED",
                "output": str(output),
                "manifest_overlap": overlap_audit["pairwise_wb_overlap"],
                "token_exposure_confound": overlap_audit[
                    "token_exposure_confound"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
