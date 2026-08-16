#!/usr/bin/env python3
"""Audit frozen WB/LD cores against the protected evaluation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


SOURCE_MANIFESTS = {
    "A0": (
        "outputs/anchor_ratio_eos_fixed/manifests/A0_source_manifest.jsonl",
        "9c4d2f5aee256c70937353a5f14e9d275d5d869d4feb2fb11c03771357f11686",
    ),
    "A5": (
        "outputs/anchor_ratio_eos_fixed/manifests/A5_source_manifest.jsonl",
        "640d344dd0ac0ac76524f880309bed13a1dd8c2d5c959b05790872e95e003ec2",
    ),
    "A10": (
        "outputs/anchor_ratio_eos_fixed/manifests/A10_source_manifest.jsonl",
        "8cca18628ff588e4a7fe8c47df807f942f16d4bbab4925e32b9132fbeedbb34d",
    ),
    "A20": (
        "outputs/anchor_ratio_eos_fixed/manifests/A20_source_manifest.jsonl",
        "41ef30bedff5224c8ec29a2e15f55a2c8bbde8334f06c4d096e4c8e82736f981",
    ),
}
RETENTION = {
    "wb_train_retention150": (
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl"
    ),
}
TRUE_HOLDOUTS = {
    "wb_unseen_holdout150": (
        "outputs/wb_ld_budget_support_replay_ablation/datasets/"
        "wb_unseen_holdout_150.jsonl"
    ),
    "ld_easy_holdout64": (
        "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/"
        "ld_easy_holdout_64.jsonl"
    ),
    "monitor64": (
        "outputs/b2_expanded_validation/datasets/"
        "monitor_minif2f_valid_64.jsonl"
    ),
    "hard_ld128": (
        "outputs/wb_ld_small_sft_ablation/evaluation/"
        "ld_holdout_manifest.jsonl"
    ),
    "full500": "outputs/b2_expanded_validation/datasets/full500.jsonl",
    "strict_unseen200": (
        "outputs/b2_expanded_validation/datasets/"
        "strict_unseen_discovery.jsonl"
    ),
    "wb_eval160": "data/processed/lean_workbook_verified_v2/eval.jsonl",
}
WB_UNSEEN_SOURCE = "data/processed/lean_workbook_verified_v2/eval.jsonl"
WB_UNSEEN_SEED = 20260729
WB_UNSEEN_SIZE = 150


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


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


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id") or "")


def theorem_group(row: dict[str, Any]) -> str:
    return str(
        row.get("theorem_group_id")
        or row.get("statement_id")
        or normalized_statement(row)
    )


def proof_hash(row: dict[str, Any]) -> str:
    proof = str(
        row.get("proof")
        or row.get("completion")
        or row.get("training_proof")
        or ""
    )
    return hashlib.sha256(re.sub(r"\s+", " ", proof).strip().encode()).hexdigest()


def identity(row: dict[str, Any]) -> dict[str, str]:
    return {
        "record_id": record_id(row),
        "theorem_group_id": theorem_group(row),
        "normalized_statement": normalized_statement(row),
        "proof_hash": proof_hash(row),
    }


def overlap(
    training: list[dict[str, Any]], protected: list[dict[str, Any]]
) -> dict[str, Any]:
    train_ids = {record_id(row) for row in training if record_id(row)}
    train_groups = {theorem_group(row) for row in training if theorem_group(row)}
    train_statements = {
        normalized_statement(row)
        for row in training
        if normalized_statement(row)
    }
    protected_ids = {record_id(row) for row in protected if record_id(row)}
    protected_groups = {
        theorem_group(row) for row in protected if theorem_group(row)
    }
    protected_statements = {
        normalized_statement(row)
        for row in protected
        if normalized_statement(row)
    }
    statement_overlap = sorted(train_statements & protected_statements)
    group_overlap = sorted(train_groups & protected_groups)
    id_overlap = sorted(train_ids & protected_ids)
    # A proof string such as ``by simp`` can legitimately occur for unrelated
    # theorems.  A cross-split proof variant requires a shared theorem identity,
    # not merely identical proof text.
    proof_variant_keys = set(statement_overlap) | set(group_overlap)
    return {
        "record_id_overlap": len(id_overlap),
        "theorem_group_overlap": len(group_overlap),
        "normalized_statement_overlap": len(statement_overlap),
        "proof_variant_overlap": len(proof_variant_keys),
        "sample_record_ids": id_overlap[:10],
        "sample_theorem_groups": group_overlap[:10],
        "sample_statement_sha256": [
            hashlib.sha256(value.encode()).hexdigest()
            for value in statement_overlap[:10]
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def proof_free_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a generation-only WB row without any reference-proof channel."""
    sanitized = dict(row)
    for key in (
        "proof",
        "completion",
        "reference_proof",
        "raw_source_context",
        "raw_declaration",
        "text",
        "metadata",
    ):
        sanitized.pop(key, None)
    sanitized["has_reference_proof"] = False
    # The runtime schema uses the generic benchmark role for protected
    # evaluation sets; the more specific semantic identity is retained in
    # evaluation_role and in the frozen manifest name.
    sanitized["data_role"] = "benchmark"
    sanitized["evaluation_role"] = "unseen_generalization"
    return sanitized


def freeze_wb_unseen_holdout(
    project: Path,
    destination: Path,
    forbidden_training_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source = project / WB_UNSEEN_SOURCE
    rows = read_jsonl(source)
    eligible = [
        row
        for row in rows
        if not (
            overlap([row], forbidden_training_rows)["theorem_group_overlap"]
            or overlap([row], forbidden_training_rows)[
                "normalized_statement_overlap"
            ]
        )
    ]
    if len(eligible) < WB_UNSEEN_SIZE:
        raise RuntimeError(
            "Not enough leak-free WB eval rows for the unseen holdout: "
            f"{len(eligible)} < {WB_UNSEEN_SIZE}"
        )
    ordered = sorted(eligible, key=lambda row: (record_id(row), theorem_group(row)))
    random.Random(WB_UNSEEN_SEED).shuffle(ordered)
    selected = ordered[:WB_UNSEEN_SIZE]
    proof_free = [proof_free_eval_row(row) for row in selected]
    for row in proof_free:
        prompt = str(row.get("prompt") or "")
        if str(row.get("proof") or row.get("completion") or ""):
            raise RuntimeError("WB unseen row retained a reference proof field")
        if prompt and not prompt.rstrip().endswith("### Lean proof"):
            raise RuntimeError("WB unseen prompt contract changed")
    if destination.exists():
        existing = read_jsonl(destination)
        if [
            record_id(row) for row in existing
        ] != [record_id(row) for row in proof_free]:
            raise RuntimeError(
                "Refusing to overwrite a different frozen WB unseen holdout"
            )
    write_jsonl(destination, proof_free)
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "selection_seed": WB_UNSEEN_SEED,
        "selection_rule": (
            "sort leak-free verified WB eval rows by record/theorem identity, "
            "shuffle once with the fixed seed, take the first 150"
        ),
        "source_rows": len(rows),
        "eligible_rows": len(eligible),
        "selected_rows": len(proof_free),
        "proof_free": True,
        "path": str(destination),
        "sha256": sha256(destination),
        "ordered_record_ids_sha256": hashlib.sha256(
            "\0".join(record_id(row) for row in proof_free).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/wb_ld_budget_support_replay_ablation"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    output = (project / args.output).resolve()

    arms: dict[str, list[dict[str, Any]]] = {}
    manifest_identity: dict[str, Any] = {}
    for arm, (relative, expected_hash) in SOURCE_MANIFESTS.items():
        path = project / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{arm} source manifest changed: {actual_hash} != {expected_hash}"
            )
        arms[arm] = read_jsonl(path)
        manifest_identity[arm] = {
            "path": str(path),
            "sha256": actual_hash,
            "rows": len(arms[arm]),
        }

    core_wb2000 = [
        row
        for row in arms["A20"]
        if str(row.get("sampling_source") or "").upper() == "WB"
    ]
    core_ld1000 = [
        row
        for row in arms["A20"]
        if str(row.get("sampling_source") or "").upper() == "LD"
    ]
    core_ld500 = [
        row
        for row in arms["A10"]
        if str(row.get("sampling_source") or "").upper() == "LD"
    ]
    core_ld250 = [
        row
        for row in arms["A5"]
        if str(row.get("sampling_source") or "").upper() == "LD"
    ]
    core_wb_ids = {record_id(row) for row in core_wb2000}
    a0_wb = [
        row
        for row in arms["A0"]
        if str(row.get("sampling_source") or "").upper() == "WB"
    ]
    extra_wb1000 = [row for row in a0_wb if record_id(row) not in core_wb_ids]
    core_sets = {
        "Core-WB2000": core_wb2000,
        "Extra-WB1000": extra_wb1000,
        "Core-LD250": core_ld250,
        "Core-LD500": core_ld500,
        "Core-LD1000": core_ld1000,
    }
    expected_counts = {
        "Core-WB2000": 2000,
        "Extra-WB1000": 1000,
        "Core-LD250": 250,
        "Core-LD500": 500,
        "Core-LD1000": 1000,
    }
    for name, rows in core_sets.items():
        if len(rows) != expected_counts[name]:
            raise RuntimeError(f"{name} count is {len(rows)}, expected {expected_counts[name]}")

    all_training_rows = a0_wb + core_ld1000
    unseen_identity = freeze_wb_unseen_holdout(
        project,
        project / TRUE_HOLDOUTS["wb_unseen_holdout150"],
        all_training_rows,
    )

    retention_identity: dict[str, Any] = {}
    retention_rows: dict[str, list[dict[str, Any]]] = {}
    for name, relative in RETENTION.items():
        path = project / relative
        retention_rows[name] = read_jsonl(path)
        retention_identity[name] = {
            "path": str(path),
            "rows": len(retention_rows[name]),
            "sha256": sha256(path),
            "role": "in_distribution_training_retention_replay",
            "training_overlap_allowed": True,
        }

    protected_identity: dict[str, Any] = {}
    protected_rows: dict[str, list[dict[str, Any]]] = {}
    for name, relative in TRUE_HOLDOUTS.items():
        path = project / relative
        protected_rows[name] = read_jsonl(path)
        protected_identity[name] = {
            "path": str(path),
            "rows": len(protected_rows[name]),
            "sha256": sha256(path),
        }

    overlaps = {
        core_name: {
            protected_name: overlap(rows, protected_rows[protected_name])
            for protected_name in TRUE_HOLDOUTS
        }
        for core_name, rows in core_sets.items()
    }
    blockers: list[dict[str, Any]] = []
    for core_name, datasets in overlaps.items():
        for protected_name, counts in datasets.items():
            if (
                counts["theorem_group_overlap"]
                or counts["normalized_statement_overlap"]
                or counts["proof_variant_overlap"]
            ):
                blockers.append(
                    {
                        "training_core": core_name,
                        "protected_dataset": protected_name,
                        **counts,
                    }
                )

    retention_training_overlaps = {
        core_name: {
            retention_name: overlap(rows, retention_rows[retention_name])
            for retention_name in RETENTION
        }
        for core_name, rows in core_sets.items()
    }
    retention_holdout_overlaps = {
        retention_name: {
            holdout_name: overlap(rows, protected_rows[holdout_name])
            for holdout_name in TRUE_HOLDOUTS
        }
        for retention_name, rows in retention_rows.items()
    }
    retention_holdout_blockers: list[dict[str, Any]] = []
    for retention_name, datasets in retention_holdout_overlaps.items():
        for holdout_name, counts in datasets.items():
            if (
                counts["theorem_group_overlap"]
                or counts["normalized_statement_overlap"]
            ):
                retention_holdout_blockers.append(
                    {
                        "retention_dataset": retention_name,
                        "true_holdout": holdout_name,
                        **counts,
                    }
                )
    blockers.extend(retention_holdout_blockers)

    ld250_ids = {record_id(row) for row in core_ld250}
    ld500_ids = {record_id(row) for row in core_ld500}
    ld1000_ids = {record_id(row) for row in core_ld1000}
    wb_a0_ids = {record_id(row) for row in a0_wb}
    core_identity = {
        name: {
            "rows": len(rows),
            "ordered_record_ids_sha256": hashlib.sha256(
                "\0".join(record_id(row) for row in rows).encode()
            ).hexdigest(),
            "record_ids": [record_id(row) for row in rows],
            "theorem_group_ids": [theorem_group(row) for row in rows],
            "statement_sha256": [
                hashlib.sha256(normalized_statement(row).encode()).hexdigest()
                for row in rows
            ],
            "proof_sha256": [proof_hash(row) for row in rows],
        }
        for name, rows in core_sets.items()
    }
    nesting = {
        "Core-WB2000_subset_A0-WB3000": core_wb_ids <= wb_a0_ids,
        "Core-WB2000_plus_Extra-WB1000_equals_A0-WB3000": (
            core_wb_ids | {record_id(row) for row in extra_wb1000}
        )
        == wb_a0_ids,
        "Core-LD250_subset_Core-LD500": ld250_ids <= ld500_ids,
        "Core-LD500_subset_Core-LD1000": ld500_ids <= ld1000_ids,
        "Core-LD250_Core-LD500_overlap": len(ld250_ids & ld500_ids),
        "Core-LD500_Core-LD1000_overlap": len(ld500_ids & ld1000_ids),
    }
    payload = {
        "status": "BLOCKED_TRUE_HOLDOUT_OVERLAP" if blockers else "PASSED",
        "manifest_identity": manifest_identity,
        "evaluation_contract": {
            "wb_train_retention150": {
                "role": "in_distribution_training_retention_replay",
                "training_overlap_allowed": True,
                "protected_split_hard_gate": False,
            },
            "wb_unseen_holdout150": {
                "role": "unseen_wb_generalization",
                "training_overlap_allowed": False,
                "protected_split_hard_gate": True,
                "proof_free_generation_manifest": True,
            },
            "ld_easy_holdout64": {
                "training_overlap_allowed": False,
                "protected_split_hard_gate": True,
            },
            "hard_ld128": {
                "training_overlap_allowed": False,
                "protected_split_hard_gate": True,
            },
            "monitor64": {
                "training_overlap_allowed": False,
                "protected_split_hard_gate": True,
            },
            "strict_unseen200": {
                "training_overlap_allowed": False,
                "protected_split_hard_gate": True,
            },
        },
        "wb_unseen_holdout_identity": unseen_identity,
        "retention_identity": retention_identity,
        "protected_identity": protected_identity,
        "core_identity": core_identity,
        "nesting": nesting,
        "overlaps": overlaps,
        "retention_training_overlaps_expected": retention_training_overlaps,
        "retention_true_holdout_overlaps": retention_holdout_overlaps,
        "blockers": blockers,
        "hard_gate": (
            "Training may start only when theorem-group and normalized-"
            "statement overlap with every true holdout are zero. "
            "WB-Train-Retention150 is explicitly exempt from this hard gate."
        ),
    }
    write_json(output / "audit/preflight_split_audit.json", payload)
    lines = [
        "# WB / LD budget-support adapted split preflight",
        "",
        f"- Status: `{payload['status']}`",
        f"- Blocking overlaps: {len(blockers)}",
        "- WB-Train-Retention150: in-distribution retention/replay; "
        "training overlap allowed; excluded from protected hard gate.",
        "- WB-Unseen-Holdout150: proof-free unseen generalization; "
        "training overlap forbidden.",
        "",
    ]
    for item in blockers:
        lines.append(
            f"- {item['training_core']} vs {item['protected_dataset']}: "
            f"groups={item['theorem_group_overlap']}, "
            f"statements={item['normalized_statement_overlap']}, "
            f"proof variants={item['proof_variant_overlap']}"
        )
    lines += [
        "",
        f"- WB unseen source rows: {unseen_identity['source_rows']}",
        f"- WB unseen selected rows: {unseen_identity['selected_rows']}",
        f"- WB unseen manifest SHA-256: `{unseen_identity['sha256']}`",
        "",
    ]
    (output / "audit/preflight_split_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if blockers:
        final_lines = [
            "# WB / LD budget-support-replay ablation: blocked preflight",
            "",
            "## Status",
            "",
            "`BLOCKED_PROTECTED_SPLIT_OVERLAP`",
            "",
            "No training manifest was created, no Trainer/DataLoader was "
            "started, and no checkpoint was written.",
            "",
            "## Blocking contradiction",
            "",
            "- Core-WB2000 must be the exact 2000 WB rows from A20.",
            "- Extra-WB1000 must be the exact remaining 1000 WB rows from A0.",
            "- WB Gate150 is declared protected and the task requires zero "
            "theorem/statement/proof-variant overlap with every protected set.",
            "- Core-WB2000 contains 68/150 WB Gate statements.",
            "- Extra-WB1000 contains the remaining 82/150 WB Gate statements.",
            "- Consequently, A0-WB3000 contains WB Gate150 in full (150/150).",
            "",
            "These requirements cannot be satisfied simultaneously without "
            "changing either the frozen training identity or the evaluation "
            "contract, both of which the task explicitly forbids.",
            "",
            "## Confirmed non-blocking identities",
            "",
            "- Core-WB2000 is a subset of A0-WB3000.",
            "- Core-WB2000 + Extra-WB1000 equals A0-WB3000 exactly.",
            "- Core-LD250 is a subset of Core-LD500.",
            "- Core-LD500 is a subset of Core-LD1000.",
            "- No Core-LD theorem/statement overlaps LD-easy64, Hard-LD128, "
            "Monitor64, Full500, Strict unseen200, or WB eval160.",
            "- Hard evaluation theorem/statement leakage is zero.",
            "",
            "## Decision required",
            "",
            "Recommended resolution: explicitly classify the existing WB "
            "Gate150 as an in-distribution retention/replay gate, exempt it "
            "from the protected-split hard gate, and add or designate a "
            "separate leak-free WB holdout for generalization. This preserves "
            "the exact A0/A20 training identities but changes the evaluation "
            "contract and therefore requires user approval.",
            "",
            "Alternative: keep WB Gate150 protected and construct new "
            "leak-free Core-WB2000/Extra-WB1000 sets. This changes the exact "
            "A20/A0 identities and breaks direct continuity with the prior "
            "experiment.",
            "",
            "The experiment must not continue until one of these contracts is "
            "explicitly revised.",
            "",
        ]
        (output / "final_report.md").write_text(
            "\n".join(final_lines), encoding="utf-8"
        )
    else:
        final_lines = [
            "# WB / LD budget-support-replay ablation: adapted split contract",
            "",
            "## Status",
            "",
            "`PREFLIGHT_PASSED`",
            "",
            "Core-WB2000, Extra-WB1000, and Core-LD1000 retain their exact "
            "frozen identities.",
            "",
            "WB-Train-Retention150 is an in-distribution retention/replay "
            "evaluation. Its training overlap is expected and is not a "
            "protected-split blocker.",
            "",
            "WB-Unseen-Holdout150 is a separately frozen, proof-free WB "
            "generalization set. It and every other true holdout have zero "
            "theorem/statement overlap with all frozen training cores.",
            "",
        ]
        (output / "final_report.md").write_text(
            "\n".join(final_lines), encoding="utf-8"
        )
    print(json.dumps(
        {
            "status": payload["status"],
            "nesting": nesting,
            "blockers": blockers,
        },
        ensure_ascii=False,
        indent=2,
    ))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
