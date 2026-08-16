#!/usr/bin/env python3
"""Materialize the triggered Random/Core/Curated WB replay experiment."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from lean_prover.lean_training.sft_pipeline.trainer import (
    assert_supervised_eos_contract,
)
from scripts.audit_wb_ld_budget_support_preflight import (
    overlap,
    read_jsonl,
    record_id,
    sha256,
    theorem_group,
    write_json,
)
from scripts.prepare_wb_ld_budget_support_ablation import (
    EXPECTED_BASE_HASH,
    OUTPUT,
    frozen_write_jsonl,
    hash_values,
    source_rows,
    summarize,
)


REPLAY_SEED = 20261501
REPLAY_ARMS = {
    "REPLAY-R-RANDOM-WB2000-LD1000": (
        "manifests/replay/REPLAY-R-RANDOM-WB2000-LD1000.jsonl"
    ),
    "REPLAY-R-CURATED-WB2000-LD1000": (
        "manifests/replay/REPLAY-R-CURATED-WB2000-LD1000.jsonl"
    ),
}


def tactic_family(row: dict[str, Any]) -> str:
    proof = str(row.get("proof") or row.get("completion") or "")
    tactics = (
        "simp",
        "norm_num",
        "linarith",
        "nlinarith",
        "aesop",
        "omega",
        "ring",
        "field_simp",
        "rw",
        "exact",
        "constructor",
        "induction",
    )
    for tactic in tactics:
        if re.search(rf"\b{re.escape(tactic)}\b", proof):
            return tactic
    return "other"


def domain_family(row: dict[str, Any]) -> str:
    statement = str(
        row.get("lean_statement") or row.get("statement") or ""
    ).lower()
    rules = (
        ("probability_measure", ("measure", "probability", "ae ", "ennreal")),
        ("topology_analysis", ("continuous", "tendsto", "filter", "topolog")),
        ("set_finset", ("finset", "set ", "set.", "subset")),
        ("sequence_container", ("list", "array", "vector", "multiset")),
        ("number_theory", ("prime", "nat.gcd", "divis", "mod ", "zmod")),
        ("algebra", ("matrix", "polynomial", "monoid", "group", "ringhom")),
        ("order_lattice", ("lattice", "sup ", "inf ", "monotone")),
        ("logic", ("↔", "→", "¬", "exists", "∀")),
        ("numeric", ("ℝ", "ℕ", "ℤ", "rat", "real", "nat ", "int ")),
    )
    for family, needles in rules:
        if any(needle.lower() in statement for needle in needles):
            return family
    return "other"


def length_bin(row: dict[str, Any]) -> str:
    value = int(row["label_tokens"])
    if value <= 16:
        return "short"
    if value <= 32:
        return "medium"
    return "long"


def curated_wb(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Round-robin across fixed metadata strata without using eval outcomes."""
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (domain_family(row), tactic_family(row), length_bin(row))
        strata[key].append(row)
    for offset, key in enumerate(sorted(strata)):
        random.Random(REPLAY_SEED + offset).shuffle(strata[key])
    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(strata)
    cursor = 0
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            if cursor < len(strata[key]):
                selected.append(strata[key][cursor])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("curated strata exhausted before requested count")
        cursor += 1
    return selected


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "domains": dict(sorted(Counter(domain_family(row) for row in rows).items())),
        "tactics": dict(sorted(Counter(tactic_family(row) for row in rows).items())),
        "length_bins": dict(sorted(Counter(length_bin(row) for row in rows).items())),
        "strata": len(
            {
                (domain_family(row), tactic_family(row), length_bin(row))
                for row in rows
            }
        ),
    }


def main() -> None:
    project = Path(".").resolve()
    root = project / OUTPUT
    trigger = json.loads(
        (root / "comparisons/replay_trigger.json").read_text(encoding="utf-8")
    )
    if trigger.get("status") != "TRIGGERED":
        raise RuntimeError("conditional replay trigger has not fired")
    base = project / "models/Qwen2.5-1.5B-Instruct"
    if sha256(base / "model.safetensors") != EXPECTED_BASE_HASH:
        raise RuntimeError("base model identity changed")

    sources = source_rows(project)
    a0_wb = [
        row for row in sources["A0"] if row.get("sampling_source") == "WB"
    ]
    core_wb = [
        row for row in sources["A20"] if row.get("sampling_source") == "WB"
    ]
    core_ld = [
        row for row in sources["A20"] if row.get("sampling_source") == "LD"
    ]
    if (len(a0_wb), len(core_wb), len(core_ld)) != (3000, 2000, 1000):
        raise RuntimeError("frozen A0/A20 core counts changed")

    random_wb = list(a0_wb)
    random.Random(REPLAY_SEED).shuffle(random_wb)
    random_wb = random_wb[:2000]
    curated = curated_wb(a0_wb, 2000)
    arms = {
        "REPLAY-R-RANDOM-WB2000-LD1000": random_wb + list(core_ld),
        "REPLAY-R-CURATED-WB2000-LD1000": curated + list(core_ld),
    }
    for offset, rows in enumerate(arms.values()):
        random.Random(REPLAY_SEED + 100 + offset).shuffle(rows)

    preflight = json.loads(
        (root / "audit/preflight_split_audit.json").read_text(encoding="utf-8")
    )
    true_holdouts: list[dict[str, Any]] = []
    for identity in preflight["protected_identity"].values():
        true_holdouts.extend(read_jsonl(Path(identity["path"])))
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError("tokenizer identity changed")

    reference_labels = sum(int(row["label_tokens"]) for row in core_wb)
    reference_total = sum(int(row["total_tokens"]) for row in core_wb)
    audit: dict[str, Any] = {
        "trigger_sha256": sha256(root / "comparisons/replay_trigger.json"),
        "selection_seed": REPLAY_SEED,
        "R-Core": {
            "reuse": "ADDON-B-WB2000-LD1000",
            "retrained": False,
        },
    }
    for name, rows in arms.items():
        ids = [record_id(row) for row in rows]
        groups = [theorem_group(row) for row in rows]
        leak = overlap(rows, true_holdouts)
        if (
            len(rows) != 3000
            or len(set(ids)) != 3000
            or len(set(groups)) != 3000
            or leak["theorem_group_overlap"]
            or leak["normalized_statement_overlap"]
            or not all(
                row.get("pantograph_verified") is True
                and row.get("statement_verified") is True
                and row.get("proof_verified") is True
                for row in rows
            )
        ):
            raise RuntimeError(f"{name} failed replay data gate")
        source_path = root / REPLAY_ARMS[name]
        frozen_write_jsonl(source_path, rows)
        effective_rows = []
        for row in rows:
            completion = str(row.get("completion") or "")
            if not completion.strip() or tokenizer.eos_token in completion:
                raise RuntimeError(f"{name} contains an invalid completion")
            effective_rows.append(
                {**row, "completion": completion + tokenizer.eos_token}
            )
        effective = root / "training" / name / "input/effective_train.jsonl"
        frozen_write_jsonl(effective, effective_rows)
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
                "effective_manifest": str(effective),
                "effective_manifest_sha256": sha256(effective),
                "hard_holdout_overlap": leak,
            }
        )
        write_json(root / "audit" / f"eos_gate_{name}.json", gate)
        wb_rows = [row for row in rows if row.get("sampling_source") == "WB"]
        audit[name] = {
            **summarize(
                rows,
                reference_labels=reference_labels,
                reference_total=reference_total,
            ),
            "selection": (
                "fixed_seed_without_replacement"
                if "RANDOM" in name
                else "domain_tactic_length_stratified_round_robin"
            ),
            "wb_distribution": distribution(wb_rows),
            "source_manifest": str(source_path),
            "source_manifest_sha256": sha256(source_path),
            "effective_manifest_sha256": sha256(effective),
            "ordered_record_ids_sha256": hash_values(ids),
            "wb_record_ids_sha256": hashlib.sha256(
                "\0".join(record_id(row) for row in wb_rows).encode()
            ).hexdigest(),
            "holdout_overlap": leak,
        }
    write_json(root / "audit/replay_manifest_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
