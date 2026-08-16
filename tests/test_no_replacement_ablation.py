from __future__ import annotations

import json
from pathlib import Path

import pytest

from lean_prover.lean_training.data.contracts import make_attestation_id
from lean_prover.lean_training.data.random_manifest import (
    build_random_subset_manifest,
    deterministic_permutation,
    read_jsonl,
    validate_manifest,
)
from lean_prover.lean_training.sft_pipeline.trainer import FixedManifestSampler


def verified_row(index: int, *, origin: str = "") -> dict:
    record_id = f"record_{index:03d}"
    environment_hash = "env"
    assembler_version = "2"
    normalization_version = "2"
    source_hash = f"source_{index:03d}"
    row = {
        "record_id": record_id,
        "statement_id": f"statement_{index:03d}",
        "id": f"statement_{index:03d}",
        "data_state": "verified",
        "data_role": "train",
        "statement_verified": True,
        "proof_verified": True,
        "pantograph_verified": True,
        "truncated": False,
        "prompt": f"prompt {index}",
        "proof": "by simp",
        "completion": "by simp",
        "environment_hash": environment_hash,
        "assembler_version": assembler_version,
        "normalization_version": normalization_version,
        "assembled_source_hash": source_hash,
    }
    if origin:
        row["origin_data_role"] = origin
    row["attestation_id"] = make_attestation_id(
        record_id=record_id,
        environment_hash=environment_hash,
        assembler_version=assembler_version,
        normalization_version=normalization_version,
        assembled_source_hash=source_hash,
    )
    return row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_random_subset_is_deterministic_and_without_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [verified_row(index) for index in range(20)]
    write_jsonl(source, rows)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    build_random_subset_manifest(source, first, 12, 20260720)
    build_random_subset_manifest(source, second, 12, 20260720)
    assert first.read_bytes() == second.read_bytes()
    selected = read_jsonl(first)
    assert len({row["record_id"] for row in selected}) == 12
    assert len({row["statement_id"] for row in selected}) == 12
    assert validate_manifest(selected, allow_origin_discovery=False)["all_verified"]


def test_random_subset_seed_and_capacity_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [verified_row(index) for index in range(20)])
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    build_random_subset_manifest(source, first, 12, 1)
    build_random_subset_manifest(source, second, 12, 2)
    assert first.read_bytes() != second.read_bytes()
    with pytest.raises(ValueError, match="eligible=20"):
        build_random_subset_manifest(source, tmp_path / "too_many.jsonl", 21, 1)


def test_master_prefix_contracts() -> None:
    master = deterministic_permutation(
        [verified_row(index) for index in range(1200)], seed=20260720
    )
    b1 = master[:1000]
    b2 = master[:847]
    c1 = master[:160]
    c3 = master[:39]
    assert b2 == b1[:847]
    assert c1 == b1[:160]
    assert c3 == b1[:39]


def test_frontier_rows_are_unique_verified_and_only_one_or_two_of_four() -> None:
    frontier = []
    for index, success_count in enumerate((1, 2, 1, 2)):
        row = verified_row(index, origin="discovery")
        row["sampling_source"] = "expert"
        row["frontier_success_count_at_4"] = success_count
        frontier.append(row)
    summary = validate_manifest(frontier, allow_origin_discovery=True)
    assert summary["unique_statement_ids"] == len(frontier)
    assert {row["frontier_success_count_at_4"] for row in frontier} == {1, 2}


def test_manifest_rejects_unverified_truncated_and_duplicate_statement() -> None:
    bad = verified_row(0)
    bad["pantograph_verified"] = False
    with pytest.raises(ValueError):
        validate_manifest([bad], allow_origin_discovery=False)
    truncated = verified_row(1)
    truncated["truncated"] = True
    with pytest.raises(ValueError):
        validate_manifest([truncated], allow_origin_discovery=False)
    duplicate = [verified_row(2), {**verified_row(3), "statement_id": "statement_002"}]
    with pytest.raises(ValueError, match="statement_id"):
        validate_manifest(duplicate, allow_origin_discovery=False)


def test_fixed_sampler_draws_100_rows_once_and_refuses_second_epoch() -> None:
    sampler = FixedManifestSampler(100, seed=20260721)
    draws = list(sampler)
    assert len(draws) == 100
    assert len(set(draws)) == 100
    assert max(__import__("collections").Counter(draws).values()) == 1
    assert (len(draws) + 4 - 1) // 4 == 25
    with pytest.raises(RuntimeError, match="second epoch"):
        list(sampler)
