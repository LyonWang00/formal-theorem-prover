from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from lean_prover.Dataset.build_verified_datasets import (
    SUCCESS,
    add_dataset_contract,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


materializer = load_script("materialize_numinamath_manual_verified_supplement")
finalizer = load_script("finalize_numinamath_expand_verification")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def parent_row() -> dict:
    statement = "import Mathlib\n\ntheorem canonical_parent (x : Nat) : x + 0 = x"
    proof = "by\n  simp"
    row = {
        "uuid": "parent-uuid",
        "record_id": "parent::row_000001",
        "problem": "Prove the additive identity.",
        "question_type": "math-word-problem",
        "lean_statement": statement,
        "formal_statement": statement + " := by",
        "proof": proof,
        "formal_ground_truth": statement + " := " + proof,
        "formal_proof": statement + " := " + proof,
        "proof_present": True,
        "proof_source": "human",
        "verification_status": SUCCESS,
        "raw_verification_status": SUCCESS,
        "raw_verifier_success": True,
        "verification_backend": "pantograph",
        "verification_scope": "full_proof",
        "timed_out": False,
        "lean_version": "Lean test",
        "mathlib_commit": "0" * 40,
        "environment_hash": "1" * 64,
        "assembled_source_hash": "2" * 64,
        "upstream_source": "canonical",
    }
    return add_dataset_contract(row, source="AI-MO/NuminaMath-LEAN", status=SUCCESS)


def supplement_row(parent: dict) -> dict:
    statement = (
        "import Mathlib\n\ntheorem manual_extension (x y : Nat) "
        ": x + y + 0 = x + y"
    )
    proof = "by\n  simp"
    row = copy.deepcopy(parent)
    row.update(
        {
            "uuid": "supplement-uuid",
            "record_id": "supplement::manual_expansion",
            "problem": "Prove the two-variable additive identity.",
            "lean_statement": statement,
            "formal_statement": statement + " := by",
            "proof": proof,
            "formal_ground_truth": statement + " := " + proof,
            "formal_proof": statement + " := " + proof,
            "upstream_source": f"parent:{parent['record_id']}",
            "repair": {
                "parent_record_id": parent["record_id"],
                "parent_statement_sha256": parent["statement_sha256"],
                "parent_proof_sha256": parent["proof_sha256"],
                "quality_tier": "high",
            },
        }
    )
    return add_dataset_contract(
        row, source="AI-MO/NuminaMath-LEAN", status=SUCCESS
    )


def test_parent_provenance_is_bound_field_by_field():
    parent = parent_row()
    candidate = {
        "record_id": "manual_candidate",
        "manual_provenance": {
            "parent_record_id": parent["record_id"],
            "parent_statement_sha256": parent["statement_sha256"],
            "parent_proof_sha256": parent["proof_sha256"],
        },
    }
    materializer.validate_parent_binding(candidate, parent)
    stale = copy.deepcopy(candidate)
    stale["manual_provenance"]["parent_proof_sha256"] = "f" * 64
    with pytest.raises(materializer.MaterializationError, match="stale parent proof"):
        materializer.validate_parent_binding(stale, parent)


def test_normalized_and_conservative_near_duplicate_gates_ignore_theorem_name():
    first = materializer.normalized_statement(
        "theorem first (x : Nat) : x + 0 = x"
    )
    renamed = materializer.normalized_statement(
        "theorem renamed (x : Nat) : x + 0 = x"
    )
    assert first == renamed
    long_statement = "theorem " + "A" * 200
    near_statement = long_statement[:-1] + "B"
    hit = materializer.conservative_near_duplicate(
        near_statement, {long_statement: "canonical-id"}
    )
    assert hit is not None and hit[0] == "canonical-id"


def test_main_finalizer_atomically_merges_hash_locked_supplement(tmp_path: Path):
    parent = parent_row()
    supplement = supplement_row(parent)
    raw = tmp_path / "raw.jsonl"
    results = tmp_path / "verification_results.jsonl"
    reviews = tmp_path / "quality_review.jsonl"
    parents = tmp_path / "parents.jsonl"
    supplement_path = tmp_path / "verified_supplement.jsonl"
    for path in (raw, results, reviews):
        write_jsonl(path, [])
    write_jsonl(parents, [parent])
    write_jsonl(supplement_path, [supplement])
    supplement_sha = hashlib.sha256(supplement_path.read_bytes()).hexdigest()
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()

    args = argparse.Namespace(
        raw=raw,
        expected_raw_sha256=raw_sha,
        verification_results=[results],
        candidate_manifests=[],
        quality_review=[reviews],
        parent_source=parents,
        verified_supplement=[supplement_path],
        expected_supplement_sha256=[supplement_sha],
        expected_environment_hash="",
        success_output=tmp_path / "success.jsonl",
        fail_output=tmp_path / "fail.jsonl",
        quality_rejected_output=tmp_path / "rejected.jsonl",
        pending_output=tmp_path / "pending.jsonl",
        audit_output=tmp_path / "audit.json",
        max_low_quality_ratio=0.05,
        require_complete=True,
        overwrite=False,
    )
    audit = finalizer.finalize(args)
    success = [json.loads(line) for line in args.success_output.read_text().splitlines()]
    assert success == [supplement]
    assert audit["verified_supplements"]["rows"] == 1
    assert audit["invariants"]["supplement_rows_merged_into_success"] is True
    assert audit["invariants"]["outputs_installed_atomically"] is True


def test_supplement_loader_rejects_error_fields_and_wrong_parent_semantics(tmp_path: Path):
    parent = parent_row()
    bad = supplement_row(parent)
    bad["repair_error"] = "must never survive on success"
    path = tmp_path / "verified_supplement.jsonl"
    write_jsonl(path, [bad])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="contains error fields"):
        finalizer.load_verified_supplements([path], [digest], {parent["record_id"]: parent})

    bad = supplement_row(parent)
    bad["question_type"] = "proof"
    path = tmp_path / "verified_supplement_semantics.jsonl"
    write_jsonl(path, [bad])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="question_type does not match"):
        finalizer.load_verified_supplements([path], [digest], {parent["record_id"]: parent})


def test_materialize_then_check_rebuilds_identical_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The report describes the artifact, so check mode is byte-identical."""
    monkeypatch.setattr(materializer, "ROOT", tmp_path)
    monkeypatch.setattr(
        materializer,
        "build",
        lambda **_kwargs: ([{"record_id": "accepted-row"}], {"stable": "value"}),
    )
    candidate = tmp_path / "candidate_manifest.jsonl"
    results = tmp_path / "verification_results.jsonl"
    review = tmp_path / "manual_review.jsonl"
    canonical = tmp_path / "canonical_success.jsonl"
    output = tmp_path / "outputs" / "supplement.jsonl"
    report = tmp_path / "outputs" / "supplement.report.json"
    common = [
        "materializer",
        "--candidate-manifest",
        str(candidate),
        "--expected-candidate-sha256",
        "0" * 64,
        "--verification-results",
        str(results),
        "--manual-review",
        str(review),
        "--canonical-success",
        str(canonical),
        "--output",
        str(output),
        "--report",
        str(report),
    ]

    monkeypatch.setattr(sys, "argv", common + ["--materialize"])
    assert materializer.main() == 0
    installed = report.read_bytes()
    installed_sha256 = hashlib.sha256(installed).hexdigest()
    assert json.loads(installed)["write_explicitly_authorized"] is True

    monkeypatch.setattr(sys, "argv", common + ["--check"])
    assert materializer.main() == 0
    assert report.read_bytes() == installed
    assert hashlib.sha256(report.read_bytes()).hexdigest() == installed_sha256
