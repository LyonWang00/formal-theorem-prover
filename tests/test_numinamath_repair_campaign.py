from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from lean_prover.Dataset.numinamath_pantograph_worker import _source
from lean_prover.Dataset.numinamath_repair_campaign import (
    IndexedFailRows,
    _api_prompt,
    _candidate_from_row,
    _interval_cases_source_variants,
    _is_api_quota_error,
    _parse_api_proof,
    _proof_from_complete_source,
    _statement_fingerprint,
    _statement_fingerprint_without_declaration_name,
    _wrap_target_declaration_with_options,
    build_index,
    parser,
    prepare_active_manual_followup_batch,
    prepare_compatibility_followup_batch,
    prepare_interval_cases_followup_batch,
    prepare_rule_batch,
)
from lean_prover.Dataset.repair_numinamath_failures import write_jsonl


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(record_id: str, source: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "record_hash": _sha(record_id),
        "formal_ground_truth": source,
        "pantograph_verified": "fail",
        "source": "AI-MO/NuminaMath-LEAN",
        "error_message": "declaration uses 'sorry'",
    }


def test_external_index_looks_up_rows_without_mutating_payload(tmp_path: Path) -> None:
    fail_file = tmp_path / "fail.jsonl"
    rows = [
        _row("r1", "import Mathlib\ntheorem one : 1 = 1 := by sorry"),
        _row("r2", "import Mathlib\ntheorem two : 2 = 2 := by sorry"),
    ]
    write_jsonl(fail_file, rows)
    index_file = tmp_path / "index.sqlite3"
    report = build_index(
        fail_file=fail_file,
        index_file=index_file,
        report_file=tmp_path / "index.json",
    )
    assert report["rows"] == 2
    index = IndexedFailRows(fail_file, index_file)
    try:
        assert index.get_row("r2") == rows[1]
        selected = index.query_easy(limit=2, max_source_chars=500, excluded_ids=set())
        assert {item["record_id"] for item in selected} == {"r1", "r2"}
        assert "index_line_number" not in index.get_row("r1")
    finally:
        index.close()


def test_external_index_detects_stale_fail_file(tmp_path: Path) -> None:
    fail_file = tmp_path / "fail.jsonl"
    write_jsonl(fail_file, [_row("r1", "theorem one : True := by sorry")])
    index_file = tmp_path / "index.sqlite3"
    build_index(fail_file=fail_file, index_file=index_file, report_file=tmp_path / "index.json")
    write_jsonl(fail_file, [_row("r2", "theorem two : True := by sorry")])
    with pytest.raises(RuntimeError, match="stale"):
        IndexedFailRows(fail_file, index_file)


def test_rule_batch_excludes_frozen_records_and_assembles_proof(tmp_path: Path) -> None:
    fail_file = tmp_path / "fail.jsonl"
    write_jsonl(
        fail_file,
        [
            _row("frozen", "import Mathlib\ntheorem frozen : 1 = 1 := by sorry"),
            _row("chosen", "import Mathlib\ntheorem chosen : 2 = 2 := by sorry"),
        ],
    )
    index_file = tmp_path / "index.sqlite3"
    build_index(fail_file=fail_file, index_file=index_file, report_file=tmp_path / "index.json")
    frozen = tmp_path / "frozen.jsonl"
    write_jsonl(frozen, [{"record_id": "frozen"}])
    args = argparse.Namespace(
        fail_file=fail_file,
        index_file=index_file,
        limit=10,
        max_source_chars=500,
        batch_id="batch_test",
        max_variants=2,
        easy_family="short_final_placeholder",
        error_family=[],
        strategy_profile="general",
        frozen_manifest=[frozen],
        output_root=tmp_path / "out",
    )
    report = prepare_rule_batch(args)
    assert report["selected_rows"] == 1
    candidate = json.loads(
        (tmp_path / "out" / "batch_test" / "candidate_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert candidate["record_id"] == "chosen"
    assembled = _source(candidate, candidate["variants"][0]["proof"])
    assert assembled == "theorem chosen : 2 = 2 := by\n  norm_num"


def test_api_contract_is_versioned_and_response_is_strict() -> None:
    prompt = _api_prompt(
        "theorem t : True := by",
        "unsolved goals",
        {
            "lean_version": "4.29.1",
            "mathlib_commit": "5e932f97",
            "environment_hash": "46b005cc",
        },
        "by\n  simp",
    )
    assert "Lean: 4.29.1" in prompt
    assert "mathlib commit: 5e932f97" in prompt
    assert "environment hash: 46b005cc" in prompt
    assert "Failed proof attempt to repair minimally" in prompt
    assert "by\n  simp" in prompt
    assert _parse_api_proof('{"proof":"by\\n  trivial"}') == "by\n  trivial"
    assert _parse_api_proof("by\n  trivial") == "by\n  trivial"
    assert _parse_api_proof("```lean\nby\n  trivial\n```") == "by\n  trivial"
    assert _parse_api_proof(
        '{"proof":"by\\n  trivial","rationale":"truncated'
    ) == "by\n  trivial"
    with pytest.raises(ValueError):
        _parse_api_proof('{"proof":"by sorry"}')


def test_api_defaults_prefer_thinking_with_large_retry_budget() -> None:
    args = parser().parse_args(
        [
            "prepare-api",
            "--source-batch",
            "source",
            "--batch-id",
            "batch_test",
        ]
    )
    assert args.thinking_mode == "enabled"
    assert args.retry_thinking_mode == "disabled"
    assert args.reasoning_effort == "high"
    assert args.max_tokens == 16384
    assert args.retry_max_tokens == 32768


def test_api_quota_errors_trigger_batch_circuit_breaker() -> None:
    assert _is_api_quota_error("402 Payment Required: insufficient balance")
    assert _is_api_quota_error("insufficient_quota")
    assert not _is_api_quota_error("ReadTimeout after 300 seconds")


def test_candidate_replaces_existing_failed_proof_without_changing_statement() -> None:
    row = _row(
        "failed-proof",
        "import Mathlib\ntheorem target : 3^2008 % 5 = 1 := by\n  norm_num",
    )
    candidate = _candidate_from_row(
        row,
        batch_id="batch",
        lane="indexed_rule_assisted",
        variants=[{"strategy": "compute_native_decide", "proof": "by\n  native_decide"}],
    )
    assert candidate["source_body"] == "theorem target : 3^2008 % 5 = 1 := by"
    assert _source(candidate, candidate["variants"][0]["proof"]) == (
        "theorem target : 3^2008 % 5 = 1 := by\n  native_decide"
    )


def test_candidate_ignores_declaration_words_inside_proof_comments() -> None:
    row = _row(
        "commented-lemma",
        "import Mathlib\n"
        "theorem target (n : Nat) : n = n := by\n"
        "  -- lemma of mod 8.\n"
        "  have helper : n = n := by rfl\n"
        "  exact helper",
    )
    candidate = _candidate_from_row(
        row,
        batch_id="batch",
        lane="indexed_rule_assisted",
        variants=[{"strategy": "rule_rfl", "proof": "by\n  rfl"}],
    )
    assert candidate["source_body"] == "theorem target (n : Nat) : n = n := by"
    assert "lemma of mod 8" not in candidate["source_body"]
    assert _source(candidate, candidate["variants"][0]["proof"]) == (
        "theorem target (n : Nat) : n = n := by\n  rfl"
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import Mathlib\ntheorem target : True :=", "theorem target : True := by"),
        ("import Mathlib\ntheorem target : True", "theorem target : True := by"),
    ],
)
def test_candidate_normalizes_statement_only_targets(source: str, expected: str) -> None:
    row = _row("missing-proof", source)
    candidate = _candidate_from_row(
        row,
        batch_id="batch",
        lane="indexed_rule_assisted",
        variants=[{"strategy": "rule_simp", "proof": "by\n  simp"}],
    )
    assert candidate["source_body"] == expected


def test_resource_options_wrap_target_declaration_after_open_preamble() -> None:
    source = (
        "open Real\n\n"
        "theorem target (x : ℝ) : sqrt (x ^ 2) = |x| := by"
    )
    wrapped = _wrap_target_declaration_with_options(
        source,
        [("maxHeartbeats", 1_000_000), ("maxRecDepth", 10_000)],
    )
    assert wrapped == (
        "open Real\n\n"
        "set_option maxHeartbeats 1000000 in\n"
        "set_option maxRecDepth 10000 in\n"
        "theorem target (x : ℝ) : sqrt (x ^ 2) = |x| := by"
    )
    assert not wrapped.startswith("set_option")


def test_raw_recovery_requires_statement_alignment_and_extracts_clean_proof() -> None:
    statement = "import Mathlib\n/- prose -/\ntheorem target (n : Nat) : n = n := by"
    proof_source = "import Mathlib\n-- alternate prose\ntheorem target (n : Nat) : n = n := by\n  rfl"
    assert _statement_fingerprint(statement) == _statement_fingerprint(proof_source)
    assert _proof_from_complete_source(proof_source) == "by\n  rfl"
    with pytest.raises(ValueError, match="forbidden"):
        _proof_from_complete_source(statement + "\n  sorry")
    renamed = proof_source.replace("theorem target", "theorem alternate_name")
    assert _statement_fingerprint(statement) != _statement_fingerprint(renamed)
    assert (
        _statement_fingerprint_without_declaration_name(statement)
        == _statement_fingerprint_without_declaration_name(renamed)
    )


def test_candidate_normalizes_lean3_finset_binder_without_semantic_change() -> None:
    row = _row(
        "sum-scope",
        "import Mathlib\ntheorem target : ∑ k in Finset.range 2, k = 1 := by sorry",
    )
    candidate = _candidate_from_row(
        row,
        batch_id="batch",
        lane="indexed_rule_assisted",
        variants=[{"strategy": "rule_norm_num", "proof": "by\n  norm_num"}],
    )
    assert "∑ k ∈ Finset.range 2" in candidate["source_body"]
    assert "∑ k in Finset.range 2" not in candidate["source_body"]
    assert candidate["statement_changed"] is False
    assert candidate["changes"] == [{
        "kind": "syntax_normalization",
        "operation": "lean3_finset_binder_in_to_membership",
        "replacement_count": 1,
        "semantic_change": False,
    }]


def test_interval_cases_repair_splits_hypothesis_and_goal_normalization() -> None:
    row = _row(
        "interval-no-goals",
        "import Mathlib\ntheorem target (n : Nat) (h : n = 3) : n = 3 := by\n"
        "  interval_cases n <;> norm_num at h ⊢ <;> all_goals trivial",
    )
    row["error_message"] = "4:20-4:35: error: No goals to be solved"
    variants = _interval_cases_source_variants(row)
    assert variants == [{
        "strategy": "rule_interval_cases_split_hypothesis_goal_norm_num",
        "proof": (
            "by\n  interval_cases n <;> norm_num at h <;> norm_num"
            " <;> all_goals trivial"
        ),
    }]

    row["formal_ground_truth"] = (
        "import Mathlib\ntheorem target (n : Nat) (h1 h2 : n = 3) : n = 3 := by\n"
        "  interval_cases n <;> norm_num at h1 h2 ⊢"
    )
    variants = _interval_cases_source_variants(row)
    assert variants[0]["proof"].endswith(
        "norm_num at h1 <;> norm_num at h2 <;> norm_num"
    )

    row["formal_ground_truth"] = (
        "import Mathlib\ntheorem target (n : Nat) (h1 h2 : n = 3) : n = 3 := by\n"
        "  interval_cases n <;> norm_num [Nat.succ_eq_add_one] at h1 h2 ⊢"
    )
    variants = _interval_cases_source_variants(row)
    assert variants[0]["proof"].endswith(
        "norm_num [Nat.succ_eq_add_one] at h1 <;> "
        "norm_num [Nat.succ_eq_add_one] at h2 <;> "
        "norm_num [Nat.succ_eq_add_one]"
    )


def test_interval_cases_repair_ignores_unrelated_proofs() -> None:
    row = _row("unrelated", "import Mathlib\ntheorem target : True := by\n  trivial")
    assert _interval_cases_source_variants(row) == []


def test_interval_followup_only_emits_changed_failed_proofs(tmp_path: Path) -> None:
    fail_file = tmp_path / "fail.jsonl"
    row = _row(
        "changed",
        "import Mathlib\ntheorem target (n : Nat) (h1 h2 : n = 3) : n = 3 := by\n"
        "  interval_cases n <;> norm_num at h1 h2 ⊢",
    )
    row["error_message"] = "error: No goals to be solved"
    write_jsonl(fail_file, [row])
    index_file = tmp_path / "index.sqlite3"
    build_index(fail_file=fail_file, index_file=index_file, report_file=tmp_path / "index.json")
    source_batch = tmp_path / "source"
    source_batch.mkdir()
    write_jsonl(source_batch / "candidate_manifest.jsonl", [{
        "record_id": "changed",
        "variants": [{"strategy": "old", "proof": "by\n  norm_num"}],
    }])
    write_jsonl(source_batch / "verification_results.jsonl", [{
        "record_id": "changed",
        "success": False,
    }])
    args = argparse.Namespace(
        fail_file=fail_file,
        index_file=index_file,
        source_batch=source_batch,
        output_root=tmp_path / "out",
        batch_id="followup",
        frozen_manifest=[],
    )
    report = prepare_interval_cases_followup_batch(args)
    assert report["selected_rows"] == 1
    candidate = json.loads(
        (tmp_path / "out" / "followup" / "candidate_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert candidate["variants"][0]["proof"].endswith(
        "norm_num at h1 <;> norm_num at h2 <;> norm_num"
    )


def test_compatibility_followup_preserves_active_repair_and_renames_api(tmp_path: Path) -> None:
    source_batch = tmp_path / "source"
    source_batch.mkdir()
    write_jsonl(source_batch / "candidate_manifest.jsonl", [{
        "record_id": "changed",
        "batch_id": "source",
        "source_body": "theorem target (n : Nat) : n ^ 2 <= n ^ 3 := by",
        "variants": [{
            "strategy": "interval_fix",
            "proof": (
                "by\n  apply Nat.pow_le_pow_of_le_right (by omega)\n  omega\n"
                "  exact Set.ncard_coe_Finset {1}"
            ),
        }],
    }])
    write_jsonl(source_batch / "verification_results.jsonl", [{
        "record_id": "changed",
        "success": False,
        "errors": ["Unknown constant `Nat.pow_le_pow_of_le_right`"],
    }])
    args = argparse.Namespace(
        source_batch=source_batch,
        output_root=tmp_path / "out",
        batch_id="compatibility_followup",
        frozen_manifest=[],
    )
    report = prepare_compatibility_followup_batch(args)
    assert report["selected_rows"] == 1
    candidate = json.loads(
        (tmp_path / "out" / "compatibility_followup" / "candidate_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    proof = candidate["variants"][0]["proof"]
    assert "Nat.pow_le_pow_right" in proof
    assert "Set.ncard_coe_finset" in proof
    assert "pow_le_pow_of_le_right" not in proof


def test_active_manual_followup_is_hash_guarded_and_exact(tmp_path: Path) -> None:
    source_batch = tmp_path / "source"
    source_batch.mkdir()
    source_candidate = {
        "record_id": "changed",
        "batch_id": "source",
        "candidate_hash": "source-hash",
        "source_body": "theorem target : True := by",
        "variants": [{"strategy": "old", "proof": "by\n  trivial\n  done"}],
    }
    write_jsonl(source_batch / "candidate_manifest.jsonl", [source_candidate])
    write_jsonl(source_batch / "verification_results.jsonl", [{
        "record_id": "changed",
        "success": False,
    }])
    edits = tmp_path / "edits.jsonl"
    write_jsonl(edits, [{
        "record_id": "changed",
        "source_candidate_hash": "source-hash",
        "rationale": "The first tactic closes the goal; remove the invalid tail.",
        "operations": [{"old": "\n  done", "new": "", "expected_count": 1}],
    }])
    args = argparse.Namespace(
        source_batch=source_batch,
        edits_file=edits,
        output_root=tmp_path / "out",
        batch_id="manual_followup",
        frozen_manifest=[],
    )
    report = prepare_active_manual_followup_batch(args)
    assert report["selected_rows"] == 1
    candidate = json.loads(
        (tmp_path / "out" / "manual_followup" / "candidate_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert candidate["variants"] == [{
        "strategy": "codex_manual_active_followup",
        "proof": "by\n  trivial",
    }]
