from __future__ import annotations

import inspect

import pytest

from lean_prover.lean_training.expert_iteration.discovery_generator import (
    _generation_id,
)
from scripts import analyze_ld_length_ablation as length_analysis
from scripts import analyze_ld_easy_pilot as pilot_analysis
from scripts import finalize_ld_difficulty_review as difficulty_finalizer
from scripts import orchestrate_ld_length_ablation as length_orchestrator
from scripts import prepare_ld_difficulty_review as difficulty_review


def test_generation_identity_isolated_by_max_new_tokens() -> None:
    common = {
        "statement_id": "statement",
        "iteration": -30,
        "checkpoint": "checkpoint",
        "sample_index": 0,
    }
    identities = {
        _generation_id(**common, max_new_tokens=length)
        for length in (256, 512, 1024)
    }
    assert len(identities) == 3


def test_paired_statistics_preserves_wins_losses_ties() -> None:
    result = length_analysis.paired_statistics(
        {"a": 1, "b": 0, "c": 1, "d": 0},
        {"a": 0, "b": 1, "c": 1, "d": 0},
        seed=42,
        draws=1000,
    )
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["ties"] == 2
    assert result["delta_pass_at_4"] == pytest.approx(0.0)
    assert result["mcnemar_exact_p"] == pytest.approx(1.0)


def test_pairing_audit_rejects_seed_or_prompt_drift() -> None:
    rows = {}
    for length in (256, 512, 1024):
        rows[length] = [
            {
                "statement_id": "s",
                "sample_index": 0,
                "prompt": "prompt",
                "generation_seed": 7,
                "max_new_tokens": length,
            }
        ]
    assert length_analysis.validate_pairing(rows)["candidate_seed_match"]
    rows[512][0]["generation_seed"] = 8
    with pytest.raises(ValueError, match="seed mismatch"):
        length_analysis.validate_pairing(rows)


def test_difficulty_preparation_has_no_automatic_label_function() -> None:
    source = inspect.getsource(difficulty_review)
    assert "difficulty =" not in source
    assert '"difficulty":' not in source
    assert "weighted_score" not in source
    assert not hasattr(difficulty_review, "classify_difficulty")


def test_review_metrics_are_auxiliary_only() -> None:
    row = {
        "id": "sample",
        "statement": "theorem t (x : Nat) : x = x",
        "proof": "by rfl",
        "proof_style": "tactic_by",
        "label_tokens": 3,
        "statement_tokens": 12,
        "tactic_trace": [{"tactic": "rfl"}],
        "premises": [
            {
                "qualified_name": "Eq.refl",
                "source_file": "Init/Prelude.lean",
                "is_same_file": False,
            }
        ],
        "source_file": "Mathlib/Data/Nat/Basic.lean",
    }
    metrics = difficulty_review.review_metrics(row)
    assert metrics["proof_tokens"] == 3
    assert metrics["tactic_steps"] == 1
    assert metrics["premise_count"] == 1
    assert "difficulty" not in metrics


def test_review_uses_immutable_default_timeout_verification_receipt() -> None:
    row = {
        "verification_status": "verified_default_timeout",
        # The retrace-era top-level field is stale in the frozen final pool.
        "pantograph_verified": False,
        "verification": {
            "compile_success": True,
            "timed_out": False,
            "error_category": "",
        },
    }
    assert difficulty_review.default_timeout_verified(row)
    row["verification"]["timed_out"] = True
    assert not difficulty_review.default_timeout_verified(row)


def test_transient_retry_is_limited_to_gpu_engine_start_failures(tmp_path) -> None:
    oom = tmp_path / "oom.log"
    oom.write_text("Engine core initialization failed: CUDA out of memory")
    semantic = tmp_path / "semantic.log"
    semantic.write_text("ValueError: unexpected frozen canary composition")
    assert length_orchestrator.is_transient_generation_start_failure(oom)
    assert not length_orchestrator.is_transient_generation_start_failure(semantic)


def test_difficulty_finalizer_does_not_infer_labels() -> None:
    source = inspect.getsource(difficulty_finalizer)
    assert "classify_difficulty" not in source
    assert "predict_difficulty" not in source
    assert "trainable_for_short_whole_proof" in source
    assert "label_validation_errors.json" in source


def test_ld_easy_promotion_requires_gain_and_retention() -> None:
    baseline_metrics = {
        "ld_easy_holdout64": {"solved": 2},
        "wb_gate150": {"solved": 80},
        "monitor64": {"solved": 10},
    }
    candidate_metrics = {
        "ld_easy_holdout64": {"solved": 3},
        "wb_gate150": {"solved": 78},
        "monitor64": {"solved": 9},
    }
    baseline_behavior = {
        "ld_easy_holdout64": {
            "repetition_ratio": 0.10,
            "length_finish_ratio": 0.10,
            "proof_extraction_success_rate": 0.95,
            "format_validity_rate": 0.95,
        }
    }
    candidate_behavior = {
        "ld_easy_holdout64": {
            "repetition_ratio": 0.11,
            "length_finish_ratio": 0.11,
            "proof_extraction_success_rate": 0.95,
            "format_validity_rate": 0.95,
        }
    }
    result = pilot_analysis.core_promotion(
        candidate_metrics,
        baseline_metrics,
        candidate_behavior,
        baseline_behavior,
    )
    assert result["promoted"]
    candidate_metrics["wb_gate150"]["solved"] = 77
    assert not pilot_analysis.core_promotion(
        candidate_metrics,
        baseline_metrics,
        candidate_behavior,
        baseline_behavior,
    )["promoted"]
