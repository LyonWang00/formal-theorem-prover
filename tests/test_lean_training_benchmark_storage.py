import json

import pytest

from lean_prover.lean_training.evaluation.benchmark import (
    JsonlStore,
    _max_lora_rank,
    classify_generation_output,
    extract_proof_body,
    read_benchmark_records,
)


def test_extractor_prefers_lean_fence_over_natural_language():
    completion = "Here is the proof:\n```lean\nby\n  simp\n```\nExtra text"
    assert extract_proof_body(completion) == "by\n  simp"


def test_extractor_rejects_unfenced_natural_language():
    assert extract_proof_body("The proof follows from the assumptions.") == ""


def test_generation_output_classifier_detects_multiple_and_repetitive_proofs():
    assert classify_generation_output("by\n  simp\n\nby\n  norm_num") == "multiple_proofs"
    repeated = "by\n" + "\n".join(["  simp"] * 24)
    assert classify_generation_output(repeated) == "repetitive_output"


def test_jsonl_store_clears_old_shards_when_not_resuming(tmp_path):
    shards = tmp_path / "attempt_shards"
    shards.mkdir()
    (shards / "old.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "attempt_shards.zip").write_text("old", encoding="utf-8")

    store = JsonlStore(tmp_path, resume=False)
    try:
        assert shards.exists()
        assert not (shards / "old.jsonl").exists()
        assert not (tmp_path / "attempt_shards.zip").exists()
    finally:
        store.close()


def test_duplicate_problem_id_rejected(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    rows = [
        {"id": "dup", "lean_statement": "theorem a : True", "prompt": "p"},
        {"id": "dup", "lean_statement": "theorem b : True", "prompt": "p"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate benchmark problem_id"):
        read_benchmark_records(str(path), None)


def test_vllm_max_lora_rank_follows_peft_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 32, "rank_pattern": {}}), encoding="utf-8"
    )
    assert _max_lora_rank(str(tmp_path)) == 32
