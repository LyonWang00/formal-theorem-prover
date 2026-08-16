from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_ablation_evaluation import dataset_uses_source_faithful_templates
from scripts.run_ei_ablation_core_evaluation import (
    DATASETS,
    assert_source_faithful_errors_clean,
    source_faithful_complete,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_ld_easy_uses_source_faithful_orchestrator() -> None:
    role, dataset, size = DATASETS["ld_easy64"]
    assert role == "ld_holdout"
    assert dataset.endswith("ld_easy_holdout_64.jsonl")
    assert size == 64


def test_generic_orchestrator_detects_source_templates(tmp_path: Path) -> None:
    source_faithful = tmp_path / "ld.jsonl"
    ordinary = tmp_path / "wb.jsonl"
    write_jsonl(source_faithful, [{"preassembled_source_template": "theorem t :=\n__P__"}])
    write_jsonl(ordinary, [{"statement": "theorem t : True"}])
    assert dataset_uses_source_faithful_templates(source_faithful)
    assert not dataset_uses_source_faithful_templates(ordinary)


def test_source_faithful_completion_contract(tmp_path: Path) -> None:
    rows = [{"generation_id": str(index)} for index in range(4)]
    write_jsonl(tmp_path / "attempts.jsonl", rows)
    write_jsonl(tmp_path / "generations.jsonl", rows)
    write_jsonl(tmp_path / "verifications.jsonl", rows)
    (tmp_path / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "data_role": "ld_holdout",
                "execution_mode": "staged_sequential_grouped_source_faithful",
                "import_group_count": 2,
            }
        ),
        encoding="utf-8",
    )
    assert source_faithful_complete(tmp_path, expected=2)

    summary = json.loads((tmp_path / "benchmark_summary.json").read_text())
    summary["data_role"] = "benchmark"
    (tmp_path / "benchmark_summary.json").write_text(json.dumps(summary))
    assert not source_faithful_complete(tmp_path, expected=2)


def test_source_faithful_error_gate_rejects_preloaded_declarations(
    tmp_path: Path,
) -> None:
    write_jsonl(
        tmp_path / "verifications.jsonl",
        [
            {
                "metadata": {
                    "compile_errors": ["error: `Existing.name` has already been declared"]
                }
            }
        ],
    )
    with pytest.raises(RuntimeError, match="preloaded declarations"):
        assert_source_faithful_errors_clean(tmp_path)

