from pathlib import Path

from scripts.summarize_base_m0_b2_comparison import evaluation_root


def test_primary_reuses_locked_m0_and_b2_results() -> None:
    output = Path("new")
    source = Path("source")
    assert evaluation_root("full500", "Base", output, source) == (
        output / "evaluations/full500/Base"
    )
    assert evaluation_root("full500", "M0", output, source) == (
        source / "evaluations/full500/M0"
    )
    assert evaluation_root("strict_unseen_primary", "B2", output, source) == (
        source / "evaluations/strict_unseen_primary/B2"
    )


def test_benchmark_uses_new_three_model_outputs() -> None:
    output = Path("new")
    source = Path("source")
    for model in ("Base", "M0", "B2"):
        assert evaluation_root("benchmark", model, output, source) == (
            output / f"evaluations/benchmark/{model}"
        )
