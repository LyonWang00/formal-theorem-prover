"""Run a real Problem -> DeepSeek -> Pantograph statement check.

This script is intentionally not part of the normal test suite. It calls the
configured LLM API and starts Pantograph/Lean.

Example:

    $env:DEEPSEEK_API_KEY = "..."
    python -m lean_prover.Planner.examples.real_test
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from lean_prover.Planner.client import create_deepseek_client_from_env
from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.schemas import TheoremProblem
from lean_prover.Planner.service import PlannerService


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def default_project_path() -> Path:
    return Path(__file__).resolve().parents[3] / "lean_project"


def main() -> int:
    configure_utf8_stdio()

    problem = TheoremProblem(
        problem_id="linear_map_rank_nullity",
        imports=["Mathlib"],
        natural_language_statement=(
            "Rank-nullity theorem for a linear map between vector spaces: "
            "if V is finite-dimensional over a field K, then the dimension "
            "of V is the sum of the dimension of the kernel of T and the "
            "dimension of the range of T. This integration test should use "
            "at least one genuine intermediate lemma about the kernel, range, "
            "or an existing rank-nullity dimension formula."
        ),
        target_lean_decl=(
            "theorem rank_nullity "
            "(K V W : Type*) [Field K] "
            "[AddCommGroup V] [Module K V] "
            "[AddCommGroup W] [Module K W] "
            "[FiniteDimensional K V] "
            "(T : V →ₗ[K] W) : "
            "Module.finrank K V = "
            "Module.finrank K T.ker + Module.finrank K T.range"
        ),
    )

    project_path = Path(
        os.environ.get(
            "LEAN_PROJECT_PATH",
            str(default_project_path()),
        )
    )
    timeout = int(os.environ.get("LEAN_BACKEND_TIMEOUT", "120"))
    client = create_deepseek_client_from_env()
    with PantographDeclarationCheckingBackend(
        project_path=project_path,
        timeout=timeout,
    ) as pantograph_backend:
        service = PlannerService(
            client=client,
            lean_checker=BlueprintLeanChecker(pantograph_backend),
            max_attempts=int(os.environ.get("PLANNER_MAX_ATTEMPTS", "3")),
        )
        result = service.plan(problem)

    report = {
        "problem_id": problem.problem_id,
        "target_lean_decl": problem.target_lean_decl,
        **result.model_dump(mode="json", by_alias=True),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
