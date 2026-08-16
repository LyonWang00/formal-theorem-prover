"""Real API/Pantograph smoke for the retrieval-generate-compile-Verify loop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lean_prover.Planner.client import OpenAICompatibleClient
from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    LeanEnvironmentIdentity,
    TheoremProblem,
    ValidationIssue,
)
from lean_prover.Planner.tests.helpers import full_node
from lean_prover.Repair.planner_subproblem import PlannerSubproblemRepairer
from lean_prover.Verify import BlueprintVerifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    arguments = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        temperature=0.0,
        max_tokens=8192,
    )
    backend = PantographDeclarationCheckingBackend(
        project_path=arguments.lean_project,
        timeout=arguments.timeout,
    )
    checker = BlueprintLeanChecker(backend)
    try:
        problem = TheoremProblem(
            problem_id="retrieval-loop-smoke",
            imports=["Mathlib"],
            natural_language_statement=(
                "For every natural number n, adding zero to n gives n."
            ),
            target_lean_decl="theorem retrieval_loop_target (n : Nat) : n + 0 = n",
        )
        environment = checker.environment_identity(problem)
        node = full_node(
            "L1",
            "lemma L1 (n : Nat) : Nat.obsolete_add_zero n = n",
        )
        node.informal_statement = (
            "For every natural number n, adding zero to n gives n."
        )
        node.preamble.imports = ["Mathlib"]
        blueprint = Blueprint(
            blueprint_summary="One deliberately stale API statement.",
            nodes=[node],
            root_dependencies=["L1"],
            environment=environment,
            problem_hash=problem.problem_hash,
        )
        initial = checker.check_blueprint_detailed(problem, blueprint)
        assert not initial.success
        assert initial.issues

        repairer = PlannerSubproblemRepairer(
            client,
            lean_checker=checker,
        )
        pre_repair_retrieval = repairer._retrieval_payload(
            problem=problem,
            previous=blueprint,
            issues=list(initial.issues),
            environment=environment,
        )
        retrieval_debug_path = arguments.output.with_suffix(
            ".retrieval.json"
        )
        retrieval_debug_path.parent.mkdir(parents=True, exist_ok=True)
        retrieval_debug_path.write_text(
            json.dumps(
                {
                    "environment": environment.model_dump(mode="json"),
                    "initial_issues": [
                        issue.model_dump(mode="json")
                        for issue in initial.issues
                    ],
                    "retrieval": pre_repair_retrieval,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        verifier = BlueprintVerifier(client, max_workers=1)
        rounds: list[dict[str, object]] = []
        current = blueprint
        issues = list(initial.issues)
        verified = False
        for round_id in range(1, 6):
            current = repairer.repair(
                problem=problem,
                previous_blueprint=current,
                issues=issues,
                environment=environment,
            )
            compile_result = checker.check_blueprint_detailed(problem, current)
            round_record: dict[str, object] = {
                "round_id": round_id,
                "statement": current.nodes[0].lean_decl,
                "compile_success": compile_result.success,
                "compile_issues": [
                    issue.model_dump(mode="json")
                    for issue in compile_result.issues
                ],
                "repair_audit": current.metadata["formalization_repair"][-1],
            }
            if compile_result.success:
                verify_results = verifier.verify_blueprint(
                    problem=problem,
                    blueprint=current,
                    environment=environment,
                    node_ids={"L1"},
                )
                verify_result = verify_results[0]
                verified = (
                    verify_result.dependency_statements_correct
                    and verify_result.formal_statement_correct
                )
                round_record["verify_result"] = verify_result.model_dump(
                    mode="json"
                )
                rounds.append(round_record)
                if verified:
                    break
                issues = [
                    ValidationIssue(
                        stage="verify",
                        code="formal_statement_verification_failed",
                        node_id="L1",
                        message=(
                            item.message
                            + " Repair reference: "
                            + item.repair_reference
                        ),
                    )
                    for item in verify_result.formal_statement_issues
                ]
                issues.extend(
                    ValidationIssue(
                        stage="verify",
                        code="dependency_verification_failed",
                        node_id="L1",
                        message=(
                            item.message
                            + " Repair reference: "
                            + item.repair_reference
                        ),
                    )
                    for item in verify_result.dependency_issues
                )
            else:
                rounds.append(round_record)
                issues = list(compile_result.issues)

        payload = {
            "schema_version": "formalization_retrieval_loop_smoke_v1",
            "model": "deepseek-v4-flash",
            "environment": environment.model_dump(mode="json"),
            "initial_statement": blueprint.nodes[0].lean_decl,
            "initial_errors": [
                issue.model_dump(mode="json") for issue in initial.issues
            ],
            "rounds": rounds,
            "success": bool(
                rounds and rounds[-1]["compile_success"] and verified
            ),
            "final_statement": current.nodes[0].lean_decl,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not payload["success"]:
            raise RuntimeError(
                "formalization retrieval loop failed within five rounds"
            )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
