"""Real DeepSeek smoke for the value-only Verify boundary."""

from __future__ import annotations

import json
import os

from lean_prover.Planner.client import OpenAICompatibleClient
from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    LeanPreamble,
    ProofLengthEstimate,
    SemanticAlignment,
    TheoremProblem,
)
from lean_prover.Verify import BlueprintVerifier


def _node(
    node_id: str,
    *,
    informal_statement: str,
    lean_statement: str,
) -> BlueprintNode:
    return BlueprintNode(
        id=node_id,
        title=node_id,
        informal_statement=informal_statement,
        lean_statement=lean_statement,
        preamble=LeanPreamble(imports=["Mathlib"], raw_header="import Mathlib"),
        semantic_alignment=SemanticAlignment(
            objects=["natural-number objects"],
            hypotheses=[],
            conclusion="the stated relation",
            alignment_notes="Verify will independently audit this claim.",
        ),
        estimated_proof_length=ProofLengthEstimate(
            estimated_lines=1,
            estimated_tokens=8,
            rationale="A statement-only smoke test.",
        ),
    )


def main() -> int:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        temperature=0.0,
        max_tokens=1024,
    )
    backend = PantographDeclarationCheckingBackend(
        project_path="lean_project", timeout=120
    )
    try:
        environment = backend.environment_identity(["Mathlib"])
        problem = TheoremProblem(
            problem_id="verify-semantic-values-api-smoke",
            imports=["Mathlib"],
            natural_language_statement="Semantic Verify smoke.",
            target_lean_decl="theorem target : True",
            header="import Mathlib",
        )
        nodes = [
            _node(
                "L1",
                informal_statement=(
                    "For every natural number n, prove n + 0 = n."
                ),
                lean_statement="lemma L1 (n : Nat) : n + 0 = n",
            ),
            _node(
                "L2",
                informal_statement=(
                    "For natural numbers a and b, assuming a - b = 0, "
                    "prove b <= a."
                ),
                lean_statement=(
                    "lemma L2 (a b : Nat) (h : a - b = 0) : a = b"
                ),
            ),
        ]
        blueprint = Blueprint(
            blueprint_summary="One exact and one mismatched semantic case.",
            nodes=nodes,
            root_dependencies=["L1", "L2"],
            environment=environment,
            problem_hash=problem.problem_hash,
        )
        results = BlueprintVerifier(client, max_workers=2).verify_blueprint(
            problem=problem,
            blueprint=blueprint,
            environment=environment,
        )
        compile_result = BlueprintLeanChecker(backend).check_blueprint_detailed(
            problem, blueprint
        )
        payload = {
            "results": [row.model_dump(mode="json") for row in results],
            "final_statements": {
                node.id: node.lean_decl for node in blueprint.nodes
            },
            "semantic_repair_counts": {
                node.id: node.metadata.get("verify_semantic_repair_count", 0)
                for node in blueprint.nodes
            },
            "pantograph_success": compile_result.success,
            "pantograph_nodes": {
                row.node_id: row.success for row in compile_result.node_results
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        expected = (
            results[0].formal_statement_correct
            and not results[1].formal_statement_correct
            and compile_result.success
            and blueprint.nodes[1].lean_decl
            == "lemma L2 (a b : Nat) (h : a - b = 0) : b <= a"
        )
        return 0 if expected else 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
