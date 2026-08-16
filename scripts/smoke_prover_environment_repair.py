"""Real API + local-index + Pantograph proof-repair smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

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
from lean_prover.Prover import BlueprintProver, GeneratedProof, NodeProofVerifier
from lean_prover.Repair.prover_proof import ProverProofRepairer
from lean_prover.lean_training.verification.pantograph import (
    PantographTheoremVerifier,
)


class DirectedFailureProofGenerator:
    model = "directed-old-api-stub"

    def __init__(self, proof: str) -> None:
        self.proof = proof

    def generate(self, request):
        return GeneratedProof(
            proof=self.proof,
            model=self.model,
            metadata={"directed_failure": self.proof},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lean-project",
        type=Path,
        default=Path("lean_project"),
    )
    parser.add_argument(
        "--failure-mode",
        choices=("old_reference", "unsolved_goals", "type_mismatch"),
        default="old_reference",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    project = args.lean_project.resolve()
    problem = TheoremProblem(
        problem_id="prover-environment-repair-smoke",
        imports=["Mathlib"],
        header="import Mathlib\nopen Nat",
        natural_language_statement=(
            "For every natural number n, prove that n plus zero equals n."
        ),
        target_lean_decl="theorem target (n : Nat) : n + 0 = n",
    )
    backend = PantographDeclarationCheckingBackend(
        project_path=project,
        timeout=120,
    )
    runtime = None
    try:
        environment = backend.environment_identity(problem.imports)
        blueprint = Blueprint(
            blueprint_summary="One direct arithmetic identity.",
            nodes=[
                BlueprintNode(
                    id="L1",
                    title="Add zero",
                    informal_statement=(
                        "For every natural number n, n plus zero equals n."
                    ),
                    informal_proof=(
                        "Apply the current-environment natural-number add-zero "
                        "identity to n."
                    ),
                    logical_ideas=["Use the add-zero identity"],
                    lean_statement="lemma L1 (n : Nat) : n + 0 = n",
                    preamble=LeanPreamble(
                        imports=["Mathlib"],
                        raw_header=problem.header,
                    ),
                    semantic_alignment=SemanticAlignment(
                        objects=["n is a natural number"],
                        hypotheses=[],
                        conclusion="n + 0 = n",
                        alignment_notes="The objects and equality match exactly.",
                    ),
                    estimated_proof_length=ProofLengthEstimate(
                        estimated_lines=1,
                        estimated_tokens=8,
                        rationale="A current environment identity suffices.",
                    ),
                    dependency_statements_verified=True,
                    formal_statement_verified=True,
                )
            ],
            root_dependencies=["L1"],
            environment=environment,
            problem_hash=problem.problem_hash,
        )
        runtime = PantographTheoremVerifier(
            project,
            imports=("Mathlib",),
            timeout=120,
        )
        directed_proof = {
            "old_reference": "exact Nat.obsolete_add_zero n",
            "unsolved_goals": "skip",
            "type_mismatch": "exact 0",
        }[args.failure_mode]
        prover = BlueprintProver(
            generator=DirectedFailureProofGenerator(directed_proof),
            verifier=NodeProofVerifier(runtime),
            proof_repairer=ProverProofRepairer(
                client=OpenAI(
                    api_key=key,
                    base_url="https://api.deepseek.com",
                ),
                model="deepseek-v4-flash",
                project_path=project,
            ),
            attempts_per_node=1,
            max_proof_repair_rounds=5,
        )
        result = prover.prove_node(
            problem=problem,
            blueprint=blueprint,
            node_id="L1",
        )
        node = result.nodes[0]
        output = {
            "success": node.verified_proof is not None,
            "verified_proof": node.verified_proof,
            "attempts": [
                {
                    "attempt_id": attempt.attempt_id,
                    "prompt_type": attempt.prompt_type,
                    "proof": attempt.proof,
                    "success": attempt.success,
                    "error_type": attempt.error_type,
                    "error_detail": attempt.error_detail,
                    "diagnostics": attempt.diagnostics,
                    "repair_round": attempt.metadata.get("repair_round"),
                    "candidate_id": attempt.metadata.get("candidate_id"),
                    "retrieved_names": [
                        item.get("full_name")
                        for item in (
                            attempt.metadata.get("retrieval", {}).get(
                                "candidate_declarations", []
                            )
                            if isinstance(
                                attempt.metadata.get("retrieval", {}), dict
                            )
                            else []
                        )
                        if isinstance(item, dict)
                    ],
                    "retrieval_grounding_required": attempt.metadata.get(
                        "retrieval_grounding_required"
                    ),
                    "cited_retrieved_declaration_names": attempt.metadata.get(
                        "cited_retrieved_declaration_names", []
                    ),
                    "used_retrieved_declaration_names": attempt.metadata.get(
                        "used_retrieved_declaration_names", []
                    ),
                    "declarations_grounded": attempt.metadata.get(
                        "declarations_grounded"
                    ),
                    "missing_required_index_evidence": attempt.metadata.get(
                        "missing_required_index_evidence"
                    ),
                }
                for attempt in node.proof_attempts
            ],
            "repair_audit": node.metadata.get("proof_repair_audit", []),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0 if output["success"] else 1
    finally:
        if runtime is not None:
            runtime.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
