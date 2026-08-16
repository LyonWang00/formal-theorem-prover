from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lean_prover.Data import ProverDataStatus
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    LeanEnvironmentIdentity,
    LeanPreamble,
    ProofLengthEstimate,
    SemanticAlignment,
    TheoremProblem,
)
from lean_prover.Prover import BlueprintProver, GeneratedProof, NodeProofVerifier
from lean_prover.lean_training.verification.pantograph import (
    PantographTheoremVerifier,
)


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="real Pantograph integration runs in the WSL Lean project",
)


class BrokenGenerator:
    def generate(self, request):
        return GeneratedProof(proof="by exact missingIdentifier", model="stub-api")


class ValidRepairer:
    def repair(self, *, request, failed_proof, feedback):
        assert feedback.error_type == "elaboration_error"
        return GeneratedProof(
            proof="by trivial",
            model="stub-repair-api",
            prompt_type="api_proof_repair",
        )


def test_repaired_proof_is_promoted_only_after_real_pantograph() -> None:
    pytest.importorskip("pantograph")
    project = Path.cwd() / "lean_project"
    if not (project / "lean-toolchain").is_file():
        pytest.skip("Lean project toolchain is unavailable")

    problem = TheoremProblem(
        problem_id="real-repair",
        input_hash="4" * 64,
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="One node.",
        nodes=[
            BlueprintNode(
                id="L1",
                title="Truth",
                informal_statement="Truth holds.",
                informal_proof="Use the constructor of True.",
                logical_ideas=["Construct True"],
                lean_statement="lemma L1 : True",
                preamble=LeanPreamble(imports=["Mathlib"]),
                semantic_alignment=SemanticAlignment(
                    objects=["True"],
                    conclusion="True",
                    alignment_notes="Exact correspondence.",
                ),
                estimated_proof_length=ProofLengthEstimate(
                    estimated_lines=1,
                    estimated_tokens=4,
                    rationale="Direct proof.",
                ),
            )
        ],
        root_dependencies=["L1"],
        environment=LeanEnvironmentIdentity(
            lean_version="Lean integration test",
            lean_commit="1" * 40,
            mathlib_commit="2" * 40,
            environment_hash="3" * 64,
        ),
        problem_hash=problem.problem_hash,
    )
    runtime = PantographTheoremVerifier(project, imports=("Mathlib",))
    try:
        result = BlueprintProver(
            generator=BrokenGenerator(),
            verifier=NodeProofVerifier(runtime),
            proof_repairer=ValidRepairer(),
            attempts_per_node=1,
        ).prove(problem=problem, blueprint=blueprint)
    finally:
        runtime.close()

    node = result.blueprint.nodes[0]
    assert result.prover_status == ProverDataStatus.SUCCESS
    assert len(node.proof_attempts) == 2
    assert not node.proof_attempts[0].success
    assert node.proof_attempts[1].success
    assert node.verified_proof == "by trivial"
