from lean_prover.pipeline import TheoremProvingPipeline
from lean_prover.Data import PlannerDataStatus, ProverDataStatus
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    LeanEnvironmentIdentity,
    LeanPreamble,
    NodeStatus,
    PlannerResult,
    ProofLengthEstimate,
    RawTheoremInput,
    SemanticAlignment,
    TheoremProblem,
)
from lean_prover.Prover import (
    BlueprintProver,
    OpenAICompatibleProofGenerator,
)
from lean_prover.Prover.tests.test_prover import FakeApiClient


class StaticPlanner:
    def __init__(self, result: PlannerResult) -> None:
        self.result = result

    def plan_input(self, raw_input: RawTheoremInput) -> PlannerResult:
        assert raw_input.input_hash == self.result.input_hash
        return self.result


def test_pipeline_preserves_hashes_and_collects_unverified_api_attempt() -> None:
    raw = RawTheoremInput(input_text="True holds.")
    problem = TheoremProblem(
        problem_id="pipeline",
        input_hash=raw.input_hash,
        natural_language_statement="True holds.",
        target_lean_decl="theorem target : True",
    )
    environment = LeanEnvironmentIdentity(
        lean_version="Lean 4.29.1 commit " + "1" * 40,
        lean_commit="1" * 40,
        mathlib_commit="2" * 40,
        environment_hash="3" * 64,
    )
    node = BlueprintNode(
        id="L1",
        title="Truth",
        informal_statement="True holds.",
        informal_proof="Use the constructor of True.",
        logical_ideas=["Construct True"],
        lean_statement="lemma L1 : True",
        preamble=LeanPreamble(imports=["Mathlib"]),
        semantic_alignment=SemanticAlignment(
            objects=["True"],
            hypotheses=[],
            conclusion="True",
            alignment_notes="Exact.",
        ),
        estimated_proof_length=ProofLengthEstimate(
            estimated_lines=2,
            estimated_tokens=12,
            rationale="Trivial.",
        ),
        proof_strategy="Use trivial.",
        status=NodeStatus.STATEMENT_VALID,
    )
    blueprint = Blueprint(
        blueprint_summary="Direct.",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment,
        problem_hash=problem.problem_hash,
    )
    planned = PlannerResult(
        success=True,
        stage="completed",
        problem=problem,
        input_hash=raw.input_hash,
        blueprint=blueprint,
        environment=environment,
    )
    pipeline = TheoremProvingPipeline(
        planner=StaticPlanner(planned),
        prover=BlueprintProver(
            generator=OpenAICompatibleProofGenerator(
                client=FakeApiClient(),
                model="api-model",
            )
        ),
    )

    result = pipeline.run(raw)

    assert result.success is False
    assert result.stage == "prover_failure"
    assert result.planner_status == PlannerDataStatus.SUCCESS
    assert result.prover_status == ProverDataStatus.FAILURE
    assert result.prover_result is not None
    assert result.input_hash == raw.input_hash
    assert result.problem_hash == problem.problem_hash
    assert result.node_outcomes[0].status == NodeStatus.PROVING
    assert result.node_outcomes[0].attempt_count == 4
