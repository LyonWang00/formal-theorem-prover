from __future__ import annotations

import lean_prover.Prover as prover_facade

from lean_prover.Data import (
    LeanFailureDetail,
    LeanVerificationStatus,
    ProverDataStatus,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    LeanEnvironmentIdentity,
    LeanFeedback,
    LeanPreamble,
    NodeStatus,
    ProofLengthEstimate,
    SemanticAlignment,
    TheoremProblem,
)
from lean_prover.Prover import (
    BlueprintProver,
    GeneratedProof,
    JsonlProverResultStore,
    NodeProofVerifier,
    OpenAICompatibleProofGenerator,
    ProofGenerationError,
    build_node_proof_prompt,
    extract_lean_proof_body,
)
from lean_prover.Repair.prover_proof import ProverProofRepairer


class FakeCompletions:
    def __init__(self, content: str | None | list[str | None]) -> None:
        self.content = content
        self.kwargs = None
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        self.kwargs = kwargs
        content = self.content.pop(0) if isinstance(self.content, list) else self.content
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeApiClient:
    def __init__(
        self,
        content: str | None | list[str | None] = "by\n  trivial",
    ) -> None:
        self.completions = FakeCompletions(content)
        self.chat = type("Chat", (), {"completions": self.completions})()


class FakeVerifierRuntime:
    def __init__(
        self,
        imports: tuple[str, ...],
        *,
        success: bool = True,
        diagnostics: str = "",
    ) -> None:
        self.imports = imports
        self.success = success
        self.diagnostics = diagnostics
        self.sources: list[str] = []

    def check_source(self, source: str):
        self.sources.append(source)
        return type(
            "Result",
            (),
            {"success": self.success, "diagnostics": self.diagnostics},
        )()


class ConditionalVerifierRuntime:
    imports = ("Mathlib",)

    def check_source(self, source: str):
        success = "by\n  trivial" in source
        return type(
            "Result",
            (),
            {
                "success": success,
                "diagnostics": "" if success else "unsolved goals\n⊢ True",
            },
        )()


class ReferenceConditionalVerifierRuntime:
    """Accept trivial and classify every other proof as a bad declaration name."""

    imports = ("Mathlib",)

    def check_source(self, source: str):
        success = "by\n  trivial" in source
        return type(
            "Result",
            (),
            {
                "success": success,
                "diagnostics": (
                    "" if success else "unknown identifier `old_api`"
                ),
            },
        )()


def environment() -> LeanEnvironmentIdentity:
    return LeanEnvironmentIdentity(
        lean_version="Lean 4.29.1 commit " + "1" * 40,
        lean_commit="1" * 40,
        mathlib_commit="2" * 40,
        environment_hash="3" * 64,
    )


def problem() -> TheoremProblem:
    return TheoremProblem(
        problem_id="api_bridge",
        input_hash="4" * 64,
        imports=["Mathlib"],
        natural_language_statement="The locally named truth holds.",
        target_lean_decl="theorem target : True",
    )


def blueprint(problem_value: TheoremProblem) -> Blueprint:
    preamble = LeanPreamble(
        imports=["Mathlib", "Mathlib.Data.Real.Basic"],
        namespaces=["PlannerProverBridge"],
        open_namespaces=["Real"],
        variable_declarations=["variable (A : Type*)"],
        local_context=["local notation \"PlannerTruth\" => True"],
    )
    return Blueprint(
        blueprint_summary="Use the complete Planner preamble.",
        nodes=[
            BlueprintNode(
                id="L1",
                title="Local truth",
                informal_statement="The locally named truth holds.",
                informal_proof="Use the local definition of truth.",
                logical_ideas=["Unfold the local truth"],
                lean_statement="lemma L1 : PlannerTruth",
                preamble=preamble,
                semantic_alignment=SemanticAlignment(
                    objects=["the proposition True"],
                    hypotheses=[],
                    conclusion="True holds",
                    alignment_notes="PlannerTruth is local notation for True.",
                ),
                estimated_proof_length=ProofLengthEstimate(
                    estimated_lines=2,
                    estimated_tokens=12,
                    rationale="The proof is trivial.",
                ),
                proof_strategy="Use trivial.",
                status=NodeStatus.STATEMENT_VALID,
            )
        ],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem_value.problem_hash,
    )


def test_api_generator_and_prover_forward_complete_planner_contract() -> None:
    problem_value = problem()
    blueprint_value = blueprint(problem_value)
    client = FakeApiClient()
    runtime = FakeVerifierRuntime(("Mathlib",))
    prover = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=client,
            model="api-model",
        ),
        verifier=NodeProofVerifier(runtime),
    )

    request = prover.build_request(
        problem=problem_value,
        blueprint=blueprint_value,
        node=blueprint_value.nodes[0],
    )
    prompt = build_node_proof_prompt(request)
    result = prover.prove_blueprint(
        problem=problem_value,
        blueprint=blueprint_value,
    )

    assert request.problem_hash == problem_value.problem_hash
    assert request.imports == ("Mathlib", "Mathlib.Data.Real.Basic")
    assert request.estimated_proof_length.estimated_tokens == 12
    assert request.preamble.local_context
    assert environment().lean_commit in prompt
    assert "estimated proof-body budget" in prompt.lower()
    assert "local notation \"PlannerTruth\" => True" in prompt
    assert "local notation" in runtime.sources[0]
    assert "namespace PlannerProverBridge" in runtime.sources[0]
    assert result.nodes[0].status == NodeStatus.PROVED
    assert result.nodes[0].verified_proof == "trivial"
    root = result.metadata["root_node"]
    assert root["id"] == "ROOT"
    assert root["depends_on"] == ["L1"]
    assert len(root["proof_attempts"]) == 4
    root_prompt = client.completions.kwargs["messages"][1]["content"]
    assert "lemma L1 : PlannerTruth" in root_prompt
    assert "lemma L1 : PlannerTruth := by\n  trivial" in runtime.sources[-1]
    assert "theorem target : True := by\n  trivial" in runtime.sources[-1]


def test_api_empty_response_uses_existing_generation_failure_collection() -> None:
    problem_value = problem()
    client = FakeApiClient(None)
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=client,
            model="api-model",
        )
    ).prove_blueprint(
        problem=problem_value,
        blueprint=blueprint(problem_value),
    )

    node = result.nodes[0]
    assert node.status == NodeStatus.FAILED
    assert len(node.proof_attempts) == 4
    assert all(not attempt.proof for attempt in node.proof_attempts)
    assert all(
        attempt.error_type == "extraction_error"
        for attempt in node.proof_attempts
    )
    assert node.lean_feedback[-1].stage == "proof_generation"
    assert node.lean_feedback[-1].error_type == "extraction_error"
    assert node.lean_feedback[-1].error_detail == "proof_extraction_failed"
    assert client.completions.call_count == 20


def test_api_generator_retries_empty_response_until_success() -> None:
    problem_value = problem()
    blueprint_value = blueprint(problem_value)
    client = FakeApiClient([None, " ", None, "", "by\n  trivial"])
    generator = OpenAICompatibleProofGenerator(client=client, model="api-model")
    request = BlueprintProver(
        generator=generator,
        attempts_per_node=1,
    ).build_request(
        problem=problem_value,
        blueprint=blueprint_value,
        node=blueprint_value.nodes[0],
    )

    generated = generator.generate(request)

    assert generated.proof == "trivial"
    assert client.completions.call_count == 5
    assert client.completions.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert "Format-only example" in client.completions.kwargs["messages"][1][
        "content"
    ]


def test_api_generator_raises_after_five_empty_responses() -> None:
    problem_value = problem()
    blueprint_value = blueprint(problem_value)
    client = FakeApiClient([None, " ", "", None, "  "])
    generator = OpenAICompatibleProofGenerator(client=client, model="api-model")
    request = BlueprintProver(
        generator=generator,
        attempts_per_node=1,
    ).build_request(
        problem=problem_value,
        blueprint=blueprint_value,
        node=blueprint_value.nodes[0],
    )

    try:
        generator.generate(request)
    except ProofGenerationError as error:
        assert str(error) == "证明节点L1时模型空响应"
    else:
        raise AssertionError("five empty API responses must fail")
    assert client.completions.call_count == 5


def test_problem_result_routes_all_verified_to_success(tmp_path) -> None:
    problem_value = problem()
    store = JsonlProverResultStore(
        success_path=tmp_path / "success.jsonl",
        failure_path=tmp_path / "failure.jsonl",
    )
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient(),
            model="api-model",
        ),
        verifier=NodeProofVerifier(FakeVerifierRuntime(("Mathlib",))),
        result_store=store,
    ).prove(problem=problem_value, blueprint=blueprint(problem_value))

    assert result.prover_status == ProverDataStatus.SUCCESS
    assert result.all_subproblems_verified
    assert result.node_results[0].verification_status == LeanVerificationStatus.SUCCESS
    assert (tmp_path / "success.jsonl").is_file()
    assert not (tmp_path / "failure.jsonl").exists()


def test_problem_result_uses_training_taxonomy_and_failure_route(tmp_path) -> None:
    problem_value = problem()
    store = JsonlProverResultStore(
        success_path=tmp_path / "success.jsonl",
        failure_path=tmp_path / "failure.jsonl",
    )
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient(),
            model="api-model",
        ),
        verifier=NodeProofVerifier(
            FakeVerifierRuntime(
                ("Mathlib",),
                success=False,
                diagnostics="unknown identifier `missingLemma`",
            )
        ),
        result_store=store,
    ).prove(problem=problem_value, blueprint=blueprint(problem_value))

    node = result.node_results[0]
    assert result.prover_status == ProverDataStatus.FAILURE
    assert not result.all_subproblems_verified
    assert node.verification_status == LeanVerificationStatus.ELABORATION_ERROR
    assert node.failure_detail == LeanFailureDetail.UNKNOWN_IDENTIFIER
    assert result.failure_counts == {
        LeanVerificationStatus.ELABORATION_ERROR: 1,
        LeanVerificationStatus.INTERNAL_ERROR: 1,
    }
    assert result.root_result.status == NodeStatus.FAILED
    assert result.root_result.skipped_reason == "dependencies not proved: L1"
    assert result.root_result.proof_attempts == []
    assert result.attempt_failure_counts == {
        LeanVerificationStatus.ELABORATION_ERROR: 4
    }
    assert result.failure_detail_counts == {LeanFailureDetail.UNKNOWN_IDENTIFIER: 4}
    assert (tmp_path / "failure.jsonl").is_file()
    assert not (tmp_path / "success.jsonl").exists()


def test_unsolved_goal_is_repaired_after_reverification() -> None:
    problem_value = problem()
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient("by\n  skip"),
            model="api-model",
        ),
        verifier=NodeProofVerifier(ConditionalVerifierRuntime()),
        proof_repairer=ProverProofRepairer(
            client=FakeApiClient(
                '{"candidates":[{"candidate_id":"C1",'
                '"proof":"trivial","used_declarations":[],'
                '"rationale":"Direct tactic."}]}'
            ),
            model="repair-api-model",
        ),
        attempts_per_node=1,
    ).prove(problem=problem_value, blueprint=blueprint(problem_value))

    node = result.blueprint.nodes[0]
    assert result.prover_status == ProverDataStatus.SUCCESS
    assert len(node.proof_attempts) == 2
    assert node.proof_attempts[0].error_type == "unsolved_goals"
    assert node.proof_attempts[1].success
    assert node.verified_proof == "trivial"


def test_reference_repair_without_local_index_evidence_is_rejected() -> None:
    problem_value = problem()
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient("by\n  exact old_api"),
            model="api-model",
        ),
        verifier=NodeProofVerifier(ReferenceConditionalVerifierRuntime()),
        proof_repairer=ProverProofRepairer(
            client=FakeApiClient(
                '{"candidates":[{"candidate_id":"C1",'
                '"proof":"trivial","used_declarations":[],'
                '"rationale":"Replace the obsolete tactic reference."}]}'
            ),
            model="repair-api-model",
        ),
        attempts_per_node=1,
    ).prove(problem=problem_value, blueprint=blueprint(problem_value))

    node = result.blueprint.nodes[0]
    assert result.prover_status == ProverDataStatus.FAILURE
    assert len(node.proof_attempts) == 2
    assert node.proof_attempts[0].error_detail == "unknown_identifier"
    assert not node.proof_attempts[1].success
    assert node.proof_attempts[1].error_detail == "forbidden_declaration"
    assert "pinned local environment index" in node.proof_attempts[1].diagnostics
    assert node.verified_proof is None


def test_reference_repair_requires_retrieved_name_in_actual_proof() -> None:
    problem_value = problem()
    blueprint_value = blueprint(problem_value)
    prover = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient(), model="api-model"
        ),
        attempts_per_node=1,
    )
    request = prover.build_request(
        problem=problem_value,
        blueprint=blueprint_value,
        node=blueprint_value.nodes[0],
    )
    feedback = LeanFeedback(
        stage="proof_verification",
        success=False,
        message="unknown identifier `old_api`",
        diagnostics="unknown identifier `old_api`",
        error_type="elaboration_error",
        error_detail="unknown_identifier",
    )
    repairer = ProverProofRepairer(
        client=FakeApiClient(
            '{"candidates":[{"candidate_id":"C1",'
            '"proof":"trivial","used_declarations":["True.intro"],'
            '"rationale":"Claimed but not used."}]}'
        ),
        model="repair-api-model",
    )
    repairer._retrieval_payload = lambda **kwargs: {
        "compiler_error_analysis": {"missing_name": "old_api"},
        "candidate_declarations": [
            {"full_name": "True.intro", "type_signature": "True"}
        ],
        "index_available": True,
    }

    candidate = repairer.repair(
        request=request,
        failed_proof="exact old_api",
        feedback=feedback,
    )

    assert candidate.metadata["cited_retrieved_declaration_names"] == [
        "True.intro"
    ]
    assert candidate.metadata["used_retrieved_declaration_names"] == []
    assert candidate.metadata["declarations_grounded"] is False
    assert candidate.metadata["missing_required_index_evidence"] is True


def test_reference_repair_accepts_exact_retrieved_name_used_in_proof() -> None:
    problem_value = problem()
    blueprint_value = blueprint(problem_value)
    request = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient(), model="api-model"
        ),
        attempts_per_node=1,
    ).build_request(
        problem=problem_value,
        blueprint=blueprint_value,
        node=blueprint_value.nodes[0],
    )
    feedback = LeanFeedback(
        stage="proof_verification",
        success=False,
        message="unknown identifier `old_api`",
        diagnostics="unknown identifier `old_api`",
        error_type="elaboration_error",
        error_detail="unknown_identifier",
    )
    repairer = ProverProofRepairer(
        client=FakeApiClient(
            '{"candidates":[{"candidate_id":"C1",'
            '"proof":"exact True.intro","used_declarations":["True.intro"],'
            '"rationale":"Use the retrieved constructor."}]}'
        ),
        model="repair-api-model",
    )
    repairer._retrieval_payload = lambda **kwargs: {
        "compiler_error_analysis": {"missing_name": "old_api"},
        "candidate_declarations": [
            {"full_name": "True.intro", "type_signature": "True"}
        ],
        "index_available": True,
    }

    candidate = repairer.repair(
        request=request,
        failed_proof="exact old_api",
        feedback=feedback,
    )

    assert candidate.metadata["used_retrieved_declaration_names"] == [
        "True.intro"
    ]
    assert candidate.metadata["declarations_grounded"] is True


def test_prover_repair_retries_empty_response_until_success() -> None:
    problem_value = problem()
    blueprint_value = blueprint(problem_value)
    client = FakeApiClient(
        [
            None,
            " ",
            None,
            "",
            '{"candidates":[{"candidate_id":"C1",'
            '"proof":"trivial","used_declarations":[],'
            '"rationale":"Direct tactic."}]}',
        ]
    )
    prover = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient(),
            model="api-model",
        ),
        attempts_per_node=1,
    )
    request = prover.build_request(
        problem=problem_value,
        blueprint=blueprint_value,
        node=blueprint_value.nodes[0],
    )
    feedback = LeanFeedback(
        stage="proof_verification",
        success=False,
        message="unsolved goals",
        error_type="unsolved_goals",
        error_detail="unsolved_goals",
    )

    generated = ProverProofRepairer(
        client=client,
        model="repair-api-model",
    ).repair(
        request=request,
        failed_proof="by\n  skip",
        feedback=feedback,
    )

    assert generated.proof == "trivial"
    assert client.completions.call_count == 5
    assert client.completions.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert client.completions.kwargs["response_format"] == {
        "type": "json_object"
    }


class TwoRoundCandidateRepairer:
    def __init__(self) -> None:
        self.calls = 0

    def generate_candidates(
        self,
        *,
        request,
        failed_proof,
        feedback,
        round_index,
        repair_history,
    ):
        self.calls += 1
        assert round_index in {1, 2}
        if round_index == 2:
            assert repair_history[0]["round"] == 1
            assert len(repair_history[0]["candidate_outcomes"]) == 1
            proofs = ["trivial"]
        else:
            proofs = ["exact missingIdentifier"]
        return [
            GeneratedProof(
                proof=proof,
                model="candidate-repair",
                prompt_type="api_proof_repair_retrieval",
                metadata={
                    "repair_round": round_index,
                    "candidate_id": f"C{index}",
                },
            )
            for index, proof in enumerate(proofs, start=1)
        ]


class UngroundedCandidateRepairer:
    def generate_candidates(
        self,
        *,
        request,
        failed_proof,
        feedback,
        round_index,
        repair_history,
    ):
        return [
            GeneratedProof(
                proof="trivial",
                model="candidate-repair",
                prompt_type="api_proof_repair_retrieval",
                metadata={
                    "repair_round": round_index,
                    "candidate_id": "C1",
                    "declarations_grounded": False,
                    "ungrounded_declared_names": ["Guessed.oldLemma"],
                },
            )
        ]


def test_pantograph_success_does_not_bypass_repair_grounding_gate() -> None:
    problem_value = problem()
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient("skip"),
            model="api-model",
        ),
        verifier=NodeProofVerifier(ReferenceConditionalVerifierRuntime()),
        proof_repairer=UngroundedCandidateRepairer(),
        attempts_per_node=1,
        max_proof_repair_rounds=1,
    ).prove_node(
        problem=problem_value,
        blueprint=blueprint(problem_value),
        node_id="L1",
    )

    node = result.nodes[0]
    assert node.status == NodeStatus.FAILED
    assert len(node.proof_attempts) == 2
    assert node.proof_attempts[-1].success is False
    assert node.proof_attempts[-1].error_detail == "forbidden_declaration"
    assert "Guessed.oldLemma" in node.proof_attempts[-1].diagnostics


def test_retrieval_repair_compiles_all_candidates_and_succeeds_in_round_two() -> None:
    problem_value = problem()
    prover = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient("skip"),
            model="api-model",
        ),
        verifier=NodeProofVerifier(ReferenceConditionalVerifierRuntime()),
        proof_repairer=TwoRoundCandidateRepairer(),
        attempts_per_node=1,
        max_proof_repair_rounds=5,
    )

    result = prover.prove_node(
        problem=problem_value,
        blueprint=blueprint(problem_value),
        node_id="L1",
    )

    node = result.nodes[0]
    assert node.status == NodeStatus.PROVED
    assert len(node.proof_attempts) == 3
    assert [attempt.success for attempt in node.proof_attempts] == [
        False,
        False,
        True,
    ]
    assert [attempt.metadata.get("repair_round") for attempt in node.proof_attempts] == [
        None,
        1,
        2,
    ]
    assert all(attempt.attempt_id for attempt in node.proof_attempts)
    assert node.metadata["proof_repair_audit"][-1]["success"] is True


def test_pass_at_four_uses_one_shared_multi_round_repair_budget() -> None:
    problem_value = problem()
    repairer = TwoRoundCandidateRepairer()
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient("skip"),
            model="api-model",
        ),
        verifier=NodeProofVerifier(ReferenceConditionalVerifierRuntime()),
        proof_repairer=repairer,
        attempts_per_node=4,
        max_proof_repair_rounds=5,
    ).prove_node(
        problem=problem_value,
        blueprint=blueprint(problem_value),
        node_id="L1",
    )

    node = result.nodes[0]
    assert repairer.calls == 2
    assert len(node.proof_attempts) == 6
    assert len(
        [
            attempt
            for attempt in node.proof_attempts
            if attempt.prompt_type == "api_proof_generation"
        ]
    ) == 4
    assert node.proof_attempts[-1].success


def test_prover_repair_raises_node_specific_error_after_five_empty_responses() -> None:
    problem_value = problem()
    blueprint_value = blueprint(problem_value)
    client = FakeApiClient([None, " ", "", None, "  "])
    prover = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient(),
            model="api-model",
        ),
        attempts_per_node=1,
    )
    request = prover.build_request(
        problem=problem_value,
        blueprint=blueprint_value,
        node=blueprint_value.nodes[0],
    )
    feedback = LeanFeedback(
        stage="proof_verification",
        success=False,
        message="unsolved goals",
        error_type="unsolved_goals",
        error_detail="unsolved_goals",
    )

    try:
        ProverProofRepairer(
            client=client,
            model="repair-api-model",
        ).repair(
            request=request,
            failed_proof="by\n  skip",
            feedback=feedback,
        )
    except ProofGenerationError as error:
        assert str(error) == "修复证明节点L1时模型空响应"
    else:
        raise AssertionError("five empty repair responses must fail")
    assert client.completions.call_count == 5


def test_default_pass_at_four_keeps_all_attempts_and_accepts_any_success() -> None:
    problem_value = problem()
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient(
                    [
                        "by\n  skip",
                        "by\n  trivial",
                        "by\n  skip",
                        "by\n  skip",
                        "by\n  trivial",
                        "by\n  trivial",
                        "by\n  trivial",
                        "by\n  trivial",
                    ]
            ),
            model="api-model",
        ),
        verifier=NodeProofVerifier(ConditionalVerifierRuntime()),
    ).prove(problem=problem_value, blueprint=blueprint(problem_value))

    node = result.blueprint.nodes[0]
    assert result.prover_status == ProverDataStatus.SUCCESS
    assert node.status == NodeStatus.PROVED
    assert len(node.proof_attempts) == 4
    assert [attempt.success for attempt in node.proof_attempts] == [
        False,
        True,
        False,
        False,
    ]
    assert node.metadata["attempt_budget"] == 4
    assert node.metadata["attempts_executed"] == 4
    assert node.metadata["successful_attempt_ids"] == [
        node.proof_attempts[1].attempt_id
    ]
    assert result.root_result.status == NodeStatus.PROVED
    assert result.root_result.dependencies == ["L1"]
    assert len(result.root_result.proof_attempts) == 4


def test_zero_subproblem_blueprint_still_proves_root_with_pass_at_four() -> None:
    problem_value = problem()
    empty_blueprint = Blueprint(
        blueprint_summary="The target is direct.",
        nodes=[],
        root_dependencies=[],
        environment=environment(),
        problem_hash=problem_value.problem_hash,
    )
    client = FakeApiClient()
    result = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=client,
            model="api-model",
        ),
        verifier=NodeProofVerifier(FakeVerifierRuntime(("Mathlib",))),
    ).prove(problem=problem_value, blueprint=empty_blueprint)

    assert result.prover_status == ProverDataStatus.SUCCESS
    assert result.all_subproblems_verified
    assert result.node_results == []
    assert result.root_result.node_id == "ROOT"
    assert result.root_result.dependencies == []
    assert result.root_result.status == NodeStatus.PROVED
    assert len(result.root_result.proof_attempts) == 4
    assert all(attempt.success for attempt in result.root_result.proof_attempts)
    assert client.completions.call_count == 4


def test_prover_facade_contains_no_reserved_7b_design() -> None:
    forbidden = (
        "Local7BProofGenerator",
        "SevenBCapabilityDecomposer",
        "SevenBCapabilityEvaluator",
        "SevenBProverProfile",
        "build_7b_decomposition_prompt",
    )
    assert all(not hasattr(prover_facade, name) for name in forbidden)


def test_extracts_direct_and_fenced_api_proofs() -> None:
    assert extract_lean_proof_body("by\n  trivial") == "trivial"
    assert extract_lean_proof_body(
        "```lean4\nlemma L1 : True := by\n  trivial\n```"
    ) == "trivial"

    try:
        extract_lean_proof_body("")
    except ProofGenerationError:
        pass
    else:
        raise AssertionError("empty API output must fail")


def test_verifier_materializes_header_then_dependencies_then_current_target() -> None:
    header = "import Mathlib\n\nopen Nat"
    problem_value = TheoremProblem(
        problem_id="header-order",
        imports=["Mathlib"],
        natural_language_statement="Truth follows from truth.",
        target_lean_decl="theorem target : True",
        header=header,
    )
    preamble = LeanPreamble(imports=["Mathlib"], raw_header=header)
    first = BlueprintNode(
        id="L1",
        title="First",
        informal_statement="Truth.",
        informal_proof="Use the constructor of True.",
        logical_ideas=["Construct True"],
        lean_statement="lemma L1 : True",
        preamble=preamble,
        semantic_alignment=SemanticAlignment(
            objects=["True"],
            hypotheses=[],
            conclusion="True",
            alignment_notes="Exact.",
        ),
        estimated_proof_length=ProofLengthEstimate(
            estimated_lines=1,
            estimated_tokens=2,
            rationale="Direct.",
        ),
        proof_strategy="trivial",
        status=NodeStatus.PROVED,
        verified_proof="trivial",
    )
    second = first.model_copy(deep=True)
    second.id = "L2"
    second.lean_decl = "lemma L2 : True"
    second.depends_on = ["L1"]
    second.status = NodeStatus.STATEMENT_VALID
    second.verified_proof = None
    blueprint_value = Blueprint(
        blueprint_summary="chain",
        nodes=[first, second],
        root_dependencies=["L2"],
        environment=environment(),
        problem_hash=problem_value.problem_hash,
    )
    runtime = FakeVerifierRuntime(("Mathlib",))
    request = BlueprintProver(
        generator=OpenAICompatibleProofGenerator(
            client=FakeApiClient("trivial"),
            model="api-model",
        ),
        verifier=NodeProofVerifier(runtime),
    ).build_request(
        problem=problem_value,
        blueprint=blueprint_value,
        node=second,
    )

    feedback = NodeProofVerifier(runtime).verify(
        request=request,
        proof="trivial",
    )

    assert feedback.success
    source = runtime.sources[-1]
    assert "import Mathlib" not in source
    assert source.count("open Nat") == 1
    assert source.index("open Nat") < source.index("lemma L1")
    assert source.index("lemma L1") < source.index("lemma L2")
    assert "lemma L1 : True := by\n  trivial" in source
    assert "lemma L2 : True := by\n  trivial" in source
