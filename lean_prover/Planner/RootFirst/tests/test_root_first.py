from __future__ import annotations

from typing import Any

import pytest

from lean_prover.Planner.RootFirst.adapters import AcceptingStatementChecker
from lean_prover.Planner.RootFirst.graph import GraphValidationError, validate_graph
from lean_prover.Planner.RootFirst.proving import RootFirstProver, negated_declaration
from lean_prover.Planner.RootFirst.refinement import BlueprintRefinement
from lean_prover.Planner.RootFirst.run_minif2f import result_record, summarize
from lean_prover.Planner.RootFirst.schemas import NodeState
from lean_prover.Planner.RootFirst.service import RootFirstService
from lean_prover.Planner.schemas import LeanFeedback, TheoremProblem
from lean_prover.Planner.tests.helpers import environment
from lean_prover.Prover.service import GeneratedProof


def ok() -> LeanFeedback:
    return LeanFeedback(
        stage="proof_verification",
        success=True,
        message="accepted",
    )


def no(
    message: str = "unsolved goals",
    *,
    detail: str = "unsolved_goals",
) -> LeanFeedback:
    return LeanFeedback(
        stage="proof_verification",
        success=False,
        message=message,
        diagnostics=message,
        error_type="lean_compilation_error",
        error_detail=detail,
    )


class QueueGenerator:
    def __init__(self, proofs: list[str]) -> None:
        self.proofs = list(proofs)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return GeneratedProof(
            proof=self.proofs.pop(0),
            model="fake",
        )


class MappingVerifier:
    def __init__(
        self,
        accepted: set[str],
        *,
        failure_detail: str = "unsolved_goals",
    ) -> None:
        self.accepted = accepted
        self.failure_detail = failure_detail
        self.calls = []

    def verify(self, *, request, proof):
        self.calls.append((request, proof))
        return (
            ok()
            if proof in self.accepted
            else no(detail=self.failure_detail)
        )


class EchoRepairer:
    def __init__(self, proof: str) -> None:
        self.proof = proof
        self.calls = 0

    def generate_candidates(self, **kwargs):
        self.calls += 1
        return [
            GeneratedProof(
                proof=self.proof,
                model="fake-repair",
                metadata={"candidate_id": f"C{self.calls}"},
            )
        ]


class QueueClient:
    def __init__(self, outputs: list[dict[str, Any] | str]) -> None:
        self.outputs = list(outputs)
        self.calls = []

    def generate_text_or_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def problem() -> TheoremProblem:
    return TheoremProblem(
        problem_id="demo",
        natural_language_statement="Truth holds.",
        target_lean_decl="theorem demo : True",
    )


def add_patch(
    *,
    new_id: str,
    target: str,
    fathers: list[str],
    statement: str | None = None,
) -> dict[str, Any]:
    numeral = max(0, int(new_id[1:]) - 1)
    return {
        "state": "failed",
        "action": "add_node",
        "target_node_id": target,
        "new_node": {
            "id": new_id,
            "informal_statement": f"Statement for {new_id}",
            "lean_statement": (
                statement or f"lemma {new_id} : {numeral} = {numeral}"
            ),
            "father_nodes": fathers,
            "children": [target],
        },
        "revised_node": None,
    }


def service(generator, verifier, client, *, attempt_disproof=False):
    return RootFirstService(
        prover=RootFirstProver(
            generator=generator,
            verifier=verifier,
            repairer=None,
            attempt_disproof=attempt_disproof,
        ),
        refinement=BlueprintRefinement(
            client=client,
            statement_checker=AcceptingStatementChecker(),
        ),
        environment=environment(),
    )


def test_direct_root_success_stops_before_refinement() -> None:
    client = QueueClient([])
    runner = service(
        QueueGenerator(["trivial"]),
        MappingVerifier({"trivial"}),
        client,
    )
    result = runner.run_problem(problem())
    assert result.success is True
    assert result.stage == "root_direct_success"
    assert result.blueprint.node_map()["L0"].state == NodeState.SUCCESS
    assert result.blueprint.node_map()["L0"].metadata[
        "attempt_count_at_freeze"
    ] == 1
    assert client.calls == []


def test_complete_pass_batch_executes_all_four_before_freeze() -> None:
    generator = QueueGenerator(["trivial"] * 4)
    prover = RootFirstProver(
        generator=generator,
        verifier=MappingVerifier({"trivial"}),
        attempt_disproof=False,
        complete_pass_batch=True,
    )
    runner = RootFirstService(
        prover=prover,
        refinement=BlueprintRefinement(
            client=QueueClient([]),
            statement_checker=AcceptingStatementChecker(),
        ),
        environment=environment(),
    )
    result = runner.run_problem(problem())
    root = result.blueprint.node_map()["L0"]
    assert result.success is True
    assert len(generator.requests) == 4
    assert root.metadata["attempt_count_at_freeze"] == 4


def test_pass_batch_gets_one_shared_reference_repair() -> None:
    repairer = EchoRepairer("trivial")
    prover = RootFirstProver(
        generator=QueueGenerator(["old_api"] * 4),
        verifier=MappingVerifier(
            {"trivial"}, failure_detail="unknown_identifier"
        ),
        repairer=repairer,
        attempt_disproof=False,
    )
    runner = RootFirstService(
        prover=prover,
        refinement=BlueprintRefinement(
            client=QueueClient([]),
            statement_checker=AcceptingStatementChecker(),
        ),
        environment=environment(),
    )
    result = runner.run_problem(problem())
    root = result.blueprint.node_map()["L0"]
    assert result.success is True
    assert [attempt.kind for attempt in root.attempts] == [
        "proof",
        "proof",
        "proof",
        "proof",
        "proof_repair",
    ]
    assert repairer.calls == 1
    history = root.metadata["proof_repair_history"]
    assert len(history) == 1
    assert history[0]["round"] == 1
    assert history[0]["candidate_outcomes"][0]["success"] is True


def test_failed_root_adds_one_node_then_proves_helper_and_root() -> None:
    # Four direct L0 failures, L1 succeeds, then L0 succeeds with L1.
    generator = QueueGenerator(["bad"] * 4 + ["l1_ok", "root_ok"])
    verifier = MappingVerifier({"l1_ok", "root_ok"})
    client = QueueClient([add_patch(new_id="L1", target="L0", fathers=[])])
    result = service(generator, verifier, client).run_problem(problem())
    assert result.success is True
    assert result.stage == "refined_success"
    nodes = result.blueprint.node_map()
    assert nodes["L1"].children == ["L0"]
    assert nodes["L0"].father_nodes == ["L1"]
    assert nodes["L1"].state == NodeState.SUCCESS
    assert nodes["L1"].frozen is True
    assert sum(request.node_id == "L1" for request in generator.requests) == 1
    validate_graph(result.blueprint)


def test_proved_l1_is_frozen_and_l2_is_inserted_between_l1_and_l0() -> None:
    # Initial L0 fail; L1 succeeds; L0 fail; L2 succeeds; L0 succeeds.
    generator = QueueGenerator(
        ["bad"] * 4
        + ["l1_ok"]
        + ["bad"] * 4
        + ["l2_ok"]
        + ["root_ok"]
    )
    verifier = MappingVerifier({"l1_ok", "l2_ok", "root_ok"})
    client = QueueClient(
        [
            add_patch(new_id="L1", target="L0", fathers=[]),
            add_patch(new_id="L2", target="L0", fathers=["L1"]),
        ]
    )
    result = service(generator, verifier, client).run_problem(problem())
    nodes = result.blueprint.node_map()
    assert result.success is True
    assert nodes["L1"].children == ["L2"]
    assert nodes["L2"].father_nodes == ["L1"]
    assert nodes["L2"].children == ["L0"]
    assert nodes["L0"].father_nodes == ["L2"]
    assert nodes["L1"].metadata["frozen_fingerprint"] == nodes[
        "L1"
    ].frozen_fingerprint()
    assert nodes["L1"].metadata["attempt_count_at_freeze"] == len(
        nodes["L1"].attempts
    )
    assert {attempt.proof_invocation for attempt in nodes["L1"].attempts} == {1}
    assert sum(request.node_id == "L1" for request in generator.requests) == 1
    validate_graph(result.blueprint)


def test_success_node_guard_never_generates_another_attempt() -> None:
    generator = QueueGenerator(["trivial"])
    verifier = MappingVerifier({"trivial"})
    runner = service(generator, verifier, QueueClient([]))
    result = runner.run_problem(problem())
    before = result.blueprint.node_map()["L0"].model_dump(mode="json")
    after_blueprint = runner.prover.prove_node(
        problem=problem(),
        blueprint=result.blueprint,
        node_id="L0",
    )
    after = after_blueprint.node_map()["L0"].model_dump(mode="json")
    assert before == after
    assert len(generator.requests) == 1
    assert len(verifier.calls) == 1


def test_result_statistics_include_freeze_and_repair_metrics() -> None:
    repairer = EchoRepairer("trivial")
    runner = RootFirstService(
        prover=RootFirstProver(
            generator=QueueGenerator(["old_api"]),
            verifier=MappingVerifier(
                {"trivial"}, failure_detail="unknown_identifier"
            ),
            repairer=repairer,
            attempt_disproof=False,
        ),
        refinement=BlueprintRefinement(
            client=QueueClient([]),
            statement_checker=AcceptingStatementChecker(),
        ),
        environment=environment(),
    )
    record = result_record(runner.run_problem(problem()), 1.0)
    summary = summarize([record])
    assert record["proof_repair_attempt_count"] == 1
    assert record["proof_repair_round_count"] == 1
    assert record["frozen_reprove_violation_count"] == 0
    assert record["node_attempt_statistics"][0][
        "repair_candidates_before_success"
    ] == 1
    assert summary["average_repair_round_count_per_proof_invocation"] == 1
    assert record["result"]["blueprint"]["nodes"][0]["attempts"][0][
        "proof_invocation"
    ] == 1


def test_graph_error_is_returned_to_refiner_and_second_patch_is_used() -> None:
    invalid = add_patch(new_id="L1", target="L0", fathers=[])
    invalid["new_node"]["children"] = ["L9"]
    client = QueueClient(
        [invalid, add_patch(new_id="L1", target="L0", fathers=[])]
    )
    generator = QueueGenerator(["bad"] * 4 + ["l1_ok", "root_ok"])
    result = service(
        generator,
        MappingVerifier({"l1_ok", "root_ok"}),
        client,
    ).run_problem(problem())
    assert result.success is True
    assert len(client.calls) == 2
    assert "children must be exactly" in client.calls[1]["user_prompt"]


def test_trivial_hypothesis_helper_is_rejected_and_regenerated() -> None:
    tautology = add_patch(
        new_id="L1",
        target="L0",
        fathers=[],
        statement="lemma L1 (p : Prop) (hp : p) : p",
    )
    client = QueueClient(
        [tautology, add_patch(new_id="L1", target="L0", fathers=[])]
    )
    generator = QueueGenerator(["bad"] * 4 + ["l1_ok", "root_ok"])
    result = service(
        generator,
        MappingVerifier({"l1_ok", "root_ok"}),
        client,
    ).run_problem(problem())

    assert result.success is True
    assert len(client.calls) == 2
    assert "conclusion exactly repeats" in client.calls[1]["user_prompt"]


def test_successor_probe_records_explicit_helper_use() -> None:
    client = QueueClient([add_patch(new_id="L1", target="L0", fathers=[])])
    generator = QueueGenerator(["bad"] * 4 + ["l1_ok", "exact L1"])
    result = service(
        generator,
        MappingVerifier({"l1_ok", "exact L1"}),
        client,
    ).run_problem(problem())

    record = result.blueprint.metadata["refinement_effectiveness"][0]
    assert record["status"] == "child_proved_using_helper"
    assert record["effective_refinement"] is True
    assert record["successful_child_attempts_referencing_helper"]


def test_non_json_refinement_is_audited_and_retried_once() -> None:
    client = QueueClient(
        ["not-json", add_patch(new_id="L1", target="L0", fathers=[])]
    )
    generator = QueueGenerator(["bad"] * 4 + ["l1_ok", "root_ok"])
    result = service(
        generator,
        MappingVerifier({"l1_ok", "root_ok"}),
        client,
    ).run_problem(problem())
    assert result.success is True
    assert len(client.calls) == 2
    assert result.refinements[0].raw_output == "not-json"
    assert "not one valid JSON" in client.calls[1]["user_prompt"]


def test_disproof_declaration_negates_full_pi_proposition() -> None:
    runner = service(QueueGenerator([]), MappingVerifier(set()), QueueClient([]))
    blueprint = runner.initial_blueprint(
        TheoremProblem(
            problem_id="p",
            target_lean_decl="theorem p (n : Nat) (h : n = 0) : n ≤ 1",
        )
    )
    assert negated_declaration(blueprint.node_map()["L0"]) == (
        "theorem L0_formal_disproof : ¬ (∀ (n : Nat) (h : n = 0), n ≤ 1)"
    )


def test_program_rejects_cycle_and_non_root_sink() -> None:
    runner = service(QueueGenerator([]), MappingVerifier(set()), QueueClient([]))
    blueprint = runner.initial_blueprint(problem())
    root = blueprint.node_map()["L0"]
    root.children = ["L0"]
    with pytest.raises(GraphValidationError):
        validate_graph(blueprint)


def test_disproved_nonroot_revision_adds_no_node() -> None:
    runner = service(QueueGenerator([]), MappingVerifier(set()), QueueClient([]))
    blueprint = runner.initial_blueprint(problem())
    refiner = BlueprintRefinement(
        client=QueueClient([add_patch(new_id="L1", target="L0", fathers=[])]),
        statement_checker=AcceptingStatementChecker(),
    )
    blueprint, _ = refiner.refine(
        problem=problem(), blueprint=blueprint.model_copy(update={"nodes": [blueprint.nodes[0].model_copy(update={"state": NodeState.FAILED})]}), target_id="L0", round_index=1
    )
    node = blueprint.node_map()["L1"]
    node.state = NodeState.DISPROVED
    node.formal_disproof = "exact formal counterexample"
    client = QueueClient([
        {
            "state": "disproved",
            "action": "revise_disproved_node",
            "target_node_id": "L1",
            "new_node": None,
            "revised_node": {
                "informal_statement": "True holds.",
                "lean_statement": "lemma L1 : True",
            },
        }
    ])
    revised, _ = BlueprintRefinement(
        client=client,
        statement_checker=AcceptingStatementChecker(),
    ).refine(problem=problem(), blueprint=blueprint, target_id="L1", round_index=2)
    assert len(revised.nodes) == 2
    assert revised.nodes_added == 1
    assert revised.node_map()["L1"].state == NodeState.PENDING
