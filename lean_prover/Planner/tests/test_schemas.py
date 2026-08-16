import pytest
from pydantic import ValidationError

from lean_prover.Planner.schemas import (
    BlueprintNode,
    BlueprintPlanNode,
    RawTheoremInput,
    TheoremProblem,
)
from lean_prover.Planner.tests.helpers import alignment, preamble, proof_length


def test_rejects_sorry() -> None:
    with pytest.raises(ValidationError):
        BlueprintNode(
            id="L1",
            title="Bad node",
            informal_statement="...",
            informal_proof="Apply the direct inference.",
            logical_ideas=["Apply the inference"],
            lean_decl="lemma L1 : True := by sorry",
            preamble=preamble(),
            semantic_alignment=alignment(),
            estimated_proof_length=proof_length(),
            depends_on=[],
            proof_strategy="",
            difficulty=1,
        )


def test_decomposition_node_requires_empty_lean_statement() -> None:
    with pytest.raises(ValidationError, match="must be the empty string"):
        BlueprintPlanNode(
            id="L1",
            title="Natural node",
            informal_statement="Every natural number equals itself.",
            informal_proof="Use reflexivity.",
            logical_ideas=["Apply reflexivity"],
            lean_statement="lemma L1 (n : Nat) : n = n",
            proof_strategy="Use reflexivity.",
            estimated_proof_length=proof_length(),
        )


def test_full_node_serializes_lean_statement_alias() -> None:
    node = BlueprintNode(
        id="L1",
        title="Natural node",
        informal_statement="True holds.",
        informal_proof="Use the constructor of True.",
        logical_ideas=["Construct True"],
        lean_statement="lemma L1 : True",
        preamble=preamble(),
        semantic_alignment=alignment(),
        estimated_proof_length=proof_length(),
        proof_strategy="Use True.intro.",
    )

    assert node.lean_decl == "lemma L1 : True"
    assert node.model_dump(by_alias=True)["lean_statement"] == "lemma L1 : True"


def test_node_rejects_more_than_three_logical_ideas() -> None:
    with pytest.raises(ValidationError):
        BlueprintNode(
            id="L1",
            title="Overloaded node",
            informal_statement="True holds.",
            informal_proof="Apply a direct construction.",
            logical_ideas=["A", "B", "C", "D"],
            lean_statement="lemma L1 : True",
            preamble=preamble(),
            semantic_alignment=alignment(),
            estimated_proof_length=proof_length(),
            proof_strategy="Construct True.",
        )


def test_input_and_problem_hashes_are_stable_under_whitespace() -> None:
    first = RawTheoremInput(
        input_text="Every   natural number equals itself.",
        imports=["Mathlib", "Mathlib"],
    )
    second = RawTheoremInput(
        input_text="Every natural number equals itself.",
        imports=["Mathlib"],
    )
    assert first.input_hash == second.input_hash

    problem_a = TheoremProblem(
        problem_id="a",
        natural_language_statement="For every n, n = n.",
        target_lean_decl="theorem target (n : Nat) : n = n",
    )
    problem_b = TheoremProblem(
        problem_id="b",
        natural_language_statement="For every n,   n = n.",
        target_lean_decl="theorem target (n : Nat) : n = n",
    )
    assert problem_a.problem_hash == problem_b.problem_hash
    assert len(problem_a.problem_hash) == 64

    routed_a = TheoremProblem(
        problem_id="routed-a",
        input_hash=first.input_hash,
        natural_language_statement="First API wording.",
        target_lean_decl="",
    )
    routed_b = TheoremProblem(
        problem_id="routed-b",
        input_hash=first.input_hash,
        natural_language_statement="Different API wording.",
        target_lean_decl="theorem target : True",
    )
    assert routed_a.problem_hash == routed_b.problem_hash
