from lean_prover.Planner.prompts import (
    DECOMPOSITION_SYSTEM_PROMPT,
    FORMALIZATION_SYSTEM_PROMPT,
    INPUT_CLASSIFICATION_SYSTEM_PROMPT,
    LEAN_DECOMPOSITION_SYSTEM_PROMPT,
    LEAN_TO_NATURAL_SYSTEM_PROMPT,
    TARGET_FORMALIZATION_SYSTEM_PROMPT,
    build_decomposition_prompt,
    build_formalization_prompt,
    build_generation_prompt,
    build_input_classification_prompt,
    build_lean_decomposition_prompt,
    build_repair_prompt,
    formalization_prompt,
)
from lean_prover.Planner.schemas import (
    BlueprintPlan,
    BlueprintPlanNode,
    RawTheoremInput,
    TheoremProblem,
    ValidationIssue,
)
from lean_prover.Planner.tests.helpers import environment, proof_length


def test_build_generation_prompt_includes_problem_context() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        imports=["Mathlib", "Mathlib.Data.Nat.Basic"],
        natural_language_statement="Every natural number equals itself.",
        target_lean_decl="theorem target (n : Nat) : n = n",
        available_definitions=["def foo := 1"],
    )

    prompt = build_generation_prompt(problem)

    assert "import Mathlib" in prompt
    assert "import Mathlib.Data.Nat.Basic" in prompt
    assert "Every natural number equals itself." in prompt
    assert problem.target_lean_decl in prompt
    assert "def foo := 1" in prompt
    assert "Every lean_statement must" in prompt
    assert "estimated_proof_length" in DECOMPOSITION_SYSTEM_PROMPT
    assert "entirely in natural mathematical" in DECOMPOSITION_SYSTEM_PROMPT
    assert "Think step by step" in DECOMPOSITION_SYSTEM_PROMPT


def test_build_repair_prompt_includes_issues() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    issue = ValidationIssue(
        stage="graph",
        code="missing_dependency",
        node_id="L1",
        message="L1 depends on L2",
    )

    prompt = build_repair_prompt(
        problem=problem,
        blueprint={"nodes": []},
        issues=[issue],
    )

    assert "theorem target : True" in prompt
    assert "[graph/missing_dependency] L1: L1 depends on L2" in prompt
    assert "output JSON" in prompt


def test_formalization_prompt_wraps_declaration() -> None:
    prompt = formalization_prompt("A theorem saying True is true.")

    assert "A theorem saying True is true." in prompt
    assert "must\nnot contain a proof body" in prompt


def test_formalization_prompt_freezes_plan_and_includes_exact_environment() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        imports=["Mathlib"],
        natural_language_statement="If p implies q and q implies r, then p implies r.",
        target_lean_decl="theorem target (p q r : Prop) : True",
    )
    plan = BlueprintPlan(
        blueprint_summary="Compose implications.",
        nodes=[
            BlueprintPlanNode(
                id="L1",
                title="Composition",
                informal_statement="If p implies q and q implies r, then p implies r.",
                informal_proof="Apply the first implication and then the second.",
                logical_ideas=["Compose the implications"],
                lean_statement="",
                proof_strategy="Apply the implications in sequence.",
                estimated_proof_length=proof_length(),
                difficulty=1,
            )
        ],
        root_dependencies=["L1"],
    )

    prompt = build_formalization_prompt(problem, plan, environment())

    assert environment().lean_commit in prompt
    assert environment().mathlib_commit in prompt
    assert '"lean_statement": ""' in prompt
    assert "semantic alignment" in prompt
    assert "complete structured preamble" in prompt
    assert "do not add, delete, reorder" in FORMALIZATION_SYSTEM_PROMPT
    assert '"alignment_notes"' in FORMALIZATION_SYSTEM_PROMPT
    assert "neither strengthens nor weakens" in FORMALIZATION_SYSTEM_PROMPT


def test_input_gate_and_translation_prompts_have_single_responsibilities() -> None:
    raw = RawTheoremInput(input_text="theorem target : True")
    gate = build_input_classification_prompt(raw)

    assert raw.input_hash in gate
    assert "natural_language|lean" in INPUT_CLASSIFICATION_SYSTEM_PROMPT
    assert "Do not translate, decompose, prove" in INPUT_CLASSIFICATION_SYSTEM_PROMPT
    assert "Preserve every object" in LEAN_TO_NATURAL_SYSTEM_PROMPT
    assert "after its\nnatural-language BlueprintPlan" in TARGET_FORMALIZATION_SYSTEM_PROMPT


def test_lean_decomposition_has_a_dedicated_non_translation_contract() -> None:
    problem = TheoremProblem(
        problem_id="lean-mode",
        target_lean_decl="theorem target (n : Nat) : n = n",
    )

    prompt = build_lean_decomposition_prompt(problem, environment())

    assert "Lean-native task-decomposition API" in LEAN_DECOMPOSITION_SYSTEM_PROMPT
    assert "Do not\ntranslate" in LEAN_DECOMPOSITION_SYSTEM_PROMPT
    assert "Think step by step" in LEAN_DECOMPOSITION_SYSTEM_PROMPT
    assert '"alignment_notes"' in LEAN_DECOMPOSITION_SYSTEM_PROMPT
    assert "return a BlueprintPlan" in prompt
    assert problem.target_lean_decl in prompt
    assert environment().mathlib_commit in prompt


def test_strict_json_and_dag_prompts_include_few_shot_examples() -> None:
    json_prompts = (
        INPUT_CLASSIFICATION_SYSTEM_PROMPT,
        LEAN_TO_NATURAL_SYSTEM_PROMPT,
        TARGET_FORMALIZATION_SYSTEM_PROMPT,
        DECOMPOSITION_SYSTEM_PROMPT,
        LEAN_DECOMPOSITION_SYSTEM_PROMPT,
        FORMALIZATION_SYSTEM_PROMPT,
    )

    for prompt in json_prompts:
        assert "EXAMPLE" in prompt
        assert "JSON OUTPUT" in prompt

    assert '"depends_on": []' in DECOMPOSITION_SYSTEM_PROMPT
    assert '"root_dependencies": ["L1"]' in DECOMPOSITION_SYSTEM_PROMPT
    assert '"semantic_alignment"' in LEAN_DECOMPOSITION_SYSTEM_PROMPT
    assert '"alignment_notes"' in FORMALIZATION_SYSTEM_PROMPT
