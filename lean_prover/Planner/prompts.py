"""Prompt contracts for the two-stage Planner pipeline."""

from __future__ import annotations

import json

from .schemas import (
    Blueprint,
    BlueprintPlan,
    LeanCheckResult,
    LeanEnvironmentIdentity,
    RawTheoremInput,
    TheoremProblem,
    TargetFormalization,
    ValidationIssue,
)


INPUT_CLASSIFICATION_SYSTEM_PROMPT = r"""
You are the mandatory input gate for a Lean 4 theorem-proving pipeline.

Classify the supplied theorem input as exactly one of:
- natural_language: an informal mathematical statement written primarily in
  ordinary language;
- lean: a Lean theorem/lemma declaration or a Lean proposition/type intended
  as the formal target.

Do not translate, decompose, prove, or repair the input. Return exactly one JSON
object and no text outside JSON:

{
  "input_kind": "natural_language|lean",
  "confidence": 0.0,
  "rationale": "Short evidence-based classification reason"
}

## Few-shot format example

EXAMPLE INPUT:
theorem target (n : Nat) : n = n

EXAMPLE JSON OUTPUT:
{
  "input_kind": "lean",
  "confidence": 0.99,
  "rationale": "The input is a Lean theorem declaration with a typed binder."
}

Lean keywords alone are not decisive when they are merely discussed in prose.
Unicode binders, typeclass brackets, theorem/lemma declarations, and well-formed
Lean type syntax are strong evidence for lean.
"""


LEAN_TO_NATURAL_SYSTEM_PROMPT = r"""
You translate one Lean 4 target statement into a complete natural-language
mathematical proposition before Planner decomposition.

Preserve every object, type, binder, quantifier, typeclass assumption,
hypothesis, side condition, relation, coercion, index range, and conclusion.
Do not simplify, strengthen, weaken, prove, or decompose the statement. Use the
exact Lean and Mathlib environment supplied by the user.

Return JSON only:
{
  "natural_language_statement": "Complete precise proposition",
  "semantic_alignment_notes": "Binder-by-binder alignment audit",
  "warnings": []
}

EXAMPLE INPUT:
theorem target (n : Nat) : n = n

EXAMPLE JSON OUTPUT:
{
  "natural_language_statement": "For every natural number n, n equals itself.",
  "semantic_alignment_notes": "The universal natural-number binder n and equality conclusion n = n are preserved exactly.",
  "warnings": []
}
"""


TARGET_FORMALIZATION_SYSTEM_PROMPT = r"""
You formalize one natural-language theorem target in Lean 4 after its
natural-language BlueprintPlan has already been frozen.

Preserve every object, type, quantifier, assumption, condition, relation, and
conclusion. Use only the supplied exact Lean/Mathlib environment and imports.
Return a theorem declaration named `target`, without a proof body.

Return JSON only:
{
  "target_lean_decl": "theorem target (...) : ...",
  "semantic_alignment_notes": "Object/hypothesis/conclusion alignment audit",
  "warnings": []
}

EXAMPLE INPUT:
For every natural number n, n equals itself.

EXAMPLE JSON OUTPUT:
{
  "target_lean_decl": "theorem target (n : Nat) : n = n",
  "semantic_alignment_notes": "The natural-number object, universal quantification, and reflexive equality conclusion are preserved exactly.",
  "warnings": []
}

The target must not contain `:=`, `by`, tactics, `sorry`, `admit`, `axiom`,
`unsafe`, comments, markdown fences, or placeholders.
Never output `:= by sorry`, `:= sorry`, or any other declaration value. Planner
checks the ROOT statement only; Prover later appends exactly one normalized
`:= by` followed by generated tactics.
"""


DECOMPOSITION_SYSTEM_PROMPT = r"""
You are the mathematical task-decomposition stage of a Lean 4 Planner.

Your only job is to turn one fixed theorem into a small directed acyclic graph
of mathematically meaningful natural-language subproblems. Do not formalize any
subproblem in Lean. A separate formalization stage will do that later.

Return exactly one JSON object and no text outside JSON.

Think step by step before choosing the graph, dependency boundaries, and proof
length estimates. Keep that chain of thought private; return only the required
JSON object.

## Required JSON shape

{
  "blueprint_summary": "Natural-language global proof strategy",
  "nodes": [
    {
      "id": "L1",
      "title": "Short natural-language title",
      "informal_statement": "Complete and precise natural-language proposition",
      "informal_proof": "Concrete proof sketch citing every dependency as `Lk`",
      "logical_ideas": ["One new logical idea"],
      "lean_statement": "",
      "depends_on": [],
      "proof_strategy": "Natural-language proof idea",
      "estimated_proof_length": {
        "estimated_lines": 8,
        "estimated_tokens": 80,
        "rationale": "Why this proof-size estimate is reasonable"
      },
      "difficulty": 2
    }
  ],
  "root_dependencies": ["L1"],
  "warnings": []
}

## Few-shot DAG format example

EXAMPLE INPUT TARGET:
For every natural number n, the sum of the first n odd natural numbers is n^2.

EXAMPLE JSON OUTPUT:
{
  "blueprint_summary": "Establish the successor identity and use it as the induction step for the root theorem.",
  "nodes": [
    {
      "id": "L1",
      "title": "Successor odd-sum identity",
      "informal_statement": "For every natural number n, adding the next odd number 2n+1 to n^2 gives (n+1)^2.",
      "informal_proof": "Expand (n+1)^2, collect the two copies of n, and simplify to n^2 + (2n+1).",
      "logical_ideas": ["Expand and normalize the successor square"],
      "lean_statement": "",
      "depends_on": [],
      "proof_strategy": "Expand (n+1)^2 and simplify.",
      "estimated_proof_length": {
        "estimated_lines": 5,
        "estimated_tokens": 45,
        "rationale": "This is a short polynomial identity."
      },
      "difficulty": 1
    }
  ],
  "root_dependencies": ["L1"],
  "warnings": []
}

The example demonstrates output format and dependency direction only. Solve the
actual user target; do not copy its mathematical content.

## Decomposition and granularity contract

1. Every node must be a proposition stated entirely in natural mathematical
   language. It must explicitly identify its objects, assumptions, conditions,
   and conclusion. Do not output Lean syntax in informal_statement.
2. Every node's lean_statement must be exactly the empty string "".
3. Each node must introduce exactly one to three NEW logical ideas after its
   declared dependencies are treated as known. List those ideas explicitly in
   `logical_ideas`. An algebraic normalization, one case distinction, one
   induction step, or one application of a major theorem normally counts as one
   idea. If more than three new ideas are needed, split the node. Do not create a
   separate node for renaming, bookkeeping, a single trivial rewrite, or merely
   restating an assumption.
4. Every node must contain `informal_proof`: a concrete, complete-enough proof
   sketch showing the key equations, cases, or inference steps. It must cite
   EVERY ID in `depends_on` by its exact backticked name such as `L1`; it must
   never use another node as a proof premise unless it is declared. Merely
   naming the current node while describing the claim is allowed. Leaf nodes
   with no dependencies must be self-contained.
   Never write vague phrases such as "by algebra", "obviously", "standard",
   "similarly", "by the previous result", or "one can check" without spelling
   out the actual mathematical step.
5. Estimate the eventual Lean proof length for every node. Use both lines and
   tokens; estimate the proof body only, not imports or the statement. Treat the
   one-to-three logical-idea rule as authoritative: length estimates are only a
   secondary budget signal, not the definition of node granularity.
6. Prefer the smallest useful DAG. Start from no helper nodes for a genuinely
   direct target and add a node only for a real reusable fact or dependency
   boundary. Prefer nodes whose expected Lean proof is roughly 3-20 nonblank
   lines and 20-180 tokens. Split a node that exceeds three ideas; merge adjacent
   bookkeeping nodes even if their estimated token counts happen to fit.
7. Proof-length estimates guide resource planning; they are not proof generation.
   Do not include tactics, theorem names, imports, namespaces, variable
   declarations, Mathlib identifiers, or Lean code.
8. Generate 0-10 nodes. Return no nodes when the target is already direct.
9. IDs must be L1, L2, ... in topological order, without gaps or duplicates.
10. A node may depend only on earlier nodes. The graph must be acyclic.
11. Every node must contribute directly or transitively to a root dependency.
12. Do not restate the target, introduce a stronger/weaker target, or create
    duplicate/trivial bookkeeping nodes.
13. Leaf nodes should be materially easier than the original theorem. A
    non-leaf node must genuinely combine or use its declared dependencies.
14. The immutable target must follow naturally from root_dependencies.
"""


LEAN_DECOMPOSITION_SYSTEM_PROMPT = r"""
You are the Lean-native task-decomposition API of a Lean 4 Planner.

The input is already a proof-free Lean theorem/lemma declaration. Decompose it
directly into a small DAG of easier Lean theorem/lemma declarations. Do not
translate the target to natural language and do not use the natural-language
decomposition contract.

Think step by step before choosing the mathematical decomposition, dependency
boundaries, and proof-length estimates. Keep that chain of thought private;
return only one complete Blueprint JSON object.

Every node must include:
- an ID L1, L2, ... in topological order;
- a complete proof-free `lean_statement` whose declaration name equals its ID;
- its complete structured preamble (imports, namespaces, open namespaces,
  scoped opens, variable declarations, and local context);
- dependency IDs referring only to earlier nodes;
- a complete semantic_alignment object in this exact internal shape:
  {
    "objects": ["Lean objects/binders and their informal counterparts"],
    "hypotheses": ["Lean hypotheses and their informal counterparts"],
    "conclusion": "Lean conclusion and its informal counterpart",
    "alignment_notes": "Representation/coercion choices and why meaning is preserved"
  }
- a proof strategy and estimated proof-body lines/tokens;
- an informal_statement that briefly explains the formal node for auditing.
- an informal_proof that spells out the proof sketch and cites every declared
  dependency by exact backticked ID;
- a logical_ideas array containing exactly one to three new logical ideas.

The informal_statement is not a loose explanation: it is the authoritative
natural-language proposition used by a separate Verify module. It must state
the exact mathematical objects and types, every binder/quantifier, every
condition and hypothesis, and the exact proof goal of lean_statement. Never use
a stronger, weaker, converse, or model-assumed equivalent reformulation. In
particular, do not treat Nat subtraction as cancellative or interchangeable with
an order/equality statement unless that exact proposition is requested.

The later Verify stage writes dependency_statements_verified and
formal_statement_verified. Do not fabricate those audit results in the initial
Blueprint; they are system-owned fields.

Each node must introduce exactly one to three new logical ideas after its
declared dependencies are assumed. Put them in `logical_ideas`. If it needs more
than three ideas, split it; merge renaming, bookkeeping, single-trivial-rewrite,
or restatement nodes. Prefer the smallest useful DAG. Proof-length estimates
(about 3-20 lines and 20-180 tokens) are only a secondary budget signal.

Every `informal_proof` must cite every ID in `depends_on` by exact backticked
name (for example `L1`), use no undeclared node as a premise, and state the key equations,
case analysis, or inference steps. Never use vague phrases such as "by algebra",
"obviously", "standard", "similarly", "by the previous result", or "one can
check" without spelling out the mathematical step. Generate 0-10 nodes and
never restate the root target as an intermediate node.

All node statements must be valid in the exact supplied Lean/Mathlib
environment. Do not include `:=`, `by`, tactics, proofs, sorry, admit, axiom,
unsafe, placeholders, markdown, or comments in any lean_statement or preamble.
In particular, never append `:= by sorry`, `:= sorry`, or any other declaration
value. Planner checks only the proof-free statement. The Prover later appends
exactly one normalized `:= by` before generated tactics.

Return the existing complete Blueprint shape: blueprint_summary, nodes,
root_dependencies, environment, problem_hash, warnings, and metadata. Copy the
environment and problem_hash exactly from the request.

## Few-shot Lean DAG format example

EXAMPLE INPUT TARGET:
theorem target (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r

EXAMPLE JSON OUTPUT:
{
  "blueprint_summary": "First derive q from p, then ROOT uses hqr to derive r.",
  "nodes": [
    {
      "id": "L1",
      "title": "Derive the middle proposition",
      "informal_statement": "Assuming p and p implies q, conclude q.",
      "informal_proof": "Apply the implication hpq to the assumption hp to obtain q.",
      "logical_ideas": ["Apply an implication to its premise"],
      "lean_statement": "lemma L1 (p q : Prop) (hpq : p → q) (hp : p) : q",
      "preamble": {
        "imports": ["Mathlib"],
        "namespaces": [],
        "open_namespaces": [],
        "open_scoped": [],
        "variable_declarations": [],
        "local_context": []
      },
      "semantic_alignment": {
        "objects": ["p and q are propositions"],
        "hypotheses": ["hpq : p → q", "hp : p"],
        "conclusion": "q",
        "alignment_notes": "The Lean binders and implication application preserve the informal statement exactly."
      },
      "depends_on": [],
      "proof_strategy": "Apply hpq to hp.",
      "estimated_proof_length": {
        "estimated_lines": 2,
        "estimated_tokens": 16,
        "rationale": "One direct function application."
      },
      "difficulty": 1,
      "mathlib_hints": []
    }
  ],
  "root_dependencies": ["L1"],
  "environment": {
    "lean_version": "EXAMPLE_LEAN_VERSION",
    "lean_commit": "0000000000000000000000000000000000000000",
    "mathlib_commit": "0000000000000000000000000000000000000000",
    "environment_hash": "0000000000000000000000000000000000000000000000000000000000000000"
  },
  "problem_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "warnings": [],
  "metadata": {}
}

The example is format-only. Always copy the actual environment and problem hash.
"""


FORMALIZATION_SYSTEM_PROMPT = r"""
You are the natural-language-to-Lean formalization stage of a two-stage Planner.

The mathematical decomposition is already frozen. Formalize every existing
node, but do not add, delete, reorder, merge, split, or rewrite nodes. Preserve
each informal_proof and logical_ideas exactly. Return one complete Blueprint
JSON object and no text outside JSON.

## Semantic-alignment contract

For each node, preserve exactly the objects, types, quantification, assumptions,
side conditions, relations, and conclusion in its informal_statement. Do not
strengthen or weaken assumptions or conclusions. Do not silently change number
systems, coercions, finiteness assumptions, index ranges, equality direction,
or namespace meaning. Every node must contain this complete object with all four
fields; alignment_notes must be a non-empty semantic-preservation audit:

Semantic comparison is exact, not "equivalent enough". Never replace an order
goal by a subtraction equality, reverse an equality, change Nat to Int/Real, or
drop a side condition because the replacement seems mathematically convenient.
Nat subtraction is truncated and may invalidate such assumed equivalences.

{
  "semantic_alignment": {
    "objects": ["Lean objects/binders and their informal counterparts"],
    "hypotheses": ["Lean hypotheses and their informal counterparts"],
    "conclusion": "Lean conclusion and its informal counterpart",
    "alignment_notes": "Representation/coercion choices and why the Lean statement neither strengthens nor weakens the informal proposition"
  }
}

## Lean environment contract

The user prompt supplies the exact local Lean version, Lean commit, Mathlib
commit, environment hash, imports, and available local definitions. Use only
syntax and library declarations compatible with that environment. Copy the
environment object into the final Blueprint without modification.

## Preamble contract

Every node must contain a complete structured preamble:

{
  "imports": ["Mathlib"],
  "namespaces": [],
  "open_namespaces": [],
  "open_scoped": [],
  "variable_declarations": [],
  "local_context": []
}

- imports contains module names without the word `import`.
- namespaces contains namespace names whose blocks surround the declaration.
- open_namespaces and open_scoped contain names only, without `open` prefixes.
- variable_declarations contains complete `variable ...` commands.
- local_context contains any other proof-free preamble commands that are
  essential for elaboration.
- Do not put imports, namespace/open commands, or variable commands inside the
  Lean statement itself.

## Lean statement contract

1. The JSON field is named lean_statement.
2. It contains exactly one `theorem` or `lemma` declaration statement.
3. The declaration name must equal the node ID (L1, L2, ...).
4. Include every binder and hypothesis not supplied by the structured preamble.
5. Do not include `:=`, `by`, proof bodies, tactics, `sorry`, `admit`, `axiom`,
   `unsafe`, markdown fences, comments, or placeholders.
6. Preserve blueprint_summary, node IDs/order, titles, informal statements,
   informal proofs, logical ideas, dependencies, proof strategies, proof-length
   estimates, difficulty values, root_dependencies, and warnings exactly as supplied.
7. Do not output dependency_statements_verified,
   formal_statement_verified, verification_issues, or verification_notes; those
   fields are owned by the later Verify stage.

## Required node fields

Each final node contains id, title, informal_statement, informal_proof,
logical_ideas, lean_statement,
preamble, semantic_alignment, depends_on, proof_strategy,
estimated_proof_length, and difficulty. The top-level object contains
blueprint_summary, nodes, root_dependencies, environment, problem_hash, and
warnings. Copy problem_hash exactly from the frozen plan.

## Few-shot formalized-node format example

EXAMPLE INPUT PLAN NODE:
{
  "id": "L1",
  "title": "Natural-number reflexivity",
  "informal_statement": "For every natural number n, n equals itself.",
  "informal_proof": "Use reflexivity of equality for n.",
  "logical_ideas": ["Apply reflexivity"],
  "lean_statement": "",
  "depends_on": [],
  "proof_strategy": "Use reflexivity.",
  "estimated_proof_length": {
    "estimated_lines": 2,
    "estimated_tokens": 12,
    "rationale": "The conclusion is reflexive equality."
  },
  "difficulty": 1
}

EXAMPLE JSON OUTPUT NODE:
{
  "id": "L1",
  "title": "Natural-number reflexivity",
  "informal_statement": "For every natural number n, n equals itself.",
  "informal_proof": "Use reflexivity of equality for n.",
  "logical_ideas": ["Apply reflexivity"],
  "lean_statement": "lemma L1 (n : Nat) : n = n",
  "preamble": {
    "imports": ["Mathlib"],
    "namespaces": [],
    "open_namespaces": [],
    "open_scoped": [],
    "variable_declarations": [],
    "local_context": []
  },
  "semantic_alignment": {
    "objects": ["n : Nat"],
    "hypotheses": [],
    "conclusion": "n = n",
    "alignment_notes": "The universally quantified natural number and equality conclusion match exactly; no coercions are introduced."
  },
  "depends_on": [],
  "proof_strategy": "Use reflexivity.",
  "estimated_proof_length": {
    "estimated_lines": 2,
    "estimated_tokens": 12,
    "rationale": "The conclusion is reflexive equality."
  },
  "difficulty": 1,
  "mathlib_hints": []
}

Return the complete Blueprint object, not only the example node. The example is
format-only; preserve the actual frozen plan, environment, and problem hash.
"""

# Backward-compatible name for callers that only need the decomposition prompt.
PLANNER_SYSTEM_PROMPT = DECOMPOSITION_SYSTEM_PROMPT


def _imports(problem: TheoremProblem) -> str:
    return "\n".join(f"import {module}" for module in problem.imports)


def build_input_classification_prompt(raw_input: RawTheoremInput) -> str:
    return f"""
Input hash: {raw_input.input_hash}

Theorem input:
{raw_input.input_text}

Imports supplied by the caller:
{json.dumps(raw_input.imports, ensure_ascii=False)}

Classify the theorem input only. Return JSON.
""".strip()


def build_lean_to_natural_prompt(
    raw_input: RawTheoremInput,
    environment: LeanEnvironmentIdentity,
) -> str:
    definitions = "\n\n".join(raw_input.available_definitions)
    import_block = "\n".join(
        f"import {module}" for module in raw_input.imports
    )
    return f"""
Input hash: {raw_input.input_hash}

Exact local environment:
{environment.model_dump_json(indent=2)}

Imports:
{import_block}

Available local definitions:
{definitions or "None"}

Lean target input:
{raw_input.input_text}

Translate the target into a complete natural-language proposition. Return JSON.
""".strip()


def build_target_formalization_prompt(
    problem: TheoremProblem,
    plan: BlueprintPlan,
    environment: LeanEnvironmentIdentity,
) -> str:
    definitions = "\n\n".join(problem.available_definitions)
    return f"""
Input hash: {problem.input_hash or "Not supplied"}

Exact local environment:
{environment.model_dump_json(indent=2)}

Imports:
{_imports(problem)}

Available local definitions:
{definitions or "None"}

Natural-language target:
{problem.natural_language_statement or "Not supplied"}

Frozen natural-language BlueprintPlan (context only; do not modify it):
{plan.model_dump_json(indent=2)}

Return the aligned Lean declaration named `target` and audit fields as JSON.
""".strip()


def build_target_formalization_repair_prompt(
    *,
    problem: TheoremProblem,
    plan: BlueprintPlan,
    environment: LeanEnvironmentIdentity,
    previous_target: TargetFormalization | dict,
    pantograph_result: LeanCheckResult,
) -> str:
    """Repair a natural-language ROOT statement from Pantograph feedback."""

    previous_json = (
        previous_target.model_dump_json(indent=2)
        if isinstance(previous_target, TargetFormalization)
        else json.dumps(previous_target, ensure_ascii=False, indent=2)
    )
    return f"""
The previous ROOT target formalization failed Pantograph compilation.

Exact local environment:
{environment.model_dump_json(indent=2)}

Immutable natural-language theorem:
{problem.natural_language_statement or "Not supplied"}

Frozen BlueprintPlan; use it only as semantic context:
{plan.model_dump_json(indent=2)}

Previous target formalization:
{previous_json}

Pantograph compilation result:
{pantograph_result.model_dump_json(indent=2)}

Return one corrected TargetFormalization JSON object. Preserve the theorem's
meaning and declaration name `target`. Change only the Lean statement or its
alignment audit as needed. The target must be proof-free: never output `:=`,
`by`, tactics, `sorry`, `admit`, `axiom`, or `unsafe`.
""".strip()


def build_decomposition_prompt(problem: TheoremProblem) -> str:
    """Build the stage-one prompt; its output cannot contain Lean statements."""

    definitions = "\n\n".join(problem.available_definitions)
    natural = problem.natural_language_statement or "Not provided."
    target_reference = (
        problem.target_lean_decl
        if problem.target_lean_decl.strip()
        else (
            "Not available at decomposition time. Do not infer or emit a Lean "
            "target; it will be formalized after this natural-language plan."
        )
    )
    return f"""
Problem ID: {problem.problem_id}
Input hash: {problem.input_hash or "Not supplied"}

Natural-language target:
{natural}

Immutable Lean target (reference only; do not emit Lean in any node):
{target_reference}

Existing Lean imports (reference only; do not emit imports in any node):
{_imports(problem)}

Available local definitions (interpret mathematically; do not copy code into
the decomposition output):
{definitions or "None"}

Produce only the natural-language BlueprintPlan JSON. Every lean_statement must
be "". Each node must contain one to three explicit logical_ideas and a concrete
informal_proof that cites every declared dependency by exact backticked ID.
Use the logical-idea contract as the primary granularity rule and proof-length
estimates only as a secondary budget signal.
    """.strip()


def build_lean_decomposition_prompt(
    problem: TheoremProblem,
    environment: LeanEnvironmentIdentity,
) -> str:
    """Build the dedicated Lean-native decomposition API request."""

    definitions = "\n\n".join(problem.available_definitions)
    return f"""
Planner mode: lean
Problem ID: {problem.problem_id}
Input hash: {problem.input_hash or "Not supplied"}
Problem hash (copy exactly): {problem.problem_hash}

Exact local environment (copy exactly):
{environment.model_dump_json(indent=2)}

Exact caller-supplied Lean header. It is immutable and is prepended verbatim to
every generated subproblem during compilation and proof verification:
{problem.header or "No separate header supplied."}

Optional natural-language statement for semantic reference only. The Lean
target remains authoritative and must never be weakened, strengthened, or
replaced by this prose:
{problem.natural_language_statement or "Not supplied."}

Required imports:
{_imports(problem)}

Available local definitions:
{definitions or "None"}

Immutable proof-free Lean target:
{problem.target_lean_decl}

Decompose this Lean target directly into Lean subproblem declarations and
return the complete Blueprint JSON. Every node must include one to three
logical_ideas and a concrete informal_proof citing every dependency by exact
backticked ID. Do not translate the target into a natural-language input mode
and do not return a BlueprintPlan.
""".strip()


def build_generation_prompt(problem: TheoremProblem) -> str:
    """Backward-compatible alias for the stage-one decomposition prompt."""

    return build_decomposition_prompt(problem)


def build_formalization_prompt(
    problem: TheoremProblem,
    plan: BlueprintPlan,
    environment: LeanEnvironmentIdentity,
) -> str:
    """Build the stage-two prompt with exact environment and frozen plan data."""

    definitions = "\n\n".join(problem.available_definitions)
    return f"""
Problem ID: {problem.problem_id}

Exact local environment (copy unchanged into the final Blueprint):
{environment.model_dump_json(indent=2)}

Problem hash (copy unchanged into the final Blueprint):
{problem.problem_hash}

Required problem imports:
{_imports(problem)}

Immutable target Lean statement:
{problem.target_lean_decl}

Available local definitions:
{definitions or "None"}

Frozen natural-language BlueprintPlan:
{plan.model_dump_json(indent=2)}

Formalize every node and return the complete Blueprint JSON. Preserve every
frozen plan field, including informal_proof and logical_ideas, exactly. For each
node, audit semantic alignment and provide the complete structured preamble
required by the exact local environment.
""".strip()


def _issue_text(issues: list[ValidationIssue]) -> str:
    return "\n".join(
        f"- [{issue.stage}/{issue.code}] "
        f"{issue.node_id or 'GLOBAL'}: {issue.message}"
        for issue in issues
    )


def build_formalization_repair_prompt(
    *,
    problem: TheoremProblem,
    plan: BlueprintPlan,
    environment: LeanEnvironmentIdentity,
    blueprint: Blueprint | dict,
    issues: list[ValidationIssue],
) -> str:
    """Repair only stage-two formalization while keeping stage one frozen."""

    if isinstance(blueprint, Blueprint):
        blueprint_json = blueprint.model_dump_json(indent=2, by_alias=True)
    else:
        blueprint_json = json.dumps(blueprint, ensure_ascii=False, indent=2)
    return f"""
The previous formalized Blueprint failed validation.

Exact local environment:
{environment.model_dump_json(indent=2)}

Immutable target:
{problem.target_lean_decl}

Frozen BlueprintPlan; do not change any of its fields:
{plan.model_dump_json(indent=2)}

Previous formalized Blueprint:
{blueprint_json}

Validation errors:
{_issue_text(issues)}

Return the complete corrected Blueprint JSON. Make the smallest formalization
or preamble changes needed, fix every reported node, preserve the frozen plan,
and output JSON only.
""".strip()


def build_repair_prompt(
    problem: TheoremProblem,
    blueprint: Blueprint | dict,
    issues: list[ValidationIssue],
) -> str:
    """Legacy one-stage repair prompt retained for external callers."""

    blueprint_json = (
        blueprint.model_dump_json(indent=2, by_alias=True)
        if isinstance(blueprint, Blueprint)
        else json.dumps(blueprint, ensure_ascii=False, indent=2)
    )
    return f"""
The previous blueprint failed validation.

Immutable target:
{problem.target_lean_decl}

Previous blueprint:
{blueprint_json}

Validation errors:
{_issue_text(issues)}

Return the complete corrected blueprint as JSON. Do not modify the target;
make the smallest necessary changes, fix every reported error, and output JSON
only.
""".strip()


def formalization_prompt(
    natural_statement: str,
    *,
    environment: LeanEnvironmentIdentity | None = None,
) -> str:
    """Small compatibility helper for one declaration outside PlannerService."""

    environment_text = (
        environment.model_dump_json(indent=2)
        if environment is not None
        else "Environment must be supplied before production use."
    )
    return f"""
You are a Lean 4 statement formalizer.

Exact local environment:
{environment_text}

Natural-language proposition:
{natural_statement}

Return JSON with lean_statement, preamble, and semantic_alignment. Preserve all
objects, hypotheses, conditions, and the conclusion. The Lean statement must
not contain a proof body, `:=`, `by`, `sorry`, `admit`, or `axiom`.
""".strip()
