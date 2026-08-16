from lean_prover.Planner.schemas import (
    BlueprintNode,
    LeanEnvironmentIdentity,
    LeanPreamble,
    ProofLengthEstimate,
    SemanticAlignment,
)


def environment() -> LeanEnvironmentIdentity:
    return LeanEnvironmentIdentity(
        lean_version=(
            "Lean (version 4.29.1, commit "
            "f72c35b3f637c8c6571d353742168ab66cc22c00)"
        ),
        lean_commit="f72c35b3f637c8c6571d353742168ab66cc22c00",
        mathlib_commit="5e932f97dd25535344f80f9dd8da3aab83df0fe6",
        environment_hash="4" * 64,
    )


def preamble() -> LeanPreamble:
    return LeanPreamble(imports=["Mathlib"])


def proof_length() -> ProofLengthEstimate:
    return ProofLengthEstimate(
        estimated_lines=3,
        estimated_tokens=24,
        rationale="A short direct proof is expected.",
    )


def alignment() -> SemanticAlignment:
    return SemanticAlignment(
        objects=["the proposition"],
        hypotheses=[],
        conclusion="the proposition holds",
        alignment_notes="The Lean declaration has the same proposition.",
    )


def full_node(
    node_id: str,
    lean_statement: str | None = None,
    *,
    depends_on: list[str] | None = None,
) -> BlueprintNode:
    return BlueprintNode(
        id=node_id,
        title=node_id,
        informal_statement=f"Statement for {node_id}",
        informal_proof=(
            "Use " + ", ".join(f"`{item}`" for item in (depends_on or []))
            + " and apply the stated inference."
            if depends_on
            else "Apply the stated direct inference."
        ),
        logical_ideas=["Apply the stated inference"],
        lean_statement=lean_statement or f"lemma {node_id} : True",
        preamble=preamble(),
        semantic_alignment=alignment(),
        estimated_proof_length=proof_length(),
        depends_on=depends_on or [],
        proof_strategy="Use the assumptions directly.",
        difficulty=1,
    )
