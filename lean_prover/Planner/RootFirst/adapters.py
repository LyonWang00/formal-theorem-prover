"""Adapters from RootFirst contracts to existing Pantograph infrastructure."""

from __future__ import annotations

from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.schemas import LeanCheckResult, TheoremProblem

from .graph import topological_order
from .schemas import RootFirstBlueprint, RootFirstNode


class PantographStatementChecker:
    """Compile one new/revised statement in its exact ancestor context."""

    def __init__(self, backend: PantographDeclarationCheckingBackend) -> None:
        self.backend = backend

    def check(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        node: RootFirstNode,
    ) -> LeanCheckResult:
        nodes = blueprint.node_map()
        order = topological_order(blueprint)
        ancestors: set[str] = set()

        def collect(node_id: str) -> None:
            for father in nodes[node_id].father_nodes:
                if father not in ancestors:
                    ancestors.add(father)
                    collect(father)

        collect(node.id)
        preceding = [
            nodes[node_id].lean_statement
            for node_id in order
            if node_id in ancestors
        ]
        requested = list(
            dict.fromkeys([*problem.imports, *node.preamble.imports])
        )
        # The umbrella module deterministically covers every Mathlib.* import.
        # Reusing it avoids starting a second Pantograph server for the long
        # miniF2F legacy header on every refinement node.
        imports = (
            ["Mathlib"]
            if requested and all(
                item == "Mathlib" or item.startswith("Mathlib.")
                for item in requested
            )
            else requested
        )
        return self.backend.check_declaration(
            imports=imports,
            declaration=node.lean_statement,
            preceding_declarations=preceding,
            preamble=node.preamble,
        )


class AcceptingStatementChecker:
    """Test-only checker; production entry points never construct this."""

    def check(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        node: RootFirstNode,
    ) -> LeanCheckResult:
        return LeanCheckResult(
            success=True,
            declaration=node.lean_statement,
            stdout="test checker accepted statement",
        )
