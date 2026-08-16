"""Repair paths used by Planner, Prover, and EI discovery."""

from .blueprint import BluePrintRepairer, JsonlBlueprintRepairStore
from .planner_subproblem import BlueprintRepairer, PlannerSubproblemRepairer


def __getattr__(name: str):
    if name == "DiscoveryAttemptRepairPipeline":
        from .discovery_attempt import DiscoveryAttemptRepairPipeline

        return DiscoveryAttemptRepairPipeline
    if name in {"ProverProofRepairer", "build_prover_repair_prompt"}:
        from .prover_proof import (
            ProverProofRepairer,
            build_prover_repair_prompt,
        )

        return {
            "ProverProofRepairer": ProverProofRepairer,
            "build_prover_repair_prompt": build_prover_repair_prompt,
        }[name]
    raise AttributeError(name)

__all__ = [
    "BluePrintRepairer",
    "BlueprintRepairer",
    "DiscoveryAttemptRepairPipeline",
    "JsonlBlueprintRepairStore",
    "PlannerSubproblemRepairer",
    "ProverProofRepairer",
    "build_prover_repair_prompt",
]
