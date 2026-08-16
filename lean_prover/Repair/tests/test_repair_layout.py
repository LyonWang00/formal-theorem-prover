from lean_prover.Repair.discovery_attempt import (
    DiscoveryAttemptRepairPipeline,
    RepairPipeline,
)
from lean_prover.Repair.planner_subproblem import PlannerSubproblemRepairer
from lean_prover.Repair.prover_proof import ProverProofRepairer
from lean_prover.lean_training.repair_pipeline import (
    RepairPipeline as CompatibilityRepairPipeline,
)


def test_three_repair_modules_have_distinct_public_names() -> None:
    assert PlannerSubproblemRepairer.__name__ == "PlannerSubproblemRepairer"
    assert ProverProofRepairer.__name__ == "ProverProofRepairer"
    assert DiscoveryAttemptRepairPipeline.__name__ == "DiscoveryAttemptRepairPipeline"
    assert RepairPipeline is DiscoveryAttemptRepairPipeline
    assert RepairPipeline is CompatibilityRepairPipeline
