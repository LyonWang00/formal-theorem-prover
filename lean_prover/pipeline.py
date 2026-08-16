"""Natural/Lean input -> Planner -> Blueprint -> Prover orchestration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .Data import (
    BlueprintData,
    PlannerDataStatus,
    ProverDataStatus,
    ProverNodeResult,
    ProverProblemResult,
    RawInferenceData,
)

from .Planner.schemas import (
    Blueprint,
    PlannerResult,
    RawTheoremInput,
)
from .Planner.service import PlannerService
from .Prover import BlueprintProver


class InferencePipelineResult(BaseModel):
    success: bool
    stage: str
    input_hash: str
    problem_hash: str = ""
    planner_status: PlannerDataStatus
    prover_status: ProverDataStatus
    raw_data: RawInferenceData
    planner_result: PlannerResult
    blueprint_data: BlueprintData
    prover_result: ProverProblemResult | None = None
    blueprint: Blueprint | None = None
    node_outcomes: list[ProverNodeResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TheoremProvingPipeline:
    """Run the configured Planner and API-only Prover in one process."""

    def __init__(
        self,
        *,
        planner: PlannerService,
        prover: BlueprintProver,
    ) -> None:
        self.planner = planner
        self.prover = prover

    def run(self, raw_input: RawTheoremInput) -> InferencePipelineResult:
        planned = self.planner.plan_input(raw_input)
        raw_data = RawInferenceData(
            input=raw_input,
            detected_input_kind=planned.planner_mode,
        )
        blueprint_data = BlueprintData.from_planner_result(planned)
        if not planned.success or planned.problem is None or planned.blueprint is None:
            return InferencePipelineResult(
                success=False,
                stage=f"planner_{planned.stage}",
                input_hash=raw_input.input_hash,
                problem_hash=(
                    planned.problem.problem_hash if planned.problem else ""
                ),
                planner_status=PlannerDataStatus.FAILURE,
                prover_status=ProverDataStatus.NOT_RUN,
                raw_data=raw_data,
                planner_result=planned,
                blueprint_data=blueprint_data,
                blueprint=planned.blueprint,
            )

        prover_result = self.prover.prove(
            problem=planned.problem,
            blueprint=planned.blueprint,
        )
        success = prover_result.prover_status == ProverDataStatus.SUCCESS
        return InferencePipelineResult(
            success=success,
            stage="prover_success" if success else "prover_failure",
            input_hash=raw_input.input_hash,
            problem_hash=planned.problem.problem_hash,
            planner_status=PlannerDataStatus.SUCCESS,
            prover_status=prover_result.prover_status,
            raw_data=raw_data,
            planner_result=planned,
            blueprint_data=blueprint_data,
            prover_result=prover_result,
            blueprint=prover_result.blueprint,
            node_outcomes=prover_result.node_results,
            warnings=[
                "Planner success/failure and Prover success/failure are stored "
                "as independent stage outcomes."
            ],
        )
