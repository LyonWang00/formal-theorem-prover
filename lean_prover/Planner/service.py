"""Input routing, two-stage planning, and atomic Lean statement gating."""

from __future__ import annotations

import json

from pydantic import ValidationError

from .client import LLMClient
from .lean_checker import BlueprintLeanChecker
from .lean_decomposition import LeanDecompositionAPI
from .prompts import (
    DECOMPOSITION_SYSTEM_PROMPT,
    FORMALIZATION_SYSTEM_PROMPT,
    INPUT_CLASSIFICATION_SYSTEM_PROMPT,
    TARGET_FORMALIZATION_SYSTEM_PROMPT,
    build_decomposition_prompt,
    build_formalization_prompt,
    build_input_classification_prompt,
    build_target_formalization_repair_prompt,
    build_target_formalization_prompt,
)
from lean_prover.Repair.planner_subproblem import PlannerSubproblemRepairer
from lean_prover.Repair.blueprint import (
    BluePrintRepairer,
    BlueprintRepairStore,
)
from lean_prover.Verify import (
    BlueprintVerificationBatchError,
    BlueprintVerifier,
    SemanticVerificationExhaustedError,
    VerifyStore,
    deduplicate_raw_header,
)
from .schemas import (
    Blueprint,
    BlueprintNodeVerification,
    BlueprintPlan,
    BlueprintPlanNode,
    InputClassification,
    LeanCheckResult,
    LeanEnvironmentIdentity,
    PlannerResult,
    ProblemInputKind,
    RawTheoremInput,
    TargetFormalization,
    TheoremProblem,
    ValidationIssue,
)
from .validator import (
    normalize_lean_target_input,
    validate_blueprint,
    validate_blueprint_plan,
    validate_formalized_blueprint,
)


class PlannerService:
    """Route raw input, plan in natural language, formalize, and compile."""

    def __init__(
        self,
        *,
        client: LLMClient,
        lean_checker: BlueprintLeanChecker,
        max_attempts: int = 3,
        max_formalization_repair_rounds: int | None = None,
        environment: LeanEnvironmentIdentity | None = None,
        verifier: BlueprintVerifier | None = None,
        verify_store: VerifyStore | None = None,
        verify_max_workers: int = 8,
        blueprint_repairer: BluePrintRepairer | None = None,
        blueprint_repair_store: BlueprintRepairStore | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if (
            max_formalization_repair_rounds is not None
            and max_formalization_repair_rounds < 1
        ):
            raise ValueError(
                "max_formalization_repair_rounds must be at least 1"
            )
        self.client = client
        self.lean_checker = lean_checker
        self.repairer = PlannerSubproblemRepairer(
            client,
            lean_checker=lean_checker,
        )
        self.lean_decomposition_api = LeanDecompositionAPI(client)
        self.blueprint_repairer = blueprint_repairer or BluePrintRepairer(
            client,
            store=blueprint_repair_store,
        )
        self.verifier = verifier or BlueprintVerifier(
            client,
            store=verify_store,
            max_workers=verify_max_workers,
        )
        self.max_attempts = max_attempts
        # Preserve callers that historically used max_attempts as a test-time
        # repair budget, while the normal/default service now uses five rounds.
        self.max_formalization_repair_rounds = (
            max_formalization_repair_rounds
            if max_formalization_repair_rounds is not None
            else (5 if max_attempts == 3 else max_attempts)
        )
        self._environment = environment

    def _generate_raw_candidate(self, **kwargs):
        """Keep malformed nonempty model output available to BluePrintRepair."""

        generate_raw = getattr(self.client, "generate_text_or_json", None)
        generate = generate_raw if callable(generate_raw) else self.client.generate_json
        return generate(**kwargs)

    def _run_blueprint_repair(self, **kwargs):
        try:
            return self.blueprint_repairer.run(**kwargs)
        except Exception as error:
            raise RuntimeError(f"BluePrintRepair failed: {error}") from error

    def _resolve_environment(
        self,
        problem: TheoremProblem,
    ) -> LeanEnvironmentIdentity:
        return self._environment or self.lean_checker.environment_identity(problem)

    @staticmethod
    def _schema_issue(stage: str, error: Exception) -> ValidationIssue:
        return ValidationIssue(
            stage=stage,
            code="schema_error",
            message=str(error),
        )

    @staticmethod
    def _problem_id(raw_input: RawTheoremInput) -> str:
        return raw_input.problem_id or f"P-{raw_input.input_hash[:16]}"

    @staticmethod
    def _failure_text(issues: list[ValidationIssue]) -> str:
        return " | ".join(issue.message for issue in issues) or "未知错误"

    @staticmethod
    def _formalization_failure(
        node_ids: list[str],
        issues: list[ValidationIssue],
    ) -> RuntimeError:
        node_label = "、".join(node_ids) or "未知"
        return RuntimeError(
            f"{node_label}节点形式化失败，错误为："
            f"{PlannerService._failure_text(issues)}"
        )

    @staticmethod
    def _apply_problem_header(
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> Blueprint:
        normalized_header, _ = deduplicate_raw_header(problem.header)
        for node in blueprint.nodes:
            node.preamble.raw_header = normalized_header
            node.preamble.imports = list(
                dict.fromkeys([*problem.imports, *node.preamble.imports])
            )
        return blueprint

    @staticmethod
    def _ensure_problem_imports(
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> Blueprint:
        for node in blueprint.nodes:
            node.preamble.imports = list(
                dict.fromkeys([*problem.imports, *node.preamble.imports])
            )
        return blueprint

    @staticmethod
    def _plan_from_repaired_blueprint(
        blueprint: Blueprint,
    ) -> BlueprintPlan:
        """Make the natural-language frozen plan match an authorized DAG repair."""

        return BlueprintPlan(
            blueprint_summary=blueprint.blueprint_summary,
            nodes=[
                BlueprintPlanNode(
                    id=node.id,
                    title=node.title,
                    informal_statement=node.informal_statement,
                    informal_proof=node.informal_proof,
                    logical_ideas=node.logical_ideas,
                    lean_statement="",
                    depends_on=node.depends_on,
                    proof_strategy=node.proof_strategy,
                    estimated_proof_length=node.estimated_proof_length,
                    difficulty=node.difficulty,
                    metadata=node.metadata,
                )
                for node in blueprint.nodes
            ],
            root_dependencies=blueprint.root_dependencies,
            problem_hash=blueprint.problem_hash,
            warnings=blueprint.warnings,
            metadata=blueprint.metadata,
        )

    @staticmethod
    def _apply_verify_results(
        blueprint: Blueprint,
        results: list[BlueprintNodeVerification],
    ) -> list[ValidationIssue]:
        nodes = {node.id: node for node in blueprint.nodes}
        issues: list[ValidationIssue] = []
        for result in results:
            node = nodes[result.node_id]
            original_raw_header = node.preamble.raw_header
            node.preamble = result.corrected_preamble
            # corrected_preamble is now constructed by deterministic code, not
            # by the model. It preserves every unique dataset-header command
            # while removing exact duplicates.
            if original_raw_header and not node.preamble.raw_header:
                node.preamble.raw_header = original_raw_header
            node.dependency_statements_verified = (
                result.dependency_statements_correct
            )
            node.formal_statement_verified = result.formal_statement_correct
            node.verification_issues = [
                *result.dependency_issues,
                *result.formal_statement_issues,
            ]
            node.verification_notes = result.verification_notes
            for item in result.dependency_issues:
                issues.append(
                    ValidationIssue(
                        stage="verify",
                        code="dependency_verification_failed",
                        node_id=node.id,
                        message=(
                            f"{item.message} Repair reference: "
                            f"{item.repair_reference}"
                        ),
                    )
                )
            for item in result.formal_statement_issues:
                issues.append(
                    ValidationIssue(
                        stage="verify",
                        code="formal_statement_verification_failed",
                        node_id=node.id,
                        message=(
                            f"{item.message} Repair reference: "
                            f"{item.repair_reference}"
                        ),
                    )
                )
        return issues

    def _verify_blueprint_nodes(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
        environment: LeanEnvironmentIdentity,
        node_ids: set[str] | None = None,
    ) -> tuple[list[BlueprintNodeVerification], list[ValidationIssue]]:
        try:
            results = self.verifier.verify_blueprint(
                problem=problem,
                blueprint=blueprint,
                environment=environment,
                node_ids=node_ids,
            )
        except BlueprintVerificationBatchError as error:
            # Successful sibling results have already been stored. Apply them
            # before reporting the aggregated node failures so no parallel
            # result is discarded merely because another model call failed.
            partial_issues = self._apply_verify_results(
                blueprint, error.partial_results
            )
            partial_issues.extend(
                ValidationIssue(
                    stage="verify",
                    code=(
                        "semantic_repair_exhausted"
                        if isinstance(failure, SemanticVerificationExhaustedError)
                        else "verify_node_failed"
                    ),
                    node_id=node_id,
                    message=f"{type(failure).__name__}: {failure}",
                )
                for node_id, failure in sorted(error.failures.items())
            )
            # A node-local model/schema failure is a failed sample, not a
            # process failure. Successful siblings and every failed node were
            # already persisted by Verify. Global API/environment errors still
            # reach the batch runner through the returned issue messages and
            # its system-failure classifier.
            self._ensure_problem_imports(problem, blueprint)
            return error.partial_results, partial_issues
        issues = self._apply_verify_results(blueprint, results)
        self._ensure_problem_imports(problem, blueprint)
        return results, issues

    @staticmethod
    def _planning_contract_issues(
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> list[ValidationIssue]:
        """Recheck program-owned narrative fields after generative repair."""

        planning_codes = {
            "invalid_logical_idea_count",
            "empty_logical_idea",
            "invalid_informal_proof_dependencies",
        }
        return [
            issue
            for issue in validate_blueprint(problem, blueprint).issues
            if issue.code in planning_codes
        ]

    @staticmethod
    def _semantic_patch_node_ids(
        blueprint: Blueprint,
        results: list[BlueprintNodeVerification],
    ) -> set[str]:
        """Consume only patches that actually changed the node statement."""

        nodes = {node.id: node for node in blueprint.nodes}
        patched: set[str] = set()
        for result in results:
            node = nodes[result.node_id]
            if node.metadata.pop("verify_semantic_patch_pending", False):
                patched.add(result.node_id)
        return patched

    def _compile_verified_blueprint(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
    ):
        """Pantograph checks exact dependencies and every current statement."""

        result = self.lean_checker.check_blueprint_detailed(problem, blueprint)
        by_id = {row.node_id: row for row in result.node_results}
        nodes = {node.id: node for node in blueprint.nodes}
        for node in blueprint.nodes:
            row = by_id.get(node.id)
            node.metadata.setdefault("verify_pantograph_checks", []).append(
                {
                    "success": bool(row and row.success),
                    "lean_statement": node.lean_decl,
                    "dependency_ids": list(node.depends_on),
                    "exact_dependency_statements": [
                        nodes[dependency].lean_decl
                        for dependency in node.depends_on
                        if dependency in nodes
                    ],
                    "diagnostics": (
                        ""
                        if row is None
                        else row.result.error_message
                        or row.result.stderr
                        or row.result.stdout
                    ),
                }
            )
        return result

    def _formalize_natural_language_root(
        self,
        *,
        provisional: TheoremProblem,
        plan: BlueprintPlan,
        environment: LeanEnvironmentIdentity,
        classification: InputClassification,
    ) -> tuple[TheoremProblem, LeanCheckResult, int] | PlannerResult:
        history: list[dict[str, str]] = []
        raw_target: dict | None = None
        target: TargetFormalization | None = None
        latest_check: LeanCheckResult | None = None
        latest_error = "未知错误"

        for attempt in range(1, self.max_attempts + 2):
            try:
                if attempt == 1:
                    prompt = build_target_formalization_prompt(
                        provisional,
                        plan,
                        environment,
                    )
                else:
                    if target is not None and latest_check is not None:
                        prompt = build_target_formalization_repair_prompt(
                            problem=provisional,
                            plan=plan,
                            environment=environment,
                            previous_target=target,
                            pantograph_result=latest_check,
                        )
                    else:
                        prompt = build_target_formalization_prompt(
                            provisional,
                            plan,
                            environment,
                        )
                raw_target = self.client.generate_json(
                    system_prompt=TARGET_FORMALIZATION_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    empty_response_message=(
                        "目标形式化模型空响应"
                        if attempt == 1
                        else "修复ROOT节点形式化时模型空响应"
                    ),
                    history=history,
                )
                target = TargetFormalization.model_validate(raw_target)
                target_decl = normalize_lean_target_input(
                    target.target_lean_decl,
                    require_target_name=True,
                )
                problem = TheoremProblem(
                    problem_id=provisional.problem_id,
                    imports=provisional.imports,
                    natural_language_statement=(
                        provisional.natural_language_statement
                    ),
                    target_lean_decl=target_decl,
                    header=provisional.header,
                    available_definitions=provisional.available_definitions,
                    input_hash=provisional.input_hash,
                )
            except Exception as error:
                latest_error = str(error)
                # Without a valid statement Pantograph cannot run. Retrying is
                # still bounded by the same three-repair budget.
                if attempt > self.max_attempts:
                    break
                history.extend(
                    [
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                raw_target or {}, ensure_ascii=False
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "ROOT statement schema/normalization failed: "
                                + latest_error
                            ),
                        },
                    ]
                )
                continue

            latest_check = self.lean_checker.check_target_declaration(problem)
            if latest_check.success:
                return problem, latest_check, attempt
            latest_error = (
                latest_check.error_message
                or latest_check.stderr
                or latest_check.stdout
                or "Lean rejected target declaration"
            )
            if attempt > self.max_attempts:
                break
            history.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(raw_target, ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Pantograph compilation result for the previous "
                            "ROOT formalization:\n"
                            + latest_check.model_dump_json(indent=2)
                        ),
                    },
                ]
            )

        failure = RuntimeError(
            f"ROOT节点形式化失败，错误为：{latest_error}"
        )
        return PlannerResult(
            success=False,
            stage="target_formalization_repair",
            problem=provisional,
            input_classification=classification,
            planner_mode=classification.input_kind,
            input_hash=provisional.input_hash,
            plan=plan,
            attempts=2 + self.max_attempts + 1,
            decomposition_attempts=1,
            classification_attempts=1,
            target_formalization_attempts=self.max_attempts + 1,
            environment=environment,
            target_check=latest_check,
            issues=[self._schema_issue("target_formalization_repair", failure)],
        )
    def _environment_failure(
        self,
        *,
        problem: TheoremProblem,
        classification: InputClassification | None = None,
        attempts: int = 0,
        error: Exception,
    ) -> PlannerResult:
        return PlannerResult(
            success=False,
            stage="environment_resolution",
            problem=problem,
            input_classification=classification,
            planner_mode=(classification.input_kind if classification else None),
            input_hash=problem.input_hash,
            attempts=attempts,
            classification_attempts=1 if classification else 0,
            issues=[
                ValidationIssue(
                    stage="environment",
                    code="environment_identity_unavailable",
                    message=str(error),
                )
            ],
        )

    def _decompose(
        self,
        *,
        problem: TheoremProblem,
        environment: LeanEnvironmentIdentity,
        classification: InputClassification | None,
        base_attempts: int,
        translation_attempts: int = 0,
        target_formalization_attempts: int = 0,
    ) -> BlueprintPlan | PlannerResult:
        try:
            raw_plan = self.client.generate_json(
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                user_prompt=build_decomposition_prompt(problem),
                empty_response_message="分解模型空响应",
            )
            plan = BlueprintPlan.model_validate(raw_plan)
            if not plan.problem_hash:
                plan.problem_hash = problem.problem_hash
        except Exception as error:
            result = PlannerResult(
                success=False,
                stage="decomposition",
                problem=problem,
                input_classification=classification,
                planner_mode=(classification.input_kind if classification else None),
                input_hash=problem.input_hash,
                attempts=base_attempts + 1,
                decomposition_attempts=1,
                classification_attempts=1 if classification else 0,
                translation_attempts=translation_attempts,
                target_formalization_attempts=target_formalization_attempts,
                environment=environment,
                issues=[self._schema_issue("decomposition", error)],
            )
            return result

        plan_validation = validate_blueprint_plan(problem, plan)
        if not plan_validation.valid:
            return PlannerResult(
                success=False,
                stage="decomposition",
                problem=problem,
                input_classification=classification,
                planner_mode=(classification.input_kind if classification else None),
                input_hash=problem.input_hash,
                plan=plan,
                attempts=base_attempts + 1,
                decomposition_attempts=1,
                classification_attempts=1 if classification else 0,
                translation_attempts=translation_attempts,
                target_formalization_attempts=target_formalization_attempts,
                environment=environment,
                issues=plan_validation.issues,
            )
        return plan

    def _target_gate(
        self,
        *,
        problem: TheoremProblem,
        environment: LeanEnvironmentIdentity,
        classification: InputClassification | None,
        attempts: int,
        decomposition_attempts: int = 0,
        translation_attempts: int = 0,
        target_formalization_attempts: int = 0,
    ) -> LeanCheckResult | PlannerResult:
        try:
            target_check = self.lean_checker.check_target_declaration(problem)
        except Exception as error:
            return PlannerResult(
                success=False,
                stage="target_lean_validation",
                problem=problem,
                input_classification=classification,
                planner_mode=(classification.input_kind if classification else None),
                input_hash=problem.input_hash,
                attempts=attempts,
                decomposition_attempts=decomposition_attempts,
                classification_attempts=1 if classification else 0,
                translation_attempts=translation_attempts,
                target_formalization_attempts=target_formalization_attempts,
                environment=environment,
                issues=[self._schema_issue("target_lean_validation", error)],
            )
        if target_check.success:
            return target_check
        message = (
            target_check.error_message
            or target_check.stderr
            or "Lean rejected target declaration"
        )
        return PlannerResult(
            success=False,
            stage="target_lean_validation",
            problem=problem,
            input_classification=classification,
            planner_mode=(classification.input_kind if classification else None),
            input_hash=problem.input_hash,
            attempts=attempts,
            decomposition_attempts=decomposition_attempts,
            classification_attempts=1 if classification else 0,
            translation_attempts=translation_attempts,
            target_formalization_attempts=target_formalization_attempts,
            environment=environment,
            target_check=target_check,
            issues=[
                ValidationIssue(
                    stage="lean",
                    code="target_declaration_failed",
                    message=message,
                )
            ],
        )

    def _formalize_and_check(
        self,
        *,
        problem: TheoremProblem,
        plan: BlueprintPlan,
        environment: LeanEnvironmentIdentity,
        classification: InputClassification | None = None,
        base_attempts: int = 1,
        translation_attempts: int = 0,
        target_formalization_attempts: int = 0,
    ) -> PlannerResult:
        blueprint_repair_rounds = []
        try:
            raw_blueprint = self._generate_raw_candidate(
                system_prompt=FORMALIZATION_SYSTEM_PROMPT,
                user_prompt=build_formalization_prompt(
                    problem,
                    plan,
                    environment,
                ),
                empty_response_message=(
                    "形式化节点"
                    + "、".join(node.id for node in plan.nodes)
                    + "时模型空响应"
                    if plan.nodes
                    else "节点形式化模型空响应"
                ),
            )
            blueprint, blueprint_repair_rounds = self._run_blueprint_repair(
                problem=problem,
                environment=environment,
                candidate=raw_blueprint,
            )
            plan = self._plan_from_repaired_blueprint(blueprint)
            blueprint = self._apply_problem_header(problem, blueprint)
        except Exception as error:
            return PlannerResult(
                success=False,
                stage=(
                    "blueprint_repair"
                    if blueprint_repair_rounds or "BluePrintRepair" in str(error)
                    else "formalization"
                ),
                problem=problem,
                input_classification=classification,
                planner_mode=(classification.input_kind if classification else None),
                input_hash=problem.input_hash,
                plan=plan,
                attempts=base_attempts + 1,
                decomposition_attempts=1,
                formalization_attempts=1,
                blueprint_repair_attempts=len(blueprint_repair_rounds),
                blueprint_repair_rounds=blueprint_repair_rounds,
                classification_attempts=1 if classification else 0,
                translation_attempts=translation_attempts,
                target_formalization_attempts=target_formalization_attempts,
                environment=environment,
                issues=[self._schema_issue("formalization", error)],
            )

        accumulated_issues: list[ValidationIssue] = []
        latest_node_checks = []
        latest_failed_node_ids: list[str] = []
        all_verify_results: list[BlueprintNodeVerification] = []
        verify_rounds = 0
        verify_node_ids: set[str] | None = None

        # Initial generation followed by up to five retrieval-guided repairs.
        for formalization_attempt in range(
            1, self.max_formalization_repair_rounds + 2
        ):
            verify_results, verify_issues = self._verify_blueprint_nodes(
                problem=problem,
                blueprint=blueprint,
                environment=environment,
                node_ids=verify_node_ids,
            )
            verify_rounds += 1
            all_verify_results.extend(verify_results)
            issues = list(verify_issues)
            issues.extend(self._planning_contract_issues(problem, blueprint))
            semantic_patch_ids = self._semantic_patch_node_ids(
                blueprint, verify_results
            )
            static_result = validate_formalized_blueprint(
                problem,
                plan,
                blueprint,
                environment,
            )
            issues.extend(static_result.issues)
            lean_result = self._compile_verified_blueprint(
                problem=problem,
                blueprint=blueprint,
            )
            latest_node_checks = lean_result.node_results
            latest_failed_node_ids = lean_result.failed_node_ids
            issues.extend(lean_result.issues)
            nonsemantic_issues = [
                issue
                for issue in issues
                if issue.code != "formal_statement_verification_failed"
            ]
            if not issues and lean_result.success:
                return PlannerResult(
                    success=True,
                    stage="completed",
                    problem=problem,
                    input_classification=classification,
                    planner_mode=(classification.input_kind if classification else None),
                    input_hash=problem.input_hash,
                    plan=plan,
                    blueprint=blueprint,
                    attempts=base_attempts + formalization_attempt,
                    decomposition_attempts=1,
                    formalization_attempts=formalization_attempt,
                    classification_attempts=1 if classification else 0,
                    translation_attempts=translation_attempts,
                    target_formalization_attempts=(
                        target_formalization_attempts
                    ),
                    verify_attempts=verify_rounds,
                    blueprint_repair_attempts=len(blueprint_repair_rounds),
                    blueprint_repair_rounds=blueprint_repair_rounds,
                    verify_results=all_verify_results,
                    environment=environment,
                    node_checks=latest_node_checks,
                    failed_node_ids=[],
                    issues=accumulated_issues,
                )

            accumulated_issues.extend(issues)
            if formalization_attempt > self.max_formalization_repair_rounds:
                break
            if semantic_patch_ids and not nonsemantic_issues and lean_result.success:
                # The Verify model supplied a semantics-preserving patch and
                # Pantograph accepted it with the exact predecessor context.
                # Re-enter Verify directly; no generative repair is needed.
                verify_node_ids = semantic_patch_ids
                continue
            try:
                verify_node_ids = {
                    issue.node_id
                    for issue in issues
                    if issue.node_id is not None
                } or None
                blueprint = self.repairer.repair(
                    problem=problem,
                    plan=plan,
                    environment=environment,
                    previous_blueprint=blueprint,
                    issues=issues,
                )
                blueprint = self._ensure_problem_imports(problem, blueprint)
                if not blueprint.problem_hash:
                    blueprint.problem_hash = problem.problem_hash
            except (ValidationError, ValueError, RuntimeError) as error:
                accumulated_issues.append(
                    self._schema_issue("formalization_repair", error)
                )
                continue

        failure = self._formalization_failure(
            latest_failed_node_ids
            or [node.id for node in blueprint.nodes],
            accumulated_issues,
        )
        accumulated_issues.append(
            self._schema_issue("formalization_repair", failure)
        )
        return PlannerResult(
            success=False,
            stage=("lean_validation" if latest_node_checks else "formalization"),
            problem=problem,
            input_classification=classification,
            planner_mode=(classification.input_kind if classification else None),
            input_hash=problem.input_hash,
            plan=plan,
            blueprint=blueprint,
            attempts=(
                base_attempts + self.max_formalization_repair_rounds + 1
            ),
            decomposition_attempts=1,
            formalization_attempts=(
                self.max_formalization_repair_rounds + 1
            ),
            classification_attempts=1 if classification else 0,
            translation_attempts=translation_attempts,
            target_formalization_attempts=target_formalization_attempts,
            verify_attempts=verify_rounds,
            blueprint_repair_attempts=len(blueprint_repair_rounds),
            blueprint_repair_rounds=blueprint_repair_rounds,
            verify_results=all_verify_results,
            environment=environment,
            node_checks=latest_node_checks,
            failed_node_ids=latest_failed_node_ids,
            issues=accumulated_issues,
        )

    def _decompose_lean_and_check(
        self,
        *,
        problem: TheoremProblem,
        environment: LeanEnvironmentIdentity,
        classification: InputClassification,
        base_attempts: int,
    ) -> PlannerResult:
        """Call the Lean-native API, then gate every generated declaration."""

        blueprint_repair_rounds = []
        try:
            raw_blueprint = self.lean_decomposition_api.decompose_candidate(
                problem=problem,
                environment=environment,
            )
            blueprint, blueprint_repair_rounds = self._run_blueprint_repair(
                problem=problem,
                environment=environment,
                candidate=raw_blueprint,
            )
            blueprint = self._apply_problem_header(problem, blueprint)
        except Exception as error:
            return PlannerResult(
                success=False,
                stage=(
                    "blueprint_repair"
                    if blueprint_repair_rounds or "BluePrintRepair" in str(error)
                    else "lean_decomposition"
                ),
                problem=problem,
                input_classification=classification,
                planner_mode=ProblemInputKind.LEAN,
                input_hash=problem.input_hash,
                attempts=base_attempts + 1,
                decomposition_attempts=1,
                blueprint_repair_attempts=len(blueprint_repair_rounds),
                blueprint_repair_rounds=blueprint_repair_rounds,
                classification_attempts=1,
                environment=environment,
                issues=[self._schema_issue("lean_decomposition", error)],
            )
            result.classification_attempts = (
                0 if classification.rationale.startswith("formal_statement") else 1
            )
            return result

        accumulated_issues: list[ValidationIssue] = []
        latest_node_checks = []
        latest_failed_node_ids: list[str] = []
        all_verify_results: list[BlueprintNodeVerification] = []
        verify_rounds = 0
        verify_node_ids: set[str] | None = None
        for lean_attempt in range(
            1, self.max_formalization_repair_rounds + 2
        ):
            verify_results, verify_issues = self._verify_blueprint_nodes(
                problem=problem,
                blueprint=blueprint,
                environment=environment,
                node_ids=verify_node_ids,
            )
            verify_rounds += 1
            all_verify_results.extend(verify_results)
            issues = list(verify_issues)
            issues.extend(self._planning_contract_issues(problem, blueprint))
            semantic_patch_ids = self._semantic_patch_node_ids(
                blueprint, verify_results
            )
            static_result = validate_blueprint(problem, blueprint)
            issues.extend(static_result.issues)
            if blueprint.environment != environment:
                issues.append(
                    ValidationIssue(
                        stage="lean_decomposition",
                        code="environment_identity_drift",
                        message=(
                            "Lean-native Blueprint environment differs from "
                            "the local environment"
                        ),
                    )
                )
            lean_result = self._compile_verified_blueprint(
                problem=problem,
                blueprint=blueprint,
            )
            latest_node_checks = lean_result.node_results
            latest_failed_node_ids = lean_result.failed_node_ids
            issues.extend(lean_result.issues)
            nonsemantic_issues = [
                issue
                for issue in issues
                if issue.code != "formal_statement_verification_failed"
            ]
            if not issues and lean_result.success:
                return PlannerResult(
                    success=True,
                    stage="completed",
                    problem=problem,
                    input_classification=classification,
                    planner_mode=ProblemInputKind.LEAN,
                    input_hash=problem.input_hash,
                    blueprint=blueprint,
                    attempts=base_attempts + lean_attempt,
                    decomposition_attempts=1,
                    formalization_attempts=0,
                    classification_attempts=1,
                    translation_attempts=0,
                    target_formalization_attempts=0,
                    verify_attempts=verify_rounds,
                    blueprint_repair_attempts=len(blueprint_repair_rounds),
                    blueprint_repair_rounds=blueprint_repair_rounds,
                    verify_results=all_verify_results,
                    environment=environment,
                    node_checks=latest_node_checks,
                    failed_node_ids=[],
                    issues=accumulated_issues,
                )

            accumulated_issues.extend(issues)
            if lean_attempt > self.max_formalization_repair_rounds:
                break
            if semantic_patch_ids and not nonsemantic_issues and lean_result.success:
                verify_node_ids = semantic_patch_ids
                continue
            try:
                verify_node_ids = {
                    issue.node_id
                    for issue in issues
                    if issue.node_id is not None
                } or None
                blueprint = self.repairer.repair(
                    problem=problem,
                    previous_blueprint=blueprint,
                    issues=issues,
                    environment=environment,
                    plan=None,
                )
                blueprint = self._ensure_problem_imports(problem, blueprint)
                if not blueprint.problem_hash:
                    blueprint.problem_hash = problem.problem_hash
            except (ValidationError, ValueError, RuntimeError) as error:
                accumulated_issues.append(
                    self._schema_issue("lean_subproblem_repair", error)
                )
                continue

        failure = self._formalization_failure(
            latest_failed_node_ids
            or [node.id for node in blueprint.nodes],
            accumulated_issues,
        )
        accumulated_issues.append(
            self._schema_issue("lean_subproblem_repair", failure)
        )
        return PlannerResult(
            success=False,
            stage=("lean_validation" if latest_node_checks else "lean_decomposition"),
            problem=problem,
            input_classification=classification,
            planner_mode=ProblemInputKind.LEAN,
            input_hash=problem.input_hash,
            blueprint=blueprint,
            attempts=(
                base_attempts + self.max_formalization_repair_rounds + 1
            ),
            decomposition_attempts=1,
            formalization_attempts=0,
            classification_attempts=1,
            translation_attempts=0,
            target_formalization_attempts=0,
            verify_attempts=verify_rounds,
            blueprint_repair_attempts=len(blueprint_repair_rounds),
            blueprint_repair_rounds=blueprint_repair_rounds,
            verify_results=all_verify_results,
            environment=environment,
            node_checks=latest_node_checks,
            failed_node_ids=latest_failed_node_ids,
            issues=accumulated_issues,
        )

    def plan(self, problem: TheoremProblem) -> PlannerResult:
        """Plan an already prepared problem containing both NL and Lean target."""

        try:
            environment = self._resolve_environment(problem)
        except Exception as error:
            return self._environment_failure(problem=problem, error=error)
        target_gate = self._target_gate(
            problem=problem,
            environment=environment,
            classification=None,
            attempts=0,
        )
        if isinstance(target_gate, PlannerResult):
            return target_gate
        decomposed = self._decompose(
            problem=problem,
            environment=environment,
            classification=None,
            base_attempts=0,
        )
        if isinstance(decomposed, PlannerResult):
            return decomposed
        result = self._formalize_and_check(
            problem=problem,
            plan=decomposed,
            environment=environment,
            base_attempts=1,
        )
        result.target_check = target_gate
        return result

    def plan_input(self, raw_input: RawTheoremInput) -> PlannerResult:
        """Run the mandatory API gate and route natural-language/Lean input."""

        explicit_formal = bool(
            raw_input.formal_statement
            and raw_input.formal_statement.strip()
        )
        if explicit_formal:
            classification = InputClassification(
                input_kind=ProblemInputKind.LEAN,
                confidence=1.0,
                rationale=(
                    "formal_statement is present; deterministic Lean-first routing"
                ),
            )
            classification_attempts = 0
        else:
            try:
                raw_classification = self.client.generate_json(
                    system_prompt=INPUT_CLASSIFICATION_SYSTEM_PROMPT,
                    user_prompt=build_input_classification_prompt(raw_input),
                    empty_response_message="输入分类模型空响应",
                )
                classification = InputClassification.model_validate(
                    raw_classification
                )
                classification_attempts = 1
            except Exception as error:
                return PlannerResult(
                    success=False,
                    stage="input_classification",
                    input_hash=raw_input.input_hash,
                    attempts=1,
                    classification_attempts=1,
                    issues=[self._schema_issue("input_classification", error)],
                )

        try:
            provisional = TheoremProblem(
                problem_id=self._problem_id(raw_input),
                imports=raw_input.imports,
                natural_language_statement=(
                    raw_input.informal_stmt or raw_input.input_text
                    if classification.input_kind
                    == ProblemInputKind.NATURAL_LANGUAGE
                    else raw_input.informal_stmt
                ),
                target_lean_decl=(
                    ""
                    if classification.input_kind
                    == ProblemInputKind.NATURAL_LANGUAGE
                    else normalize_lean_target_input(
                        raw_input.formal_statement or raw_input.input_text
                    )
                ),
                header=raw_input.header,
                available_definitions=raw_input.available_definitions,
                input_hash=raw_input.input_hash,
            )
        except Exception as error:
            return PlannerResult(
                success=False,
                stage="lean_target_validation",
                input_classification=classification,
                planner_mode=classification.input_kind,
                input_hash=raw_input.input_hash,
                attempts=1,
                classification_attempts=classification_attempts,
                issues=[self._schema_issue("lean_target_validation", error)],
            )
        try:
            environment = self._resolve_environment(provisional)
        except Exception as error:
            return self._environment_failure(
                problem=provisional,
                classification=classification,
                attempts=1,
                error=error,
            )

        if classification.input_kind == ProblemInputKind.LEAN:
            problem = provisional
            target_gate = self._target_gate(
                problem=problem,
                environment=environment,
                classification=classification,
                attempts=1,
            )
            if isinstance(target_gate, PlannerResult):
                target_gate.classification_attempts = classification_attempts
                target_gate.attempts -= 1 - classification_attempts
                return target_gate
            result = self._decompose_lean_and_check(
                problem=problem,
                environment=environment,
                classification=classification,
                base_attempts=1,
            )
            result.classification_attempts = classification_attempts
            result.attempts -= 1 - classification_attempts
            result.target_check = target_gate
            return result

        decomposed = self._decompose(
            problem=provisional,
            environment=environment,
            classification=classification,
            base_attempts=1,
        )
        if isinstance(decomposed, PlannerResult):
            return decomposed
        root_result = self._formalize_natural_language_root(
            provisional=provisional,
            plan=decomposed,
            environment=environment,
            classification=classification,
        )
        if isinstance(root_result, PlannerResult):
            return root_result
        problem, target_gate, root_attempts = root_result
        decomposed.problem_hash = problem.problem_hash
        result = self._formalize_and_check(
            problem=problem,
            plan=decomposed,
            environment=environment,
            classification=classification,
            base_attempts=3,
            target_formalization_attempts=root_attempts,
        )
        result.target_check = target_gate
        return result
