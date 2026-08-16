"""Self-looping repair gate between Blueprint generation and Verify."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from lean_prover.Planner.client import LLMClient
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintRepairOutput,
    BlueprintRepairRound,
    BlueprintRepairState,
    LeanEnvironmentIdentity,
    TheoremProblem,
    ValidationIssue,
)
from lean_prover.Planner.validator import validate_blueprint


BLUEPRINT_REPAIR_SYSTEM_PROMPT = r"""
You are BluePrintRepair, the strict repair gate immediately after a Planner
generates a complete Lean Blueprint and before the separate Verify module.

Independently inspect the current input only. You receive no prior-round
conversation and must not infer hidden history.

MANDATORY EXECUTION ORDER -- perform these steps in exactly this order:
1. SNAPSHOT: treat the received candidate as immutable input and remember its
   complete data-level value, excluding no Blueprint fields.
2. INSPECT: check JSON/schema completeness and objective DAG structure using
   the rules below. Inspect every node before editing. Do not judge the
   mathematical or logical adequacy of declared dependencies.
3. REPAIR: if you found any defect, immediately fix the actual Blueprint fields
   that cause it. Do not merely notice, describe, flag, or plan a repair. Recheck
   the entire repaired result from scratch and continue fixing until the returned
   Blueprint itself satisfies every check.
4. COMPARE: compare the final returned Blueprint, excluding only the new `state`
   field, against the exact received candidate at the data level.
5. ASSIGN STATE LAST:
   - If and only if inspection found zero defects and comparison shows zero data
     changes, set `state` to `success`.
   - If inspection found one or more defects, the returned Blueprint MUST contain
     the concrete repairs, comparison MUST show a data change, and `state` MUST be
     `failed`. Here `failed` means "this round's INPUT was defective and has now
     been repaired"; it does not mean "I found a possible concern" or "repair
   failed".

Although you decide the state only after comparison, serialize `state` as the
FIRST top-level JSON key. The field is mandatory and may never be omitted or
null. The first two output characters after `{` and whitespace must begin the
key `"state"`.

ABSOLUTE STATE INVARIANTS:
- NEVER return `state="failed"` with an unchanged Blueprint.
- NEVER return `state="failed"` as a substitute for performing the repair.
- NEVER return `state="success"` after changing, normalizing, reordering,
  deleting, or adding any Blueprint data.
- Do not choose `failed` merely because your role is named BluePrintRepair, the
  prompt lists possible checks, or an earlier round might have made a repair.
- If deterministic issues are supplied for this current input, repair every one
  in the returned fields; do not just acknowledge them.
- If the current candidate is malformed, truncated, or not valid JSON, do not
  copy its broken suffix. Reconstruct one complete valid JSON object containing
  every required Blueprint field and the final state. Close every string,
  object, and array. Never return a partial JSON prefix.
- Do not output a diagnosis or proposed patch. The complete returned Blueprint
  IS the executed repair.

Check and, when necessary, minimally repair:
1. JSON/schema completeness: return every required Blueprint and BlueprintNode
   field with the correct JSON type. Preserve valid fields. Each node must have
   a complete informal_statement and a proof-free lean_statement.
2. DAG structure: IDs are L1..Ln, dependencies reference existing earlier
   nodes, the graph is acyclic, every node reaches the virtual ROOT, and ROOT is
   the unique sink. root_dependencies are ROOT's incoming predecessors; never
   place a ROOT node in nodes.
3. Node planning fields: every node has exactly one to three non-empty
   `logical_ideas` and a concrete `informal_proof`. Treat a present one-to-three
   item list as the authoritative granularity audit: do not independently judge
   or rewrite its mathematical logic. The proof sketch must cite every declared
   dependency by exact backticked ID and must not use an undeclared node as a
   proof premise. A descriptive mention of the current node is not a dependency.

If `logical_ideas` or `informal_proof` is missing, do not invent mathematical
content in this repair model. Copy an existing non-empty `proof_strategy` into
`informal_proof` and use that same existing strategy as the single
`logical_ideas` entry; add exact backticked dependency IDs to that copied text
when they are absent. This is schema recovery only. Verify does not audit these
fields, and later actual proving evaluates whether the plan is adequate.

Do NOT judge whether a dependency is mathematically necessary, sufficient, or
logically correct. Do NOT add, delete, or rewire nodes because of a model-level
opinion about the proof strategy. Mathematical dependency adequacy is evaluated
later by actual proving, not by BluePrintRepair. Change graph fields only for
objective schema/structural defects such as missing IDs, cycles, order, or ROOT
reachability.

Return exactly one JSON object: the complete Blueprint plus top-level state.
Do not return JSONL fences, prose, diagnostics, proofs, tactics, :=, by, sorry,
admit, axiom, unsafe, comments, placeholders, or a node with id ROOT.

Few-shot malformed-input example:
INPUT:
{"blueprint_summary":"Use a helper.","nodes":[{"id":"L2","title":"Helper","informal_statement":"For every natural n, n equals n.","informal_proof":"Use reflexivity for n.","logical_ideas":["Apply reflexivity"],"lean_statement":"lemma L2 (n : Nat) : n = n","depends_on":["L9"]}],"root_dependencies":[]}

OUTPUT:
{
  "state":"failed",
  "blueprint_summary":"Use a helper.",
  "nodes":[{
    "id":"L1",
    "title":"Helper",
    "informal_statement":"For every natural number n, n equals n.",
    "informal_proof":"Use reflexivity of equality for n.",
    "logical_ideas":["Apply reflexivity"],
    "lean_statement":"lemma L1 (n : Nat) : n = n",
    "preamble":{"imports":["Mathlib"],"raw_header":"","namespaces":[],"open_namespaces":[],"open_scoped":[],"variable_declarations":[],"local_context":[]},
    "semantic_alignment":{"objects":["n : Nat"],"hypotheses":[],"conclusion":"n = n","alignment_notes":"The natural object and equality goal are identical."},
    "estimated_proof_length":{"estimated_lines":1,"estimated_tokens":4,"rationale":"Reflexivity."},
    "depends_on":[],"proof_strategy":"Use reflexivity.","mathlib_hints":[],"difficulty":1
  }],
  "root_dependencies":["L1"],
  "environment":{"lean_version":"Lean EXAMPLE","lean_commit":"0000000000000000000000000000000000000000","mathlib_commit":"0000000000000000000000000000000000000000","environment_hash":"0000000000000000000000000000000000000000000000000000000000000000"},
  "problem_hash":"0000000000000000000000000000000000000000000000000000000000000000",
  "warnings":[],"metadata":{}
}

Why `failed` is required in this example: the output visibly repairs the node ID
L2 -> L1, removes the missing L9 dependency, adds all required fields, and makes
L1 a predecessor of ROOT. Returning the original malformed data with only
`state="failed"` would be invalid.

Few-shot correct-input behavior:
If the input already has valid complete fields and an acyclic fully
ROOT-reachable DAG with ROOT as unique sink, copy the complete Blueprint
byte-for-byte at the data level and add only "state":"success". Do not make
this decision depend on a mathematical judgment about the dependency chain.

Few-shot state decision for an already-correct input:
INPUT Blueprint data: B
INSPECTION: no defect
FINAL Blueprint data excluding state: exactly B
OUTPUT: `state":"success"` as the first top-level key, with every field of B
unchanged after it.

Forbidden output for that same correct input:
`state":"failed"` plus all fields of B unchanged
This is forbidden because no concrete repair was executed.

The `status`, `parent_id`, `children`, `local_context`, `proof_attempts`,
`verified_proof`, `lean_feedback`, `retrieval_hints`, Verify-result fields, and
node `metadata` are optional runtime/default fields. Preserve them when present;
do not invent proof results before Verify or Prover.

Before emitting JSON, silently perform this final assertion:
`state == "failed"` if and only if the returned Blueprint data is different
because at least one detected defect was concretely repaired. Otherwise the
Blueprint must be unchanged and `state == "success"`.
""".strip()


class BlueprintRepairStore(Protocol):
    def record(self, payload: dict[str, object]) -> None:
        ...


class JsonlBlueprintRepairStore:
    def __init__(
        self,
        path: str | Path = "outputs/blueprint_repair/results.jsonl",
    ) -> None:
        self.path = Path(path)

    def record(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _candidate_json(candidate: dict[str, Any] | str | Blueprint) -> str:
    if isinstance(candidate, Blueprint):
        return candidate.model_dump_json(indent=2, by_alias=True)
    if isinstance(candidate, str):
        return candidate
    return json.dumps(candidate, ensure_ascii=False, indent=2)


def _candidate_data(candidate: dict[str, Any] | str | Blueprint) -> Any:
    if isinstance(candidate, Blueprint):
        return candidate.model_dump(mode="json", by_alias=True)
    if isinstance(candidate, str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return candidate
    return candidate


def _changed_paths(before: Any, after: Any, path: str = "") -> list[str]:
    """Return deterministic JSON paths changed by an executed repair."""

    if isinstance(before, dict) and isinstance(after, dict):
        paths: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            if key not in before or key not in after:
                paths.append(child)
            else:
                paths.extend(_changed_paths(before[key], after[key], child))
        return paths
    if isinstance(before, list) and isinstance(after, list):
        paths = []
        for index in range(max(len(before), len(after))):
            child = f"{path}[{index}]"
            if index >= len(before) or index >= len(after):
                paths.append(child)
            else:
                paths.extend(_changed_paths(before[index], after[index], child))
        return paths
    return [path or "$ROOT"] if before != after else []


def deterministic_blueprint_issues(
    *,
    problem: TheoremProblem,
    candidate: dict[str, Any] | str | Blueprint,
    environment: LeanEnvironmentIdentity,
) -> tuple[Blueprint | None, list[ValidationIssue]]:
    parsed_for_completeness: Any = candidate
    try:
        if isinstance(candidate, str):
            parsed = json.loads(candidate)
            parsed_for_completeness = parsed
        elif isinstance(candidate, Blueprint):
            parsed = candidate
        else:
            parsed = candidate
        blueprint = Blueprint.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        return None, [
            ValidationIssue(
                stage="blueprint_repair",
                code="json_or_schema_error",
                message=str(error),
            )
        ]

    issues: list[ValidationIssue] = []
    if isinstance(parsed_for_completeness, dict):
        required_top = {
            "blueprint_summary", "nodes", "root_dependencies", "environment",
            "problem_hash", "warnings", "metadata",
        }
        missing_top = sorted(required_top - set(parsed_for_completeness))
        if missing_top:
            issues.append(ValidationIssue(
                stage="blueprint_repair",
                code="incomplete_blueprint_json",
                message=f"Missing top-level fields: {missing_top}",
            ))
        required_node = {
            "id", "title", "informal_statement", "informal_proof",
            "logical_ideas", "lean_statement", "preamble",
            "semantic_alignment", "estimated_proof_length", "depends_on",
            "proof_strategy", "mathlib_hints", "difficulty",
        }
        for index, node in enumerate(parsed_for_completeness.get("nodes", [])):
            if not isinstance(node, dict):
                continue
            available = set(node)
            if "lean_decl" in available:
                available.add("lean_statement")
            missing_node = sorted(required_node - available)
            if missing_node:
                issues.append(ValidationIssue(
                    stage="blueprint_repair",
                    code="incomplete_blueprint_node_json",
                    node_id=str(node.get("id") or f"index:{index}"),
                    message=f"Missing node fields: {missing_node}",
                ))
    issues.extend(validate_blueprint(problem, blueprint).issues)
    if blueprint.problem_hash != problem.problem_hash:
        issues.append(ValidationIssue(
            stage="blueprint_repair",
            code="missing_or_incorrect_problem_hash",
            message="Blueprint problem_hash must equal the root problem hash",
        ))
    if blueprint.environment != environment:
        issues.append(
            ValidationIssue(
                stage="blueprint_repair",
                code="environment_identity_drift",
                message="Blueprint environment differs from the exact local environment",
            )
        )
    # With edges dependency -> dependent plus node -> ROOT, every node reaching
    # ROOT is exactly the condition that virtual ROOT is the sole sink.
    if any(issue.code == "disconnected_node" for issue in issues):
        issues.append(
            ValidationIssue(
                stage="blueprint_repair",
                code="root_not_unique_sink",
                message="Virtual ROOT is not the unique sink because at least one node cannot reach ROOT",
            )
        )
    return blueprint, issues


class BluePrintRepairer:
    """Run at most three stateless inspection/repair rounds."""

    def __init__(
        self,
        client: LLMClient,
        *,
        store: BlueprintRepairStore | None = None,
        max_rounds: int = 3,
    ) -> None:
        if max_rounds != 3:
            raise ValueError("BluePrintRepair max_rounds is fixed at 3")
        self.client = client
        self.store = store
        self.max_rounds = max_rounds

    def _prompt(
        self,
        *,
        problem: TheoremProblem,
        environment: LeanEnvironmentIdentity,
        candidate: dict[str, Any] | str | Blueprint,
        issues: list[ValidationIssue],
    ) -> str:
        return f"""
Exact root problem:
{problem.model_dump_json(indent=2)}

Exact local environment:
{environment.model_dump_json(indent=2)}

Deterministic JSON/schema/DAG issues found before this round:
{json.dumps([issue.model_dump(mode="json") for issue in issues], ensure_ascii=False, indent=2)}

Current candidate only; independently inspect and repair it:
BEGIN_CURRENT_BLUEPRINT_CANDIDATE
{_candidate_json(candidate)}
END_CURRENT_BLUEPRINT_CANDIDATE

Now execute SNAPSHOT -> INSPECT -> REPAIR -> COMPARE -> ASSIGN STATE LAST.
This round is invalid if you return `failed` without a visible Blueprint data
change. If there is a defect, fix its actual field now. If there is no defect,
copy the Blueprint without any normalization and return `success`.
After deciding it last, serialize mandatory non-null `state` as the first
top-level key of the complete JSON object.
""".strip()

    def run(
        self,
        *,
        problem: TheoremProblem,
        environment: LeanEnvironmentIdentity,
        candidate: dict[str, Any] | str | Blueprint,
    ) -> tuple[Blueprint, list[BlueprintRepairRound]]:
        current = candidate
        rounds: list[BlueprintRepairRound] = []

        def record(row: BlueprintRepairRound) -> None:
            rounds.append(row)
            if self.store is not None:
                self.store.record({
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "problem_id": problem.problem_id,
                    "problem_hash": problem.problem_hash,
                    **row.model_dump(mode="json"),
                })

        for round_id in range(1, self.max_rounds + 1):
            input_snapshot = _candidate_data(current)
            before, issues = deterministic_blueprint_issues(
                problem=problem,
                candidate=current,
                environment=environment,
            )
            generate_raw = getattr(self.client, "generate_text_or_json", None)
            generate = (
                generate_raw if callable(generate_raw) else self.client.generate_json
            )
            raw = generate(
                system_prompt=BLUEPRINT_REPAIR_SYSTEM_PROMPT,
                user_prompt=self._prompt(
                    problem=problem,
                    environment=environment,
                    candidate=current,
                    issues=issues,
                ),
                empty_response_message="BluePrintRepair模型空响应",
                history=None,
            )
            try:
                output = BlueprintRepairOutput.model_validate(raw)
            except (ValidationError, ValueError) as error:
                raw_blueprint_data: Any = raw
                if isinstance(raw, dict):
                    raw_blueprint_data = dict(raw)
                    raw_blueprint_data.pop("state", None)
                row = BlueprintRepairRound(
                    round_id=round_id,
                    state=None,
                    input_was_valid_blueprint=before is not None and not issues,
                    deterministic_issues_before=issues,
                    changed=_candidate_data(current) != raw_blueprint_data,
                    changed_fields=_changed_paths(
                        _candidate_data(current), raw_blueprint_data
                    ),
                    input_snapshot=input_snapshot,
                    raw_output=raw,
                    output_error=str(error),
                )
                record(row)
                current = raw
                continue
            returned = Blueprint.model_validate(
                output.model_dump(exclude={"state"}, by_alias=True)
            )
            returned_data = returned.model_dump(mode="json", by_alias=True)
            raw_returned = dict(raw)
            raw_returned.pop("state", None)
            comparison_before = _candidate_data(current)
            comparison_output: Any = raw_returned
            if not issues and before is not None:
                comparison_before = before.model_dump(mode="json", by_alias=True)
                comparison_output = returned_data
            changed_fields = _changed_paths(comparison_before, comparison_output)
            changed = bool(changed_fields)
            contract_error: str | None = None
            effective_state = output.state
            raw_output_for_audit: dict[str, object] | str | None = None
            # `state` is a deterministic fact about input/output data, not a
            # creative model decision.  If an already-valid Blueprint is
            # returned byte-for-byte with `failed`, construct the canonical
            # success state in code and retain the raw response for audit.
            if (
                output.state == BlueprintRepairState.FAILED
                and not changed
                and not issues
                and before is not None
            ):
                effective_state = BlueprintRepairState.SUCCESS
                raw_output_for_audit = raw
            if issues and output.state != BlueprintRepairState.FAILED:
                contract_error = (
                    "BluePrintRepair must set state=failed when deterministic "
                    "repair was required"
                )
            if output.state == BlueprintRepairState.SUCCESS:
                if issues:
                    contract_error = "state=success cannot bypass deterministic issues"
                elif changed:
                    contract_error = "state=success must not modify Blueprint data"
            elif not changed and effective_state == BlueprintRepairState.FAILED:
                contract_error = "state=failed requires an actual repair to Blueprint data"
            row = BlueprintRepairRound(
                round_id=round_id,
                state=effective_state,
                input_was_valid_blueprint=before is not None and not issues,
                deterministic_issues_before=issues,
                changed=changed,
                changed_fields=changed_fields,
                input_snapshot=input_snapshot,
                blueprint=returned,
                raw_output=raw if contract_error else raw_output_for_audit,
                output_error=contract_error,
            )
            record(row)
            current = returned
            if effective_state == BlueprintRepairState.SUCCESS and contract_error is None:
                break
        final, final_issues = deterministic_blueprint_issues(
            problem=problem,
            candidate=current,
            environment=environment,
        )
        if final is None or final_issues:
            raise ValueError(
                "BluePrintRepair exhausted three rounds with invalid output: "
                + " | ".join(issue.message for issue in final_issues)
            )
        return final, rounds


__all__ = [
    "BLUEPRINT_REPAIR_SYSTEM_PROMPT",
    "BluePrintRepairer",
    "BlueprintRepairStore",
    "JsonlBlueprintRepairStore",
    "deterministic_blueprint_issues",
]
