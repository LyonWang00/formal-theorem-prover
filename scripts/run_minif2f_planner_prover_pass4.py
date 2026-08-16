#!/usr/bin/env python3
"""Run a restartable controlled miniF2F Planner/Prover pass@k audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any

from openai import OpenAI

from lean_prover.Data import ProverDataStatus
from lean_prover.pipeline import TheoremProvingPipeline
from lean_prover.Planner.client import OpenAICompatibleClient
from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.pantograph_checker import PantographDeclarationCheckingBackend
from lean_prover.Planner.preamble import imports_from_header
from lean_prover.Planner.lean_decomposition import LeanDecompositionAPI
from lean_prover.Planner.prompts import (
    LEAN_DECOMPOSITION_SYSTEM_PROMPT,
    build_lean_decomposition_prompt,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    LeanEnvironmentIdentity,
    ProblemInputKind,
    RawTheoremInput,
    TheoremProblem,
)
from lean_prover.Planner.service import PlannerService
from lean_prover.Prover import (
    BlueprintProver,
    JsonlProverResultStore,
    JsonlProverTelemetryStore,
    NodeProofVerifier,
    OpenAICompatibleProofGenerator,
)
from lean_prover.Prover.repair_policy import DEFAULT_CONTROLLED_PROOF_BUDGET
from lean_prover.Verify import BlueprintVerifier, JsonlVerifyStore
from lean_prover.Repair.blueprint import (
    BluePrintRepairer,
    JsonlBlueprintRepairStore,
)
from lean_prover.Repair.prover_proof import ProverProofRepairer
from lean_prover.lean_training.verification.pantograph import (
    PantographTheoremVerifier,
)


MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_TOTAL_GRAPH_NODES = 4
MAX_HELPER_NODES = MAX_TOTAL_GRAPH_NODES - 1


ORIGINAL_BLUEPRINT_SYSTEM_PROMPT = r"""
You are the Lean-native task-decomposition API of a Lean 4 Planner. The input
is an immutable proof-free Lean theorem or lemma declaration. Produce one
complete JSON Blueprint containing a small DAG of easier proof-free Lean
theorem or lemma declarations. Return exactly one JSON object and no prose.

Every helper node must contain: id L1, L2, ... in topological order; title;
an exact informal_statement; a proof-free lean_statement whose declaration
name equals the node id; a complete preamble; semantic_alignment; depends_on;
proof_strategy; estimated_proof_length; difficulty; and mathlib_hints. The
top-level object contains blueprint_summary, nodes, root_dependencies,
environment, problem_hash, warnings, and metadata. Copy the supplied exact
environment and problem_hash. Dependencies refer only to earlier nodes. The
graph is acyclic, every helper reaches virtual ROOT, and ROOT is the unique
sink. Never include a node whose id is ROOT.

This is the original whole-Blueprint comparison arm. Do not generate or reason
from `informal_proof` or `logical_ideas`; the caller adds compatibility values
only after your response so that the current storage schema can represent this
historical arm. Choose the overall DAG using proof_strategy and proof-size
estimates. Do not include :=, by, tactics, proofs, sorry, admit, axiom, unsafe,
comments, markdown, imports, or namespace commands inside lean_statement.

HARD EXPERIMENT BUDGET: the nodes array may contain zero, one, two, or three
helper nodes only. Therefore the complete graph contains at most four nodes
after counting the virtual ROOT. Never return four or more helper nodes.

Few-shot format example (values are illustrative only):
{
  "blueprint_summary":"Use one intermediate implication.",
  "nodes":[{
    "id":"L1",
    "title":"Derive q",
    "informal_statement":"Assuming p and p implies q, conclude q.",
    "lean_statement":"lemma L1 (p q : Prop) (hpq : p → q) (hp : p) : q",
    "preamble":{"imports":["Mathlib"],"raw_header":"","namespaces":[],"open_namespaces":[],"open_scoped":[],"variable_declarations":[],"local_context":[]},
    "semantic_alignment":{"objects":["p and q are propositions"],"hypotheses":["hpq : p → q","hp : p"],"conclusion":"q","alignment_notes":"The binders, assumptions, and conclusion match exactly."},
    "depends_on":[],
    "proof_strategy":"Apply hpq to hp.",
    "estimated_proof_length":{"estimated_lines":2,"estimated_tokens":16,"rationale":"One direct application."},
    "difficulty":1,
    "mathlib_hints":[]
  }],
  "root_dependencies":["L1"],
  "environment":{"lean_version":"Lean EXAMPLE","lean_commit":"0000000000000000000000000000000000000000","mathlib_commit":"0000000000000000000000000000000000000000","environment_hash":"0000000000000000000000000000000000000000000000000000000000000000"},
  "problem_hash":"0000000000000000000000000000000000000000000000000000000000000000",
  "warnings":[],
  "metadata":{}
}
Always solve the actual target and copy its actual identity values.
""".strip()


def _add_original_schema_compatibility(raw: Any) -> Any:
    """Populate current storage-only fields without exposing them to the model."""

    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
    if not isinstance(parsed, dict):
        return raw
    for node in parsed.get("nodes", []):
        if not isinstance(node, dict):
            continue
        strategy = str(node.get("proof_strategy") or "Follow the stated plan.")
        node.setdefault("informal_proof", strategy)
        node.setdefault("logical_ideas", [strategy])
    return parsed


class ArchitectureLeanDecompositionAPI(LeanDecompositionAPI):
    def __init__(self, client: object, *, architecture: str) -> None:
        super().__init__(client)
        self.architecture = architecture

    def decompose_candidate(
        self,
        *,
        problem: TheoremProblem,
        environment: LeanEnvironmentIdentity,
    ) -> dict[str, Any] | str:
        prompt = (
            ORIGINAL_BLUEPRINT_SYSTEM_PROMPT
            if self.architecture == "original"
            else LEAN_DECOMPOSITION_SYSTEM_PROMPT
            + "\n\nHARD EXPERIMENT BUDGET: return at most three helper nodes in "
            "the nodes array, so the complete graph including virtual ROOT "
            "has at most four nodes. Never return four or more helpers."
        )
        generate_raw = getattr(self.client, "generate_text_or_json", None)
        generate = (
            generate_raw if callable(generate_raw) else self.client.generate_json
        )
        raw = generate(
            system_prompt=prompt,
            user_prompt=build_lean_decomposition_prompt(problem, environment),
            empty_response_message="分解模型空响应",
        )
        if self.architecture == "original":
            return _add_original_schema_compatibility(raw)
        return raw


class NodeBudgetBluePrintRepairer(BluePrintRepairer):
    """Expose the experiment cap to repair and reject any final violation."""

    def _prompt(self, **kwargs: Any) -> str:
        return super()._prompt(**kwargs) + (
            "\n\nHARD EXPERIMENT BUDGET: preserve or repair the candidate so "
            "that nodes contains at most three helper nodes (at most four total "
            "nodes including virtual ROOT)."
        )

    def run(self, **kwargs: Any):
        blueprint, rounds = super().run(**kwargs)
        if len(blueprint.nodes) > MAX_HELPER_NODES:
            raise ValueError(
                "experiment node budget exceeded: "
                f"{len(blueprint.nodes)} helper nodes > {MAX_HELPER_NODES}"
            )
        return blueprint, rounds

LEAN_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_']*|[0-9]+|:=|=>|<;>|<=|>=|!=|[^\s]"
)


def lexical_proof_token_count(proof: str) -> int:
    """Count Lean-like lexical tokens when provider token usage is unavailable."""

    return len(LEAN_TOKEN_PATTERN.findall(proof))


def tactic_command_count(proof: str) -> int:
    """Approximate Lean tactic commands while excluding layout-only syntax."""

    count = 0
    for raw_line in proof.splitlines():
        line = raw_line.split("--", 1)[0].strip()
        while line.startswith("·"):
            line = line[1:].lstrip()
        if not line or line == "by":
            continue
        if line.startswith("by "):
            line = line[3:].lstrip()
        branch = re.match(r"^(?:\||case\s+)[^=]*=>\s*(.*)$", line)
        if branch is not None:
            line = branch.group(1).strip()
            if not line:
                continue
        commands = [
            part.strip()
            for part in re.split(r"\s*<;>\s*|\s*;\s*", line)
            if part.strip()
        ]
        count += len(commands)
    return count


def attempt_proof_token_count(attempt: Any) -> tuple[int, str]:
    metadata = getattr(attempt, "metadata", None) or {}
    if getattr(attempt, "prompt_type", "") == "api_proof_repair_retrieval":
        # One structured repair response contains up to three candidates. The
        # provider completion count belongs to that shared JSON envelope; do
        # not assign the same count to every candidate proof.
        return (
            lexical_proof_token_count(getattr(attempt, "proof", "")),
            "lean_lexical_repair_candidate",
        )
    usage = metadata.get("api_usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens >= 0:
        return completion_tokens, "api_completion_tokens"
    return lexical_proof_token_count(getattr(attempt, "proof", "")), "lean_lexical"


def node_attempt_statistics(nodes: list[Any], root_result: Any | None) -> list[dict[str, Any]]:
    all_nodes: list[tuple[str, list[Any], bool]] = [
        (node.id, list(node.proof_attempts), False) for node in nodes
    ]
    if root_result is not None:
        all_nodes.append(("ROOT", list(root_result.proof_attempts), True))

    statistics: list[dict[str, Any]] = []
    for node_id, attempts, is_root in all_nodes:
        token_measurements = [attempt_proof_token_count(attempt) for attempt in attempts]
        token_counts = [value for value, _ in token_measurements]
        tactic_counts = [tactic_command_count(attempt.proof) for attempt in attempts]
        sources = sorted({source for _, source in token_measurements})
        attempt_count = len(attempts)
        base_attempt_count = sum(
            attempt.prompt_type != "api_proof_repair_retrieval"
            for attempt in attempts
        )
        repair_attempt_count = attempt_count - base_attempt_count
        base_success_count = sum(
            bool(getattr(attempt, "success", False))
            and attempt.prompt_type != "api_proof_repair_retrieval"
            for attempt in attempts
        )
        repair_success_count = sum(
            bool(getattr(attempt, "success", False))
            and attempt.prompt_type == "api_proof_repair_retrieval"
            for attempt in attempts
        )
        statistics.append(
            {
                "node_id": node_id,
                "is_root": is_root,
                "attempt_count": attempt_count,
                "base_attempt_count": base_attempt_count,
                "repair_attempt_count": repair_attempt_count,
                "base_success_count": base_success_count,
                "repair_success_count": repair_success_count,
                "base_pass_at_k": base_success_count > 0,
                # Backward-compatible alias for older analysis scripts.
                "base_pass_at_4": base_success_count > 0,
                "repair_rescued": (
                    base_success_count == 0 and repair_success_count > 0
                ),
                "proof_token_count_source": sources,
                "proof_token_count_total": sum(token_counts),
                "average_proof_token_count": (
                    round(sum(token_counts) / attempt_count, 4)
                    if attempt_count
                    else None
                ),
                "tactic_count_total": sum(tactic_counts),
                "average_tactic_count": (
                    round(sum(tactic_counts) / attempt_count, 4)
                    if attempt_count
                    else None
                ),
                "attempts": [
                    {
                        "attempt_id": attempt.attempt_id,
                        "prompt_type": attempt.prompt_type,
                        "success": bool(getattr(attempt, "success", False)),
                        "proof_token_count": token_count,
                        "proof_token_count_source": token_source,
                        "tactic_count": tactic_count,
                    }
                    for attempt, (token_count, token_source), tactic_count in zip(
                        attempts, token_measurements, tactic_counts, strict=True
                    )
                ],
            }
        )
    return statistics


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_rows(
    dataset_paths: list[Path],
    manifest_path: Path,
    count: int,
    seed: int,
    source_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if manifest_path.is_file():
        manifest = load_jsonl(manifest_path)
        if len(manifest) != count:
            raise RuntimeError(
                f"existing manifest has {len(manifest)} rows, expected {count}"
            )
        return manifest
    rows = [
        row
        for dataset_path in dataset_paths
        for row in load_jsonl(dataset_path)
    ]
    all_source_ids = [row["id"] for row in rows]
    if len(all_source_ids) != len(set(all_source_ids)):
        raise ValueError("dataset inputs contain duplicate source IDs")
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a {len(rows)}-row dataset")
    if source_ids:
        by_id = {row["id"]: row for row in rows}
        missing = [source_id for source_id in source_ids if source_id not in by_id]
        if missing:
            raise ValueError(f"requested source IDs are absent: {missing}")
        selected = [by_id[source_id] for source_id in source_ids]
    else:
        selected = random.Random(seed).sample(rows, count)
    for sample_index, row in enumerate(selected):
        append_jsonl(
            manifest_path,
            {
                "sample_index": sample_index,
                "seed": seed,
                "source_id": row["id"],
                "source_split": row.get("split"),
                "informal_stmt": row["informal_stmt"],
                "reference_formal_statement": row["formal_statement"],
                "source_header": row.get("header", ""),
            },
        )
    return load_jsonl(manifest_path)


def completed_ids(results_path: Path) -> set[str]:
    if not results_path.is_file():
        return set()
    return {row["source_id"] for row in load_jsonl(results_path)}


def system_failure_message(result: Any) -> str | None:
    text = " ".join(issue.message for issue in result.planner_result.issues).casefold()
    api_text_signals = (
        "authentication",
        "api key",
        "connection error",
        "connection refused",
        "model not found",
        "unknown model",
    )
    api_status_signal = re.search(
        r"(?:http(?: status)?|status(?: code)?|error)\s*[:=]?\s*(?:401|403)\b"
        r"|\b(?:401|403)\s+(?:unauthorized|forbidden)\b",
        text,
    )
    if api_status_signal or any(signal in text for signal in api_text_signals):
        return "global model API failure detected"
    environment_signals = (
        "object file",
        ".olean' of module",
        "unknown module",
        "module not found",
    )
    if any(signal in text for signal in environment_signals):
        return "Lean/header environment incompatibility detected"
    return None


def write_node_attempt_statistics_report(
    rows: list[dict[str, Any]],
    report_path: Path,
) -> None:
    """Materialize one compact JSONL record per problem/node pair."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for node_statistics in row.get("node_attempt_statistics", []):
                payload = {
                    "schema_version": "minif2f_node_attempt_statistics_v1",
                    "sample_index": row["sample_index"],
                    "source_id": row["source_id"],
                    "source_split": row["source_split"],
                    "planner_success": row["planner_success"],
                    "prover_success": row["prover_success"],
                    **node_statistics,
                }
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                )


def summarize(results_path: Path, summary_path: Path) -> dict[str, Any]:
    rows = load_jsonl(results_path) if results_path.is_file() else []
    planner_successes = sum(row["planner_success"] for row in rows)
    prover_successes = sum(row["prover_success"] for row in rows)
    decomposed = sum(row["node_count"] > 0 for row in rows)
    node_count = sum(row["node_count"] for row in rows)
    proved_nodes = sum(row["proved_node_count"] for row in rows)
    base_proved_nodes = sum(
        row.get("base_proved_node_count", row["proved_node_count"])
        for row in rows
    )
    repair_rescued_nodes = sum(row.get("repair_rescued_node_count", 0) for row in rows)
    root_results = sum(row.get("root_attempted", False) for row in rows)
    proved_roots = sum(row.get("root_proved", False) for row in rows)
    base_proved_roots = sum(
        row.get("root_base_proved", row.get("root_proved", False)) for row in rows
    )
    repair_rescued_roots = sum(row.get("root_repair_rescued", False) for row in rows)
    attempt_count = sum(row.get("attempt_count", 0) for row in rows)
    base_attempt_count = sum(row.get("base_attempt_count", 0) for row in rows)
    repair_attempt_count = sum(row.get("repair_attempt_count", 0) for row in rows)
    proof_token_total = sum(row.get("attempt_proof_token_total", 0) for row in rows)
    tactic_total = sum(row.get("attempt_tactic_total", 0) for row in rows)
    architecture = rows[0].get("architecture") if rows else None
    total_graph_nodes = sum(
        row.get("total_graph_node_count", row["node_count"] + 1)
        for row in rows
    )
    summary = {
        "schema_version": "minif2f_planner_prover_passk_summary_v1",
        "model": MODEL,
        "architecture": architecture,
        "attempts_per_node": (
            rows[0].get("attempts_per_node") if rows else None
        ),
        "max_formalization_repair_rounds": (
            rows[0].get("max_formalization_repair_rounds") if rows else None
        ),
        "max_prover_repair_rounds": (
            rows[0].get("max_prover_repair_rounds") if rows else None
        ),
        "max_prover_repair_candidates_per_node": (
            rows[0].get("max_prover_repair_candidates_per_node")
            if rows
            else None
        ),
        "prover_repair_candidates_per_generation": 1,
        "max_total_graph_nodes": MAX_TOTAL_GRAPH_NODES,
        "processed_problem_count": len(rows),
        "planner_success_count": planner_successes,
        "planner_success_rate": planner_successes / len(rows) if rows else 0.0,
        "problems_with_nonempty_decomposition": decomposed,
        "nonempty_decomposition_rate": decomposed / len(rows) if rows else 0.0,
        "prover_success_count": prover_successes,
        "prover_success_rate": prover_successes / len(rows) if rows else 0.0,
        "total_node_count": node_count,
        "total_graph_node_count_including_root": total_graph_nodes,
        "average_total_graph_node_count_including_root": (
            total_graph_nodes / len(rows) if rows else 0.0
        ),
        "average_planner_generated_subgoal_count": (
            node_count / len(rows) if rows else 0.0
        ),
        "proved_node_count": proved_nodes,
        "base_proved_node_count": base_proved_nodes,
        "repair_rescued_node_count": repair_rescued_nodes,
        "node_pass_at_k": base_proved_nodes / node_count if node_count else 0.0,
        "node_success_after_repair_rate": (
            proved_nodes / node_count if node_count else 0.0
        ),
        "root_result_count": root_results,
        "root_proved_count": proved_roots,
        "root_base_proved_count": base_proved_roots,
        "root_repair_rescued_count": repair_rescued_roots,
        "root_pass_at_k": (
            base_proved_roots / root_results if root_results else 0.0
        ),
        "root_success_after_repair_rate": (
            proved_roots / root_results if root_results else 0.0
        ),
        "attempt_count": attempt_count,
        "base_attempt_count": base_attempt_count,
        "repair_attempt_count": repair_attempt_count,
        "attempt_proof_token_total": proof_token_total,
        "average_attempt_proof_token_count": (
            proof_token_total / attempt_count if attempt_count else None
        ),
        "attempt_tactic_total": tactic_total,
        "average_attempt_tactic_count": (
            tactic_total / attempt_count if attempt_count else None
        ),
        "per_problem_elapsed_seconds": {
            str(row["source_id"]): float(row["elapsed_seconds"])
            for row in rows
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_node_attempt_statistics_report(
        rows,
        summary_path.parent / "node_attempt_statistics.jsonl",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Select an exact source ID; may be repeated for directed tests.",
    )
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--enable-prover-repair",
        action="store_true",
        help=(
            "Enable environment-index retrieval/generation/Pantograph proof "
            "repair (one shared --max-prover-repair-rounds budget per node "
            "after all pass@k base attempts fail)."
        ),
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        default=DEFAULT_CONTROLLED_PROOF_BUDGET.pass_k,
    )
    parser.add_argument(
        "--max-prover-repair-rounds",
        type=int,
        default=DEFAULT_CONTROLLED_PROOF_BUDGET.repair_rounds_per_node,
    )
    parser.add_argument(
        "--max-prover-repair-candidates-per-node",
        type=int,
        default=DEFAULT_CONTROLLED_PROOF_BUDGET.repair_candidates_per_node,
    )
    parser.add_argument("--max-formalization-repair-rounds", type=int, default=2)
    parser.add_argument(
        "--architecture",
        choices=("original", "logical_ideas"),
        default="logical_ideas",
        help="Select the whole-Blueprint comparison arm.",
    )
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    if args.count < 1:
        raise ValueError("count must be positive")
    for name in (
        "pass_k",
        "max_prover_repair_rounds",
        "max_prover_repair_candidates_per_node",
        "max_formalization_repair_rounds",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.source_id:
        if len(args.source_id) != len(set(args.source_id)):
            raise ValueError("--source-id values must be unique")
        args.count = len(args.source_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sample_manifest.jsonl"
    results_path = args.output_dir / "problem_results.jsonl"
    summary_path = args.output_dir / "summary.json"
    fatal_path = args.output_dir / "fatal_error.json"
    manifest = selected_rows(
        args.dataset,
        manifest_path,
        args.count,
        args.seed,
        args.source_id,
    )
    remaining = [
        row for row in manifest if row["source_id"] not in completed_ids(results_path)
    ]
    if args.stop_after is not None:
        remaining = remaining[: args.stop_after]

    planner_client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=BASE_URL,
        model=MODEL,
        temperature=0.2,
        max_tokens=8192,
    )
    include_blueprint_reasoning = args.architecture == "logical_ideas"
    proof_generator = OpenAICompatibleProofGenerator(
        client=OpenAI(api_key=api_key, base_url=BASE_URL),
        model=MODEL,
        temperature=0.2,
        max_tokens=2048,
        include_blueprint_reasoning=include_blueprint_reasoning,
    )
    planner_backend = PantographDeclarationCheckingBackend(
        project_path=args.lean_project,
        timeout=args.timeout,
    )
    proof_runtime: PantographTheoremVerifier | None = None
    try:
        proof_runtime = PantographTheoremVerifier(
            args.lean_project,
            imports=("Mathlib",),
            timeout=args.timeout,
        )
        planner = PlannerService(
                client=planner_client,
                lean_checker=BlueprintLeanChecker(planner_backend),
                max_attempts=3,
                max_formalization_repair_rounds=(
                    args.max_formalization_repair_rounds
                ),
                verifier=BlueprintVerifier(
                    planner_client,
                    store=JsonlVerifyStore(
                        args.output_dir / "verify_results.jsonl"
                    ),
                    max_semantic_repairs=(
                        args.max_formalization_repair_rounds
                    ),
                ),
                blueprint_repairer=NodeBudgetBluePrintRepairer(
                    planner_client,
                    store=JsonlBlueprintRepairStore(
                        args.output_dir / "blueprint_repair_results.jsonl"
                    ),
                ),
            )
        planner.lean_decomposition_api = ArchitectureLeanDecompositionAPI(
            planner_client,
            architecture=args.architecture,
        )
        pipeline = TheoremProvingPipeline(
            planner=planner,
            prover=BlueprintProver(
                generator=proof_generator,
                verifier=NodeProofVerifier(proof_runtime),
                proof_repairer=(
                    ProverProofRepairer(
                        client=OpenAI(api_key=api_key, base_url=BASE_URL),
                        model=MODEL,
                        project_path=args.lean_project,
                        temperature=0.0,
                        max_tokens=2048,
                        include_blueprint_reasoning=include_blueprint_reasoning,
                    )
                    if args.enable_prover_repair
                    else None
                ),
                max_proof_repair_rounds=args.max_prover_repair_rounds,
                max_proof_repair_candidates_per_node=(
                    args.max_prover_repair_candidates_per_node
                ),
                attempts_per_node=args.pass_k,
                telemetry_store=JsonlProverTelemetryStore(
                    attempts_path=args.output_dir / "attempts.jsonl"
                ),
                result_store=JsonlProverResultStore(
                    success_path=args.output_dir / "prover_success.jsonl",
                    failure_path=args.output_dir / "prover_failure.jsonl",
                ),
            ),
        )
        for position, source in enumerate(remaining, start=1):
            start = time.monotonic()
            raw_input = RawTheoremInput(
                input_text=source["reference_formal_statement"],
                formal_statement=source["reference_formal_statement"],
                informal_stmt=source["informal_stmt"],
                header=source.get("source_header", ""),
                problem_id=source["source_id"],
                imports=(
                    imports_from_header(source.get("source_header", ""))
                    or ["Mathlib"]
                ),
            )
            print(f"[{position}/{len(remaining)}] {source['source_id']} starting", flush=True)
            result = pipeline.run(raw_input)
            elapsed = round(time.monotonic() - start, 4)
            nodes = result.blueprint.nodes if result.blueprint is not None else []
            root_result = (
                result.prover_result.root_result
                if result.prover_result is not None
                else None
            )
            attempt_statistics = node_attempt_statistics(nodes, root_result)
            attempt_count = sum(
                item["attempt_count"] for item in attempt_statistics
            )
            base_attempt_count = sum(
                item["base_attempt_count"] for item in attempt_statistics
            )
            repair_attempt_count = sum(
                item["repair_attempt_count"] for item in attempt_statistics
            )
            attempt_proof_token_total = sum(
                item["proof_token_count_total"] for item in attempt_statistics
            )
            attempt_tactic_total = sum(
                item["tactic_count_total"] for item in attempt_statistics
            )
            record = {
                "schema_version": "minif2f_planner_prover_pass4_problem_v3",
                "sample_index": source["sample_index"],
                "source_id": source["source_id"],
                "source_split": source["source_split"],
                "source_informal_statement": source["informal_stmt"],
                "reference_formal_statement": source["reference_formal_statement"],
                "raw_planner_input": raw_input.model_dump(mode="json"),
                "model": MODEL,
                "architecture": args.architecture,
                "blueprint_reasoning_exposed_to_prover": (
                    include_blueprint_reasoning
                ),
                "attempts_per_node": args.pass_k,
                "max_formalization_repair_rounds": (
                    args.max_formalization_repair_rounds
                ),
                "max_prover_repair_rounds": args.max_prover_repair_rounds,
                "max_prover_repair_candidates_per_node": (
                    args.max_prover_repair_candidates_per_node
                ),
                "prover_repair_candidates_per_generation": 1,
                "max_total_graph_nodes": MAX_TOTAL_GRAPH_NODES,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "pipeline_success": result.success,
                "pipeline_stage": result.stage,
                "planner_success": result.planner_result.success,
                "planner_stage": result.planner_result.stage,
                "planner_mode": result.planner_result.planner_mode,
                "lean_native_route_verified": (
                    result.planner_result.planner_mode
                    == ProblemInputKind.LEAN
                    and result.planner_result.classification_attempts == 0
                    and raw_input.formal_statement
                    == source["reference_formal_statement"]
                ),
                "node_count": len(nodes),
                "total_graph_node_count": len(nodes) + 1,
                "planner_generated_subgoal_count": len(nodes),
                "proved_node_count": sum(
                    node.status.value == "proved" for node in nodes
                ),
                "base_proved_node_count": sum(
                    item["base_pass_at_4"]
                    for item in attempt_statistics
                    if not item["is_root"]
                ),
                "repair_rescued_node_count": sum(
                    item["repair_rescued"]
                    for item in attempt_statistics
                    if not item["is_root"]
                ),
                "root_attempted": root_result is not None,
                "root_proved": (
                    root_result is not None
                    and root_result.status.value == "proved"
                    and bool(root_result.verified_proof)
                ),
                "root_base_proved": any(
                    item["is_root"] and item["base_pass_at_4"]
                    for item in attempt_statistics
                ),
                "root_repair_rescued": any(
                    item["is_root"] and item["repair_rescued"]
                    for item in attempt_statistics
                ),
                "root_attempt_count": (
                    len(root_result.proof_attempts)
                    if root_result is not None
                    else 0
                ),
                "node_attempt_statistics": attempt_statistics,
                "attempt_count": attempt_count,
                "base_attempt_count": base_attempt_count,
                "repair_attempt_count": repair_attempt_count,
                "attempt_proof_token_total": attempt_proof_token_total,
                "average_attempt_proof_token_count": (
                    round(attempt_proof_token_total / attempt_count, 4)
                    if attempt_count
                    else None
                ),
                "attempt_tactic_total": attempt_tactic_total,
                "average_attempt_tactic_count": (
                    round(attempt_tactic_total / attempt_count, 4)
                    if attempt_count
                    else None
                ),
                "prover_success": (
                    result.prover_result is not None
                    and result.prover_result.prover_status == ProverDataStatus.SUCCESS
                ),
                "pipeline_result": result.model_dump(mode="json", by_alias=True),
            }
            append_jsonl(results_path, record)
            summarize(results_path, summary_path)
            print(
                f"[{position}/{len(remaining)}] {source['source_id']} "
                f"planner={record['planner_success']} nodes={len(nodes)} "
                f"proved={record['proved_node_count']} "
                f"prover={record['prover_success']} seconds={elapsed}",
                flush=True,
            )
            if failure := system_failure_message(result):
                raise RuntimeError(failure + "; stopping immediately")
            if not record["lean_native_route_verified"]:
                raise RuntimeError(
                    f"system design violation for {source['source_id']}: "
                    "miniF2F row did not use deterministic Lean-native routing"
                )
            for node in nodes:
                if node.preamble.raw_header != source.get("source_header", "").strip():
                    raise RuntimeError(
                        f"system design violation for {source['source_id']}/{node.id}: "
                        "source header was not preserved on the subgoal"
                    )
                if any(
                    token in node.lean_decl.casefold()
                    for token in (":=", " by", "sorry", "admit")
                ):
                    raise RuntimeError(
                        f"system design violation for {source['source_id']}/{node.id}: "
                        "Planner emitted a proof-bearing Lean statement"
                    )
    except Exception as error:
        fatal_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if proof_runtime is not None:
            proof_runtime.close()
        planner_backend.close()

    print(json.dumps(summarize(results_path, summary_path), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
