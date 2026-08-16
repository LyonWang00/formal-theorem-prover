#!/usr/bin/env python3
"""Run the isolated RootFirst experiment on a fixed miniF2F manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from openai import OpenAI

from lean_prover.Planner.client import (
    OpenAICompatibleClient,
    PlannerClientConfig,
)
from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.schemas import RawTheoremInput
from lean_prover.Prover.service import (
    NodeProofVerifier,
    OpenAICompatibleProofGenerator,
)
from lean_prover.Prover.repair_policy import DEFAULT_CONTROLLED_PROOF_BUDGET
from lean_prover.Repair.planner_subproblem import PlannerSubproblemRepairer
from lean_prover.Repair.prover_proof import ProverProofRepairer
from lean_prover.Verify import BlueprintVerifier, JsonlVerifyStore
from lean_prover.lean_training.verification.pantograph import (
    PantographTheoremVerifier,
)

from .adapters import PantographStatementChecker
from .formalization_gate import RootFirstFormalizationGate
from .proving import RootFirstProver
from .refinement import BlueprintRefinement
from .schemas import NodeState, RootFirstRunResult
from .service import RootFirstService


MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

LEAN_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_'.]*|[0-9]+(?:\.[0-9]+)?|:=|=>|<-|<=|>=|!=|[^\s]"
)
TACTIC_RE = re.compile(
    r"\b(intro|intros|rintro|exact|refine|apply|rw|rfl|simp|simpa|norm_num|"
    r"linarith|nlinarith|omega|ring|ring_nf|field_simp|positivity|aesop|"
    r"constructor|left|right|use|obtain|have|show|change|unfold|induction|"
    r"cases|rcases|by_cases|contradiction|exfalso|assumption|decide|native_decide)\b"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_progress(path: Path, **payload: object) -> None:
    append_jsonl(
        path,
        {
            "created_monotonic": time.monotonic(),
            **payload,
        },
    )


def imports_from_header(header: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(1)
            for line in header.splitlines()
            if (match := re.match(r"^\s*import\s+([^\s]+)", line))
        )
    )


def proof_token_count(attempt) -> int:
    usage = attempt.metadata.get("api_usage") if attempt.metadata else None
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        return int(usage["completion_tokens"])
    return len(LEAN_TOKEN_RE.findall(attempt.proof))


def tactic_count(proof: str) -> int:
    """Stable lexical estimate used only for cross-run comparison."""

    count = len(TACTIC_RE.findall(proof))
    return count or int(bool(proof.strip()))


def node_attempt_statistics(node) -> dict[str, object]:
    proof_attempts = [
        attempt
        for attempt in node.attempts
        if attempt.kind in {"proof", "proof_repair", "generation_error"}
    ]
    repair_attempts = [
        attempt for attempt in proof_attempts if attempt.kind == "proof_repair"
    ]
    # New records carry proof_invocation explicitly. For pre-field records,
    # recover invocation boundaries deterministically: every new base attempt
    # with attempt_index=1 starts another node-level pass@k invocation.
    invocation = 0
    invocation_by_attempt_id: dict[str, int] = {}
    for attempt in proof_attempts:
        if attempt.proof_invocation:
            invocation = attempt.proof_invocation
        elif attempt.kind != "proof_repair" and attempt.attempt_index == 1:
            invocation += 1
        invocation_by_attempt_id[attempt.attempt_id] = max(1, invocation)
    repair_rounds = {
        (
            invocation_by_attempt_id[attempt.attempt_id],
            attempt.attempt_index,
            attempt.repair_round,
        )
        for attempt in repair_attempts
        if attempt.repair_round is not None
    }
    first_success = next(
        (index for index, attempt in enumerate(proof_attempts) if attempt.success),
        None,
    )
    attempts_before_success = (
        proof_attempts[: first_success + 1]
        if first_success is not None
        else proof_attempts
    )
    frozen_count = node.metadata.get("attempt_count_at_freeze")
    post_freeze_attempt_count = (
        max(0, len(node.attempts) - int(frozen_count))
        if frozen_count is not None
        else 0
    )
    token_counts = [proof_token_count(attempt) for attempt in proof_attempts]
    tactic_counts = [tactic_count(attempt.proof) for attempt in proof_attempts]
    return {
        "node_id": node.id,
        "state": node.state.value,
        "proof_invocation_count": int(
            node.metadata.get("proof_invocation_count", 0)
        ),
        "attempt_count": len(proof_attempts),
        "repair_candidate_count": len(repair_attempts),
        "repair_round_count": len(repair_rounds),
        "repair_candidates_before_success": sum(
            attempt.kind == "proof_repair"
            for attempt in attempts_before_success
        ),
        "average_attempt_proof_token_count": (
            sum(token_counts) / len(token_counts) if token_counts else 0.0
        ),
        "average_attempt_tactic_count": (
            sum(tactic_counts) / len(tactic_counts) if tactic_counts else 0.0
        ),
        "attempt_count_at_freeze": frozen_count,
        "post_freeze_attempt_count": post_freeze_attempt_count,
        "frozen_reprove_violation": post_freeze_attempt_count > 0,
    }


def backfill_proof_invocations(result: RootFirstRunResult) -> RootFirstRunResult:
    """Upgrade records written before proof_invocation became explicit."""

    upgraded = result.model_copy(deep=True)
    for node in upgraded.blueprint.nodes:
        invocation = 0
        for attempt in node.attempts:
            if attempt.proof_invocation:
                invocation = attempt.proof_invocation
            elif attempt.kind not in {"proof_repair", "disproof_repair"}:
                if attempt.attempt_index == 1:
                    invocation += 1
                attempt.proof_invocation = max(1, invocation)
            else:
                attempt.proof_invocation = max(1, invocation)
    return upgraded


def refresh_result_record(record: dict[str, Any]) -> dict[str, object]:
    result = backfill_proof_invocations(
        RootFirstRunResult.model_validate(record["result"])
    )
    refreshed = result_record(result, float(record["elapsed_seconds"]))
    for key, value in record.items():
        if key not in refreshed and key != "result":
            refreshed[key] = value
    return refreshed


def result_record(result: RootFirstRunResult, elapsed: float) -> dict[str, object]:
    nodes = result.blueprint.nodes
    effectiveness = list(
        result.blueprint.metadata.get("refinement_effectiveness", [])
    )
    attempts = [attempt for node in nodes for attempt in node.attempts]
    base_attempts = [
        attempt for attempt in attempts if attempt.kind in {"proof", "disproof"}
    ]
    repair_attempts = [
        attempt
        for attempt in attempts
        if attempt.kind in {"proof_repair", "disproof_repair"}
    ]
    node_stats = [node_attempt_statistics(node) for node in nodes]
    proof_attempts = [
        attempt
        for node in nodes
        for attempt in node.attempts
        if attempt.kind in {"proof", "proof_repair", "generation_error"}
    ]
    proof_repair_attempts = [
        attempt for attempt in proof_attempts if attempt.kind == "proof_repair"
    ]
    proof_invocations = sum(
        int(row["proof_invocation_count"]) for row in node_stats
    )
    proof_repair_rounds = sum(
        int(row["repair_round_count"]) for row in node_stats
    )
    token_counts = [proof_token_count(attempt) for attempt in proof_attempts]
    tactic_counts = [tactic_count(attempt.proof) for attempt in proof_attempts]
    proved_stats = [
        row for row in node_stats if row["state"] == NodeState.SUCCESS.value
    ]
    return {
        "schema_version": "root_first_minif2f_v2",
        "problem_id": result.problem_id,
        "success": result.success,
        "stage": result.stage,
        "elapsed_seconds": round(elapsed, 4),
        "node_count_including_l0": len(nodes),
        "helper_node_count": len(nodes) - 1,
        "effective_helper_count": sum(
            bool(item.get("effective_refinement"))
            for item in effectiveness
            if isinstance(item, dict)
        ),
        "refinement_effectiveness": effectiveness,
        "refinement_round": result.blueprint.refinement_round,
        "nodes_added": result.blueprint.nodes_added,
        "total_attempt_count": len(attempts),
        "base_attempt_count": len(base_attempts),
        "repair_attempt_count": len(repair_attempts),
        "proof_attempt_count": len(proof_attempts),
        "proof_repair_attempt_count": len(proof_repair_attempts),
        "proof_repair_round_count": proof_repair_rounds,
        "proof_invocation_count": proof_invocations,
        "average_repair_candidate_count_per_proof_invocation": (
            len(proof_repair_attempts) / proof_invocations
            if proof_invocations
            else 0.0
        ),
        "average_repair_candidate_count_per_base_attempt": (
            len(proof_repair_attempts) / len(
                [a for a in proof_attempts if a.kind in {"proof", "generation_error"}]
            )
            if any(
                a.kind in {"proof", "generation_error"} for a in proof_attempts
            )
            else 0.0
        ),
        "average_repair_round_count_per_proof_invocation": (
            proof_repair_rounds / proof_invocations
            if proof_invocations
            else 0.0
        ),
        "average_repairs_before_success_for_proved_nodes": (
            sum(int(row["repair_candidates_before_success"]) for row in proved_stats)
            / len(proved_stats)
            if proved_stats
            else 0.0
        ),
        "average_attempt_proof_token_count": (
            sum(token_counts) / len(token_counts) if token_counts else 0.0
        ),
        "average_attempt_tactic_count": (
            sum(tactic_counts) / len(tactic_counts) if tactic_counts else 0.0
        ),
        "frozen_reprove_violation_count": sum(
            bool(row["frozen_reprove_violation"]) for row in node_stats
        ),
        "node_attempt_statistics": node_stats,
        "direct_root_success": result.stage == "root_direct_success",
        "proved_node_count": sum(node.state == NodeState.SUCCESS for node in nodes),
        "disproved_node_count": sum(node.state == NodeState.DISPROVED for node in nodes),
        "node_states": {node.id: node.state.value for node in nodes},
        "result": result.model_dump(mode="json"),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, object]:
    records = [refresh_result_record(row) for row in records]
    total = len(records)
    success = sum(bool(row["success"]) for row in records)
    direct = sum(bool(row["direct_root_success"]) for row in records)
    attempt_count = sum(int(row["proof_attempt_count"]) for row in records)
    proof_invocations = sum(int(row["proof_invocation_count"]) for row in records)
    repair_rounds = sum(int(row["proof_repair_round_count"]) for row in records)
    repair_candidates = sum(
        int(row["proof_repair_attempt_count"]) for row in records
    )
    proved_node_stats = [
        node
        for row in records
        for node in row["node_attempt_statistics"]
        if node["state"] == NodeState.SUCCESS.value
    ]
    proof_base_attempts = sum(
        sum(
            int(node["attempt_count"]) - int(node["repair_candidate_count"])
            for node in row["node_attempt_statistics"]
        )
        for row in records
    )
    weighted_token_total = sum(
        float(row["average_attempt_proof_token_count"])
        * int(row["proof_attempt_count"])
        for row in records
    )
    weighted_tactic_total = sum(
        float(row["average_attempt_tactic_count"])
        * int(row["proof_attempt_count"])
        for row in records
    )
    return {
        "schema_version": "root_first_minif2f_summary_v2",
        "pass_k": records[0].get("pass_k") if records else None,
        "proof_repair_rounds_per_node": (
            records[0].get("proof_repair_rounds_per_node") if records else None
        ),
        "proof_repair_candidates_per_node": (
            records[0].get("proof_repair_candidates_per_node") if records else None
        ),
        "proof_repair_candidates_per_generation": 1,
        "formalization_repair_rounds": (
            records[0].get("formalization_repair_rounds") if records else None
        ),
        "max_total_graph_nodes": 4,
        "processed": total,
        "success_count": success,
        "success_rate": success / total if total else 0.0,
        "direct_root_success_count": direct,
        "direct_root_success_rate": direct / total if total else 0.0,
        "refined_success_count": sum(
            row["stage"] == "refined_success" for row in records
        ),
        "stage_distribution": dict(Counter(str(row["stage"]) for row in records)),
        "average_helper_node_count": (
            sum(int(row["helper_node_count"]) for row in records) / total
            if total
            else 0.0
        ),
        "effective_helper_count": sum(
            int(row.get("effective_helper_count", 0)) for row in records
        ),
        "helper_effectiveness_rate": (
            sum(int(row.get("effective_helper_count", 0)) for row in records)
            / sum(int(row["helper_node_count"]) for row in records)
            if sum(int(row["helper_node_count"]) for row in records)
            else 0.0
        ),
        "total_attempt_count": sum(
            int(row["total_attempt_count"]) for row in records
        ),
        "repair_attempt_count": sum(
            int(row["repair_attempt_count"]) for row in records
        ),
        "proof_repair_round_count": repair_rounds,
        "average_repair_candidate_count_per_proof_invocation": (
            repair_candidates / proof_invocations if proof_invocations else 0.0
        ),
        "average_repair_candidate_count_per_base_attempt": (
            repair_candidates / proof_base_attempts if proof_base_attempts else 0.0
        ),
        "average_repair_round_count_per_proof_invocation": (
            repair_rounds / proof_invocations if proof_invocations else 0.0
        ),
        "average_repairs_before_success_for_proved_nodes": (
            sum(
                int(node["repair_candidates_before_success"])
                for node in proved_node_stats
            )
            / len(proved_node_stats)
            if proved_node_stats
            else 0.0
        ),
        "average_attempt_proof_token_count": (
            weighted_token_total / attempt_count if attempt_count else 0.0
        ),
        "average_attempt_tactic_count": (
            weighted_tactic_total / attempt_count if attempt_count else 0.0
        ),
        "frozen_reprove_violation_count": sum(
            int(row["frozen_reprove_violation_count"]) for row in records
        ),
        "per_problem_elapsed_seconds": {
            str(row["problem_id"]): float(row["elapsed_seconds"])
            for row in records
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--pass-k",
        type=int,
        default=DEFAULT_CONTROLLED_PROOF_BUDGET.pass_k,
    )
    parser.add_argument(
        "--proof-repair-rounds",
        type=int,
        default=DEFAULT_CONTROLLED_PROOF_BUDGET.repair_rounds_per_node,
    )
    parser.add_argument(
        "--proof-repair-candidates-per-node",
        type=int,
        default=DEFAULT_CONTROLLED_PROOF_BUDGET.repair_candidates_per_node,
    )
    parser.add_argument("--formalization-repair-rounds", type=int, default=2)
    args = parser.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    if args.count < 1:
        raise ValueError("count must be positive")
    for name in (
        "pass_k",
        "proof_repair_rounds",
        "proof_repair_candidates_per_node",
        "formalization_repair_rounds",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")

    reference = load_jsonl(args.reference_manifest)
    if args.source_id:
        by_id = {str(row["source_id"]): row for row in reference}
        missing = [item for item in args.source_id if item not in by_id]
        if missing:
            raise ValueError(f"source IDs missing from manifest: {missing}")
        selected = [by_id[item] for item in args.source_id]
    else:
        selected = reference[: args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.jsonl"
    progress_path = args.output_dir / "progress.jsonl"
    completed = {
        str(row["problem_id"])
        for row in load_jsonl(result_path)
    } if result_path.is_file() else set()

    api_key = os.environ["DEEPSEEK_API_KEY"]
    planner_client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=BASE_URL,
        model=MODEL,
        temperature=0.2,
        max_tokens=8192,
    )
    openai_client = OpenAI(api_key=api_key, base_url=BASE_URL)
    generator = OpenAICompatibleProofGenerator(
        client=openai_client,
        model=MODEL,
        temperature=0.2,
        max_tokens=2048,
    )
    declaration_backend = PantographDeclarationCheckingBackend(
        project_path=args.lean_project,
        timeout=args.timeout,
    )
    proof_runtime = PantographTheoremVerifier(
        args.lean_project,
        imports=("Mathlib",),
        timeout=args.timeout,
    )
    environment = declaration_backend.environment_identity(["Mathlib"])
    lean_checker = BlueprintLeanChecker(declaration_backend)
    formalization_gate = RootFirstFormalizationGate(
        verifier=BlueprintVerifier(
            planner_client,
            store=JsonlVerifyStore(
                args.output_dir / "refinement_verify_results.jsonl"
            ),
            max_workers=1,
            max_semantic_repairs=args.formalization_repair_rounds,
        ),
        lean_checker=lean_checker,
        repairer=PlannerSubproblemRepairer(
            planner_client,
            lean_checker=lean_checker,
        ),
        max_repair_rounds=args.formalization_repair_rounds,
    )
    service = RootFirstService(
        prover=RootFirstProver(
            generator=generator,
            verifier=NodeProofVerifier(proof_runtime),
            repairer=ProverProofRepairer(
                client=openai_client,
                model=MODEL,
                project_path=args.lean_project,
                temperature=0.0,
                max_tokens=2048,
            ),
            attempts_per_node=args.pass_k,
            max_repair_rounds=args.proof_repair_rounds,
            max_repair_candidates_per_node=(
                args.proof_repair_candidates_per_node
            ),
            complete_pass_batch=True,
            attempt_disproof=False,
        ),
        refinement=BlueprintRefinement(
            client=planner_client,
            statement_checker=PantographStatementChecker(declaration_backend),
            formalization_gate=formalization_gate,
        ),
        environment=environment,
        max_refinement_rounds=3,
    )
    try:
        for index, source in enumerate(selected, start=1):
            source_id = str(source["source_id"])
            if source_id in completed:
                continue
            header = str(source.get("source_header") or "")
            raw = RawTheoremInput(
                input_text=str(source["reference_formal_statement"]),
                formal_statement=str(source["reference_formal_statement"]),
                informal_stmt=str(source.get("informal_stmt") or ""),
                header=header,
                problem_id=source_id,
                imports=imports_from_header(header) or ["Mathlib"],
            )
            print(f"[{index}/{len(selected)}] {source_id} root-first", flush=True)
            start = time.monotonic()
            write_progress(
                progress_path,
                problem_id=source_id,
                event="problem_started",
            )
            result = service.run_input(raw)
            record = result_record(result, time.monotonic() - start)
            record["source_split"] = source.get("source_split")
            record["pass_k"] = args.pass_k
            record["proof_repair_rounds_per_node"] = args.proof_repair_rounds
            record["proof_repair_candidates_per_node"] = (
                args.proof_repair_candidates_per_node
            )
            record["proof_repair_candidates_per_generation"] = 1
            record["formalization_repair_rounds"] = (
                args.formalization_repair_rounds
            )
            record["max_total_graph_nodes"] = 4
            append_jsonl(result_path, record)
            write_progress(
                progress_path,
                problem_id=source_id,
                event="problem_completed",
                stage=result.stage,
                success=result.success,
                helper_nodes=result.blueprint.nodes_added,
                attempt_count=record["total_attempt_count"],
            )
            current = load_jsonl(result_path)
            (args.output_dir / "summary.json").write_text(
                json.dumps(summarize(current), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"  stage={result.stage} success={result.success} "
                f"helpers={result.blueprint.nodes_added} "
                f"attempts={record['total_attempt_count']}",
                flush=True,
            )
    finally:
        proof_runtime.close()
        declaration_backend.close()
    print(
        json.dumps(summarize(load_jsonl(result_path)), ensure_ascii=False, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
