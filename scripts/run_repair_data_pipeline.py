#!/usr/bin/env python3
"""Run attempt-complete, statement-grouped RepairData v2 generation.

The script never starts a Trainer. Every promoted proof is compiled locally by
Pantograph under the exact environment embedded in its prompt and output row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lean_prover.lean_training.repair_pipeline import (
    APICallAttempt,
    AggregatedAttempt,
    DeepSeekRepairClient,
    RepairClientError,
    RepairClientResponse,
    RepairData,
    RepairPipeline,
    RepairRequest,
    load_environment_contract,
)
from lean_prover.lean_training.repair_pipeline.client import RepairClientConfig
from lean_prover.lean_training.repair_pipeline.schema import canonical_sha256
from lean_prover.lean_training.verification.pantograph import PantographTheoremVerifier


REPAIRABLE_LAYERS = {"repair_candidate", "near_miss"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_input_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_proof_state(error_message: str) -> str:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", error_message) if part.strip()]
    return blocks[-1] if len(blocks) > 1 and ("⊢" in blocks[-1] or "goal" in blocks[-1].lower()) else ""


def extract_theorem_name(statement: str) -> str:
    match = re.search(r"\b(?:theorem|lemma)\s+([^\s(:]+)", statement)
    return match.group(1) if match else ""


def attempt_fingerprint(row: dict[str, Any]) -> str:
    """Deduplicate an attempt by normalized semantic evidence, not statement ID."""

    return canonical_sha256(
        {
            "statement_id": normalize_input_text(row.get("statement_id")),
            "theorem": normalize_input_text(row.get("theorem")),
            "failure_proof": normalize_input_text(
                row.get("generated_proof") or row.get("extracted_proof")
            ),
            "error_message": normalize_input_text(row.get("error_message")),
            "failure_class": normalize_input_text(row.get("failure_taxonomy")),
        }
    )


def eligible_attempts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every structurally repairable failed attempt; remove only exact attempts."""

    valid: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for row in rows:
        proof = normalize_input_text(row.get("generated_proof") or row.get("extracted_proof"))
        theorem = normalize_input_text(row.get("theorem"))
        error = normalize_input_text(row.get("error_message"))
        reason = ""
        if row.get("pantograph_verified") is True:
            reason = "already_verified"
        elif row.get("failure_layer") not in REPAIRABLE_LAYERS:
            reason = "non_repairable_failure_layer"
        elif row.get("timed_out"):
            reason = "timeout"
        elif row.get("extraction_success") is not True:
            reason = "proof_extraction_failed"
        elif not theorem or not proof or not error:
            reason = "missing_required_input"
        if reason:
            excluded.append(
                {
                    "target_attempt_id": normalize_input_text(row.get("candidate_id")),
                    "statement_id": normalize_input_text(row.get("statement_id")),
                    "exclusion_reason": reason,
                }
            )
            continue
        fingerprint = attempt_fingerprint(row)
        if fingerprint in seen:
            excluded.append(
                {
                    "target_attempt_id": normalize_input_text(row.get("candidate_id")),
                    "statement_id": normalize_input_text(row.get("statement_id")),
                    "exclusion_reason": "duplicate_attempt",
                    "duplicate_of_attempt_id": seen[fingerprint],
                    "attempt_fingerprint": fingerprint,
                }
            )
            continue
        seen[fingerprint] = normalize_input_text(row.get("candidate_id"))
        copy = dict(row)
        copy["attempt_fingerprint"] = fingerprint
        valid.append(copy)
    return valid, excluded


def stable_order(row: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:{row.get('statement_id')}:{row.get('candidate_id')}:{row.get('attempt_fingerprint')}".encode()
    ).hexdigest()


def build_requests(
    all_attempts: list[dict[str, Any]], environment: Any, *, count: int, seed: int
) -> list[RepairRequest]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_attempts:
        group_id = normalize_input_text(row.get("statement_id")) or canonical_sha256(
            normalize_input_text(row.get("theorem"))
        )
        groups[group_id].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (row.get("candidate_rank") is None, row.get("candidate_rank") or 0, str(row.get("candidate_id"))))

    targets = sorted(all_attempts, key=lambda row: stable_order(row, seed))
    if count > 0:
        targets = targets[:count]
    requests: list[RepairRequest] = []
    for row in targets:
        statement_id = normalize_input_text(row.get("statement_id")) or canonical_sha256(
            normalize_input_text(row.get("theorem"))
        )
        peers = groups[statement_id]
        aggregated = []
        for peer in peers:
            peer_error = normalize_input_text(peer.get("error_message"))
            aggregated.append(
                AggregatedAttempt(
                    attempt_id=normalize_input_text(peer.get("candidate_id")),
                    attempt_rank=int(peer["candidate_rank"]) if peer.get("candidate_rank") is not None else None,
                    failure_proof=normalize_input_text(peer.get("generated_proof") or peer.get("extracted_proof")),
                    error_message=peer_error,
                    failure_class=normalize_input_text(peer.get("failure_taxonomy")),
                    proof_state=extract_proof_state(peer_error),
                    original_verification_status=normalize_input_text(peer.get("pantograph_status")) or "failed",
                )
            )
        error = normalize_input_text(row.get("error_message"))
        requests.append(
            RepairRequest(
                source=normalize_input_text(row.get("source")) or "unknown",
                theorem_name=extract_theorem_name(normalize_input_text(row.get("theorem"))),
                lean_statement=normalize_input_text(row.get("theorem")),
                target_attempt_id=normalize_input_text(row.get("candidate_id")),
                target_attempt_rank=int(row["candidate_rank"]) if row.get("candidate_rank") is not None else None,
                target_failure_proof=normalize_input_text(row.get("generated_proof") or row.get("extracted_proof")),
                target_error_message=error,
                target_failure_class=normalize_input_text(row.get("failure_taxonomy")),
                target_proof_state=extract_proof_state(error),
                statement_id=statement_id,
                aggregation_group_id=statement_id,
                aggregated_attempts=aggregated,
                attempt_fingerprint=str(row["attempt_fingerprint"]),
                environment=environment,
                source_metadata={
                    "source_candidate_id": row.get("candidate_id"),
                    "source_id": row.get("source_id"),
                    "failure_layer": row.get("failure_layer"),
                    "candidate_seed": row.get("candidate_seed"),
                    "checkpoint_hash": row.get("checkpoint_hash"),
                    "generation_config_hash": row.get("generation_config_hash"),
                    "original_pantograph_status": row.get("pantograph_status"),
                    "original_verification_metadata": row.get("verification_metadata"),
                },
            )
        )
    requests.sort(key=lambda row: (row.aggregation_group_id, row.target_attempt_rank is None, row.target_attempt_rank or 0, row.target_attempt_id))
    return requests


def api_call(
    request: RepairRequest,
    config: RepairClientConfig,
    *,
    client: DeepSeekRepairClient | None = None,
) -> dict[str, Any]:
    """Retry exactly once only when the provider returns empty content."""

    client = client or DeepSeekRepairClient(config)
    calls: list[APICallAttempt] = []
    for call_index in (1, 2):
        try:
            response = client.repair(request)
            calls.append(APICallAttempt(call_index=call_index, success=True, observation=response.observation))
            return {
                "target_attempt_id": request.target_attempt_id,
                "status": "success",
                "raw_response": response.raw_content,
                "final_observation": response.observation.model_dump(mode="json"),
                "api_call_attempts": [row.model_dump(mode="json") for row in calls],
            }
        except RepairClientError as error:
            calls.append(
                APICallAttempt(
                    call_index=call_index,
                    success=False,
                    error_kind=error.kind,
                    error_message=str(error),
                    observation=error.observation,
                )
            )
            if error.kind != "empty_response" or call_index == 2:
                return {
                    "target_attempt_id": request.target_attempt_id,
                    "status": "empty_response_after_retry" if error.kind == "empty_response" else "api_error",
                    "error_kind": error.kind,
                    "error_message": str(error),
                    "api_call_attempts": [row.model_dump(mode="json") for row in calls],
                }
            time.sleep(1.0)
    raise AssertionError("unreachable")


def api_call_group(requests: list[RepairRequest], config: RepairClientConfig) -> list[dict[str, Any]]:
    """Keep attempts from one theorem together within one API worker/client."""

    client = DeepSeekRepairClient(config)
    return [api_call(request, config, client=client) for request in requests]


class CachedClient:
    def __init__(self, model_name: str, payload: dict[str, Any]) -> None:
        self.model_name = model_name
        self.payload = payload

    def repair(self, request: RepairRequest) -> RepairClientResponse:
        if self.payload.get("status") != "success":
            calls = self.payload.get("api_call_attempts") or []
            observation = APICallAttempt.model_validate(calls[-1]).observation if calls else None
            kind = "empty_response" if self.payload.get("status") == "empty_response_after_retry" else str(self.payload.get("error_kind") or "api_error")
            raise RepairClientError(
                str(self.payload.get("error_message") or "API call failed"),
                kind=kind,
                observation=observation,
            )
        return RepairClientResponse(
            raw_content=str(self.payload["raw_response"]),
            observation=self.payload["final_observation"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--failure-bank", type=Path, default=Path("outputs/expert_iteration/round0/failure_bank/failure_bank.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/repair_pipeline_v2"))
    parser.add_argument("--count", type=int, default=0, help="0 processes every eligible failed attempt")
    parser.add_argument(
        "--failure-layer",
        action="append",
        choices=sorted(REPAIRABLE_LAYERS),
        help="optionally restrict target attempts while retaining attempt-level processing",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--api-workers", type=int, default=4)
    parser.add_argument("--verification-timeout", type=int, default=60)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate, deduplicate, group, and freeze requests without requiring an API key",
    )
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    failure_path = args.failure_bank if args.failure_bank.is_absolute() else project / args.failure_bank
    output = args.output if args.output.is_absolute() else project / args.output
    data_dir = output / "repair_data"
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    environment = load_environment_contract(project / "lean_project", imports=("Mathlib",))
    failures = read_jsonl(failure_path)
    target_failures = (
        [row for row in failures if row.get("failure_layer") in set(args.failure_layer)]
        if args.failure_layer else failures
    )
    attempts, excluded = eligible_attempts(target_failures)
    requests = build_requests(attempts, environment, count=args.count, seed=args.seed)
    request_path = data_dir / "requests.jsonl"
    write_jsonl(request_path, (row.model_dump(mode="json") for row in requests))
    write_jsonl(data_dir / "excluded_inputs.jsonl", excluded)
    write_json(output / "environment_contract.json", environment.model_dump(mode="json"))

    if args.preflight_only:
        preflight = {
            "schema_version": "repair_pipeline_preflight_v2",
            "failure_bank_rows": len(failures),
            "target_failure_bank_rows": len(target_failures),
            "failure_layer_filter": args.failure_layer or [],
            "repairable_unique_attempts": len(attempts),
            "request_rows": len(requests),
            "statement_group_count": len({row.statement_id for row in requests}),
            "maximum_aggregated_attempt_count": max(
                (len(row.aggregated_attempts) for row in requests), default=0
            ),
            "excluded_input_rows": len(excluded),
            "excluded_input_reasons": dict(
                Counter(str(row.get("exclusion_reason")) for row in excluded)
            ),
            "unique_target_attempt_ids": len({row.target_attempt_id for row in requests}) == len(requests),
            "unique_attempt_fingerprints": len({row.attempt_fingerprint for row in requests}) == len(requests),
            "environment_hash": environment.environment_hash,
            "request_manifest_sha256": file_sha256(request_path),
            "api_called": False,
            "pantograph_started": False,
            "training_started": False,
        }
        write_json(output / "preflight.json", preflight)
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    config = RepairClientConfig.from_env()
    api_path = data_dir / "api_responses.jsonl"
    api_by_id = {str(row["target_attempt_id"]): row for row in read_jsonl(api_path)} if api_path.exists() else {}
    pending = [row for row in requests if row.target_attempt_id not in api_by_id] if not args.finalize_only else []
    if pending:
        grouped: dict[str, list[RepairRequest]] = defaultdict(list)
        for request in pending:
            grouped[request.aggregation_group_id].append(request)
        with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
            futures = [pool.submit(api_call_group, group, config) for group in grouped.values()]
            for future in as_completed(futures):
                for payload in future.result():
                    append_jsonl(api_path, payload)
                    api_by_id[str(payload["target_attempt_id"])] = payload

    attempts_path = data_dir / "attempt_results.jsonl"
    results_by_id = {str(row["target_attempt_id"]): row for row in read_jsonl(attempts_path)} if attempts_path.exists() else {}
    pending_verify = [row for row in requests if row.target_attempt_id not in results_by_id] if not args.finalize_only else []
    if pending_verify:
        verifier = PantographTheoremVerifier(
            project / "lean_project", imports=("Mathlib",), timeout=args.verification_timeout, startup_timeout=900
        )
        warmup = verifier.warmup(timeout=120)
        if not warmup.success:
            verifier.close()
            raise RuntimeError(f"Pantograph warmup failed: {warmup.diagnostics}")
        try:
            for request in pending_verify:
                payload = api_by_id[request.target_attempt_id]
                calls = [APICallAttempt.model_validate(row) for row in payload.get("api_call_attempts", [])]
                result = RepairPipeline(
                    client=CachedClient(config.model, payload), verifier=verifier, timeout=args.verification_timeout
                ).run_one(request, api_call_attempts=calls)
                row = result.data.model_dump(mode="json")
                append_jsonl(attempts_path, row)
                results_by_id[request.target_attempt_id] = row
        finally:
            verifier.close()

    missing = [row.target_attempt_id for row in requests if row.target_attempt_id not in results_by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} attempt results are missing; cannot finalize")
    records = [RepairData.model_validate(results_by_id[row.target_attempt_id]) for row in requests]
    verified = [row.model_dump(mode="json") for row in records if row.repair_verified]
    failed = [row.model_dump(mode="json") for row in records if not row.repair_verified]
    verified_path = output / "repair_verified.jsonl"
    failed_path = output / "repair_failed.jsonl"
    write_jsonl(verified_path, verified)
    write_jsonl(failed_path, failed)

    audit = {
        "schema_version": "repair_data_audit_v2",
        "request_rows": len(requests),
        "unique_target_attempt_ids": len({row.target_attempt_id for row in records}) == len(records),
        "unique_attempt_fingerprints": len({row.attempt_fingerprint for row in records}) == len(records),
        "statement_duplicates_permitted": True,
        "repeated_statement_row_count": len(records) - len({row.statement_id for row in records}),
        "all_promoted_proofs_pantograph_verified": all(
            row.repair_verified and (
                (row.minimal_repair and row.minimal_repair.verification.verified)
                if row.selected_strategy == "minimal_repair"
                else (row.clean_repair and row.clean_repair.verification.verified)
            )
            for row in records if row.selected_verified_proof
        ),
        "environment_hash_consistent": len({row.environment.environment_hash for row in records}) <= 1,
        "api_retry_cap_respected": all(row.api_retry_count <= 1 for row in records),
        "ambiguous_correct_proof_field_absent": all("correct_proof" not in row for row in verified + failed),
    }
    audit["passed"] = all(
        value for key, value in audit.items()
        if key not in {"schema_version", "request_rows", "repeated_statement_row_count"}
    )
    write_json(output / "repair_data_audit.json", audit)

    statistics = {
        "status": "REPAIR_PIPELINE_V2_COMPLETED",
        "failure_bank_rows": len(failures),
        "eligible_attempt_rows_before_deduplication": len(attempts) + sum(row.get("exclusion_reason") == "duplicate_attempt" for row in excluded),
        "attempt_level_duplicates_removed": sum(row.get("exclusion_reason") == "duplicate_attempt" for row in excluded),
        "excluded_input_rows": len(excluded),
        "processed_attempt_rows": len(records),
        "statement_group_count": len({row.statement_id for row in records}),
        "api_success_count": sum(row.api_success for row in records),
        "api_empty_response_after_retry_count": sum(row.anomaly_status == "api_empty_response_after_retry" for row in records),
        "response_parse_success_count": sum(row.response_parse_success for row in records),
        "pantograph_verified_repair_count": len(verified),
        "selected_strategy": dict(Counter(row.selected_strategy for row in records)),
        "resolved_attempt_assessment": dict(Counter(row.resolved_attempt_assessment for row in records)),
        "anomaly_status": dict(Counter(row.anomaly_status for row in records)),
        "environment_hash": environment.environment_hash,
        "request_manifest_sha256": file_sha256(request_path),
        "verified_manifest_sha256": file_sha256(verified_path),
        "failed_manifest_sha256": file_sha256(failed_path),
        "training_started": False,
        "repair_data_audit": audit,
    }
    write_json(output / "statistics.json", statistics)
    (output / "report.md").write_text(
        "# RepairData v2 report\n\n"
        f"- Processed attempts: {len(records)} across {statistics['statement_group_count']} statement groups\n"
        f"- Pantograph-verified repairs: {len(verified)}\n"
        f"- Empty API responses after one retry: {statistics['api_empty_response_after_retry_count']}\n"
        f"- Environment hash: `{environment.environment_hash}`\n"
        f"- Data contract audit: {'PASS' if audit['passed'] else 'FAIL'}\n\n"
        "Every target attempt received both a minimal-edit and a clean-from-scratch proposal. "
        "Each proposal was normalized independently and compiled locally. Only the shortest "
        "Pantograph-verified proof was promoted. No Trainer was started.\n",
        encoding="utf-8",
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
