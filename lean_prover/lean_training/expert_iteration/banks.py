"""Idempotent Proof Bank and Failure Bank storage."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from lean_prover.lean_training.data.contracts import make_attestation_id

from .schemas import (
    DataRole,
    FailureBankRecord,
    GenerationRecord,
    ProofBankRecord,
    StatementRecord,
    VerificationRecord,
    normalized_proof,
    stable_hash,
)
from .utils import append_jsonl, iter_jsonl, read_jsonl, write_jsonl_atomic


class ProofBank:
    def __init__(
        self,
        path: str | Path,
        *,
        max_per_statement: int = 3,
        max_training_per_statement: int = 1,
        selection_strategy: str = "random_verified",
        seed: int = 42,
    ) -> None:
        self.path = Path(path)
        self.max_per_statement = max_per_statement
        self.max_training_per_statement = max_training_per_statement
        self.selection_strategy = selection_strategy
        self.seed = seed
        self.records = [ProofBankRecord.model_validate(row) for row in read_jsonl(self.path)]
        self._mark_environment_changes()

    def _mark_environment_changes(self) -> None:
        environments = {record.environment_hash for record in self.records}
        if len(environments) > 1:
            for record in self.records:
                record.needs_reverification = True

    def mark_for_environment(self, environment_hash: str) -> int:
        """Mark historical proofs verified in a different Lean environment."""

        changed = 0
        statement_ids: set[str] = set()
        for record in self.records:
            needs_reverification = record.environment_hash != environment_hash
            if record.needs_reverification != needs_reverification:
                record.needs_reverification = needs_reverification
                changed += 1
            statement_ids.add(record.statement_id)
        for statement_id in statement_ids:
            self.select_primaries(statement_id)
        if changed:
            self.flush()
        return changed

    def insert_verified(
        self,
        statement: StatementRecord,
        generation: GenerationRecord,
        verification: VerificationRecord,
    ) -> ProofBankRecord | None:
        if (
            statement.data_role is not DataRole.DISCOVERY
            or generation.data_role is not DataRole.DISCOVERY
            or verification.data_role is not DataRole.DISCOVERY
        ):
            raise ValueError(f"Proof Bank accepts discovery only: {statement.statement_id}")
        if generation.statement_id != statement.statement_id or verification.statement_id != statement.statement_id:
            raise ValueError("Proof Bank statement IDs do not match")
        if verification.generation_id != generation.generation_id:
            raise ValueError("Proof Bank generation IDs do not match")
        if generation.iteration is None or verification.iteration != generation.iteration:
            raise ValueError("Proof Bank requires a matching discovery iteration")
        if not verification.verified:
            raise ValueError(f"Proof Bank requires verified proof: {generation.generation_id}")
        proof = (generation.extracted_proof or "").strip()
        if not proof:
            raise ValueError(f"verified generation has no extracted proof: {generation.generation_id}")
        proof_key = stable_hash(statement.statement_id, normalized_proof(proof))
        proof_id = f"proof_{proof_key[:24]}"
        verification_attestation = _verification_attestation(
            proof_id=proof_id,
            verification=verification,
        )
        duplicate = next(
            (record for record in self.records if record.proof_id == proof_id),
            None,
        )
        if duplicate is not None:
            if duplicate.metadata.get("verification_attestation") != verification_attestation:
                duplicate.metadata["verification_attestation"] = verification_attestation
                duplicate.metadata["last_reverified_generation_id"] = generation.generation_id
            if (
                duplicate.environment_hash != verification.environment_hash
                or duplicate.needs_reverification
            ):
                duplicate.environment_hash = verification.environment_hash
                duplicate.lean_version = verification.lean_version
                duplicate.mathlib_commit = verification.mathlib_commit
                duplicate.compile_time_ms = verification.compile_time_ms
                duplicate.needs_reverification = False
                duplicate.metadata["last_reverified_generation_id"] = generation.generation_id
                self.select_primaries(statement.statement_id)
            self.flush()
            return None
        existing = [record for record in self.records if record.statement_id == statement.statement_id]
        if len(existing) >= self.max_per_statement:
            return None
        record = ProofBankRecord(
            proof_id=proof_id,
            statement_id=statement.statement_id,
            proof=proof,
            iteration_found=generation.iteration,
            generator_checkpoint=generation.checkpoint,
            generation_id=generation.generation_id,
            proof_tokens=int(generation.metadata.get("completion_tokens") or len(proof.split())),
            proof_chars=len(proof),
            compile_time_ms=verification.compile_time_ms,
            selected_as_primary=False,
            environment_hash=verification.environment_hash,
            lean_version=verification.lean_version,
            mathlib_commit=verification.mathlib_commit,
            metadata={
                "origin_data_role": DataRole.DISCOVERY.value,
                "current_train_eligible": True,
                "statement": statement.model_dump(mode="json", exclude={"reference_proof"}),
                "verification_attestation": verification_attestation,
            },
        )
        self.records.append(record)
        self.select_primaries(statement.statement_id)
        self.flush()
        return record

    def select_primaries(self, statement_id: str) -> list[ProofBankRecord]:
        statement_records = [record for record in self.records if record.statement_id == statement_id]
        for record in statement_records:
            record.selected_as_primary = False
        candidates = [record for record in statement_records if not record.needs_reverification]
        if self.selection_strategy == "shortest":
            ordered = sorted(candidates, key=lambda item: (item.proof_tokens, item.proof_id))
        elif self.selection_strategy == "fastest_compile":
            ordered = sorted(candidates, key=lambda item: (item.compile_time_ms or 10**12, item.proof_id))
        else:
            ordered = list(candidates)
            random.Random(stable_hash(self.seed, statement_id)).shuffle(ordered)
        selected = ordered[: self.max_training_per_statement]
        for record in selected:
            record.selected_as_primary = True
        return selected

    def selected_for_training(self) -> list[ProofBankRecord]:
        return [record for record in self.records if record.selected_as_primary and not record.needs_reverification]

    def flush(self) -> None:
        write_jsonl_atomic(self.path, (record.model_dump(mode="json") for record in self.records))


def _verification_attestation(
    *,
    proof_id: str,
    verification: VerificationRecord,
) -> dict[str, str | None]:
    """Persist the exact successful compile attestation needed by SFT."""

    assembler_version = str(verification.assembler_version or "")
    normalization_version = str(verification.normalization_version or "")
    assembled_source_hash = str(verification.assembled_source_hash or "")
    if not assembler_version or not normalization_version or not assembled_source_hash:
        raise ValueError(
            f"verified proof {proof_id} is missing compile attestation fields"
        )
    return {
        "record_id": proof_id,
        "environment_hash": verification.environment_hash,
        "assembler_version": assembler_version,
        "normalization_version": normalization_version,
        "assembled_source_hash": assembled_source_hash,
        "attestation_id": make_attestation_id(
            record_id=proof_id,
            environment_hash=verification.environment_hash,
            assembler_version=assembler_version,
            normalization_version=normalization_version,
            assembled_source_hash=assembled_source_hash,
        ),
        "lean_version": verification.lean_version,
        "mathlib_commit": verification.mathlib_commit,
        "attested_at": verification.verified_at,
    }


class FailureBank:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._failure_ids = {
            str(row["failure_id"])
            for row in iter_jsonl(self.path)
            if row.get("failure_id")
        }

    @property
    def records(self) -> list[FailureBankRecord]:
        """Compatibility view loaded only when explicitly requested."""

        return [FailureBankRecord.model_validate(row) for row in iter_jsonl(self.path)]

    def __len__(self) -> int:
        return len(self._failure_ids)

    def insert_failed(
        self,
        statement: StatementRecord,
        generation: GenerationRecord,
        verification: VerificationRecord,
    ) -> FailureBankRecord | None:
        if (
            statement.data_role is not DataRole.DISCOVERY
            or generation.data_role is not DataRole.DISCOVERY
            or verification.data_role is not DataRole.DISCOVERY
        ):
            raise ValueError(f"Failure Bank accepts discovery only: {statement.statement_id}")
        if generation.statement_id != statement.statement_id or verification.statement_id != statement.statement_id:
            raise ValueError("Failure Bank statement IDs do not match")
        if verification.generation_id != generation.generation_id:
            raise ValueError("Failure Bank generation IDs do not match")
        if generation.iteration is None or verification.iteration != generation.iteration:
            raise ValueError("Failure Bank requires a matching discovery iteration")
        if verification.verified:
            raise ValueError(f"Failure Bank rejects successful generation: {generation.generation_id}")
        failure_id = f"failure_{stable_hash(generation.generation_id, verification.status)[:24]}"
        if failure_id in self._failure_ids:
            return None
        record = FailureBankRecord(
            failure_id=failure_id,
            generation_id=generation.generation_id,
            statement_id=statement.statement_id,
            iteration=generation.iteration,
            generator_checkpoint=generation.checkpoint,
            raw_output=generation.raw_output,
            extracted_proof=generation.extracted_proof,
            status=verification.status,
            error_type=verification.error_type,
            error_message=verification.error_message,
            first_error_position=verification.first_error_position,
            valid_prefix_length=verification.valid_prefix_length,
            remaining_goals=verification.remaining_goals,
            timed_out=verification.timed_out,
            environment_hash=verification.environment_hash,
            metadata={"origin_data_role": DataRole.DISCOVERY.value},
        )
        append_jsonl(self.path, (record.model_dump(mode="json"),))
        self._failure_ids.add(failure_id)
        return record

    def query(
        self,
        *,
        statement_id: str | None = None,
        iteration: int | None = None,
        error_type: str | None = None,
    ) -> list[FailureBankRecord]:
        return [
            record
            for record in (
                FailureBankRecord.model_validate(row) for row in iter_jsonl(self.path)
            )
            if (statement_id is None or record.statement_id == statement_id)
            and (iteration is None or record.iteration == iteration)
            and (error_type is None or record.error_type == error_type)
        ]

    def flush(self) -> None:
        """Records are appended and flushed at insertion time."""
