"""Data-role loading, prompt isolation, and overlap validation."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable

from .schemas import DataRole, StatementRecord
from .utils import read_jsonl, write_json_atomic


def load_statements(path: str | Path, role: DataRole) -> list[StatementRecord]:
    return [StatementRecord.from_prepared(row, role) for row in read_jsonl(path)]


def role_overlap_report(
    datasets: dict[DataRole, list[StatementRecord]],
) -> dict[str, object]:
    overlaps: list[dict[str, object]] = []
    for left_role, right_role in itertools.combinations(sorted(datasets, key=str), 2):
        left = datasets[left_role]
        right = datasets[right_role]
        left_ids = {record.statement_id for record in left}
        right_ids = {record.statement_id for record in right}
        left_hashes = {record.statement_hash for record in left}
        right_hashes = {record.statement_hash for record in right}
        left_source_ids = {
            f"{record.source}:{record.source_id}" for record in left if record.source_id
        }
        right_source_ids = {
            f"{record.source}:{record.source_id}" for record in right if record.source_id
        }
        shared_ids = sorted(left_ids & right_ids)
        shared_hashes = sorted(left_hashes & right_hashes)
        shared_source_ids = sorted(left_source_ids & right_source_ids)
        if shared_ids or shared_hashes or shared_source_ids:
            overlaps.append(
                {
                    "left_role": left_role.value,
                    "right_role": right_role.value,
                    "statement_ids": shared_ids,
                    "statement_hashes": shared_hashes,
                    "source_ids": shared_source_ids,
                }
            )
    return {"valid": not overlaps, "overlaps": overlaps}


def validate_role_isolation(
    datasets: dict[DataRole, list[StatementRecord]],
    *,
    report_path: str | Path | None = None,
    policy: str = "error",
) -> dict[str, object]:
    report = role_overlap_report(datasets)
    if report_path:
        write_json_atomic(report_path, report)
    if not report["valid"] and policy == "error":
        pairs = [
            f"{item['left_role']}∩{item['right_role']}"
            for item in report["overlaps"]
        ]
        raise ValueError(f"data-role overlap detected: {pairs}; see {report_path}")
    return report


def assert_prompt_has_no_reference_proof(record: StatementRecord, prompt: str) -> None:
    if record.reference_proof and record.reference_proof.strip() in prompt:
        raise ValueError(
            f"reference proof leaked into {record.data_role} prompt: {record.statement_id}"
        )


def index_statements(records: Iterable[StatementRecord]) -> dict[str, StatementRecord]:
    result: dict[str, StatementRecord] = {}
    for record in records:
        if record.statement_id in result:
            raise ValueError(f"duplicate statement_id in role {record.data_role}: {record.statement_id}")
        result[record.statement_id] = record
    return result
