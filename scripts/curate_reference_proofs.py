#!/usr/bin/env python3
"""Build a deterministic, provenance-preserving 30-proof PRECHECK fixture.

The source datasets are not rewritten.  Candidate reference proofs are passed
through the production extractor-independent assembler and the persistent
Pantograph pool.  Only proofs that compile successfully are selected, and the
selected 30 are compiled a second time to produce the final round-trip report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    compose_lean_theorem,
    normalize_proof_for_assembly,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def eligible_rows(path: Path, rng: random.Random) -> list[tuple[int, dict[str, Any]]]:
    rows = [
        (index, row)
        for index, row in enumerate(read_jsonl(path))
        if row.get("lean_statement") and (row.get("proof") or row.get("completion"))
    ]
    rng.shuffle(rows)
    return rows


def make_task(
    row: dict[str, Any],
    *,
    source_file: Path,
    source_index: int,
    problem_index: int,
    generation_id: str,
    group: str,
) -> tuple[VerificationTask, dict[str, Any]]:
    statement = str(row["lean_statement"]).strip()
    reference_proof = str(row.get("proof") or row.get("completion") or "").strip()
    normalized, proof_format = normalize_proof_for_assembly(reference_proof)
    declaration = compose_lean_theorem(
        statement,
        normalized,
        proof_format=proof_format,
    )
    imports = tuple(row.get("imports") or ("Mathlib",))
    context_lines = tuple(row.get("context_lines") or [])
    provenance = {
        "group": group,
        "id": str(row.get("id") or f"row-{source_index}"),
        "source_file": str(source_file.resolve()),
        "source_index": source_index,
        "statement": statement,
        "reference_proof": reference_proof,
        "normalized_proof": normalized,
        "proof_format": proof_format.value,
        "imports": list(imports),
        "context_lines": list(context_lines),
        "statement_hash": str(row.get("statement_hash") or sha256_text(statement)),
        "reference_proof_hash": str(
            row.get("reference_proof_hash") or sha256_text(reference_proof)
        ),
        "assembler_version": ASSEMBLER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
    }
    task = VerificationTask(
        priority=problem_index,
        problem_index=problem_index,
        attempt_index=0,
        problem_id=provenance["id"],
        prompt=str(row.get("prompt") or ""),
        generated_proof=normalized,
        raw_completion=reference_proof,
        lean_code=declaration,
        imports=imports,
        context_lines=context_lines,
        payload={"generation_id": generation_id, **provenance},
        reject_forbidden=True,
    )
    provenance["assembled_source"] = build_labeled_lean_code(
        task, include_imports=True
    )
    provenance["assembled_source_hash"] = sha256_text(
        provenance["assembled_source"]
    )
    return task, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--project", default="lean_project")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    args = parser.parse_args()

    train_path = Path(args.train)
    eval_path = Path(args.eval)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_candidates = eligible_rows(train_path, rng)
    eval_candidates = eligible_rows(eval_path, rng)
    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=args.project,
            imports=("Mathlib",),
            timeout=args.timeout,
            warmup_timeout=args.startup_timeout,
            num_workers=args.num_workers,
            queue_maxsize=max(32, args.batch_size * 2),
            max_worker_restarts=3,
            max_task_retries=1,
            shutdown_timeout=15,
        )
    )
    attempts: list[dict[str, Any]] = []
    selected_train: list[dict[str, Any]] = []
    selected_eval: list[dict[str, Any]] = []
    task_counter = 0

    def scan(
        candidates: list[tuple[int, dict[str, Any]]],
        *,
        source_file: Path,
        required: int,
        destination: list[dict[str, Any]],
        label: str,
    ) -> None:
        nonlocal task_counter
        cursor = 0
        while len(destination) < required and cursor < len(candidates):
            chunk = candidates[cursor : cursor + args.batch_size]
            cursor += len(chunk)
            tasks: list[VerificationTask] = []
            provenance_by_id: dict[str, dict[str, Any]] = {}
            for source_index, row in chunk:
                generation_id = f"curate:{label}:{source_index}"
                try:
                    task, provenance = make_task(
                        row,
                        source_file=source_file,
                        source_index=source_index,
                        problem_index=task_counter,
                        generation_id=generation_id,
                        group=label,
                    )
                except Exception as error:
                    attempts.append(
                        {
                            "generation_id": generation_id,
                            "group": label,
                            "source_file": str(source_file.resolve()),
                            "source_index": source_index,
                            "id": row.get("id"),
                            "success": False,
                            "status": "assembly_error",
                            "diagnostics": f"{type(error).__name__}: {error}",
                        }
                    )
                    task_counter += 1
                    continue
                tasks.append(task)
                provenance_by_id[generation_id] = provenance
                task_counter += 1
            if tasks:
                run = pool.run_batch(tasks)
                if run.fatal_errors:
                    raise RuntimeError(f"Pantograph pool failed: {run.fatal_errors}")
                for result in run.results:
                    generation_id = str(result["generation_id"])
                    merged = {**provenance_by_id[generation_id], **result}
                    attempts.append(merged)
                    if result.get("success") and len(destination) < required:
                        destination.append(provenance_by_id[generation_id])
            print(
                json.dumps(
                    {
                        "phase": "curation",
                        "group": label,
                        "scanned": cursor,
                        "selected": len(destination),
                        "required": required,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if len(destination) < required:
            raise RuntimeError(
                f"only found {len(destination)}/{required} verified proofs for {label}"
            )

    try:
        scan(
            train_candidates,
            source_file=train_path,
            required=20,
            destination=selected_train,
            label="train_pool",
        )
        scan(
            eval_candidates,
            source_file=eval_path,
            required=10,
            destination=selected_eval,
            label="eval",
        )
        manifest = []
        for group, rows in (
            ("train", selected_train[:10]),
            ("eval", selected_eval),
            ("train_extra", selected_train[10:20]),
        ):
            for row in rows:
                manifest.append({**row, "group": group, "pantograph_verified": True})

        final_tasks: list[VerificationTask] = []
        final_provenance: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(manifest):
            source_rows = read_jsonl(Path(record["source_file"]))
            source_row = source_rows[int(record["source_index"])]
            generation_id = f"roundtrip:{record['group']}:{record['id']}"
            task, provenance = make_task(
                source_row,
                source_file=Path(record["source_file"]),
                source_index=int(record["source_index"]),
                problem_index=index,
                generation_id=generation_id,
                group=str(record["group"]),
            )
            final_tasks.append(task)
            final_provenance[generation_id] = provenance
        final_run = pool.run_batch(final_tasks)
        final_records = [
            {**final_provenance[str(result["generation_id"])], **result}
            for result in final_run.results
        ]
        identity = environment_identity(args.project, ("Mathlib",))
        report = {
            "success": len(final_records) == 30
            and all(row.get("success") for row in final_records)
            and not final_run.fatal_errors,
            "passed": sum(bool(row.get("success")) for row in final_records),
            "total": len(final_records),
            "groups": {
                group: sum(
                    bool(row.get("success")) and row.get("group") == group
                    for row in final_records
                )
                for group in ("train", "eval", "train_extra")
            },
            "candidate_attempts": len(attempts),
            "environment": identity,
            "warmup_reports": pool.warmup_reports,
            "runtime_stats": final_run.runtime_stats,
            "fatal_errors": final_run.fatal_errors,
            "records": final_records,
        }
        write_jsonl(output_dir / "reference_manifest.jsonl", manifest)
        write_jsonl(output_dir / "curation_attempts.jsonl", attempts)
        (output_dir / "reference_roundtrip_30.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "records"},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit(0 if report["success"] else 1)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
