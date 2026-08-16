"""Generate and Pantograph-verify conservative LeanWorkbook analogies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from lean_prover.Dataset.build_verified_datasets import (
    ASSEMBLER_VERSION,
    FAIL,
    NORMALIZATION_VERSION,
    SUCCESS,
    _result_error,
    _write_json,
    add_dataset_contract,
    dataset_record_hash,
    sha256_file,
    sha256_text,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
from lean_prover.lean_training.verification.pool import VerificationPool, VerificationPoolConfig
from lean_prover.lean_training.verification.schema import VerificationTask


SOURCE = "InternLM/Lean-Workbook"
EXPANSION_VERSION = "leanworkbook_analogy_v3"
_DECLARATION = re.compile(r"(?m)^\s*(?:theorem|lemma)\s+([^\s(:{]+)")
_REORDERABLE = re.compile(
    r"\b(rw|simp|simpa|linarith|nlinarith)\s*\[([^\[\]\n]+,[^\[\]\n]+)\]"
)
_NUMERIC_HEADER = re.compile(r"(?:\b(?:Nat|Int|Rat|Real)\b|[\u2115\u2124\u211a\u211d])")
_RELATIONS = ("\u2264", "\u2265", "<", ">", "=")
HIGH_QUALITY_METHODS = frozenset(
    {
        "equality_symmetry",
        "relation_translate_one",
        "relation_scale_two",
        "equality_sub_zero",
    }
)
LOW_QUALITY_METHODS = frozenset(
    {"true_and", "or_false", "extra_true_assumption"}
)
PROOF_VARIATION_METHODS = frozenset({"proof_argument_reorder"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def split_statement(statement: str) -> tuple[str, str, str]:
    """Return declaration name, header before goal colon, and proposition."""

    match = _DECLARATION.search(statement)
    if match is None:
        raise ValueError("statement has no theorem/lemma declaration")
    depth = 0
    in_string = False
    escaped = False
    for index in range(match.end(), len(statement)):
        char = statement[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            proposition = statement[index + 1 :].strip()
            if not proposition:
                raise ValueError("empty theorem proposition")
            return match.group(1), statement[:index].rstrip(), proposition
    raise ValueError("could not locate top-level theorem goal colon")


def replace_declaration_name(statement: str, new_name: str) -> str:
    match = _DECLARATION.search(statement)
    if match is None:
        raise ValueError("statement has no declaration name")
    return statement[: match.start(1)] + new_name + statement[match.end(1) :]


def _split_arguments(value: str) -> list[str]:
    values: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            values.append(value[start:index].strip())
            start = index + 1
    values.append(value[start:].strip())
    return [item for item in values if item]


def reordered_proof(proof: str) -> str | None:
    match = _REORDERABLE.search(proof)
    if match is None:
        return None
    arguments = _split_arguments(match.group(2))
    if len(arguments) < 2:
        return None
    replacement = f"{match.group(1)} [{', '.join(reversed(arguments))}]"
    candidate = proof[: match.start()] + replacement + proof[match.end() :]
    return candidate if candidate != proof else None


def parenthesized_proof(proof: str) -> str:
    lines = proof.strip().splitlines()
    if not lines:
        raise ValueError("empty parent proof")
    if lines[0].strip() != "by":
        return f"({proof.strip()})"
    body = textwrap.dedent("\n".join(lines[1:])).strip("\n")
    if not body:
        raise ValueError("empty by proof")
    return "(by\n" + textwrap.indent(body, "      ") + "\n    )"


def split_top_level_relation(proposition: str) -> tuple[str, str, str] | None:
    """Split a direct top-level numeric relation without touching nested hypotheses."""

    depth = 0
    in_string = False
    escaped = False
    candidate: tuple[int, int, str] | None = None
    index = 0
    while index < len(proposition):
        char = proposition[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            if proposition.startswith("->", index) or char in ("\u2192", "\u2194"):
                return None
            operator = None
            width = 1
            if proposition.startswith("<=", index):
                operator, width = "<=", 2
            elif proposition.startswith(">=", index):
                operator, width = ">=", 2
            elif char in _RELATIONS:
                operator = char
            if operator is not None:
                if candidate is not None:
                    return None
                candidate = (index, width, operator)
                index += width
                continue
        index += 1
    if candidate is None:
        return None
    position, width, operator = candidate
    left = proposition[:position].strip()
    right = proposition[position + width :].strip()
    if not left or not right:
        return None
    return left, operator, right


def applicable_high_quality_methods(statement: str) -> list[str]:
    """Return mathematically meaningful analogy methods supported by the goal."""

    _, header, proposition = split_statement(statement)
    relation = split_top_level_relation(proposition)
    if relation is None or _NUMERIC_HEADER.search(header) is None:
        return []
    _, operator, _ = relation
    methods = ["relation_translate_one", "relation_scale_two"]
    if operator == "=":
        methods.insert(0, "equality_symmetry")
        # Nat subtraction is truncated, so only use the sub-zero equivalence
        # where an additive group is explicit in the declaration header.
        if re.search(r"(?:\b(?:Int|Rat|Real)\b|[\u2124\u211a\u211d])", header):
            methods.append("equality_sub_zero")
    return methods


def _parent_fact(proposition: str, proof: str) -> str:
    return (
        "by\n"
        f"  have h_parent : ({proposition}) := {parenthesized_proof(proof)}\n"
    )


def transform(statement: str, proof: str, method: str, new_name: str) -> tuple[str, str]:
    _, header, proposition = split_statement(statement)
    header = replace_declaration_name(header, new_name)
    if method == "true_and":
        return (
            f"{header} : True ∧ ({proposition})",
            f"by\n  constructor\n  · trivial\n  · exact {parenthesized_proof(proof)}",
        )
    if method == "or_false":
        return (
            f"{header} : ({proposition}) ∨ False",
            f"by\n  exact Or.inl {parenthesized_proof(proof)}",
        )
    if method == "extra_true_assumption":
        return f"{header} (h_extra : True) : {proposition}", proof
    if method == "proof_argument_reorder":
        changed = reordered_proof(proof)
        if changed is None:
            raise ValueError("proof contains no reorderable tactic argument list")
        return f"{header} : {proposition}", changed
    relation = split_top_level_relation(proposition)
    if method in HIGH_QUALITY_METHODS:
        if relation is None or _NUMERIC_HEADER.search(header) is None:
            raise ValueError("goal is not a supported top-level numeric relation")
        left, operator, right = relation
        parent_fact = _parent_fact(proposition, proof)
        if method == "equality_symmetry":
            if operator != "=":
                raise ValueError("symmetry requires an equality goal")
            return f"{header} : {right} = {left}", parent_fact + "  exact h_parent.symm"
        if method == "relation_translate_one":
            # Match the orientation of mathlib's add_*_add_right lemmas in the
            # frozen environment (the added term is printed on the left).
            goal = f"1 + ({left}) {operator} 1 + ({right})"
            if operator == "=":
                finish = "  exact congrArg (fun z => 1 + z) h_parent"
            elif operator in ("\u2264", "\u2265", "<=", ">="):
                finish = "  exact add_le_add_right h_parent 1"
            else:
                finish = "  exact add_lt_add_right h_parent 1"
            return f"{header} : {goal}", parent_fact + finish
        if method == "relation_scale_two":
            goal = f"2 * ({left}) {operator} 2 * ({right})"
            if operator == "=":
                finish = "  exact congrArg (fun z => 2 * z) h_parent"
            elif operator in ("\u2264", "\u2265", "<=", ">="):
                finish = "  exact mul_le_mul_of_nonneg_left h_parent (by norm_num)"
            else:
                finish = "  exact mul_lt_mul_of_pos_left h_parent (by norm_num)"
            return f"{header} : {goal}", parent_fact + finish
        if method == "equality_sub_zero":
            if operator != "=":
                raise ValueError("sub-zero equivalence requires an equality goal")
            return f"{header} : ({left}) - ({right}) = 0", parent_fact + "  exact sub_eq_zero.mpr h_parent"
    raise ValueError(f"unsupported expansion method: {method}")


def _informal(original: str, method: str) -> str:
    text = original.strip()
    if method == "true_and":
        return f"Equivalent reformulation: prove True together with the original claim. Original problem: {text}"
    if method == "or_false":
        return f"Equivalent reformulation: prove the original claim or False. Original problem: {text}"
    if method == "extra_true_assumption":
        return f"Assume additionally that True holds; prove the original claim. Original problem: {text}"
    if method == "equality_symmetry":
        return f"Prove the symmetric form of the original equality. Original problem: {text}"
    if method == "relation_translate_one":
        return f"Add one to both sides of the original relation and prove the resulting relation. Original problem: {text}"
    if method == "relation_scale_two":
        return f"Multiply both sides of the original relation by two and prove the resulting relation. Original problem: {text}"
    if method == "equality_sub_zero":
        return f"Express the original equality as a zero-difference equality. Original problem: {text}"
    return text


def expanded_row(
    parent: Mapping[str, Any], *, method: str, ordinal: int, raw_path: Path
) -> dict[str, Any]:
    original_statement = str(parent.get("statement") or "").strip()
    original_proof = str(parent.get("proof") or "").strip()
    if not original_statement or not original_proof:
        raise ValueError("parent row is missing statement or proof")
    old_name, _, _ = split_statement(original_statement)
    suffix = hashlib.sha256(
        f"{parent.get('record_id')}:{method}:{EXPANSION_VERSION}".encode()
    ).hexdigest()[:12]
    new_name = f"{old_name}__expand_{method}_{suffix}"
    statement, proof = transform(original_statement, original_proof, method, new_name)
    record_id = new_name
    row = dict(parent)
    row.update(
        {
            "record_id": record_id,
            "statement_id": f"stmt_{sha256_text(statement)[:24]}",
            "data_state": "raw",
            "source_file": str(raw_path),
            "source_declaration": new_name,
            "source_span": {
                "start_row": ordinal,
                "end_row": ordinal,
                "start_line": None,
                "end_line": None,
            },
            "informal_statement": _informal(str(parent.get("informal_statement") or ""), method),
            "statement": statement,
            "proof": proof,
            "raw_declaration": f"{statement} := by sorry",
            "raw_source_context": parent.get("raw_source_context"),
            "statement_verified": False,
            "proof_verified": False,
            "reference_proof_verified": False,
            "pantograph_verified": "pending",
            "verification_status": "pending",
            "verification_error_type": None,
            "verification_error_message": None,
            "assembled_source_hash": sha256_text(f"import Mathlib\n\n{statement} := {proof}\n"),
            "recovered_source_hash": sha256_text(f"{statement}\n{proof}"),
            "statement_sha256": sha256_text(statement),
            "proof_sha256": sha256_text(proof),
        }
    )
    metadata = dict(parent.get("metadata") or {})
    if method in HIGH_QUALITY_METHODS:
        quality = "high"
    elif method in LOW_QUALITY_METHODS:
        quality = "low"
    else:
        quality = "proof_variation"
    metadata.update(
        {
            "expansion_version": EXPANSION_VERSION,
            "expansion_method": method,
            "expansion_quality": quality,
            "parent_record_id": parent.get("record_id"),
            "parent_record_hash": parent.get("record_hash"),
        }
    )
    row["metadata"] = metadata
    row["record_hash"] = dataset_record_hash(row, source=SOURCE)
    if set(row) != set(parent):
        raise AssertionError("raw expansion changed the parent row field set")
    return row


def generate(args: argparse.Namespace) -> None:
    parents = read_jsonl(args.success_file)
    eligible = [
        row
        for row in parents
        if len(str(row.get("statement") or "")) <= args.max_statement_chars
        and len(str(row.get("proof") or "")) <= args.max_proof_chars
        and str(row.get("pantograph_verified")) == SUCCESS
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row.get('record_id')}:{row.get('record_hash')}".encode()
        ).hexdigest()
    )
    planned: list[tuple[dict[str, Any], str]] = []
    high_parent_count = 0
    for row in eligible:
        if high_parent_count >= args.high_quality_parents:
            break
        methods = applicable_high_quality_methods(str(row.get("statement") or ""))
        if not methods:
            continue
        # Rotate the method order deterministically so the cap does not always
        # suppress the same valid transformation.
        offset = int(
            hashlib.sha256(f"{args.seed}:{row.get('record_id')}:high".encode()).hexdigest(),
            16,
        ) % len(methods)
        methods = methods[offset:] + methods[:offset]
        planned.extend(
            (row, method) for method in methods[: args.high_quality_variants_per_parent]
        )
        high_parent_count += 1

    low_methods = ("extra_true_assumption", "true_and", "or_false")
    low_parent_count = 0
    for row in eligible:
        if low_parent_count >= args.low_quality_parents:
            break
        offset = int(
            hashlib.sha256(f"{args.seed}:{row.get('record_id')}:low".encode()).hexdigest(),
            16,
        ) % len(low_methods)
        methods = low_methods[offset:] + low_methods[:offset]
        planned.extend(
            (row, method) for method in methods[: args.low_quality_variants_per_parent]
        )
        low_parent_count += 1
    reorder_count = 0
    for row in eligible:
        if reorder_count >= args.reorder_limit:
            break
        if reordered_proof(str(row.get("proof") or "")):
            planned.append((row, "proof_argument_reorder"))
            reorder_count += 1
    rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for ordinal, (parent, method) in enumerate(planned):
        try:
            rows.append(expanded_row(parent, method=method, ordinal=ordinal, raw_path=args.raw_output.resolve()))
        except ValueError as error:
            failures[f"{method}:{error}"] += 1
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("expansion record IDs are not unique")
    hashes = [str(row["record_hash"]) for row in rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("expansion record hashes are not unique")
    exact_pairs = [(str(row["statement"]), str(row["proof"])) for row in rows]
    if len(exact_pairs) != len(set(exact_pairs)):
        raise RuntimeError("expansion contains duplicate statement/proof pairs")
    low_per_parent = Counter(
        str((row.get("metadata") or {}).get("parent_record_id"))
        for row in rows
        if (row.get("metadata") or {}).get("expansion_quality") == "low"
    )
    if low_per_parent and max(low_per_parent.values()) > 2:
        raise RuntimeError("a parent received more than two low-quality expansions")
    write_jsonl(args.raw_output, rows)
    report = {
        "schema_version": "leanworkbook_expansion_generation_v1",
        "parent_success_rows": len(parents),
        "eligible_parent_rows": len(eligible),
        "generated_rows": len(rows),
        "high_quality_parent_rows": high_parent_count,
        "low_quality_parent_rows": low_parent_count,
        "methods": dict(Counter(str((row.get("metadata") or {}).get("expansion_method")) for row in rows)),
        "quality": dict(Counter(str((row.get("metadata") or {}).get("expansion_quality")) for row in rows)),
        "generation_failures": dict(failures),
        "duplicate_record_ids": len(ids) - len(set(ids)),
        "duplicate_record_hashes": len(hashes) - len(set(hashes)),
        "duplicate_statement_proof_pairs": len(exact_pairs) - len(set(exact_pairs)),
        "max_low_quality_expansions_per_parent": max(low_per_parent.values(), default=0),
        "parent_field_sets_preserved": True,
        "raw_sha256": sha256_file(args.raw_output),
        "pantograph_started": False,
    }
    _write_json(args.report_dir / "generation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_task(row: Mapping[str, Any], index: int, environment_hash: str) -> VerificationTask:
    task = VerificationTask(
        priority=index,
        problem_index=index,
        attempt_index=0,
        problem_id=str(row["record_id"]),
        prompt="",
        generated_proof=str(row["proof"]),
        raw_completion=str(row["proof"]),
        lean_code=f"{str(row['statement']).strip()} := {str(row['proof']).strip()}",
        imports=("Mathlib",),
        payload={
            "record_id": row["record_id"],
            "environment_hash": environment_hash,
            "expansion_version": EXPANSION_VERSION,
            "assembler_version": ASSEMBLER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
        },
        reject_forbidden=True,
    )
    task.payload["assembled_source_hash"] = sha256_text(
        build_labeled_lean_code(task, include_imports=True)
    )
    return task


def compatible(result: Mapping[str, Any], task: VerificationTask, environment_hash: str) -> bool:
    return (
        str(result.get("record_id") or result.get("problem_id")) == task.problem_id
        and result.get("environment_hash") == environment_hash
        and result.get("expansion_version") == EXPANSION_VERSION
        and result.get("assembler_version") == ASSEMBLER_VERSION
        and result.get("normalization_version") == NORMALIZATION_VERSION
        and result.get("assembled_source_hash") == task.payload.get("assembled_source_hash")
    )


def compact(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "problem_id", "record_id", "problem_index", "attempt_index", "status",
        "success", "diagnostics", "compile_errors", "compile_warnings", "timed_out",
        "verifier_backend", "rejected_reason", "verification_seconds",
        "assembled_source_hash", "environment_hash", "expansion_version",
        "assembler_version", "normalization_version", "worker_pid", "worker_generation",
        "worker_restart_count", "pantograph_restart_count",
    )
    return {key: result.get(key) for key in keys if key in result}


def verify(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.raw_output)
    identity = environment_identity(args.lean_project, ("Mathlib",))
    environment_hash = str(identity["environment_hash"])
    cached_rows = read_jsonl(args.results_cache) if args.results_cache.exists() else []
    cache: dict[str, list[dict[str, Any]]] = {}
    for result in cached_rows:
        cache.setdefault(str(result.get("record_id") or result.get("problem_id")), []).append(result)

    def find(task: VerificationTask) -> dict[str, Any] | None:
        return next(
            (row for row in reversed(cache.get(task.problem_id, ())) if compatible(row, task, environment_hash)),
            None,
        )

    pending = [
        build_task(row, index, environment_hash)
        for index, row in enumerate(rows)
        if find(build_task(row, index, environment_hash)) is None
    ]
    runtime: dict[str, Any] = {}
    completed = 0

    def persist(raw: dict[str, Any]) -> None:
        nonlocal completed
        result = compact(raw)
        append_jsonl(args.results_cache, result)
        cache.setdefault(str(result.get("record_id") or result.get("problem_id")), []).append(result)
        completed += 1
        if completed % 100 == 0:
            print(json.dumps({"completed": completed, "pending": len(pending)}), flush=True)

    if pending:
        pool = VerificationPool(
            VerificationPoolConfig(
                lean_project_path=str(args.lean_project), imports=("Mathlib",), timeout=args.timeout,
                warmup_timeout=3600, num_workers=1, queue_maxsize=8,
                max_worker_restarts=3, max_task_retries=1, shutdown_timeout=15,
                task_spool_dir=str(args.report_dir / "runtime" / "task_spool"),
            )
        )
        with pool:
            for start in range(0, len(pending), args.batch_size):
                run = pool.run_batch(pending[start : start + args.batch_size], on_result=persist)
                runtime = run.runtime_stats
                if run.fatal_errors:
                    raise RuntimeError(str(run.fatal_errors))

    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    unresolved = 0
    statuses: Counter[str] = Counter()
    for index, raw in enumerate(rows):
        task = build_task(raw, index, environment_hash)
        result = find(task)
        if result is None:
            unresolved += 1
            continue
        row = dict(raw)
        row.update(
            {
                "lean_version": identity["lean_version"],
                "mathlib_commit": identity["mathlib_commit"],
                "environment_hash": environment_hash,
                "assembled_source_hash": result.get("assembled_source_hash"),
                "verification_backend": result.get("verifier_backend") or "pantograph",
                "verification_status": "success" if result.get("success") else result.get("status"),
                "statement_verified": bool(result.get("success")),
                "proof_verified": bool(result.get("success")),
                "reference_proof_verified": bool(result.get("success")),
                "verification_error_type": None if result.get("success") else result.get("status"),
                "verification_error_message": None if result.get("success") else _result_error(result),
                "data_state": "verified" if result.get("success") else "quarantined",
            }
        )
        if result.get("success"):
            output = add_dataset_contract(row, source=SOURCE, status=SUCCESS)
            success.append(output)
        else:
            output = add_dataset_contract(
                row, source=SOURCE, status=FAIL, error_message=_result_error(result)
            )
            failed.append(output)
            statuses[str(result.get("status") or "unknown")] += 1
    write_jsonl(args.success_output, success)
    write_jsonl(args.fail_output, failed)
    final_rows = success + failed
    method_stats: dict[str, Counter[str]] = {}
    quality_stats: dict[str, Counter[str]] = {}
    raw_by_id = {str(row["record_id"]): row for row in rows}
    expected_fail_error_field_additions = 0
    unexpected_field_set_mismatches = 0
    for row in final_rows:
        metadata = row.get("metadata") or {}
        method = str(metadata.get("expansion_method") or "unknown")
        quality = str(metadata.get("expansion_quality") or "unknown")
        outcome = str(row.get("pantograph_verified") or "unknown")
        for table, key in ((method_stats, method), (quality_stats, quality)):
            table.setdefault(key, Counter())["total"] += 1
            table[key][outcome] += 1
        raw_parent = raw_by_id.get(str(row.get("record_id")))
        if raw_parent is None:
            unexpected_field_set_mismatches += 1
        else:
            added = set(row) - set(raw_parent)
            removed = set(raw_parent) - set(row)
            if outcome == FAIL and added == {"error_message"} and not removed:
                expected_fail_error_field_additions += 1
            elif added or removed:
                unexpected_field_set_mismatches += 1
    record_ids = [str(row.get("record_id")) for row in final_rows]
    record_hashes = [str(row.get("record_hash")) for row in final_rows]
    statement_proof_pairs = [
        (str(row.get("statement")), str(row.get("proof"))) for row in final_rows
    ]
    report = {
        "schema_version": "leanworkbook_expansion_verification_v1",
        "expansion_version": EXPANSION_VERSION,
        "raw_rows": len(rows),
        "new_compilations": completed,
        "success": len(success),
        "fail": len(failed),
        "unresolved": unresolved,
        "failure_statuses": dict(statuses),
        "method_outcomes": {key: dict(value) for key, value in sorted(method_stats.items())},
        "quality_outcomes": {key: dict(value) for key, value in sorted(quality_stats.items())},
        "duplicate_record_ids": len(record_ids) - len(set(record_ids)),
        "duplicate_record_hashes": len(record_hashes) - len(set(record_hashes)),
        "duplicate_statement_proof_pairs": len(statement_proof_pairs) - len(set(statement_proof_pairs)),
        "raw_parent_field_sets_preserved_during_generation": True,
        "expected_fail_error_message_field_additions": expected_fail_error_field_additions,
        "unexpected_field_set_mismatches": unexpected_field_set_mismatches,
        "source_values": dict(Counter(str(row.get("source")) for row in final_rows)),
        "pantograph_verified_values": dict(
            Counter(str(row.get("pantograph_verified")) for row in final_rows)
        ),
        "complete": unresolved == 0 and len(success) + len(failed) == len(rows),
        "environment": identity,
        "raw_sha256": sha256_file(args.raw_output),
        "success_sha256": sha256_file(args.success_output),
        "fail_sha256": sha256_file(args.fail_output),
        "runtime": runtime,
    }
    _write_json(args.report_dir / "verification_report.json", report)
    method_lines = "\n".join(
        f"| {method} | {counts.get('total', 0)} | {counts.get(SUCCESS, 0)} | "
        f"{counts.get(FAIL, 0)} |"
        for method, counts in sorted(method_stats.items())
    )
    markdown = f"""# LeanWorkbook Analogy Expansion Report

- Expansion version: `{EXPANSION_VERSION}`
- Parent verified-success rows: 10,037
- Raw expansions: {len(rows)}
- Pantograph success: {len(success)}
- Pantograph fail: {len(failed)}
- Unresolved: {unresolved}
- Complete: {str(report['complete']).lower()}
- Raw parent field sets preserved during generation: true
- Expected fail-contract `error_message` additions: {expected_fail_error_field_additions}
- Unexpected field-set mismatches: {unexpected_field_set_mismatches}
- Duplicate record IDs / hashes / exact statement-proof pairs: {report['duplicate_record_ids']} / {report['duplicate_record_hashes']} / {report['duplicate_statement_proof_pairs']}

| Method | Total | Success | Fail |
|---|---:|---:|---:|
{method_lines}

## Frozen environment

- Lean: `{identity['lean_version']}`
- mathlib commit: `{identity['mathlib_commit']}`
- environment hash: `{environment_hash}`

## Artifact hashes

- raw: `{report['raw_sha256']}`
- success: `{report['success_sha256']}`
- fail: `{report['fail_sha256']}`
"""
    (args.report_dir / "final_report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("generate", "verify"))
    value.add_argument("--success-file", type=Path, default=root / "verified_data" / "leanworkbook_verified_success_2.jsonl")
    value.add_argument("--raw-output", type=Path, default=root / "raw_data" / "leanworkbook_expand_raw.jsonl")
    value.add_argument("--success-output", type=Path, default=root / "verified_data" / "leanworkbook_expand_verified_success.jsonl")
    value.add_argument("--fail-output", type=Path, default=root / "verified_data" / "leanworkbook_expand_verified_fail.jsonl")
    value.add_argument("--results-cache", type=Path, default=root / "verified_data" / "leanworkbook_expand_verification_results_v3.jsonl")
    value.add_argument("--report-dir", type=Path, default=Path("outputs/leanworkbook_expansion"))
    value.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    value.add_argument("--seed", type=int, default=20260807)
    value.add_argument("--high-quality-parents", type=int, default=4000)
    value.add_argument("--high-quality-variants-per-parent", type=int, default=3)
    value.add_argument("--low-quality-parents", type=int, default=1500)
    value.add_argument("--low-quality-variants-per-parent", type=int, default=2)
    value.add_argument("--reorder-limit", type=int, default=1500)
    value.add_argument("--max-statement-chars", type=int, default=3000)
    value.add_argument("--max-proof-chars", type=int, default=5000)
    value.add_argument("--timeout", type=int, default=60)
    value.add_argument("--batch-size", type=int, default=256)
    return value


def main() -> None:
    args = parser().parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        generate(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
