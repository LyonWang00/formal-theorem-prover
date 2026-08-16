"""Route NuminaMath failures into mutually exclusive repair workflows.

The router is conservative: records are never declared invalid merely because
their proof failed.  Suspected statement failures are rechecked in the frozen
Pantograph environment with a placeholder proof before final routing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from lean_prover.Dataset.build_verified_datasets import sha256_file
from lean_prover.Dataset.verify_external_datasets import NUMINA_SOURCE, split_imports
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import PantographTheoremVerifier


SCHEMA = "numinamath_failure_routing_v2"
FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.I)
DECLARATION = re.compile(r"(?m)^\s*(?:theorem|lemma)\s+([^\s(:{]+)")
EMBEDDED_COMMAND = re.compile(
    r"(?m)^(?:import|open|namespace|section|end|theorem|lemma|def|example|instance)\b"
)
ERROR_LINE = re.compile(r"(?mi)^\s*(\d+):\d+(?:-\d+:\d+)?:\s*error:\s*(.+)$")
UNKNOWN = re.compile(r"Unknown (?:identifier|constant) `([^`]+)`", re.I)
CONTEXT_NAME = re.compile(r"^(?:lemma|aux|helper|claim|step|main)_?\w*$", re.I)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def error_families(message: str) -> list[str]:
    lowered = message.lower()
    families = []
    rules = (
        ("timeout_resource", ("timeout", "timed out", "maximum number of heartbeats", "maximum recursion depth")),
        ("unknown_identifier", ("unknown identifier", "unknown constant", "invalid field")),
        ("syntax_error", ("syntax error", "unexpected token", "unexpected end of input", "expected command")),
        ("type_mismatch", ("type mismatch", "application type mismatch", "function expected")),
        ("rewrite_mismatch", ("rewrite failed", "did not find an occurrence")),
        ("no_goals", ("no goals to be solved",)),
        ("tactic_error", ("tactic `", "tactic failed", "made no progress")),
        ("unsolved_goals", ("unsolved goal",)),
        ("elaboration_error", ("failed to synthesize", "cannot synthesize", "contains metavariables")),
    )
    for family, needles in rules:
        if any(needle in lowered for needle in needles):
            families.append(family)
    return families or ["other"]


def only_missing_proof_diagnostics(message: str) -> bool:
    lowered = message.lower()
    return (
        "unsolved goals" in lowered
        and ("unexpected token 'end'" in lowered or "unexpected end of input" in lowered)
        and not any(
            marker in lowered
            for marker in (
                "unknown identifier", "unknown constant", "type mismatch",
                "failed to synthesize", "invalid field", "declaration has metavariables",
            )
        )
    )


def statement_complexity(statement: str) -> int:
    without_comments = re.sub(r"/-.*?-/", " ", statement, flags=re.S)
    score = 0
    if len(without_comments) > 1800:
        score += 2
    elif len(without_comments) > 1000:
        score += 1
    logical = sum(without_comments.count(token) for token in ("∀", "∃", "→", "↔", "∧", "∨"))
    if logical >= 8:
        score += 2
    elif logical >= 4:
        score += 1
    if len(re.findall(r"\([^()]*(?::|∈)[^()]*\)", without_comments)) >= 10:
        score += 1
    if re.search(r"\b(?:Polynomial|Matrix|Measure|Filter|Finset|Set|Complex|deriv|Integral)\b", without_comments):
        score += 1
    return score


def routing_entry(row: Mapping[str, Any], category: str, reasons: list[str]) -> dict[str, Any]:
    proof = str(row.get("proof") or "").strip()
    statement = str(row.get("lean_statement") or row.get("formal_statement") or "").strip()
    message = str(row.get("error_message") or "")
    return {
        "schema_version": SCHEMA,
        "record_id": str(row["record_id"]),
        "record_hash": str(row.get("record_hash") or ""),
        "statement_sha256": str(row.get("statement_sha256") or ""),
        "repair_category": category,
        "reason_codes": reasons,
        "error_families": error_families(message),
        "proof_present": bool(proof),
        "proof_forbidden": bool(FORBIDDEN.search(proof)),
        "proof_chars": len(proof),
        "statement_chars": len(statement),
        "statement_complexity": statement_complexity(statement),
        "routing_hash": canonical_hash({
            "record_id": row["record_id"],
            "record_hash": row.get("record_hash"),
            "category": category,
            "reasons": reasons,
        }),
    }


def provisional_route(row: Mapping[str, Any]) -> tuple[str, list[str], bool]:
    proof = str(row.get("proof") or "").strip()
    statement = str(row.get("lean_statement") or row.get("formal_statement") or "").strip()
    message = str(row.get("error_message") or "")
    lowered = message.lower()
    declarations = DECLARATION.findall(statement)
    proof_commands = EMBEDDED_COMMAND.findall(proof)
    support_sorry = bool(re.search(r":=\s*by\s+(?:sorry|admit)\b", statement, flags=re.I))
    unknown_names = UNKNOWN.findall(message)
    statement_lines = len(statement.splitlines())
    error_locations = [(int(line), detail) for line, detail in ERROR_LINE.findall(message)]
    error_in_statement = any(line <= statement_lines for line, _ in error_locations)

    if support_sorry:
        return "need_decompose", ["support_declaration_contains_placeholder"], False
    if len(declarations) > 1 and proof_commands:
        return "missing_context_suspect", ["multiple_declarations_and_embedded_proof_commands"], False
    if proof_commands:
        return "missing_context_suspect", ["proof_field_contains_top_level_commands"], False
    if any(CONTEXT_NAME.match(name) for name in unknown_names):
        return "missing_context_suspect", ["unknown_project_local_helper"], False
    if row.get("timed_out") or any(
        marker in lowered for marker in ("timeout", "timed out", "maximum number of heartbeats", "maximum recursion depth")
    ):
        return "timeout_resource", ["resource_limit_diagnostic"], False

    proof_forbidden = bool(FORBIDDEN.search(proof))
    if not proof or proof_forbidden:
        if only_missing_proof_diagnostics(message):
            if statement_complexity(statement) >= 3:
                return "need_decompose", ["missing_proof_high_statement_complexity"], False
            return "from_scratch", ["missing_proof_statement_elaborated"], False
        return "invalid_statement_suspect", ["missing_proof_with_statement_phase_error"], True

    structural = (
        "expected type must not contain free variables",
        "declaration has metavariables",
        "unknown module prefix",
        "unknown namespace",
        "invalid import",
    )
    if any(marker in lowered for marker in structural):
        return "invalid_statement_suspect", ["structural_statement_diagnostic"], True
    if error_in_statement and any(
        marker in lowered
        for marker in ("unknown identifier", "unknown constant", "unexpected token", "failed to synthesize", "invalid field")
    ):
        return "invalid_statement_suspect", ["error_location_within_statement"], True

    errors = ERROR_LINE.findall(message)
    families = set(error_families(message))
    local_families = {
        "unknown_identifier", "syntax_error", "type_mismatch", "rewrite_mismatch",
        "no_goals", "tactic_error", "unsolved_goals", "elaboration_error",
    }
    if len(proof) <= 8000 and len(errors) <= 12 and families & local_families:
        return "local_repair", ["bounded_proof_near_miss"], False
    return "from_scratch", ["proof_present_but_not_local_repair"], False


def classify(args: argparse.Namespace) -> dict[str, Any]:
    frozen_ids = {
        str(row["record_id"])
        for path in args.frozen_manifest
        for row in iter_jsonl(path)
        if row.get("record_id")
    }
    output_root = args.output_root
    buckets: dict[str, list[dict[str, Any]]] = {
        category: []
        for category in (
            "local_repair", "missing_context_suspect", "need_decompose",
            "timeout_resource", "invalid_statement_suspect", "from_scratch",
            "frozen_excluded",
        )
    }
    audit_candidates: list[dict[str, Any]] = []
    input_rows = 0
    for row in iter_jsonl(args.fail_file):
        input_rows += 1
        record_id = str(row["record_id"])
        if record_id in frozen_ids:
            category, reasons, needs_audit = "frozen_excluded", ["frozen_external_assignment"], False
        else:
            category, reasons, needs_audit = provisional_route(row)
        entry = routing_entry(row, category, reasons)
        buckets[category].append(entry)
        if needs_audit:
            audit_candidates.append(entry)

    for category, entries in buckets.items():
        write_jsonl(output_root / f"{category}_manifest.jsonl", entries)
    write_jsonl(output_root / "statement_audit_manifest.jsonl", audit_candidates)
    report = {
        "schema_version": SCHEMA,
        "command": "classify",
        "input_fail_rows": input_rows,
        "frozen_rows": len(frozen_ids),
        "category_counts": {key: len(value) for key, value in buckets.items()},
        "statement_audit_candidates": len(audit_candidates),
        "partition_total": sum(len(value) for value in buckets.values()),
        "fail_file_sha256": sha256_file(args.fail_file),
        "frozen_manifest_sha256": {str(path): sha256_file(path) for path in args.frozen_manifest},
    }
    if report["partition_total"] != input_rows:
        raise RuntimeError("routing partition does not cover fail exactly once")
    write_json(output_root / "classification_report.json", report)
    return report


def placeholder_elaborated(result: Any) -> bool:
    if result.timed_out:
        return False
    diagnostics = "\n".join([
        str(result.diagnostics or ""),
        *(str(item) for item in result.errors),
        *(str(item) for item in result.warnings),
    ])
    errors = re.findall(r"(?mi)^.*?error:.*$", diagnostics)
    return not errors and "declaration uses `sorry`" in diagnostics


def audit_statements(args: argparse.Namespace) -> dict[str, Any]:
    candidates = read_jsonl(args.audit_manifest)
    candidate_ids = {str(row["record_id"]) for row in candidates}
    existing = {str(row["record_id"]): row for row in read_jsonl(args.output)}
    rows_by_id = {
        str(row["record_id"]): row
        for row in iter_jsonl(args.fail_file)
        if str(row["record_id"]) in candidate_ids
    }
    missing = candidate_ids - set(rows_by_id)
    if missing:
        raise RuntimeError(f"audit candidates absent from fail: {sorted(missing)[:5]}")

    identity = environment_identity(args.lean_project, ("Mathlib",))
    verifier = PantographTheoremVerifier(
        args.lean_project, imports=("Mathlib",), timeout=args.timeout, startup_timeout=900
    )
    warmup = verifier.warmup(timeout=180)
    if not warmup.success:
        verifier.close()
        raise RuntimeError(f"Pantograph warmup failed: {warmup.diagnostics}")
    try:
        for index, candidate in enumerate(candidates, start=1):
            record_id = str(candidate["record_id"])
            if record_id in existing:
                continue
            row = rows_by_id[record_id]
            statement = str(row.get("lean_statement") or row.get("formal_statement") or "").strip()
            imports, body = split_imports(statement)
            if ":=" in body:
                status = "requires_context_review"
                diagnostics = "statement view contains supporting declaration bodies"
                checked = None
            else:
                checked = verifier.check_source(
                    body.rstrip() + "\n:= by sorry", timeout=args.timeout, reject_forbidden=False
                )
                status = "elaborated" if checked.success or placeholder_elaborated(checked) else "failed"
                diagnostics = str(checked.diagnostics or "")
            existing[record_id] = {
                "schema_version": SCHEMA,
                "record_id": record_id,
                "record_hash": row.get("record_hash"),
                "statement_audit_status": status,
                "error_type": checked.error_type if checked is not None else "context_required",
                "timed_out": bool(checked.timed_out) if checked is not None else False,
                "diagnostics": diagnostics,
                "errors": list(checked.errors) if checked is not None else [],
                "warnings": list(checked.warnings) if checked is not None else [],
                "lean_version": identity["lean_version"],
                "mathlib_commit": identity["mathlib_commit"],
                "environment_hash": identity["environment_hash"],
            }
            if index % 25 == 0:
                write_jsonl(args.output, existing.values())
        write_jsonl(args.output, existing.values())
    finally:
        verifier.close()
    counts = Counter(str(row["statement_audit_status"]) for row in existing.values())
    report = {
        "schema_version": SCHEMA,
        "command": "audit-statements",
        "candidate_rows": len(candidates),
        "completed_rows": len(existing),
        "status_counts": dict(counts),
        "output_sha256": sha256_file(args.output),
        "environment": identity,
    }
    write_json(args.report, report)
    return report


def invalid_type_from_diagnostics(diagnostics: str) -> str:
    lowered = diagnostics.lower()
    if any(marker in lowered for marker in ("unknown identifier", "unknown constant", "unknown namespace", "unknown module")):
        return "missing_context"
    if any(marker in lowered for marker in ("unexpected token", "expected type", "contains metavariables", "failed to synthesize")):
        return "invalid_statement"
    return "other_invalid_data"


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    provisional: dict[str, dict[str, Any]] = {}
    for path in args.routing_root.glob("*_manifest.jsonl"):
        if path.name == "statement_audit_manifest.jsonl":
            continue
        for row in iter_jsonl(path):
            record_id = str(row["record_id"])
            if record_id in provisional:
                raise RuntimeError(f"duplicate provisional route: {record_id}")
            provisional[record_id] = row
    audits = {str(row["record_id"]): row for row in iter_jsonl(args.audit_results)}
    final: dict[str, list[dict[str, Any]]] = {
        category: []
        for category in (
            "local_repair", "missing_context", "need_decompose", "timeout_resource",
            "invalid_statement", "from_scratch", "frozen_excluded",
        )
    }
    invalid_types: dict[str, str] = {}
    for record_id, entry in provisional.items():
        category = str(entry["repair_category"])
        if category == "missing_context_suspect":
            final_category = "missing_context"
            invalid_types[record_id] = "missing_context"
        elif category == "invalid_statement_suspect":
            audit = audits.get(record_id)
            if audit is None:
                raise RuntimeError(f"missing statement audit result: {record_id}")
            status = str(audit["statement_audit_status"])
            if status == "elaborated":
                final_category = (
                    "from_scratch"
                    if not entry["proof_present"] or entry.get("proof_forbidden")
                    else "local_repair"
                )
            elif status == "requires_context_review":
                final_category = "missing_context"
                invalid_types[record_id] = "missing_context"
            else:
                invalid_type = invalid_type_from_diagnostics(str(audit.get("diagnostics") or ""))
                final_category = "missing_context" if invalid_type == "missing_context" else "invalid_statement"
                invalid_types[record_id] = invalid_type
        else:
            final_category = category
        updated = dict(entry)
        updated["repair_category"] = final_category
        updated["provisional_category"] = category
        final[final_category].append(updated)

    for category, entries in final.items():
        write_jsonl(args.output_root / f"{category}_manifest.jsonl", entries)
    write_jsonl(
        args.output_root / "invalid_routing_manifest.jsonl",
        ({"record_id": record_id, "invalid_type": invalid_type} for record_id, invalid_type in sorted(invalid_types.items())),
    )
    report = {
        "schema_version": SCHEMA,
        "command": "finalize",
        "category_counts": {key: len(value) for key, value in final.items()},
        "partition_total": sum(len(value) for value in final.values()),
        "invalid_type_counts": dict(Counter(invalid_types.values())),
        "audit_results_sha256": sha256_file(args.audit_results),
    }
    write_json(args.output_root / "final_routing_report.json", report)
    return report


def apply_invalid(args: argparse.Namespace) -> dict[str, Any]:
    invalid_routes = {
        str(row["record_id"]): str(row["invalid_type"])
        for row in iter_jsonl(args.invalid_manifest)
    }
    frozen_ids = {
        str(row["record_id"])
        for path in args.frozen_manifest
        for row in iter_jsonl(path)
        if row.get("record_id")
    }
    if set(invalid_routes) & frozen_ids:
        raise RuntimeError("invalid routing overlaps frozen records")
    existing_invalid = read_jsonl(args.invalid_file)
    existing_ids = {str(row["record_id"]) for row in existing_invalid}
    remaining = []
    moved = []
    for row in iter_jsonl(args.fail_file):
        record_id = str(row["record_id"])
        invalid_type = invalid_routes.get(record_id)
        if invalid_type is None:
            remaining.append(row)
            continue
        if record_id in existing_ids:
            raise RuntimeError(f"record already exists in numinamath_invalid: {record_id}")
        updated = dict(row)
        updated["invalid_type"] = invalid_type
        moved.append(updated)
    if len(moved) != len(invalid_routes):
        raise RuntimeError(f"not all invalid routes found in fail: {len(moved)} != {len(invalid_routes)}")
    before_total = len(remaining) + len(moved)
    write_jsonl(args.fail_file, remaining)
    write_jsonl(args.invalid_file, [*existing_invalid, *moved])
    report = {
        "schema_version": SCHEMA,
        "command": "apply-invalid",
        "fail_before": before_total,
        "moved_to_numinamath_invalid": len(moved),
        "fail_after": len(remaining),
        "numinamath_invalid_after": len(existing_invalid) + len(moved),
        "invalid_type_counts": dict(Counter(row["invalid_type"] for row in moved)),
        "fail_sha256": sha256_file(args.fail_file),
        "invalid_sha256": sha256_file(args.invalid_file),
        "frozen_hashes": {str(path): sha256_file(path) for path in args.frozen_manifest},
        "only_added_field_to_moved_rows": "invalid_type",
    }
    write_json(args.report, report)
    return report


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    classify_parser = sub.add_parser("classify")
    classify_parser.add_argument("--fail-file", type=Path, default=root / "verified_data" / "numinamath_verified_fail.jsonl")
    classify_parser.add_argument("--output-root", type=Path, required=True)
    classify_parser.add_argument("--frozen-manifest", type=Path, action="append", default=[])

    audit_parser = sub.add_parser("audit-statements")
    audit_parser.add_argument("--fail-file", type=Path, default=root / "verified_data" / "numinamath_verified_fail.jsonl")
    audit_parser.add_argument("--audit-manifest", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--report", type=Path, required=True)
    audit_parser.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    audit_parser.add_argument("--timeout", type=int, default=30)

    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--routing-root", type=Path, required=True)
    finalize_parser.add_argument("--audit-results", type=Path, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)

    apply_parser = sub.add_parser("apply-invalid")
    apply_parser.add_argument("--fail-file", type=Path, default=root / "verified_data" / "numinamath_verified_fail.jsonl")
    apply_parser.add_argument("--invalid-file", type=Path, default=root / "verified_data" / "numinamath_invalid.jsonl")
    apply_parser.add_argument("--invalid-manifest", type=Path, required=True)
    apply_parser.add_argument("--frozen-manifest", type=Path, action="append", default=[])
    apply_parser.add_argument("--report", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "classify":
        report = classify(args)
    elif args.command == "audit-statements":
        report = audit_statements(args)
    elif args.command == "finalize":
        report = finalize(args)
    else:
        report = apply_invalid(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
