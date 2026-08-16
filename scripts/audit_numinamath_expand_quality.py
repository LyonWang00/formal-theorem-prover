#!/usr/bin/env python3
"""Build and validate a human-review ledger for NuminaMath expansions.

This tool is intentionally separated from Pantograph compilation and from the
writers of ``*_verified_success.jsonl`` / ``*_verified_fail.jsonl``.  It never
changes the input dataset.  Its outputs are:

* a deterministic reviewer packet JSONL (safe to rebuild atomically),
* an append-only ledger containing deterministic seed decisions and manual
  decision events, and
* JSON summaries / validation reports.

No candidate is automatically accepted.  Ten known no-value transformations
are locked to ``drop``.  Scope/parser hazards are marked ``repair`` or ``drop``;
all other candidates start as ``pending``.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCRIPT_VERSION = "numinamath_expand_quality_audit_v1"
PACKET_SCHEMA = "numinamath_expand_quality_packet_v1"
LEDGER_SCHEMA = "numinamath_expand_quality_ledger_event_v1"
SUMMARY_SCHEMA = "numinamath_expand_quality_summary_v1"
VALIDATION_SCHEMA = "numinamath_expand_quality_validation_v1"

EXPECTED_INPUT_SHA256 = (
    "dd66edc02676dc3d459eef84304e02bc4354c3c63a5d4bd8791c78b1ee79d60e"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "lean_prover" / "Dataset"
DEFAULT_INPUT = DATASET_ROOT / "raw_data" / "numinamath_expand.jsonl"
DEFAULT_PARENTS = DATASET_ROOT / "verified_data" / "numinamath_verified_success.jsonl"
DEFAULT_AUDIT_DIR = DATASET_ROOT / "quality_audit"
DEFAULT_LEDGER = DEFAULT_AUDIT_DIR / "numinamath_expand_quality_ledger.jsonl"
DEFAULT_PACKETS = DEFAULT_AUDIT_DIR / "numinamath_expand_quality_packets.jsonl"
DEFAULT_SUMMARY = DEFAULT_AUDIT_DIR / "numinamath_expand_quality_summary.json"
DEFAULT_VALIDATION = DEFAULT_AUDIT_DIR / "numinamath_expand_quality_validation.json"

DECISIONS = frozenset({"pending", "accept_high", "accept_low", "repair", "drop"})
ACCEPT_DECISIONS = frozenset({"accept_high", "accept_low"})

# These methods carry no new mathematical content.  The expected total on the
# frozen input is 11,214.  Each reason is both machine-readable and Chinese so
# reviewers do not need an external taxonomy to interpret the ledger.
HARD_DROP_METHODS: dict[str, tuple[str, str]] = {
    "eq_add_one": (
        "extreme_low_eq_add_same_constant",
        "极低质量：仅在等式两边同时加一，没有新增数学内容。",
    ),
    "eq_square": (
        "extreme_low_eq_apply_same_square",
        "极低质量：仅在等式两边同时平方，没有形成独立有价值的新命题。",
    ),
    "eq_symmetry": (
        "extreme_low_eq_symmetry",
        "极低质量：仅交换等式两边。",
    ),
    "and_commute": (
        "extreme_low_and_commutation",
        "极低质量：仅交换合取命题顺序。",
    ),
    "or_commute": (
        "extreme_low_or_commutation",
        "极低质量：仅交换析取命题顺序。",
    ),
    "proof_argument_reorder": (
        "extreme_low_proof_argument_reorder",
        "极低质量：命题不变，仅重排 linarith/nlinarith 的参数。",
    ),
    "iff_as_implications": (
        "extreme_low_iff_repackaging",
        "极低质量：仅把充要条件机械拆装为两个蕴含。",
    ),
    "implication_as_disjunction": (
        "extreme_low_implication_disjunction_repackaging",
        "极低质量：仅把蕴含机械改写为经典逻辑析取。",
    ),
    "le_to_not_reverse_lt": (
        "extreme_low_le_negated_reverse",
        "极低质量：仅把 a ≤ b 改写为 ¬ b < a。",
    ),
    "lt_to_not_reverse_le": (
        "extreme_low_lt_negated_reverse",
        "极低质量：仅把 a < b 改写为 ¬ b ≤ a。",
    ),
}

RISK_REASONS: dict[str, str] = {
    "missing_parent": "无法在父数据中定位 upstream parent，必须人工修复来源。",
    "parent_goal_parse_error": "无法可靠提取父命题目标，禁止自动接受。",
    "candidate_goal_parse_error": "无法可靠提取扩充命题目标，禁止自动接受。",
    "scope_exists": "父目标以存在量词开头；生成器可能把量词内部连接符误当顶层结构。",
    "scope_forall": "父目标以全称量词开头；生成器可能丢失量词作用域。",
    "scope_negation": "父目标以否定开头；对内部关系的机械变换通常不保持原语义。",
    "scope_let_if": "父目标以 let/if 开头；机械切分可能跨越局部作用域。",
    "parent_multiple_declarations": "父 lean_statement 含多个 theorem/lemma 声明，存在拼接污染。",
    "candidate_multiple_declarations": "扩充 lean_statement 含多个 theorem/lemma 声明，存在拼接污染。",
    "parent_embedded_proof_assignment": "父 lean_statement 含 := by，超出 statement-only 契约。",
    "candidate_embedded_proof_assignment": "扩充 lean_statement 含 := by，目标可能混入证明文本。",
}

WARNING_REASONS: dict[str, str] = {
    "question_type_overwritten": "扩充记录用方法名覆盖了父记录 question_type，最终导出前应复核语义域。",
    "template_problem": "problem 为固定扩充模板加原题，需人工确认与新 formal target 一致。",
}

DECLARATION = re.compile(r"\b(?:theorem|lemma)\s+([^\s:{(]+)")


class AuditError(RuntimeError):
    """Raised when an audit invariant is violated."""


@dataclass(frozen=True)
class RawMeta:
    row_number: int
    record_id: str
    parent_id: str
    method: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise AuditError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def event_with_id(payload: Mapping[str, Any]) -> dict[str, Any]:
    event = dict(payload)
    event.pop("event_id", None)
    event["event_id"] = sha256_text(canonical_json(event))
    return event


def verify_event_id(event: Mapping[str, Any]) -> bool:
    claimed = str(event.get("event_id") or "")
    payload = dict(event)
    payload.pop("event_id", None)
    return claimed == sha256_text(canonical_json(payload))


def parent_id_from(row: Mapping[str, Any]) -> str:
    upstream = str(row.get("upstream_source") or "")
    return upstream[len("parent:") :] if upstream.startswith("parent:") else upstream


def declaration_count(statement: str) -> int:
    return len(DECLARATION.findall(statement))


def extract_goal(statement: str) -> str:
    """Extract the goal after the first declaration colon.

    This is a lexical extractor, not a Lean parser.  It intentionally reports
    contaminated trailing text so that the risk detector can quarantine it.
    """

    match = DECLARATION.search(statement)
    if match is None:
        raise AuditError("statement has no theorem/lemma declaration")
    depth = 0
    in_string = False
    escaped = False
    index = match.end()
    while index < len(statement):
        char = statement[index]
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
        elif char == ":" and depth == 0:
            if index + 1 < len(statement) and statement[index + 1] == "=":
                index += 2
                continue
            goal = statement[index + 1 :].strip()
            if not goal:
                raise AuditError("empty theorem goal")
            return goal
        index += 1
    raise AuditError("could not locate theorem goal colon")


def unwrap_outer_parentheses(value: str) -> str:
    value = value.strip()
    while len(value) >= 2 and value[0] == "(":
        depth = 0
        closing = None
        in_string = False
        escaped = False
        for index, char in enumerate(value):
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
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing != len(value) - 1:
            break
        value = value[1:-1].strip()
    return value


def leading_scope(goal: str) -> str | None:
    goal = unwrap_outer_parentheses(goal)
    if goal.startswith("∃"):
        return "exists"
    if goal.startswith("∀"):
        return "forall"
    if goal.startswith("¬"):
        return "negation"
    if goal.startswith("let ") or goal.startswith("if "):
        return "let_if"
    return None


def compact_unified_diff(
    before: str,
    after: str,
    before_name: str,
    after_name: str,
    max_lines: int,
) -> dict[str, Any]:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=before_name,
            tofile=after_name,
            lineterm="",
            n=3,
        )
    )
    total = len(lines)
    truncated = max_lines > 0 and total > max_lines
    if truncated:
        head = max_lines // 2
        tail = max_lines - head
        omitted = total - max_lines
        lines = lines[:head] + [f"... {omitted} diff lines omitted ..."] + lines[-tail:]
    return {
        "text": "\n".join(lines),
        "total_lines": total,
        "truncated": truncated,
        "before_sha256": sha256_text(before),
        "after_sha256": sha256_text(after),
    }


def collect_raw_meta(input_path: Path) -> tuple[list[RawMeta], dict[str, list[dict[str, str]]]]:
    metadata: list[RawMeta] = []
    siblings: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    seen: set[str] = set()
    for row_number, row in read_jsonl(input_path):
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise AuditError(f"{input_path}:{row_number}: missing record_id")
        if record_id in seen:
            raise AuditError(f"{input_path}:{row_number}: duplicate record_id {record_id}")
        seen.add(record_id)
        method = str(row.get("question_type") or "")
        parent_id = parent_id_from(row)
        metadata.append(RawMeta(row_number, record_id, parent_id, method))
        siblings[parent_id].append({"record_id": record_id, "method": method})
    for values in siblings.values():
        values.sort(key=lambda item: (item["method"], item["record_id"]))
    return metadata, siblings


def load_needed_parents(parent_path: Path, needed_ids: set[str]) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    for line_number, row in read_jsonl(parent_path):
        record_id = str(row.get("record_id") or "")
        if record_id not in needed_ids:
            continue
        if record_id in parents:
            raise AuditError(f"{parent_path}:{line_number}: duplicate parent record_id {record_id}")
        parents[record_id] = row
    return parents


def risk_analysis(
    candidate: Mapping[str, Any],
    method: str,
    parent: Mapping[str, Any] | None,
) -> tuple[str, bool, list[str], list[str], list[str], str, str]:
    """Return decision, lock, reason codes/text, warnings, and both goals."""

    reasons: list[str] = []
    warnings: list[str] = []
    parent_goal = ""
    candidate_goal = ""

    if parent is None:
        reasons.append("missing_parent")
    else:
        parent_statement = str(parent.get("lean_statement") or "")
        try:
            parent_goal = extract_goal(parent_statement)
        except AuditError:
            reasons.append("parent_goal_parse_error")
        if declaration_count(parent_statement) > 1:
            reasons.append("parent_multiple_declarations")
        if ":= by" in parent_statement:
            reasons.append("parent_embedded_proof_assignment")

    candidate_statement = str(candidate.get("lean_statement") or "")
    try:
        candidate_goal = extract_goal(candidate_statement)
    except AuditError:
        reasons.append("candidate_goal_parse_error")
    if declaration_count(candidate_statement) > 1:
        reasons.append("candidate_multiple_declarations")
    if ":= by" in candidate_statement:
        reasons.append("candidate_embedded_proof_assignment")

    # proof_argument_reorder does not transform the goal, so a leading binder
    # is not itself a scope-cut risk for that one method.
    if parent_goal and method != "proof_argument_reorder":
        scope = leading_scope(parent_goal)
        if scope is not None:
            reasons.append(f"scope_{scope}")

    if parent is not None and str(parent.get("question_type")) != str(method):
        warnings.append("question_type_overwritten")
    if "Original problem:" in str(candidate.get("problem") or ""):
        warnings.append("template_problem")

    hard = HARD_DROP_METHODS.get(method)
    if hard is not None:
        code, chinese = hard
        codes = [code] + sorted(set(reasons))
        text = [chinese] + [RISK_REASONS[code] for code in sorted(set(reasons))]
        return "drop", True, codes, text, sorted(set(warnings)), parent_goal, candidate_goal

    unique_reasons = sorted(set(reasons))
    if unique_reasons:
        # Negated goals are normally invalid under the generator's inner
        # relation rewrite.  Default to drop, but leave the event unlocked so
        # a reviewer may explicitly repair and recompile it.
        decision = "drop" if "scope_negation" in unique_reasons else "repair"
        return (
            decision,
            False,
            unique_reasons,
            [RISK_REASONS[code] for code in unique_reasons],
            sorted(set(warnings)),
            parent_goal,
            candidate_goal,
        )

    return "pending", False, [], [], sorted(set(warnings)), parent_goal, candidate_goal


def seed_event_from_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    return event_with_id(
        {
            "schema_version": LEDGER_SCHEMA,
            "script_version": SCRIPT_VERSION,
            "event_type": "seed",
            "input_sha256": packet["input_sha256"],
            "source_row": packet["source_row"],
            "record_id": packet["record_id"],
            "parent_id": packet["parent_id"],
            "method": packet["method"],
            "machine_decision": packet["machine_decision"],
            "decision": packet["machine_decision"],
            "decision_source": "machine",
            "decision_locked": packet["decision_locked"],
            "reason_codes": packet["reason_codes"],
            "reason_zh": packet["reason_zh"],
            "warning_codes": packet["warning_codes"],
        }
    )


def build_packet(
    row_number: int,
    candidate: Mapping[str, Any],
    parent: Mapping[str, Any] | None,
    siblings: Sequence[Mapping[str, str]],
    input_sha256: str,
    parent_sha256: str,
    max_proof_diff_lines: int,
) -> dict[str, Any]:
    record_id = str(candidate.get("record_id") or "")
    parent_id = parent_id_from(candidate)
    method = str(candidate.get("question_type") or "")
    decision, locked, reason_codes, reason_zh, warnings, parent_goal, candidate_goal = (
        risk_analysis(candidate, method, parent)
    )
    parent_statement = str(parent.get("lean_statement") or "") if parent else ""
    parent_proof = str(parent.get("proof") or "") if parent else ""
    candidate_statement = str(candidate.get("lean_statement") or "")
    candidate_proof = str(candidate.get("proof") or "")
    return {
        "schema_version": PACKET_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "input_sha256": input_sha256,
        "parent_source_sha256": parent_sha256,
        "source_row": row_number,
        "record_id": record_id,
        "parent_id": parent_id,
        "method": method,
        "machine_decision": decision,
        "decision_locked": locked,
        "reason_codes": reason_codes,
        "reason_zh": reason_zh,
        "warning_codes": warnings,
        "warning_zh": [WARNING_REASONS[code] for code in warnings],
        "siblings": list(siblings),
        "parent": {
            "record_id": parent_id,
            "question_type": parent.get("question_type") if parent else None,
            "lean_statement": parent_statement,
            "goal": parent_goal,
            "proof_sha256": sha256_text(parent_proof),
            "proof_line_count": len(parent_proof.splitlines()),
        },
        "candidate": {
            "question_type": candidate.get("question_type"),
            "lean_statement": candidate_statement,
            "goal": candidate_goal,
            "proof_sha256": sha256_text(candidate_proof),
            "proof_line_count": len(candidate_proof.splitlines()),
        },
        "diff": {
            "statement": compact_unified_diff(
                parent_statement,
                candidate_statement,
                "parent_statement",
                "candidate_statement",
                0,
            ),
            "goal": compact_unified_diff(
                parent_goal, candidate_goal, "parent_goal", "candidate_goal", 0
            ),
            "proof": compact_unified_diff(
                parent_proof,
                candidate_proof,
                "parent_proof",
                "candidate_proof",
                max_proof_diff_lines,
            ),
        },
    }


def parse_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [event for _, event in read_jsonl(path)]


def append_missing_seeds(path: Path, seeds: Sequence[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if event.get("event_type") == "seed":
                record_id = str(event.get("record_id") or "")
                if record_id in existing:
                    raise AuditError(f"{path}:{line_number}: duplicate seed for {record_id}")
                existing[record_id] = event
        appended = 0
        handle.seek(0, os.SEEK_END)
        for seed_value in seeds:
            seed = dict(seed_value)
            record_id = str(seed["record_id"])
            previous = existing.get(record_id)
            if previous is not None:
                if canonical_json(previous) != canonical_json(seed):
                    raise AuditError(
                        f"existing seed for {record_id} differs from deterministic rebuild"
                    )
                continue
            handle.write(canonical_json(seed) + "\n")
            existing[record_id] = seed
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return appended


def check_expected_sha(actual: str, expected: str) -> None:
    if expected and actual != expected:
        raise AuditError(f"input SHA-256 mismatch: expected {expected}, got {actual}")


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    parent_path = Path(args.parents)
    packet_path = Path(args.packets)
    ledger_path = Path(args.ledger)
    summary_path = Path(args.summary)

    input_sha = sha256_file(input_path)
    check_expected_sha(input_sha, str(args.expected_input_sha256 or ""))
    parent_sha = sha256_file(parent_path)
    metadata, sibling_map = collect_raw_meta(input_path)
    needed = {meta.parent_id for meta in metadata if meta.parent_id}
    parents = load_needed_parents(parent_path, needed)

    packet_path.parent.mkdir(parents=True, exist_ok=True)
    seeds: list[dict[str, Any]] = []
    decision_counts: collections.Counter[str] = collections.Counter()
    method_counts: collections.Counter[str] = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    warning_counts: collections.Counter[str] = collections.Counter()
    hard_drop_counts: collections.Counter[str] = collections.Counter()

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=packet_path.parent, prefix=f".{packet_path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for (line_number, candidate), meta in zip(read_jsonl(input_path), metadata, strict=True):
            if line_number != meta.row_number or str(candidate.get("record_id")) != meta.record_id:
                raise AuditError("input changed while packets were being generated")
            packet = build_packet(
                line_number,
                candidate,
                parents.get(meta.parent_id),
                sibling_map.get(meta.parent_id, []),
                input_sha,
                parent_sha,
                int(args.max_proof_diff_lines),
            )
            handle.write(canonical_json(packet) + "\n")
            seeds.append(seed_event_from_packet(packet))
            decision_counts[packet["machine_decision"]] += 1
            method_counts[packet["method"]] += 1
            for code in packet["reason_codes"]:
                reason_counts[code] += 1
            for code in packet["warning_codes"]:
                warning_counts[code] += 1
            if packet["method"] in HARD_DROP_METHODS:
                hard_drop_counts[packet["method"]] += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, packet_path)
    packet_sha = sha256_file(packet_path)
    appended = append_missing_seeds(ledger_path, seeds)
    ledger_events = parse_ledger(ledger_path)

    hard_drop_total = sum(hard_drop_counts.values())
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "input": str(input_path),
        "input_sha256": input_sha,
        "input_rows": len(metadata),
        "parent_source": str(parent_path),
        "parent_source_sha256": parent_sha,
        "parents_resolved": len(parents),
        "parents_missing": len(needed - set(parents)),
        "packets": str(packet_path),
        "packets_sha256": packet_sha,
        "ledger": str(ledger_path),
        "ledger_events": len(ledger_events),
        "new_seed_events_appended": appended,
        "machine_decisions": dict(sorted(decision_counts.items())),
        "methods": dict(sorted(method_counts.items())),
        "reason_codes": dict(sorted(reason_counts.items())),
        "warning_codes": dict(sorted(warning_counts.items())),
        "hard_drop_methods": dict(sorted(hard_drop_counts.items())),
        "hard_drop_total": hard_drop_total,
        "expected_hard_drop_total": 11214,
        "hard_drop_total_matches_expected": hard_drop_total == 11214,
        "automatic_accept_count": 0,
        "max_proof_diff_lines": int(args.max_proof_diff_lines),
    }
    atomic_write_json(summary_path, summary)
    if args.enforce_expected_counts and hard_drop_total != 11214:
        raise AuditError(
            f"hard-drop total mismatch: expected 11214, got {hard_drop_total}"
        )
    return summary


def append_manual_decision(args: argparse.Namespace) -> dict[str, Any]:
    input_sha = sha256_file(Path(args.input))
    check_expected_sha(input_sha, str(args.expected_input_sha256 or ""))
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    decision = str(args.decision)
    if decision not in DECISIONS - {"pending"}:
        raise AuditError(f"invalid manual decision: {decision}")
    if not str(args.reviewer or "").strip():
        raise AuditError("--reviewer is required")
    if not str(args.note or "").strip():
        raise AuditError("--note is required")

    with ledger_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AuditError(f"{ledger_path}:{line_number}: invalid JSON: {error}") from error
        record_events = [event for event in events if event.get("record_id") == args.record_id]
        if not record_events:
            raise AuditError(f"record_id has no seed event: {args.record_id}")
        seed = next((event for event in record_events if event.get("event_type") == "seed"), None)
        if seed is None:
            raise AuditError(f"record_id has no seed event: {args.record_id}")
        if seed.get("input_sha256") != input_sha:
            raise AuditError("seed input SHA does not match the current input")
        if bool(seed.get("decision_locked")) and decision != "drop":
            raise AuditError("hard-drop decision is locked and cannot be accepted or repaired")
        if decision in ACCEPT_DECISIONS and args.pantograph_result != "success":
            raise AuditError("accept decisions require --pantograph-result success")
        if (
            decision in ACCEPT_DECISIONS
            and seed.get("machine_decision") != "pending"
            and not str(args.repair_evidence or "").strip()
        ):
            raise AuditError(
                "accepting a machine-flagged record requires --repair-evidence after recompile"
            )
        previous = record_events[-1]
        payload = {
            "schema_version": LEDGER_SCHEMA,
            "script_version": SCRIPT_VERSION,
            "event_type": "manual_decision",
            "input_sha256": input_sha,
            "record_id": args.record_id,
            "parent_id": seed.get("parent_id"),
            "method": seed.get("method"),
            "decision": decision,
            "decision_source": "human",
            "reviewer": str(args.reviewer),
            "note": str(args.note),
            "pantograph_result": str(args.pantograph_result),
            "repair_evidence": str(args.repair_evidence or ""),
            "previous_event_id": previous.get("event_id"),
        }
        event = event_with_id(payload)
        if event["event_id"] == previous.get("event_id"):
            raise AuditError("manual event is identical to the latest event")
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return event


def low_ratio_is_valid(low: int, accepted_total: int) -> bool:
    return accepted_total == 0 or (low / accepted_total) < 0.05


def validate_outputs(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    input_sha = sha256_file(input_path)
    check_expected_sha(input_sha, str(args.expected_input_sha256 or ""))
    metadata, _ = collect_raw_meta(input_path)
    input_ids = {meta.record_id for meta in metadata}
    errors: list[str] = []

    packet_ids: set[str] = set()
    packet_seed_view: dict[str, tuple[str, bool, tuple[str, ...]]] = {}
    packet_path = Path(args.packets)
    if not packet_path.exists():
        errors.append(f"packet file does not exist: {packet_path}")
    else:
        for line_number, packet in read_jsonl(packet_path):
            record_id = str(packet.get("record_id") or "")
            if packet.get("input_sha256") != input_sha:
                errors.append(f"packet line {line_number} has a different input SHA")
            if record_id in packet_ids:
                errors.append(f"duplicate packet record_id: {record_id}")
            packet_ids.add(record_id)
            packet_seed_view[record_id] = (
                str(packet.get("machine_decision")),
                bool(packet.get("decision_locked")),
                tuple(packet.get("reason_codes") or ()),
            )

    events = parse_ledger(Path(args.ledger))
    seeds: dict[str, dict[str, Any]] = {}
    latest: dict[str, dict[str, Any]] = {}
    last_event_ids: dict[str, str] = {}
    for event_number, event in enumerate(events, 1):
        record_id = str(event.get("record_id") or "")
        if event.get("schema_version") != LEDGER_SCHEMA:
            errors.append(f"ledger event {event_number} has an unknown schema")
        if event.get("input_sha256") != input_sha:
            errors.append(f"ledger event {event_number} has a different input SHA")
        if not verify_event_id(event):
            errors.append(f"ledger event {event_number} has an invalid event_id")
        event_type = event.get("event_type")
        if event_type == "seed":
            if record_id in seeds:
                errors.append(f"duplicate seed for {record_id}")
            seeds[record_id] = event
            if record_id in last_event_ids:
                errors.append(f"seed for {record_id} appears after another event")
        elif event_type == "manual_decision":
            if record_id not in seeds:
                errors.append(f"manual decision before seed for {record_id}")
            expected_previous = last_event_ids.get(record_id)
            if event.get("previous_event_id") != expected_previous:
                errors.append(f"broken event chain for {record_id}")
            decision = str(event.get("decision"))
            if decision not in DECISIONS - {"pending"}:
                errors.append(f"invalid manual decision for {record_id}: {decision}")
            seed = seeds.get(record_id, {})
            if bool(seed.get("decision_locked")) and decision != "drop":
                errors.append(f"locked hard-drop was overridden for {record_id}")
            if decision in ACCEPT_DECISIONS and event.get("pantograph_result") != "success":
                errors.append(f"accepted record lacks Pantograph success: {record_id}")
            if (
                decision in ACCEPT_DECISIONS
                and seed.get("machine_decision") != "pending"
                and not str(event.get("repair_evidence") or "").strip()
            ):
                errors.append(f"flagged accepted record lacks repair evidence: {record_id}")
        else:
            errors.append(f"ledger event {event_number} has invalid event_type {event_type}")
        last_event_ids[record_id] = str(event.get("event_id") or "")
        latest[record_id] = event

    for record_id, seed in seeds.items():
        packet_view = packet_seed_view.get(record_id)
        seed_view = (
            str(seed.get("machine_decision")),
            bool(seed.get("decision_locked")),
            tuple(seed.get("reason_codes") or ()),
        )
        if packet_view is not None and seed_view != packet_view:
            errors.append(f"seed/packet classification differs for {record_id}")

    missing_packets = input_ids - packet_ids
    extra_packets = packet_ids - input_ids
    missing_seeds = input_ids - set(seeds)
    extra_seeds = set(seeds) - input_ids
    if missing_packets:
        errors.append(f"missing packets: {len(missing_packets)}")
    if extra_packets:
        errors.append(f"extra packets: {len(extra_packets)}")
    if missing_seeds:
        errors.append(f"missing seeds: {len(missing_seeds)}")
    if extra_seeds:
        errors.append(f"extra seeds: {len(extra_seeds)}")

    effective_counts: collections.Counter[str] = collections.Counter()
    for record_id in input_ids:
        decision = str(latest.get(record_id, {}).get("decision") or "missing")
        effective_counts[decision] += 1
    accepted_high = effective_counts["accept_high"]
    accepted_low = effective_counts["accept_low"]
    accepted_total = accepted_high + accepted_low
    low_ratio = accepted_low / accepted_total if accepted_total else 0.0
    if not low_ratio_is_valid(accepted_low, accepted_total):
        errors.append(
            f"accepted low-quality ratio must be <5%, got {accepted_low}/{accepted_total}={low_ratio:.6f}"
        )
    if args.require_final:
        unfinished = effective_counts["pending"] + effective_counts["repair"] + effective_counts["missing"]
        if unfinished:
            errors.append(f"require-final failed: {unfinished} records are unfinished")

    report = {
        "schema_version": VALIDATION_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "input": str(input_path),
        "input_sha256": input_sha,
        "input_rows": len(metadata),
        "packet_rows": len(packet_ids),
        "seed_rows": len(seeds),
        "ledger_events": len(events),
        "record_id_completeness": not (missing_packets or extra_packets or missing_seeds or extra_seeds),
        "effective_decisions": dict(sorted(effective_counts.items())),
        "accepted_high": accepted_high,
        "accepted_low": accepted_low,
        "accepted_total": accepted_total,
        "accepted_low_ratio": low_ratio,
        "accepted_low_ratio_strictly_below_5_percent": low_ratio_is_valid(
            accepted_low, accepted_total
        ),
        "require_final": bool(args.require_final),
        "valid": not errors,
        "errors": errors,
    }
    atomic_write_json(Path(args.report), report)
    if errors:
        raise AuditError("validation failed:\n- " + "\n- ".join(errors[:30]))
    return report


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def run_selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="numinamath-quality-selftest-") as directory:
        root = Path(directory)
        parents_path = root / "parents.jsonl"
        input_path = root / "input.jsonl"
        packets_path = root / "packets.jsonl"
        ledger_path = root / "ledger.jsonl"
        summary_path = root / "summary.json"
        report_path = root / "validation.json"

        parents = [
            {
                "record_id": "p-hard",
                "question_type": "math-word-problem",
                "lean_statement": "import Mathlib\ntheorem pHard (x : ℝ) : x = x",
                "proof": "by rfl",
            },
            {
                "record_id": "p-scope",
                "question_type": "proof",
                "lean_statement": "import Mathlib\ntheorem pScope (f : ℕ → ℕ) : ∀ n, f n = f n",
                "proof": "by intro n; rfl",
            },
            {
                "record_id": "p-clean",
                "question_type": "MCQ",
                "lean_statement": "import Mathlib\ntheorem pClean (x : ℝ) : x = x",
                "proof": "by rfl",
            },
            {
                "record_id": "p-multi",
                "question_type": "proof",
                "lean_statement": (
                    "import Mathlib\ntheorem pMulti : True := by trivial\n"
                    "theorem pOther : True"
                ),
                "proof": "by trivial",
            },
        ]
        candidates = [
            {
                "record_id": "c-hard",
                "upstream_source": "parent:p-hard",
                "question_type": "eq_add_one",
                "lean_statement": "import Mathlib\ntheorem cHard (x : ℝ) : 1 + x = 1 + x",
                "proof": "by rfl",
                "problem": "Original problem: hard",
            },
            {
                "record_id": "c-scope",
                "upstream_source": "parent:p-scope",
                "question_type": "eq_to_le_forward",
                "lean_statement": "import Mathlib\ntheorem cScope (f : ℕ → ℕ) : True",
                "proof": "by trivial",
                "problem": "Original problem: scope",
            },
            {
                "record_id": "c-clean",
                "upstream_source": "parent:p-clean",
                "question_type": "eq_to_le_forward",
                "lean_statement": "import Mathlib\ntheorem cClean (x : ℝ) : x ≤ x",
                "proof": "by exact le_rfl",
                "problem": "Original problem: clean",
            },
            {
                "record_id": "c-multi",
                "upstream_source": "parent:p-multi",
                "question_type": "and_left",
                "lean_statement": "import Mathlib\ntheorem cMulti : True",
                "proof": "by trivial",
                "problem": "Original problem: multi",
            },
        ]
        write_jsonl(parents_path, parents)
        write_jsonl(input_path, candidates)
        input_sha = sha256_file(input_path)
        common = argparse.Namespace(
            input=input_path,
            parents=parents_path,
            packets=packets_path,
            ledger=ledger_path,
            summary=summary_path,
            report=report_path,
            expected_input_sha256=input_sha,
            max_proof_diff_lines=100,
            enforce_expected_counts=False,
            require_final=False,
        )
        summary = build_outputs(common)
        assert summary["input_rows"] == 4
        first_packet_sha = summary["packets_sha256"]
        rebuilt = build_outputs(common)
        assert rebuilt["new_seed_events_appended"] == 0
        assert rebuilt["packets_sha256"] == first_packet_sha
        assert rebuilt["ledger_events"] == 4
        packets = {row["record_id"]: row for _, row in read_jsonl(packets_path)}
        assert packets["c-hard"]["machine_decision"] == "drop"
        assert packets["c-hard"]["decision_locked"] is True
        assert packets["c-scope"]["machine_decision"] == "repair"
        assert "scope_forall" in packets["c-scope"]["reason_codes"]
        assert packets["c-clean"]["machine_decision"] == "pending"
        assert packets["c-multi"]["machine_decision"] == "repair"
        validate_outputs(common)

        decide = argparse.Namespace(
            input=input_path,
            expected_input_sha256=input_sha,
            ledger=ledger_path,
            record_id="c-clean",
            decision="accept_high",
            reviewer="selftest",
            note="clean test record",
            pantograph_result="success",
            repair_evidence="",
        )
        append_manual_decision(decide)
        validate_outputs(common)
        rebuilt_after_manual = build_outputs(common)
        assert rebuilt_after_manual["new_seed_events_appended"] == 0
        assert rebuilt_after_manual["ledger_events"] == 5

        decide.record_id = "c-hard"
        try:
            append_manual_decision(decide)
        except AuditError:
            pass
        else:
            raise AssertionError("hard-drop override was not rejected")

        decide.record_id = "c-scope"
        try:
            append_manual_decision(decide)
        except AuditError:
            pass
        else:
            raise AssertionError("flagged accept without repair evidence was not rejected")

        assert not low_ratio_is_valid(1, 20)  # exactly 5% is forbidden
        assert low_ratio_is_valid(1, 21)
    print("selftest: PASS")


def add_common_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--expected-input-sha256",
        default=EXPECTED_INPUT_SHA256,
        help="fail closed if the raw input no longer matches this SHA-256",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="rebuild packets and append missing seed events")
    add_common_input_arguments(build)
    build.add_argument("--parents", type=Path, default=DEFAULT_PARENTS)
    build.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    build.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    build.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    build.add_argument(
        "--max-proof-diff-lines",
        type=int,
        default=400,
        help="0 keeps the complete proof diff; positive values keep deterministic head/tail excerpts",
    )
    build.add_argument(
        "--enforce-expected-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require the frozen input to contain exactly 11,214 hard-drop records",
    )

    decide = subparsers.add_parser("decide", help="append one human decision event")
    add_common_input_arguments(decide)
    decide.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    decide.add_argument("--record-id", required=True)
    decide.add_argument(
        "--decision", required=True, choices=sorted(DECISIONS - {"pending"})
    )
    decide.add_argument("--reviewer", required=True)
    decide.add_argument("--note", required=True)
    decide.add_argument(
        "--pantograph-result", choices=("success", "fail", "not_run"), default="not_run"
    )
    decide.add_argument(
        "--repair-evidence",
        default="",
        help="required to accept a record initially flagged repair/drop",
    )

    validate = subparsers.add_parser("validate", help="validate ledger integrity and review gates")
    add_common_input_arguments(validate)
    validate.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    validate.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    validate.add_argument("--report", type=Path, default=DEFAULT_VALIDATION)
    validate.add_argument(
        "--require-final",
        action="store_true",
        help="also require that no pending/repair records remain",
    )

    subparsers.add_parser("selftest", help="run isolated unit/self checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "build":
            result = build_outputs(args)
        elif args.command == "decide":
            result = append_manual_decision(args)
        elif args.command == "validate":
            result = validate_outputs(args)
        elif args.command == "selftest":
            run_selftest()
            return
        else:  # pragma: no cover - argparse prevents this
            raise AuditError(f"unsupported command: {args.command}")
    except AuditError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
