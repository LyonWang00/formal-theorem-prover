"""LeanDojo Benchmark 4 ingestion without coupling it to Lean-Workbook.

LeanDojo stores theorem locations and tactic traces in the benchmark split,
while declaration text and imports live in ``corpus.jsonl``.  This module
joins those sources and reconstructs a self-contained theorem from the first
proof state so that namespace, universes, variables, hypotheses, and instances
are retained for strict Pantograph verification.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..contracts import DataState, LeanDataRecord, SourceSpan
from ..preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    ProofFormat,
    contains_forbidden_proof_token,
    lean_code_tokens,
    statement_hash,
)


LEANDOJO_DATASET = "LeanDojo Benchmark 4"
LEANDOJO_CONTEXT_RECOVERY_VERSION = "leandojo_proof_state_v1"
FORBIDDEN_DECLARATION_TOKENS = {"axiom", "unsafe"}
_LOCAL_DECLARATION = re.compile(r"^(.+?)\s+:\s+(.+)$")
_SYNTHETIC_UNIVERSE = re.compile(r"\?u\.(\d+)")
_SYNTHETIC_LOCAL = re.compile(
    r"[A-Za-z_α-ωΑ-Ω][^\s:(),{}\[\]]*?[✝†][⁰¹²³⁴⁵⁶⁷⁸⁹]*"
)
_IDENTIFIER = re.compile(r"(?:[^\W\d]|_)[\w']*", re.UNICODE)


def normalize_leandojo_sft_record(
    record: Mapping[str, Any],
    index: int,
    *,
    source_name: str = LEANDOJO_DATASET,
):
    """Adapt one already Pantograph-verified LeanDojo SFT record."""

    from ..preparation import normalize_generic_lean_example

    if str(record.get("pantograph_verified") or "").lower() not in {
        "success",
        "true",
    }:
        raise ValueError("LeanDojo SFT row is not Pantograph verified")
    payload = dict(record)
    payload["statement"] = str(
        record.get("training_statement") or record.get("statement") or ""
    )
    payload["proof"] = str(
        record.get("training_proof") or record.get("proof") or ""
    )
    normalized = normalize_generic_lean_example(
        payload,
        source="leandojo-sft",
        index=index,
        source_name=source_name,
    )
    proof = normalized.proof.strip()
    if proof and not (
        proof == "by" or proof.startswith(("by ", "by\n"))
    ):
        proof = "by\n  exact " + proof.replace("\n", "\n  ")
    namespace_parts = [str(value) for value in record.get("namespace_stack") or ()]
    qualified_namespaces = [
        ".".join(namespace_parts[:index])
        for index in range(1, len(namespace_parts) + 1)
    ]
    raw_open_namespaces = _plain_open_namespaces(
        record.get("open_namespaces") or ()
    )
    open_namespaces = list(
        dict.fromkeys(
            raw_open_namespaces + qualified_namespaces
        )
    )
    open_namespaces = [
        value
        for value in open_namespaces
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*", value)
    ]
    if "MeasureTheory.Lp" in open_namespaces:
        open_namespaces = [
            value for value in open_namespaces if value != "MeasureTheory.Lp"
        ]
    metadata = record.get("metadata") or {}
    assembled_source = str(metadata.get("assembled_source") or "")
    declaration_source = str(record.get("declaration_source") or "")
    source_prefix = assembled_source.split(declaration_source, 1)[0]
    if declaration_source not in assembled_source:
        source_prefix = assembled_source
    if namespace_parts in (["MeasureTheory", "Lp"], ["MeasureTheory.Lp"]):
        normalized = replace(
            normalized,
            lean_statement=_qualify_namespace_locals(
                normalized.lean_statement,
                namespace="MeasureTheory.Lp",
                source_prefix=source_prefix,
            ),
        )
        proof = _qualify_namespace_locals(
            proof,
            namespace="MeasureTheory.Lp",
            source_prefix=source_prefix,
        )
    context: list[str] = _standalone_declaration_context(
        normalized.lean_statement,
        variables=record.get("variables") or (),
        local_instances=record.get("local_instances") or (),
        source_prefix=source_prefix,
    )
    if open_namespaces:
        context.insert(0, "open " + " ".join(open_namespaces))
    open_scopes = [
        str(value)
        for value in record.get("open_scopes") or ()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", str(value))
    ]
    if open_scopes:
        context.insert(1 if open_namespaces else 0, "open scoped " + " ".join(open_scopes))
    return replace(
        normalized,
        proof=proof,
        context_lines=tuple(context) + normalized.context_lines,
        pantograph_verified=True,
    )


def _standalone_declaration_context(
    statement: str,
    *,
    variables: Iterable[Mapping[str, Any]],
    local_instances: Iterable[Any],
    source_prefix: str,
) -> list[str]:
    """Recover only section variables needed by a standalone declaration.

    Verified LeanDojo rows retain a declaration exactly as it appeared inside
    its source file. Section variables are consequently not repeated in that
    declaration. Training prompts cannot keep namespace/section blocks open
    after the proof hole, so this adapter emits equivalent top-level variable
    commands instead of copying the complete source prefix.
    """

    variable_rows = [dict(value) for value in variables if isinstance(value, Mapping)]
    explicit_binders = _declaration_bound_names(statement)
    statement_tokens = set(_IDENTIFIER.findall(statement))
    selected = {
        str(row.get("name") or "")
        for row in variable_rows
        if str(row.get("name") or "") in statement_tokens
        and str(row.get("name") or "") not in explicit_binders
    }

    changed = True
    while changed:
        changed = False
        selected_types = " ".join(
            str(row.get("type") or "")
            for row in variable_rows
            if str(row.get("name") or "") in selected
        )
        type_tokens = set(_IDENTIFIER.findall(selected_types))
        for row in variable_rows:
            name = str(row.get("name") or "")
            if name and name in type_tokens and name not in selected:
                selected.add(name)
                changed = True

    # Some Mathlib APIs carry the measurable-space parameter as an ordinary
    # implicit variable while downstream notation expects it as an instance.
    for row in variable_rows:
        name = str(row.get("name") or "")
        type_expression = str(row.get("type") or "").strip()
        if not type_expression.startswith("MeasurableSpace "):
            continue
        dependencies = set(_IDENTIFIER.findall(type_expression)) - {
            "MeasurableSpace"
        }
        if dependencies <= selected and (
            "Measure " in statement or "∂" in statement or "μ" in statement
        ):
            selected.add(name)

    selected_instances: list[str] = []
    for value in local_instances:
        instance = _repair_instance_fragment(
            str(value).strip(), source_prefix, required_names=selected
        )
        if not _balanced_lean_fragment(instance):
            continue
        if set(_IDENTIFIER.findall(instance)) & selected:
            selected_instances.append(instance)

    fragments = [
        str(row.get("type") or "")
        for row in variable_rows
        if str(row.get("name") or "") in selected
    ] + selected_instances
    universes = sorted(
        {
            match.group(1)
            for fragment in fragments
            for match in re.finditer(r"(?:\bType\s+|\.\{)([a-z][A-Za-z0-9_]*)", fragment)
        }
    )
    context = ["universe " + " ".join(universes)] if universes else []
    deferred_variables: list[str] = []
    for row in variable_rows:
        name = str(row.get("name") or "")
        type_expression = _qualify_context_type(str(row.get("type") or "").strip())
        if name not in selected or not type_expression:
            continue
        if not _balanced_lean_fragment(type_expression):
            continue
        binder = str(row.get("binder") or "explicit")
        if type_expression.startswith("MeasurableSpace "):
            binder = "instance"
        delimiters = {
            "implicit": ("{", "}"),
            "instance": ("[", "]"),
        }.get(binder, ("(", ")"))
        command = f"variable {delimiters[0]}{name} : {type_expression}{delimiters[1]}"
        if re.fullmatch(r"Type(?:\s+[A-Za-z_][A-Za-z0-9_]*)?", type_expression):
            context.append(command)
        else:
            deferred_variables.append(command)
    context.extend(f"variable [{instance}]" for instance in selected_instances)
    context.extend(deferred_variables)
    return context


def _declaration_bound_names(statement: str) -> set[str]:
    """Extract identifiers declared before the first top-level colon per binder."""

    bound: set[str] = set()
    index = 0
    while index < len(statement):
        opening = statement[index]
        if opening not in "({[":
            index += 1
            continue
        closing = {"(": ")", "{": "}", "[": "]"}[opening]
        depth = 1
        cursor = index + 1
        colon = None
        while cursor < len(statement) and depth:
            character = statement[cursor]
            if character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
            elif character == ":" and depth == 1 and colon is None:
                colon = cursor
            cursor += 1
        if depth == 0 and colon is not None:
            head = statement[index + 1 : colon]
            bound.update(_IDENTIFIER.findall(head))
        index = max(cursor, index + 1)
    return bound


def _balanced_lean_fragment(value: str) -> bool:
    pairs = {"(": ")", "{": "}", "[": "]"}
    stack: list[str] = []
    for character in value:
        if character in pairs:
            stack.append(pairs[character])
        elif character in pairs.values():
            if not stack or stack.pop() != character:
                return False
    return not stack


def _repair_instance_fragment(
    value: str,
    source_prefix: str,
    *,
    required_names: set[str] | None = None,
) -> str:
    """Recover an instance expression truncated by legacy context parsing."""

    if _balanced_lean_fragment(value):
        return value
    if not value or not source_prefix:
        return value
    candidates = re.findall(r"\[([^\]\n]+)\]", source_prefix)
    matches = [candidate.strip() for candidate in candidates if candidate.startswith(value)]
    if required_names:
        for candidate in reversed(matches):
            if set(_IDENTIFIER.findall(candidate)) & required_names:
                return candidate
    if matches:
        return matches[-1]
    return value


def _plain_open_namespaces(values: Iterable[Any]) -> list[str]:
    """Discard parenthesized selective-open payloads from legacy token lists."""

    result: list[str] = []
    skipping = False
    for raw_value in values:
        value = str(raw_value).strip()
        if value.startswith("("):
            skipping = not value.endswith(")")
            continue
        if skipping:
            if value.endswith(")"):
                skipping = False
            continue
        result.append(value)
    return result


def _qualify_context_type(value: str) -> str:
    """Qualify context types whose parent namespace is intentionally not opened."""

    return re.sub(r"(?<![A-Za-z0-9_.])Measure\s+", "MeasureTheory.Measure ", value)


def _qualify_namespace_locals(
    value: str,
    *,
    namespace: str,
    source_prefix: str,
) -> str:
    """Qualify earlier declarations from a namespace that cannot be safely opened."""

    declaration_names = {
        match.group(1)
        for match in re.finditer(
            r"(?m)^\s*(?:theorem|lemma|def|abbrev)\s+"
            r"(?:_root_\.)?([A-Za-z_][A-Za-z0-9_']*)",
            source_prefix,
        )
    }
    declaration_names.difference_update(namespace.split("."))
    if not declaration_names:
        return value
    alternatives = "|".join(
        re.escape(name) for name in sorted(declaration_names, key=len, reverse=True)
    )
    return re.sub(
        rf"(?<![A-Za-z0-9_.'])(?:{alternatives})\b",
        lambda match: f"{namespace}.{match.group(0)}",
        value,
    )


@dataclass(frozen=True)
class CorpusDeclaration:
    """Source-side declaration metadata joined from ``corpus.jsonl``."""

    path: str
    full_name: str
    code: str
    imports: tuple[str, ...]
    start: tuple[int, int] | None = None
    end: tuple[int, int] | None = None
    kind: str | None = None


def iter_json_array(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array.

    ``ijson`` is already present in the vLLM environment.  A standard-library
    fallback keeps unit tests and small fixtures usable without that optional
    dependency.
    """

    source = Path(path)
    try:
        import ijson
    except ImportError:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON array: {source}")
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"expected object rows in {source}")
            yield row
        return

    with source.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            if not isinstance(row, dict):
                raise ValueError(f"expected object rows in {source}")
            yield row


def reservoir_sample(
    rows: Iterable[Mapping[str, Any]], *, size: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    """Select a deterministic uniform sample without materializing the split."""

    if size <= 0:
        raise ValueError("sample size must be positive")
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    total = 0
    for total, row in enumerate(rows, start=1):
        materialized = dict(row)
        if total <= size:
            sample.append(materialized)
            continue
        replacement = rng.randrange(total)
        if replacement < size:
            sample[replacement] = materialized
    if total < size:
        raise ValueError(f"requested {size} rows from a split containing {total}")
    return sample, total


def corpus_key(file_path: str, full_name: str) -> tuple[str, str]:
    """Normalize the two fields that identify a declaration across artifacts."""

    normalized = file_path.replace("\\", "/")
    for prefix in (
        ".lake/packages/mathlib/",
        "Mathlib/",
    ):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    return normalized, full_name


def load_corpus_declarations(
    corpus_path: str | Path,
    requested: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], CorpusDeclaration]:
    """Load only declarations needed by the selected split records."""

    wanted = set(requested)
    found: dict[tuple[str, str], CorpusDeclaration] = {}
    if not wanted:
        return found
    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            path = str(row.get("path") or "")
            imports = tuple(str(value) for value in row.get("imports") or ())
            for declaration in row.get("premises") or ():
                key = corpus_key(path, str(declaration.get("full_name") or ""))
                if key not in wanted:
                    continue
                found[key] = CorpusDeclaration(
                    path=path,
                    full_name=key[1],
                    code=str(declaration.get("code") or ""),
                    imports=imports,
                    start=_position(declaration.get("start")),
                    end=_position(declaration.get("end")),
                    kind=str(declaration.get("kind") or "") or None,
                )
            if len(found) == len(wanted):
                break
    return found


def reconstruct_leandojo_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    corpus_declarations: Mapping[tuple[str, str], CorpusDeclaration],
    split_name: str,
) -> tuple[list[LeanDataRecord], dict[str, Any]]:
    """Normalize LeanDojo split records into the shared audited contract."""

    records: list[LeanDataRecord] = []
    reasons: Counter[str] = Counter()
    seen_statements: dict[str, str] = {}
    proof_hashes: Counter[str] = Counter()

    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        file_path = str(row.get("file_path") or "")
        full_name = str(row.get("full_name") or "")
        declaration = corpus_declarations.get(corpus_key(file_path, full_name))
        record = reconstruct_leandojo_record(
            row,
            declaration=declaration,
            split_name=split_name,
            source_index=index,
        )
        reason = record.verification_error_type
        if reason is None:
            previous = seen_statements.get(record.statement_id)
            if previous is not None:
                record = _quarantine(
                    record,
                    "duplicate_statement",
                    f"normalized statement duplicates record {previous}",
                )
                reason = "duplicate_statement"
            else:
                seen_statements[record.statement_id] = record.record_id
        if record.proof:
            proof_hashes[_normalized_proof_hash(record.proof)] += 1
        if reason:
            reasons[reason] += 1
        records.append(record)

    duplicated_proof_rows = sum(count - 1 for count in proof_hashes.values())
    return records, {
        "raw_records": len(records),
        "clean_records": sum(
            record.data_state is not DataState.QUARANTINED for record in records
        ),
        "quarantined_records": sum(
            record.data_state is DataState.QUARANTINED for record in records
        ),
        "quarantine_reasons": dict(reasons),
        "unique_statement_hashes": len(seen_statements),
        "unique_normalized_proofs": len(proof_hashes),
        "duplicate_proof_rows": duplicated_proof_rows,
        "duplicate_proof_ratio": duplicated_proof_rows / max(1, len(records)),
    }


def reconstruct_leandojo_record(
    row: Mapping[str, Any],
    *,
    declaration: CorpusDeclaration | None,
    split_name: str,
    source_index: int,
) -> LeanDataRecord:
    """Reconstruct one proof-state trajectory as a complete theorem record."""

    file_path = str(row.get("file_path") or "")
    full_name = str(row.get("full_name") or "")
    commit = str(row.get("commit") or "")
    stable_payload = json.dumps(
        {
            "url": row.get("url"),
            "commit": commit,
            "file_path": file_path,
            "full_name": full_name,
            "start": row.get("start"),
            "end": row.get("end"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    record_id = "leandojo_" + _sha256(stable_payload)[:32]
    trace = list(row.get("traced_tactics") or ())
    raw_context = json.dumps(trace, ensure_ascii=False)
    proof = render_tactic_trace(trace)
    initial_state = str(trace[0].get("state_before") or "") if trace else ""
    final_state = str(trace[-1].get("state_after") or "") if trace else ""
    namespace = _namespace_for(full_name)
    statement = ""
    universe_context = None
    recovery_error: str | None = None
    try:
        statement, universes, local_replacements = _theorem_from_proof_state(
            initial_state,
            record_id=record_id,
        )
        for source_name, replacement in local_replacements.items():
            proof = proof.replace(source_name, replacement) if proof else proof
        universe_context = f"universe {' '.join(universes)}" if universes else None
    except ValueError as error:
        recovery_error = "invalid_syntax"
        statement = f"theorem leandojo_invalid_{source_index} : True"
        proof = None

    quarantine_reason: str | None = recovery_error
    quarantine_message: str | None = (
        "failed to reconstruct theorem from initial proof state"
        if recovery_error
        else None
    )
    transition_ok = _trace_transitions_are_contiguous(trace)
    if declaration is None and quarantine_reason is None:
        quarantine_reason = "missing_context"
        quarantine_message = "matching corpus declaration was not found"
    elif not trace and quarantine_reason is None:
        quarantine_reason = "empty_proof"
        quarantine_message = "tactic trace is empty"
    elif not transition_ok and quarantine_reason is None:
        quarantine_reason = "corrupt_tactic_trace"
        quarantine_message = "state_after/state_before transition mismatch"
    elif final_state.strip() != "no goals" and quarantine_reason is None:
        quarantine_reason = "unsolved_goals"
        quarantine_message = f"final proof state is {final_state!r}"
    elif not proof and quarantine_reason is None:
        quarantine_reason = "empty_proof"
        quarantine_message = "reconstructed proof is empty"
    elif _has_forbidden_content(
        statement=statement,
        proof=proof or "",
        source_code=declaration.code if declaration else "",
    ) and quarantine_reason is None:
        quarantine_reason = "forbidden_declaration"
        quarantine_message = "record contains sorry/admit/axiom/unsafe"
    elif "_private." in full_name and quarantine_reason is None:
        quarantine_reason = "private_declaration"
        quarantine_message = "private declaration namespace is not portable"

    statement_digest = statement_hash(
        "\n".join(filter(None, (namespace, universe_context, statement)))
    )
    source_span = SourceSpan(
        start_line=_line_number(row.get("start")),
        end_line=_line_number(row.get("end")),
    )
    record = LeanDataRecord(
        record_id=record_id,
        statement_id=f"stmt_{statement_digest}",
        data_state=(
            DataState.QUARANTINED if quarantine_reason else DataState.RAW
        ),
        source_dataset=LEANDOJO_DATASET,
        source_file=file_path or None,
        source_module=_module_name(file_path),
        source_declaration=full_name or None,
        source_commit=commit or None,
        source_span=source_span,
        imports=["Mathlib"],
        namespace=namespace,
        variable_context=universe_context,
        informal_statement="",
        statement=statement,
        proof=proof if quarantine_reason is None else None,
        proof_format=ProofFormat.FULL_PROOF,
        raw_declaration=declaration.code if declaration else None,
        raw_source_context=raw_context,
        verification_status="quarantined" if quarantine_reason else "raw",
        verification_error_type=quarantine_reason,
        verification_error_message=quarantine_message,
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        context_recovery_version=LEANDOJO_CONTEXT_RECOVERY_VERSION,
        recovered_source_hash=_sha256(raw_context),
        metadata={
            "split": split_name,
            "source_index": source_index,
            "source_url": row.get("url"),
            "source_imports": list(declaration.imports) if declaration else [],
            "source_corpus_path": declaration.path if declaration else None,
            "source_declaration_kind": declaration.kind if declaration else None,
            "source_declaration_start": declaration.start if declaration else None,
            "source_declaration_end": declaration.end if declaration else None,
            "trajectory_steps": len(trace),
            "transition_ok": transition_ok,
            "initial_state": initial_state,
            "final_state": final_state,
            "imports_provenance": "fixed_target_environment",
        },
    )
    return record


def theorem_from_proof_state(
    state: str, *, record_id: str
) -> tuple[str, tuple[str, ...]]:
    """Turn Lean's initial tactic state into an explicit theorem statement."""

    statement, universes, _ = _theorem_from_proof_state(
        state,
        record_id=record_id,
    )
    return statement, universes


def _theorem_from_proof_state(
    state: str, *, record_id: str
) -> tuple[str, tuple[str, ...], dict[str, str]]:
    if "⊢" not in state:
        raise ValueError("proof state has no target marker")
    local_text, target = state.rsplit("⊢", 1)
    target = target.strip()
    if not target:
        raise ValueError("proof state target is empty")
    local_text = "\n".join(
        line for line in local_text.splitlines() if not line.strip().startswith("case ")
    ).strip()
    combined = f"{local_text}\n{target}"
    replacements: dict[str, str] = {}
    synthetic_names = list(dict.fromkeys(_SYNTHETIC_LOCAL.findall(combined)))
    for raw_line in local_text.splitlines():
        match = _LOCAL_DECLARATION.match(raw_line.strip())
        if not match:
            continue
        synthetic_names.extend(
            name
            for name in match.group(1).split()
            if _needs_identifier_sanitization(name)
            and name not in synthetic_names
        )
    for index, token in enumerate(synthetic_names):
        prefix = "inst_ld" if token.startswith(("inst", "_inst")) else "ld_local"
        replacements[token] = f"{prefix}_{index}"
    for token, replacement in replacements.items():
        local_text = local_text.replace(token, replacement)
        target = target.replace(token, replacement)
    local_text = _SYNTHETIC_UNIVERSE.sub(r"ldu_\1", local_text)
    target = _SYNTHETIC_UNIVERSE.sub(r"ldu_\1", target)

    declarations = _parse_local_declarations(local_text)
    binders: list[str] = []
    for names, type_expression in declarations:
        clean_names = [name for name in names if name and name != "_"]
        if not clean_names:
            binders.append(f"[{type_expression}]")
            continue
        if all(_looks_like_instance(name) for name in clean_names):
            if len(clean_names) == 1:
                binders.append(f"[{clean_names[0]} : {type_expression}]")
            else:
                binders.extend(
                    f"[{name} : {type_expression}]" for name in clean_names
                )
        else:
            binders.append(f"({' '.join(clean_names)} : {type_expression})")

    universe_names = _universe_names("\n".join((local_text, target)))
    theorem_name = "ld_" + record_id.removeprefix("leandojo_")[:24]
    binder_text = (" " + " ".join(binders)) if binders else ""
    return (
        f"theorem {theorem_name}{binder_text} : {target}",
        universe_names,
        replacements,
    )


def render_tactic_trace(trace: Iterable[Mapping[str, Any]]) -> str:
    """Render ordered LeanDojo tactics as one complete ``by`` proof."""

    lines = ["by"]
    for step in trace:
        tactic = str(step.get("tactic") or "").strip()
        if not tactic:
            continue
        lines.extend(f"  {line}" if line.strip() else "" for line in tactic.splitlines())
    return "\n".join(lines).rstrip() if len(lines) > 1 else ""


def leandojo_prompt_statement(record: LeanDataRecord) -> str:
    """Return a proof-free prompt statement carrying required source context."""

    lines: list[str] = []
    if record.namespace:
        lines.append(f"namespace {record.namespace}")
    if record.variable_context:
        lines.append(record.variable_context)
    lines.append(record.statement)
    return "\n\n".join(lines)


def _parse_local_declarations(text: str) -> list[tuple[list[str], str]]:
    declarations: list[tuple[list[str], str]] = []
    current_names: list[str] | None = None
    current_type: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LOCAL_DECLARATION.match(line)
        if match and not line.startswith(("⊢", "case ")):
            if current_names is not None:
                declarations.append((current_names, " ".join(current_type).strip()))
            left, right = match.groups()
            current_names = left.split()
            current_type = [right]
        elif current_names is not None:
            current_type.append(line)
        else:
            raise ValueError(f"unrecognized local proof-state line: {line}")
    if current_names is not None:
        declarations.append((current_names, " ".join(current_type).strip()))
    if any(not type_expression for _, type_expression in declarations):
        raise ValueError("empty local declaration type")
    return declarations


def _universe_names(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for match in re.finditer(r"\b(?:Type|Sort)\s+([^)\],}\n]+|\([^)\n]+\))", text):
        for token in _IDENTIFIER.findall(match.group(1)):
            if token not in {
                "Type",
                "Sort",
                "max",
                "imax",
                "succ",
            } and token not in candidates:
                candidates.append(token)
    for token in re.findall(r"\bldu_\d+\b", text):
        if token not in candidates:
            candidates.append(token)
    local_names = {
        name
        for names, _ in _parse_local_declarations(text.rsplit("⊢", 1)[0])
        for name in names
    } if "⊢" in text else set()
    return tuple(token for token in candidates if token not in local_names)


def _trace_transitions_are_contiguous(trace: list[Mapping[str, Any]]) -> bool:
    return all(
        str(trace[index - 1].get("state_after") or "").strip()
        == str(trace[index].get("state_before") or "").strip()
        for index in range(1, len(trace))
    )


def _has_forbidden_content(*, statement: str, proof: str, source_code: str) -> bool:
    if contains_forbidden_proof_token(statement) or contains_forbidden_proof_token(proof):
        return True
    source_tokens = set(lean_code_tokens(source_code))
    return bool(source_tokens & FORBIDDEN_DECLARATION_TOKENS)


def _quarantine(
    record: LeanDataRecord, error_type: str, message: str
) -> LeanDataRecord:
    payload = record.model_dump(mode="json")
    payload.update(
        {
            "data_state": DataState.QUARANTINED,
            "proof": None,
            "verification_status": "quarantined",
            "verification_error_type": error_type,
            "verification_error_message": message,
        }
    )
    return LeanDataRecord.model_validate(payload)


def _namespace_for(full_name: str) -> str | None:
    namespace, separator, _ = full_name.rpartition(".")
    return namespace if separator and namespace else None


def _module_name(file_path: str) -> str | None:
    normalized = file_path.replace("\\", "/")
    if normalized.endswith(".lean"):
        normalized = normalized[:-5]
    return normalized.replace("/", ".") or None


def _looks_like_instance(name: str) -> bool:
    return name.startswith(("inst", "_inst")) or name in {"thisInstance"}


def _needs_identifier_sanitization(name: str) -> bool:
    return any(
        character in {"✝", "†"}
        or unicodedata.category(character) in {"Lm", "No"}
        for character in name
    )


def _position(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def _line_number(value: Any) -> int | None:
    position = _position(value)
    return position[0] if position else None


def _normalized_proof_hash(proof: str) -> str:
    normalized = re.sub(r"\s+", " ", proof).strip()
    return _sha256(normalized)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
