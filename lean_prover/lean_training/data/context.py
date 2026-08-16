"""Context-aware Lean source contracts, hashing, and deterministic assembly."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .contracts import LeanDataRecord
from .preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    ProofFormat,
    compose_lean_theorem,
)


CONTEXT_SCHEMA_VERSION = "lean-context-v4"
CONTEXT_ASSEMBLER_VERSION = f"{ASSEMBLER_VERSION}+context-v4"
CONTEXT_NORMALIZATION_VERSION = (
    f"{NORMALIZATION_VERSION}+context-v4"
)


def stable_json_hash(value: Any) -> str:
    """Hash JSON-compatible data with a canonical, Unicode-preserving encoding."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_lines(values: Iterable[str]) -> str:
    return stable_json_hash([str(value) for value in values])


@dataclass(frozen=True)
class LeanSourceContext:
    """Source-backed context active immediately before one Lean declaration."""

    imports: tuple[str, ...] = ()
    namespaces: tuple[str, ...] = ()
    open_namespaces: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    variables: tuple[dict[str, Any], ...] = ()
    hypotheses: tuple[dict[str, Any], ...] = ()
    variable_commands: tuple[str, ...] = ()
    local_instances: tuple[str, ...] = ()
    local_notations: tuple[str, ...] = ()
    local_attributes: tuple[str, ...] = ()
    option_commands: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    active_commands: tuple[str, ...] = ()
    closing_commands: tuple[str, ...] = ()
    source_file: str | None = None
    source_commit: str | None = None
    source_line: int | None = None
    context_recovered: bool = False
    recovery_method: str = ""
    recovery_status: str = "unavailable"
    recovery_sources: tuple[str, ...] = ()
    context_warnings: tuple[str, ...] = ()
    recovery_errors: tuple[str, ...] = ()
    source_hash: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "imports": list(self.imports),
            "namespaces": list(self.namespaces),
            "open_namespaces": list(self.open_namespaces),
            "scopes": list(self.scopes),
            "variables": list(self.variables),
            "hypotheses": list(self.hypotheses),
            "local_instances": list(self.local_instances),
            "local_notations": list(self.local_notations),
            "local_attributes": list(self.local_attributes),
            "variable_commands": list(self.variable_commands),
            "option_commands": list(self.option_commands),
            "sections": list(self.sections),
            "active_commands": list(self.active_commands),
            "closing_commands": list(self.closing_commands),
            "source_file": self.source_file,
            "source_commit": self.source_commit,
            "source_line": self.source_line,
            "context_recovered": self.context_recovered,
            "recovery_method": self.recovery_method,
            "recovery_status": self.recovery_status,
            "recovery_sources": list(self.recovery_sources),
            "context_warnings": list(self.context_warnings),
            "recovery_errors": list(self.recovery_errors),
            "source_hash": self.source_hash,
        }

    def hashes(self, *, environment_hash: str) -> dict[str, str]:
        namespace_payload = {
            "lexical": self.namespaces,
            "sections": self.sections,
        }
        scope_payload = {
            "open_namespaces": self.open_namespaces,
            "scopes": self.scopes,
            "local_notations": self.local_notations,
            "local_attributes": self.local_attributes,
            "options": self.option_commands,
        }
        variable_payload = {
            "variables": self.variables,
            "hypotheses": self.hypotheses,
            "variable_commands": self.variable_commands,
            "local_instances": self.local_instances,
        }
        context_payload = {
            "namespace": namespace_payload,
            "scope": scope_payload,
            "variables": variable_payload,
            "active_commands": self.active_commands,
            "closing_commands": self.closing_commands,
            "source_hash": self.source_hash,
            "recovery_method": self.recovery_method,
        }
        return {
            "imports_hash": hash_lines(self.imports),
            "namespace_hash": stable_json_hash(namespace_payload),
            "scope_hash": stable_json_hash(scope_payload),
            "variable_context_hash": stable_json_hash(variable_payload),
            "context_hash": stable_json_hash(context_payload),
            "environment_hash": environment_hash,
        }


def apply_context_to_record(
    record: LeanDataRecord,
    context: LeanSourceContext,
    *,
    environment_hash: str,
) -> LeanDataRecord:
    """Copy the unified context schema and its provenance onto a data record."""

    hashes = context.hashes(environment_hash=environment_hash)
    section_context: dict[str, Any] = {
        "sections": list(context.sections),
        "active_commands": list(context.active_commands),
        "closing_commands": list(context.closing_commands),
    }
    namespace = ".".join(context.namespaces) or record.namespace
    metadata = {
        **record.metadata,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "context_summary": context.summary(),
        "context_hashes": hashes,
        "source_lookup_attempted": bool(context.source_file),
        "context_recovery_method": context.recovery_method,
    }
    return record.model_copy(
        update={
            "id": record.record_id,
            "source": (
                "leandojo"
                if "leandojo" in record.source_dataset.lower()
                else "lean_workbook"
            ),
            "imports": list(context.imports or tuple(record.imports)),
            "namespace": namespace,
            "namespaces": list(context.namespaces),
            "open_namespaces": list(context.open_namespaces),
            "scopes": list(context.scopes),
            "variables": list(context.variables),
            "hypotheses": list(context.hypotheses),
            "local_instances": list(context.local_instances),
            "local_notations": list(context.local_notations),
            "local_attributes": list(context.local_attributes),
            "section_context": section_context,
            "source_file": context.source_file or record.source_file,
            "source_commit": context.source_commit or record.source_commit,
            "environment_hash": environment_hash,
            "assembler_version": CONTEXT_ASSEMBLER_VERSION,
            "normalization_version": CONTEXT_NORMALIZATION_VERSION,
            "context_recovered": context.context_recovered,
            "context_recovery_method": context.recovery_method,
            "context_recovery_status": context.recovery_status,
            "context_recovery_sources": list(context.recovery_sources),
            "context_warnings": list(context.context_warnings),
            "recovered_source_hash": context.source_hash,
            "metadata": metadata,
        }
    )


def record_context_summary(record: LeanDataRecord) -> dict[str, Any]:
    """Return a stable context summary for verification records and cache keys."""

    from_metadata = record.metadata.get("context_summary")
    if isinstance(from_metadata, Mapping):
        return dict(from_metadata)
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "imports": list(record.imports),
        "namespaces": list(record.namespaces),
        "open_namespaces": list(record.open_namespaces),
        "scopes": list(record.scopes),
        "variables": list(record.variables),
        "hypotheses": list(record.hypotheses),
        "local_instances": list(record.local_instances),
        "local_notations": list(record.local_notations),
        "local_attributes": list(record.local_attributes),
        "section_context": record.section_context,
        "source_file": record.source_file,
        "source_commit": record.source_commit,
        "context_recovered": record.context_recovered,
        "context_recovery_method": record.context_recovery_method,
        "context_recovery_status": record.context_recovery_status,
        "context_recovery_sources": list(record.context_recovery_sources),
        "context_warnings": list(record.context_warnings),
    }


def record_context_hashes(
    record: LeanDataRecord,
    *,
    environment_hash: str,
) -> dict[str, str]:
    """Build every context component required by the verification cache contract."""

    existing = record.metadata.get("context_hashes")
    if isinstance(existing, Mapping) and existing.get("context_hash"):
        hashes = {str(key): str(value) for key, value in existing.items()}
        hashes["environment_hash"] = environment_hash
        return hashes
    namespace_payload = {
        "namespace": record.namespace,
        "namespaces": record.namespaces,
    }
    scope_payload = {
        "open_namespaces": record.open_namespaces,
        "scopes": record.scopes,
        "local_notations": record.local_notations,
        "local_attributes": record.local_attributes,
        "section_context": record.section_context,
    }
    variable_payload = {
        "variables": record.variables,
        "hypotheses": record.hypotheses,
        "local_instances": record.local_instances,
        "variable_context": record.variable_context,
        "local_context": record.local_context,
    }
    summary = record_context_summary(record)
    return {
        "imports_hash": hash_lines(record.imports),
        "namespace_hash": stable_json_hash(namespace_payload),
        "scope_hash": stable_json_hash(scope_payload),
        "variable_context_hash": stable_json_hash(variable_payload),
        "context_hash": stable_json_hash(summary),
        "environment_hash": environment_hash,
    }


def assemble_context_aware_declaration(
    record: LeanDataRecord,
    *,
    proof: str | None = None,
) -> str:
    """Assemble a theorem with its exact recovered lexical context.

    Context-aware records store replayable source commands in metadata.  The
    declaration header comes from the source corpus when available; otherwise
    the normalized statement is used.  The outer evaluation namespace is still
    added later by the unchanged Pantograph task assembler.
    """

    declaration = compose_lean_theorem(
        record.statement,
        record.proof if proof is None else proof,
        proof_format=record.proof_format,
    )
    summary = record_context_summary(record)
    active = [
        str(value).rstrip()
        for value in summary.get("active_commands", ())
        if str(value).strip()
    ]
    closing = [
        str(value).rstrip()
        for value in summary.get("closing_commands", ())
        if str(value).strip()
    ]
    if not active:
        active = _fallback_context_commands(record)
        closing = _fallback_closing_commands(record)
    return "\n\n".join([*active, declaration, *closing]).strip()


def _fallback_context_commands(record: LeanDataRecord) -> list[str]:
    commands: list[str] = []
    if record.variable_context:
        commands.append(record.variable_context)
    commands.extend(f"open {name}" for name in record.open_namespaces)
    commands.extend(f"open scoped {name}" for name in record.scopes)
    commands.extend(record.local_notations)
    commands.extend(record.local_attributes)
    for namespace in record.namespaces:
        commands.append(f"namespace {namespace}")
    if not record.namespaces and record.namespace:
        commands.append(f"namespace {record.namespace}")
    return commands


def _fallback_closing_commands(record: LeanDataRecord) -> list[str]:
    namespaces: Sequence[str] = record.namespaces
    if not namespaces and record.namespace:
        namespaces = (record.namespace,)
    return [f"end {name}" for name in reversed(namespaces)]


_VARIABLE_BINDER = re.compile(r"([\(\{\[])([^\)\}\]]+)[\)\}\]]")


def parse_variable_command(command: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract structured binders while retaining the exact raw command."""

    compact = " ".join(line.strip() for line in command.splitlines())
    compact = re.sub(r"^variables?\s+", "", compact).strip()
    variables: list[dict[str, Any]] = []
    instances: list[str] = []
    for match in _VARIABLE_BINDER.finditer(compact):
        opener, body = match.groups()
        if opener == "[":
            instances.append(body.strip())
            continue
        if ":" not in body:
            continue
        names, type_expression = body.split(":", 1)
        binder = "implicit" if opener == "{" else "explicit"
        for name in names.split():
            variables.append(
                {
                    "name": name,
                    "type": type_expression.strip(),
                    "binder": binder,
                    "raw": match.group(0),
                }
            )
    return variables, instances


def variables_from_proof_state(state: str) -> tuple[dict[str, Any], ...]:
    """Best-effort structured view of locals already explicit in a proof state."""

    if "⊢" not in state:
        return ()
    local_text = state.rsplit("⊢", 1)[0]
    variables: list[dict[str, Any]] = []
    for raw_line in local_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("case ") or ":" not in line:
            continue
        names, type_expression = line.split(":", 1)
        for name in names.split():
            if name == "_":
                continue
            variables.append(
                {
                    "name": name,
                    "type": type_expression.strip(),
                    "binder": (
                        "instance"
                        if name.startswith(("inst", "_inst"))
                        else "explicit"
                    ),
                    "raw": line,
                }
            )
    return tuple(variables)


_PROPOSITION_MARKERS = (
    " = ",
    " ≠ ",
    " < ",
    " > ",
    " ≤ ",
    " ≥ ",
    " ∣ ",
    " ∈ ",
    " → ",
    " ↔ ",
    "¬",
)


def split_proof_state_locals(
    state: str,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[str, ...],
]:
    """Conservatively classify proof-state locals for schema/audit purposes.

    These values are never replayed as Lean commands.  They describe locals
    already bound by the theorem or introduced by tactics, and therefore must
    not be mistaken for recoverable file-level declarations.
    """

    variables: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    instances: list[str] = []
    if "⊢" not in state:
        return (), (), ()
    for raw_line in state.rsplit("⊢", 1)[0].splitlines():
        line = raw_line.strip()
        if not line or line.startswith("case ") or ":" not in line:
            continue
        names_text, type_expression = line.split(":", 1)
        names = [name for name in names_text.split() if name != "_"]
        if not names:
            continue
        type_expression = type_expression.strip()
        is_instance = all(
            name.startswith(("inst", "_inst")) for name in names
        )
        is_hypothesis = any(
            name.startswith(("h", "this")) for name in names
        ) or any(marker in f" {type_expression} " for marker in _PROPOSITION_MARKERS)
        target = hypotheses if is_hypothesis else variables
        for name in names:
            target.append(
                {
                    "name": name,
                    "type": type_expression,
                    "binder": "instance" if is_instance else "explicit",
                    "raw": line,
                    "source": "proof_state",
                }
            )
        if is_instance:
            instances.append(type_expression)
    return tuple(variables), tuple(hypotheses), tuple(instances)


def context_record_payload(record: LeanDataRecord) -> dict[str, Any]:
    """Serialize the public unified schema requested by context-aware audits."""

    return {
        "id": record.id or record.record_id,
        "source": record.source or record.source_dataset,
        "statement": record.statement,
        "proof": record.proof,
        "imports": list(record.imports),
        "namespaces": list(record.namespaces),
        "open_namespaces": list(record.open_namespaces),
        "scopes": list(record.scopes),
        "variables": list(record.variables),
        "hypotheses": list(record.hypotheses),
        "local_instances": list(record.local_instances),
        "local_notations": list(record.local_notations),
        "section_context": record.section_context or {},
        "source_file": record.source_file or "",
        "environment_hash": record.environment_hash or "",
        "context_recovered": record.context_recovered,
        "context_recovery_method": record.context_recovery_method,
        "context_recovery_status": record.context_recovery_status,
        "context_recovery_sources": list(record.context_recovery_sources),
        "context_warnings": list(record.context_warnings),
        "assembled_source_hash": record.assembled_source_hash or "",
        "metadata": record.metadata,
        "record": record.model_dump(mode="json"),
    }
