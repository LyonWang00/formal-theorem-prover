"""Small shared checks for role-specific verified-dataset adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_FINAL_ASSIGNMENT = re.compile(r"\s*:=\s*$", re.S)
_TARGET_DECLARATION = re.compile(
    r"(?ms)^[ \t]*(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:private|protected|nonrec)\s+)*"
    r"(?:theorem|lemma|example)(?:\s+|\()"
)


def has_supported_declaration_prefix(statement: str) -> bool:
    """Return whether text begins with a supported theorem declaration."""

    return bool(_TARGET_DECLARATION.match(statement))


def require_verified_scope(record: Mapping[str, Any], *, scope: str) -> None:
    status = record.get("pantograph_verified")
    if status not in {"success", True}:
        raise ValueError("record is not Pantograph verified")
    if str(record.get("verification_scope") or "") != scope:
        raise ValueError(f"record does not have verification_scope={scope}")


def split_imports_and_body(source: str) -> tuple[tuple[str, ...], str]:
    imports: list[str] = []
    body: list[str] = []
    for line in source.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            module = stripped.removeprefix("import ").strip()
            if module and module not in imports:
                imports.append(module)
        else:
            body.append(line)
    return tuple(imports), "\n".join(body).strip()


def strip_final_assignment(statement: str) -> str:
    return _FINAL_ASSIGNMENT.sub("", statement).strip()


def split_context_and_target(body: str) -> tuple[tuple[str, ...], str]:
    matches = list(_TARGET_DECLARATION.finditer(body))
    if not matches:
        raise ValueError("source has no theorem, lemma, or example declaration")
    target_start = matches[-1].start()
    context = tuple(line for line in body[:target_start].splitlines() if line.strip())
    return context, body[target_start:].strip()


def record_id(record: Mapping[str, Any], *, prefix: str, index: int) -> str:
    return str(
        record.get("record_id")
        or record.get("statement_id")
        or record.get("id")
        or f"{prefix}/{index}"
    ).strip()
