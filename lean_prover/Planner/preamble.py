"""Render the structured Planner preamble for Lean/Pantograph consumers."""

from __future__ import annotations

import re

from .schemas import LeanPreamble


_SECTION_PATTERN = re.compile(
    r"^(?:noncomputable\s+)?section"
    r"(?:\s+([A-Za-z_][A-Za-z0-9_']*))?$"
)


def imports_from_header(header: str) -> list[str]:
    """Extract imports without rewriting the caller-supplied Lean header."""

    return [
        match.group(1)
        for line in header.splitlines()
        if (
            match := re.fullmatch(
                r"\s*import\s+([A-Za-z_][A-Za-z0-9_'.]*(?:\.[A-Za-z_][A-Za-z0-9_']*)*)\s*",
                line,
            )
        )
    ]


def header_context_without_imports(header: str) -> str:
    """Return header commands that Pantograph may accept after server startup.

    Pantograph loads modules while constructing ``Server(imports=...)`` and
    wraps ``load_sorry`` / goal sources internally.  Re-emitting an ``import``
    command in those sources is rejected because it is no longer at the
    beginning of the generated Lean file.  Non-import commands remain in their
    original order and the complete raw header remains stored in the schema.
    """

    return "\n".join(
        line
        for line in header.splitlines()
        if not re.fullmatch(
            r"\s*import\s+[A-Za-z_][A-Za-z0-9_'.]*(?:\.[A-Za-z_][A-Za-z0-9_']*)*\s*",
            line,
        )
    ).strip()


def ordered_imports(*groups: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for group in groups:
        for module in group:
            if module not in result:
                result.append(module)
    return result


def render_preamble_context(preamble: LeanPreamble) -> str:
    """Render structured commands once, even when raw_header contains them."""

    raw_lines = {
        re.sub(r"\s+", " ", line.strip())
        for line in preamble.raw_header.splitlines()
        if line.strip()
    }
    candidates = [
        *(f"open {name}" for name in preamble.open_namespaces),
        *(f"open scoped {name}" for name in preamble.open_scoped),
        *preamble.variable_declarations,
        *preamble.local_context,
    ]
    rendered: list[str] = []
    seen = set(raw_lines)
    for line in candidates:
        normalized = re.sub(r"\s+", " ", line.strip())
        if normalized and normalized not in seen:
            rendered.append(line.strip())
            seen.add(normalized)
    return "\n".join(rendered)


def wrap_source_with_preamble(
    source: str,
    preamble: LeanPreamble,
    *,
    include_raw_header: bool = True,
) -> str:
    """Materialize namespaces/context and close every opened section."""

    section_closings: list[str] = []
    for line in preamble.local_context:
        match = _SECTION_PATTERN.fullmatch(line)
        if match:
            name = match.group(1)
            section_closings.append(f"end {name}" if name else "end")

    namespace_opening = "\n".join(
        f"namespace {name}" for name in preamble.namespaces
    )
    namespace_closing = "\n".join(
        f"end {name}" for name in reversed(preamble.namespaces)
    )
    scope_closing = "\n".join(
        [*reversed(section_closings), namespace_closing]
    ).strip()
    return "\n\n".join(
        part
        for part in (
            preamble.raw_header if include_raw_header else "",
            namespace_opening,
            render_preamble_context(preamble),
            source.strip(),
            scope_closing,
        )
        if part.strip()
    )
