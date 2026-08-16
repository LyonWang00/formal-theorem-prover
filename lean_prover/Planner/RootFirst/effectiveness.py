"""Deterministic quality and effectiveness checks for refinement helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import RootFirstNode


_DECLARATION_RE = re.compile(
    r"^(?P<kind>theorem|lemma)\s+(?P<name>[^\s(:]+)(?P<tail>.*)$",
    re.DOTALL,
)
_IDENTIFIER_CHAR = r"A-Za-z0-9_'"


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def declaration_shape(statement: str) -> str:
    """Compare declarations while ignoring only their declaration names."""

    match = _DECLARATION_RE.match(statement.strip())
    if match is None:
        return _compact(statement)
    return _compact(match.group("kind") + " _ " + match.group("tail"))


def _top_level_signature_colon(declaration: str) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    candidates: list[int] = []
    for index, char in enumerate(declaration):
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
            candidates.append(index)
    return candidates[-1] if candidates else None


def _top_level_colon(text: str) -> int | None:
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return index
    return None


def _binder_types(prefix: str) -> list[str]:
    """Extract explicit binder types conservatively from balanced groups."""

    results: list[str] = []
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(prefix):
        if char in "([{":
            stack.append((char, index))
        elif char in pairs and stack:
            opening, start = stack.pop()
            if opening != pairs[char] or stack:
                continue
            content = prefix[start + 1 : index]
            separator = _top_level_colon(content)
            if separator is not None:
                results.append(content[separator + 1 :].strip())
    return results


def helper_quality_errors(
    new_node: RootFirstNode,
    existing_nodes: Iterable[RootFirstNode],
) -> list[str]:
    """Reject duplicate and assumption-as-conclusion helper propositions."""

    errors: list[str] = []
    new_shape = declaration_shape(new_node.lean_statement)
    new_informal = _compact(new_node.informal_statement)
    for existing in existing_nodes:
        if new_shape == declaration_shape(existing.lean_statement):
            errors.append(
                f"new helper duplicates Lean proposition of existing node {existing.id}"
            )
            break
        if new_informal and new_informal == _compact(existing.informal_statement):
            errors.append(
                f"new helper duplicates informal proposition of existing node {existing.id}"
            )
            break

    separator = _top_level_signature_colon(new_node.lean_statement)
    if separator is not None:
        prefix = new_node.lean_statement[:separator]
        conclusion = _compact(new_node.lean_statement[separator + 1 :])
        if conclusion in {_compact(item) for item in _binder_types(prefix)}:
            errors.append(
                "new helper is trivial: its conclusion exactly repeats an explicit hypothesis"
            )
        if conclusion in {"true", "proptru"}:
            errors.append("new helper is trivial: its conclusion is True")
    return errors


def proof_mentions_name(proof: str, name: str) -> bool:
    return re.search(
        rf"(?<![{_IDENTIFIER_CHAR}]){re.escape(name)}(?![{_IDENTIFIER_CHAR}])",
        proof,
    ) is not None


def dependency_usage(node: RootFirstNode) -> dict[str, object]:
    """Audit explicit use of every declared predecessor in a verified proof."""

    proof = node.verified_proof or ""
    used = [name for name in node.father_nodes if proof_mentions_name(proof, name)]
    unused = [name for name in node.father_nodes if name not in used]
    return {
        "declared_father_nodes": list(node.father_nodes),
        "explicitly_used_father_nodes": used,
        "unused_father_nodes": unused,
        "all_declared_dependencies_used": not unused,
    }


__all__ = [
    "declaration_shape",
    "dependency_usage",
    "helper_quality_errors",
    "proof_mentions_name",
]
