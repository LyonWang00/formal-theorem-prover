"""Deterministic miniF2F import compatibility for the pinned Mathlib tree."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path


MAPPING_VERSION = "minif2f-mathlib-5e932f97-imports-v2"
STATEMENT_COMPATIBILITY_VERSION = "minif2f-mathlib-5e932f97-statements-v1"

# Explicit identity entries make the contract fully auditable and ensure an
# unexpected miniF2F import is never silently accepted.
MINIF2F_IMPORT_COMPATIBILITY: dict[str, tuple[str, ...]] = {
    "Mathlib.Algebra.BigOperators.Basic": (
        "Mathlib.Algebra.BigOperators.Group.Finset.Basic",
    ),
    "Mathlib.Data.Real.Basic": ("Mathlib.Data.Real.Basic",),
    "Mathlib.Data.Complex.Basic": ("Mathlib.Data.Complex.Basic",),
    "Mathlib.Data.Nat.Log": ("Mathlib.Data.Nat.Log",),
    "Mathlib.Data.Complex.Exponential": (
        "Mathlib.Analysis.Complex.Exponential",
        # The legacy miniF2F aggregate header exposed the real logarithm,
        # logarithm-base, and trigonometric declarations transitively.  Their
        # current homes must be explicit in the pinned modular Mathlib tree.
        "Mathlib.Analysis.SpecialFunctions.Log.Base",
        "Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic",
        "Mathlib.NumberTheory.Real.Irrational",
    ),
    "Mathlib.NumberTheory.Divisors": ("Mathlib.NumberTheory.Divisors",),
    "Mathlib.Data.ZMod.Defs": ("Mathlib.Data.ZMod.Defs",),
    "Mathlib.Data.ZMod.Basic": ("Mathlib.Data.ZMod.Basic",),
    "Mathlib.Topology.Basic": ("Mathlib.Topology.Basic",),
    "Mathlib.Data.Nat.Digits": ("Mathlib.Data.Nat.Digits.Lemmas",),
}

_IMPORT_LINE = re.compile(
    r"^(?P<indent>\s*)import\s+(?P<module>[A-Za-z0-9_.]+)\s*$"
)


def mapped_imports(
    header: str,
    *,
    mapping: Mapping[str, Sequence[str]] = MINIF2F_IMPORT_COMPATIBILITY,
) -> list[str]:
    """Return mapped imports, rejecting every unregistered input module."""

    imports: list[str] = []
    seen: set[str] = set()
    for line in header.splitlines():
        match = _IMPORT_LINE.fullmatch(line)
        if match is None:
            continue
        old_module = match.group("module")
        if old_module not in mapping:
            raise ValueError(f"unmapped miniF2F import: {old_module}")
        for new_module in mapping[old_module]:
            if new_module not in seen:
                seen.add(new_module)
                imports.append(new_module)
    if not imports:
        raise ValueError("miniF2F header contains no import declarations")
    return imports


def rewrite_header(
    header: str,
    *,
    mapping: Mapping[str, Sequence[str]] = MINIF2F_IMPORT_COMPATIBILITY,
) -> str:
    """Rewrite import lines while preserving all non-import header text."""

    output: list[str] = []
    for line in header.splitlines():
        match = _IMPORT_LINE.fullmatch(line)
        if match is None:
            output.append(line)
            continue
        old_module = match.group("module")
        if old_module not in mapping:
            raise ValueError(f"unmapped miniF2F import: {old_module}")
        indent = match.group("indent")
        output.extend(f"{indent}import {module}" for module in mapping[old_module])
    return "\n".join(output).strip()


def validate_mapped_modules_exist(
    mathlib_root: str | Path,
    *,
    mapping: Mapping[str, Sequence[str]] = MINIF2F_IMPORT_COMPATIBILITY,
) -> None:
    """Fail before Pantograph if a mapped module has no current source file."""

    root = Path(mathlib_root)
    missing: list[str] = []
    for targets in mapping.values():
        for module in targets:
            source = root / (module.replace(".", "/") + ".lean")
            if not source.is_file():
                missing.append(module)
    if missing:
        raise FileNotFoundError(
            "mapped Mathlib modules do not exist: " + ", ".join(sorted(set(missing)))
        )


def _uncomment_declaration(statement: str) -> tuple[str, bool]:
    """Restore a declaration commented out by miniF2F conversion tooling."""

    lines = statement.splitlines()
    theorem_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"\s*--\s*(?:theorem|lemma)\b", line)
        ),
        None,
    )
    if theorem_index is None:
        return statement, False

    restored: list[str] = []
    for line in lines[theorem_index:]:
        match = re.match(r"^(?P<indent>\s*)--(?:\s?)(?P<body>.*)$", line)
        restored.append(
            f"{match.group('indent')}{match.group('body')}"
            if match is not None
            else line
        )
    return "\n".join(restored).strip(), True


def rewrite_statement(statement: str) -> tuple[str, list[str]]:
    """Apply deterministic, semantics-preserving current-Mathlib rewrites."""

    rewritten, uncommented = _uncomment_declaration(statement.strip())
    repairs: list[str] = []
    if uncommented:
        repairs.append("uncomment_formal_declaration")

    # Mathlib removed the deprecated `∑/∏ x in s, f x` parser.  The current
    # spelling has identical Finset semantics and uses the membership binder.
    big_operator = re.compile(r"(?P<operator>[∑∏])(?P<binder>[^,\n]*?)\s+in\s+")
    rewritten, replacement_count = big_operator.subn(
        lambda match: (
            f"{match.group('operator')}{match.group('binder')} ∈ "
        ),
        rewritten,
    )
    if replacement_count:
        repairs.append("rewrite_big_operator_in_to_membership")

    # `Complex.abs` was the complex norm as a real number.  Its current
    # notation is `‖z‖`; this miniF2F corpus contains one fixed occurrence.
    old_complex_abs = "Complex.abs (a - b)"
    if old_complex_abs in rewritten:
        rewritten = rewritten.replace(old_complex_abs, "‖a - b‖")
        repairs.append("rewrite_complex_abs_to_norm")

    # The unqualified name became ambiguous after a second `lcm` API entered
    # scope.  Both arguments are Nat in this fixed corpus entry.
    rewritten, replacement_count = re.subn(
        r"(?<![A-Za-z0-9_.])lcm(?=\s+m\s+n\b)",
        "Nat.lcm",
        rewritten,
    )
    if replacement_count:
        repairs.append("qualify_nat_lcm")

    exact_rewrites = (
        (
            "∏ k ∈ Finset.Icc 1 n, (1 + 1 / k^3)",
            "∏ k ∈ Finset.Icc 1 n, (1 + (1 : ℝ) / (k : ℝ)^3)",
            "disambiguate_real_product_term",
        ),
        (
            "(h₀ : 2^a = 32)",
            "(h₀ : (2 : ℝ)^a = 32)",
            "disambiguate_real_power_base",
        ),
        (
            "(n = 3) :",
            "(h₂ : n = 3) :",
            "name_anonymous_equality_hypothesis",
        ),
    )
    for old, new, repair in exact_rewrites:
        if old in rewritten:
            rewritten = rewritten.replace(old, new)
            repairs.append(repair)

    rewritten, replacement_count = re.subn(
        r"(?<![A-Za-z0-9_])irrational\b",
        "Irrational",
        rewritten,
    )
    if replacement_count:
        repairs.append("capitalize_irrational_predicate")

    # One source row omitted only the declaration command while retaining a
    # theorem-shaped identifier, binders, result type, and placeholder proof.
    if rewritten.startswith("imosl_2007_algebra_p6\n"):
        rewritten = "theorem " + rewritten
        repairs.append("restore_missing_theorem_keyword")

    return rewritten.strip(), repairs
