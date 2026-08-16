"""Batchwise, locally verified minimal repairs for NuminaMath-LEAN failures.

The first repair lane is intentionally conservative: it considers only rows
whose final declaration has no proof, contains no forbidden proof token, and
is short enough for a small Codex-curated portfolio of standalone tactics.
Every promoted proof must pass the frozen local Pantograph environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from lean_prover.Dataset.build_verified_datasets import (
    FAIL,
    SUCCESS,
    _write_json,
    add_dataset_contract,
    dataset_record_hash,
    sha256_file,
    sha256_text,
)
from lean_prover.Dataset.verify_external_datasets import (
    NUMINA_REVISION,
    NUMINA_SOURCE,
    select_numina_source,
    split_imports,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import PantographTheoremVerifier


REPAIR_VERSION = "numinamath_codex_minimal_repair_v1"
_FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.I)
_DECLARATION = re.compile(r"\b(?:theorem|lemma)\s+[^\s(:{]+")
_EMPTY_PROOF_END = re.compile(r":=\s*by\s*$", re.S)
_PLACEHOLDER_PROOF_END = re.compile(r":=\s*by\s+(?:sorry|admit)\s*$", re.I | re.S)
_DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")
_INTEGER = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_PURE_ARITHMETIC = re.compile(r"[\s\d+\-*/^=<>≤≥().]+")
_ERROR_LINE = re.compile(
    r"^\s*\d+:\d+(?:-\d+:\d+)?:\s*error:\s*(.+)$", re.I | re.M
)
_COMPLEX_ABS = re.compile(r"\bComplex\.abs\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def error_family(message: str) -> str:
    lowered = message.lower()
    ordered = (
        ("timeout", ("timeout", "timed out")),
        ("unknown_identifier", ("unknown identifier", "unknown constant", "invalid field")),
        ("syntax_error", ("syntax error", "unexpected token", "expected ','")),
        ("type_mismatch", ("type mismatch", "application type mismatch")),
        ("tactic_error", ("tactic `", "tactic failed", "no goals to be solved", "made no progress")),
        ("unsolved_goals", ("unsolved goal",)),
        ("elaboration_error", ("failed to synthesize", "cannot synthesize")),
    )
    for label, needles in ordered:
        if any(needle in lowered for needle in needles):
            return label
    return "other"


def classify_row(row: Mapping[str, Any]) -> str:
    proof = str(row.get("proof") or "").strip()
    _, selected_source = select_numina_source(row)
    if not proof:
        if _FORBIDDEN.search(selected_source):
            return "missing_proof_with_forbidden_support"
        return "missing_proof_clean"
    if _FORBIDDEN.search(proof) or _FORBIDDEN.search(selected_source):
        return "incomplete_or_placeholder_proof"
    return f"proof_present_{error_family(str(row.get('error_message') or ''))}"


def source_body_for_repair(row: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    _, selected_source = select_numina_source(row)
    imports, body = split_imports(selected_source)
    return tuple(imports), body.strip()


def declaration_goal_colon(source_body: str) -> int | None:
    declarations = list(_DECLARATION.finditer(source_body))
    if not declarations:
        return None
    start = declarations[-1].end()
    depth = 0
    for index in range(start, len(source_body)):
        char = source_body[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return index
    return None


def annotate_first_decimal_as_rational(source_body: str) -> tuple[str, dict[str, str]] | None:
    """Add one minimal type annotation to a closed decimal proposition."""

    goal_colon = declaration_goal_colon(source_body)
    declarations = list(_DECLARATION.finditer(source_body))
    if goal_colon is None or not declarations:
        return None
    # This first statement-repair lane is restricted to closed theorems.  Any
    # binders or local typeclass assumptions are left for a later batch.
    header_tail = source_body[declarations[-1].end() : goal_colon].strip()
    if header_tail:
        return None
    proof_start = source_body.rfind(":=", goal_colon)
    if proof_start < goal_colon:
        return None
    proposition = source_body[goal_colon + 1 : proof_start]
    match = _DECIMAL.search(proposition)
    if match is None:
        return None
    original = match.group(1)
    replacement = f"({original} : ℚ)"
    absolute_start = goal_colon + 1 + match.start(1)
    absolute_end = goal_colon + 1 + match.end(1)
    repaired = source_body[:absolute_start] + replacement + source_body[absolute_end:]
    return repaired, {
        "kind": "disambiguate_decimal_type",
        "from": original,
        "to": replacement,
    }


def annotate_pure_closed_arithmetic(
    source_body: str,
) -> tuple[str, dict[str, str]] | None:
    """Disambiguate a variable-free, identifier-free arithmetic proposition."""

    goal_colon = declaration_goal_colon(source_body)
    declarations = list(_DECLARATION.finditer(source_body))
    if goal_colon is None or not declarations:
        return None
    if source_body[declarations[-1].end() : goal_colon].strip():
        return None
    proof_start = source_body.rfind(":=", goal_colon)
    if proof_start < goal_colon:
        return None
    raw_proposition = source_body[goal_colon + 1 : proof_start]
    proposition = raw_proposition.strip()
    if not proposition or _PURE_ARITHMETIC.fullmatch(proposition) is None:
        return None
    # Fractional/decimal exponents require Real.rpow rather than a local
    # numeric annotation, so they belong to a later statement-repair lane.
    if re.search(r"\^\s*\([^)]*/[^)]*\)", proposition) or re.search(
        r"\^\s*\(?\s*\d+\.\d+", proposition
    ):
        return None
    if any(operator in proposition for operator in (">", "<", "≥", "≤")) and "^" in proposition:
        literal_exponents = [
            int(value) for value in re.findall(r"\^\s*\(?\s*(\d+)", proposition)
        ]
        if proposition.count("^") >= 3 or any(value > 64 for value in literal_exponents):
            return None
    if any(operator in proposition for operator in (">", "<", "≥", "≤")) and "^" in proposition:
        literal_exponents = [
            int(value) for value in re.findall(r"\^\s*\(?\s*(\d+)", proposition)
        ]
        if proposition.count("^") >= 3 or any(value > 64 for value in literal_exponents):
            return None
    match = _NUMBER.search(raw_proposition)
    if match is None:
        return None
    target_type = (
        "ℚ"
        if "/" in proposition or _DECIMAL.search(proposition)
        else "ℤ"
        if "-" in proposition
        else "ℕ"
    )
    original = match.group(1)
    replacement = f"({original} : {target_type})"
    absolute_start = goal_colon + 1 + match.start(1)
    absolute_end = goal_colon + 1 + match.end(1)
    repaired = source_body[:absolute_start] + replacement + source_body[absolute_end:]
    return repaired, {
        "kind": "disambiguate_pure_arithmetic_type",
        "from": original,
        "to": replacement,
        "target_type": target_type,
    }


def eligible_missing_proof(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str]:
    if classify_row(row) != "missing_proof_clean":
        return False, "not_clean_missing_proof"
    if row.get("repair_error"):
        return False, "already_attempted"
    _, body = source_body_for_repair(row)
    if len(body) > max_source_chars:
        return False, "source_too_long"
    if len(_DECLARATION.findall(body)) > max_declarations:
        return False, "too_many_declarations"
    if not _EMPTY_PROOF_END.search(body):
        return False, "missing_supported_empty_proof_ending"
    if _FORBIDDEN.search(body):
        return False, "forbidden_source"
    return True, "eligible"


def eligible_closed_decimal_rational(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str, str | None, dict[str, str] | None]:
    if classify_row(row) != "missing_proof_clean":
        return False, "not_clean_missing_proof", None, None
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair.get("lane") == "closed_decimal_rational":
        return False, "already_attempted_decimal_lane", None, None
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair.get("lane") == "closed_decimal_rational":
        return False, "already_attempted_decimal_lane", None, None
    _, body = source_body_for_repair(row)
    if len(body) > max_source_chars:
        return False, "source_too_long", None, None
    if len(_DECLARATION.findall(body)) > max_declarations:
        return False, "too_many_declarations", None, None
    if not _EMPTY_PROOF_END.search(body):
        return False, "missing_supported_empty_proof_ending", None, None
    changed = annotate_first_decimal_as_rational(body)
    if changed is None:
        return False, "not_closed_decimal_goal", None, None
    repaired, change = changed
    return True, "eligible", repaired, change


def eligible_closed_integer_norm_num(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str]:
    if classify_row(row) != "missing_proof_clean":
        return False, "not_clean_missing_proof"
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair.get("lane") == "closed_integer_norm_num":
        return False, "already_attempted_integer_lane"
    _, body = source_body_for_repair(row)
    if len(body) > max_source_chars:
        return False, "source_too_long"
    declarations = list(_DECLARATION.finditer(body))
    if len(declarations) > max_declarations:
        return False, "too_many_declarations"
    if not declarations or not _EMPTY_PROOF_END.search(body):
        return False, "missing_supported_empty_proof_ending"
    goal_colon = declaration_goal_colon(body)
    if goal_colon is None:
        return False, "missing_goal_colon"
    if body[declarations[-1].end() : goal_colon].strip():
        return False, "not_closed_goal"
    proof_start = body.rfind(":=", goal_colon)
    proposition = body[goal_colon + 1 : proof_start]
    if not re.search(r"\d", proposition) or _DECIMAL.search(proposition):
        return False, "not_integer_numeric_goal"
    return True, "eligible"


def eligible_pure_closed_arithmetic(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str, str | None, dict[str, str] | None]:
    classification = classify_row(row)
    _, selected_source = select_numina_source(row)
    has_only_final_placeholder = bool(_PLACEHOLDER_PROOF_END.search(selected_source))
    if classification not in {"missing_proof_clean", "incomplete_or_placeholder_proof"} and not (
        classification == "missing_proof_with_forbidden_support"
        and has_only_final_placeholder
    ):
        return False, "not_missing_or_placeholder_proof", None, None
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair.get("lane") == "pure_closed_arithmetic":
        return False, "already_attempted_pure_arithmetic_lane", None, None
    _, body = source_body_for_repair(row)
    if len(body) > max_source_chars:
        return False, "source_too_long", None, None
    if len(_DECLARATION.findall(body)) != 1:
        return False, "not_single_declaration", None, None
    if not (_EMPTY_PROOF_END.search(body) or _PLACEHOLDER_PROOF_END.search(body)):
        return False, "not_empty_or_placeholder_final_proof", None, None
    # Ignore only the final placeholder; any earlier forbidden declaration is
    # a different repair problem and must not enter this lane.
    prefix = _PLACEHOLDER_PROOF_END.sub(":= by", body)
    if _FORBIDDEN.search(prefix):
        return False, "forbidden_support", None, None
    changed = annotate_pure_closed_arithmetic(body)
    if changed is None:
        return False, "not_pure_closed_arithmetic", None, None
    repaired, change = changed
    return True, "eligible", repaired, change


def only_no_goals_errors(message: str) -> bool:
    errors = [item.strip() for item in _ERROR_LINE.findall(message)]
    return bool(errors) and all(item == "No goals to be solved" for item in errors)


def trailing_tactic_variants(proof: str, *, max_variants: int) -> list[dict[str, str]]:
    """Remove only trailing, flat tactic lines from a proof that over-runs its goal."""

    lines = proof.strip().splitlines()
    if not lines or lines[0].strip() != "by":
        return []
    body = lines[1:]
    while body and (not body[-1].strip() or body[-1].lstrip().startswith("--")):
        body.pop()
    tactic_indexes: list[int] = []
    for index, line in enumerate(body):
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        # This lane is deliberately limited to one-line top-level tactics.
        # Nested bullets/cases or continuation indentation require syntax-aware
        # editing and are left to a different repair strategy.
        if not line.startswith("  ") or line.startswith("    "):
            return []
        if stripped.startswith(("·", "|", "case ", "next ", "all_goals", "first |")) or stripped.endswith(" with"):
            return []
        tactic_indexes.append(index)
    if len(tactic_indexes) < 2:
        return []
    variants: list[dict[str, str]] = []
    for trim_count in range(1, min(max_variants, len(tactic_indexes) - 1) + 1):
        cut_at = tactic_indexes[-trim_count]
        kept = body[:cut_at]
        while kept and (not kept[-1].strip() or kept[-1].lstrip().startswith("--")):
            kept.pop()
        if not kept:
            continue
        variants.append({
            "strategy": f"trim_{trim_count}_trailing_tactic_lines",
            "proof": "\n".join(["by", *kept]),
        })
    return variants


def eligible_trim_trailing_no_goals(
    row: Mapping[str, Any], *, max_source_chars: int, max_variants: int
) -> tuple[bool, str, str | None, list[dict[str, str]]]:
    proof = str(row.get("proof") or "").strip()
    if not proof:
        return False, "missing_proof", None, []
    if _FORBIDDEN.search(proof):
        return False, "forbidden_proof", None, []
    if not only_no_goals_errors(str(row.get("error_message") or "")):
        return False, "not_only_no_goals", None, []
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair.get("lane") == "trim_trailing_no_goals":
        return False, "already_attempted_trim_lane", None, []
    statement_source = str(row.get("lean_statement") or row.get("formal_statement") or "").strip()
    imports, body = split_imports(statement_source)
    if not body or len(body) > max_source_chars:
        return False, "statement_missing_or_too_long", None, []
    if ":=" in body:
        return False, "statement_contains_proof", None, []
    variants = trailing_tactic_variants(proof, max_variants=max_variants)
    if not variants:
        return False, "proof_not_flat_trimmable", None, []
    return True, "eligible", body.rstrip() + " := by", variants


def eligible_replace_complex_abs(
    row: Mapping[str, Any], *, max_source_chars: int
) -> tuple[bool, str, str | None, list[dict[str, str]], list[dict[str, str]], bool]:
    proof = str(row.get("proof") or "").strip()
    if not proof:
        return False, "missing_proof", None, [], [], False
    if _FORBIDDEN.search(proof):
        return False, "forbidden_proof", None, [], [], False
    message = str(row.get("error_message") or "")
    if not re.search(r"Unknown (?:identifier|constant) `Complex\.abs`", message):
        return False, "missing_complex_abs_error", None, [], [], False
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair.get("lane") == "replace_complex_abs":
        return False, "already_attempted_complex_abs_lane", None, [], [], False
    statement_source = str(row.get("lean_statement") or row.get("formal_statement") or "").strip()
    _, body = split_imports(statement_source)
    if not body or len(body) + len(proof) > max_source_chars:
        return False, "source_missing_or_too_long", None, [], [], False
    if ":=" in body:
        return False, "statement_contains_proof", None, [], [], False
    repaired_body, statement_replacements = _COMPLEX_ABS.subn("norm", body)
    repaired_proof, proof_replacements = _COMPLEX_ABS.subn("norm", proof)
    if statement_replacements + proof_replacements == 0:
        return False, "complex_abs_not_in_statement_or_proof", None, [], [], False
    changes = [{
        "kind": "replace_removed_complex_abs_api",
        "from": "Complex.abs",
        "to": "norm",
        "statement_replacements": str(statement_replacements),
        "proof_replacements": str(proof_replacements),
    }]
    variants = [{"strategy": "replace_complex_abs_with_norm", "proof": repaired_proof}]
    return (
        True,
        "eligible",
        repaired_body.rstrip() + " := by",
        variants,
        changes,
        bool(statement_replacements),
    )


API_COMPATIBILITY_RENAMES = {
    "Set.ncard_coe_Finset": "Set.ncard_coe_finset",
    "Real.sqrt_eq_iff_sq_eq": "Real.sqrt_eq_iff_eq_sq",
    "Set.encard_insert_of_not_mem": "Set.encard_insert_of_notMem",
    "Finset.card_insert_of_not_mem": "Finset.card_insert_of_notMem",
    "Nat.dvd_sub'": "Nat.dvd_sub",
    "Nat.not_eq_zero_of_lt": "Nat.ne_zero_of_lt",
    "le_or_lt": "le_or_gt",
    "lt_or_le": "lt_or_ge",
    "Nat.pow_le_pow_of_le_left": "Nat.pow_le_pow_left",
    "Nat.pow_le_pow_of_le_right": "Nat.pow_le_pow_right",
    "div_le_div_iff": "div_le_div_iff₀",
    "div_lt_div_iff": "div_lt_div_iff₀",
    "pow_le_pow_left": "pow_le_pow_left₀",
    "div_le_div_right": "div_le_div_iff_of_pos_right",
    "Set.ncard_insert_of_not_mem": "Set.ncard_insert_of_notMem",
    "lt_div_iff": "lt_div_iff₀",
    "div_lt_iff": "div_lt_iff₀",
    "div_le_iff": "div_le_iff₀",
    "le_div_iff": "le_div_iff₀",
    "pow_le_pow_right": "pow_le_pow_right₀",
}

_BIG_OPERATOR_IN_BINDER = re.compile(r"([∑∏]\s+[^\s,]+)\s+in\s+")


def eligible_api_compatibility_renames(
    row: Mapping[str, Any], *, max_source_chars: int, old_name: str | None = None
) -> tuple[bool, str, str | None, list[dict[str, str]], list[dict[str, str]], bool]:
    proof = str(row.get("proof") or row.get("formal_proof") or "").strip()
    if not proof:
        return False, "missing_proof", None, [], [], False
    if _FORBIDDEN.search(proof):
        return False, "forbidden_proof", None, [], [], False
    message = str(row.get("repair_error") or row.get("error_message") or "")
    statement_source = str(
        row.get("lean_statement") or row.get("formal_statement") or ""
    ).strip()
    imports, body = split_imports(statement_source)
    del imports
    if not body or len(body) + len(proof) > max_source_chars:
        return False, "source_missing_or_too_long", None, [], [], False
    if ":=" in body:
        return False, "statement_contains_proof", None, [], [], False
    repair = row.get("repair") or {}
    attempted_names = {
        str(change.get("from") or "")
        for change in (repair.get("changes") or [])
        if isinstance(change, Mapping)
    } if isinstance(repair, Mapping) else set()
    selected = []
    for previous, current in API_COMPATIBILITY_RENAMES.items():
        if old_name and previous != old_name:
            continue
        if previous in attempted_names:
            continue
        if previous not in message:
            continue
        if previous not in body and previous not in proof:
            continue
        selected.append((previous, current))
    if not selected:
        reason = (
            "already_attempted_selected_api"
            if old_name in attempted_names
            else "missing_selected_api_error"
        )
        return False, reason, None, [], [], False
    repaired_body = body
    repaired_proof = proof
    changes = []
    statement_changed = False
    for previous, current in selected:
        statement_replacements = repaired_body.count(previous)
        proof_replacements = repaired_proof.count(previous)
        repaired_body = repaired_body.replace(previous, current)
        repaired_proof = repaired_proof.replace(previous, current)
        statement_changed = statement_changed or statement_replacements > 0
        changes.append({
            "kind": "replace_removed_or_renamed_mathlib_api",
            "from": previous,
            "to": current,
            "statement_replacements": str(statement_replacements),
            "proof_replacements": str(proof_replacements),
        })
    strategy_suffix = "+".join(previous for previous, _ in selected)
    variants = [{
        "strategy": f"api_compatibility:{strategy_suffix}",
        "proof": repaired_proof,
    }]
    return (
        True,
        "eligible",
        repaired_body.rstrip() + " := by",
        variants,
        changes,
        statement_changed,
    )


def eligible_big_operator_in_syntax(
    row: Mapping[str, Any], *, max_source_chars: int
) -> tuple[bool, str, str | None, list[dict[str, str]], list[dict[str, str]], bool]:
    proof = str(row.get("proof") or row.get("formal_proof") or "").strip()
    if not proof:
        return False, "missing_proof", None, [], [], False
    if _FORBIDDEN.search(proof):
        return False, "forbidden_proof", None, [], [], False
    message = str(row.get("repair_error") or row.get("error_message") or "")
    if "unexpected token 'in'; expected ','" not in message:
        return False, "missing_operator_in_diagnostic", None, [], [], False
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair.get("lane") == "big_operator_in_syntax":
        return False, "already_attempted_operator_syntax", None, [], [], False
    statement_source = str(
        row.get("lean_statement") or row.get("formal_statement") or ""
    ).strip()
    _, body = split_imports(statement_source)
    if not body or len(body) + len(proof) > max_source_chars:
        return False, "source_missing_or_too_long", None, [], [], False
    if ":=" in body:
        return False, "statement_contains_proof", None, [], [], False
    repaired_body, statement_replacements = _BIG_OPERATOR_IN_BINDER.subn(
        lambda match: f"{match.group(1)} ∈ ",
        body,
    )
    repaired_proof, proof_replacements = _BIG_OPERATOR_IN_BINDER.subn(
        lambda match: f"{match.group(1)} ∈ ",
        proof,
    )
    if statement_replacements + proof_replacements == 0:
        return False, "missing_operator_in_source", None, [], [], False
    changes = [{
        "kind": "replace_removed_big_operator_in_binder_syntax",
        "from": "∑/∏ binder in collection",
        "to": "∑/∏ binder ∈ collection",
        "statement_replacements": str(statement_replacements),
        "proof_replacements": str(proof_replacements),
    }]
    return (
        True,
        "eligible",
        repaired_body.rstrip() + " := by",
        [{"strategy": "big_operator_in_to_membership", "proof": repaired_proof}],
        changes,
        statement_replacements > 0,
    )


def short_completion_portfolio(*, max_variants: int) -> list[dict[str, str]]:
    candidates = (
        ("short_native_decide", "by\n  native_decide"),
        ("short_linarith", "by\n  linarith"),
        ("short_norm_num", "by\n  norm_num"),
        ("short_grind", "by\n  grind"),
        ("short_rfl", "by\n  rfl"),
        ("short_omega", "by\n  omega"),
        ("short_nlinarith", "by\n  nlinarith"),
        ("short_ring", "by\n  ring"),
        ("short_decide", "by\n  decide"),
        ("short_aesop", "by\n  aesop"),
    )
    return [
        {"strategy": strategy, "proof": proof}
        for strategy, proof in candidates[:max_variants]
    ]


_COMPLEX_AUTOMATION = re.compile(
    r"Set\.|Finset|IsLeast|IsGreatest|∑|∏|Nat\.digits|Real\.sqrt|\bsqrt\b|√|"
    r"∀|∃|Nat\.factorial|\.roots|Polynomial|\babs\b|\||/\s*\("
)


def is_simple_automation_source(source: str) -> bool:
    return _COMPLEX_AUTOMATION.search(source) is None


def has_hazardous_power(source: str) -> bool:
    """Reject powers known to panic Pantograph/Lean during automated closure."""

    if "^" not in source:
        return False
    if re.search(r"\^\s*\d+\s*\^", source):
        return True
    for match in re.finditer(r"\^\s*(?:\(([^)]*)\)|(\S+))", source):
        exponent = (match.group(1) or match.group(2) or "").strip()
        if "^" in exponent:
            return True
        if exponent.isdigit():
            if int(exponent) > 512:
                return True
            continue
        # Variable, functional, or otherwise non-literal exponents can trigger
        # unbounded reduction in native_decide/grind on this corpus.
        return True
    return False


def eligible_short_placeholder(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str]:
    _, body = source_body_for_repair(row)
    if len(body) > max_source_chars:
        return False, "source_too_long"
    if len(_DECLARATION.findall(body)) > max_declarations:
        return False, "too_many_declarations"
    if not _PLACEHOLDER_PROOF_END.search(body):
        return False, "not_final_placeholder"
    if has_hazardous_power(body):
        return False, "hazardous_power"
    # Remove only the final placeholder and reject any forbidden support.
    prefix = _PLACEHOLDER_PROOF_END.sub(lambda _: ":= by", body)
    if _FORBIDDEN.search(prefix):
        return False, "forbidden_support"
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair.get("lane") in {
        "short_placeholder_portfolio",
        "simple_placeholder_portfolio",
    }:
        return False, "already_attempted_placeholder_lane"
    return True, "eligible"


def eligible_simple_placeholder(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str]:
    allowed, reason = eligible_short_placeholder(
        row,
        max_source_chars=max_source_chars,
        max_declarations=max_declarations,
    )
    if not allowed:
        return allowed, reason
    _, body = source_body_for_repair(row)
    if not is_simple_automation_source(body):
        return False, "complex_automation_source"
    return True, "eligible"


def eligible_simple_empty_proof(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str]:
    if classify_row(row) != "missing_proof_clean":
        return False, "not_clean_missing_proof"
    if row.get("repair_error"):
        return False, "already_attempted"
    _, body = source_body_for_repair(row)
    if len(body) > max_source_chars:
        return False, "source_too_long"
    if len(_DECLARATION.findall(body)) != max_declarations:
        return False, "unexpected_declaration_count"
    if not _EMPTY_PROOF_END.search(body):
        return False, "missing_supported_empty_proof_ending"
    if _FORBIDDEN.search(body):
        return False, "forbidden_source"
    if has_hazardous_power(body):
        return False, "hazardous_power"
    if not is_simple_automation_source(body):
        return False, "complex_automation_source"
    return True, "eligible"


def eligible_guarded_complex_placeholder(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str, str | None]:
    allowed, reason = eligible_short_placeholder(
        row,
        max_source_chars=max_source_chars,
        max_declarations=max_declarations,
    )
    if not allowed:
        return allowed, reason, None
    _, body = source_body_for_repair(row)
    if is_simple_automation_source(body):
        return False, "simple_automation_source", None
    guarded = (
        "set_option maxRecDepth 256\n"
        "set_option maxHeartbeats 100000\n\n"
        + body
    )
    return True, "eligible", guarded


def eligible_targeted_placeholder(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str, str | None]:
    allowed, reason = eligible_short_placeholder(
        row,
        max_source_chars=max_source_chars,
        max_declarations=max_declarations,
    )
    if not allowed:
        return allowed, reason, None
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair:
        return False, "already_attempted_other_repair", None
    _, body = source_body_for_repair(row)
    guarded = (
        "set_option maxRecDepth 256\n"
        "set_option maxHeartbeats 100000\n\n"
        + body
    )
    return True, "eligible", guarded


def eligible_hazardous_power_grind_placeholder(
    row: Mapping[str, Any], *, max_source_chars: int, max_declarations: int
) -> tuple[bool, str, str | None]:
    _, body = source_body_for_repair(row)
    if len(body) > max_source_chars:
        return False, "source_too_long", None
    if len(_DECLARATION.findall(body)) > max_declarations:
        return False, "too_many_declarations", None
    if not _PLACEHOLDER_PROOF_END.search(body):
        return False, "not_final_placeholder", None
    prefix = _PLACEHOLDER_PROOF_END.sub(lambda _: ":= by", body)
    if _FORBIDDEN.search(prefix):
        return False, "forbidden_support", None
    if not has_hazardous_power(body):
        return False, "not_hazardous_power", None
    repair = row.get("repair") or {}
    if isinstance(repair, Mapping) and repair:
        return False, "already_attempted_other_repair", None
    guarded = (
        "set_option maxRecDepth 256\n"
        "set_option maxHeartbeats 50000\n\n"
        + body
    )
    return True, "eligible", guarded


def proof_portfolio(source_body: str, *, max_variants: int) -> list[dict[str, str]]:
    """Return short, standalone proof attempts ordered by goal signals."""

    lowered = source_body.lower()
    arithmetic = (
        any(token in source_body for token in ("ℕ", "ℤ", "ℚ", "ℝ"))
        or any(token in lowered for token in ("nat", "int", "real", "finset"))
        or bool(re.search(r"\d", source_body))
    )
    logical = any(token in source_body for token in ("∀", "∃", "→", "↔", "∧", "∨", "¬"))
    equality = " = " in source_body
    candidates: list[tuple[str, str]] = []
    if logical:
        candidates.extend((
            ("clean_aesop", "by\n  aesop"),
            ("clean_grind", "by\n  grind"),
            ("clean_simp", "by\n  simp"),
        ))
    if arithmetic:
        candidates.extend((
            ("clean_omega", "by\n  omega"),
            ("clean_norm_num", "by\n  norm_num"),
            ("clean_nlinarith", "by\n  nlinarith"),
            ("clean_linarith", "by\n  linarith"),
        ))
    if equality:
        candidates.extend((
            ("clean_ring", "by\n  ring"),
            ("clean_ring_nf", "by\n  ring_nf"),
        ))
    candidates.extend((
        ("clean_simp", "by\n  simp"),
        ("clean_aesop", "by\n  aesop"),
        ("clean_grind", "by\n  grind"),
    ))
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for strategy, proof in candidates:
        if proof in seen:
            continue
        seen.add(proof)
        unique.append({"strategy": strategy, "proof": proof})
        if len(unique) >= max_variants:
            break
    return unique


def complete_source(source_body: str, proof: str) -> str:
    proof = proof.strip()
    if not proof.startswith("by"):
        raise ValueError("repair proof must be a complete `by` proof")
    if _EMPTY_PROOF_END.search(source_body):
        return _EMPTY_PROOF_END.sub(lambda _: f":= {proof}", source_body)
    if _PLACEHOLDER_PROOF_END.search(source_body):
        return _PLACEHOLDER_PROOF_END.sub(lambda _: f":= {proof}", source_body)
    raise ValueError("source does not end in a supported empty/placeholder proof")


def with_imports(imports: Iterable[str], body: str) -> str:
    block = "\n".join(f"import {module}" for module in imports)
    return (block + "\n\n" if block else "") + body.strip() + "\n"


def prepare(args: argparse.Namespace) -> None:
    categories: Counter[str] = Counter()
    exclusion: Counter[str] = Counter()
    eligible: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in iter_jsonl(args.fail_file):
        category = classify_row(row)
        categories[category] += 1
        repaired_body: str | None = None
        prepared_variants: list[dict[str, str]] | None = None
        statement_changed_override: bool | None = None
        changes: list[dict[str, str]] = []
        if args.lane == "closed_decimal_rational":
            allowed, reason, repaired_body, change = eligible_closed_decimal_rational(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
            if change is not None:
                changes.append(change)
        elif args.lane == "closed_integer_norm_num":
            allowed, reason = eligible_closed_integer_norm_num(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
        elif args.lane == "pure_closed_arithmetic":
            allowed, reason, repaired_body, change = eligible_pure_closed_arithmetic(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
            if change is not None:
                changes.append(change)
        elif args.lane == "trim_trailing_no_goals":
            allowed, reason, repaired_body, prepared_variants = eligible_trim_trailing_no_goals(
                row,
                max_source_chars=args.max_source_chars,
                max_variants=args.max_variants,
            )
        elif args.lane == "replace_complex_abs":
            (
                allowed,
                reason,
                repaired_body,
                prepared_variants,
                changes,
                statement_changed_override,
            ) = eligible_replace_complex_abs(
                row,
                max_source_chars=args.max_source_chars,
            )
        elif args.lane == "api_compatibility_renames":
            (
                allowed,
                reason,
                repaired_body,
                prepared_variants,
                changes,
                statement_changed_override,
            ) = eligible_api_compatibility_renames(
                row,
                max_source_chars=args.max_source_chars,
                old_name=args.api_old_name,
            )
        elif args.lane == "big_operator_in_syntax":
            (
                allowed,
                reason,
                repaired_body,
                prepared_variants,
                changes,
                statement_changed_override,
            ) = eligible_big_operator_in_syntax(
                row,
                max_source_chars=args.max_source_chars,
            )
        elif args.lane == "short_placeholder_portfolio":
            allowed, reason = eligible_short_placeholder(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
        elif args.lane == "simple_placeholder_portfolio":
            allowed, reason = eligible_simple_placeholder(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
        elif args.lane == "simple_empty_portfolio":
            allowed, reason = eligible_simple_empty_proof(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
        elif args.lane == "guarded_complex_placeholder":
            allowed, reason, repaired_body = eligible_guarded_complex_placeholder(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
        elif args.lane == "targeted_placeholder_two_tactic":
            allowed, reason, repaired_body = eligible_targeted_placeholder(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
            prepared_variants = [
                {"strategy": "targeted_native_decide", "proof": "by\n  native_decide"},
                {"strategy": "targeted_grind", "proof": "by\n  grind"},
            ][: args.max_variants]
        elif args.lane == "hazardous_power_grind_placeholder":
            allowed, reason, repaired_body = eligible_hazardous_power_grind_placeholder(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
            prepared_variants = [
                {"strategy": "hazardous_power_guarded_grind", "proof": "by\n  grind"},
            ][: args.max_variants]
        else:
            allowed, reason = eligible_missing_proof(
                row,
                max_source_chars=args.max_source_chars,
                max_declarations=args.max_declarations,
            )
        if not allowed:
            exclusion[reason] += 1
            continue
        imports, original_body = source_body_for_repair(row)
        body = repaired_body if repaired_body is not None else original_body
        record_id = str(row["record_id"])
        variants = prepared_variants or (
            [{"strategy": "typed_decimal_norm_num", "proof": "by\n  norm_num"}]
            if args.lane == "closed_decimal_rational"
            else [{"strategy": "closed_integer_norm_num", "proof": "by\n  norm_num"}]
            if args.lane == "closed_integer_norm_num"
            else [{"strategy": "typed_pure_arithmetic_norm_num", "proof": "by\n  norm_num"}]
            if args.lane == "pure_closed_arithmetic"
            else short_completion_portfolio(max_variants=args.max_variants)
            if args.lane in {
                "short_placeholder_portfolio",
                "simple_placeholder_portfolio",
                "simple_empty_portfolio",
                "guarded_complex_placeholder",
            }
            else proof_portfolio(body, max_variants=args.max_variants)
        )
        payload = {
            "schema_version": "numinamath_repair_candidate_v1",
            "repair_version": REPAIR_VERSION,
            "batch_id": args.batch_id,
            "record_id": record_id,
            "category": category,
            "lane": args.lane,
            "imports": list(imports),
            "source_body": body,
            "original_source_body": original_body,
            "statement_changed": (
                statement_changed_override
                if statement_changed_override is not None
                else bool(changes)
            ),
            "changes": changes,
            "original_error_message": str(row.get("error_message") or ""),
            "original_record_hash": str(row.get("record_hash") or ""),
            "variants": variants,
        }
        payload["candidate_hash"] = canonical_hash(payload)
        rank = (
            len(body),
            hashlib.sha256(f"{args.seed}:{record_id}".encode()).hexdigest(),
        )
        eligible.append((rank, payload))
    eligible.sort(key=lambda item: item[0])
    selected = [payload for _, payload in eligible[: args.batch_size]]
    batch_dir = args.output_root / args.batch_id
    write_jsonl(batch_dir / "candidate_manifest.jsonl", selected)
    report = {
        "schema_version": "numinamath_repair_prepare_report_v1",
        "repair_version": REPAIR_VERSION,
        "batch_id": args.batch_id,
        "lane": args.lane,
        "input_fail_rows": sum(categories.values()),
        "classification": dict(categories),
        "eligible_clean_missing_proof": len(eligible),
        "selected_rows": len(selected),
        "selection_exclusions": dict(exclusion),
        "max_source_chars": args.max_source_chars,
        "max_declarations": args.max_declarations,
        "max_variants": args.max_variants,
        "seed": args.seed,
        "candidate_manifest_sha256": sha256_file(batch_dir / "candidate_manifest.jsonl"),
        "external_api_called": False,
    }
    _write_json(args.output_root / "failure_classification.json", {
        "schema_version": "numinamath_failure_classification_v1",
        "rows": sum(categories.values()),
        "classification": dict(categories),
    })
    _write_json(batch_dir / "prepare_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _runtime_failure(result: Mapping[str, Any]) -> bool:
    return bool(result.get("timed_out")) or result.get("error_type") == "pantograph_error"


def start_verifier(args: argparse.Namespace) -> PantographTheoremVerifier:
    verifier = PantographTheoremVerifier(
        args.lean_project,
        imports=("Mathlib",),
        timeout=args.timeout,
        startup_timeout=900,
    )
    warmup = verifier.warmup(timeout=180)
    if not warmup.success:
        verifier.close()
        raise RuntimeError(f"Pantograph warmup failed: {warmup.diagnostics}")
    return verifier


def _result_error(result: Mapping[str, Any]) -> str:
    diagnostics = str(result.get("diagnostics") or "").strip()
    if diagnostics:
        return diagnostics
    errors = result.get("errors") or ()
    return "\n".join(str(item) for item in errors).strip() or str(result.get("error_type") or "failed")


def verify(args: argparse.Namespace) -> None:
    batch_dir = args.output_root / args.batch_id
    candidates = read_jsonl(batch_dir / "candidate_manifest.jsonl")
    if not candidates:
        raise RuntimeError("candidate manifest is empty; run prepare first")
    identity = environment_identity(args.lean_project, ("Mathlib",))
    result_path = batch_dir / "verification_results.jsonl"
    cached = {
        str(row["variant_hash"]): row
        for row in read_jsonl(result_path)
        if row.get("repair_version") == REPAIR_VERSION and row.get("variant_hash")
    }
    selected: dict[str, dict[str, Any]] = {}
    verifier: PantographTheoremVerifier | None = None
    new_attempts = 0
    restarts = 0
    try:
        for index, candidate in enumerate(candidates, start=1):
            chosen: dict[str, Any] | None = None
            attempts: list[dict[str, Any]] = []
            for variant in candidate["variants"]:
                full_body = complete_source(candidate["source_body"], variant["proof"])
                source = with_imports((), full_body)
                variant_hash = canonical_hash({
                    "candidate_hash": candidate["candidate_hash"],
                    "strategy": variant["strategy"],
                    "proof": variant["proof"],
                    "environment_hash": identity["environment_hash"],
                })
                result = cached.get(variant_hash)
                if result is None:
                    if verifier is None:
                        verifier = start_verifier(args)
                    checked = verifier.check_source(source, timeout=args.timeout, reject_forbidden=True)
                    restart_count = 0
                    if checked.error_type == "pantograph_error" or checked.timed_out:
                        verifier.close()
                        verifier = start_verifier(args)
                        restarts += 1
                        restart_count = 1
                        checked = verifier.check_source(source, timeout=args.timeout, reject_forbidden=True)
                    result = {
                        "schema_version": "numinamath_repair_verification_result_v1",
                        "repair_version": REPAIR_VERSION,
                        "batch_id": args.batch_id,
                        "record_id": candidate["record_id"],
                        "candidate_hash": candidate["candidate_hash"],
                        "variant_hash": variant_hash,
                        "strategy": variant["strategy"],
                        "proof": variant["proof"],
                        "success": checked.success,
                        "pantograph_verified": SUCCESS if checked.success else FAIL,
                        "diagnostics": checked.diagnostics,
                        "errors": list(checked.errors),
                        "warnings": list(checked.warnings),
                        "timed_out": checked.timed_out,
                        "error_type": checked.error_type,
                        "verification_seconds": checked.check_seconds,
                        "pantograph_restart_count": restart_count,
                        "assembled_source_hash": sha256_text(source),
                        "environment": identity,
                    }
                    append_jsonl(result_path, result)
                    cached[variant_hash] = result
                    new_attempts += 1
                attempts.append(result)
                chosen = result
                if result.get("success"):
                    break
            if chosen is None:
                raise RuntimeError(f"candidate {candidate['record_id']} has no variants")
            selected[str(candidate["record_id"])] = {
                "candidate": candidate,
                "chosen": chosen,
                "attempts": attempts,
            }
            if index % 25 == 0:
                solved = sum(bool(item["chosen"].get("success")) for item in selected.values())
                print(json.dumps({
                    "records_completed": index,
                    "records_total": len(candidates),
                    "records_solved": solved,
                    "new_variant_attempts": new_attempts,
                }), flush=True)
    finally:
        if verifier is not None:
            verifier.close()

    if args.command == "verify-candidates":
        solved = sum(bool(item["chosen"].get("success")) for item in selected.values())
        report = {
            "schema_version": "numinamath_candidate_verification_report_v1",
            "repair_version": REPAIR_VERSION,
            "batch_id": args.batch_id,
            "input_candidates": len(candidates),
            "records_completed": len(selected),
            "records_solved": solved,
            "records_failed": len(selected) - solved,
            "new_variant_attempts": new_attempts,
            "pantograph_restarts": restarts,
            "candidate_manifest_sha256": sha256_file(batch_dir / "candidate_manifest.jsonl"),
            "verification_results_sha256": sha256_file(result_path),
            "environment": identity,
            "master_dataset_mutated": False,
            "external_api_called": False,
        }
        _write_json(batch_dir / "candidate_verification_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    success_rows = read_jsonl(args.success_file)
    success_ids = {str(row["record_id"]) for row in success_rows}
    repaired_success: list[dict[str, Any]] = []
    remaining_fail: list[dict[str, Any]] = []
    attempted_fail = 0
    for row in iter_jsonl(args.fail_file):
        record_id = str(row["record_id"])
        selection = selected.get(record_id)
        if selection is None:
            remaining_fail.append(row)
            continue
        candidate = selection["candidate"]
        chosen = selection["chosen"]
        attempts = selection["attempts"]
        lane = candidate.get("lane", "clean_missing_proof")
        manual_rationale = str(candidate.get("manual_rationale") or "").strip()
        repair_payload = {
            "repair_version": REPAIR_VERSION,
            "batch_id": args.batch_id,
            "classification": candidate["category"],
            "lane": lane,
            "minimal_edit": (
                manual_rationale
                if lane == "codex_manual" and manual_rationale
                else "applied explicit hash-locked edits selected by Codex manual review"
                if lane == "codex_manual"
                else
                "replaced only the final placeholder proof with a locally verified short proof"
                if lane in {
                    "short_placeholder_portfolio",
                    "simple_placeholder_portfolio",
                    "simple_empty_portfolio",
                    "guarded_complex_placeholder",
                    "targeted_placeholder_two_tactic",
                    "hazardous_power_grind_placeholder",
                }
                else "replaced the removed `Complex.abs` API with the current equivalent `norm` identifier"
                if lane == "replace_complex_abs"
                else "replaced only explicitly diagnosed removed or renamed mathlib API identifiers"
                if lane == "api_compatibility_renames"
                else "replaced only diagnosed removed `∑`/`∏` binder `in` syntax with membership syntax"
                if lane == "big_operator_in_syntax"
                else "removed only trailing top-level tactic lines executed after the proof goal was already closed"
                if lane == "trim_trailing_no_goals"
                else "added one numeric type annotation to disambiguate a pure closed arithmetic goal and completed the final proof"
                if lane == "pure_closed_arithmetic"
                else "added one ℚ annotation to disambiguate the closed decimal goal and completed `:= by`"
                if candidate.get("statement_changed")
                else "completed the original empty final `:= by` proof"
            ),
            "statement_changed": bool(candidate.get("statement_changed")),
            "changes": candidate.get("changes") or [],
            "attempted_strategies": [item["strategy"] for item in attempts],
            "selected_strategy": chosen["strategy"] if chosen.get("success") else None,
            "review_method": (
                "codex_explicit_hash_locked" if lane == "codex_manual" else "rule_based_candidate"
            ),
            "selected_source_sha256": candidate.get("selected_source_sha256"),
            "external_api_called": False,
        }
        if chosen.get("success"):
            full_body = complete_source(candidate["source_body"], chosen["proof"])
            full_source = with_imports(candidate["imports"], full_body)
            updated = dict(row)
            updated.update({
                "formal_ground_truth": full_source.strip(),
                "ground_truth_type": "complete",
                "proof": chosen["proof"],
                "proof_present": True,
                "proof_source": (
                    "codex_manual_repair" if lane == "codex_manual" else "codex_minimal_repair"
                ),
                "verification_status": "success",
                "raw_verification_status": "success",
                "raw_verifier_success": True,
                "verification_backend": "pantograph",
                "timed_out": False,
                "lean_version": identity["lean_version"],
                "mathlib_commit": identity["mathlib_commit"],
                "environment_hash": identity["environment_hash"],
                "assembled_source_hash": chosen["assembled_source_hash"],
                "repair": repair_payload,
            })
            if candidate.get("statement_changed"):
                statement_body = _EMPTY_PROOF_END.sub("", candidate["source_body"]).strip()
                updated["lean_statement"] = with_imports(candidate["imports"], statement_body).strip()
                updated["formal_statement"] = with_imports(
                    candidate["imports"], candidate["source_body"]
                ).strip()
            updated.pop("repair_error", None)
            updated["record_hash"] = dataset_record_hash(updated, source=NUMINA_SOURCE)
            repaired_success.append(add_dataset_contract(updated, source=NUMINA_SOURCE, status=SUCCESS))
        else:
            updated = dict(row)
            updated["repair"] = repair_payload
            updated["repair_error"] = "\n\n".join(
                f"{item['strategy']}: {_result_error(item)}" for item in attempts
            )
            remaining_fail.append(updated)
            attempted_fail += 1

    promoted_ids = {str(row["record_id"]) for row in repaired_success}
    if promoted_ids & success_ids:
        raise RuntimeError("a promoted repair record already exists in success")
    if len(promoted_ids) != len(repaired_success):
        raise RuntimeError("promoted repairs contain duplicate record IDs")
    success_before = len(success_rows)
    fail_before = len(remaining_fail) + len(repaired_success)
    success_rows.extend(repaired_success)
    write_jsonl(args.success_file, success_rows)
    write_jsonl(args.fail_file, remaining_fail)

    report = {
        "schema_version": "numinamath_repair_batch_report_v1",
        "repair_version": REPAIR_VERSION,
        "batch_id": args.batch_id,
        "input_candidates": len(candidates),
        "new_variant_attempts": new_attempts,
        "pantograph_restarts": restarts,
        "promoted_to_success": len(repaired_success),
        "attempted_but_still_fail": attempted_fail,
        "success_before": success_before,
        "success_after": len(success_rows),
        "fail_before": fail_before,
        "fail_after": len(remaining_fail),
        "total_rows_after": len(success_rows) + len(remaining_fail),
        "success_sha256": sha256_file(args.success_file),
        "fail_sha256": sha256_file(args.fail_file),
        "verification_results_sha256": sha256_file(result_path),
        "environment": identity,
        "external_api_called": False,
    }
    _write_json(batch_dir / "batch_report.json", report)

    if args.verification_report.exists():
        verification_report = json.loads(args.verification_report.read_text(encoding="utf-8"))
        verification_report["success"] = len(success_rows)
        verification_report["fail"] = len(remaining_fail)
        verification_report["unresolved"] = 0
        verification_report["complete"] = True
        verification_report.setdefault("repair_batches", []).append({
            "batch_id": args.batch_id,
            "repair_version": REPAIR_VERSION,
            "promoted_to_success": len(repaired_success),
            "attempted_but_still_fail": attempted_fail,
            "report": str((batch_dir / "batch_report.json").resolve()),
        })
        verification_report.setdefault("outputs", {}).setdefault("success", {}).update({
            "rows": len(success_rows), "sha256": report["success_sha256"]
        })
        verification_report.setdefault("outputs", {}).setdefault("fail", {}).update({
            "rows": len(remaining_fail), "sha256": report["fail_sha256"]
        })
        _write_json(args.verification_report, verification_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("prepare", "verify", "verify-candidates"))
    value.add_argument("--fail-file", type=Path, default=root / "verified_data" / "numinamath_verified_fail.jsonl")
    value.add_argument("--success-file", type=Path, default=root / "verified_data" / "numinamath_verified_success.jsonl")
    value.add_argument("--verification-report", type=Path, default=root / "verified_data" / "numinamath_verification_report.json")
    value.add_argument("--output-root", type=Path, default=Path("outputs/numinamath_repair"))
    value.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    value.add_argument("--batch-id", default="batch_0001")
    value.add_argument(
        "--lane",
        choices=(
            "clean_missing_proof",
            "closed_decimal_rational",
            "closed_integer_norm_num",
            "pure_closed_arithmetic",
            "trim_trailing_no_goals",
            "replace_complex_abs",
            "api_compatibility_renames",
            "big_operator_in_syntax",
            "short_placeholder_portfolio",
            "simple_placeholder_portfolio",
            "simple_empty_portfolio",
            "guarded_complex_placeholder",
            "targeted_placeholder_two_tactic",
            "hazardous_power_grind_placeholder",
        ),
        default="clean_missing_proof",
    )
    value.add_argument("--batch-size", type=int, default=200)
    value.add_argument("--max-source-chars", type=int, default=2500)
    value.add_argument("--max-declarations", type=int, default=1)
    value.add_argument("--max-variants", type=int, default=5)
    value.add_argument(
        "--api-old-name",
        choices=tuple(API_COMPATIBILITY_RENAMES),
        default=None,
    )
    value.add_argument("--timeout", type=int, default=20)
    value.add_argument("--seed", type=int, default=20260808)
    return value


def main() -> None:
    args = parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
