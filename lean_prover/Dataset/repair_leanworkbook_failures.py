"""Prioritized, restartable repair of verified LeanWorkbook failures.

API proposal generation is deliberately separate from Pantograph finalization
so it can run while another memory-heavy verification job owns the sole safe
Pantograph worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from lean_prover.Data.compile_errors import (
    LeanFailureDetail,
    LeanVerificationStatus,
    classify_lean_diagnostics,
)
from lean_prover.Dataset.build_verified_datasets import (
    FAIL,
    SUCCESS,
    _write_json,
    add_dataset_contract,
    sha256_file,
)
from lean_prover.Dataset.expand_leanworkbook import split_statement
from lean_prover.Repair.discovery_attempt.client import (
    DeepSeekRepairClient,
    RepairClientConfig,
    RepairClientError,
)
from lean_prover.Repair.discovery_attempt.pipeline import RepairPipeline
from lean_prover.Repair.discovery_attempt.schema import (
    APICallAttempt,
    AggregatedAttempt,
    EnvironmentContract,
    RepairClientResponse,
    RepairData,
    RepairRequest,
    canonical_sha256,
)
from lean_prover.lean_training.verification.pantograph import (
    PantographTheoremVerifier,
)


DEFAULT_COUNT = 500
DEFAULT_SEED = 20260807
SOURCE = "InternLM/Lean-Workbook"
MANUAL_REPAIR_VERSION = "codex_manual_full_v1"

# These are deliberately conservative, source-level compatibility repairs for
# identifiers that are present in the frozen LeanWorkbook failure diagnostics.
# Every resulting declaration is still required to pass local Pantograph; a
# rule match is never treated as evidence of correctness by itself.
REAL_IDENTIFIERS = {
    "sin", "cos", "tan", "tanh", "exp", "log", "logb", "sinh", "cosh", "arcsin", "arccos",
    "sin_add", "cos_add", "sin_sub", "cos_sub", "sin_neg", "cos_neg",
    "sin_sq_add_cos_sq", "cos_sq_add_sin_sq", "sin_sq", "cos_sq",
    "sin_two_mul", "cos_two_mul", "sin_three_mul", "cos_three_mul",
    "tan_eq_sin_div_cos", "cos_le_one", "sin_le_one", "abs_sin_le_one",
    "abs_cos_le_one", "cos_zero", "sin_zero", "cos_pi_div_four",
    "sin_pi_div_four", "cos_pi_div_two", "sin_pi_div_two", "cos_pi_div_three",
    "cos_pi_div_two_sub", "arcsin_eq_pi_div_two_sub_arccos", "exp_pos", "exp_add",
    "exp_neg", "exp_injective", "exp_le_exp", "add_one_le_exp",
    "log_le_sub_one_of_pos", "log_mul", "log_nonneg", "rpow_def", "pi_pos",
    "sqrt_eq_iff_mul_self_eq_of_pos", "logb_div", "norm_eq_abs",
    "sinh_neg", "cosh_eq", "exp_log", "log_rpow", "rpow_mul", "one_le_rpow",
    "sq_sqrt", "neg_one_le_cos", "continuous_cos", "continuous_sin",
    "sin_pi_div_two_sub", "tanh_eq_sinh_div_cosh", "sinh_add", "cosh_add",
    "le_sqrt_of_sq_le", "sqrt_le_left", "exp_ne_zero", "sin_pi_sub",
    "sin_pos_of_pos_of_lt_pi", "cos_pi", "sin_eq_zero_iff",
    "cos_pos_of_mem_Ioo", "log_inv", "log_lt_iff_lt_exp",
    "logb_eq_iff_rpow_eq", "log_lt_sub_one_of_pos",
    "cos_pi_sub", "logb_mul", "exp_mul", "log_pos", "sin_pi",
    "cos_nonneg_of_mem_Icc", "sin_nonneg_of_nonneg_of_le_pi",
    "sin_pi_div_three", "lt_sqrt_of_sq_lt", "log_div",
    "cos_pi_sub", "logb_mul", "exp_mul", "log_pos", "sin_pi",
    "cos_nonneg_of_mem_Icc", "sin_nonneg_of_nonneg_of_le_pi",
    "sin_pi_div_three", "lt_sqrt_of_sq_lt", "log_div",
}
COMPLEX_IDENTIFIERS = {
    "sin", "cos", "tan", "exp", "sinh", "cosh",
    "sin_add", "cos_add", "sin_sub", "cos_sub", "sin_neg", "cos_neg",
    "sin_sq_add_cos_sq", "cos_sq_add_sin_sq", "sin_two_mul", "cos_two_mul",
    "tan_eq_sin_div_cos", "exp_pos", "exp_add", "exp_neg",
}
NAT_IDENTIFIERS = {
    "choose", "choose_succ_succ", "choose_symm", "choose_symm_of_eq_add",
    "choose_le_middle", "choose_mul", "fib", "fib_add_two", "fib_gcd",
    "factorial", "totient_mul", "totient_pos", "divisors", "ModEq",
    "zero_lt_succ", "add_succ", "pow_mod",
    "totient", "prime_two", "Coprime", "choose_two_right",
    "fib_two", "fib_two_mul", "fib_two_mul_add_one",
    "fib_two", "fib_two_mul", "fib_two_mul_add_one",
}
RENAMED_IDENTIFIERS = {
    "div_le_div_iff": "div_le_div_iff₀",
    "div_le_iff": "div_le_iff₀",
    "div_lt_div_iff": "div_lt_div_iff₀",
    "div_lt_iff": "div_lt_iff₀",
    "le_div_iff": "le_div_iff₀",
    "lt_div_iff": "lt_div_iff₀",
    "nat_sub_dvd_pow_sub_pow": "Nat.sub_dvd_pow_sub_pow",
    "Real.sqrt_eq_iff_sq_eq": "Real.sqrt_eq_iff_eq_sq",
    "sqrt_eq_iff_sq_eq": "Real.sqrt_eq_iff_eq_sq",
    "true_and_iff": "true_and",
    "true_or_iff": "true_or",
    "le_or_lt": "le_or_gt",
    "φ": "Nat.totient",
    "Complex.abs": "norm",
    "Complex.abs_pow": "Complex.norm_pow",
    "Complex.abs_def": "Complex.norm_def",
    "Int.ceil_add_int": "Int.ceil_add_intCast",
    "Int.floor_add_int": "Int.floor_add_intCast",
    "rank_quotient_add_rank": "Submodule.rank_quotient_add_rank",
    "Coprime.gcd_mul_left_cancel": "Nat.Coprime.gcd_mul_left_cancel",
    "one_le_pow_of_one_le": "one_le_pow_of_one_le'",
    "Commutative": "Function.Commutative",
    "floor": "Int.floor",
    "ceil": "Int.ceil",
    "Int.mod_self": "Int.emod_self",
    "Function.funext_iff": "funext_iff",
    "Nat.odd_iff_not_even": "Nat.not_even_iff_odd",
    "Complex.normSq_eq_abs": "Complex.normSq_eq_norm_sq",
    "abs_add": "abs_add_le",
    "Int.mod_self": "Int.emod_self",
    "Function.funext_iff": "funext_iff",
    "Nat.odd_iff_not_even": "Nat.not_even_iff_odd",
    "Complex.normSq_eq_abs": "Complex.normSq_eq_norm_sq",
    "abs_add": "abs_add_le",
}

# Individually reviewed clean proofs for records where an obsolete helper lemma
# should be removed rather than mechanically renamed.  These are candidates,
# not accepted repairs; local Pantograph remains the only success criterion.
SPECIAL_PROOF_REPAIRS = {
    "lean_workbook_plus_1072": "by\n  rfl",
    "lean_workbook_plus_38899": "by\n  simpa using norm_mul z w",
    "lean_workbook_plus_45835": "by\n  simpa using norm_mul a b",
    "lean_workbook_plus_60366": "by\n  intro h\n  simpa [one_div, h] using norm_inv z",
    "lean_workbook_plus_43943": "by\n  simpa using norm_pow z 2",
    "lean_workbook_plus_70512": "by\n  simpa using norm_div z w",
    "lean_workbook_plus_31408": "by\n  simpa using norm_pow z n",
    "lean_workbook_plus_18882": "by\n  simpa using norm_pow z 15",
    "lean_workbook_plus_37716": "by\n  simpa using norm_mul c z",
    "lean_workbook_plus_32084": "by\n  exact ⟨_, rfl⟩",
    "lean_workbook_plus_58490": "by\n  intro a b\n  simpa using Complex.norm_add_mul_I a b",
    "lean_workbook_plus_52395": "by\n  simpa using Complex.norm_add_mul_I a b",
    "lean_workbook_plus_2117": (
        "by\n  intro h\n  nlinarith [sq_nonneg (‖z‖ ^ 2 - 1)]"
    ),
    "lean_workbook_plus_26682": "by\n  simpa [h₁, h₂]",
    "lean_workbook_plus_39894": "by\n  simpa [h]",
    "lean_workbook_plus_13497": "by\n  exact h₁",
    "lean_workbook_plus_4250": "by\n  exact h",
    "lean_workbook_plus_46936": "by\n  norm_num",
    "lean_workbook_plus_7627": "by\n  rfl",
    "lean_workbook_plus_31690": "by\n  ring_nf",
    "lean_workbook_plus_53373": (
        "by\n  simpa only [sub_eq_add_neg, abs_neg, Real.norm_eq_abs] "
        "using norm_add_le a (-b)"
    ),
    "lean_workbook_plus_9252": (
        "by\n  simpa only [sub_eq_add_neg, abs_neg, Real.norm_eq_abs] "
        "using norm_add_le a (-b)"
    ),
    "lean_workbook_plus_35590": (
        "by\n  simpa using norm_mul (a + b * I) (c + d * I)"
    ),
    "lean_workbook_plus_73420": "by\n  norm_num [hp, hq]",
    "lean_workbook_plus_75453": (
        "by\n  intro n\n  exact ⟨Nat.even_iff.symm, Nat.odd_iff.symm⟩"
    ),
    "lean_workbook_plus_23305": "by\n  rw [h₁]\n  field_simp",
    "lean_workbook_plus_2893": (
        "by\n  have hn : ‖x‖ < 1 := by "
        "simpa [Real.norm_eq_abs, abs_of_nonneg hx.1] using hx.2\n"
        "  simpa [one_div] using (tsum_geometric_of_norm_lt_one (ξ := x) hn)"
    ),
    "lean_workbook_plus_13924": (
        "by\n  have hn : ‖x‖ < 1 := by "
        "simpa [Real.norm_eq_abs, abs_of_pos hx.1] using hx.2\n"
        "  simpa [one_div] using (tsum_geometric_of_norm_lt_one (ξ := x) hn)"
    ),
    "lean_workbook_plus_18395": (
        "by\n  rw [tsum_mul_left]\n"
        "  have hn : ‖r‖ < 1 := by "
        "simpa [Real.norm_eq_abs, abs_of_pos h] using h'\n"
        "  rw [tsum_geometric_of_norm_lt_one hn]\n  simp [div_eq_mul_inv]"
    ),
    "lean_workbook_plus_1680": (
        "by\n  rw [tsum_mul_left]\n"
        "  have hn : ‖r‖ < 1 := by simpa [Real.norm_eq_abs] using h\n"
        "  rw [tsum_geometric_of_norm_lt_one hn]\n  simp [div_eq_mul_inv]"
    ),
    "lean_workbook_plus_64464": (
        "by\n  intro x hx\n"
        "  have hn : ‖x‖ < 1 := by simpa [Real.norm_eq_abs] using hx\n"
        "  simpa [one_div] using (tsum_geometric_of_norm_lt_one (ξ := x) hn).symm"
    ),
    "lean_workbook_plus_58808": "by\n  exact h₁.mul h₂",
    "lean_workbook_plus_28806": (
        "by\n  exact hgh.orderOf_mul_eq_mul_orderOf_of_coprime hmn"
    ),
    "lean_workbook_plus_31794": (
        "by\n  constructor\n"
        "  · intro h x y hxy hfeq\n    exact hxy (h hfeq)\n"
        "  · intro h x y hfeq\n    by_contra hxy\n    exact (h x y hxy) hfeq"
    ),
    "lean_workbook_plus_71340": "by\n  positivity",
    "lean_workbook_plus_16111": "by\n  positivity",
    "lean_workbook_plus_65017": "by\n  norm_num",
    "lean_workbook_plus_47687": "by\n  subst a\n  omega",
    "lean_workbook_plus_63763": (
        "by\n  intro x hx\n"
        "  exact ⟨⟨hx.1, hx.2.1⟩, fun h => hx.2.2 h.2⟩"
    ),
    "lean_workbook_plus_61970": (
        "by\n  intro x hx\n"
        "  apply (div_le_div_iff₀ (by norm_num : (0 : ℝ) < 2) (by linarith)).2\n"
        "  nlinarith"
    ),
    "lean_workbook_plus_32000": "by\n  fun_prop",
    "lean_workbook_plus_52119": "by\n  omega",
    "lean_workbook_plus_78123": (
        "by\n  constructor\n"
        "  · congr 1\n    ring\n"
        "  · simpa only [Real.norm_eq_abs] using "
        "norm_add_le (x - f x) (f x - f y)"
    ),
    "lean_workbook_plus_32950": (
        "by\n  apply le_antisymm ?_ (hx 1 zero_lt_one).1\n"
        "  by_contra h\n"
        "  have xpos : 0 < x := lt_of_not_ge h\n"
        "  have hsmall := (hx (x / 2) (by positivity)).2\n"
        "  linarith"
    ),
    "lean_workbook_plus_61844": "by\n  intro h\n  exact h.neg",
    "lean_workbook_plus_19940": "by\n  intro a b\n  positivity",
    "lean_workbook_plus_8155": (
        "by\n  rintro ⟨h0, h1⟩\n"
        "  rcases hf with ⟨a, b, hab⟩\n"
        "  have e0 := hab 0\n  have e1 := hab 1\n  have en := hab (-10)\n"
        "  norm_num at e0 e1 en ⊢\n  linarith"
    ),
    "lean_workbook_plus_36959": (
        "by\n  rcases Int.even_or_odd x with hx | hx\n"
        "  · exact (hx.pow_of_ne_zero (by norm_num)).add "
        "(hx.pow_of_ne_zero (by norm_num))\n"
        "  · exact hx.pow.add_odd hx.pow"
    ),
    "lean_workbook_plus_44231": (
        "by\n  apply Nat.pow_right_injective (by norm_num : 2 ≤ 5)\n"
        "  calc\n    5 ^ n = 3125 := h₀\n    _ = 5 ^ 5 := by norm_num"
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _replace_identifier(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace a complete Lean identifier without touching qualified names."""

    pattern = re.compile(rf"(?<![\w.]){re.escape(old)}(?![\w])")
    return pattern.subn(new, text)


_OLD_BOUNDED_OPERATOR = re.compile(r"([\u2211\u220f]\s*[^,\n]+?)\s+in\b")
_FACTORIAL_TACTIC_NAMES = {
    "aesop",
    "assumption",
    "by_contra",
    "constructor",
    "contrapose",
    "exact",
    "field_simp",
    "grind",
    "linarith",
    "norm_num",
    "omega",
    "positivity",
    "ring",
    "ring_nf",
    "simp",
    "simp_all",
    "simpa",
}


def _replace_old_bounded_operator_notation(value: str) -> tuple[str, int]:
    """Translate Lean 3 ``sum/prod ... in`` binders to current mathlib syntax."""

    return _OLD_BOUNDED_OPERATOR.subn(
        lambda match: match.group(1).rstrip() + " \u2208", value
    )


def _replace_postfix_nat_factorial(value: str) -> tuple[str, int]:
    """Translate legacy postfix Nat factorial without touching tactic ``!`` forms."""

    result = value
    replacements = 0
    cursor = 0
    while cursor < len(result):
        marker = result.find("!", cursor)
        if marker < 0:
            break
        if marker + 1 < len(result) and result[marker + 1] == "=":
            cursor = marker + 1
            continue
        end = marker - 1
        while end >= 0 and result[end].isspace():
            end -= 1
        start = end
        if end >= 0 and result[end] == ")":
            depth = 1
            start = end - 1
            while start >= 0 and depth:
                if result[start] == ")":
                    depth += 1
                elif result[start] == "(":
                    depth -= 1
                start -= 1
            if depth:
                cursor = marker + 1
                continue
            start += 1
        else:
            while start >= 0 and (result[start].isalnum() or result[start] in "_'"):
                start -= 1
            start += 1
        operand = result[start : end + 1]
        if not operand or operand in _FACTORIAL_TACTIC_NAMES:
            cursor = marker + 1
            continue
        replacement = f"Nat.factorial {operand}"
        result = result[:start] + replacement + result[marker + 1 :]
        replacements += 1
        cursor = start + len(replacement)
    return result, replacements


def manual_compatibility_repair(
    statement: str,
    proof: str,
    *,
    record_id: str = "",
) -> tuple[str, str, list[dict[str, Any]]]:
    """Apply auditable high-confidence compatibility edits.

    The namespace family is selected from the theorem's explicit scalar type.
    Ambiguous theorems containing both real and complex scalars are skipped for
    the shared analytic identifiers instead of guessing.
    """

    repaired_statement = _text(statement)
    repaired_proof = _text(proof)
    changes: list[dict[str, Any]] = []

    repaired_statement, statement_count = _replace_old_bounded_operator_notation(
        repaired_statement
    )
    repaired_proof, proof_count = _replace_old_bounded_operator_notation(repaired_proof)
    if statement_count or proof_count:
        changes.append(
            {
                "from": "legacy_bounded_operator_in",
                "to": "current_bounded_operator_membership",
                "reason": "current_mathlib_syntax",
                "statement_occurrences": statement_count,
                "proof_occurrences": proof_count,
            }
        )

    repaired_statement, statement_count = _replace_postfix_nat_factorial(
        repaired_statement
    )
    repaired_proof, proof_count = _replace_postfix_nat_factorial(repaired_proof)
    if statement_count or proof_count:
        changes.append(
            {
                "from": "legacy_postfix_nat_factorial",
                "to": "Nat.factorial",
                "reason": "current_mathlib_syntax",
                "statement_occurrences": statement_count,
                "proof_occurrences": proof_count,
            }
        )

    has_real = "ℝ" in repaired_statement or "Real." in repaired_statement
    has_complex = "ℂ" in repaired_statement or "Complex." in repaired_statement
    analytic_namespace = None
    analytic_names: set[str] = set()
    if has_real and not has_complex:
        analytic_namespace = "Real"
        analytic_names = REAL_IDENTIFIERS
    elif has_complex and not has_real:
        analytic_namespace = "Complex"
        analytic_names = COMPLEX_IDENTIFIERS
    elif not has_complex:
        # LeanWorkbook's unqualified analytic vocabulary (sin/cos/exp/π) is
        # its legacy real-number notation. Qualifying it as Real also restores
        # the intended type of otherwise auto-implicit variables such as x.
        analytic_namespace = "Real"
        analytic_names = REAL_IDENTIFIERS

    replacements: list[tuple[str, str, str]] = []
    if analytic_namespace:
        replacements.extend(
            (name, f"{analytic_namespace}.{name}", "namespace_qualification")
            for name in analytic_names
        )
    replacements.extend(
        (name, f"Nat.{name}", "namespace_qualification") for name in NAT_IDENTIFIERS
    )
    replacements.extend(
        (old, new, "current_mathlib_name") for old, new in RENAMED_IDENTIFIERS.items()
    )
    if analytic_namespace == "Real":
        replacements.append(("π", "Real.pi", "namespace_qualification"))

    # Long names must be handled first so shorter independent names do not
    # affect them.
    for old, new, reason in sorted(replacements, key=lambda row: len(row[0]), reverse=True):
        repaired_statement, statement_count = _replace_identifier(repaired_statement, old, new)
        repaired_proof, proof_count = _replace_identifier(repaired_proof, old, new)
        if statement_count or proof_count:
            changes.append(
                {
                    "from": old,
                    "to": new,
                    "reason": reason,
                    "statement_occurrences": statement_count,
                    "proof_occurrences": proof_count,
                }
            )
    special_proof = SPECIAL_PROOF_REPAIRS.get(record_id)
    if special_proof and repaired_proof != special_proof:
        repaired_proof = special_proof
        changes.append(
            {
                "from": "original_proof",
                "to": "individually_reviewed_clean_proof",
                "reason": "remove_obsolete_or_unnecessary_helper_lemma",
                "statement_occurrences": 0,
                "proof_occurrences": 1,
            }
        )
    return repaired_statement, repaired_proof, changes


def _proof_state(diagnostics: str) -> str:
    blocks = [part.strip() for part in diagnostics.split("\n\n") if part.strip()]
    return blocks[-1] if blocks and ("goal" in blocks[-1].casefold() or "⊢" in blocks[-1]) else ""


def _classification(row: dict[str, Any]) -> tuple[str, str]:
    diagnostics = _text(row.get("error_message"))
    lowered = diagnostics.casefold()
    classified = classify_lean_diagnostics(
        diagnostics,
        timed_out="timeout" in lowered,
    )
    # Preserve the more actionable primary error when Pantograph also appends
    # a generic unsolved-goals block after an identifier/elaboration failure.
    if re.search(r"unknown\s+identifier", diagnostics, re.IGNORECASE):
        return LeanVerificationStatus.ELABORATION_ERROR.value, LeanFailureDetail.UNKNOWN_IDENTIFIER.value
    if re.search(r"unknown\s+constant", diagnostics, re.IGNORECASE):
        return LeanVerificationStatus.ELABORATION_ERROR.value, LeanFailureDetail.UNKNOWN_CONSTANT.value
    if re.search(r"invalid\s+field", diagnostics, re.IGNORECASE):
        return LeanVerificationStatus.ELABORATION_ERROR.value, LeanFailureDetail.INVALID_FIELD.value
    return classified.status.value, classified.detail.value


DETAIL_PRIORITY = {
    LeanFailureDetail.UNKNOWN_IDENTIFIER.value: 0,
    LeanFailureDetail.UNKNOWN_CONSTANT.value: 0,
    LeanFailureDetail.INVALID_FIELD.value: 1,
    LeanFailureDetail.UNSOLVED_GOALS.value: 2,
    LeanFailureDetail.TACTIC_EXECUTION_FAILED.value: 3,
    LeanFailureDetail.APPLICATION_TYPE_MISMATCH.value: 4,
    LeanFailureDetail.TYPE_MISMATCH.value: 4,
    LeanFailureDetail.FAILED_TO_SYNTHESIZE.value: 5,
    LeanFailureDetail.UNEXPECTED_TOKEN.value: 6,
    LeanFailureDetail.PARSER_ERROR.value: 6,
    LeanFailureDetail.UNCLASSIFIED_ELABORATION.value: 7,
}


def repair_rank(row: dict[str, Any], *, seed: int) -> tuple[Any, ...]:
    status, detail = _classification(row)
    raw_status = _text((row.get("metadata") or {}).get("raw_status"))
    steps = int((row.get("metadata") or {}).get("trajectory_steps") or 0)
    stable = hashlib.sha256(
        f"{seed}:{row.get('record_id')}:{row.get('record_hash')}".encode()
    ).hexdigest()
    return (
        0 if raw_status == "proved" else 1,
        DETAIL_PRIORITY.get(detail, 99),
        0 if status != LeanVerificationStatus.TIMEOUT.value else 1,
        steps,
        len(_text(row.get("proof"))),
        stable,
    )


def eligible_rows(rows: list[dict[str, Any]], *, count: int, seed: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    eligible: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        record_id = _text(row.get("record_id"))
        statement = _text(row.get("statement"))
        proof = _text(row.get("proof"))
        error = _text(row.get("error_message"))
        status, _ = _classification(row)
        reason = ""
        if not record_id or not statement or not proof or not error:
            reason = "missing_required_input"
        elif status in {
            LeanVerificationStatus.TIMEOUT.value,
            LeanVerificationStatus.ENVIRONMENT_ERROR.value,
            LeanVerificationStatus.INTERNAL_ERROR.value,
        }:
            reason = status
        fingerprint = canonical_sha256(
            {"record_id": record_id, "statement": statement, "proof": proof, "error": error}
        )
        if not reason and fingerprint in seen:
            reason = "duplicate_attempt"
        if reason:
            excluded[reason] += 1
            continue
        seen.add(fingerprint)
        copy = dict(row)
        copy["attempt_fingerprint"] = fingerprint
        eligible.append(copy)
    eligible.sort(key=lambda row: repair_rank(row, seed=seed))
    return (eligible[:count] if count > 0 else eligible), excluded


def build_request(row: dict[str, Any], environment: EnvironmentContract) -> RepairRequest:
    status, detail = _classification(row)
    attempt_id = _text(row["record_id"])
    error = _text(row["error_message"])
    proof = _text(row["proof"])
    statement_id = _text(row.get("statement_id")) or canonical_sha256(row["statement"])
    failure_class = f"{status}:{detail}"
    attempt = AggregatedAttempt(
        attempt_id=attempt_id,
        failure_proof=proof,
        error_message=error,
        failure_class=failure_class,
        proof_state=_proof_state(error),
        original_verification_status="failed",
    )
    return RepairRequest(
        source=SOURCE,
        theorem_name=_text(row.get("source_declaration")),
        lean_statement=_text(row["statement"]),
        target_attempt_id=attempt_id,
        target_failure_proof=proof,
        target_error_message=error,
        target_failure_class=failure_class,
        target_proof_state=_proof_state(error),
        statement_id=statement_id,
        aggregation_group_id=statement_id,
        aggregated_attempts=[attempt],
        attempt_fingerprint=_text(row["attempt_fingerprint"]),
        environment=environment,
        source_metadata={
            "record_id": attempt_id,
            "record_hash": row.get("record_hash"),
            "raw_status": (row.get("metadata") or {}).get("raw_status"),
            "trajectory_steps": (row.get("metadata") or {}).get("trajectory_steps"),
            "failure_status": status,
            "failure_detail": detail,
        },
    )


def api_call(request: RepairRequest, config: RepairClientConfig) -> dict[str, Any]:
    client = DeepSeekRepairClient(config)
    calls: list[APICallAttempt] = []
    for call_index in (1, 2):
        try:
            response = client.repair(request)
            calls.append(APICallAttempt(call_index=call_index, success=True, observation=response.observation))
            return {
                "target_attempt_id": request.target_attempt_id,
                "status": "success",
                "raw_response": response.raw_content,
                "final_observation": response.observation.model_dump(mode="json"),
                "api_call_attempts": [row.model_dump(mode="json") for row in calls],
            }
        except RepairClientError as error:
            calls.append(
                APICallAttempt(
                    call_index=call_index,
                    success=False,
                    error_kind=error.kind,
                    error_message=str(error),
                    observation=error.observation,
                )
            )
            if error.kind != "empty_response" or call_index == 2:
                return {
                    "target_attempt_id": request.target_attempt_id,
                    "status": "empty_response_after_retry" if error.kind == "empty_response" else "api_error",
                    "error_kind": error.kind,
                    "error_message": str(error),
                    "api_call_attempts": [row.model_dump(mode="json") for row in calls],
                }
            time.sleep(1.0)
    raise AssertionError("unreachable")


class CachedClient:
    def __init__(self, model_name: str, payload: dict[str, Any]) -> None:
        self.model_name = model_name
        self.payload = payload

    def repair(self, _: RepairRequest) -> RepairClientResponse:
        if self.payload.get("status") != "success":
            raise RepairClientError(
                _text(self.payload.get("error_message")) or "cached API call failed",
                kind=_text(self.payload.get("error_kind")) or "api_error",
            )
        return RepairClientResponse(
            raw_content=_text(self.payload["raw_response"]),
            observation=self.payload["final_observation"],
        )


def prepare(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.fail_file)
    environment = EnvironmentContract.model_validate(
        json.loads(args.environment_contract.read_text(encoding="utf-8"))
    )
    selected, excluded = eligible_rows(rows, count=args.count, seed=args.seed)
    requests = [build_request(row, environment) for row in selected]
    write_jsonl(args.output / "requests.jsonl", (row.model_dump(mode="json") for row in requests))
    report = {
        "schema_version": "leanworkbook_repair_preflight_v1",
        "input_fail_rows": len(rows),
        "selected_attempts": len(requests),
        "excluded": dict(excluded),
        "failure_statuses": dict(Counter(row.target_failure_class.split(":", 1)[0] for row in requests)),
        "failure_details": dict(Counter(row.target_failure_class.split(":", 1)[1] for row in requests)),
        "raw_statuses": dict(Counter(str(row.source_metadata.get("raw_status")) for row in requests)),
        "request_sha256": sha256_file(args.output / "requests.jsonl"),
        "environment_hash": environment.environment_hash,
        "api_called": False,
        "pantograph_started": False,
    }
    _write_json(args.output / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def restore_disproved_statement(statement: str) -> tuple[str, bool]:
    """Restore the negated theorem proved by a LeanWorkbook disproval trace."""

    normalized, _ = normalize_missing_goal_colon(statement)
    _, header, proposition = split_statement(normalized)
    if proposition.lstrip().startswith(("¬", "Not ")):
        return normalized, False
    return f"{header} : ¬ ({proposition})", True


def normalize_missing_goal_colon(statement: str) -> tuple[str, bool]:
    """Repair legacy rows where the colon after the declaration name is absent."""

    value = _text(statement)
    declaration = re.search(r"(?m)^(\s*(?:theorem|lemma)\s+[^\s(:{]+)", value)
    if declaration is None:
        raise ValueError("statement has no theorem/lemma declaration")
    tail = value[declaration.end() :].lstrip()
    # A valid declaration can continue with explicit/implicit binders before
    # its top-level goal colon.  Any other leading token is already the
    # proposition.  Checking this before ``split_statement`` is essential for
    # rows such as ``theorem t forall x : Nat, ...``: the binder type colon is
    # not the missing declaration colon.
    if tail and not tail.startswith((":", "(", "{", "[")):
        proposition = tail
        return f"{value[:declaration.end()].rstrip()} : {proposition}", True
    try:
        split_statement(value)
        return value, False
    except ValueError:
        proposition = value[declaration.end() :].strip()
        if not proposition:
            raise ValueError("declaration has no proposition after its name")
        return f"{value[:declaration.end()].rstrip()} : {proposition}", True


def _statement_from_initial_state(record_id: str, initial_state: str) -> str | None:
    """Rebuild a truncated declaration from Lean's frozen initial goal state."""

    state = _text(initial_state)
    if "\u22a2" not in state:
        return None
    context, goal = state.split("\u22a2", 1)
    goal = goal.strip()
    if not goal:
        return None
    blocks: list[str] = []
    for line in context.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line[:1].isspace() and blocks:
            blocks[-1] += " " + stripped
        else:
            blocks.append(stripped)
    if any(":" not in block for block in blocks):
        return None
    binders = " ".join(f"({block})" for block in blocks)
    spacing = f" {binders}" if binders else ""
    return f"theorem {record_id}{spacing} : {goal}"


def _targeted_proof_variants(primary: str) -> list[dict[str, str]]:
    """Produce small proof edits for explicit rewrite/redundant-tactic failures."""

    lines = _text(primary).splitlines()
    if not lines or lines[0].strip() != "by":
        return []
    variants: list[dict[str, str]] = []
    seen = {_text(primary)}

    def add(strategy: str, candidate_lines: list[str]) -> None:
        proof = "\n".join(candidate_lines).strip()
        if proof and proof != "by" and proof not in seen:
            seen.add(proof)
            variants.append({"strategy": strategy, "proof": proof})

    # In many migrated proofs ``field_simp`` has already removed every
    # division, so a following div_* rewrite is both redundant and the exact
    # reported failure.  Preserve the remaining tactic chain when removing it.
    for index, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if not re.search(r"\b(?:rw|rewrite)\b", stripped):
            continue
        chained = re.sub(r"\s*<;>\s*(?:rw|rewrite)\s*\[[^\]]*\]", "", line)
        if chained != line:
            changed = list(lines)
            changed[index] = chained
            add(f"remove_redundant_chained_rewrite_line_{index}", changed)
        match = re.match(
            r"^(\s*)((?:all_goals\s+)?)(?:rw|rewrite)\s*\[[^\]]*\]\s*<;>\s*(.+)$",
            line,
        )
        if match:
            changed = list(lines)
            changed[index] = f"{match.group(1)}{match.group(2)}{match.group(3)}"
            add(f"keep_post_rewrite_tactic_line_{index}", changed)
        if re.match(r"^\s*(?:all_goals\s+)?(?:rw|rewrite)\b", line):
            changed = lines[:index] + lines[index + 1 :]
            add(f"remove_redundant_rewrite_line_{index}", changed)
            simp_line = re.sub(r"\b(?:rw|rewrite)\b", "simp only", line, count=1)
            changed = list(lines)
            changed[index] = simp_line
            add(f"rewrite_to_simp_only_line_{index}", changed)

    # A common current-Lean diagnostic is ``No goals to be solved`` after an
    # earlier tactic became stronger.  Testing short suffix removals is a
    # minimal edit and cannot be accepted unless the prefix already closes.
    for count in (1, 2):
        if len(lines) > count + 1:
            add(f"drop_redundant_trailing_tactics_{count}", lines[:-count])

    simple_tactic = re.compile(
        r"^(?:all_goals\s+)?(?:aesop|assumption|exact|field_simp|grind|linarith|"
        r"nlinarith|norm_num|omega|positivity|ring|ring_nf|simp|simpa|rw|rewrite)\b"
    )
    for index, line in enumerate(lines[1:], start=1):
        if not simple_tactic.match(line.strip()):
            continue
        if index > 1 and lines[index - 1].rstrip().endswith(("=>", ":= by")):
            continue
        add(f"remove_explicit_tactic_line_{index}", lines[:index] + lines[index + 1 :])
        if len(variants) >= 16:
            break
    return variants[:16]


def _targeted_proof_variants(primary: str) -> list[dict[str, str]]:
    """Produce small proof edits for explicit rewrite/redundant-tactic failures."""

    lines = _text(primary).splitlines()
    if not lines or lines[0].strip() != "by":
        return []
    variants: list[dict[str, str]] = []
    seen = {_text(primary)}

    def add(strategy: str, candidate_lines: list[str]) -> None:
        proof = "\n".join(candidate_lines).strip()
        if proof and proof != "by" and proof not in seen:
            seen.add(proof)
            variants.append({"strategy": strategy, "proof": proof})

    # In many migrated proofs ``field_simp`` has already removed every
    # division, so a following div_* rewrite is both redundant and the exact
    # reported failure.  Preserve the remaining tactic chain when removing it.
    for index, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if not re.search(r"\b(?:rw|rewrite)\b", stripped):
            continue
        chained = re.sub(r"\s*<;>\s*(?:rw|rewrite)\s*\[[^\]]*\]", "", line)
        if chained != line:
            changed = list(lines)
            changed[index] = chained
            add(f"remove_redundant_chained_rewrite_line_{index}", changed)
        match = re.match(
            r"^(\s*)((?:all_goals\s+)?)(?:rw|rewrite)\s*\[[^\]]*\]\s*<;>\s*(.+)$",
            line,
        )
        if match:
            changed = list(lines)
            changed[index] = f"{match.group(1)}{match.group(2)}{match.group(3)}"
            add(f"keep_post_rewrite_tactic_line_{index}", changed)
        if re.match(r"^\s*(?:all_goals\s+)?(?:rw|rewrite)\b", line):
            changed = lines[:index] + lines[index + 1 :]
            add(f"remove_redundant_rewrite_line_{index}", changed)
            simp_line = re.sub(r"\b(?:rw|rewrite)\b", "simp only", line, count=1)
            changed = list(lines)
            changed[index] = simp_line
            add(f"rewrite_to_simp_only_line_{index}", changed)

    # A common current-Lean diagnostic is ``No goals to be solved`` after an
    # earlier tactic became stronger.  Testing short suffix removals is a
    # minimal edit and cannot be accepted unless the prefix already closes.
    for count in (1, 2):
        if len(lines) > count + 1:
            add(f"drop_redundant_trailing_tactics_{count}", lines[:-count])

    simple_tactic = re.compile(
        r"^(?:all_goals\s+)?(?:aesop|assumption|exact|field_simp|grind|linarith|"
        r"nlinarith|norm_num|omega|positivity|ring|ring_nf|simp|simpa|rw|rewrite)\b"
    )
    for index, line in enumerate(lines[1:], start=1):
        if not simple_tactic.match(line.strip()):
            continue
        if index > 1 and lines[index - 1].rstrip().endswith(("=>", ":= by")):
            continue
        add(f"remove_explicit_tactic_line_{index}", lines[:index] + lines[index + 1 :])
        if len(variants) >= 16:
            break
    return variants[:16]


def _proof_portfolio(primary: str, statement: str, raw_status: str) -> list[dict[str, str]]:
    """Return an ordered local proof portfolio; Pantograph accepts the winner."""

    variants: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(strategy: str, proof: str) -> None:
        normalized = _text(proof)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append({"strategy": strategy, "proof": normalized})

    combinatorial = any(
        token in statement for token in ("\u2211", "\u220f", "Nat.factorial", "Finset.")
    )
    try:
        _, declaration_header, _ = split_statement(statement)
        closed_declaration = not any(
            token in declaration_header for token in ("(", "{", "[")
        )
    except ValueError:
        closed_declaration = False

    # Large closed finite computations are dramatically cheaper with
    # native_decide than by expanding sums/factorials through ring or norm_num.
    if combinatorial and closed_declaration:
        add("clean_native_decide", "by\n  native_decide")
    add("manual_compatibility_or_reviewed", primary)
    if "push_neg" in primary:
        add("replace_deprecated_push_neg", primary.replace("push_neg", "push Not"))
    for targeted in _targeted_proof_variants(primary):
        add(targeted["strategy"], targeted["proof"])
    for targeted in _targeted_proof_variants(primary):
        add(targeted["strategy"], targeted["proof"])

    # These are clean from-scratch proof candidates, never tactic suffixes
    # appended to a known-bad attempt.  They are intentionally small and
    # deterministic; local Pantograph is the sole acceptance gate.
    add("clean_norm_num", "by\n  norm_num")
    add("clean_omega", "by\n  omega")
    if not combinatorial:
        add("clean_ring", "by\n  ring")
        add("clean_ring_nf", "by\n  ring_nf")
    if not combinatorial and any(token in statement for token in ("≤", "≥", "<", ">")):
        add("clean_linarith", "by\n  linarith")
        add("clean_nlinarith", "by\n  nlinarith")
        add("clean_positivity", "by\n  positivity")
    add("clean_simp_all", "by\n  simp_all")
    add("clean_aesop", "by\n  aesop")
    add("clean_grind", "by\n  grind")
    return variants


def manual_all_candidates(args: argparse.Namespace) -> None:
    """Create one full-coverage repair record for every fail_2 row."""

    rows = read_jsonl(args.fail_file)
    environment = EnvironmentContract.model_validate(
        json.loads(args.environment_contract.read_text(encoding="utf-8"))
    )
    candidates: list[dict[str, Any]] = []
    replacement_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    negation_restored = 0
    for row in rows:
        record_id = _text(row.get("record_id"))
        raw_status = _text((row.get("metadata") or {}).get("raw_status"))
        original_statement = _text(row.get("statement"))
        original_proof = _text(row.get("proof"))
        changes: list[dict[str, Any]] = []
        original_error = _text(row.get("error_message"))
        structurally_truncated = bool(
            re.search(
                r"unexpected token ['\"](?:\)|:)['\"]; expected ':='",
                original_error,
                re.IGNORECASE,
            )
        )
        rebuilt_statement = (
            _statement_from_initial_state(
                record_id,
                _text((row.get("metadata") or {}).get("initial_state")),
            )
            if structurally_truncated
            else None
        )
        if rebuilt_statement:
            repaired_statement = rebuilt_statement
            goal_colon_restored = False
            changes.append(
                {
                    "from": "truncated_legacy_declaration",
                    "to": "declaration_rebuilt_from_frozen_initial_state",
                    "reason": "restore_verified_binders_and_goal",
                    "statement_occurrences": 1,
                    "proof_occurrences": 0,
                }
            )
        else:
            repaired_statement, goal_colon_restored = normalize_missing_goal_colon(
                original_statement
            )
        if goal_colon_restored:
            changes.append(
                {
                    "from": "declaration_missing_top_level_goal_colon",
                    "to": "declaration_with_explicit_goal_colon",
                    "reason": "restore_legacy_statement_syntax",
                    "statement_occurrences": 1,
                    "proof_occurrences": 0,
                }
            )
        if raw_status == "disproved":
            try:
                repaired_statement, changed = restore_disproved_statement(repaired_statement)
            except ValueError as error:
                raise ValueError(
                    f"{record_id}: cannot restore disproved statement: {error}; "
                    f"statement={original_statement!r}"
                ) from error
            if changed:
                negation_restored += 1
                changes.append(
                    {
                        "from": "positive_statement_saved_by_legacy_reconstruction",
                        "to": "negated_statement_proved_by_disproval_trace",
                        "reason": "restore_raw_disproved_target",
                        "statement_occurrences": 1,
                        "proof_occurrences": 0,
                    }
                )
        repaired_statement, repaired_proof, compatibility_changes = manual_compatibility_repair(
            repaired_statement,
            original_proof,
            record_id=record_id,
        )
        changes.extend(compatibility_changes)
        if not changes:
            changes.append(
                {
                    "from": "original_attempt",
                    "to": "full_coverage_local_reverification",
                    "reason": "no_safe_source_edit; verify_original_then_clean_portfolio",
                    "statement_occurrences": 0,
                    "proof_occurrences": 0,
                }
            )
        for change in changes:
            replacement_counts[f"{change['from']} -> {change['to']}"] += (
                int(change["statement_occurrences"]) + int(change["proof_occurrences"])
            )
        proof_candidates = _proof_portfolio(repaired_proof, repaired_statement, raw_status)
        status, detail = _classification(row)
        status_counts[f"{status}:{detail}"] += 1
        payload: dict[str, Any] = {
            "schema_version": "leanworkbook_manual_full_repair_candidate_v1",
            "manual_repair_version": MANUAL_REPAIR_VERSION,
            "candidate_id": f"{record_id}:full_repair",
            "target_attempt_id": record_id,
            "source": SOURCE,
            "source_record_hash": row.get("record_hash"),
            "statement_id": row.get("statement_id"),
            "raw_status": raw_status,
            "original_statement": original_statement,
            "repaired_statement": repaired_statement,
            "original_proof": original_proof,
            "repaired_proof": repaired_proof,
            "proof_candidates": proof_candidates,
            "statement_changed": repaired_statement != original_statement,
            "proof_changed": repaired_proof != original_proof,
            "changes": changes,
            "original_error_message": _text(row.get("error_message")),
            "failure_class": f"{status}:{detail}",
            "environment": environment.model_dump(mode="json"),
            "external_api_called": False,
        }
        payload["candidate_hash"] = canonical_sha256(payload)
        candidates.append(payload)

    write_jsonl(args.output / "manual_candidates.jsonl", candidates)
    write_jsonl(args.output / "manual_skipped.jsonl", ())
    report = {
        "schema_version": "leanworkbook_manual_full_candidate_report_v1",
        "manual_repair_version": MANUAL_REPAIR_VERSION,
        "input_fail_2_rows": len(rows),
        "candidates": len(candidates),
        "record_id_unique": len({row["target_attempt_id"] for row in candidates}),
        "raw_status": dict(Counter(row["raw_status"] for row in candidates)),
        "disproved_statement_negation_restored": negation_restored,
        "statement_changed": sum(bool(row["statement_changed"]) for row in candidates),
        "proof_changed": sum(bool(row["proof_changed"]) for row in candidates),
        "total_proof_variants": sum(len(row["proof_candidates"]) for row in candidates),
        "failure_class": dict(status_counts),
        "replacement_occurrences": dict(replacement_counts.most_common()),
        "candidate_manifest_sha256": sha256_file(args.output / "manual_candidates.jsonl"),
        "external_api_called": False,
        "pantograph_started": False,
        "full_coverage": len(candidates) == len(rows),
    }
    _write_json(args.output / "manual_candidate_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def manual_candidates(args: argparse.Namespace) -> None:
    """Create local Codex-curated candidates without any network/API call."""

    requests = [RepairRequest.model_validate(row) for row in read_jsonl(args.output / "requests.jsonl")]
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    replacement_counts: Counter[str] = Counter()
    for request in requests:
        statement, proof, changes = manual_compatibility_repair(
            request.lean_statement,
            request.target_failure_proof,
            record_id=request.target_attempt_id,
        )
        if not changes:
            skipped.append(
                {
                    "target_attempt_id": request.target_attempt_id,
                    "reason": "no_high_confidence_manual_rule",
                    "failure_class": request.target_failure_class,
                }
            )
            continue
        for change in changes:
            replacement_counts[f"{change['from']} -> {change['to']}"] += (
                int(change["statement_occurrences"]) + int(change["proof_occurrences"])
            )
        payload: dict[str, Any] = {
            "schema_version": "leanworkbook_manual_repair_candidate_v1",
            "manual_repair_version": MANUAL_REPAIR_VERSION,
            "candidate_id": f"{request.target_attempt_id}:compatibility",
            "target_attempt_id": request.target_attempt_id,
            "source": SOURCE,
            "source_record_hash": request.source_metadata.get("record_hash"),
            "statement_id": request.statement_id,
            "original_statement": request.lean_statement,
            "repaired_statement": statement,
            "original_proof": request.target_failure_proof,
            "repaired_proof": proof,
            "statement_changed": statement != request.lean_statement,
            "proof_changed": proof != request.target_failure_proof,
            "changes": changes,
            "original_error_message": request.target_error_message,
            "failure_class": request.target_failure_class,
            "environment": request.environment.model_dump(mode="json"),
            "external_api_called": False,
        }
        payload["candidate_hash"] = canonical_sha256(payload)
        candidates.append(payload)
    write_jsonl(args.output / "manual_candidates.jsonl", candidates)
    write_jsonl(args.output / "manual_skipped.jsonl", skipped)
    report = {
        "schema_version": "leanworkbook_manual_candidate_report_v1",
        "manual_repair_version": MANUAL_REPAIR_VERSION,
        "frozen_requests": len(requests),
        "candidates": len(candidates),
        "skipped_without_high_confidence_rule": len(skipped),
        "statement_changed": sum(bool(row["statement_changed"]) for row in candidates),
        "proof_changed": sum(bool(row["proof_changed"]) for row in candidates),
        "replacement_occurrences": dict(replacement_counts.most_common()),
        "candidate_manifest_sha256": sha256_file(args.output / "manual_candidates.jsonl"),
        "external_api_called": False,
        "pantograph_started": False,
    }
    _write_json(args.output / "manual_candidate_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _manual_source(candidate: dict[str, Any], proof: str | None = None) -> str:
    statement = _text(candidate["repaired_statement"])
    selected_proof = _text(proof if proof is not None else candidate["repaired_proof"])
    if ":=" in statement:
        raise ValueError(
            f"statement for {candidate['candidate_id']} already contains ':='; "
            "manual repair refuses ambiguous assembly"
        )
    # ``PantographTheoremVerifier`` starts its persistent server with Mathlib
    # already imported.  Supplying another top-level import to ``check_source``
    # after warmup is rejected because imports are only legal in the file
    # header.  The evaluation verifier follows the same contract and sends
    # declaration-only source to the initialized server.
    return f"{statement} := {selected_proof}\n"


def _manual_repair_description(candidate: dict[str, Any], strategy: str) -> str:
    edits = ", ".join(
        f"{change['from']}->{change['to']}"
        f"(statement={change['statement_occurrences']},proof={change['proof_occurrences']})"
        for change in candidate["changes"]
    )
    return (
        f"method={MANUAL_REPAIR_VERSION}; external_api_called=false; "
        f"candidate_id={candidate['candidate_id']}; selected_strategy={strategy}; edits=[{edits}]"
    )


def _manual_proof_variants(candidate: dict[str, Any]) -> list[dict[str, str]]:
    raw = candidate.get("proof_candidates") or [
        {"strategy": "manual_compatibility_or_reviewed", "proof": candidate["repaired_proof"]}
    ]
    variants: list[dict[str, str]] = []
    for item in raw:
        strategy = _text(item.get("strategy")) or "unnamed"
        proof = _text(item.get("proof"))
        variant_hash = canonical_sha256(
            {
                "candidate_hash": candidate["candidate_hash"],
                "strategy": strategy,
                "proof": proof,
            }
        )
        variants.append({"strategy": strategy, "proof": proof, "variant_hash": variant_hash})
    return variants


def _manual_result_requires_retry(result: dict[str, Any]) -> bool:
    """Reject cached transport/runtime failures as proof-verification evidence."""

    diagnostics = _text(result.get("diagnostics")).casefold()
    return bool(
        result.get("timed_out")
        or result.get("error_type") == "pantograph_error"
        or any(
            marker in diagnostics
            for marker in (
                "server not running",
                "server closed",
                "connection reset",
                "connection refused",
                "broken pipe",
                "transport endpoint",
                "timeout limit",
            )
        )
    )


def _start_manual_verifier(args: argparse.Namespace) -> PantographTheoremVerifier:
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


def manual_finalize(args: argparse.Namespace) -> None:
    """Pantograph-check every fail_2 row and atomically migrate verified rows."""

    candidates = read_jsonl(args.output / "manual_candidates.jsonl")
    if not candidates:
        raise RuntimeError("manual candidate manifest is empty; run manual-all-candidates first")
    if len({str(row["target_attempt_id"]) for row in candidates}) != len(candidates):
        raise RuntimeError("manual full candidate manifest has duplicate target_attempt_id values")
    result_path = args.output / "manual_attempt_results.jsonl"
    result_rows = read_jsonl(result_path)
    results = {
        str(row["variant_hash"]): row
        for row in result_rows
        if row.get("variant_hash")
        and row.get("manual_repair_version") == MANUAL_REPAIR_VERSION
        and not _manual_result_requires_retry(row)
    }
    selected_by_id: dict[str, dict[str, Any]] = {}
    verifier: PantographTheoremVerifier | None = None
    new_attempts = 0
    pantograph_restarts = 0
    try:
        for index, candidate in enumerate(candidates, start=1):
            target_id = str(candidate["target_attempt_id"])
            chosen: dict[str, Any] | None = None
            attempted_for_record = 0
            for variant in _manual_proof_variants(candidate):
                attempted_for_record += 1
                result = results.get(variant["variant_hash"])
                if result is None:
                    if verifier is None:
                        verifier = _start_manual_verifier(args)
                    source = _manual_source(candidate, variant["proof"])
                    checked = verifier.check_source(source, timeout=args.timeout)
                    restart_count = 0
                    if checked.error_type == "pantograph_error" or checked.timed_out:
                        verifier.close()
                        verifier = _start_manual_verifier(args)
                        pantograph_restarts += 1
                        restart_count = 1
                        checked = verifier.check_source(source, timeout=args.timeout)
                        if checked.error_type == "pantograph_error" or checked.timed_out:
                            # A second runtime failure is recorded for this
                            # variant, but the next variant must receive a new
                            # live server rather than inheriting a dead one.
                            verifier.close()
                            verifier = None
                    result = {
                        "schema_version": "leanworkbook_manual_full_repair_result_v1",
                        "manual_repair_version": MANUAL_REPAIR_VERSION,
                        "candidate_id": candidate["candidate_id"],
                        "candidate_hash": candidate["candidate_hash"],
                        "variant_hash": variant["variant_hash"],
                        "strategy": variant["strategy"],
                        "selected_proof": variant["proof"],
                        "target_attempt_id": target_id,
                        "pantograph_verified": SUCCESS if checked.success else FAIL,
                        "success": checked.success,
                        "diagnostics": checked.diagnostics,
                        "errors": list(checked.errors),
                        "warnings": list(checked.warnings),
                        "timed_out": checked.timed_out,
                        "error_type": checked.error_type,
                        "verification_seconds": checked.check_seconds,
                        "assembled_source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                        "environment_hash": candidate["environment"]["environment_hash"],
                        "external_api_called": False,
                        "pantograph_restart_count": restart_count,
                    }
                    append_jsonl(result_path, result)
                    results[variant["variant_hash"]] = result
                    new_attempts += 1
                chosen = {
                    "candidate": candidate,
                    "result": result,
                    "strategy": variant["strategy"],
                    "proof": variant["proof"],
                    "attempted_for_record": attempted_for_record,
                }
                if result["success"]:
                    break
            if chosen is None:
                raise RuntimeError(f"candidate {candidate['candidate_id']} has no proof variants")
            selected_by_id[target_id] = chosen
            if index % 25 == 0:
                solved = sum(bool(row["result"]["success"]) for row in selected_by_id.values())
                print(
                    json.dumps(
                        {
                            "records_completed": index,
                            "records_total": len(candidates),
                            "records_solved": solved,
                            "new_variant_attempts": new_attempts,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        if verifier is not None:
            verifier.close()

    verified_candidates: list[dict[str, Any]] = []
    failed_candidates: list[dict[str, Any]] = []
    for target_id, selection in selected_by_id.items():
        materialized = dict(selection["candidate"])
        materialized.update(
            {
                "selected_strategy": selection["strategy"],
                "selected_verified_proof": selection["proof"] if selection["result"]["success"] else None,
                "selected_result_variant_hash": selection["result"]["variant_hash"],
                "proof_variants_attempted": selection["attempted_for_record"],
            }
        )
        if selection["result"]["success"]:
            verified_candidates.append(materialized)
        else:
            failed_candidates.append(materialized)
    write_jsonl(args.output / "manual_repair_verified.jsonl", verified_candidates)
    write_jsonl(args.output / "manual_repair_failed.jsonl", failed_candidates)

    by_id = {str(row["target_attempt_id"]): row for row in candidates}
    result_by_id = {
        target_id: selection["result"] for target_id, selection in selected_by_id.items()
    }

    fail_snapshot = args.output / "leanworkbook_verified_fail_2.before_manual_repair.jsonl"
    success_snapshot = args.output / "leanworkbook_verified_success_2.before_manual_repair.jsonl"
    if not fail_snapshot.exists():
        shutil.copy2(args.fail_file, fail_snapshot)
    if not success_snapshot.exists():
        shutil.copy2(args.success_file, success_snapshot)
    original_fail = read_jsonl(fail_snapshot)
    original_success = read_jsonl(success_snapshot)
    promoted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    attempted_ids = set(by_id)
    for row in original_fail:
        record_id = _text(row.get("record_id"))
        candidate = by_id.get(record_id)
        result = result_by_id.get(record_id)
        if candidate is None or result is None:
            remaining.append(row)
            continue
        selection = selected_by_id[record_id]
        description = _manual_repair_description(candidate, selection["strategy"])
        if result["success"]:
            updated = dict(row)
            updated["statement"] = candidate["repaired_statement"]
            updated["proof"] = selection["proof"]
            updated["repair"] = description
            updated.update(
                {
                    "statement_verified": True,
                    "proof_verified": True,
                    "reference_proof_verified": True,
                    "verification_status": "success",
                    "verification_error_type": None,
                    "verification_error_message": None,
                    "assembled_source_hash": result["assembled_source_hash"],
                }
            )
            updated.pop("error_message", None)
            updated.pop("repair_fail", None)
            promoted.append(add_dataset_contract(updated, source=SOURCE, status=SUCCESS))
        else:
            updated = dict(row)
            updated["repair"] = description
            updated["repair_fail"] = (
                f"proof_variants_attempted={selection['attempted_for_record']}\n"
                + (_text(result.get("diagnostics"))
                or "\n".join(str(value) for value in result.get("errors") or ())
                or str(result.get("error_type") or "Pantograph verification failed"))
            )
            remaining.append(
                add_dataset_contract(
                    updated,
                    source=SOURCE,
                    status=FAIL,
                    error_message=_text(row.get("error_message")),
                )
            )

    promoted_ids = {_text(row.get("record_id")) for row in promoted}
    success = [row for row in original_success if _text(row.get("record_id")) not in promoted_ids]
    success.extend(promoted)
    write_jsonl(args.success_file, success)
    write_jsonl(args.fail_file, remaining)
    report = {
        "schema_version": "leanworkbook_manual_repair_final_report_v1",
        "manual_repair_version": MANUAL_REPAIR_VERSION,
        "input_candidates": len(candidates),
        "attempted": len(attempted_ids),
        "promoted_to_success_2": len(promoted),
        "repair_failed": len(failed_candidates),
        "records_with_statement_change": sum(bool(row["statement_changed"]) for row in candidates),
        "disproved_statement_negation_restored": sum(
            any(change["reason"] == "restore_raw_disproved_target" for change in row["changes"])
            for row in candidates
        ),
        "selected_strategy": dict(
            Counter(selection["strategy"] for selection in selected_by_id.values())
        ),
        "new_variant_attempts": new_attempts,
        "total_cached_or_new_variant_results": len(results),
        "pantograph_restarts": pantograph_restarts,
        "success_2_before": len(original_success),
        "success_2_after": len(success),
        "fail_2_before": len(original_fail),
        "fail_2_after": len(remaining),
        "success_sha256": sha256_file(args.success_file),
        "fail_sha256": sha256_file(args.fail_file),
        "candidate_manifest_sha256": sha256_file(args.output / "manual_candidates.jsonl"),
        "result_manifest_sha256": sha256_file(result_path),
        "all_promoted_pantograph_verified": all(
            result_by_id[record_id]["success"] for record_id in promoted_ids
        ),
        "full_fail_2_coverage": len(attempted_ids) == len(original_fail),
        "external_api_called": False,
    }
    _write_json(args.output / "manual_final_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def generate(args: argparse.Namespace) -> None:
    requests = [RepairRequest.model_validate(row) for row in read_jsonl(args.output / "requests.jsonl")]
    response_path = args.output / "api_responses.jsonl"
    existing = {str(row["target_attempt_id"]): row for row in read_jsonl(response_path)}
    pending = [row for row in requests if row.target_attempt_id not in existing]
    config = RepairClientConfig.from_env()
    with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
        futures = {pool.submit(api_call, request, config): request for request in pending}
        for future in as_completed(futures):
            payload = future.result()
            append_jsonl(response_path, payload)
            existing[str(payload["target_attempt_id"])] = payload
            if len(existing) % 25 == 0:
                print(json.dumps({"api_completed": len(existing), "api_total": len(requests)}), flush=True)
    report = {
        "requests": len(requests),
        "responses": len(existing),
        "status": dict(Counter(str(row.get("status")) for row in existing.values())),
        "response_manifest_sha256": sha256_file(response_path),
        "pantograph_started": False,
    }
    _write_json(args.output / "api_generation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _repair_description(record: RepairData) -> str:
    changes = ""
    if record.raw_api_response:
        try:
            changes = _text(json.loads(record.raw_api_response).get("minimal_repair_change_summary"))
        except (json.JSONDecodeError, AttributeError):
            pass
    return (
        f"assessment={record.resolved_attempt_assessment}; "
        f"minimal_change={changes or 'unavailable'}; "
        f"selected_strategy={record.selected_strategy}; anomaly={record.anomaly_status}"
    )


def _repair_failure(record: RepairData) -> str:
    diagnostics: list[str] = []
    for label, candidate in (("minimal_repair", record.minimal_repair), ("clean_repair", record.clean_repair)):
        if candidate and not candidate.verification.verified:
            message = candidate.verification.diagnostics or "\n".join(candidate.verification.errors)
            diagnostics.append(f"{label}: {message.strip() or candidate.verification.status}")
    return "\n\n".join(diagnostics) or record.failure_reason or record.anomaly_status


def finalize(args: argparse.Namespace) -> None:
    requests = [RepairRequest.model_validate(row) for row in read_jsonl(args.output / "requests.jsonl")]
    responses = {str(row["target_attempt_id"]): row for row in read_jsonl(args.output / "api_responses.jsonl")}
    if any(row.target_attempt_id not in responses for row in requests):
        raise RuntimeError("API response manifest is incomplete")
    config = RepairClientConfig.from_env()
    result_path = args.output / "attempt_results.jsonl"
    results = {str(row["target_attempt_id"]): row for row in read_jsonl(result_path)}
    pending = [row for row in requests if row.target_attempt_id not in results]
    if pending:
        verifier = PantographTheoremVerifier(
            args.lean_project, imports=("Mathlib",), timeout=args.timeout, startup_timeout=900
        )
        warmup = verifier.warmup(timeout=180)
        if not warmup.success:
            verifier.close()
            raise RuntimeError(f"Pantograph warmup failed: {warmup.diagnostics}")
        try:
            for index, request in enumerate(pending, start=1):
                payload = responses[request.target_attempt_id]
                calls = [APICallAttempt.model_validate(row) for row in payload.get("api_call_attempts", [])]
                result = RepairPipeline(
                    client=CachedClient(config.model, payload), verifier=verifier, timeout=args.timeout
                ).run_one(request, api_call_attempts=calls)
                serialized = result.data.model_dump(mode="json")
                append_jsonl(result_path, serialized)
                results[request.target_attempt_id] = serialized
                if index % 25 == 0:
                    print(json.dumps({"verified_attempts": index, "pending_total": len(pending)}), flush=True)
        finally:
            verifier.close()

    records = [RepairData.model_validate(results[row.target_attempt_id]) for row in requests]
    by_id = {row.target_attempt_id: row for row in records}
    write_jsonl(args.output / "repair_verified.jsonl", (row.model_dump(mode="json") for row in records if row.repair_verified))
    write_jsonl(args.output / "repair_failed.jsonl", (row.model_dump(mode="json") for row in records if not row.repair_verified))

    args.output.mkdir(parents=True, exist_ok=True)
    fail_snapshot = args.output / "leanworkbook_verified_fail_2.before_repair.jsonl"
    success_snapshot = args.output / "leanworkbook_verified_success_2.before_repair.jsonl"
    if not fail_snapshot.exists():
        shutil.copy2(args.fail_file, fail_snapshot)
    if not success_snapshot.exists():
        shutil.copy2(args.success_file, success_snapshot)
    original_fail = read_jsonl(fail_snapshot)
    original_success = read_jsonl(success_snapshot)
    promoted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for row in original_fail:
        record_id = _text(row.get("record_id"))
        repair = by_id.get(record_id)
        if repair is None:
            remaining.append(row)
            continue
        if repair.repair_verified:
            updated = dict(row)
            updated["proof"] = repair.selected_verified_proof
            updated.update(
                {
                    "statement_verified": True,
                    "proof_verified": True,
                    "reference_proof_verified": True,
                    "verification_status": "success",
                    "verification_error_type": None,
                    "verification_error_message": None,
                    "assembled_source_hash": (
                        repair.minimal_repair.verification.assembled_source_sha256
                        if repair.selected_strategy == "minimal_repair"
                        else repair.clean_repair.verification.assembled_source_sha256
                    ),
                }
            )
            updated.pop("error_message", None)
            promoted.append(add_dataset_contract(updated, source=SOURCE, status=SUCCESS))
        else:
            updated = dict(row)
            updated["repair"] = _repair_description(repair)
            updated["repair_fail"] = _repair_failure(repair)
            remaining.append(
                add_dataset_contract(
                    updated,
                    source=SOURCE,
                    status=FAIL,
                    error_message=_text(row.get("error_message")),
                )
            )
    promoted_ids = {_text(row.get("record_id")) for row in promoted}
    success = [row for row in original_success if _text(row.get("record_id")) not in promoted_ids]
    success.extend(promoted)
    write_jsonl(args.success_file, success)
    write_jsonl(args.fail_file, remaining)
    report = {
        "schema_version": "leanworkbook_failure_repair_report_v1",
        "attempted": len(records),
        "promoted_to_success_2": len(promoted),
        "repair_failed": len(records) - len(promoted),
        "success_2_before": len(original_success),
        "success_2_after": len(success),
        "fail_2_before": len(original_fail),
        "fail_2_after": len(remaining),
        "selected_strategy": dict(Counter(row.selected_strategy for row in records)),
        "anomaly_status": dict(Counter(row.anomaly_status for row in records)),
        "success_sha256": sha256_file(args.success_file),
        "fail_sha256": sha256_file(args.fail_file),
        "all_promoted_pantograph_verified": all(row.repair_verified for row in records if row.target_attempt_id in promoted_ids),
    }
    _write_json(args.output / "final_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command",
        choices=(
            "prepare",
            "manual-candidates",
            "manual-all-candidates",
            "manual-finalize",
            "generate",
            "finalize",
        ),
    )
    value.add_argument("--fail-file", type=Path, default=root / "verified_data" / "leanworkbook_verified_fail_2.jsonl")
    value.add_argument("--success-file", type=Path, default=root / "verified_data" / "leanworkbook_verified_success_2.jsonl")
    value.add_argument("--environment-contract", type=Path, default=Path("outputs/repair_pipeline_v2/environment_contract.json"))
    value.add_argument("--output", type=Path, default=Path("outputs/leanworkbook_failure_repair"))
    value.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    value.add_argument("--count", type=int, default=DEFAULT_COUNT)
    value.add_argument("--seed", type=int, default=DEFAULT_SEED)
    value.add_argument("--api-workers", type=int, default=6)
    value.add_argument("--timeout", type=int, default=60)
    return value


def main() -> None:
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "manual-candidates":
        manual_candidates(args)
    elif args.command == "manual-all-candidates":
        manual_all_candidates(args)
    elif args.command == "manual-finalize":
        manual_finalize(args)
    elif args.command == "generate":
        generate(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
