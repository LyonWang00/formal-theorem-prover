"""Indexed, staged NuminaMath repair campaign orchestration.

This module keeps acceleration metadata outside the canonical JSONL rows.  It
supports one read-only SQLite index, conservative rule-assisted candidate
preparation, DeepSeek-assisted candidate preparation, explicit invalid
statement manifests, and one atomic campaign commit after all Pantograph jobs
have finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from openai import OpenAI

from lean_prover.Dataset.build_verified_datasets import (
    FAIL,
    SUCCESS,
    _write_json,
    add_dataset_contract,
    dataset_record_hash,
    sha256_file,
)
from lean_prover.Dataset.repair_numinamath_failures import (
    REPAIR_VERSION,
    _result_error,
    append_jsonl,
    canonical_hash,
    classify_row,
    complete_source,
    error_family,
    iter_jsonl,
    read_jsonl,
    write_jsonl,
)
from lean_prover.Dataset.verify_external_datasets import (
    NUMINA_SOURCE,
    select_numina_source,
    split_imports,
)


INDEX_SCHEMA = "numinamath_fail_index_v1"
CAMPAIGN_SCHEMA = "numinamath_repair_campaign_v1"
INVALID_SCHEMA = "numinamath_invalid_statement_v1"
_FINAL_PLACEHOLDER = re.compile(r":=\s*by\s+(?:sorry|admit)\s*$", re.I | re.S)
_PROOF_START = re.compile(r":=\s*by\b")
# Only top-level declaration commands may delimit the target statement.  A
# loose word-boundary match also sees prose such as ``-- lemma of mod 8``
# inside a failed proof.  That made candidate extraction choose a later local
# ``:= by`` and silently splice proof text into the statement scaffold.
_DECLARATION = re.compile(r"(?m)^[ \t]*(?:theorem|lemma|example)\b")
_FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.I)
_SUSPICIOUS = re.compile(
    r"Set\.(?:ncard|encard)|\.ncard|\.encard|∑ᶠ|∏ᶠ|∑'|∫|MeasureTheory|"
    r"Polynomial|\[X\]|\.roots|\.aroots|\.rootSet|Real\.(?:log|logb|sin|cos|tan|rpow)|"
    r"Complex|Matrix|EuclideanSpace|Nat\.digits|Nat\.divisors|multiplicity",
)
_LINEARISH = re.compile(r"\b(?:linarith|nlinarith|omega|norm_num|ring)\b")
_FOLLOWUP_COMPATIBILITY_RENAMES = {
    "Nat.pow_le_pow_of_le_left": "Nat.pow_le_pow_left",
    "Nat.pow_le_pow_of_le_right": "Nat.pow_le_pow_right",
    "Nat.dvd_sub'": "Nat.dvd_sub",
    "Real.sqrt_eq_iff_sq_eq": "Real.sqrt_eq_iff_eq_sq",
    "Real.pi_gt_31415": "Real.pi_gt_d4",
    "Real.pi_lt_31416": "Real.pi_lt_d4",
    "pi_gt_31415": "pi_gt_d4",
    "pi_lt_31416": "pi_lt_d4",
    "Int.dvd_iff_mod_eq_zero": "Int.dvd_iff_emod_eq_zero",
    "ZMod.eq_iff_modEq_nat": "ZMod.natCast_eq_natCast_iff",
    "Set.eq_empty_iff_forall_not_mem": "Set.eq_empty_iff_forall_notMem",
    "Set.ncard_insert_of_not_mem": "Set.ncard_insert_of_notMem",
    "Set.ncard_coe_Finset": "Set.ncard_coe_finset",
    "Finset.card_insert_of_not_mem": "Finset.card_insert_of_notMem",
    # `Complex.abs` was the pre-norm-API name for the complex modulus.
    # The current mathlib exposes the same operation through the generic norm.
    "Complex.abs": "norm",
    "div_lt_iff'": "div_lt_iff\u2080'",
    "div_le_iff'": "div_le_iff\u2080'",
    "lt_div_iff'": "lt_div_iff\u2080'",
    "le_div_iff'": "le_div_iff\u2080'",
}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_features(source: str) -> tuple[str, int, int, int]:
    final_placeholder = int(bool(_FINAL_PLACEHOLDER.search(source.strip())))
    declaration_count = len(_DECLARATION.findall(source))
    suspicious_count = len(_SUSPICIOUS.findall(source))
    proof_prefix = source.rsplit(":=", 1)[-1] if ":=" in source else ""
    nonfinal_forbidden = int(
        bool(_FORBIDDEN.search(_FINAL_PLACEHOLDER.sub("", source)))
    )
    if final_placeholder and declaration_count == 1 and not suspicious_count:
        family = "short_final_placeholder"
    elif final_placeholder and declaration_count == 1:
        family = "advanced_final_placeholder"
    elif "no goals to be solved" in source.lower() or _LINEARISH.search(proof_prefix):
        family = "proof_repair"
    else:
        family = "other"
    return family, final_placeholder, declaration_count, nonfinal_forbidden


def build_index(*, fail_file: Path, index_file: Path, report_file: Path) -> dict[str, Any]:
    index_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_file.with_suffix(index_file.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.execute(
        """
        CREATE TABLE records (
          record_id TEXT PRIMARY KEY,
          line_number INTEGER NOT NULL,
          byte_offset INTEGER NOT NULL,
          byte_length INTEGER NOT NULL,
          record_hash TEXT NOT NULL,
          selected_field TEXT NOT NULL,
          selected_source_sha256 TEXT NOT NULL,
          source_chars INTEGER NOT NULL,
          classification TEXT NOT NULL,
          error_family TEXT NOT NULL,
          easy_family TEXT NOT NULL,
          final_placeholder INTEGER NOT NULL,
          declaration_count INTEGER NOT NULL,
          nonfinal_forbidden INTEGER NOT NULL,
          repair_lane TEXT,
          error_head TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX idx_easy ON records(easy_family, source_chars)")
    connection.execute("CREATE INDEX idx_error ON records(error_family, source_chars)")
    rows = 0
    families: dict[str, int] = {}
    error_families: dict[str, int] = {}
    with fail_file.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            text = raw.decode("utf-8-sig" if rows == 0 else "utf-8")
            row = json.loads(text)
            selected_field, source = select_numina_source(row)
            family, final_placeholder, declarations, nonfinal_forbidden = _source_features(source)
            message = str(row.get("repair_error") or row.get("error_message") or "")
            err = error_family(message)
            repair = row.get("repair") if isinstance(row.get("repair"), Mapping) else {}
            connection.execute(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(row["record_id"]),
                    rows + 1,
                    offset,
                    len(raw),
                    str(row.get("record_hash") or ""),
                    selected_field,
                    _sha256_text(source),
                    len(source),
                    classify_row(row),
                    err,
                    family,
                    final_placeholder,
                    declarations,
                    nonfinal_forbidden,
                    str(repair.get("lane") or ""),
                    message.splitlines()[0][:500] if message else "",
                ),
            )
            rows += 1
            families[family] = families.get(family, 0) + 1
            error_families[err] = error_families.get(err, 0) + 1
    meta = {
        "schema_version": INDEX_SCHEMA,
        "fail_file": str(fail_file.resolve()),
        "fail_sha256": sha256_file(fail_file),
        "rows": rows,
        "created_unix": int(time.time()),
    }
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO meta VALUES (?,?)",
        [(key, json.dumps(value, ensure_ascii=False)) for key, value in meta.items()],
    )
    connection.commit()
    connection.close()
    temporary.replace(index_file)
    report = {
        **meta,
        "index_file": str(index_file.resolve()),
        "index_sha256": sha256_file(index_file),
        "easy_family": dict(sorted(families.items())),
        "error_family": dict(sorted(error_families.items())),
        "temporary_fields_are_external_only": True,
    }
    _json_dump(report_file, report)
    return report


class IndexedFailRows:
    def __init__(self, fail_file: Path, index_file: Path) -> None:
        self.fail_file = fail_file
        self.connection = sqlite3.connect(index_file)
        meta = {
            key: json.loads(value)
            for key, value in self.connection.execute("SELECT key,value FROM meta")
        }
        actual = sha256_file(fail_file)
        if meta.get("schema_version") != INDEX_SCHEMA or meta.get("fail_sha256") != actual:
            raise RuntimeError("temporary record index is stale for the current fail file")

    def close(self) -> None:
        self.connection.close()

    def query_easy(
        self,
        *,
        limit: int,
        max_source_chars: int,
        excluded_ids: set[str],
        easy_family: str = "short_final_placeholder",
        error_families: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        where = [
            "easy_family=?",
            "source_chars<=?",
            "repair_lane NOT IN ('short_placeholder_portfolio','simple_placeholder_portfolio',"
            "'simple_empty_portfolio','targeted_placeholder_two_tactic',"
            "'hazardous_power_grind_placeholder')",
        ]
        parameters: list[Any] = [easy_family, max_source_chars]
        if easy_family == "short_final_placeholder":
            where.extend(("final_placeholder=1", "declaration_count=1", "nonfinal_forbidden=0"))
        if error_families:
            placeholders = ",".join("?" for _ in error_families)
            where.append(f"error_family IN ({placeholders})")
            parameters.extend(error_families)
        cursor = self.connection.execute(
            f"""
            SELECT record_id,line_number,byte_offset,byte_length,record_hash,
                   selected_field,selected_source_sha256,source_chars,
                   classification,error_family,easy_family,error_head
            FROM records
            WHERE {' AND '.join(where)}
            ORDER BY source_chars ASC, record_id ASC
            """,
            tuple(parameters),
        )
        selected: list[dict[str, Any]] = []
        for values in cursor:
            item = dict(zip((
                "record_id","line_number","byte_offset","byte_length","record_hash",
                "selected_field","selected_source_sha256","source_chars",
                "classification","error_family","easy_family","error_head",
            ), values))
            if item["record_id"] in excluded_ids:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def get_row(self, record_id: str) -> dict[str, Any]:
        found = self.connection.execute(
            "SELECT byte_offset,byte_length FROM records WHERE record_id=?", (record_id,)
        ).fetchone()
        if found is None:
            raise KeyError(record_id)
        offset, length = found
        with self.fail_file.open("rb") as handle:
            handle.seek(int(offset))
            raw = handle.read(int(length))
        row = json.loads(raw.decode("utf-8-sig" if int(offset) == 0 else "utf-8"))
        if str(row.get("record_id")) != record_id:
            raise RuntimeError(f"record index offset mismatch for {record_id}")
        return row


def _frozen_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        result.update(str(row["record_id"]) for row in read_jsonl(path))
    return result


def _statement_fingerprint(source: str) -> str:
    """Return a whitespace/comment-insensitive fingerprint before the target proof."""
    _, body = split_imports(source)
    declarations = list(_DECLARATION.finditer(body))
    if declarations:
        proof_start = _PROOF_START.search(body, declarations[-1].start())
        if proof_start is not None:
            body = body[: proof_start.start()]
    body = re.sub(r"/-.*?-/", "", body, flags=re.S)
    body = re.sub(r"--[^\n]*", "", body)
    return _sha256_text(re.sub(r"\s+", "", body))


def _statement_fingerprint_without_declaration_name(source: str) -> str:
    """Fingerprint a statement while deliberately ignoring only its declaration name."""
    _, body = split_imports(source)
    declarations = list(_DECLARATION.finditer(body))
    if declarations:
        proof_start = _PROOF_START.search(body, declarations[-1].start())
        if proof_start is not None:
            body = body[: proof_start.start()]
    body = re.sub(r"/-.*?-/", "", body, flags=re.S)
    body = re.sub(r"--[^\n]*", "", body)
    body = re.sub(
        r"\b(theorem|lemma|example)\s+[^\s({:\[]+",
        r"\1 __DECLARATION_NAME__",
        body,
        count=1,
    )
    return _sha256_text(re.sub(r"\s+", "", body))


def _proof_from_complete_source(source: str) -> str:
    """Extract the final declaration's proof without trusting upstream status labels."""
    _, body = split_imports(source)
    declarations = list(_DECLARATION.finditer(body))
    if not declarations:
        raise ValueError("raw proof source contains no declaration")
    proof_start = _PROOF_START.search(body, declarations[-1].start())
    if proof_start is None:
        raise ValueError("raw proof source contains no ':= by' proof")
    proof = re.sub(r"^:=\s*", "", body[proof_start.start():].strip(), count=1)
    if not proof.startswith("by"):
        raise ValueError("raw proof does not start with 'by'")
    if _FORBIDDEN.search(proof):
        raise ValueError("raw proof contains a forbidden placeholder")
    return proof


def prepare_raw_recovery_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Recover statement-aligned alternate proofs from the immutable raw parquet."""
    import pyarrow.parquet as parquet

    excluded = _frozen_ids(args.frozen_manifest)
    source_failure_ids: set[str] | None = None
    if args.source_batch_failures is not None:
        source_candidates = read_jsonl(args.source_batch_failures / "candidate_manifest.jsonl")
        source_results = read_jsonl(args.source_batch_failures / "verification_results.jsonl")
        source_successes = {
            str(row["record_id"]) for row in source_results if row.get("success")
        }
        source_failure_ids = {
            str(row["record_id"]) for row in source_candidates
            if str(row["record_id"]) not in source_successes
        }
    raw_parquet_sha256 = sha256_file(args.raw_parquet)
    indexed = IndexedFailRows(args.fail_file, args.index_file)
    try:
        raw_rows = parquet.read_table(
            args.raw_parquet,
            columns=["uuid", args.raw_proof_field, "rl_data"],
        ).to_pylist()
        candidates: list[dict[str, Any]] = []
        rejected = {
            "not_in_current_fail": 0,
            "excluded": 0,
            "empty_or_forbidden_proof": 0,
            "statement_mismatch": 0,
            "malformed_proof": 0,
        }
        current_by_uuid: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(args.fail_file):
            uuid = str(row.get("uuid") or "")
            if uuid:
                current_by_uuid[uuid] = row
        for raw in raw_rows:
            row = current_by_uuid.get(str(raw.get("uuid") or ""))
            if row is None:
                rejected["not_in_current_fail"] += 1
                continue
            if str(row["record_id"]) in excluded:
                rejected["excluded"] += 1
                continue
            if source_failure_ids is not None and str(row["record_id"]) not in source_failure_ids:
                rejected["not_in_current_fail"] += 1
                continue
            raw_source = str(raw.get(args.raw_proof_field) or "").strip()
            if not raw_source or _FORBIDDEN.search(raw_source):
                rejected["empty_or_forbidden_proof"] += 1
                continue
            _, current_source = select_numina_source(row)
            exact_statement_match = (
                _statement_fingerprint(current_source) == _statement_fingerprint(raw_source)
            )
            name_only_statement_match = (
                args.allow_declaration_name_mismatch
                and _statement_fingerprint_without_declaration_name(current_source)
                == _statement_fingerprint_without_declaration_name(raw_source)
            )
            if not exact_statement_match and not name_only_statement_match:
                rejected["statement_mismatch"] += 1
                continue
            try:
                proof = _proof_from_complete_source(raw_source)
            except ValueError:
                rejected["malformed_proof"] += 1
                continue
            rl_data = raw.get("rl_data") if isinstance(raw.get("rl_data"), Mapping) else {}
            candidates.append(_candidate_from_row(
                row,
                batch_id=args.batch_id,
                lane="raw_upstream_proof_recovery",
                variants=[{"strategy": "raw_formal_proof_recovery", "proof": proof}],
                metadata={
                    "raw_parquet_sha256": raw_parquet_sha256,
                    "raw_uuid": str(raw.get("uuid") or ""),
                    "raw_proof_field": args.raw_proof_field,
                    "upstream_correct_proof_count": int(rl_data.get("n_correct_proofs") or 0),
                    "upstream_proof_count": int(rl_data.get("n_proofs") or 0),
                    "statement_fingerprint_match": exact_statement_match,
                    "statement_match_ignoring_declaration_name": name_only_statement_match,
                },
            ))
            if args.limit and len(candidates) >= args.limit:
                break
    finally:
        indexed.close()

    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, candidates)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-raw-recovery",
        "batch_id": args.batch_id,
        "selected_rows": len(candidates),
        "variants_per_row": 1,
        "raw_parquet": str(args.raw_parquet.resolve()),
        "raw_parquet_sha256": raw_parquet_sha256,
        "raw_proof_field": args.raw_proof_field,
        "source_batch_failures": (
            str(args.source_batch_failures.resolve())
            if args.source_batch_failures is not None else None
        ),
        "candidate_manifest_sha256": sha256_file(manifest),
        "frozen_record_count": len(excluded),
        "overlap_with_frozen": 0,
        "rejected": rejected,
        "statement_changed": False,
        "allow_declaration_name_mismatch": args.allow_declaration_name_mismatch,
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def _rule_variants(max_variants: int, *, profile: str = "general") -> list[dict[str, str]]:
    general = [
        ("rule_norm_num", "by\n  norm_num"),
        ("rule_omega", "by\n  omega"),
        ("rule_simp", "by\n  simp"),
        ("rule_native_decide", "by\n  native_decide"),
        ("rule_aesop", "by\n  aesop"),
        ("rule_grind", "by\n  grind"),
    ]
    compute = [
        ("compute_native_decide", "by\n  native_decide"),
        ("compute_norm_num", "by\n  norm_num"),
        ("compute_decide", "by\n  decide"),
        ("compute_grind", "by\n  grind"),
    ]
    algebra = [
        ("algebra_rfl", "by\n  rfl"),
        ("algebra_simp_all", "by\n  simp_all"),
        ("algebra_linarith", "by\n  linarith"),
        ("algebra_nlinarith", "by\n  nlinarith"),
        ("algebra_ring", "by\n  ring"),
        ("algebra_ring_nf", "by\n  ring_nf"),
    ]
    portfolio = compute if profile == "compute" else algebra if profile == "algebra" else general
    return [{"strategy": name, "proof": proof} for name, proof in portfolio[:max_variants]]


def _interval_cases_proof_variants(
    proof: str, error_message: str
) -> list[dict[str, str]]:
    """Sequentialize one verified-failing interval proof without changing its plan."""
    message = str(error_message or "")
    if "No goals to be solved" not in message:
        return []
    proof = str(proof or "").strip()
    if not proof.startswith("by") or _FORBIDDEN.search(proof):
        return []
    if "interval_cases" not in proof:
        return []
    def split_hypothesis_goal_norm_num(match: re.Match[str]) -> str:
        config = match.group(1) or ""
        hypotheses = [item for item in match.group(2).split() if item]
        if not hypotheses or hypotheses == ["*"]:
            return match.group(0)
        steps = [f"norm_num{config} at {hypothesis}" for hypothesis in hypotheses]
        steps.append(f"norm_num{config}")
        return " <;> ".join(steps)

    repaired, replacements = re.subn(
        "norm_num(\\s*\\[[^\\]\\n]*\\])?\\s+at\\s+([^\\n<]*?)\\s*⊢",
        split_hypothesis_goal_norm_num,
        proof,
    )
    if not replacements or repaired == proof:
        return []
    return [{
        "strategy": "rule_interval_cases_split_hypothesis_goal_norm_num",
        "proof": repaired,
    }]


def _remove_no_goals_span_variants(
    source_body: str, proof: str, error_message: str
) -> list[dict[str, str]]:
    """Remove only tactic spans Lean identified as running after all goals closed."""
    if "No goals to be solved" not in error_message:
        return []
    proof = str(proof or "").strip()
    if not proof.startswith("by") or _FORBIDDEN.search(proof):
        return []
    proof_lines = proof.splitlines()
    source_line_offset = len(str(source_body).splitlines()) - 1
    edits: dict[int, list[tuple[int, int]]] = {}
    for match in re.finditer(
        r"(?m)^(\d+):(\d+)-(\d+):(\d+): error: No goals to be solved",
        error_message,
    ):
        start_line, start_col, end_line, end_col = map(int, match.groups())
        if start_line != end_line:
            continue
        proof_index = start_line - source_line_offset - 1
        if proof_index <= 0 or proof_index >= len(proof_lines):
            continue
        line = proof_lines[proof_index]
        if not (0 <= start_col < end_col <= len(line)):
            continue
        if not line[start_col:end_col].strip():
            continue
        edits.setdefault(proof_index, []).append((start_col, end_col))
    if not edits:
        return []
    for proof_index, spans in edits.items():
        line = proof_lines[proof_index]
        for start_col, end_col in sorted(set(spans), reverse=True):
            line = line[:start_col] + line[end_col:]
        line = re.sub(r"<;>\s*<;>", "<;>", line)
        indent = line[: len(line) - len(line.lstrip())]
        stripped = line.strip()
        if stripped.startswith("<;>"):
            stripped = stripped[3:].lstrip()
        if stripped.endswith("<;>"):
            stripped = stripped[:-3].rstrip()
        proof_lines[proof_index] = indent + stripped if stripped else ""
    repaired = "\n".join(
        line for index, line in enumerate(proof_lines)
        if line.strip() or index not in edits
    ).strip()
    if repaired == proof or not repaired.startswith("by") or _FORBIDDEN.search(repaired):
        return []
    return [{
        "strategy": "rule_remove_verified_no_goals_tactic_span",
        "proof": repaired,
    }]


def _interval_cases_source_variants(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build a narrow, proof-preserving repair for a known no-goals pattern."""
    message = str(row.get("repair_error") or row.get("error_message") or "")
    _, source = select_numina_source(row)
    _, body = split_imports(source)
    declarations = list(_DECLARATION.finditer(body))
    if len(declarations) != 1:
        return []
    proof_start = _PROOF_START.search(body, declarations[-1].start())
    if proof_start is None:
        return []
    proof = re.sub(r"^:=\s*", "", body[proof_start.start():].strip(), count=1)
    return _interval_cases_proof_variants(proof, message)


def _interval_cases_source_variants_legacy(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build a narrow, proof-preserving repair for a known no-goals pattern."""
    message = str(row.get("repair_error") or row.get("error_message") or "")
    if "No goals to be solved" not in message:
        return []
    _, source = select_numina_source(row)
    _, body = split_imports(source)
    declarations = list(_DECLARATION.finditer(body))
    if len(declarations) != 1:
        return []
    proof_start = _PROOF_START.search(body, declarations[-1].start())
    if proof_start is None:
        return []
    proof = body[proof_start.start():].strip()
    proof = re.sub(r"^:=\s*", "", proof, count=1)
    if not proof.startswith("by") or _FORBIDDEN.search(proof):
        return []
    if "interval_cases" not in proof:
        return []
    repaired, replacements = re.subn(
        r"norm_num\s+at\s+([^\n<]*?)\s*⊢",
        lambda match: f"norm_num at {match.group(1).strip()} <;> norm_num",
        proof,
    )
    if not replacements or repaired == proof:
        return []
    return [{
        "strategy": "rule_interval_cases_split_hypothesis_goal_norm_num",
        "proof": repaired,
    }]


def _candidate_from_row(
    row: Mapping[str, Any], *, batch_id: str, lane: str, variants: list[dict[str, str]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_field, source = select_numina_source(row)
    imports, body = split_imports(source)
    source_body = _FINAL_PLACEHOLDER.sub(":= by", body.strip())
    if not re.search(r":=\s*by\s*$", source_body, re.S):
        declarations = list(_DECLARATION.finditer(source_body))
        if not declarations:
            raise ValueError(f"cannot isolate target declaration for {row['record_id']}")
        proof_start = _PROOF_START.search(source_body, declarations[-1].start())
        if proof_start is None:
            source_body = source_body.rstrip()
            source_body += " by" if re.search(r":=\s*$", source_body) else " := by"
        else:
            source_body = source_body[: proof_start.start()].rstrip() + " := by"
    if not re.search(r":=\s*by\s*$", source_body, re.S):
        declarations = list(_DECLARATION.finditer(source_body))
        if not declarations:
            raise ValueError(f"cannot isolate target declaration for {row['record_id']}")
        proof_start = _PROOF_START.search(source_body, declarations[-1].start())
        if proof_start is None:
            source_body = source_body.rstrip()
            source_body += " by" if re.search(r":=\s*$", source_body) else " := by"
        else:
            source_body = source_body[: proof_start.start()].rstrip() + " := by"
    source_body, lean3_finset_binders = re.subn(
        r"([∑∏]\s+[\w₀-₉']+)\s+in\s+",
        r"\1 ∈ ",
        source_body,
    )
    changes: list[dict[str, Any]] = []
    if lean3_finset_binders:
        changes.append({
            "kind": "syntax_normalization",
            "operation": "lean3_finset_binder_in_to_membership",
            "replacement_count": lean3_finset_binders,
            "semantic_change": False,
        })
    candidate = {
        "schema_version": "numinamath_repair_candidate_v1",
        "repair_version": REPAIR_VERSION,
        "batch_id": batch_id,
        "record_id": str(row["record_id"]),
        "category": classify_row(row),
        "lane": lane,
        "imports": list(imports),
        "source_body": source_body,
        "original_source_body": body.strip(),
        "selected_source_field": selected_field,
        "selected_source_sha256": _sha256_text(source),
        "statement_changed": False,
        "changes": changes,
        "original_error_message": str(row.get("repair_error") or row.get("error_message") or ""),
        "original_record_hash": str(row.get("record_hash") or ""),
        "variants": variants,
        "campaign_metadata": dict(metadata or {}),
    }
    candidate["candidate_hash"] = canonical_hash(candidate)
    return candidate


def prepare_rule_batch(args: argparse.Namespace) -> dict[str, Any]:
    excluded = _frozen_ids(args.frozen_manifest)
    indexed = IndexedFailRows(args.fail_file, args.index_file)
    try:
        selected = indexed.query_easy(
            limit=args.limit,
            max_source_chars=args.max_source_chars,
            excluded_ids=excluded,
            easy_family=args.easy_family,
            error_families=tuple(args.error_family),
        )
        candidates = []
        retained_selection = []
        for item in selected:
            row = indexed.get_row(item["record_id"])
            variants = (
                _interval_cases_source_variants(row)
                if args.strategy_profile == "interval_cases"
                else _rule_variants(args.max_variants, profile=args.strategy_profile)
            )
            if not variants:
                continue
            candidates.append(_candidate_from_row(
                row,
                batch_id=args.batch_id,
                lane="indexed_rule_assisted",
                variants=variants[: args.max_variants],
                metadata={"index_line_number": item["line_number"], "easy_family": item["easy_family"]},
            ))
            retained_selection.append(item)
        selected = retained_selection
    finally:
        indexed.close()
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, candidates)
    selection = batch_dir / "selection_manifest.jsonl"
    write_jsonl(selection, selected)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-rule",
        "batch_id": args.batch_id,
        "selected_rows": len(candidates),
        "variant_count_distribution": {
            str(count): sum(1 for candidate in candidates if len(candidate["variants"]) == count)
            for count in sorted({len(candidate["variants"]) for candidate in candidates})
        },
        "easy_family": args.easy_family,
        "error_family": list(args.error_family),
        "strategy_profile": args.strategy_profile,
        "candidate_manifest_sha256": sha256_file(manifest),
        "selection_manifest_sha256": sha256_file(selection),
        "frozen_record_count": len(excluded),
        "overlap_with_frozen": 0,
        "temporary_index_fields_in_candidates_only": True,
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def prepare_followup_batch(args: argparse.Namespace) -> dict[str, Any]:
    source_batch = args.source_batch
    candidates = read_jsonl(source_batch / "candidate_manifest.jsonl")
    results = read_jsonl(source_batch / "verification_results.jsonl")
    succeeded = {str(row["record_id"]) for row in results if row.get("success")}
    first_result: dict[str, dict[str, Any]] = {}
    for row in results:
        first_result.setdefault(str(row["record_id"]), row)
    diagnostic_markers = {
        "free_variables": "Expected type must not contain free variables",
        "noncomputable": "noncomputable",
        "decidable_missing": "failed to synthesize",
        "syntax_preamble": "unexpected token",
        "unknown_constant": "Unknown constant",
    }
    marker = diagnostic_markers[args.diagnostic_family]
    selected = [
        row for row in candidates
        if str(row["record_id"]) not in succeeded
        and marker in str(first_result.get(str(row["record_id"]), {}).get("diagnostics") or "")
    ]
    if args.limit:
        selected = selected[: args.limit]
    portfolios = {
        "context": [
            ("context_omega", "by\n  omega"),
            ("context_simp_all", "by\n  simp_all"),
            ("context_aesop", "by\n  aesop"),
            ("context_norm_num_all", "by\n  norm_num at *"),
            ("context_linarith", "by\n  linarith"),
            ("context_nlinarith", "by\n  nlinarith"),
        ],
        "algebra": [
            ("algebra_ring_nf", "by\n  ring_nf"),
            ("algebra_norm_num", "by\n  norm_num"),
            ("algebra_field_simp", "by\n  field_simp"),
            ("algebra_nlinarith", "by\n  nlinarith"),
        ],
    }
    variants = [
        {"strategy": strategy, "proof": proof}
        for strategy, proof in portfolios[args.strategy_profile][: args.max_variants]
    ]
    generated: list[dict[str, Any]] = []
    for source_candidate in selected:
        candidate = dict(source_candidate)
        candidate.update({
            "batch_id": args.batch_id,
            "lane": "indexed_rule_followup",
            "variants": variants,
            "campaign_metadata": {
                "source_batch": str(source_batch),
                "diagnostic_family": args.diagnostic_family,
                "strategy_profile": args.strategy_profile,
            },
        })
        candidate.pop("candidate_hash", None)
        candidate["candidate_hash"] = canonical_hash(candidate)
        generated.append(candidate)
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, generated)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-followup",
        "batch_id": args.batch_id,
        "source_batch": str(source_batch),
        "diagnostic_family": args.diagnostic_family,
        "strategy_profile": args.strategy_profile,
        "selected_rows": len(generated),
        "variants_per_row": len(variants),
        "candidate_manifest_sha256": sha256_file(manifest),
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def prepare_interval_cases_followup_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Retry failed interval-case repairs only when sequentialization changes proof text."""
    source_candidates = read_jsonl(args.source_batch / "candidate_manifest.jsonl")
    results = read_jsonl(args.source_batch / "verification_results.jsonl")
    succeeded = {str(row["record_id"]) for row in results if row.get("success")}
    failed_diagnostics: dict[str, str] = {}
    for result in results:
        if result.get("success"):
            continue
        record_id = str(result["record_id"])
        diagnostic = "\n".join([
            str(result.get("diagnostics") or ""),
            *(str(item) for item in result.get("errors") or []),
        ])
        if diagnostic:
            failed_diagnostics[record_id] = diagnostic
    frozen = _frozen_ids(args.frozen_manifest)
    indexed = IndexedFailRows(args.fail_file, args.index_file)
    generated: list[dict[str, Any]] = []
    unchanged_proofs = 0
    missing_current_fail = 0
    try:
        for source_candidate in source_candidates:
            record_id = str(source_candidate["record_id"])
            if record_id in succeeded or record_id in frozen:
                continue
            try:
                row = indexed.get_row(record_id)
            except KeyError:
                missing_current_fail += 1
                continue
            diagnostic = failed_diagnostics.get(record_id, "")
            variants: list[dict[str, str]] = []
            for source_variant in source_candidate.get("variants") or []:
                variants.extend(_remove_no_goals_span_variants(
                    str(source_candidate.get("source_body") or ""),
                    str(source_variant.get("proof") or ""),
                    diagnostic,
                ))
            if not variants:
                for source_variant in source_candidate.get("variants") or []:
                    variants.extend(_interval_cases_proof_variants(
                        str(source_variant.get("proof") or ""), diagnostic
                    ))
            if not variants:
                variants = _interval_cases_source_variants(row)
            previous_proofs = {
                str(variant.get("proof") or "").strip()
                for variant in source_candidate.get("variants") or []
            }
            variants = [
                variant for variant in variants
                if str(variant.get("proof") or "").strip() not in previous_proofs
            ]
            if not variants:
                unchanged_proofs += 1
                continue
            generated.append(_candidate_from_row(
                row,
                batch_id=args.batch_id,
                lane="indexed_rule_followup",
                variants=variants,
                metadata={
                    "source_batch": str(args.source_batch),
                    "followup_reason": "sequentialize_norm_num_hypotheses",
                },
            ))
    finally:
        indexed.close()
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, generated)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-interval-followup",
        "batch_id": args.batch_id,
        "source_batch": str(args.source_batch),
        "selected_rows": len(generated),
        "unchanged_proofs_skipped": unchanged_proofs,
        "missing_current_fail": missing_current_fail,
        "overlap_with_frozen": 0,
        "candidate_manifest_sha256": sha256_file(manifest),
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def prepare_compatibility_followup_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Apply exact Lean/mathlib API renames to active proofs from a failed batch."""
    source_candidates = read_jsonl(args.source_batch / "candidate_manifest.jsonl")
    results = read_jsonl(args.source_batch / "verification_results.jsonl")
    succeeded = {str(row["record_id"]) for row in results if row.get("success")}
    failed_ids = {str(row["record_id"]) for row in results if not row.get("success")}
    frozen = _frozen_ids(args.frozen_manifest)
    generated: list[dict[str, Any]] = []
    unchanged_proofs = 0
    for source_candidate in source_candidates:
        record_id = str(source_candidate["record_id"])
        if record_id in succeeded or record_id not in failed_ids or record_id in frozen:
            continue
        source_body = str(source_candidate.get("source_body") or "")
        variants: list[dict[str, str]] = []
        applied_names: set[str] = set()
        for source_variant in source_candidate.get("variants") or []:
            proof = str(source_variant.get("proof") or "")
            repaired_proof = proof
            variant_names: list[str] = []
            for previous, current in _FOLLOWUP_COMPATIBILITY_RENAMES.items():
                if previous in repaired_proof:
                    repaired_proof = repaired_proof.replace(previous, current)
                    applied_names.add(previous)
                    variant_names.append(previous)
            if repaired_proof != proof:
                variants.append({
                    "strategy": "followup_api_compatibility:" + "+".join(sorted(variant_names)),
                    "proof": repaired_proof,
                })
        repaired_source_body = source_body
        for previous, current in _FOLLOWUP_COMPATIBILITY_RENAMES.items():
            repaired_source_body = repaired_source_body.replace(previous, current)
        if not variants and repaired_source_body == source_body:
            unchanged_proofs += 1
            continue
        candidate = dict(source_candidate)
        candidate.update({
            "batch_id": args.batch_id,
            "lane": "indexed_rule_followup",
            "source_body": repaired_source_body,
            "variants": variants or list(source_candidate.get("variants") or []),
            "campaign_metadata": {
                "source_batch": str(args.source_batch),
                "followup_reason": "exact_lean_mathlib_api_compatibility",
                "renamed_identifiers": sorted(applied_names),
            },
        })
        candidate.pop("candidate_hash", None)
        candidate["candidate_hash"] = canonical_hash(candidate)
        generated.append(candidate)
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, generated)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-compatibility-followup",
        "batch_id": args.batch_id,
        "source_batch": str(args.source_batch),
        "selected_rows": len(generated),
        "unchanged_proofs_skipped": unchanged_proofs,
        "overlap_with_frozen": 0,
        "candidate_manifest_sha256": sha256_file(manifest),
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def _wrap_target_declaration_with_options(
    source_body: str, options: list[tuple[str, int]]
) -> str:
    """Apply finite Lean options to the target declaration, not its preamble.

    ``set_option ... in`` scopes exactly one following command.  Prefixing it
    to a source body beginning with ``open Real`` therefore scopes the option
    to the ``open`` command and makes that namespace unavailable to the target
    theorem.  Insert the wrappers immediately before the final declaration so
    preceding ``open``/``namespace``/``variable`` commands retain their normal
    scope while the resource limits apply to the proof being retried.
    """
    declarations = list(_DECLARATION.finditer(source_body))
    if not declarations:
        raise ValueError("cannot isolate target declaration for resource followup")
    target_start = declarations[-1].start()
    prefix = source_body[:target_start]
    target = source_body[target_start:]
    wrappers = "".join(
        f"set_option {option} {limit} in\n" for option, limit in options
    )
    return prefix + wrappers + target


def prepare_resource_limit_followup_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Retry otherwise unchanged proofs under finite, explicitly audited Lean limits."""
    source_candidates = read_jsonl(args.source_batch / "candidate_manifest.jsonl")
    results = read_jsonl(args.source_batch / "verification_results.jsonl")
    succeeded = {str(row["record_id"]) for row in results if row.get("success")}
    diagnostics_by_id: dict[str, list[str]] = {}
    for row in results:
        if row.get("success"):
            continue
        diagnostics_by_id.setdefault(str(row["record_id"]), []).append(
            "\n".join([
                str(row.get("diagnostics") or ""),
                *(str(item) for item in row.get("errors") or []),
            ])
        )
    frozen = _frozen_ids(args.frozen_manifest)
    generated: list[dict[str, Any]] = []
    for source_candidate in source_candidates:
        record_id = str(source_candidate["record_id"])
        if record_id in succeeded or record_id in frozen:
            continue
        diagnostic = "\n".join(diagnostics_by_id.get(record_id, []))
        options: list[tuple[str, int]] = []
        if "maximum number of heartbeats" in diagnostic:
            options.append(("maxHeartbeats", 1_000_000))
        if "maximum recursion depth" in diagnostic:
            options.append(("maxRecDepth", 10_000))
        if not options:
            continue
        source_body = _wrap_target_declaration_with_options(
            str(source_candidate["source_body"]), options
        )
        candidate = dict(source_candidate)
        candidate.update({
            "batch_id": args.batch_id,
            "lane": "resource_limit_followup",
            "source_body": source_body,
            "changes": [
                *(source_candidate.get("changes") or []),
                *({
                    "kind": "environment_limit",
                    "operation": f"set_option_{option}",
                    "value": limit,
                    "semantic_change": False,
                } for option, limit in options),
            ],
            "campaign_metadata": {
                "source_batch": str(args.source_batch),
                "followup_reason": "finite_resource_limit_retry",
                "resource_options": {option: limit for option, limit in options},
            },
        })
        candidate.pop("candidate_hash", None)
        candidate["candidate_hash"] = canonical_hash(candidate)
        generated.append(candidate)
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, generated)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-resource-followup",
        "batch_id": args.batch_id,
        "source_batch": str(args.source_batch),
        "selected_rows": len(generated),
        "overlap_with_frozen": 0,
        "candidate_manifest_sha256": sha256_file(manifest),
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def prepare_wall_timeout_followup_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Stage unchanged candidates that exhausted the Pantograph wall-clock limit.

    This is deliberately separate from Lean's heartbeat/recursion options: the
    theorem and proof bytes stay unchanged, while the queue job records the
    larger finite wall-clock allowance when it is enqueued.
    """
    source_candidates = read_jsonl(args.source_batch / "candidate_manifest.jsonl")
    results = read_jsonl(args.source_batch / "verification_results.jsonl")
    succeeded = {str(row["record_id"]) for row in results if row.get("success")}
    timed_out = {
        str(row["record_id"])
        for row in results
        if not row.get("success")
        and (
            row.get("timed_out")
            or "timeout" in str(row.get("diagnostics") or "").lower()
            or any("timeout" in str(item).lower() for item in row.get("errors") or [])
        )
    }
    frozen = _frozen_ids(args.frozen_manifest)
    generated: list[dict[str, Any]] = []
    for source_candidate in source_candidates:
        record_id = str(source_candidate["record_id"])
        if record_id in succeeded or record_id in frozen or record_id not in timed_out:
            continue
        candidate = dict(source_candidate)
        candidate.update({
            "batch_id": args.batch_id,
            "lane": "wall_timeout_followup",
            "changes": [
                *(source_candidate.get("changes") or []),
                {
                    "kind": "verification_contract",
                    "operation": "increase_finite_wall_timeout",
                    "value_seconds": int(args.wall_timeout),
                    "semantic_change": False,
                },
            ],
            "campaign_metadata": {
                "source_batch": str(args.source_batch),
                "followup_reason": "pantograph_wall_timeout",
                "wall_timeout_seconds": int(args.wall_timeout),
            },
        })
        candidate.pop("candidate_hash", None)
        candidate["candidate_hash"] = canonical_hash(candidate)
        generated.append(candidate)
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, generated)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-wall-timeout-followup",
        "batch_id": args.batch_id,
        "source_batch": str(args.source_batch),
        "selected_rows": len(generated),
        "wall_timeout_seconds": int(args.wall_timeout),
        "overlap_with_frozen": 0,
        "candidate_manifest_sha256": sha256_file(manifest),
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def prepare_routed_resource_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Retry routed timeout rows with their original proof under finite limits.

    This avoids an intentionally redundant first compile: the canonical fail row
    already records the resource-limit failure.  Only clean, explicit `by`
    proofs are admitted, and the theorem statement is left unchanged.
    """
    routed = read_jsonl(args.routing_manifest)
    routed.sort(key=lambda row: (int(row.get("proof_chars") or 0), str(row["record_id"])))
    if args.limit:
        routed = routed[: args.limit]
    frozen = _frozen_ids(args.frozen_manifest)
    routed_ids = {str(row["record_id"]) for row in routed}
    overlap = routed_ids & frozen
    if overlap:
        raise ValueError(f"resource route overlaps frozen records: {sorted(overlap)[:5]}")

    indexed = IndexedFailRows(args.fail_file, args.index_file)
    candidates: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    try:
        for route in routed:
            record_id = str(route["record_id"])
            row = indexed.get_row(record_id)
            proof = str(row.get("proof") or "").strip()
            if not proof.startswith("by"):
                skipped["proof_not_by"] = skipped.get("proof_not_by", 0) + 1
                continue
            if _FORBIDDEN.search(proof):
                skipped["forbidden_proof"] = skipped.get("forbidden_proof", 0) + 1
                continue
            diagnostic = str(row.get("repair_error") or row.get("error_message") or "")
            options: list[tuple[str, int]] = []
            lowered = diagnostic.lower()
            if row.get("timed_out") or "heartbeat" in lowered or "timeout" in lowered or "timed out" in lowered:
                options.append(("maxHeartbeats", 1_000_000))
            if "recursion depth" in lowered:
                options.append(("maxRecDepth", 10_000))
            if not options:
                skipped["no_resource_marker"] = skipped.get("no_resource_marker", 0) + 1
                continue
            candidate = _candidate_from_row(
                row,
                batch_id=args.batch_id,
                lane="routed_resource_limit_retry",
                variants=[{"strategy": "original_proof_finite_resource_retry", "proof": proof}],
                metadata={
                    "routing_manifest": str(args.routing_manifest),
                    "resource_options": {option: limit for option, limit in options},
                },
            )
            candidate["source_body"] = _wrap_target_declaration_with_options(
                str(candidate["source_body"]), options
            )
            candidate["changes"] = [
                *(candidate.get("changes") or []),
                *(
                    {
                        "kind": "environment_limit",
                        "operation": f"set_option_{option}",
                        "value": limit,
                        "semantic_change": False,
                    }
                    for option, limit in options
                ),
            ]
            candidate.pop("candidate_hash", None)
            candidate["candidate_hash"] = canonical_hash(candidate)
            candidates.append(candidate)
    finally:
        indexed.close()

    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, candidates)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-routed-resource",
        "batch_id": args.batch_id,
        "routed_rows": len(routed),
        "selected_rows": len(candidates),
        "skipped": skipped,
        "candidate_manifest_sha256": sha256_file(manifest),
        "overlap_with_frozen": 0,
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def split_recovery_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze completed candidates and deterministically chunk the untouched remainder."""
    source_candidates = read_jsonl(args.source_batch / "candidate_manifest.jsonl")
    source_results = read_jsonl(args.source_batch / "verification_results.jsonl")
    frozen = _frozen_ids(args.frozen_manifest)
    if {str(row["record_id"]) for row in source_candidates} & frozen:
        raise RuntimeError("source recovery batch overlaps frozen records")
    results_by_hash: dict[str, list[dict[str, Any]]] = {}
    for result in source_results:
        results_by_hash.setdefault(str(result["candidate_hash"]), []).append(result)
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    completed_hashes: set[str] = set()
    for candidate in source_candidates:
        attempts = results_by_hash.get(str(candidate["candidate_hash"]), [])
        is_complete = any(row.get("success") for row in attempts) or (
            len(attempts) >= len(candidate.get("variants") or []) > 0
        )
        if is_complete:
            completed.append(candidate)
            completed_hashes.add(str(candidate["candidate_hash"]))
        else:
            pending.append(candidate)

    completed_dir = args.output_root / f"{args.batch_prefix}_verified_subset"
    if completed_dir.exists() and any(completed_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {completed_dir}")
    completed_dir.mkdir(parents=True, exist_ok=True)
    completed_manifest = completed_dir / "candidate_manifest.jsonl"
    completed_results = completed_dir / "verification_results.jsonl"
    write_jsonl(completed_manifest, completed)
    write_jsonl(
        completed_results,
        [row for row in source_results if str(row["candidate_hash"]) in completed_hashes],
    )
    completed_report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "split-recovery-batch-verified-subset",
        "batch_id": f"{args.batch_prefix}_verified_subset",
        "source_batch": str(args.source_batch.resolve()),
        "selected_rows": len(completed),
        "candidate_manifest_sha256": sha256_file(completed_manifest),
        "verification_results_sha256": sha256_file(completed_results),
        "overlap_with_frozen": 0,
        "source_batch_mutated": False,
        "fully_verified": True,
    }
    _json_dump(completed_dir / "prepare_report.json", completed_report)

    def priority(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        variants = candidate.get("variants") or []
        proof_chars = min((len(str(item.get("proof") or "")) for item in variants), default=10**9)
        metadata = candidate.get("campaign_metadata") or {}
        correct = int(metadata.get("upstream_correct_proof_count") or 0)
        total = int(metadata.get("upstream_proof_count") or 0)
        rate = correct / total if total else 0.0
        return (proof_chars, -rate, str(candidate["record_id"]))

    pending.sort(key=priority)
    chunk_dirs: list[Path] = []
    for offset in range(0, len(pending), args.chunk_size):
        ordinal = offset // args.chunk_size + 1
        batch_id = f"{args.batch_prefix}_pending_{ordinal:02d}"
        batch_dir = args.output_root / batch_id
        if batch_dir.exists() and any(batch_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
        batch_dir.mkdir(parents=True, exist_ok=True)
        rewritten: list[dict[str, Any]] = []
        for original in pending[offset : offset + args.chunk_size]:
            candidate = dict(original)
            candidate["batch_id"] = batch_id
            candidate["campaign_metadata"] = {
                **(original.get("campaign_metadata") or {}),
                "split_source_batch": str(args.source_batch.resolve()),
                "split_priority": "proof_chars_asc_then_upstream_rate_desc",
            }
            candidate.pop("candidate_hash", None)
            candidate["candidate_hash"] = canonical_hash(candidate)
            rewritten.append(candidate)
        manifest = batch_dir / "candidate_manifest.jsonl"
        write_jsonl(manifest, rewritten)
        prepare_report = {
            "schema_version": CAMPAIGN_SCHEMA,
            "command": "split-recovery-batch-pending-chunk",
            "batch_id": batch_id,
            "source_batch": str(args.source_batch.resolve()),
            "selected_rows": len(rewritten),
            "candidate_manifest_sha256": sha256_file(manifest),
            "overlap_with_frozen": 0,
            "source_batch_mutated": False,
            "fully_verified": False,
        }
        _json_dump(batch_dir / "prepare_report.json", prepare_report)
        chunk_dirs.append(batch_dir)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "split-recovery-batch",
        "source_batch": str(args.source_batch.resolve()),
        "source_candidates": len(source_candidates),
        "verified_subset_rows": len(completed),
        "pending_rows": len(pending),
        "chunk_size": args.chunk_size,
        "chunk_dirs": [str(path.resolve()) for path in chunk_dirs],
        "overlap_with_frozen": 0,
        "source_batch_mutated": False,
    }
    _json_dump(args.output_root / f"{args.batch_prefix}_split_report.json", report)
    return report


def prepare_active_manual_followup_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Apply reviewed exact text edits to proofs already repaired by a source batch."""
    edits = read_jsonl(args.edits_file)
    edit_by_id = {str(row.get("record_id") or ""): row for row in edits}
    if not edit_by_id or "" in edit_by_id or len(edit_by_id) != len(edits):
        raise ValueError("active manual edits must have unique nonempty record IDs")
    candidates = {
        str(row["record_id"]): row
        for row in read_jsonl(args.source_batch / "candidate_manifest.jsonl")
    }
    results = read_jsonl(args.source_batch / "verification_results.jsonl")
    succeeded = {str(row["record_id"]) for row in results if row.get("success")}
    frozen = _frozen_ids(args.frozen_manifest)
    generated: list[dict[str, Any]] = []
    for record_id, edit in edit_by_id.items():
        if record_id in succeeded:
            raise ValueError(f"manual followup targets an already-successful record: {record_id}")
        if record_id in frozen:
            raise ValueError(f"manual followup overlaps a frozen record: {record_id}")
        source_candidate = candidates.get(record_id)
        if source_candidate is None:
            raise ValueError(f"manual followup record absent from source batch: {record_id}")
        if str(edit.get("source_candidate_hash") or "") != str(source_candidate.get("candidate_hash") or ""):
            raise ValueError(f"source candidate hash mismatch for {record_id}")
        operations = edit.get("operations") or []
        if not operations:
            raise ValueError(f"manual followup has no operations for {record_id}")
        variants: list[dict[str, str]] = []
        for source_variant in source_candidate.get("variants") or []:
            proof = str(source_variant.get("proof") or "")
            repaired = proof
            for operation in operations:
                previous = str(operation.get("old") or "")
                current = str(operation.get("new") or "")
                expected_count = int(operation.get("expected_count", 1))
                actual_count = repaired.count(previous)
                if not previous or actual_count != expected_count:
                    raise ValueError(
                        f"manual operation count mismatch for {record_id}: "
                        f"expected {expected_count}, observed {actual_count}"
                    )
                repaired = repaired.replace(previous, current)
            repaired = repaired.strip()
            if not repaired.startswith("by") or _FORBIDDEN.search(repaired):
                raise ValueError(f"manual followup produced an invalid proof for {record_id}")
            variants.append({"strategy": "codex_manual_active_followup", "proof": repaired})
        candidate = dict(source_candidate)
        candidate.update({
            "batch_id": args.batch_id,
            "lane": "codex_manual_active_followup",
            "variants": variants,
            "campaign_metadata": {
                "source_batch": str(args.source_batch),
                "rationale": str(edit.get("rationale") or ""),
                "reviewer": "codex_manual_review",
            },
        })
        candidate.pop("candidate_hash", None)
        candidate["candidate_hash"] = canonical_hash(candidate)
        generated.append(candidate)
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, generated)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-active-manual-followup",
        "batch_id": args.batch_id,
        "source_batch": str(args.source_batch),
        "selected_rows": len(generated),
        "candidate_manifest_sha256": sha256_file(manifest),
        "edits_file_sha256": sha256_file(args.edits_file),
        "overlap_with_frozen": 0,
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def prepare_indexed_manual_batch(args: argparse.Namespace) -> dict[str, Any]:
    edits = read_jsonl(args.edits_file)
    record_ids = [str(item.get("record_id") or "") for item in edits]
    if not record_ids or any(not item for item in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("manual edit records must be nonempty and unique")
    frozen = _frozen_ids(args.frozen_manifest)
    overlap = set(record_ids) & frozen
    if overlap:
        raise ValueError(f"manual batch overlaps frozen records: {sorted(overlap)[:5]}")
    indexed = IndexedFailRows(args.fail_file, args.index_file)
    candidates: list[dict[str, Any]] = []
    try:
        for edit in edits:
            record_id = str(edit["record_id"])
            row = indexed.get_row(record_id)
            selected_field, selected_source = select_numina_source(row)
            if str(edit.get("original_record_hash") or "") != str(row.get("record_hash") or ""):
                raise ValueError(f"record hash mismatch for {record_id}")
            if str(edit.get("selected_source_sha256") or "") != _sha256_text(selected_source):
                raise ValueError(f"selected source hash mismatch for {record_id}")
            variants = [dict(item) for item in edit.get("proof_variants") or []]
            if not variants:
                proof = str(edit.get("replacement_proof") or "").strip()
                variants = [{"strategy": "codex_manual_proof", "proof": proof}]
            for variant in variants:
                proof = str(variant.get("proof") or "").strip()
                if not proof.startswith("by") or _FORBIDDEN.search(proof):
                    raise ValueError(f"invalid manual proof variant for {record_id}")
            candidates.append(_candidate_from_row(
                row,
                batch_id=args.batch_id,
                lane="codex_manual",
                variants=variants,
                metadata={
                    "reviewer": "codex_manual_review",
                    "rationale": str(edit.get("rationale") or ""),
                    "selected_source_field": selected_field,
                },
            ))
    finally:
        indexed.close()
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, candidates)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-indexed-manual",
        "batch_id": args.batch_id,
        "selected_rows": len(candidates),
        "candidate_manifest_sha256": sha256_file(manifest),
        "edits_file_sha256": sha256_file(args.edits_file),
        "temporary_index_fields_in_candidates": False,
        "master_dataset_mutated": False,
        "external_api_called": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def _parse_api_proof(content: str) -> str:
    stripped = content.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        json_match = re.search(r"```json\s*(.*?)```", stripped, re.S | re.I)
        lean_match = re.search(r"```(?:lean|lean4)?\s*(by\b.*?)```", stripped, re.S | re.I)
        complete_proof_field = re.search(
            r'"(?:proof|lean_proof)"\s*:\s*("(?:\\.|[^"\\])*")',
            stripped,
            re.S,
        )
        if json_match:
            value = json.loads(json_match.group(1))
        elif lean_match:
            value = {"proof": lean_match.group(1).strip()}
        elif complete_proof_field:
            value = {"proof": json.loads(complete_proof_field.group(1))}
        elif stripped.startswith("by"):
            value = {"proof": stripped}
        else:
            raise ValueError("DeepSeek response is neither valid JSON nor a standalone Lean proof")
    if not isinstance(value, Mapping):
        raise ValueError("DeepSeek response JSON must be an object")
    proof = str(value.get("proof") or value.get("lean_proof") or "").strip()
    if not proof.startswith("by") or _FORBIDDEN.search(proof):
        raise ValueError("DeepSeek proof must be a complete forbidden-token-free `by` proof")
    return proof


def _api_prompt(
    source_body: str,
    error_message: str,
    environment: Mapping[str, str],
    failed_proof: str = "",
) -> str:
    failed_attempt = (
        f"\nFailed proof attempt to repair minimally:\n```lean\n{failed_proof}\n```\n"
        if failed_proof.strip()
        else ""
    )
    return f"""Repair one Lean 4 theorem proof.

Frozen environment:
- Lean: {environment['lean_version']}
- mathlib commit: {environment['mathlib_commit']}
- environment hash: {environment['environment_hash']}

You must target exactly these Lean and mathlib commits. Every imported module,
declaration name, theorem name, tactic, and syntax form must exist in this
frozen environment; do not rely on APIs from another mathlib revision.

Rules:
1. Do not change the theorem statement, imports, binders, hypotheses, or conclusion.
2. Return one concise complete proof beginning with `by`.
3. Never use sorry, admit, axiom, or unsafe declarations.
4. Prefer a semantically meaningful proof over irrelevant tactics.
5. Output strict JSON only: {{"proof":"by\\n  ...","failure_assessment":"...","rationale":"..."}}.
6. Use bounded analysis: identify the decisive lemma or tactic, check it against
   the frozen environment, and emit JSON promptly. Once a plausible complete
   proof is found, do not explore many alternative proof architectures.

Current source:
```lean
{source_body}
```
{failed_attempt}

Current verifier diagnostic:
```text
{error_message}
```
"""


def _is_api_quota_error(error: BaseException | str) -> bool:
    message = str(error).casefold()
    signals = (
        "insufficient balance",
        "insufficient_balance",
        "insufficient quota",
        "insufficient_quota",
        "quota exceeded",
        "quota_exceeded",
        "recharge",
        "payment required",
        "402",
    )
    return any(signal in message for signal in signals)


def prepare_api_batch(args: argparse.Namespace) -> dict[str, Any]:
    from lean_prover.lean_training.expert_iteration.utils import environment_identity

    source_batch = args.source_batch
    candidates = read_jsonl(source_batch / "candidate_manifest.jsonl")
    results = read_jsonl(source_batch / "verification_results.jsonl")
    succeeded = {str(row["record_id"]) for row in results if row.get("success")}
    latest_failed_result: dict[str, dict[str, Any]] = {}
    for result in results:
        if not result.get("success"):
            latest_failed_result[str(result["record_id"])] = result
    failed_candidates = [row for row in candidates if str(row["record_id"]) not in succeeded]
    skipped_not_current_fail = 0
    if args.fail_file is not None or args.index_file is not None:
        if args.fail_file is None or args.index_file is None:
            raise ValueError("--fail-file and --index-file must be provided together")
        indexed = IndexedFailRows(args.fail_file, args.index_file)
        refreshed: list[dict[str, Any]] = []
        try:
            for old_candidate in failed_candidates:
                record_id = str(old_candidate["record_id"])
                try:
                    current_row = indexed.get_row(record_id)
                except KeyError:
                    skipped_not_current_fail += 1
                    continue
                refreshed_candidate = _candidate_from_row(
                    current_row,
                    batch_id=args.batch_id,
                    lane="deepseek_v4_flash_assisted",
                    variants=[],
                    metadata={
                        "source_batch": str(source_batch),
                        "refreshed_from_current_fail_index": True,
                    },
                )
                feedback = latest_failed_result.get(record_id)
                if feedback is not None:
                    refreshed_candidate["original_error_message"] = _result_error(feedback)
                    refreshed_candidate["previous_failed_proof"] = str(feedback.get("proof") or "")
                    refreshed_candidate["campaign_metadata"].update({
                        "feedback_source_batch": str(source_batch),
                        "feedback_variant_hash": str(feedback.get("variant_hash") or ""),
                        "feedback_error_type": str(feedback.get("error_type") or ""),
                    })
                    refreshed_candidate.pop("candidate_hash", None)
                    refreshed_candidate["candidate_hash"] = canonical_hash(refreshed_candidate)
                refreshed.append(refreshed_candidate)
        finally:
            indexed.close()
        failed_candidates = refreshed
    excluded_ids = _frozen_ids(args.exclude_manifest)
    excluded_by_manifest = sum(
        str(candidate["record_id"]) in excluded_ids for candidate in failed_candidates
    )
    failed_candidates = [
        candidate for candidate in failed_candidates
        if str(candidate["record_id"]) not in excluded_ids
    ]
    if args.limit:
        failed_candidates = failed_candidates[: args.limit]
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    identity = environment_identity(args.lean_project, ("Mathlib",))
    client = OpenAI(api_key=key, base_url=base_url, timeout=args.api_timeout)
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    candidate_journal = batch_dir / "candidate_journal.jsonl"
    observation_journal = batch_dir / "api_observations.journal.jsonl"
    write_jsonl(candidate_journal, [])
    write_jsonl(observation_journal, [])
    selection_manifest = batch_dir / "api_selection_manifest.jsonl"
    write_jsonl(selection_manifest, failed_candidates)
    quota_exhausted = threading.Event()

    def call(candidate: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        prompt = _api_prompt(
            str(candidate["source_body"]),
            str(candidate.get("original_error_message") or ""),
            identity,
            str(candidate.get("previous_failed_proof") or ""),
        )
        last_error = ""
        call_observations: list[dict[str, Any]] = []
        for attempt in (1, 2):
            if quota_exhausted.is_set():
                call_observations.append({
                    "record_id": candidate["record_id"],
                    "attempt": attempt,
                    "provider": "deepseek",
                    "model": model,
                    "status": "skipped_after_quota_exhaustion",
                    "thinking_mode": args.thinking_mode,
                    "reasoning_effort": args.reasoning_effort,
                    "prompt_sha256": _sha256_text(prompt),
                    "environment_hash": identity["environment_hash"],
                })
                return None, call_observations
            started = time.monotonic()
            try:
                request_max_tokens = args.max_tokens if attempt == 1 else args.retry_max_tokens
                request_thinking_mode = (
                    args.thinking_mode if attempt == 1 else args.retry_thinking_mode
                )
                extra_body: dict[str, Any] = {
                    "thinking": {"type": request_thinking_mode}
                }
                request: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a Lean 4 proof repair specialist."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": request_max_tokens,
                    "response_format": {"type": "json_object"},
                    "extra_body": extra_body,
                }
                if request_thinking_mode == "enabled":
                    request["reasoning_effort"] = args.reasoning_effort
                else:
                    request["temperature"] = 0.1
                response = client.chat.completions.create(**request)
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                reasoning = str(getattr(message, "reasoning_content", None) or "")
                usage = getattr(response, "usage", None)
                observation = {
                    "record_id": candidate["record_id"],
                    "attempt": attempt,
                    "provider": "deepseek",
                    "model": str(getattr(response, "model", None) or model),
                    "request_id": str(getattr(response, "id", None) or ""),
                    "finish_reason": str(getattr(choice, "finish_reason", None) or ""),
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "response_empty": not bool(content.strip()),
                    "response_sha256": _sha256_text(content),
                    "reasoning_chars": len(reasoning),
                    "reasoning_sha256": _sha256_text(reasoning),
                    "thinking_mode": request_thinking_mode,
                    "reasoning_effort": args.reasoning_effort,
                    "requested_max_tokens": request_max_tokens,
                    "prompt_sha256": _sha256_text(prompt),
                    "environment_hash": identity["environment_hash"],
                }
                if not content.strip():
                    last_error = "empty_response"
                    if attempt == 1:
                        call_observations.append({**observation, "status": "empty_response_retrying"})
                        continue
                    call_observations.append({**observation, "status": "empty_response_after_retry"})
                    return None, call_observations
                try:
                    proof = _parse_api_proof(content)
                except ValueError as error:
                    call_observations.append({
                        **observation,
                        "status": "invalid_response_format",
                        "error": f"{type(error).__name__}: {error}",
                        "response_preview": content[:4000],
                    })
                    return None, call_observations
                generated = dict(candidate)
                generated.update({
                    "batch_id": args.batch_id,
                    "lane": "deepseek_v4_flash_assisted",
                    "variants": [{"strategy": "deepseek_v4_flash", "proof": proof}],
                    "campaign_metadata": {
                        "source_batch": str(source_batch),
                        "provider": "deepseek",
                        "model": model,
                        "request_id": observation["request_id"],
                    },
                })
                generated.pop("candidate_hash", None)
                generated["candidate_hash"] = canonical_hash(generated)
                call_observations.append({**observation, "status": "candidate_created"})
                return generated, call_observations
            except Exception as error:  # observable, never accepted without Pantograph
                last_error = f"{type(error).__name__}: {error}"
                if _is_api_quota_error(error):
                    quota_exhausted.set()
                if "empty" in last_error.lower() and attempt == 1:
                    call_observations.append({
                        "record_id": candidate["record_id"], "attempt": attempt,
                        "provider": "deepseek", "model": model,
                        "status": "empty_response_retrying", "error": last_error,
                        "latency_ms": round((time.monotonic() - started) * 1000),
                        "thinking_mode": request_thinking_mode,
                        "reasoning_effort": args.reasoning_effort,
                        "requested_max_tokens": request_max_tokens,
                        "prompt_sha256": _sha256_text(prompt),
                        "environment_hash": identity["environment_hash"],
                    })
                    continue
                call_observations.append({
                    "record_id": candidate["record_id"], "attempt": attempt,
                    "provider": "deepseek", "model": model, "status": "api_error",
                    "error": last_error, "latency_ms": round((time.monotonic() - started) * 1000),
                    "thinking_mode": request_thinking_mode,
                    "reasoning_effort": args.reasoning_effort,
                    "requested_max_tokens": request_max_tokens,
                    "prompt_sha256": _sha256_text(prompt),
                    "environment_hash": identity["environment_hash"],
                })
                return None, call_observations
        call_observations.append({
            "record_id": candidate["record_id"], "status": "api_error", "error": last_error,
            "prompt_sha256": _sha256_text(prompt),
            "environment_hash": identity["environment_hash"],
        })
        return None, call_observations

    generated: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.api_workers) as executor:
        futures = [executor.submit(call, candidate) for candidate in failed_candidates]
        for future in as_completed(futures):
            candidate, call_observations = future.result()
            observations.extend(call_observations)
            for observation in call_observations:
                append_jsonl(observation_journal, observation)
            if candidate is not None:
                generated.append(candidate)
                append_jsonl(candidate_journal, candidate)
    generated.sort(key=lambda row: str(row["record_id"]))
    observations.sort(key=lambda row: (str(row.get("record_id")), int(row.get("attempt", 0))))
    manifest = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest, generated)
    write_jsonl(batch_dir / "api_observations.jsonl", observations)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "prepare-api",
        "batch_id": args.batch_id,
        "source_failed_rows": len(failed_candidates),
        "skipped_not_current_fail": skipped_not_current_fail,
        "excluded_by_manifest": excluded_by_manifest,
        "api_candidates_created": len(generated),
        "api_anomalies": len(failed_candidates) - len(generated),
        "provider": "deepseek",
        "model": model,
        "thinking_mode": args.thinking_mode,
        "retry_thinking_mode": args.retry_thinking_mode,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "retry_max_tokens": args.retry_max_tokens,
        "environment": identity,
        "api_selection_manifest_sha256": sha256_file(selection_manifest),
        "candidate_manifest_sha256": sha256_file(manifest),
        "candidate_journal_sha256": sha256_file(candidate_journal),
        "observation_journal_sha256": sha256_file(observation_journal),
        "external_api_called": True,
        "quota_exhausted": quota_exhausted.is_set(),
        "master_dataset_mutated": False,
    }
    _json_dump(batch_dir / "prepare_report.json", report)
    return report


def validate_invalid_manifest(
    path: Path | None, *, current_rows: Mapping[str, Mapping[str, Any]], frozen_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(path):
        record_id = str(item.get("record_id") or "")
        if not record_id or record_id in result:
            raise ValueError("invalid_statement manifest has missing or duplicate record_id")
        if record_id not in current_rows:
            raise ValueError(f"invalid_statement record not present in current fail: {record_id}")
        if record_id in frozen_ids:
            raise ValueError(f"invalid_statement overlaps frozen batch: {record_id}")
        row = current_rows[record_id]
        if str(item.get("original_record_hash") or "") != str(row.get("record_hash") or ""):
            raise ValueError(f"invalid_statement record hash mismatch: {record_id}")
        _, source = select_numina_source(row)
        if str(item.get("selected_source_sha256") or "") != _sha256_text(source):
            raise ValueError(f"invalid_statement source hash mismatch: {record_id}")
        if str(item.get("reason_category") or "") not in {
            "false_statement", "underspecified_statement", "inconsistent_encoding", "data_corruption"
        }:
            raise ValueError(f"invalid invalid_statement reason_category: {record_id}")
        result[record_id] = dict(item)
    return result


def _load_campaign_selections(batch_dirs: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    grouped: dict[str, dict[str, Any]] = {}
    result_hashes: list[str] = []
    for batch_dir in batch_dirs:
        candidates = read_jsonl(batch_dir / "candidate_manifest.jsonl")
        results = read_jsonl(batch_dir / "verification_results.jsonl")
        result_hashes.append(sha256_file(batch_dir / "verification_results.jsonl"))
        by_candidate: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            by_candidate.setdefault(str(result["candidate_hash"]), []).append(result)
        for candidate in candidates:
            attempts = by_candidate.get(str(candidate["candidate_hash"]), [])
            if not attempts:
                raise RuntimeError(f"batch is not fully verified: {batch_dir} {candidate['record_id']}")
            success = next((row for row in attempts if row.get("success")), None)
            record_id = str(candidate["record_id"])
            entry = grouped.setdefault(record_id, {"attempts": [], "successful": None})
            entry["attempts"].extend(attempts)
            if success is not None:
                entry["successful"] = {"candidate": candidate, "chosen": success}
    return grouped, result_hashes


def commit_campaign(args: argparse.Namespace) -> dict[str, Any]:
    frozen_ids = _frozen_ids(args.frozen_manifest)
    fail_rows = list(iter_jsonl(args.fail_file))
    fail_by_id = {str(row["record_id"]): row for row in fail_rows}
    if len(fail_by_id) != len(fail_rows):
        raise RuntimeError("current fail file contains duplicate record IDs")
    selections, result_hashes = _load_campaign_selections(args.batch_dir)
    invalid_new = validate_invalid_manifest(
        args.invalid_manifest, current_rows=fail_by_id, frozen_ids=frozen_ids
    )
    overlap = (set(selections) | set(invalid_new)) & frozen_ids
    if overlap:
        raise RuntimeError(f"campaign overlaps frozen records: {sorted(overlap)[:5]}")
    if set(selections) & set(invalid_new):
        raise RuntimeError("a record cannot be both a repair candidate and invalid_statement")

    success_rows = read_jsonl(args.success_file)
    success_ids = {str(row["record_id"]) for row in success_rows}
    invalid_rows = read_jsonl(args.invalid_file)
    invalid_ids = {str(row["record_id"]) for row in invalid_rows}
    promoted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    newly_invalid: list[dict[str, Any]] = []
    attempted_still_fail = 0
    for row in fail_rows:
        record_id = str(row["record_id"])
        invalid = invalid_new.get(record_id)
        if invalid is not None:
            updated = dict(row)
            updated["invalid_statement"] = {
                "schema_version": INVALID_SCHEMA,
                "reason_category": invalid["reason_category"],
                "reason_detail": str(invalid.get("reason_detail") or ""),
                "evidence": invalid.get("evidence") or [],
                "reviewer": str(invalid.get("reviewer") or "codex_manual_review"),
                "campaign_id": args.campaign_id,
            }
            updated["record_hash"] = dataset_record_hash(updated, source=NUMINA_SOURCE)
            newly_invalid.append(updated)
            continue
        selection = selections.get(record_id)
        if selection is None:
            remaining.append(row)
            continue
        successful = selection.get("successful")
        attempts = selection["attempts"]
        if successful is None:
            updated = dict(row)
            updated["repair"] = {
                "repair_version": REPAIR_VERSION,
                "campaign_id": args.campaign_id,
                "lanes": sorted({str(item.get("strategy") or "") for item in attempts}),
                "attempted_strategies": [str(item.get("strategy") or "") for item in attempts],
                "selected_strategy": None,
                "review_method": "staged_rule_api_manual",
            }
            updated["repair_error"] = "\n\n".join(
                f"{item.get('strategy')}: {_result_error(item)}" for item in attempts
            )
            remaining.append(updated)
            attempted_still_fail += 1
            continue
        candidate = successful["candidate"]
        chosen = successful["chosen"]
        lane = str(candidate.get("lane") or "")
        full_body = complete_source(str(candidate["source_body"]), str(chosen["proof"]))
        imports = candidate.get("imports") or []
        full_source = "\n".join([*(f"import {name}" for name in imports), full_body]).strip()
        updated = dict(row)
        updated.update({
            "formal_ground_truth": full_source,
            "ground_truth_type": "complete",
            "proof": chosen["proof"],
            "proof_present": True,
            "proof_source": (
                "deepseek_v4_flash_repair_pantograph_verified"
                if lane == "deepseek_v4_flash_assisted"
                else "codex_manual_repair"
                if lane.startswith("codex_manual")
                else "rule_assisted_repair_pantograph_verified"
            ),
            "verification_status": "success",
            "raw_verification_status": "success",
            "raw_verifier_success": True,
            "pantograph_verified": "success",
            "verification_backend": "pantograph",
            "timed_out": False,
            "lean_version": chosen["environment"]["lean_version"],
            "mathlib_commit": chosen["environment"]["mathlib_commit"],
            "environment_hash": chosen["environment"]["environment_hash"],
            "assembled_source_hash": chosen["assembled_source_hash"],
            "repair": {
                "repair_version": REPAIR_VERSION,
                "campaign_id": args.campaign_id,
                "lane": lane,
                "attempted_strategies": [str(item.get("strategy") or "") for item in attempts],
                "selected_strategy": chosen["strategy"],
                "review_method": "staged_rule_api_manual",
                "external_api_called": lane == "deepseek_v4_flash_assisted",
                "candidate_changes": candidate.get("changes") or [],
            },
        })
        updated.pop("repair_error", None)
        updated["record_hash"] = dataset_record_hash(updated, source=NUMINA_SOURCE)
        promoted.append(add_dataset_contract(updated, source=NUMINA_SOURCE, status=SUCCESS))

    promoted_ids = {str(row["record_id"]) for row in promoted}
    if promoted_ids & success_ids or set(invalid_new) & invalid_ids:
        raise RuntimeError("campaign output duplicates an existing success/invalid record")
    success_rows.extend(promoted)
    invalid_rows.extend(newly_invalid)
    write_jsonl(args.success_file, success_rows)
    write_jsonl(args.fail_file, remaining)
    write_jsonl(args.invalid_file, invalid_rows)
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "command": "commit",
        "campaign_id": args.campaign_id,
        "batch_dirs": [str(path) for path in args.batch_dir],
        "promoted_to_success": len(promoted),
        "attempted_but_still_fail": attempted_still_fail,
        "moved_to_invalid_statement": len(newly_invalid),
        "success_after": len(success_rows),
        "fail_after": len(remaining),
        "invalid_after": len(invalid_rows),
        "total_rows_after": len(success_rows) + len(remaining) + len(invalid_rows),
        "success_sha256": sha256_file(args.success_file),
        "fail_sha256": sha256_file(args.fail_file),
        "invalid_sha256": sha256_file(args.invalid_file),
        "verification_result_hashes": result_hashes,
        "temporary_index_fields_archived": False,
        "single_atomic_writeback_per_output": True,
    }
    _json_dump(args.report_file, report)
    return report


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fail-file", type=Path, default=root / "verified_data" / "numinamath_verified_fail.jsonl")
    common.add_argument("--index-file", type=Path, default=Path("outputs/numinamath_repair/campaign/fail_index.sqlite3"))

    build = sub.add_parser("build-index", parents=[common])
    build.add_argument("--report-file", type=Path, default=Path("outputs/numinamath_repair/campaign/fail_index_report.json"))

    rule = sub.add_parser("prepare-rule", parents=[common])
    rule.add_argument("--output-root", type=Path, default=Path("outputs/numinamath_repair"))
    rule.add_argument("--batch-id", required=True)
    rule.add_argument("--limit", type=int, default=100)
    rule.add_argument("--max-source-chars", type=int, default=650)
    rule.add_argument("--max-variants", type=int, default=4)
    rule.add_argument(
        "--easy-family",
        choices=("short_final_placeholder", "advanced_final_placeholder", "proof_repair", "other"),
        default="short_final_placeholder",
    )
    rule.add_argument(
        "--error-family",
        action="append",
        choices=("timeout", "unknown_identifier", "syntax_error", "type_mismatch", "tactic_error", "unsolved_goals", "elaboration_error", "other"),
        default=[],
    )
    rule.add_argument(
        "--strategy-profile",
        choices=("general", "compute", "algebra", "interval_cases"),
        default="general",
    )
    rule.add_argument("--frozen-manifest", type=Path, action="append", default=[])

    api = sub.add_parser("prepare-api")
    api.add_argument("--source-batch", type=Path, required=True)
    api.add_argument("--fail-file", type=Path)
    api.add_argument("--index-file", type=Path)
    api.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    api.add_argument("--output-root", type=Path, default=Path("outputs/numinamath_repair"))
    api.add_argument("--batch-id", required=True)
    api.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    api.add_argument("--limit", type=int, default=0)
    api.add_argument("--api-workers", type=int, default=4)
    api.add_argument("--api-timeout", type=float, default=120.0)
    api.add_argument("--max-tokens", type=int, default=16384)
    api.add_argument("--retry-max-tokens", type=int, default=32768)
    api.add_argument("--thinking-mode", choices=("disabled", "enabled"), default="enabled")
    api.add_argument("--retry-thinking-mode", choices=("disabled", "enabled"), default="disabled")
    api.add_argument("--reasoning-effort", choices=("high", "max"), default="high")

    raw_recovery = sub.add_parser("prepare-raw-recovery", parents=[common])
    raw_recovery.add_argument("--raw-parquet", type=Path, required=True)
    raw_recovery.add_argument(
        "--raw-proof-field",
        choices=("formal_proof", "formal_ground_truth"),
        default="formal_proof",
    )
    raw_recovery.add_argument("--output-root", type=Path, default=Path("outputs/numinamath_repair"))
    raw_recovery.add_argument("--batch-id", required=True)
    raw_recovery.add_argument("--limit", type=int, default=0)
    raw_recovery.add_argument("--source-batch-failures", type=Path)
    raw_recovery.add_argument("--allow-declaration-name-mismatch", action="store_true")
    raw_recovery.add_argument("--frozen-manifest", type=Path, action="append", default=[])

    followup = sub.add_parser("prepare-followup")
    followup.add_argument("--source-batch", type=Path, required=True)
    followup.add_argument("--output-root", type=Path, default=Path("outputs/numinamath_repair"))
    followup.add_argument("--batch-id", required=True)
    followup.add_argument("--diagnostic-family", choices=("free_variables", "noncomputable", "decidable_missing", "syntax_preamble", "unknown_constant"), required=True)
    followup.add_argument("--strategy-profile", choices=("context", "algebra"), required=True)
    followup.add_argument("--max-variants", type=int, default=6)
    followup.add_argument("--limit", type=int, default=0)

    interval_followup = sub.add_parser("prepare-interval-followup", parents=[common])
    interval_followup.add_argument("--source-batch", type=Path, required=True)
    interval_followup.add_argument(
        "--output-root", type=Path, default=Path("outputs/numinamath_repair")
    )
    interval_followup.add_argument("--batch-id", required=True)
    interval_followup.add_argument(
        "--frozen-manifest", type=Path, action="append", default=[]
    )

    compatibility_followup = sub.add_parser("prepare-compatibility-followup")
    compatibility_followup.add_argument("--source-batch", type=Path, required=True)
    compatibility_followup.add_argument(
        "--output-root", type=Path, default=Path("outputs/numinamath_repair")
    )
    compatibility_followup.add_argument("--batch-id", required=True)
    compatibility_followup.add_argument(
        "--frozen-manifest", type=Path, action="append", default=[]
    )

    resource_followup = sub.add_parser("prepare-resource-followup")
    resource_followup.add_argument("--source-batch", type=Path, required=True)
    resource_followup.add_argument(
        "--output-root", type=Path, default=Path("outputs/numinamath_repair")
    )
    resource_followup.add_argument("--batch-id", required=True)
    resource_followup.add_argument(
        "--frozen-manifest", type=Path, action="append", default=[]
    )

    wall_timeout_followup = sub.add_parser("prepare-wall-timeout-followup")
    wall_timeout_followup.add_argument("--source-batch", type=Path, required=True)
    wall_timeout_followup.add_argument(
        "--output-root", type=Path, default=Path("outputs/numinamath_repair")
    )
    wall_timeout_followup.add_argument("--batch-id", required=True)
    wall_timeout_followup.add_argument("--wall-timeout", type=int, default=120)
    wall_timeout_followup.add_argument(
        "--frozen-manifest", type=Path, action="append", default=[]
    )

    routed_resource = sub.add_parser("prepare-routed-resource", parents=[common])
    routed_resource.add_argument("--routing-manifest", type=Path, required=True)
    routed_resource.add_argument(
        "--output-root", type=Path, default=Path("outputs/numinamath_repair")
    )
    routed_resource.add_argument("--batch-id", required=True)
    routed_resource.add_argument("--limit", type=int, default=0)
    routed_resource.add_argument(
        "--frozen-manifest", type=Path, action="append", default=[]
    )

    split_batch = sub.add_parser("split-recovery-batch")
    split_batch.add_argument("--source-batch", type=Path, required=True)
    split_batch.add_argument("--output-root", type=Path, default=Path("outputs/numinamath_repair"))
    split_batch.add_argument("--batch-prefix", required=True)
    split_batch.add_argument("--chunk-size", type=int, default=400)
    split_batch.add_argument("--frozen-manifest", type=Path, action="append", default=[])

    active_manual = sub.add_parser("prepare-active-manual-followup")
    active_manual.add_argument("--source-batch", type=Path, required=True)
    active_manual.add_argument("--edits-file", type=Path, required=True)
    active_manual.add_argument(
        "--output-root", type=Path, default=Path("outputs/numinamath_repair")
    )
    active_manual.add_argument("--batch-id", required=True)
    active_manual.add_argument("--frozen-manifest", type=Path, action="append", default=[])

    manual = sub.add_parser("prepare-indexed-manual", parents=[common])
    manual.add_argument("--edits-file", type=Path, required=True)
    manual.add_argument("--output-root", type=Path, default=Path("outputs/numinamath_repair"))
    manual.add_argument("--batch-id", required=True)
    manual.add_argument("--frozen-manifest", type=Path, action="append", default=[])

    commit = sub.add_parser("commit")
    commit.add_argument("--campaign-id", required=True)
    commit.add_argument("--batch-dir", type=Path, action="append", required=True)
    commit.add_argument("--fail-file", type=Path, default=root / "verified_data" / "numinamath_verified_fail.jsonl")
    commit.add_argument("--success-file", type=Path, default=root / "verified_data" / "numinamath_verified_success.jsonl")
    commit.add_argument("--invalid-file", type=Path, default=root / "verified_data" / "numinamath_invalid_statement.jsonl")
    commit.add_argument("--invalid-manifest", type=Path)
    commit.add_argument("--frozen-manifest", type=Path, action="append", default=[])
    commit.add_argument("--report-file", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-index":
        report = build_index(fail_file=args.fail_file, index_file=args.index_file, report_file=args.report_file)
    elif args.command == "prepare-rule":
        report = prepare_rule_batch(args)
    elif args.command == "prepare-api":
        report = prepare_api_batch(args)
    elif args.command == "prepare-raw-recovery":
        report = prepare_raw_recovery_batch(args)
    elif args.command == "prepare-followup":
        report = prepare_followup_batch(args)
    elif args.command == "prepare-interval-followup":
        report = prepare_interval_cases_followup_batch(args)
    elif args.command == "prepare-compatibility-followup":
        report = prepare_compatibility_followup_batch(args)
    elif args.command == "prepare-resource-followup":
        report = prepare_resource_limit_followup_batch(args)
    elif args.command == "prepare-wall-timeout-followup":
        report = prepare_wall_timeout_followup_batch(args)
    elif args.command == "prepare-routed-resource":
        report = prepare_routed_resource_batch(args)
    elif args.command == "split-recovery-batch":
        report = split_recovery_batch(args)
    elif args.command == "prepare-active-manual-followup":
        report = prepare_active_manual_followup_batch(args)
    elif args.command == "prepare-indexed-manual":
        report = prepare_indexed_manual_batch(args)
    else:
        report = commit_campaign(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
