"""Adapter for LeanDojo-v2 traces produced from the current mathlib commit.

This module is deliberately separate from the historical LeanDojo Benchmark
adapter.  LeanDojo-v2 is imported lazily so the project's normal training and
verification processes do not acquire the tracing environment as a dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping

from .leandojo_context_builder import extract_active_source_context


CURRENT_MATHLIB_COMMIT = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"
CURRENT_LEAN_VERSION = "4.29.1"
CURRENT_LEAN_COMMIT = "f72c35b3f637c8c6571d353742168ab66cc22c00"
CURRENT_ENVIRONMENT_HASH = (
    "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
)
LEANDOJO_V2_COMMIT = "936ea0dd32ffc2305fe7855db7fbc5bc8557dcf6"
ADAPTER_VERSION = "leandojo-v2-current-mathlib-v3"

_DECLARATION_KIND = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?P<kind>theorem|lemma|example)\b"
)
_FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom|unsafe)\b")
_TEST_PATH = re.compile(r"(?:^|/)(?:test|tests|testdata|fixtures)(?:/|$)", re.I)


def sha256_text(value: str) -> str:
    """Return a stable SHA-256 digest for UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_proof_tokens(proof: str) -> int:
    """Estimate Lean proof tokens without importing a model tokenizer.

    The estimate separates identifiers, numerals, and punctuation.  It is used
    only for auditable sampling statistics, not model truncation.
    """

    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_']*|\d+|[^\s]", proof))


def assemble_source_prefix(
    source: str,
    *,
    end_line: int,
    end_column: int,
    closing_commands: Iterable[str],
) -> str:
    """Keep the exact source through a declaration and close active scopes."""

    lines = source.splitlines(keepends=True)
    if end_line <= 0 or end_line > len(lines):
        raise ValueError(f"invalid declaration end line: {end_line}")
    prefix = "".join(lines[: end_line - 1])
    final_line = lines[end_line - 1].removesuffix("\n")
    if end_column <= 0 or end_column > len(final_line) + 1:
        raise ValueError(f"invalid declaration end column: {end_column}")
    prefix += final_line[: end_column - 1]
    closings = "\n".join(
        command.strip() for command in closing_commands if command.strip()
    )
    return "\n".join(part for part in (prefix.rstrip(), closings) if part).rstrip() + "\n"


def _span(start: Any, end: Any) -> dict[str, int]:
    return {
        "start_line": int(start.line_nb),
        "start_column": int(start.column_nb),
        "end_line": int(end.line_nb),
        "end_column": int(end.column_nb),
    }


def _repo_identity() -> Any:
    """Construct a read-only LeanDojo repo identity without cloning a repo.

    ``LeanGitRepo`` normally populates its metadata by cloning/caching a local
    repository.  The traced files already live in a verified isolated clone, so
    the adapter supplies the immutable identity fields directly and never calls
    that network/cache-producing constructor.
    """

    from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo, RepoType

    repo = object.__new__(LeanGitRepo)
    object.__setattr__(
        repo, "url", "https://github.com/leanprover-community/mathlib4"
    )
    object.__setattr__(repo, "commit", CURRENT_MATHLIB_COMMIT)
    object.__setattr__(repo, "lean_version", f"v{CURRENT_LEAN_VERSION}")
    object.__setattr__(repo, "repo_type", RepoType.GITHUB)
    return repo


def _proof_style(proof: str, tactic_trace: list[dict[str, Any]]) -> str:
    stripped = proof.lstrip()
    if tactic_trace and re.match(r"^(?::=\s*)?(?:by|begin)\b", stripped):
        return "tactic"
    if tactic_trace:
        return "mixed"
    return "term"


def _premises(theorem: Any, source_file: str) -> list[dict[str, Any]]:
    from lean_dojo_v2.lean_dojo.data_extraction.ast import IdentNode

    rows: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...]] = set()

    def collect(node: Any, _: list[Any]) -> None:
        full_name = getattr(node, "full_name", None)
        if not full_name:
            return
        def_path = getattr(node, "def_path", None)
        def_start = getattr(node, "def_start", None)
        def_end = getattr(node, "def_end", None)
        usage_start = getattr(node, "start", None)
        usage_end = getattr(node, "end", None)
        if def_start is None or def_end is None:
            definition_span: dict[str, int] = {}
        else:
            definition_span = _span(def_start, def_end)
        usage_span = (
            _span(usage_start, usage_end)
            if usage_start is not None and usage_end is not None
            else {}
        )
        normalized_path = str(def_path or "").replace("\\", "/")
        signature = (
            str(full_name),
            normalized_path,
            tuple(usage_span.values()),
        )
        if signature in signatures:
            return
        signatures.add(signature)
        rows.append(
            {
                "qualified_name": str(full_name),
                "source_file": normalized_path,
                "definition_span": definition_span,
                "usage_span": usage_span,
                "is_same_file": normalized_path == source_file,
            }
        )

    theorem.ast.traverse_preorder(collect, IdentNode)
    return rows


def _tactic_trace(theorem: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, tactic in enumerate(theorem.get_traced_tactics()):
        _, provenances = tactic.get_annotated_tactic()
        rows.append(
            {
                "step_index": index,
                "state_before": tactic.state_before,
                "tactic": tactic.tactic,
                "state_after": tactic.state_after,
                "premises_used": [
                    {
                        "qualified_name": str(item.get("full_name") or ""),
                        "source_file": str(item.get("def_path") or "").replace(
                            "\\", "/"
                        ),
                        "definition_span": {
                            "start_line": int((item.get("def_pos") or [0, 0])[0]),
                            "start_column": int(
                                (item.get("def_pos") or [0, 0])[1]
                            ),
                            "end_line": int(
                                (item.get("def_end_pos") or [0, 0])[0]
                            ),
                            "end_column": int(
                                (item.get("def_end_pos") or [0, 0])[1]
                            ),
                        },
                    }
                    for item in provenances
                ],
            }
        )
    return rows


def _trace_paths(repo_root: Path, source_file: str) -> tuple[Path, Path]:
    relative = Path(source_file)
    ast_path = repo_root / ".lake/build/ir" / relative.with_suffix(".ast.json")
    dep_path = repo_root / ".lake/build/ir" / relative.with_suffix(".dep_paths")
    return ast_path, dep_path


def iter_current_mathlib_records(
    *,
    repo_root: str | Path,
    source_files: Iterable[str],
) -> Iterator[dict[str, Any]]:
    """Yield declaration-centric records from current-commit trace artifacts."""

    from lean_dojo_v2.lean_dojo.data_extraction.traced_data import TracedFile

    repo_root = Path(repo_root).resolve(strict=True)
    repo = _repo_identity()
    traced_repo = SimpleNamespace(repo=repo, dependencies={})
    for source_file in source_files:
        normalized_file = str(source_file).replace("\\", "/")
        ast_path, dep_path = _trace_paths(repo_root, normalized_file)
        traced_file = TracedFile.from_traced_file(repo_root, ast_path, repo)
        traced_file.traced_repo = traced_repo
        source = traced_file.abs_path.read_text(encoding="utf-8")
        dependencies = [
            line.strip()
            for line in dep_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for theorem in traced_file.get_traced_theorems():
            proof_start, proof_end = theorem.locate_proof()
            statement = theorem.get_theorem_statement().strip()
            raw_proof_source = traced_file.lean_file[
                proof_start:proof_end
            ].strip()
            proof = theorem.get_tactic_proof() or raw_proof_source
            declaration_source = traced_file.lean_file[
                theorem.start : theorem.end
            ].strip()
            kind_match = _DECLARATION_KIND.match(declaration_source)
            kind = kind_match.group("kind") if kind_match else "theorem"
            context = extract_active_source_context(
                source,
                declaration_line=theorem.start.line_nb,
                source_file=normalized_file,
                source_commit=CURRENT_MATHLIB_COMMIT,
            )
            assembled_source = assemble_source_prefix(
                source,
                end_line=theorem.end.line_nb,
                end_column=theorem.end.column_nb,
                closing_commands=context.closing_commands,
            )
            tactic_trace = _tactic_trace(theorem)
            premises = _premises(theorem, normalized_file)
            source_span = _span(theorem.start, theorem.end)
            trace_identity = {
                "ast_sha256": sha256_file(ast_path),
                "dep_paths_sha256": sha256_file(dep_path),
                "qualified_name": theorem.theorem.full_name,
                "source_span": source_span,
            }
            record_id = "ldv2_" + sha256_text(
                f"{CURRENT_MATHLIB_COMMIT}:{normalized_file}:"
                f"{theorem.theorem.full_name}:{source_span}"
            )[:24]
            yield {
                "id": record_id,
                "source": "leandojo_v2_current_mathlib",
                "repository": "mathlib4",
                "repository_commit": CURRENT_MATHLIB_COMMIT,
                "lean_version": CURRENT_LEAN_VERSION,
                "lean_commit": CURRENT_LEAN_COMMIT,
                "leandojo_v2_commit": LEANDOJO_V2_COMMIT,
                "source_file": normalized_file,
                "qualified_name": theorem.theorem.full_name,
                "declaration_kind": kind,
                "source_span": source_span,
                "proof_span": _span(proof_start, proof_end),
                "imports": list(context.imports),
                "file_dependencies": dependencies,
                "namespace_stack": list(context.namespaces),
                "open_namespaces": list(context.open_namespaces),
                "open_scopes": list(context.scopes),
                "section_context": list(context.sections),
                "variables": list(context.variables),
                "hypotheses": [],
                "local_instances": list(context.local_instances),
                "local_notations": list(context.local_notations),
                "local_attributes": list(context.local_attributes),
                "statement": statement,
                "proof": proof,
                "declaration_source": declaration_source,
                "proof_style": _proof_style(proof, tactic_trace),
                "tactic_trace": tactic_trace,
                "premises": premises,
                "environment_hash": CURRENT_ENVIRONMENT_HASH,
                "trace_hash": sha256_text(
                    json.dumps(trace_identity, sort_keys=True)
                ),
                "assembled_source_hash": sha256_text(assembled_source),
                "pantograph_verified": False,
                "metadata": {
                    "adapter_version": ADAPTER_VERSION,
                    "source_sha256": sha256_text(source),
                    "ast_path": str(ast_path),
                    "ast_sha256": trace_identity["ast_sha256"],
                    "dep_paths_path": str(dep_path),
                    "dep_paths_sha256": trace_identity["dep_paths_sha256"],
                    "proof_token_estimate": estimate_proof_tokens(proof),
                    "proof_token_estimator": "lean-lexeme-regex-v1",
                    "raw_proof_source": raw_proof_source,
                    "context_recovery": context.summary(),
                    "assembled_source": assembled_source,
                    "assembly_mode": "source_prefix_through_declaration",
                    "field_availability": {
                        "hypotheses": (
                            "not separately emitted by LeanDojo-v2; retained "
                            "in statement, section variables, and tactic states"
                        ),
                        "local_instances": "recovered from exact source prefix",
                        "local_notations": "recovered from exact source prefix",
                        "local_attributes": "recovered from exact source prefix",
                    },
                },
            }


def quarantine_reason(
    record: Mapping[str, Any],
    *,
    max_proof_tokens: int = 1024,
) -> str | None:
    """Return the first deterministic candidate-filter rejection reason."""

    statement = str(record.get("statement") or "").strip()
    proof = str(record.get("proof") or "").strip()
    declaration = str(record.get("declaration_source") or "").strip()
    source_file = str(record.get("source_file") or "").replace("\\", "/")
    if not statement:
        return "empty_statement"
    if not proof:
        return "empty_proof"
    if _FORBIDDEN.search("\n".join((statement, proof, declaration))):
        return "forbidden_token"
    if _TEST_PATH.search(source_file):
        return "generated_or_test_only_path"
    if record.get("repository_commit") != CURRENT_MATHLIB_COMMIT:
        return "mathlib_commit_mismatch"
    if record.get("lean_version") != CURRENT_LEAN_VERSION:
        return "lean_version_mismatch"
    if not record.get("qualified_name") or not record.get("source_span"):
        return "unlocatable_declaration"
    token_count = int(
        (record.get("metadata") or {}).get("proof_token_estimate")
        or estimate_proof_tokens(proof)
    )
    if token_count > max_proof_tokens:
        return "proof_over_sampling_token_limit"
    trace = list(record.get("tactic_trace") or [])
    if record.get("proof_style") in {"tactic", "mixed"}:
        if not trace:
            return "missing_tactic_trace"
        for left, right in zip(trace, trace[1:]):
            if str(left.get("state_after") or "").strip() != str(
                right.get("state_before") or ""
            ).strip():
                return "non_contiguous_tactic_trace"
        final_state = str(trace[-1].get("state_after") or "").strip().lower()
        if final_state not in {"no goals", "no goals to be solved"}:
            return "tactic_trace_has_open_goal"
    return None


def validate_current_record(record: Mapping[str, Any]) -> list[str]:
    """Return schema/alignment errors for one adapted declaration record."""

    required = {
        "id",
        "source",
        "repository",
        "repository_commit",
        "lean_version",
        "leandojo_v2_commit",
        "source_file",
        "qualified_name",
        "declaration_kind",
        "source_span",
        "imports",
        "file_dependencies",
        "namespace_stack",
        "open_namespaces",
        "open_scopes",
        "section_context",
        "variables",
        "hypotheses",
        "local_instances",
        "local_notations",
        "local_attributes",
        "statement",
        "proof",
        "declaration_source",
        "proof_style",
        "tactic_trace",
        "premises",
        "environment_hash",
        "trace_hash",
        "assembled_source_hash",
        "pantograph_verified",
        "metadata",
    }
    errors = [f"missing:{key}" for key in sorted(required - record.keys())]
    if record.get("repository_commit") != CURRENT_MATHLIB_COMMIT:
        errors.append("repository_commit_mismatch")
    if record.get("lean_version") != CURRENT_LEAN_VERSION:
        errors.append("lean_version_mismatch")
    if record.get("leandojo_v2_commit") != LEANDOJO_V2_COMMIT:
        errors.append("leandojo_v2_commit_mismatch")
    if record.get("environment_hash") != CURRENT_ENVIRONMENT_HASH:
        errors.append("environment_hash_mismatch")
    if record.get("declaration_kind") not in {"theorem", "lemma", "example"}:
        errors.append("invalid_declaration_kind")
    for index, step in enumerate(record.get("tactic_trace") or []):
        missing = {
            "step_index",
            "state_before",
            "tactic",
            "state_after",
            "premises_used",
        } - step.keys()
        errors.extend(f"tactic_trace[{index}].missing:{key}" for key in sorted(missing))
    for index, premise in enumerate(record.get("premises") or []):
        missing = {
            "qualified_name",
            "source_file",
            "definition_span",
            "usage_span",
            "is_same_file",
        } - premise.keys()
        errors.extend(f"premises[{index}].missing:{key}" for key in sorted(missing))
    metadata = record.get("metadata") or {}
    assembled = metadata.get("assembled_source")
    if assembled and sha256_text(str(assembled)) != record.get(
        "assembled_source_hash"
    ):
        errors.append("assembled_source_hash_mismatch")
    return errors
