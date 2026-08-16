"""Source-file context recovery for LeanDojo Benchmark records."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .context import (
    LeanSourceContext,
    apply_context_to_record,
    parse_variable_command,
)
from .contracts import DataState, LeanDataRecord
from .preparation import ProofFormat, split_lean_statement_and_proof, statement_hash


LEANDOJO_SOURCE_CONTEXT_VERSION = "leandojo-source-context-v2"

_CONTEXT_PREFIXES = (
    "open ",
    "open scoped ",
    "variable ",
    "variables ",
    "universe ",
    "universes ",
    "include ",
    "omit ",
    "local notation ",
    "local infix",
    "local prefix",
    "local postfix",
    "local macro ",
    "local syntax ",
    "local attribute ",
    "attribute [local",
    "set_option ",
    "noncomputable section",
)
_IMPORT = re.compile(
    r"^\s*(?:(?:public|private|protected|meta)\s+)*"
    r"import\s+([A-Za-z0-9_.'/-]+)"
)
_NAMESPACE = re.compile(r"^\s*namespace\s+([A-Za-z0-9_'.]+)\s*$")
_SECTION = re.compile(
    r"^\s*(?:@\[[^\]]+\]\s*)?"
    r"(?:(?:public|private|protected|meta)\s+)*"
    r"section(?:\s+([A-Za-z0-9_'.]+))?\s*$"
)
_NONCOMPUTABLE_SECTION = re.compile(
    r"^\s*noncomputable\s+section(?:\s+([A-Za-z0-9_'.]+))?\s*$"
)
_END = re.compile(r"^\s*end(?:\s+([A-Za-z0-9_'.]+))?\s*$")


@dataclass
class _ScopeFrame:
    kind: str
    name: str
    opening: str
    commands: list[str] = field(default_factory=list)


class GitSourceProvider:
    """Read historical Lean sources from already-present local Git objects."""

    def __init__(self, *, lean_project: str | Path) -> None:
        self.lean_project = Path(lean_project).resolve()
        self.mathlib_repo = self.lean_project / ".lake" / "packages" / "mathlib"
        self._manifest_cache: dict[str, dict[str, str]] = {}

    def read_source(self, *, commit: str, source_file: str) -> tuple[str, dict[str, Any]]:
        normalized = source_file.replace("\\", "/")
        if normalized.startswith("Mathlib/"):
            return self._git_show(self.mathlib_repo, commit, normalized), {
                "repository": str(self.mathlib_repo),
                "repository_commit": commit,
                "repository_path": normalized,
            }
        package_match = re.match(r"^\.lake/packages/([^/]+)/(.*)$", normalized)
        if package_match:
            package, relative = package_match.groups()
            package_repo = self.lean_project / ".lake" / "packages" / package
            package_commit = self._package_revisions(commit).get(package)
            if not package_commit:
                raise FileNotFoundError(
                    f"package {package!r} has no revision in mathlib {commit}"
                )
            return self._git_show(package_repo, package_commit, relative), {
                "repository": str(package_repo),
                "repository_commit": package_commit,
                "repository_path": relative,
                "parent_mathlib_commit": commit,
            }
        raise FileNotFoundError(f"unsupported LeanDojo source path: {source_file}")

    def _package_revisions(self, mathlib_commit: str) -> dict[str, str]:
        cached = self._manifest_cache.get(mathlib_commit)
        if cached is not None:
            return cached
        payload = self._git_show(
            self.mathlib_repo,
            mathlib_commit,
            "lake-manifest.json",
        )
        manifest = json.loads(payload)
        revisions = {
            str(package.get("name")): str(package.get("rev"))
            for package in manifest.get("packages", ())
            if package.get("name") and package.get("rev")
        }
        self._manifest_cache[mathlib_commit] = revisions
        return revisions

    @staticmethod
    def _git_show(repository: Path, commit: str, path: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), "show", f"{commit}:{path}"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise FileNotFoundError(
                completed.stderr.strip()
                or f"{commit}:{path} is unavailable in {repository}"
            )
        return completed.stdout


def build_leandojo_source_context(
    record: LeanDataRecord,
    *,
    source_provider: GitSourceProvider,
    environment_hash: str,
) -> LeanDataRecord:
    """Recover the exact source declaration header and its active lexical context."""

    source_file = record.source_file or ""
    source_commit = record.source_commit or ""
    source_line = (
        record.source_span.start_line if record.source_span is not None else None
    )
    errors: list[str] = []
    try:
        if not source_file or not source_commit or source_line is None:
            raise FileNotFoundError(
                "record lacks source_file, source_commit, or declaration line"
            )
        source, provenance = source_provider.read_source(
            commit=source_commit,
            source_file=source_file,
        )
        context = extract_active_source_context(
            source,
            declaration_line=source_line,
            source_file=source_file,
            source_commit=source_commit,
        )
        source_statement = _source_declaration_header(record)
        if not source_statement:
            raise ValueError("corpus declaration has no parseable declaration header")
        source_proof = _raw_trace_proof(record) or record.proof
        updated = record.model_copy(
            update={
                "statement": source_statement,
                "statement_id": f"stmt_{statement_hash(source_statement)}",
                "proof": source_proof,
                "proof_format": ProofFormat.FULL_PROOF,
                "raw_source_context": json.dumps(
                    context.summary(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "metadata": {
                    **record.metadata,
                    "source_repository": provenance,
                    "proof_state_statement": record.statement,
                    "source_declaration_header": source_statement,
                    "source_declaration_recovered": True,
                },
            }
        )
        updated = apply_context_to_record(
            updated,
            context,
            environment_hash=environment_hash,
        )
        return updated.model_copy(
            update={
                "context_recovery_version": LEANDOJO_SOURCE_CONTEXT_VERSION,
                "metadata": {
                    **updated.metadata,
                    "source_context_required": True,
                    "proof_state_materialized_as_explicit_binders": False,
                },
            }
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
        context = LeanSourceContext(
            imports=tuple(record.imports),
            namespaces=tuple(
                part for part in (record.namespace or "").split(".") if part
            ),
            source_file=source_file or None,
            source_commit=source_commit or None,
            source_line=source_line,
            context_recovered=False,
            recovery_method="source_file_lookup_failed",
            recovery_errors=tuple(errors),
        )
        updated = apply_context_to_record(
            record,
            context,
            environment_hash=environment_hash,
        )
        return updated.model_copy(
            update={
                "data_state": DataState.QUARANTINED,
                "proof_verified": False,
                "pantograph_verified": False,
                "verification_status": "quarantined",
                "verification_error_type": "missing_context",
                "verification_error_message": "; ".join(errors),
                "context_recovery_version": LEANDOJO_SOURCE_CONTEXT_VERSION,
            }
        )


def extract_active_source_context(
    source: str,
    *,
    declaration_line: int,
    source_file: str,
    source_commit: str,
) -> LeanSourceContext:
    """Extract source commands active immediately before a 1-based line."""

    if declaration_line <= 0:
        raise ValueError("declaration_line must be positive")
    lines = source.splitlines()
    if declaration_line > len(lines) + 1:
        raise ValueError(
            f"declaration line {declaration_line} exceeds {len(lines)} source lines"
        )
    prefix = lines[: declaration_line - 1]
    sanitized = _mask_block_comments(prefix)
    imports: list[str] = []
    root = _ScopeFrame(kind="root", name="", opening="")
    frames: list[_ScopeFrame] = [root]
    pending_command: list[str] = []

    def flush_pending() -> None:
        if pending_command:
            frames[-1].commands.append("\n".join(pending_command).rstrip())
            pending_command.clear()

    for original, visible in zip(prefix, sanitized, strict=True):
        stripped = visible.strip()
        scope_visible = visible.split("--", 1)[0]
        if not stripped or stripped.startswith("--"):
            continue
        import_match = _IMPORT.match(scope_visible)
        if import_match:
            flush_pending()
            imports.append(import_match.group(1))
            continue
        namespace_match = _NAMESPACE.match(scope_visible)
        if namespace_match:
            flush_pending()
            name = namespace_match.group(1)
            frames.append(_ScopeFrame("namespace", name, f"namespace {name}"))
            continue
        section_match = _SECTION.match(scope_visible)
        if section_match:
            flush_pending()
            name = section_match.group(1) or ""
            opening = f"section {name}".rstrip()
            frames.append(_ScopeFrame("section", name, opening))
            continue
        noncomputable_section_match = _NONCOMPUTABLE_SECTION.match(
            scope_visible
        )
        if noncomputable_section_match:
            flush_pending()
            name = noncomputable_section_match.group(1) or ""
            opening = f"noncomputable section {name}".rstrip()
            frames.append(_ScopeFrame("section", name, opening))
            continue
        end_match = _END.match(scope_visible)
        if end_match and len(frames) > 1:
            flush_pending()
            frames.pop()
            continue
        if original[:1].isspace() and pending_command:
            pending_command.append(original)
            continue
        flush_pending()
        if _is_context_command(stripped):
            pending_command.append(original)
    flush_pending()

    active_commands: list[str] = []
    closing_commands: list[str] = []
    for frame in frames:
        if frame.kind != "root":
            active_commands.append(frame.opening)
        active_commands.extend(
            command
            for command in frame.commands
            if _command_is_replayable(command)
        )
    for frame in reversed(frames[1:]):
        closing_commands.append(
            f"end {frame.name}".rstrip()
            if frame.name
            else "end"
        )

    namespaces = tuple(
        frame.name for frame in frames if frame.kind == "namespace" and frame.name
    )
    sections = tuple(
        frame.name for frame in frames if frame.kind == "section" and frame.name
    )
    all_commands = [command for frame in frames for command in frame.commands]
    open_namespaces: list[str] = []
    scopes: list[str] = []
    variables: list[dict[str, Any]] = []
    variable_commands: list[str] = []
    local_instances: list[str] = []
    local_notations: list[str] = []
    local_attributes: list[str] = []
    option_commands: list[str] = []
    for command in all_commands:
        compact = " ".join(line.strip() for line in command.splitlines())
        if compact.startswith("open scoped "):
            scopes.extend(compact.removeprefix("open scoped ").split())
        elif compact.startswith("open "):
            open_namespaces.extend(compact.removeprefix("open ").split())
        elif compact.startswith(("variable ", "variables ")):
            parsed, instances = parse_variable_command(command)
            variables.extend(parsed)
            local_instances.extend(instances)
            variable_commands.append(command)
        elif compact.startswith(
            (
                "local notation ",
                "local infix",
                "local prefix",
                "local postfix",
                "local macro ",
                "local syntax ",
            )
        ):
            local_notations.append(command)
        elif compact.startswith(("local attribute ", "attribute [local")):
            local_attributes.append(command)
        elif compact.startswith("set_option "):
            option_commands.append(command)

    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return LeanSourceContext(
        imports=tuple(dict.fromkeys(imports)),
        namespaces=namespaces,
        open_namespaces=tuple(dict.fromkeys(open_namespaces)),
        scopes=tuple(dict.fromkeys(scopes)),
        variables=tuple(variables),
        variable_commands=tuple(variable_commands),
        local_instances=tuple(local_instances),
        local_notations=tuple(local_notations),
        local_attributes=tuple(local_attributes),
        option_commands=tuple(option_commands),
        sections=sections,
        active_commands=tuple(active_commands),
        closing_commands=tuple(closing_commands),
        source_file=source_file,
        source_commit=source_commit,
        source_line=declaration_line,
        context_recovered=True,
        recovery_method="historical_git_source_and_declaration_location",
        recovery_status="full",
        recovery_sources=("source_file", "declaration_location"),
        source_hash=source_hash,
    )


def _source_declaration_header(record: LeanDataRecord) -> str:
    declaration = str(record.raw_declaration or "").strip()
    if not declaration:
        return ""
    statement, _ = split_lean_statement_and_proof(declaration)
    return statement.strip()


def _raw_trace_proof(record: LeanDataRecord) -> str:
    try:
        trace = json.loads(record.raw_source_context or "[]")
    except json.JSONDecodeError:
        return ""
    if not isinstance(trace, list):
        return ""
    lines = ["by"]
    for step in trace:
        if not isinstance(step, dict):
            continue
        tactic = str(step.get("tactic") or "").strip()
        if tactic:
            lines.extend(
                f"  {line}" if line.strip() else ""
                for line in tactic.splitlines()
            )
    return "\n".join(lines).rstrip() if len(lines) > 1 else ""


def _is_context_command(stripped: str) -> bool:
    return stripped.startswith(_CONTEXT_PREFIXES)


def _command_is_replayable(command: str) -> bool:
    """Keep semantic options while omitting obsolete, diagnostics-only linters."""

    compact = " ".join(line.strip() for line in command.splitlines())
    if compact.startswith("set_option linter."):
        return False
    # Command-scoped prefixes such as ``variable (x) in`` apply only to the
    # immediately following source declaration.  If that declaration is before
    # our target, the prefix is no longer active at the target location.
    return not compact.endswith(" in")


def _mask_block_comments(lines: Iterable[str]) -> list[str]:
    """Mask nested block comments while preserving line and column counts."""

    depth = 0
    output: list[str] = []
    for line in lines:
        chars = list(line)
        index = 0
        while index < len(chars):
            pair = "".join(chars[index : index + 2])
            if pair == "/-":
                depth += 1
                chars[index] = chars[index + 1] = " "
                index += 2
                continue
            if pair == "-/" and depth:
                depth -= 1
                chars[index] = chars[index + 1] = " "
                index += 2
                continue
            if depth:
                chars[index] = " "
            index += 1
        output.append("".join(chars))
    return output
