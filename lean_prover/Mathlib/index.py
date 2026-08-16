"""Commit-pinned offline declaration index built from local Mathlib sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable

from lean_prover.Planner.schemas import LeanEnvironmentIdentity


DECLARATION_KINDS = (
    "def",
    "theorem",
    "lemma",
    "abbrev",
    "structure",
    "class",
    "inductive",
    "instance",
    "opaque",
)

_DECLARATION_RE = re.compile(
    r"^\s*(?:(?:@\[[^\n]*?\]\s*)*)"
    r"(?P<modifiers>(?:(?:noncomputable|protected|private|nonrec|unsafe)\s+)*)"
    r"(?P<kind>def|theorem|lemma|abbrev|structure|class|inductive|instance|opaque)"
    r"\s+(?P<name>[^\s(:={]+)",
)
_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_'.]*)\s*$")
_SECTION_RE = re.compile(
    r"^\s*section(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?\s*$"
)
_END_RE = re.compile(r"^\s*end(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?\s*$")


@dataclass(frozen=True)
class MathlibDeclaration:
    full_name: str
    short_name: str
    declaration_kind: str
    type_signature: str
    namespace: str
    module: str
    required_import: str
    source_path: str
    source_line: int
    first_explicit_parameter_type: str | None
    domain: str
    declaration_origin: str = "mathlib"


@dataclass(frozen=True)
class DeclarationCandidate:
    declaration: MathlibDeclaration
    match_stage: str
    score: float

    def prompt_record(self) -> dict[str, object]:
        return {
            **asdict(self.declaration),
            "match_stage": self.match_stage,
            "score": round(self.score, 6),
        }


def _strip_comments_preserve_lines(source: str) -> str:
    """Mask nested block and line comments while preserving line numbers."""

    output: list[str] = []
    index = 0
    block_depth = 0
    in_string = False
    while index < len(source):
        pair = source[index:index + 2]
        char = source[index]
        if block_depth:
            if pair == "/-":
                block_depth += 1
                output.extend("  ")
                index += 2
            elif pair == "-/":
                block_depth -= 1
                output.extend("  ")
                index += 2
            else:
                output.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if not in_string and pair == "/-":
            block_depth = 1
            output.extend("  ")
            index += 2
            continue
        if not in_string and pair == "--":
            newline = source.find("\n", index)
            if newline < 0:
                output.extend(" " * (len(source) - index))
                break
            output.extend(" " * (newline - index))
            output.append("\n")
            index = newline + 1
            continue
        if char == '"' and (index == 0 or source[index - 1] != "\\"):
            in_string = not in_string
        output.append(char)
        index += 1
    return "".join(output)


def _module_from_path(mathlib_root: Path, source_path: Path) -> str:
    relative = source_path.relative_to(mathlib_root.parent).with_suffix("")
    return ".".join(relative.parts)


def _domain_for(module: str, signature: str = "") -> str:
    value = f"{module} {signature}".casefold()
    rules = (
        ("number_theory", ("numbertheory", "prime", "divisor", "factorization", "zmod")),
        ("algebra", ("algebra", "ring", "group", "monoid", "polynomial")),
        ("analysis", ("analysis", "topology", "continuous", "deriv", "integral")),
        ("geometry", ("geometry", "euclidean", "affine")),
        ("combinatorics", ("combinatorics", "finset", "multiset", "graph")),
        ("probability", ("probability", "measure", "random")),
        ("logic", ("logic", "modeltheory", "settheory")),
        ("data_nat", ("data.nat", " nat", "ℕ")),
    )
    for domain, needles in rules:
        if any(needle in value for needle in needles):
            return domain
    return "general"


def infer_math_domain(text: str) -> str:
    return _domain_for(text.replace(" ", "."), text)


def _balanced_group(text: str, start: int, opening: str, closing: str) -> tuple[str, int] | None:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    return None


def _first_explicit_parameter_type(signature: str) -> str | None:
    index = 0
    while index < len(signature):
        if signature[index] == "(":
            group = _balanced_group(signature, index, "(", ")")
            if group is None:
                break
            content, index = group
            if ":" in content:
                parameter_type = content.split(":", 1)[1].strip()
                parameter_type = parameter_type.split(":=", 1)[0].strip()
                return " ".join(parameter_type.split()) or None
            continue
        index += 1
    # Declarations without named explicit binders can still expose A -> B.
    type_part = signature.split(":", 1)[1].strip() if ":" in signature else ""
    paren = brace = bracket = 0
    index = 0
    while index < len(type_part):
        char = type_part[index]
        if char == "(": paren += 1
        elif char == ")": paren = max(0, paren - 1)
        elif char == "{": brace += 1
        elif char == "}": brace = max(0, brace - 1)
        elif char == "[": bracket += 1
        elif char == "]": bracket = max(0, bracket - 1)
        if paren == brace == bracket == 0:
            arrow_length = 1 if char == "→" else (
                2 if type_part.startswith("->", index) else 0
            )
            if arrow_length:
                return " ".join(type_part[:index].split()) or None
        index += 1
    return None


def _header_until_body(lines: list[str], start: int) -> tuple[str, int]:
    collected: list[str] = []
    paren = brace = bracket = 0
    for index in range(start, min(len(lines), start + 80)):
        line = lines[index]
        # Equation-compiler branches are declaration bodies even when the
        # source omits an explicit ``:=`` (common for defs and theorems).
        if index > start and line.lstrip().startswith("|"):
            break
        collected.append(line.strip())
        in_string = False
        position = 0
        while position < len(line):
            char = line[position]
            if char == '"' and (position == 0 or line[position - 1] != "\\"):
                in_string = not in_string
            if in_string:
                position += 1
                continue
            if char == "(": paren += 1
            elif char == ")": paren = max(0, paren - 1)
            elif char == "{": brace += 1
            elif char == "}": brace = max(0, brace - 1)
            elif char == "[": bracket += 1
            elif char == "]": bracket = max(0, bracket - 1)
            if paren == brace == bracket == 0:
                if line.startswith(":=", position):
                    joined = " ".join(part for part in collected if part)
                    return joined.rsplit(":=", 1)[0].strip(), index
                if re.match(r"\bwhere\b", line[position:]):
                    joined = " ".join(part for part in collected if part)
                    return re.split(r"\bwhere\b", joined, maxsplit=1)[0].strip(), index
            position += 1
        # A declaration header followed by a body on the next indented line may
        # omit := only for structures/inductives. Stop once the next declaration
        # begins rather than swallowing it.
        if index > start and _DECLARATION_RE.match(lines[index + 1] if index + 1 < len(lines) else ""):
            break
    return " ".join(part for part in collected if part).strip(), start


def extract_declarations(
    source_path: Path,
    mathlib_root: Path,
    *,
    declaration_origin: str = "mathlib",
    required_import: str | None = None,
) -> Iterable[MathlibDeclaration]:
    source = source_path.read_text(encoding="utf-8", errors="replace")
    lines = _strip_comments_preserve_lines(source).splitlines()
    # Keep sections in the scope stack as well.  An unlabelled ``end`` closes
    # the innermost scope, which is not necessarily a namespace.
    scope_stack: list[tuple[str, str | None, list[str]]] = []
    module = _module_from_path(mathlib_root, source_path)
    for index, line in enumerate(lines):
        namespace_match = _NAMESPACE_RE.match(line)
        if namespace_match:
            parts = namespace_match.group(1).split(".")
            scope_stack.append(("namespace", parts[-1], parts))
            continue
        section_match = _SECTION_RE.match(line)
        if section_match:
            scope_stack.append(("section", section_match.group(1), []))
            continue
        end_match = _END_RE.match(line)
        if end_match and scope_stack:
            label = end_match.group(1)
            if label:
                while scope_stack:
                    _, scope_label, _ = scope_stack.pop()
                    if scope_label == label:
                        break
            else:
                scope_stack.pop()
            continue
        match = _DECLARATION_RE.match(line)
        if not match or "private" in match.group("modifiers").split():
            continue
        raw_header, _ = _header_until_body(lines, index)
        # Remove attributes/modifiers/kind/name from the collected header.
        header_match = _DECLARATION_RE.match(raw_header)
        if header_match is None:
            continue
        name = header_match.group("name").strip()
        valid_name = all(
            re.fullmatch(r"[^\W\d][\w']*", part, flags=re.UNICODE)
            for part in name.split(".")
        )
        if name == "_" or name.startswith("«") or not valid_name:
            continue
        signature_tail = raw_header[header_match.end():].strip()
        namespace_parts = [
            part
            for scope_kind, _, parts in scope_stack
            if scope_kind == "namespace"
            for part in parts
        ]
        namespace = ".".join(namespace_parts)
        full_name = (
            name
            if "." in name and not namespace
            else ".".join(part for part in (namespace, name) if part)
        )
        declaration_namespace = (
            full_name.rsplit(".", 1)[0] if "." in full_name else ""
        )
        type_signature = f"{full_name} {signature_tail}".strip()
        yield MathlibDeclaration(
            full_name=full_name,
            short_name=name.rsplit(".", 1)[-1],
            declaration_kind=match.group("kind"),
            type_signature=type_signature,
            namespace=declaration_namespace,
            module=module,
            required_import=(
                module if required_import is None else required_import
            ),
            source_path=str(source_path),
            source_line=index + 1,
            first_explicit_parameter_type=_first_explicit_parameter_type(signature_tail),
            domain=_domain_for(module, signature_tail),
            declaration_origin=declaration_origin,
        )


class LocalMathlibDeclarationIndex:
    SCHEMA_VERSION = "mathlib_declaration_index_v4"

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @classmethod
    def for_project(
        cls,
        *,
        project_path: str | Path,
        environment: LeanEnvironmentIdentity,
        cache_root: str | Path | None = None,
    ) -> "LocalMathlibDeclarationIndex":
        project = Path(project_path).resolve()
        mathlib_root = project / ".lake" / "packages" / "mathlib" / "Mathlib"
        if not mathlib_root.is_dir():
            raise FileNotFoundError(f"Local Mathlib source tree not found: {mathlib_root}")
        cache = Path(cache_root) if cache_root else project.parent / ".cache" / "mathlib_index"
        key = f"{environment.mathlib_commit}_{environment.environment_hash}"
        database = cache / f"{key}.sqlite3"
        index = cls(database)
        index.ensure_built(mathlib_root=mathlib_root, environment=environment)
        return index

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def ensure_built(
        self,
        *,
        mathlib_root: Path,
        environment: LeanEnvironmentIdentity,
    ) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        if self.database_path.is_file():
            with self._connect() as connection:
                try:
                    meta = dict(connection.execute("SELECT key, value FROM metadata"))
                except sqlite3.DatabaseError:
                    meta = {}
                if (
                    meta.get("schema_version") == self.SCHEMA_VERSION
                    and meta.get("mathlib_commit") == environment.mathlib_commit
                    and meta.get("environment_hash") == environment.environment_hash
                ):
                    return
        temporary = self.database_path.with_suffix(".tmp.sqlite3")
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE declarations (
                    full_name TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    declaration_kind TEXT NOT NULL,
                    type_signature TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    module TEXT NOT NULL,
                    required_import TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_line INTEGER NOT NULL,
                    first_explicit_parameter_type TEXT,
                    domain TEXT NOT NULL,
                    declaration_origin TEXT NOT NULL,
                    PRIMARY KEY (full_name, module, source_line)
                );
                CREATE INDEX declarations_full_name ON declarations(full_name);
                CREATE INDEX declarations_short_name ON declarations(short_name);
                CREATE INDEX declarations_namespace ON declarations(namespace);
                CREATE INDEX declarations_domain ON declarations(domain);
            """)
            metadata = {
                "schema_version": self.SCHEMA_VERSION,
                "mathlib_commit": environment.mathlib_commit,
                "environment_hash": environment.environment_hash,
                "lean_version": environment.lean_version,
                "source_root": str(mathlib_root),
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
            )
            count = 0
            for source_path in sorted(mathlib_root.rglob("*.lean")):
                rows = [asdict(item) for item in extract_declarations(source_path, mathlib_root)]
                if not rows:
                    continue
                connection.executemany(
                    """INSERT OR IGNORE INTO declarations VALUES (
                    :full_name, :short_name, :declaration_kind, :type_signature,
                    :namespace, :module, :required_import, :source_path,
                    :source_line, :first_explicit_parameter_type, :domain,
                    :declaration_origin)""",
                    rows,
                )
                count += len(rows)
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ("declaration_count", str(count)),
            )
            connection.commit()
        finally:
            connection.close()
        temporary.replace(self.database_path)

    @staticmethod
    def _row(row: sqlite3.Row) -> MathlibDeclaration:
        return MathlibDeclaration(**dict(row))

    @staticmethod
    def _normalize_type(value: str | None) -> set[str]:
        if not value:
            return set()
        aliases = {"nat": "ℕ", "int": "ℤ", "real": "ℝ", "rat": "ℚ"}
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_'.]*|[ℕℤℚℝℂ]", value.casefold())
        return {aliases.get(token, token) for token in tokens}

    def search(
        self,
        query: str,
        *,
        parameter_type: str | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[DeclarationCandidate]:
        query = query.strip().strip("`'")
        if not query or limit < 1:
            return []
        query_short = query.rsplit(".", 1)[-1]
        query_namespace = query.rsplit(".", 1)[0] if "." in query else ""
        with self._connect() as connection:
            exact_rows = connection.execute(
                "SELECT * FROM declarations WHERE full_name = ? OR short_name = ? LIMIT 200",
                (query, query_short),
            ).fetchall()
            suffix_rows = connection.execute(
                "SELECT * FROM declarations WHERE full_name LIKE ? LIMIT 500",
                (f"%.{query_short}",),
            ).fetchall()
            raw_terms = [
                part
                for part in re.split(
                    r"[_'\W]+|(?<=[a-z0-9])(?=[A-Z])",
                    query_short,
                )
                if len(part) >= 3
            ]
            legacy_markers = {
                "old", "obsolete", "deprecated", "legacy", "removed",
            }
            semantic_terms = [
                term for term in raw_terms
                if term.casefold() not in legacy_markers
            ]
            search_terms = [query_short]
            if semantic_terms:
                search_terms.extend(semantic_terms)
                search_terms.append("_".join(semantic_terms))
                if len(semantic_terms) > 1:
                    search_terms.append("_".join(semantic_terms[-2:]))
            fuzzy_by_key: dict[tuple[str, str, int], sqlite3.Row] = {}
            for term in dict.fromkeys(search_terms):
                prefix = term[: max(1, min(4, len(term)))]
                rows = connection.execute(
                    "SELECT * FROM declarations "
                    "WHERE short_name LIKE ? OR full_name LIKE ? LIMIT 3000",
                    (f"%{prefix}%", f"%{term}%"),
                ).fetchall()
                for row in rows:
                    fuzzy_by_key[
                        (row["full_name"], row["module"], row["source_line"])
                    ] = row
            fuzzy_pool = list(fuzzy_by_key.values())
            if not fuzzy_pool:
                fuzzy_pool = connection.execute(
                    "SELECT * FROM declarations LIMIT 30000"
                ).fetchall()

        staged: dict[tuple[str, str, int], tuple[MathlibDeclaration, str, float]] = {}
        for row in exact_rows:
            declaration = self._row(row)
            staged[(declaration.full_name, declaration.module, declaration.source_line)] = (
                declaration,
                "exact",
                4.5 if declaration.full_name == query else 4.0,
            )
        for row in suffix_rows:
            declaration = self._row(row)
            staged.setdefault(
                (declaration.full_name, declaration.module, declaration.source_line),
                (declaration, "namespace_suffix", 3.0),
            )
        query_variants = [
            query_short.casefold(),
            *(
                "_".join(semantic_terms[index:]).casefold()
                for index in range(len(semantic_terms))
            ),
        ]
        fuzzy_ranked = sorted(
            [
                (
                    max(
                        SequenceMatcher(
                            None,
                            variant,
                            str(row["short_name"]).casefold(),
                        ).ratio()
                        for variant in query_variants
                    ),
                    row,
                )
                for row in fuzzy_pool
            ],
            key=lambda item: item[0],
        )[-200:]
        for similarity, row in fuzzy_ranked:
            if similarity < 0.35:
                continue
            declaration = self._row(row)
            staged.setdefault(
                (declaration.full_name, declaration.module, declaration.source_line),
                (declaration, "fuzzy", 1.0 + similarity),
            )

        parameter_tokens = self._normalize_type(parameter_type)
        candidates: list[DeclarationCandidate] = []
        for declaration, stage, score in staged.values():
            declaration_tokens = self._normalize_type(
                declaration.first_explicit_parameter_type
            )
            if parameter_tokens:
                overlap = parameter_tokens & declaration_tokens
                if overlap:
                    score += 0.75 + 0.1 * len(overlap)
                elif declaration_tokens:
                    score -= 0.6
            if domain and declaration.domain == domain:
                score += 0.5
            elif domain and declaration.domain == "general":
                score += 0.05
            semantic_short = "_".join(semantic_terms).casefold()
            if (
                semantic_short
                and declaration.short_name.casefold() == semantic_short
            ):
                score += 0.65
            if query_namespace and declaration.namespace == query_namespace:
                score += 0.35
            candidates.append(
                DeclarationCandidate(
                    declaration=declaration,
                    match_stage=stage,
                    score=score,
                )
            )
        candidates.sort(
            key=lambda item: (
                item.score,
                item.declaration.domain == domain,
                -len(item.declaration.full_name),
            ),
            reverse=True,
        )
        return candidates[:limit]

    def metadata(self) -> dict[str, str]:
        with self._connect() as connection:
            return dict(connection.execute("SELECT key, value FROM metadata"))


__all__ = [
    "DeclarationCandidate",
    "LocalMathlibDeclarationIndex",
    "MathlibDeclaration",
    "extract_declarations",
    "infer_math_domain",
]
