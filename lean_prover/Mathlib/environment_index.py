"""Merged declaration index for pinned Lean Core, Std, and Mathlib sources."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sqlite3
import subprocess

from lean_prover.Planner.schemas import LeanEnvironmentIdentity

from .index import (
    DeclarationCandidate,
    LocalMathlibDeclarationIndex,
    MathlibDeclaration,
    extract_declarations,
)


def _lean_prefix(project_path: Path) -> Path:
    completed = subprocess.run(
        ["lake", "env", "lean", "--print-prefix"],
        cwd=project_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    prefix = Path(completed.stdout.strip()).resolve()
    if not prefix.is_dir():
        raise FileNotFoundError(f"Lean toolchain prefix not found: {prefix}")
    return prefix


class LeanCoreStdDeclarationIndex(LocalMathlibDeclarationIndex):
    """SQLite companion index for the exact pinned Init and Std sources."""

    SCHEMA_VERSION = "lean_core_std_declaration_index_v1"

    @classmethod
    def for_project(
        cls,
        *,
        project_path: str | Path,
        environment: LeanEnvironmentIdentity,
        cache_root: str | Path | None = None,
    ) -> "LeanCoreStdDeclarationIndex":
        project = Path(project_path).resolve()
        prefix = _lean_prefix(project)
        source_root = prefix / "src" / "lean"
        init_root = source_root / "Init"
        std_root = source_root / "Std"
        if not init_root.is_dir() or not std_root.is_dir():
            raise FileNotFoundError(
                "Pinned Lean toolchain does not contain Init/Std sources: "
                f"{source_root}"
            )
        cache = (
            Path(cache_root)
            if cache_root
            else project.parent / ".cache" / "lean_core_std_index"
        )
        key = f"{environment.lean_commit}_{environment.environment_hash}"
        database = cache / f"{key}.sqlite3"
        index = cls(database)
        index.ensure_core_std_built(
            source_root=source_root,
            init_root=init_root,
            std_root=std_root,
            environment=environment,
        )
        return index

    def ensure_core_std_built(
        self,
        *,
        source_root: Path,
        init_root: Path,
        std_root: Path,
        environment: LeanEnvironmentIdentity,
    ) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        if self.database_path.is_file():
            with self._connect() as connection:
                try:
                    metadata = dict(
                        connection.execute("SELECT key, value FROM metadata")
                    )
                except sqlite3.DatabaseError:
                    metadata = {}
                if (
                    metadata.get("schema_version") == self.SCHEMA_VERSION
                    and metadata.get("lean_commit") == environment.lean_commit
                    and metadata.get("environment_hash")
                    == environment.environment_hash
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
                "lean_commit": environment.lean_commit,
                "environment_hash": environment.environment_hash,
                "lean_version": environment.lean_version,
                "source_root": str(source_root),
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            count = 0
            for library_root, origin in (
                (init_root, "lean_core"),
                (std_root, "lean_std"),
            ):
                for source_path in sorted(library_root.rglob("*.lean")):
                    rows: list[dict[str, object]] = []
                    for declaration in extract_declarations(
                        source_path,
                        library_root,
                        declaration_origin=origin,
                    ):
                        row = asdict(declaration)
                        # Init is available without an explicit import.  Std
                        # declarations retain their precise import module.
                        if origin == "lean_core":
                            row["required_import"] = ""
                        rows.append(row)
                    if not rows:
                        continue
                    connection.executemany(
                        """INSERT OR IGNORE INTO declarations VALUES (
                        :full_name, :short_name, :declaration_kind,
                        :type_signature, :namespace, :module,
                        :required_import, :source_path, :source_line,
                        :first_explicit_parameter_type, :domain,
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


class EnvironmentDeclarationIndex:
    """Search Core/Std and Mathlib as one pinned declaration environment."""

    def __init__(
        self,
        *,
        mathlib: LocalMathlibDeclarationIndex,
        core_std: LeanCoreStdDeclarationIndex,
    ) -> None:
        self.mathlib = mathlib
        self.core_std = core_std

    @classmethod
    def for_project(
        cls,
        *,
        project_path: str | Path,
        environment: LeanEnvironmentIdentity,
    ) -> "EnvironmentDeclarationIndex":
        return cls(
            mathlib=LocalMathlibDeclarationIndex.for_project(
                project_path=project_path,
                environment=environment,
            ),
            core_std=LeanCoreStdDeclarationIndex.for_project(
                project_path=project_path,
                environment=environment,
            ),
        )

    def search(
        self,
        query: str,
        *,
        parameter_type: str | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[DeclarationCandidate]:
        merged: dict[tuple[str, str, int], DeclarationCandidate] = {}
        component_limit = max(10, limit * 2)
        for component in (self.core_std, self.mathlib):
            for candidate in component.search(
                query,
                parameter_type=parameter_type,
                domain=domain,
                limit=component_limit,
            ):
                declaration = candidate.declaration
                key = (
                    declaration.full_name,
                    declaration.module,
                    declaration.source_line,
                )
                previous = merged.get(key)
                if previous is None or candidate.score > previous.score:
                    merged[key] = candidate
        results = sorted(
            merged.values(),
            key=lambda candidate: (
                candidate.score,
                candidate.declaration.full_name == query,
                candidate.declaration.domain == domain,
            ),
            reverse=True,
        )
        return results[:limit]

    def metadata(self) -> dict[str, object]:
        return {
            "mathlib": self.mathlib.metadata(),
            "lean_core_std": self.core_std.metadata(),
            "mathlib_database": str(self.mathlib.database_path),
            "lean_core_std_database": str(self.core_std.database_path),
        }


__all__ = [
    "EnvironmentDeclarationIndex",
    "LeanCoreStdDeclarationIndex",
]
