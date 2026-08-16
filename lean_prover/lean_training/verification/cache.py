"""Disk-backed verification cache scoped by Lean and assembler identity."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Any


def make_context_cache_key(
    *,
    statement_hash: str,
    proof_hash: str,
    imports_hash: str,
    namespace_hash: str,
    scope_hash: str,
    variable_context_hash: str,
    context_hash: str,
    assembled_source_hash: str,
    environment_hash: str,
    verifier_version: str,
    assembler_version: str,
    normalization_version: str,
) -> str:
    """Bind a verification cache entry to every compilation-relevant input."""

    payload = {
        "statement_hash": statement_hash,
        "proof_hash": proof_hash,
        "imports_hash": imports_hash,
        "namespace_hash": namespace_hash,
        "scope_hash": scope_hash,
        "variable_context_hash": variable_context_hash,
        "context_hash": context_hash,
        "assembled_source_hash": assembled_source_hash,
        "environment_hash": environment_hash,
        "verifier_version": verifier_version,
        "assembler_version": assembler_version,
        "normalization_version": normalization_version,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class VerificationCache:
    """Query one proof at a time without materializing the cache in RAM."""

    def __init__(self, path: str | Path) -> None:
        requested = Path(path)
        self.path = requested if requested.suffix == ".sqlite" else requested.with_suffix(".sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS verification_cache (
                cache_key TEXT PRIMARY KEY,
                proof_hash TEXT NOT NULL,
                statement_hash TEXT NOT NULL,
                environment_hash TEXT NOT NULL,
                assembler_version TEXT NOT NULL,
                normalization_version TEXT NOT NULL,
                status TEXT NOT NULL,
                error_type TEXT,
                source_path TEXT,
                verification_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS verification_environment_idx "
            "ON verification_cache(environment_hash, assembler_version, normalization_version)"
        )
        self.connection.commit()

    def get(self, cache_key: str, *, environment_hash: str, assembler_version: str, normalization_version: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT verification_json FROM verification_cache
            WHERE cache_key = ? AND environment_hash = ?
              AND assembler_version = ? AND normalization_version = ?
            """,
            (cache_key, environment_hash, assembler_version, normalization_version),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, *, cache_key: str, proof_hash: str, statement_hash: str, environment_hash: str, assembler_version: str, normalization_version: str, status: str, error_type: str | None, source_path: str | None, verification: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO verification_cache (
                cache_key, proof_hash, statement_hash, environment_hash,
                assembler_version, normalization_version, status, error_type,
                source_path, verification_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(cache_key) DO UPDATE SET
                proof_hash=excluded.proof_hash,
                statement_hash=excluded.statement_hash,
                environment_hash=excluded.environment_hash,
                assembler_version=excluded.assembler_version,
                normalization_version=excluded.normalization_version,
                status=excluded.status,
                error_type=excluded.error_type,
                source_path=excluded.source_path,
                verification_json=excluded.verification_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                cache_key, proof_hash, statement_hash, environment_hash,
                assembler_version, normalization_version, status, error_type,
                source_path, json.dumps(verification, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM verification_cache").fetchone()[0])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "VerificationCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
