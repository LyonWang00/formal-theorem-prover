#!/usr/bin/env python3
"""Freeze the audited 4,803-row LeanDojo three-tier difficulty pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {"easy": 1831, "medium": 2363, "difficult": 609}
SOURCE_NAMES = {
    "easy": "ld_easy_manifest.jsonl",
    "medium": "ld_medium_manifest.jsonl",
    "difficult": "ld_difficult_manifest.jsonl",
}
SUPPORT_FILES = (
    "classification_report.md",
    "classification_statistics.json",
    "difficulty_audit.json",
    "excluded_manifest.jsonl",
    "theorem_group_aliases.jsonl",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/leandojo_difficulty_classification"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ld_difficulty_classification_freeze/three_tier_v1"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    source = args.source if args.source.is_absolute() else project / args.source
    output = args.output if args.output.is_absolute() else project / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen archive: {output}")

    audit = read_json(source / "difficulty_audit.json")
    stats = read_json(source / "classification_statistics.json")
    if audit.get("status") != "PASSED" or stats.get("audit_status") != "PASSED":
        raise RuntimeError("source difficulty audit is not PASSED")
    if audit.get("official_classified_rows") != sum(EXPECTED_COUNTS.values()):
        raise RuntimeError("official classified row count drifted")

    rows_by_tier: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for tier, filename in SOURCE_NAMES.items():
        rows = read_jsonl(source / filename)
        if len(rows) != EXPECTED_COUNTS[tier]:
            raise RuntimeError(f"{tier} count drifted: {len(rows)}")
        if any(str(row.get("difficulty") or "").lower() != tier for row in rows):
            raise RuntimeError(f"{tier} manifest contains a foreign label")
        rows_by_tier[tier] = rows
        all_rows.extend(rows)

    for key in ("record_id", "theorem_group_id", "theorem_name"):
        values = [str(row.get(key) or "") for row in all_rows]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise RuntimeError(f"missing or duplicate {key}")
    if any(row.get("pantograph_verified") is not True for row in all_rows):
        raise RuntimeError("unverified row entered the official pool")
    if any(str(row.get("verification_status") or "") not in {"verified", "verified_default_timeout"} for row in all_rows):
        raise RuntimeError("invalid verification status entered the official pool")

    environment = {
        "lean_commits": sorted({str(row.get("lean_commit") or "") for row in all_rows}),
        "mathlib_repository_commits": sorted(
            {str(row.get("repository_commit") or "") for row in all_rows}
        ),
        "leandojo_v2_commits": sorted(
            {str(row.get("leandojo_v2_commit") or "") for row in all_rows}
        ),
        "repositories": sorted({str(row.get("repository") or "") for row in all_rows}),
    }
    if any(not value or len(value) != 1 for value in environment.values()):
        raise RuntimeError(f"environment identity is incomplete or mixed: {environment}")

    output.mkdir(parents=True)
    copied: list[Path] = []
    for filename in (*SOURCE_NAMES.values(), *SUPPORT_FILES):
        source_path = source / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = output / filename
        shutil.copy2(source_path, destination)
        copied.append(destination)

    manifest = {
        "freeze_name": "LeanDojo verified three-tier difficulty classification freeze",
        "freeze_version": "ld-difficulty-three-tier-v1",
        "status": "FROZEN_READ_ONLY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_pool_rows": int(stats["source_pool_rows"]),
        "eligible_rows": int(stats["eligible_rows"]),
        "official_classified_rows": len(all_rows),
        "difficulty_counts": EXPECTED_COUNTS,
        "classification_origins": dict(
            Counter(str(row.get("classification_origin") or "unknown") for row in all_rows)
        ),
        "environment": environment,
        "hard_audit": {
            "duplicate_theorem": audit["duplicate_theorem"],
            "duplicate_record_id": audit["duplicate_record_id"],
            "theorem_group_duplicate": audit["theorem_group_duplicate"],
            "source_provenance_complete": audit["source_provenance_complete"],
            "protected_evaluation_overlap": audit["protected_evaluation_overlap"],
        },
        "source_identity": {
            "input_sha256": audit["input_sha256"],
            "old_frozen_labels_sha256": audit["old_frozen_labels_sha256"],
            "source_manifest_sha256": audit["manifest_sha256"],
        },
        "immutability": {
            "archive_directory_mode": "0555",
            "archive_files_mode": "0444",
            "integrity_file": "SHA256SUMS",
            "overwrite_policy": "refuse_if_exists",
            "source_files_modified": False,
        },
    }
    freeze_manifest = output / "freeze_manifest.json"
    write_json(freeze_manifest, manifest)
    copied.append(freeze_manifest)

    readme = output / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# LeanDojo verified three-tier difficulty freeze",
                "",
                "Status: **FROZEN_READ_ONLY**",
                "",
                "| Tier | Rows |",
                "|---|---:|",
                "| Easy | 1,831 |",
                "| Medium | 2,363 |",
                "| Difficult | 609 |",
                "| Total | 4,803 |",
                "",
                "All rows are Pantograph verified, theorem-group unique, source-provenance complete,",
                "and zero-overlap with every protected evaluation set on all audited identity axes.",
                "Frozen manual judgments remain authoritative; previously unlabeled rows use the",
                "codex-local semantic rubric recorded in each row.",
                "",
                "Verify archive integrity with `sha256sum -c SHA256SUMS`.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    copied.append(readme)

    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(copied)),
        encoding="utf-8",
        newline="\n",
    )
    copied.append(sums)
    for path in copied:
        os.chmod(path, 0o444)
    os.chmod(output, 0o555)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
