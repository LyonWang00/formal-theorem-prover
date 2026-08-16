"""Freeze the two completed LeanDojo manual-classification rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROUND1 = Path("outputs/ld_length_difficulty_pipeline/difficulty/difficulty_labels.jsonl")
ROUND1_SUMMARY = Path("outputs/ld_length_difficulty_pipeline/difficulty/difficulty_summary.json")
ROUND2 = Path(
    "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion/expansion_labels.jsonl"
)
ROUND2_SUMMARY = Path(
    "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion/expansion_summary.json"
)
ROUND2_SELECTION = Path(
    "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion/selection_audit.json"
)
TRAINABLE_EASY = Path(
    "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion/trainable_easy_pool.jsonl"
)
SOURCE_POOL = Path(
    "outputs/leandojo_v2_dataset_build/final_pool/leandojo_v2_final_train_candidates.jsonl"
)
DEFAULT_OUTPUT = Path("outputs/ld_manual_classification_freeze/two_round_v1")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id") or row.get("id")
    if not value:
        raise ValueError("manual label without sample_id")
    return str(value)


def validate_round(
    rows: list[dict[str, Any]], expected_rows: int, expected_counts: dict[str, int]
) -> tuple[Counter[str], set[str]]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, found {len(rows)}")
    ids = [sample_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample IDs inside manual classification round")
    counts = Counter(str(row.get("difficulty")) for row in rows)
    if dict(counts) != expected_counts:
        raise RuntimeError(f"difficulty counts drifted: {dict(counts)} != {expected_counts}")
    if any(str(row.get("reviewed_by")) != "codex" for row in rows):
        raise RuntimeError("non-manual/non-Codex label found")
    return counts, set(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen archive: {output}")

    sources = {
        "round1_labels": project / ROUND1,
        "round1_summary": project / ROUND1_SUMMARY,
        "round2_labels": project / ROUND2,
        "round2_summary": project / ROUND2_SUMMARY,
        "round2_selection": project / ROUND2_SELECTION,
        "trainable_easy_pool": project / TRAINABLE_EASY,
        "leandojo_final_source_pool": project / SOURCE_POOL,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")

    round1 = list(iter_jsonl(sources["round1_labels"]))
    round2 = list(iter_jsonl(sources["round2_labels"]))
    counts1, ids1 = validate_round(
        round1, 600, {"easy": 204, "medium": 260, "difficult": 108, "exclude": 28}
    )
    counts2, ids2 = validate_round(
        round2, 1125, {"easy": 864, "medium": 255, "exclude": 6}
    )
    overlap = sorted(ids1 & ids2)
    if overlap:
        raise RuntimeError(f"manual rounds overlap: {overlap[:10]}")

    output.mkdir(parents=True)
    try:
        round1_copy = output / "round1_manual_labels_600.jsonl"
        round2_copy = output / "round2_expansion_manual_labels_1125.jsonl"
        shutil.copyfile(sources["round1_labels"], round1_copy)
        shutil.copyfile(sources["round2_labels"], round2_copy)

        combined_path = output / "combined_manual_labels_1725.jsonl"
        combined_rows = (
            {
                "freeze_round": "round1_initial",
                "freeze_source_classification_version": row.get("classification_version"),
                **row,
            }
            for row in round1
        )
        combined_count = write_jsonl(combined_path, combined_rows)
        with combined_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in round2:
                frozen = {
                    "freeze_round": "round2_easy_expansion",
                    "freeze_source_classification_version": row.get("classification_version"),
                    **row,
                }
                handle.write(json.dumps(frozen, ensure_ascii=False, sort_keys=True) + "\n")
                combined_count += 1
        if combined_count != 1725:
            raise RuntimeError(f"combined freeze row mismatch: {combined_count}")

        easy_ids_path = output / "trainable_easy_identity_1004.jsonl"
        easy_ids: set[str] = set()

        def compact_easy_rows() -> Iterator[dict[str, Any]]:
            for row in iter_jsonl(sources["trainable_easy_pool"]):
                sid = sample_id(row)
                if sid in easy_ids:
                    raise RuntimeError(f"duplicate trainable easy ID: {sid}")
                easy_ids.add(sid)
                annotation = row.get("difficulty_annotation") or {}
                yield {
                    "sample_id": sid,
                    "qualified_name": row.get("qualified_name") or annotation.get("qualified_name"),
                    "theorem_group_id": row.get("theorem_group_id"),
                    "source_file": row.get("source_file"),
                    "difficulty": annotation.get("difficulty", "easy"),
                    "classification_version": annotation.get("classification_version"),
                    "proof_hash_exact": row.get("proof_hash_exact"),
                    "statement_hash_exact": row.get("statement_hash_exact"),
                }

        easy_count = write_jsonl(easy_ids_path, compact_easy_rows())
        if easy_count != 1004:
            raise RuntimeError(f"expected 1004 trainable easy identities, found {easy_count}")
        if not easy_ids <= (ids1 | ids2):
            raise RuntimeError("trainable easy snapshot contains IDs outside the two manual rounds")

        combined_counts = counts1 + counts2
        source_identity = {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sources.items()
        }
        source_identity["round1_labels"]["rows"] = 600
        source_identity["round2_labels"]["rows"] = 1125
        source_identity["trainable_easy_pool"]["rows"] = 1004
        source_identity["leandojo_final_source_pool"]["rows"] = 5311

        manifest = {
            "freeze_name": "LeanDojo two-round manual classification freeze",
            "freeze_version": "ld-manual-two-round-v1",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "status": "FROZEN_READ_ONLY",
            "automatic_difficulty_labels": 0,
            "source_pool_rows": 5311,
            "rounds": {
                "round1_initial": {
                    "reviewed": 600,
                    "classification_version": "ld-manual-v1",
                    "difficulty_counts": dict(counts1),
                },
                "round2_easy_expansion": {
                    "selected": 1200,
                    "reviewed": 1125,
                    "unreviewed": 75,
                    "classification_version": "ld-manual-anchor-expansion-v1",
                    "difficulty_counts": dict(counts2),
                },
            },
            "combined": {
                "reviewed_unique": 1725,
                "cross_round_duplicate_sample_ids": 0,
                "difficulty_counts": dict(combined_counts),
                "valid_difficulty_rows": combined_counts["easy"]
                + combined_counts["medium"]
                + combined_counts["difficult"],
                "excluded_rows": combined_counts["exclude"],
            },
            "trainable_easy": {
                "old_easy_total": 204,
                "old_protected_overlap_removed": 64,
                "old_trainable_easy": 140,
                "new_trainable_easy": 864,
                "total_trainable_easy": 1004,
                "round1_sft_used": 1000,
                "unused_after_round1_sft": 4,
            },
            "source_identity": source_identity,
            "immutability": {
                "source_files_modified": False,
                "archive_files_mode": "0444",
                "archive_directory_mode": "0555",
                "integrity_file": "SHA256SUMS",
                "overwrite_policy": "refuse_if_exists",
            },
        }
        manifest_path = output / "freeze_manifest.json"
        write_json(manifest_path, manifest)

        report = output / "README.md"
        report.write_text(
            "# LeanDojo two-round manual classification freeze\n\n"
            "Status: **FROZEN_READ_ONLY**\n\n"
            "| Round | Reviewed | Easy | Medium | Difficult | Exclude |\n"
            "|---|---:|---:|---:|---:|---:|\n"
            "| Initial manual classification | 600 | 204 | 260 | 108 | 28 |\n"
            "| LD-easy expansion | 1125 | 864 | 255 | 0 | 6 |\n"
            "| Combined unique | 1725 | 1068 | 515 | 108 | 34 |\n\n"
            "The expansion pool selected 1,200 records; 1,125 received manual labels and 75 remain unreviewed. "
            "No automatic difficulty labels are included. The two reviewed rounds have zero sample-ID overlap.\n\n"
            "The protected-clean trainable easy pool contains 1,004 identities: 140 retained from the initial round "
            "after removing 64 protected overlaps, plus 864 from the expansion. Round-1 SFT used 1,000, leaving 4 unused.\n\n"
            "All archive files are read-only. Verify integrity with `sha256sum -c SHA256SUMS`.\n",
            encoding="utf-8",
        )

        artifacts = [round1_copy, round2_copy, combined_path, easy_ids_path, manifest_path, report]
        checksums = output / "SHA256SUMS"
        checksums.write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
            encoding="utf-8",
        )
        artifacts.append(checksums)
        for path in artifacts:
            os.chmod(path, 0o444)
        os.chmod(output, 0o555)
    except Exception:
        # Only clean the brand-new incomplete target. Existing archives are
        # never touched because creation refuses when the target exists.
        if output.exists():
            os.chmod(output, 0o755)
            shutil.rmtree(output)
        raise

    print(
        json.dumps(
            {
                "status": "FROZEN_READ_ONLY",
                "output": str(output),
                "reviewed_unique": 1725,
                "difficulty_counts": dict(counts1 + counts2),
                "trainable_easy": 1004,
                "source_files_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
