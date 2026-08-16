#!/usr/bin/env python3
"""Identify NuminaMath rows that depend on a sorry-backed helper proposition.

This intentionally does not classify an ordinary main-theorem placeholder as
need_decompose.  A row is selected only when a named helper proposition has a
sorry/admit body and that helper name is referenced later in the same source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


SORRY = re.compile(r"\b(?:sorry|admit)\b")
TOP_DECL = re.compile(
    r"(?m)^[ \t]*(?:private[ \t]+|protected[ \t]+)?(?:theorem|lemma)[ \t]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_'.]*)\b"
)
LOCAL_HELPER = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?:have|suffices)[ \t]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_']*)\b"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{number}: {exc}") from exc
    return rows


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_text(row: dict[str, Any]) -> tuple[str, str]:
    for field in (
        "formal_ground_truth",
        "formal_proof",
        "lean_code",
        "source_body",
        "statement",
    ):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return field, value
    return "", ""


def mask_comments(source: str) -> str:
    # Preserve character positions and newlines while masking Lean comments.
    chars = list(source)
    depth = 0
    index = 0
    while index < len(chars):
        pair = "".join(chars[index:index + 2])
        if depth == 0 and pair == "--":
            end = source.find("\n", index)
            if end < 0:
                end = len(chars)
            for pos in range(index, end):
                chars[pos] = " "
            index = end
            continue
        if pair == "/-":
            depth += 1
            chars[index:index + 2] = [" ", " "]
            index += 2
            continue
        if depth and pair == "-/":
            depth -= 1
            chars[index:index + 2] = [" ", " "]
            index += 2
            continue
        if depth and chars[index] != "\n":
            chars[index] = " "
        index += 1
    return "".join(chars)


def referenced_later(source: str, name: str, offset: int) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_']){re.escape(name)}(?![A-Za-z0-9_'])", source[offset:]) is not None


def classify(source: str) -> list[dict[str, Any]]:
    masked = mask_comments(source)
    evidence: list[dict[str, Any]] = []

    top = list(TOP_DECL.finditer(masked))
    for index, match in enumerate(top[:-1]):
        end = top[index + 1].start()
        segment = masked[match.start():end]
        sorry = SORRY.search(segment)
        if sorry is None:
            continue
        name = match.group("name")
        if referenced_later(masked, name, end):
            evidence.append({
                "kind": "top_level_helper_with_placeholder",
                "helper_name": name,
                "declaration_line": masked.count("\n", 0, match.start()) + 1,
                "placeholder_line": masked.count("\n", 0, match.start() + sorry.start()) + 1,
                "referenced_after_helper": True,
            })

    line_starts = [0]
    for match in re.finditer("\n", masked):
        line_starts.append(match.end())
    for helper in LOCAL_HELPER.finditer(masked):
        start = helper.start()
        indent = len(helper.group("indent").replace("\t", "    "))
        start_line = masked.count("\n", 0, start)
        end = len(masked)
        for next_line in range(start_line + 1, len(line_starts)):
            pos = line_starts[next_line]
            line_end = masked.find("\n", pos)
            if line_end < 0:
                line_end = len(masked)
            line = masked[pos:line_end]
            if not line.strip():
                continue
            next_indent = len(line) - len(line.lstrip(" \t"))
            if next_indent <= indent:
                end = pos
                break
        block = masked[start:end]
        sorry = SORRY.search(block)
        if sorry is None:
            continue
        name = helper.group("name")
        if referenced_later(masked, name, end):
            evidence.append({
                "kind": "local_helper_with_placeholder",
                "helper_name": name,
                "declaration_line": start_line + 1,
                "placeholder_line": masked.count("\n", 0, start + sorry.start()) + 1,
                "referenced_after_helper": True,
            })
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    fail_rows = read_jsonl(args.fail_file)
    excluded_ids: set[str] = set()
    excluded_hashes: dict[str, str] = {}
    for manifest in args.exclude_manifest:
        excluded_ids |= {str(row["record_id"]) for row in read_jsonl(manifest)}
        excluded_hashes[str(manifest)] = sha256_file(manifest)
    existing = read_jsonl(args.output_file) if args.output_file.exists() else []
    existing_by_id = {
        str(row["record_id"]): row
        for row in existing
        if str(row["record_id"]) not in excluded_ids
    }
    detected: list[dict[str, Any]] = []
    excluded_detected = 0
    evidence_kind_counts: dict[str, int] = {}
    source_field_counts: dict[str, int] = {}
    for row in fail_rows:
        field, source = source_text(row)
        evidence = classify(source)
        if not evidence:
            continue
        if str(row["record_id"]) in excluded_ids:
            excluded_detected += 1
            continue
        updated = dict(row)
        updated["need_decompose"] = {
            "schema_version": "numinamath_need_decompose_v1",
            "status": "need_decompose",
            "reason_category": "referenced_helper_contains_placeholder",
            "source_field": field,
            "evidence": evidence,
            "repair_eligible": False,
        }
        detected.append(updated)
        source_field_counts[field] = source_field_counts.get(field, 0) + 1
        for item in evidence:
            kind = str(item["kind"])
            evidence_kind_counts[kind] = evidence_kind_counts.get(kind, 0) + 1

    detected_ids = {str(row["record_id"]) for row in detected}
    for row in detected:
        existing_by_id[str(row["record_id"])] = row
    output_rows = [existing_by_id[key] for key in sorted(existing_by_id)]
    atomic_write_jsonl(args.output_file, output_rows)
    if args.apply:
        remaining = [row for row in fail_rows if str(row["record_id"]) not in detected_ids]
        atomic_write_jsonl(args.fail_file, remaining)
    else:
        remaining = fail_rows

    report = {
        "schema_version": "numinamath_need_decompose_audit_v1",
        "mode": "apply" if args.apply else "audit_only",
        "fail_before": len(fail_rows),
        "newly_detected": len(detected),
        "detected_but_excluded_frozen": excluded_detected,
        "need_decompose_total": len(output_rows),
        "fail_after": len(remaining),
        "evidence_kind_counts": evidence_kind_counts,
        "source_field_counts": source_field_counts,
        "output_sha256": sha256_file(args.output_file),
        "fail_sha256": sha256_file(args.fail_file),
        "main_theorem_placeholder_alone_is_excluded": True,
        "excluded_manifest_hashes": excluded_hashes,
    }
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
