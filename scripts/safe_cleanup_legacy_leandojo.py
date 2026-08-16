"""Audit and safely remove explicitly configured legacy LeanDojo artifacts.

The command never discovers deletion targets by glob.  It consumes a JSON
configuration containing literal candidate paths, resolves and validates every
path against allowlisted roots and protected paths, writes a dry-run inventory,
and only deletes entries that the inventory explicitly marks ``delete``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


FORBIDDEN_ROOTS = {Path("/")}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _same_or_contains(left: Path, right: Path) -> bool:
    return left == right or _is_relative_to(right, left)


def assert_safe_deletion_target(
    target: Path,
    allowed_roots: list[Path],
    protected_paths: list[Path],
) -> Path:
    """Resolve and validate a literal deletion target.

    A target must exist below, but not equal to, an allowlisted root.  It may
    neither equal, contain, nor be contained by a protected path.  Symlinks are
    rejected because deleting them cannot establish ownership of their target.
    """

    expanded = target.expanduser()
    if not expanded.exists() and not expanded.is_symlink():
        raise ValueError(f"target does not exist: {expanded}")
    if expanded.is_symlink():
        raise ValueError(f"symlink deletion target is forbidden: {expanded}")

    resolved = expanded.resolve(strict=True)
    roots = [path.expanduser().resolve(strict=True) for path in allowed_roots]
    protected = [
        path.expanduser().resolve(strict=True)
        for path in protected_paths
        if path.expanduser().exists()
    ]

    if resolved in FORBIDDEN_ROOTS or resolved == Path(resolved.anchor):
        raise ValueError(f"filesystem root is forbidden: {resolved}")
    if resolved == Path.home().resolve():
        raise ValueError(f"home directory is forbidden: {resolved}")
    if not any(resolved != root and _is_relative_to(resolved, root) for root in roots):
        raise ValueError(f"target is outside allowed roots: {resolved}")
    for protected_path in protected:
        if _same_or_contains(resolved, protected_path) or _same_or_contains(
            protected_path, resolved
        ):
            raise ValueError(
                f"target overlaps protected path {protected_path}: {resolved}"
            )
    return resolved


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            child = root_path / name
            if not child.is_symlink():
                total += child.stat().st_size
    return total


def _git_tracked(project_root: Path | None, path: Path) -> bool:
    if project_root is None or not _is_relative_to(path, project_root):
        return False
    relative = path.relative_to(project_root).as_posix()
    result = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "--", relative],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AuditContext:
    platform: str
    project_root: Path | None
    allowed_roots: list[Path]
    protected_paths: list[Path]


def _candidate_record(
    candidate: dict[str, Any],
    context: AuditContext,
) -> dict[str, Any]:
    literal = Path(candidate["path"]).expanduser()
    record: dict[str, Any] = {
        "path": str(literal.absolute()),
        "platform": context.platform,
        "type": candidate.get("type", "uncertain"),
        "size_bytes": 0,
        "is_git_tracked": False,
        "is_symlink": literal.is_symlink(),
        "resolved_target": "",
        "evidence": list(candidate.get("evidence", [])),
        "protected_path_overlap": False,
        "recommended_action": candidate.get("recommended_action", "uncertain"),
        "reason": candidate.get("reason", ""),
        "archive_path": candidate.get("archive_path"),
    }
    if not literal.exists() and not literal.is_symlink():
        record["recommended_action"] = "keep"
        record["reason"] = "candidate path does not exist"
        return record

    resolved = literal.resolve(strict=True)
    record["resolved_target"] = str(resolved)
    record["size_bytes"] = _size_bytes(literal)
    record["is_git_tracked"] = _git_tracked(context.project_root, resolved)

    for protected in context.protected_paths:
        if not protected.exists():
            continue
        protected_resolved = protected.resolve(strict=True)
        if _same_or_contains(resolved, protected_resolved) or _same_or_contains(
            protected_resolved, resolved
        ):
            record["protected_path_overlap"] = True
            break

    expected = candidate.get("evidence_file")
    expected_hash = candidate.get("evidence_sha256")
    if expected and expected_hash:
        evidence_file = Path(expected).expanduser().resolve(strict=True)
        actual_hash = _sha256(evidence_file)
        record["evidence"].append(
            {
                "file": str(evidence_file),
                "sha256": actual_hash,
                "matches_expected": actual_hash == expected_hash,
            }
        )
        if actual_hash != expected_hash:
            record["recommended_action"] = "uncertain"
            record["reason"] = "provenance evidence hash mismatch"

    if record["is_git_tracked"]:
        record["recommended_action"] = "keep"
        record["reason"] = "Git-tracked content is protected"
    elif record["is_symlink"]:
        record["recommended_action"] = "keep"
        record["reason"] = "symbolic links are never deleted"
    elif record["protected_path_overlap"]:
        record["recommended_action"] = "keep"
        record["reason"] = "candidate overlaps a protected path"
    elif record["recommended_action"] in {"delete", "archive_then_delete"}:
        try:
            if context.project_root is not None and resolved == context.project_root:
                raise ValueError(f"project root is forbidden: {resolved}")
            assert_safe_deletion_target(
                literal, context.allowed_roots, context.protected_paths
            )
        except ValueError as error:
            record["recommended_action"] = "keep"
            record["reason"] = str(error)
    return record


def _summary(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "candidate_count": len(records),
        "total_candidate_size": sum(item["size_bytes"] for item in records),
        "safe_to_delete_count": sum(
            item["recommended_action"] == "delete" for item in records
        ),
        "archive_then_delete_count": sum(
            item["recommended_action"] == "archive_then_delete" for item in records
        ),
        "uncertain_count": sum(
            item["recommended_action"] == "uncertain" for item in records
        ),
        "protected_rejected_count": sum(
            item["recommended_action"] == "keep" for item in records
        ),
    }


def _markdown_inventory(payload: dict[str, Any]) -> str:
    lines = [
        "# Legacy LeanDojo cleanup inventory",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| Platform | Action | Type | Bytes | Path | Reason |",
        "|---|---|---|---:|---|---|",
    ]
    for item in payload["candidates"]:
        lines.append(
            "| {platform} | {action} | {type} | {size} | `{path}` | {reason} |".format(
                platform=item["platform"],
                action=item["recommended_action"],
                type=item["type"],
                size=item["size_bytes"],
                path=item["path"].replace("|", r"\|"),
                reason=item["reason"].replace("|", r"\|"),
            )
        )
    lines.extend(["", "## Summary", "", "```json"])
    lines.append(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines) + "\n"


def _load_config(path: Path) -> tuple[AuditContext, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    project_value = payload.get("project_root")
    context = AuditContext(
        platform=str(payload["platform"]),
        project_root=Path(project_value).resolve() if project_value else None,
        allowed_roots=[Path(value) for value in payload["allowed_roots"]],
        protected_paths=[Path(value) for value in payload["protected_paths"]],
    )
    return context, list(payload["candidates"])


def create_inventory(config_path: Path, output_json: Path, output_md: Path) -> None:
    context, candidates = _load_config(config_path)
    records = [_candidate_record(candidate, context) for candidate in candidates]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path.resolve()),
        "candidates": records,
        "summary": _summary(records),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(_markdown_inventory(payload), encoding="utf-8")


def apply_inventory(
    config_path: Path,
    inventory_path: Path,
    result_path: Path,
) -> None:
    context, _ = _load_config(config_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    deleted: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for item in inventory["candidates"]:
        action = item["recommended_action"]
        target = Path(item["path"])
        if action == "archive_then_delete":
            archive_value = item.get("archive_path")
            if not archive_value or not Path(archive_value).expanduser().is_file():
                kept.append(
                    {**item, "apply_reason": "required archive does not exist"}
                )
                continue
        if action not in {"delete", "archive_then_delete"}:
            kept.append(item)
            continue
        if context.project_root is not None and target.resolve() == context.project_root:
            raise ValueError(f"project root is forbidden: {target}")
        resolved = assert_safe_deletion_target(
            target, context.allowed_roots, context.protected_paths
        )
        size = _size_bytes(resolved)
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        if resolved.exists() or resolved.is_symlink():
            raise RuntimeError(f"deletion verification failed: {resolved}")
        deleted.append({"path": str(resolved), "size_bytes": size, "action": action})

    payload = {
        "completed_at": datetime.now(UTC).isoformat(),
        "deleted": deleted,
        "deleted_bytes": sum(item["size_bytes"] for item in deleted),
        "kept": kept,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    inventory = subcommands.add_parser("inventory")
    inventory.add_argument("--config", type=Path, required=True)
    inventory.add_argument("--output-json", type=Path, required=True)
    inventory.add_argument("--output-md", type=Path, required=True)

    apply = subcommands.add_parser("apply")
    apply.add_argument("--config", type=Path, required=True)
    apply.add_argument("--inventory", type=Path, required=True)
    apply.add_argument("--result", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "inventory":
        create_inventory(args.config, args.output_json, args.output_md)
    else:
        apply_inventory(args.config, args.inventory, args.result)


if __name__ == "__main__":
    main()
