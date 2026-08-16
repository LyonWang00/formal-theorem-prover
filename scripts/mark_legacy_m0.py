#!/usr/bin/env python3
"""Mark an existing M0 as legacy without deleting or changing model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_artifact(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    identity_files = [
        item
        for item in files
        if item.name
        in {
            "adapter_config.json",
            "adapter_model.safetensors",
            "config.json",
            "generation_config.json",
            "training_config.json",
            "trainer_state.json",
        }
    ]
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "file_count": len(files),
        "identity_file_sha256": {
            str(item.relative_to(path)): file_sha256(item) for item in identity_files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--generation-reverification", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    initial_sft = run_dir / "initial_sft"
    payload = {
        "status": "legacy_unverified_data_m0",
        "excluded_from_formal_expert_iteration": True,
        "reason": (
            "This M0 was trained on the legacy prepared split before current-target "
            "Pantograph proof attestations were required. It is retained only for "
            "diagnosis and reproducibility."
        ),
        "marked_at": datetime.now(timezone.utc).isoformat(),
        "artifacts_modified": False,
        "artifacts": {
            "adapter": inspect_artifact(initial_sft / "checkpoint"),
            "merged_anchor": inspect_artifact(initial_sft / "merged_anchor"),
        },
        "evidence": {
            "dataset_audit_summary": str(Path(args.audit_summary).expanduser().resolve()),
            "legacy_generation_reverification": str(
                Path(args.generation_reverification).expanduser().resolve()
            ),
        },
    }
    destination = initial_sft / "legacy_status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
