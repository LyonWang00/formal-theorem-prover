from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-version", default="dynamic-grpo-v2.1.2")
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    version = json.loads((root / "PROJECT_VERSION.json").read_text(encoding="utf-8"))
    assert manifest["architecture_version"] == args.expected_version
    assert version["architecture_version"] == args.expected_version
    expected = {row["path"]: row for row in manifest["files"]}
    actual = {
        p.relative_to(root).as_posix(): p
        for p in root.rglob("*")
        if p.is_file() and p.name != "SOURCE_MANIFEST.json"
    }
    assert set(actual) == set(expected), (sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected)))
    for relative, path in actual.items():
        row = expected[relative]
        assert path.stat().st_size == row["bytes"], relative
        assert digest(path) == row["sha256"], relative
    print(json.dumps({"status": "PASS", "architecture_version": args.expected_version, "file_count": len(actual)}))


if __name__ == "__main__":
    main()
