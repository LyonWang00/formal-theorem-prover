from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    spec = json.loads((base / "PROJECT_SOURCE_SPEC.json").read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise FileExistsError(f"immutable bundle already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    excluded = set(spec["exclude_names"])
    sources: dict[str, Path] = {}
    for pattern in spec["root_patterns"]:
        for source in base.glob(pattern):
            if source.name not in excluded and source.resolve() != output:
                sources[source.name] = source
    for relative, source in spec["mapped_files"].items():
        sources[relative] = (base / source).resolve()

    for relative, source in sorted(sources.items()):
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    version = {
        "schema": "dynamic_grpo_project_version_v1",
        "architecture_version": spec["architecture_version"],
        "bundle_name": spec["bundle_name"],
        "immutable": True,
    }
    (output / "PROJECT_VERSION.json").write_text(
        json.dumps(version, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative == "SOURCE_MANIFEST.json":
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": "cross_endpoint_source_manifest_v1",
        "architecture_version": spec["architecture_version"],
        "bundle_name": spec["bundle_name"],
        "file_count": len(files),
        "files": files,
    }
    (output / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(output), "file_count": len(files)}))


if __name__ == "__main__":
    main()
