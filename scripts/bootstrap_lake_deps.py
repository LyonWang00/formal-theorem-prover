"""Restore Lake Git dependencies at exactly the locked manifest commits."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    project = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    manifest = json.loads(
        (project / "lake-manifest.json").read_text(encoding="utf-8")
    )
    packages = project / manifest.get("packagesDir", ".lake/packages")
    packages.mkdir(parents=True, exist_ok=True)

    for package in manifest["packages"]:
        if package["type"] != "git":
            continue
        name = package["name"]
        url = package["url"]
        revision = package["rev"]
        input_revision = package.get("inputRev", "")
        destination = packages / name
        print(f"Restoring {name} at {revision}", flush=True)

        for attempt in range(1, 6):
            shutil.rmtree(destination, ignore_errors=True)
            destination.mkdir(parents=True)
            try:
                run("git", "init", "-q", str(destination))
                run("git", "remote", "add", "origin", url, cwd=destination)
                run(
                    "git",
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    revision,
                    cwd=destination,
                )
                if input_revision.startswith("v"):
                    run(
                        "git",
                        "fetch",
                        "--depth",
                        "1",
                        "origin",
                        f"refs/tags/{input_revision}:"
                        f"refs/tags/{input_revision}",
                        cwd=destination,
                    )
                run(
                    "git",
                    "-c",
                    "advice.detachedHead=false",
                    "checkout",
                    "-q",
                    "FETCH_HEAD",
                    cwd=destination,
                )
                actual = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=destination,
                    text=True,
                ).strip()
                if actual != revision:
                    raise RuntimeError(
                        f"{name}: expected {revision}, got {actual}"
                    )
                break
            except (OSError, subprocess.CalledProcessError, RuntimeError):
                if attempt == 5:
                    raise
                print(
                    f"{name}: fetch attempt {attempt} failed; retrying",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(5)


if __name__ == "__main__":
    main()
