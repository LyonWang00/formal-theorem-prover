"""Build or inspect the environment-pinned offline Mathlib declaration index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_prover.Mathlib import LocalMathlibDeclarationIndex
from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-path",
        type=Path,
        default=Path("lean_project"),
    )
    parser.add_argument("--query")
    parser.add_argument("--parameter-type")
    parser.add_argument("--domain")
    parser.add_argument("--limit", type=int, default=10)
    arguments = parser.parse_args()

    backend = PantographDeclarationCheckingBackend(
        project_path=arguments.project_path,
    )
    try:
        environment = backend.environment_identity(["Mathlib"])
    finally:
        backend.close()
    index = LocalMathlibDeclarationIndex.for_project(
        project_path=arguments.project_path,
        environment=environment,
    )
    output: dict[str, object] = {
        "database_path": str(index.database_path),
        "metadata": index.metadata(),
    }
    if arguments.query:
        output["results"] = [
            candidate.prompt_record()
            for candidate in index.search(
                arguments.query,
                parameter_type=arguments.parameter_type,
                domain=arguments.domain,
                limit=arguments.limit,
            )
        ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
