from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:120] or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard an attempts.jsonl file by problem.")
    parser.add_argument("attempts_jsonl")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    attempts_path = Path(args.attempts_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else attempts_path.parent
    shards_dir = output_dir / "attempt_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    for old_path in shards_dir.glob("*.jsonl"):
        old_path.unlink()

    count = 0
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        problem_index = int(row.get("problem_index", -1))
        problem_id = sanitize(str(row.get("problem_id", "unknown")))
        shard_path = shards_dir / f"problem_{problem_index:05d}_{problem_id}.jsonl"
        with shard_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1

    archive_path = output_dir / "attempt_shards.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for shard_path in sorted(shards_dir.glob("*.jsonl")):
            archive.write(shard_path, arcname=f"attempt_shards/{shard_path.name}")

    print(
        json.dumps(
            {
                "attempt_rows": count,
                "shard_files": len(list(shards_dir.glob("*.jsonl"))),
                "archive": str(archive_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
