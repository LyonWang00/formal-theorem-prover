from __future__ import annotations

import json
import re
import sys
from pathlib import Path


FORBIDDEN_RE = re.compile(r"\b(?:sorry|admit)\b")


def main() -> None:
    path = Path(sys.argv[1])
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bad = [row for row in rows if FORBIDDEN_RE.search(row.get("text", ""))]
    print(f"bad_rows={len(bad)}")
    for row in bad:
        print("ID", row.get("id"))
        for field in ("informal_statement", "lean_statement", "proof", "text"):
            value = row.get(field, "")
            print(f"{field}_has_forbidden={bool(FORBIDDEN_RE.search(value))}")
            if FORBIDDEN_RE.search(value):
                match = FORBIDDEN_RE.search(value)
                start = max(0, match.start() - 160)
                end = min(len(value), match.end() + 160)
                print(value[start:end].encode("unicode_escape").decode("ascii"))
        print("---")


if __name__ == "__main__":
    main()

