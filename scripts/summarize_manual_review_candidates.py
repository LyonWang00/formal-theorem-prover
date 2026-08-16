#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
args = parser.parse_args()
for index, line in enumerate(args.path.read_text(encoding="utf-8").splitlines()):
    if not line.strip():
        continue
    row = json.loads(line)
    error_head = str(row.get("error_message") or "").splitlines()
    proof = str(row.get("proof") or "")
    print(
        index,
        row.get("record_id"),
        row.get("source_chars"),
        row.get("error_family"),
        len(proof),
        error_head[0] if error_head else "",
    )
