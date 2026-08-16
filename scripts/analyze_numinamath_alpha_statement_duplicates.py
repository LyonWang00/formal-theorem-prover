#!/usr/bin/env python3
"""Find alpha-equivalent unresolved statements with a locally verified proof."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path


IDENT = re.compile(r"^[^\W\d]\w*(?:[₀-₉]+)?$", re.UNICODE)


def rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def clean_statement(source: str) -> str:
    source = re.sub(r"/-.*?-/", " ", source or "", flags=re.S)
    source = re.sub(r"(?m)^\s*(?:import|open|namespace|end)\b.*$", " ", source)
    source = re.sub(r"\btheorem\s+[^\s:(]+", "theorem _", source, count=1)
    source = re.sub(r":=\s*by\s*$", "", source.strip(), flags=re.S)
    return re.sub(r"\s+", " ", source).strip()


def top_level_binders(source: str) -> list[str]:
    start = source.find("theorem _")
    if start < 0:
        return []
    i = start + len("theorem _")
    groups = []
    while i < len(source):
        while i < len(source) and source[i].isspace():
            i += 1
        if i >= len(source) or source[i] not in "({":
            break
        opening = source[i]
        closing = ")" if opening == "(" else "}"
        depth = 0
        j = i
        while j < len(source):
            char = source[j]
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    groups.append(source[i + 1 : j])
                    i = j + 1
                    break
            j += 1
        else:
            break
    return groups


def alpha_key(source: str) -> tuple[str, dict[str, str]]:
    source = clean_statement(source)
    names = []
    for group in top_level_binders(source):
        if ":" not in group:
            continue
        name_side = group.split(":", 1)[0].strip()
        for token in name_side.split():
            token = token.strip("⦃⦄[]")
            if IDENT.match(token) and token not in names:
                names.append(token)
    original_to_canonical = {name: f"v{index}" for index, name in enumerate(names)}
    canonical_to_original = {canonical: original for original, canonical in original_to_canonical.items()}
    for original in sorted(original_to_canonical, key=len, reverse=True):
        source = re.sub(
            rf"(?<![\w₀-₉]){re.escape(original)}(?![\w₀-₉])",
            original_to_canonical[original],
            source,
        )
    return source, canonical_to_original


def notation_key(source: str) -> tuple[str, dict[str, str]]:
    key, binder_map = alpha_key(source)
    replacements = {
        "Real.sqrt": "sqrt",
        "√": "sqrt ",
        "Real.sin": "sin",
        "Real.cos": "cos",
        "Real.tan": "tan",
        "Real.exp": "exp",
        "Real.log": "log",
        "Real.pi": "pi",
        "π": "pi",
        "Set.Infinite": "Infinite",
        "Function.Periodic": "Periodic",
        "Nat.Prime": "Prime",
        "Set.range": "range",
        "λ": "fun",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    # Explicit versus implicit theorem binders do not change the proposition
    # that an already assembled proof must establish.
    start = key.find("theorem _") + len("theorem _")
    chars = list(key)
    cursor = start
    matching = {"(": ")", "{": "}", "[": "]"}
    while cursor < len(chars):
        while cursor < len(chars) and chars[cursor].isspace():
            cursor += 1
        if cursor >= len(chars) or chars[cursor] not in "({":
            break
        opening_index = cursor
        stack = [chars[cursor]]
        cursor += 1
        while cursor < len(chars) and stack:
            char = chars[cursor]
            if char in matching:
                stack.append(char)
            elif char in matching.values() and matching[stack[-1]] == char:
                stack.pop()
            cursor += 1
        if stack:
            break
        closing_index = cursor - 1
        chars[opening_index] = "("
        chars[closing_index] = ")"
    key = "".join(chars)
    return re.sub(r"\s+", " ", key).strip(), binder_map


def layout_key(source: str) -> tuple[str, dict[str, str]]:
    """Normalize only whitespace around Lean punctuation and operators."""
    key, binder_map = alpha_key(source)
    key = re.sub(r"\s*([(){}\[\],:;^=+*/<>≤≥∧∨¬∈∉→↔|])\s*", r"\1", key)
    key = re.sub(r"\s+-\s+", "-", key)
    return re.sub(r"\s+", " ", key).strip(), binder_map


def binder_layout_key(source: str) -> tuple[str, dict[str, str]]:
    """Normalize layout plus explicit/implicit top-level theorem binders."""
    key, binder_map = layout_key(source)
    start = key.find("theorem _") + len("theorem _")
    chars = list(key)
    cursor = start
    matching = {"(": ")", "{": "}", "[": "]"}
    while cursor < len(chars):
        while cursor < len(chars) and chars[cursor].isspace():
            cursor += 1
        if cursor >= len(chars) or chars[cursor] not in "({":
            break
        opening_index = cursor
        stack = [chars[cursor]]
        cursor += 1
        while cursor < len(chars) and stack:
            char = chars[cursor]
            if char in matching:
                stack.append(char)
            elif char in matching.values() and matching[stack[-1]] == char:
                stack.pop()
            cursor += 1
        if stack:
            break
        chars[opening_index] = "("
        chars[cursor - 1] = ")"
    return "".join(chars), binder_map


def type_layout_key(source: str) -> tuple[str, dict[str, str]]:
    """Also normalize the standard Unicode aliases for core numeric types."""
    key, binder_map = binder_layout_key(source)
    for old, new in (
        ("ℝ", "Real"), ("ℕ", "Nat"), ("ℤ", "Int"),
        ("ℂ", "Complex"), ("ℚ", "Rat"),
    ):
        key = key.replace(old, new)
    return key, binder_map


def real_notation_layout_key(source: str) -> tuple[str, dict[str, str]]:
    """Normalize layout and definitionally identical Real notation only."""
    key, binder_map = type_layout_key(source)
    replacements = {
        "Real.sqrt": "sqrt",
        "√": "sqrt ",
        "Real.sin": "sin",
        "Real.cos": "cos",
        "Real.tan": "tan",
        "Real.exp": "exp",
        "Real.log": "log",
        "Real.pi": "pi",
        "π": "pi",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return re.sub(r"\s+", " ", key).strip(), binder_map


def scoped_notation_layout_key(source: str) -> tuple[str, dict[str, str]]:
    """Normalize qualified versus opened Set/Function constants."""
    key, binder_map = type_layout_key(source)
    for namespace in ("Set", "Function"):
        for name in (
            "Ioo", "Icc", "Ico", "Ioc", "range", "Infinite", "Finite",
            "Periodic", "Injective", "Surjective", "Bijective",
        ):
            key = key.replace(f"{namespace}.{name}", name)
    return key, binder_map


def atomic_parentheses_layout_key(source: str) -> tuple[str, dict[str, str]]:
    """Remove redundant parentheses around canonical binders and numerals."""
    key, binder_map = type_layout_key(source)
    previous = None
    while previous != key:
        previous = key
        key = re.sub(r"\((v\d+|-?\d+)\)", r"\1", key)
    return key, binder_map


def relational_layout_key(source: str) -> tuple[str, dict[str, str]]:
    """Canonicalize atomic `>`/`≥` comparisons by reversing their operands."""
    key, binder_map = type_layout_key(source)
    atom = r"(?:v\d+|-?\d+)"
    key = re.sub(rf"({atom})>({atom})", lambda m: f"{m.group(2)}<{m.group(1)}", key)
    key = re.sub(rf"({atom})≥({atom})", lambda m: f"{m.group(2)}≤{m.group(1)}", key)
    return key, binder_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--fail", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, action="append", required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalization",
        choices=(
            "alpha", "notation", "layout", "binder_layout", "type_layout",
            "real_notation_layout", "scoped_notation_layout",
            "atomic_parentheses_layout",
            "relational_layout",
        ),
        default="alpha",
    )
    args = parser.parse_args()
    key_function = {
        "alpha": alpha_key,
        "notation": notation_key,
        "layout": layout_key,
        "binder_layout": binder_layout_key,
        "type_layout": type_layout_key,
        "real_notation_layout": real_notation_layout_key,
        "scoped_notation_layout": scoped_notation_layout_key,
        "atomic_parentheses_layout": atomic_parentheses_layout_key,
        "relational_layout": relational_layout_key,
    }[args.normalization]

    excluded = {
        str(row["record_id"])
        for path in args.exclude
        for row in rows(path)
        if row.get("record_id")
    }
    verified: dict[str, list[dict]] = defaultdict(list)
    solved_ids: set[str] = set()

    for row in rows(args.success):
        proof = row.get("proof") or row.get("formal_proof")
        statement = row.get("lean_statement") or row.get("formal_statement") or ""
        key, binder_map = key_function(statement)
        if key and proof:
            verified[key].append(
                {
                    "record_id": str(row.get("record_id") or "canonical_success"),
                    "proof": proof,
                    "lean_statement": statement,
                    "binder_map": binder_map,
                }
            )

    result_paths = sorted(
        {
            result
            for root in args.batch_root
            for result in glob.glob(str(root / "batch_*" / "verification_results.jsonl"))
        }
    )
    for result_name in result_paths:
        result_path = Path(result_name)
        manifest_path = result_path.parent / "candidate_manifest.jsonl"
        if not manifest_path.is_file():
            continue
        candidates = {str(row["record_id"]): row for row in rows(manifest_path)}
        for result in rows(result_path):
            if result.get("success") is not True and result.get("pantograph_verified") != "success":
                continue
            record_id = str(result["record_id"])
            solved_ids.add(record_id)
            candidate = candidates.get(record_id, {})
            statement = candidate.get("original_source_body") or candidate.get("source_body") or ""
            key, binder_map = key_function(statement)
            proof = result.get("proof")
            if key and proof:
                verified[key].append(
                    {
                        "record_id": record_id,
                        "proof": proof,
                        "lean_statement": statement,
                        "binder_map": binder_map,
                    }
                )

    matches = []
    for row in rows(args.fail):
        record_id = str(row.get("record_id") or "")
        if not record_id or record_id in excluded or record_id in solved_ids:
            continue
        statement = row.get("lean_statement") or row.get("formal_statement") or ""
        key, binder_map = key_function(statement)
        peers = verified.get(key)
        if not peers:
            continue
        peer = min(peers, key=lambda item: len(str(item["proof"])))
        if clean_statement(statement) == clean_statement(peer["lean_statement"]):
            continue
        matches.append(
            {
                "record_id": record_id,
                "lean_statement": statement,
                "target_binder_map": binder_map,
                "verified_record_id": peer["record_id"],
                "verified_lean_statement": peer["lean_statement"],
                "source_binder_map": peer["binder_map"],
                "proof": peer["proof"],
                "verified_proofs_for_alpha_statement": len(peers),
            }
        )

    matches.sort(key=lambda row: row["record_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in matches:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "batch_result_files": len(result_paths),
                "locally_solved_record_ids": len(solved_ids),
                "verified_alpha_keys": len(verified),
                "new_alpha_matches": len(matches),
                "normalization": args.normalization,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
