"""Conservative, reproducible fingerprints for Lean dataset auditing."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

_HORIZONTAL_SPACE = re.compile(r"[ \t]+")
_DECL_NAME = re.compile(
    r"^(\s*(?:theorem|lemma|example|def|instance)\s+)([^\s:(\[{]+)",
    re.MULTILINE,
)
_DECL_KIND_AND_NAME = re.compile(
    r"^\s*(?:theorem|lemma|example|def|instance)\s+[^\s:(\[{]+",
    re.MULTILINE,
)
_TRAILING_ASSIGN = re.compile(r"\s*:=\s*$")
_TOKEN = re.compile(
    r"[A-Za-z_\u0080-\uffff][\w'.\u0080-\uffff]*|\d+|:=|=>|->|[^\s]",
    re.UNICODE,
)


def exact_hash(value: str) -> str:
    """Hash exact UTF-8 text without changing its bytes."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return exact_hash(payload)


def normalize_lexical(value: str, *, strip_outer: bool = True) -> str:
    """Apply only semantics-preserving lexical normalization.

    This representation is for hashing/search only. It must never replace the
    source-faithful text submitted to Lean.
    """

    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    lines = [_HORIZONTAL_SPACE.sub(" ", line.rstrip()) for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip() if strip_outer else text.rstrip()


def normalized_hash(value: str) -> str:
    return exact_hash(normalize_lexical(value))


def normalize_statement_lexical(value: str) -> str:
    """Normalize the non-semantic declaration wrapper for theorem identity."""

    normalized = _TRAILING_ASSIGN.sub("", normalize_lexical(value))
    return _DECL_NAME.sub(r"\1<DECL_NAME>", normalized, count=1)


def statement_hash(value: str) -> str:
    return exact_hash(normalize_statement_lexical(value))


def source_identity_hash(record: Mapping[str, Any]) -> str:
    identity = {
        "repository_commit": record.get("repository_commit")
        or record.get("source_commit")
        or record.get("mathlib_commit"),
        "source_file": record.get("source_file"),
        "qualified_name": record.get("qualified_name")
        or record.get("source_declaration"),
        "source_span": record.get("source_span") or {},
    }
    return stable_json_hash(identity)


def _premise_name(premise: Any) -> str:
    if isinstance(premise, str):
        return premise
    if isinstance(premise, Mapping):
        for key in ("qualified_name", "full_name", "name", "declaration"):
            if premise.get(key):
                return str(premise[key])
    return stable_json_hash(premise)


def canonical_premises(premises: Iterable[Any]) -> list[str]:
    return sorted({_premise_name(item) for item in premises})


def premise_set_hash(premises: Iterable[Any]) -> str:
    return stable_json_hash(canonical_premises(premises))


def theorem_group_id(statement: str) -> str:
    return f"theorem_{statement_hash(statement)}"


def structural_fingerprint(statement: str) -> str:
    """Remove only a declaration name; retain variables and all type syntax."""

    normalized = _TRAILING_ASSIGN.sub("", normalize_lexical(statement))
    skeleton = _DECL_KIND_AND_NAME.sub("<DECL> <DECL_NAME>", normalized, count=1)
    return exact_hash(" ".join(_TOKEN.findall(skeleton)))


def lexical_tokens(value: str) -> list[str]:
    return _TOKEN.findall(normalize_lexical(value))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(lexical_tokens(left))
    right_tokens = set(lexical_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
