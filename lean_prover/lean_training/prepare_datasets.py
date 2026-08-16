"""Prepare Lean SFT validation and benchmark datasets.

This module turns source datasets such as Lean-Workbook and miniF2F into a
small, stable JSONL schema consumed by the training and benchmark scripts.
Training records include a theorem statement and proof. Benchmark records keep
only the statement-side prompt so generated proofs cannot see the gold proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


LEAN_WORKBOOK_ALIASES = {"lean-workbook", "lean_workbook", "workbook"}
MINIF2F_ALIASES = {"minif2f", "miniF2F", "mini-f2f", "mini_f2f"}

DEFAULT_DATASET_IDS = {
    "lean-workbook": "InternLM/Lean-Workbook",
    "minif2f": "cat-searcher/minif2f-lean4",
}

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_LENGTH = 1024
FORBIDDEN_PROOF_RE = re.compile(r"\b(?:sorry|admit)\b")

INFORMAL_KEYS = (
    "informal_statement",
    "informal_stmt",
    "natural_language_statement",
    "nl_statement",
    "problem",
    "question",
    "description",
    "informal",
)
STATEMENT_KEYS = (
    "lean_statement",
    "formal_statement",
    "formal_stmt",
    "statement",
    "theorem",
    "decl",
    "declaration",
    "target",
)
PROOF_KEYS = (
    "tactic",
    "proof",
    "formal_proof",
    "full_proof",
    "tactic_proof",
    "completion",
    "answer",
)
CODE_KEYS = (
    "lean_code",
    "formal",
    "code",
    "source",
)


@dataclass(frozen=True)
class LeanPreamble:
    """Lean context that must appear before a theorem declaration."""

    imports: tuple[str, ...] = ()
    context_lines: tuple[str, ...] = ()
    unknown_preamble_lines: tuple[str, ...] = ()

    def to_json(self) -> dict[str, list[str]]:
        return {
            "imports": list(self.imports),
            "context_lines": list(self.context_lines),
            "unknown_preamble_lines": list(self.unknown_preamble_lines),
        }


@dataclass
class PreparationStats:
    """Count accepted records and each rejection category during normalization."""

    total: int = 0
    accepted: int = 0
    missing_statement: int = 0
    invalid_statement: int = 0
    missing_proof: int = 0
    forbidden_proof: int = 0
    invalid_imports: int = 0
    duplicate_id: int = 0
    duplicate_statement: int = 0
    statement_proof_mismatch: int = 0
    normalization_error: int = 0
    pantograph_rejected: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "missing_statement": self.missing_statement,
            "invalid_statement": self.invalid_statement,
            "missing_proof": self.missing_proof,
            "forbidden_proof": self.forbidden_proof,
            "invalid_imports": self.invalid_imports,
            "duplicate_id": self.duplicate_id,
            "duplicate_statement": self.duplicate_statement,
            "statement_proof_mismatch": self.statement_proof_mismatch,
            "normalization_error": self.normalization_error,
            "pantograph_rejected": self.pantograph_rejected,
        }


@dataclass(frozen=True)
class NormalizedExample:
    """Uniform Lean proof example used by the training/benchmark pipeline."""

    id: str
    source: str
    informal_statement: str
    lean_statement: str
    proof: str = ""
    imports: tuple[str, ...] = ()
    context_lines: tuple[str, ...] = ()
    unknown_preamble_lines: tuple[str, ...] = ()
    source_name: str = ""

    def training_text(self) -> str:
        proof = self.proof.strip()
        if not proof:
            raise ValueError(f"training example {self.id} has no proof")
        if contains_forbidden_proof_token(proof):
            raise ValueError(
                f"training example {self.id} contains forbidden proof token"
            )
        text = (
            "### Informal statement\n"
            f"{self.informal_statement.strip()}\n\n"
            "### Lean statement\n"
            f"{self.lean_statement.strip()}\n\n"
            "### Lean proof\n"
            f"{proof}"
        )
        return text

    def completion(self) -> str:
        proof = self.proof.strip()
        if not proof:
            raise ValueError(f"training example {self.id} has no proof")
        if contains_forbidden_proof_token(proof):
            raise ValueError(
                f"training example {self.id} contains forbidden proof token"
            )
        return proof

    def generation_prompt(self) -> str:
        return (
            "### Informal statement\n"
            f"{self.informal_statement.strip()}\n\n"
            "### Lean statement\n"
            f"{self.lean_statement.strip()}\n\n"
            "### Lean proof\n"
        )

    def to_grpo_training_record(self) -> dict[str, Any]:
        """Return a GRPO prompt row with theorem context and proof-audit hashes.

        Raises:
            ValueError: If the example has no usable reference proof.
        """
        proof = self.proof.strip()
        if not proof:
            raise ValueError(f"GRPO example {self.id} has no reference proof")
        if contains_forbidden_proof_token(proof):
            raise ValueError(
                f"GRPO example {self.id} contains forbidden proof token"
            )
        return {
            "id": self.id,
            "source": self.source,
            "data_kind": self.source,
            "source_name": self.source_name or self.source,
            "informal_statement": self.informal_statement,
            "lean_statement": self.lean_statement,
            "statement_hash": statement_hash(self.lean_statement),
            "prompt": self.generation_prompt(),
            "imports": list(self.imports),
            "context_lines": list(self.context_lines),
            "unknown_preamble_lines": list(self.unknown_preamble_lines),
            "preamble": LeanPreamble(
                self.imports,
                self.context_lines,
                self.unknown_preamble_lines,
            ).to_json(),
            "reference_proof_hash": proof_hash(proof),
            "has_reference_proof": True,
        }

    def to_sft_training_record(self) -> dict[str, Any]:
        """Return an SFT prompt/completion row with Lean context metadata.

        Raises:
            ValueError: If the example has no usable proof completion.
        """
        prompt = self.generation_prompt()
        completion = self.completion()
        return {
            "id": self.id,
            "source": self.source,
            "data_kind": self.source,
            "source_name": self.source_name or self.source,
            "informal_statement": self.informal_statement,
            "lean_statement": self.lean_statement,
            "proof": self.proof,
            "statement_hash": statement_hash(self.lean_statement),
            "prompt": prompt,
            "completion": completion,
            "imports": list(self.imports),
            "context_lines": list(self.context_lines),
            "unknown_preamble_lines": list(self.unknown_preamble_lines),
            "preamble": LeanPreamble(
                self.imports,
                self.context_lines,
                self.unknown_preamble_lines,
            ).to_json(),
            "text": prompt + completion,
        }

    def to_benchmark_prompt_record(self) -> dict[str, Any]:
        """Return a proof-free benchmark prompt and theorem-context row."""
        return {
            "id": self.id,
            "source": self.source,
            "data_kind": self.source,
            "source_name": self.source_name or self.source,
            "informal_statement": self.informal_statement,
            "lean_statement": self.lean_statement,
            "statement_hash": statement_hash(self.lean_statement),
            "imports": list(self.imports),
            "context_lines": list(self.context_lines),
            "unknown_preamble_lines": list(self.unknown_preamble_lines),
            "preamble": LeanPreamble(
                self.imports,
                self.context_lines,
                self.unknown_preamble_lines,
            ).to_json(),
            "prompt": self.generation_prompt(),
        }


def load_dataset_source(
    dataset_name: str,
    split: str,
    config_name: str | None = None,
    revision: str | None = None,
):
    """Load a local file or a Hugging Face dataset split.

    Args:
        dataset_name: Local path, supported alias, or Hugging Face dataset ID.
        split: Dataset split to load.
        config_name: Optional Hugging Face dataset configuration.
        revision: Optional dataset repository revision.

    Returns:
        A list-like local record collection or Hugging Face ``Dataset``.
    """

    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        return load_local_records(dataset_path)

    dataset_kind = normalize_dataset_name(dataset_name)
    if dataset_kind == "lean-workbook":
        return load_lean_workbook_records(
            dataset_name,
            split=split,
            config_name=config_name,
            revision=revision,
        )
    if dataset_kind == "minif2f":
        return load_minif2f_records(
            dataset_name,
            split=split,
            config_name=config_name,
            revision=revision,
        )

    from datasets import Dataset, DatasetDict, load_dataset

    resolved_name = DEFAULT_DATASET_IDS.get(dataset_name, dataset_name)
    loaded = (
        load_dataset(resolved_name, config_name, revision=revision)
        if config_name
        else load_dataset(resolved_name, revision=revision)
    )
    if isinstance(loaded, DatasetDict):
        return loaded[split]
    if isinstance(loaded, Dataset):
        return loaded
    raise TypeError(f"unsupported dataset object: {type(loaded)!r}")


def load_local_records(dataset_path: Path):
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "train", "validation", "test"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"unsupported JSON dataset shape: {dataset_path}")
    if suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        from datasets import load_dataset

        return load_dataset(
            "parquet",
            data_files=str(dataset_path),
            split="train",
        )
    raise ValueError(
        "local datasets must be .json, .jsonl, .csv, or .parquet files; "
        f"got {dataset_path}"
    )


def load_lean_workbook_records(
    dataset_name: str,
    *,
    split: str,
    config_name: str | None = None,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    dataset_id = DEFAULT_DATASET_IDS.get(dataset_name, dataset_name)
    filename_by_split = {
        "train": "wkbk_1009.parquet",
    }
    filename = filename_by_split.get(split)
    if filename is None:
        raise ValueError(
            "Lean-Workbook currently supports split='train' via "
            "wkbk_1009.parquet; got split={!r}".format(split)
        )
    if config_name:
        raise ValueError(
            "Lean-Workbook loader does not expose dataset configs; "
            f"got config_name={config_name!r}"
        )
    parquet_path = hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    return load_local_records(Path(parquet_path))


def load_minif2f_records(
    dataset_name: str,
    *,
    split: str,
    config_name: str | None = None,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    dataset_id = DEFAULT_DATASET_IDS.get(dataset_name, dataset_name)
    if config_name:
        raise ValueError(
            "miniF2F loader does not expose dataset configs; "
            f"got config_name={config_name!r}"
        )
    filename_by_split = {
        "validation": "valid.jsonl",
        "valid": "valid.jsonl",
        "test": "test.jsonl",
    }
    filename = filename_by_split.get(split, f"{split}.jsonl")
    jsonl_path = hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    return load_local_records(Path(jsonl_path))


def sample_dataset(dataset, sample_size: int | None, seed: int | None):
    """Shuffle and optionally take the first sample_size records."""

    if isinstance(dataset, list):
        shuffled = list(dataset)
        random.Random(seed).shuffle(shuffled)
        return shuffled[:sample_size] if sample_size is not None else shuffled

    shuffled = dataset.shuffle(seed=seed)
    if sample_size is None:
        return shuffled
    size = min(sample_size, len(shuffled))
    return shuffled.select(range(size))


def normalize_dataset_name(dataset_name: str) -> str:
    name = dataset_name.strip()
    lowered = name.lower()
    if lowered in LEAN_WORKBOOK_ALIASES:
        return "lean-workbook"
    if lowered in {alias.lower() for alias in MINIF2F_ALIASES}:
        return "minif2f"
    if "lean-workbook" in lowered:
        return "lean-workbook"
    if "minif2f" in lowered or "mini-f2f" in lowered:
        return "minif2f"
    return "generic"


def normalize_example(
    example: Mapping[str, Any],
    *,
    dataset_kind: str,
    source_name: str,
    index: int,
) -> NormalizedExample:
    """Dispatch one raw row to its schema-specific normalizer.

    Args:
        example: Raw source record.
        dataset_kind: Normalized schema kind.
        source_name: Original dataset name retained as provenance.
        index: Source index used for fallback identifiers.

    Returns:
        A uniform Lean theorem/proof example.
    """

    if dataset_kind == "lean-workbook":
        return normalize_lean_workbook_example(example, index, source_name=source_name)
    if dataset_kind == "minif2f":
        return normalize_minif2f_example(example, index, source_name=source_name)
    return normalize_generic_lean_example(
        example,
        dataset_kind,
        index,
        source_name=source_name,
    )


def normalize_lean_workbook_example(
    example: Mapping[str, Any],
    index: int,
    *,
    source_name: str = "lean-workbook",
) -> NormalizedExample:
    """Adapt Lean-Workbook records.

    Lean-Workbook stores theorem declarations in formal_statement with a
    placeholder suffix such as ``:= by sorry``. The usable proof is stored in
    tactic, so training data must strip the placeholder from formal_statement
    and pair the resulting statement with tactic.
    """

    example_id = _first_identifier_text(
        example,
        ("id", "name", "problem_id", "theorem_name", "uuid"),
        default=f"lean-workbook/{index}",
    )
    informal = _first_text(example, INFORMAL_KEYS, default="")
    raw_statement = _first_text(example, ("formal_statement",), default="")
    tactic = _first_text(example, ("tactic",), default="")
    if not raw_statement:
        raise ValueError(
            f"Lean-Workbook example {example_id!r} lacks formal_statement"
        )
    statement = strip_dataset_placeholder_proof(raw_statement)
    if not has_supported_declaration_prefix(statement):
        raise ValueError(
            f"invalid Lean-Workbook statement in example {example_id!r}"
        )
    proof = normalize_proof_rhs(tactic)
    if proof and _looks_like_full_declaration(proof):
        tactic_statement, proof = split_lean_statement_and_proof(proof)
        if not equivalent_statement(statement, tactic_statement):
            raise ValueError(
                "Lean-Workbook formal_statement/tactic declaration mismatch "
                f"in example {example_id!r}"
            )
    preamble = _extract_preamble(example)
    return NormalizedExample(
        id=str(example_id),
        source="lean-workbook",
        informal_statement=informal,
        lean_statement=statement,
        proof=proof,
        imports=preamble.imports,
        context_lines=preamble.context_lines,
        unknown_preamble_lines=preamble.unknown_preamble_lines,
        source_name=source_name,
    )


def normalize_minif2f_example(
    example: Mapping[str, Any],
    index: int,
    *,
    source_name: str = "minif2f",
) -> NormalizedExample:
    """Adapt miniF2F-style records to the common schema."""

    example_id = _first_identifier_text(
        example,
        ("id", "name", "problem_id", "theorem_name", "uuid"),
        default=f"minif2f/{index}",
    )
    informal = _first_text(example, INFORMAL_KEYS, default="")
    raw_statement = _first_text(example, STATEMENT_KEYS, default="")
    if not raw_statement:
        raise ValueError(f"miniF2F example {example_id!r} lacks Lean statement")
    statement = strip_dataset_placeholder_proof(raw_statement)
    if not has_supported_declaration_prefix(statement):
        raise ValueError(f"invalid miniF2F statement in example {example_id!r}")
    preamble = _extract_preamble(example)
    return NormalizedExample(
        id=str(example_id),
        source="minif2f",
        informal_statement=informal,
        lean_statement=statement,
        proof="",
        imports=preamble.imports,
        context_lines=preamble.context_lines,
        unknown_preamble_lines=preamble.unknown_preamble_lines,
        source_name=source_name,
    )


def normalize_generic_lean_example(
    example: Mapping[str, Any],
    source: str,
    index: int,
    *,
    source_name: str | None = None,
) -> NormalizedExample:
    """Extract a theorem, optional proof, and preamble from a generic row.

    Args:
        example: Raw mapping using supported field-name variants.
        source: Logical source/data kind.
        index: Source index used for a fallback ID.
        source_name: Optional original dataset identifier.

    Returns:
        The normalized theorem example.

    Raises:
        ValueError: If no valid statement can be extracted or fields disagree.
    """
    example_id = _first_identifier_text(
        example,
        ("id", "name", "problem_id", "theorem_name", "uuid"),
        default=f"{source}/{index}",
    )
    informal = _first_text(example, INFORMAL_KEYS, default="")
    statement = _first_text(example, STATEMENT_KEYS, default="")
    proof = _first_text(example, PROOF_KEYS, default="")
    code = _first_text(example, CODE_KEYS, default="")

    if proof and _looks_like_full_declaration(proof):
        proof_statement, proof_body = split_lean_statement_and_proof(proof)
        if statement and not equivalent_statement(statement, proof_statement):
            raise ValueError(
                f"statement/proof declaration mismatch in {source} example "
                f"{example_id!r}"
            )
        statement = statement or proof_statement
        proof = proof_body

    if not statement and code:
        statement, extracted_proof = split_lean_statement_and_proof(code)
        proof = proof or extracted_proof
    elif statement:
        statement, extracted_proof = split_lean_statement_and_proof(statement)
        proof = proof or extracted_proof

    if not proof and code and code.strip() != statement.strip():
        code_statement, extracted_proof = split_lean_statement_and_proof(code)
        if statement and code_statement and not equivalent_statement(
            statement,
            code_statement,
        ):
            raise ValueError(
                f"statement/code declaration mismatch in {source} example "
                f"{example_id!r}"
            )
        proof = extracted_proof

    preamble = _extract_preamble(example)
    statement = strip_lean_proof_body(statement)
    if not statement:
        raise ValueError(
            f"could not find Lean theorem statement in {source} example "
            f"{example_id!r}; keys={sorted(example.keys())}"
        )
    if not has_supported_declaration_prefix(statement):
        raise ValueError(
            f"invalid Lean theorem statement in {source} example {example_id!r}"
        )

    return NormalizedExample(
        id=str(example_id),
        source=source,
        informal_statement=informal,
        lean_statement=statement,
        proof=normalize_proof_rhs(proof),
        imports=preamble.imports,
        context_lines=preamble.context_lines,
        unknown_preamble_lines=preamble.unknown_preamble_lines,
        source_name=source_name or source,
    )


def split_lean_statement_and_proof(lean_code: str) -> tuple[str, str]:
    """Split a Lean declaration at its top-level proof assignment.

    Args:
        lean_code: Declaration text, optionally inside a Markdown fence.

    Returns:
        ``(statement, proof_body)``; proof is empty when no top-level assignment exists.
    """

    code = _strip_markdown_fence(lean_code).strip()
    assignment_index = find_top_level_declaration_assignment(code)
    if assignment_index is None:
        return code, ""
    statement = code[:assignment_index].rstrip()
    proof = code[assignment_index + 2 :].strip()
    return statement, normalize_proof_rhs(proof)


def find_top_level_declaration_assignment(lean_code: str) -> int | None:
    """Return the index of the declaration-level ``:=`` assignment.

    The scan skips strings, line comments, nested block comments, and bracketed
    terms so ``let x := ...`` inside a theorem type is not mistaken for the
    theorem proof separator.
    """

    code = lean_code
    block_comment_depth = 0
    in_string = False
    in_line_comment = False
    escaped = False
    brackets = {
        "(": ")",
        "[": "]",
        "{": "}",
        "\u27e8": "\u27e9",
    }
    closers = set(brackets.values())
    stack: list[str] = []
    candidates: list[int] = []
    index = 0
    while index < len(code):
        char = code[index]
        next_char = code[index + 1] if index + 1 < len(code) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if block_comment_depth:
            if char == "/" and next_char == "-":
                block_comment_depth += 1
                index += 2
                continue
            if char == "-" and next_char == "/":
                block_comment_depth -= 1
                index += 2
                continue
            index += 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == "-" and next_char == "-":
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "-":
            block_comment_depth = 1
            index += 2
            continue
        if char == '"':
            in_string = True
            index += 1
            continue

        if char in brackets:
            stack.append(brackets[char])
            index += 1
            continue
        if char in closers and stack and char == stack[-1]:
            stack.pop()
            index += 1
            continue

        if char == ":" and next_char == "=" and not stack:
            candidates.append(index)
            index += 2
            continue

        index += 1

    if not candidates:
        return None
    proof_like = [
        position
        for position in candidates
        if _rhs_starts_like_proof(code[position + 2 :])
    ]
    if proof_like:
        return proof_like[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("ambiguous Lean declaration assignment")


def _rhs_starts_like_proof(rhs: str) -> bool:
    stripped = _strip_leading_space_and_comments(rhs)
    if not stripped:
        return False
    first = _first_token(stripped)
    return first in {
        "by",
        "fun",
        "show",
        "exact",
        "calc",
        "match",
        "nomatch",
        "if",
        "let",
        "have",
        "suffices",
        "by_cases",
        "by_contra",
    }


def _strip_leading_space_and_comments(text: str) -> str:
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if text.startswith("--", index):
            newline = text.find("\n", index)
            if newline == -1:
                return ""
            index = newline + 1
            continue
        if text.startswith("/-", index):
            end = _find_block_comment_end(text, index)
            if end is None:
                return ""
            index = end
            continue
        break
    return text[index:]


def _find_block_comment_end(text: str, start: int) -> int | None:
    depth = 0
    index = start
    while index < len(text) - 1:
        pair = text[index : index + 2]
        if pair == "/-":
            depth += 1
            index += 2
            continue
        if pair == "-/":
            depth -= 1
            index += 2
            if depth == 0:
                return index
            continue
        index += 1
    return None


def _first_token(text: str) -> str:
    match = re.match(r"[A-Za-z_][A-Za-z0-9_']*", text)
    return match.group(0) if match else text[:1]


def strip_lean_proof_body(lean_code: str) -> str:
    """Return only the theorem/lemma statement, removing any proof body."""

    statement, _ = split_lean_statement_and_proof(lean_code)
    return statement.strip()


def strip_dataset_placeholder_proof(lean_code: str) -> str:
    """Remove dataset proof placeholders while keeping only the declaration."""

    statement = strip_lean_proof_body(lean_code)
    statement = re.sub(
        r"(?is)\s*:?\s*=\s*by\s+(?:sorry|admit)\s*$",
        "",
        statement,
    )
    statement = re.sub(
        r"(?is)\s+by\s+(?:sorry|admit)\s*$",
        "",
        statement,
    )
    statement = re.sub(r"(?is)\s+(?:sorry|admit)\s*$", "", statement)
    return statement.strip()


def normalize_proof_rhs(proof: str) -> str:
    """Normalize a declaration RHS without rewriting term proofs as tactics."""

    proof = _strip_markdown_fence(proof).strip()
    if proof.startswith(":="):
        proof = proof[2:].strip()
    return proof


def cleanup_proof_body(proof: str) -> str:
    """Backward-compatible wrapper for proof RHS normalization."""

    return normalize_proof_rhs(proof)


def compose_lean_theorem(lean_statement: str, proof: str) -> str:
    """Combine a proof-free Lean declaration with a generated proof body.

    Args:
        lean_statement: Target theorem/lemma declaration without its proof.
        proof: Proof term, tactic block, or matching complete declaration.

    Returns:
        A complete Lean declaration ready for compilation.

    Raises:
        ValueError: If the proof is empty or a complete declaration targets a
            different statement.
    """

    statement = strip_lean_proof_body(lean_statement)
    proof = normalize_proof_rhs(proof)
    if _looks_like_full_declaration(proof):
        proof_statement, proof_body = split_lean_statement_and_proof(proof)
        if not equivalent_statement(statement, proof_statement):
            raise ValueError("generated declaration does not match target statement")
        proof = proof_body
    if not proof.strip():
        raise ValueError("empty proof RHS")
    return f"{statement} := {proof.strip()}"


def contains_forbidden_proof_token(text: str) -> bool:
    """Return True when text uses placeholders that must not count as proofs."""

    return any(token in {"sorry", "admit", "sorryAx"} for token in lean_code_tokens(text))


def lean_code_tokens(text: str) -> Iterable[str]:
    """Yield Lean-like identifiers outside strings and comments."""

    block_comment_depth = 0
    in_string = False
    in_line_comment = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            index += 1
            continue
        if block_comment_depth:
            if char == "/" and next_char == "-":
                block_comment_depth += 1
                index += 2
                continue
            if char == "-" and next_char == "/":
                block_comment_depth -= 1
                index += 2
                continue
            index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == "-" and next_char == "-":
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "-":
            block_comment_depth = 1
            index += 2
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < len(text) and (
                text[index].isalnum() or text[index] in {"_", "'"}
            ):
                index += 1
            yield text[start:index]
            continue
        index += 1


def has_supported_declaration_prefix(statement: str) -> bool:
    stripped = statement.lstrip()
    return stripped.startswith("theorem ") or stripped.startswith("lemma ")


def equivalent_statement(left: str, right: str) -> bool:
    """Compare two declaration statements after proof/body cleanup."""

    left_statement = strip_lean_proof_body(left)
    right_statement = strip_lean_proof_body(right)
    return _normalize_statement_for_compare(left_statement) == _normalize_statement_for_compare(
        right_statement
    )


def statement_hash(statement: str) -> str:
    normalized = _normalize_statement_for_compare(strip_lean_proof_body(statement))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def proof_hash(proof: str) -> str:
    normalized = re.sub(r"\s+", " ", normalize_proof_rhs(proof).strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_records(
    dataset,
    *,
    dataset_kind: str,
    source_name: str | None = None,
    require_proof: bool,
    limit: int | None = None,
) -> list[NormalizedExample]:
    """Normalize, validate, deduplicate, and optionally limit source rows.

    Args:
        dataset: Iterable source dataset.
        dataset_kind: Schema adapter selected for the source.
        source_name: Optional provenance name stored on output records.
        require_proof: Whether rows without a valid proof must be rejected.
        limit: Optional maximum number of accepted records.

    Returns:
        Accepted normalized examples in source order.

    Side Effects:
        Prints aggregate preparation and rejection statistics.
    """
    if dataset_kind == "minif2f" and require_proof:
        raise ValueError(
            "miniF2F normalization currently produces benchmark prompts only; "
            "do not use dataset_kind=minif2f with require_proof=True"
        )
    records: list[NormalizedExample] = []
    stats = PreparationStats()
    seen_ids: set[str] = set()
    seen_statement_hashes: set[str] = set()
    for index, example in enumerate(dataset):
        stats.total += 1
        if not _has_valid_imports_shape(example):
            stats.invalid_imports += 1
            continue
        try:
            normalized = normalize_example(
                example,
                dataset_kind=dataset_kind,
                source_name=source_name or dataset_kind,
                index=index,
            )
        except ValueError as error:
            _record_normalization_error(stats, error)
            continue
        if normalized.id in seen_ids:
            stats.duplicate_id += 1
            continue
        seen_ids.add(normalized.id)
        normalized_hash = statement_hash(normalized.lean_statement)
        if normalized_hash in seen_statement_hashes:
            stats.duplicate_statement += 1
            continue
        seen_statement_hashes.add(normalized_hash)
        if not normalized.lean_statement.strip():
            stats.missing_statement += 1
            continue
        if not has_supported_declaration_prefix(normalized.lean_statement):
            stats.invalid_statement += 1
            continue
        if require_proof and not normalized.proof.strip():
            stats.missing_proof += 1
            continue
        if normalized.proof and contains_forbidden_proof_token(normalized.proof):
            stats.forbidden_proof += 1
            continue
        if require_proof:
            try:
                normalized.training_text()
            except ValueError as error:
                _record_normalization_error(stats, error)
                continue
        records.append(normalized)
        stats.accepted += 1
        if limit is not None and len(records) >= limit:
            break
    print(
        "PREPARATION_STATS "
        + json.dumps(
            {
                "data_kind": dataset_kind,
                "require_proof": require_proof,
                **stats.to_json(),
            },
            ensure_ascii=False,
        )
    )
    return records


def _record_normalization_error(stats: PreparationStats, error: ValueError) -> None:
    message = str(error).lower()
    if "lacks formal_statement" in message or "lacks lean statement" in message:
        stats.missing_statement += 1
    elif "could not find lean theorem statement" in message:
        stats.missing_statement += 1
    elif "invalid" in message and "statement" in message:
        stats.invalid_statement += 1
    elif "mismatch" in message:
        stats.statement_proof_mismatch += 1
    elif "no proof" in message:
        stats.missing_proof += 1
    elif "forbidden" in message or "sorry" in message or "admit" in message:
        stats.forbidden_proof += 1
    else:
        stats.normalization_error += 1


def validate_records_with_pantograph(
    records: Iterable[NormalizedExample],
    *,
    lean_project_path: str,
    imports: tuple[str, ...] = ("Mathlib",),
    timeout: int = 120,
    require_proof: bool,
    num_workers: int = 1,
    rejected_output_path: Path | None = None,
) -> list[NormalizedExample]:
    """Compile normalized records with persistent Pantograph workers.

    Args:
        records: Normalized examples to validate.
        lean_project_path: Lean/Lake project used for compilation.
        imports: Imports loaded by each Pantograph server.
        timeout: Per-record and warmup timeout in seconds.
        require_proof: Compile complete proofs when true; otherwise validate statements.
        num_workers: Number of Pantograph worker processes.
        rejected_output_path: Optional JSONL destination for rejected rows.

    Returns:
        Records accepted by Pantograph, preserving original order.

    Raises:
        RuntimeError: If the verification pool reports a fatal backend error.
    """

    from lean_prover.lean_training.verification_pool import (
        VerificationPoolConfig,
        run_verification_pool,
    )
    from lean_prover.lean_training.verification_schema import (
        VerificationTask,
    )

    records_list = list(records)
    accepted: list[NormalizedExample] = []
    pre_rejected = 0
    rejected_rows: list[dict[str, Any]] = []
    tasks: list[VerificationTask] = []
    for record_index, record in enumerate(records_list):
        if _record_contains_forbidden_token(record):
            pre_rejected += 1
            rejected_rows.append(
                {
                    "id": record.id,
                    "record_index": record_index,
                    "reason": "forbidden token in statement/proof/context",
                }
            )
            continue
        if not has_supported_declaration_prefix(record.lean_statement):
            pre_rejected += 1
            rejected_rows.append(
                {
                    "id": record.id,
                    "record_index": record_index,
                    "reason": "unsupported declaration prefix",
                }
            )
            continue
        try:
            declaration, reject_forbidden = _record_validation_declaration(
                record,
                require_proof=require_proof,
            )
        except ValueError:
            pre_rejected += 1
            rejected_rows.append(
                {
                    "id": record.id,
                    "record_index": record_index,
                    "reason": "failed to build validation declaration",
                }
            )
            continue
        tasks.append(
            VerificationTask(
                priority=record_index,
                problem_index=record_index,
                attempt_index=0,
                problem_id=record.id,
                prompt=record.generation_prompt(),
                generated_proof=record.proof,
                raw_completion=record.proof,
                lean_code=declaration,
                imports=record.imports or imports,
                context_lines=record.context_lines,
                enqueue_time=0.0,
                payload={"record_index": record_index},
                reject_forbidden=reject_forbidden,
            )
        )

    accepted_indices: set[int] = set()

    def on_result(result: Mapping[str, Any]) -> None:
        if result.get("success"):
            accepted_indices.add(int(result["record_index"]))
        else:
            rejected_rows.append(dict(result))

    run = run_verification_pool(
        tasks,
        VerificationPoolConfig(
            lean_project_path=lean_project_path,
            imports=imports,
            timeout=timeout,
            warmup_timeout=timeout,
            num_workers=max(1, num_workers),
            cancel_on_success=False,
        ),
        on_result=on_result,
    )
    if run.fatal_errors:
        raise RuntimeError(
            "Pantograph validation failed: " + "; ".join(run.fatal_errors)
        )
    accepted = [record for index, record in enumerate(records_list) if index in accepted_indices]
    if rejected_output_path is not None:
        write_jsonl(rejected_rows, rejected_output_path)
    print(
        "PANTOGRAPH_VALIDATION_STATS "
        + json.dumps(
            {
                "before": len(records_list),
                "pre_rejected": pre_rejected,
                "compiled": len(tasks),
                "accepted": len(accepted),
                "pantograph_rejected": len(tasks) - len(accepted),
                "num_workers": max(1, num_workers),
            },
            ensure_ascii=False,
        )
    )
    return accepted


def should_validate_dataset_with_pantograph(dataset_kind: str) -> bool:
    """Only generic datasets need compile-backed normalization validation."""

    return dataset_kind == "generic"


def _record_contains_forbidden_token(record: NormalizedExample) -> bool:
    fields = [record.lean_statement, record.proof, *record.context_lines]
    return any(contains_forbidden_proof_token(field) for field in fields if field)


def _record_validation_declaration(
    record: NormalizedExample,
    *,
    require_proof: bool,
) -> tuple[str, bool]:
    if require_proof:
        declaration = compose_lean_theorem(record.lean_statement, record.proof)
        return declaration, True
    statement = strip_lean_proof_body(record.lean_statement)
    if not statement:
        raise ValueError(f"record {record.id} has no statement")
    declaration = f"{statement} := by\n  sorry"
    return declaration, False


def build_lean_source_with_preamble(
    declaration: str,
    *,
    context_lines: Iterable[str] = (),
    namespace: str | None = None,
    label: str | None = None,
) -> str:
    """Wrap a declaration with context lines and balanced Lean scopes.

    Args:
        declaration: Complete Lean declaration to place in the source.
        context_lines: Namespace, section, variable, and other context lines.
        namespace: Optional outer namespace.
        label: Optional identifying source comment.

    Returns:
        Compile-ready Lean source ending with a newline.
    """
    context = tuple(line.strip() for line in context_lines if line.strip())
    body: list[str] = []
    if namespace:
        body.append(f"namespace {namespace}")
        body.append("")
    if context:
        body.extend(context)
        body.append("")
    if label:
        body.append(f"-- label: {label}")
    body.append(declaration.strip())
    closing_lines = _context_closing_lines(context)
    if closing_lines:
        body.append("")
        body.extend(closing_lines)
    if namespace:
        body.append("")
        body.append(f"end {namespace}")
    return "\n".join(body).strip() + "\n"


def _context_closing_lines(context_lines: Iterable[str]) -> tuple[str, ...]:
    stack: list[str] = []
    for line in context_lines:
        stripped = line.strip()
        if stripped.startswith("namespace "):
            stack.append("namespace")
        elif stripped == "section" or stripped.startswith("section "):
            stack.append("section")
        elif stripped == "end" or stripped.startswith("end "):
            if stack:
                stack.pop()
    return tuple("end" for _ in reversed(stack))


def split_training_validation_records(
    records: list[NormalizedExample],
    *,
    validation_ratio: float,
    seed: int | None,
) -> tuple[list[NormalizedExample], list[NormalizedExample]]:
    """Split normalized training records into train and in-training validation sets."""

    if validation_ratio <= 0 or len(records) < 2:
        return records, []
    validation_count = int(round(len(records) * validation_ratio))
    validation_count = min(max(validation_count, 1), len(records) - 1)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_ids = {record.id for record in shuffled[:validation_count]}
    train_records = [record for record in records if record.id not in validation_ids]
    validation_records = [record for record in records if record.id in validation_ids]
    return train_records, validation_records


def token_length_summary(
    texts: Iterable[str],
    *,
    tokenizer_name_or_path: str,
    max_length: int,
) -> dict[str, Any]:
    """Compute token length percentiles and truncation ratio for texts."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    lengths = [
        len(tokenizer(text, add_special_tokens=True)["input_ids"])
        for text in texts
    ]
    if not lengths:
        return {
            "num_records": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "max_length": max_length,
            "truncation_ratio": 0.0,
        }
    sorted_lengths = sorted(lengths)
    truncated = sum(1 for length in lengths if length > max_length)
    return {
        "num_records": len(lengths),
        "median": _percentile(sorted_lengths, 0.50),
        "p90": _percentile(sorted_lengths, 0.90),
        "p95": _percentile(sorted_lengths, 0.95),
        "p99": _percentile(sorted_lengths, 0.99),
        "max": sorted_lengths[-1],
        "max_length": max_length,
        "truncation_ratio": truncated / len(lengths),
    }


def filter_records_by_token_length(
    records: list[NormalizedExample],
    *,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    min_completion_tokens: int,
    enabled: bool,
) -> list[NormalizedExample]:
    """Optionally remove records that would be silently truncated by SFT."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    kept: list[NormalizedExample] = []
    counters = {
        "total": len(records),
        "kept": 0,
        "prompt_too_long": 0,
        "sequence_too_long": 0,
        "completion_too_short": 0,
        "empty_completion": 0,
    }
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    for record in records:
        prompt = record.generation_prompt()
        completion = record.completion()
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        completion_tokens = len(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )
        total_tokens = len(
            tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
        )
        prompt_lengths.append(prompt_tokens)
        completion_lengths.append(completion_tokens)
        total_lengths.append(total_tokens)
        reason = None
        if not completion.strip():
            reason = "empty_completion"
        elif completion_tokens < min_completion_tokens:
            reason = "completion_too_short"
        elif prompt_tokens >= max_seq_length:
            reason = "prompt_too_long"
        elif total_tokens > max_seq_length:
            reason = "sequence_too_long"
        if reason is not None:
            counters[reason] += 1
            if enabled:
                continue
        kept.append(record)
    counters["kept"] = len(kept)
    summary = {
        **counters,
        "max_seq_length": max_seq_length,
        "min_completion_tokens": min_completion_tokens,
        "filter_enabled": enabled,
        "prompt_tokens": _length_distribution(prompt_lengths),
        "completion_tokens": _length_distribution(completion_lengths),
        "total_tokens": _length_distribution(total_lengths),
        "overlength_ratio": (
            counters["sequence_too_long"] / len(records) if records else 0.0
        ),
    }
    print("LENGTH_FILTER_STATS " + json.dumps(summary, ensure_ascii=False))
    return kept


def filter_grpo_records_by_prompt_length(
    records: list[NormalizedExample],
    *,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    enabled: bool,
) -> list[NormalizedExample]:
    """Optionally remove GRPO records whose prompts would exceed context."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    kept: list[NormalizedExample] = []
    prompt_lengths: list[int] = []
    prompt_too_long = 0
    for record in records:
        prompt_tokens = len(
            tokenizer(record.generation_prompt(), add_special_tokens=True)["input_ids"]
        )
        prompt_lengths.append(prompt_tokens)
        if prompt_tokens >= max_seq_length:
            prompt_too_long += 1
            if enabled:
                continue
        kept.append(record)
    print(
        "GRPO_PROMPT_LENGTH_FILTER_STATS "
        + json.dumps(
            {
                "total": len(records),
                "kept": len(kept),
                "prompt_too_long": prompt_too_long,
                "max_seq_length": max_seq_length,
                "filter_enabled": enabled,
                "prompt_tokens": _length_distribution(prompt_lengths),
                "overlength_ratio": (
                    prompt_too_long / len(records) if records else 0.0
                ),
            },
            ensure_ascii=False,
        )
    )
    return kept


def _length_distribution(lengths: list[int]) -> dict[str, int | None]:
    if not lengths:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    sorted_lengths = sorted(lengths)
    return {
        "p50": _percentile(sorted_lengths, 0.50),
        "p90": _percentile(sorted_lengths, 0.90),
        "p95": _percentile(sorted_lengths, 0.95),
        "p99": _percentile(sorted_lengths, 0.99),
        "max": sorted_lengths[-1],
    }


def print_token_length_summary(label: str, summary: Mapping[str, Any]) -> None:
    print(
        "TOKEN_LENGTH_SUMMARY "
        + json.dumps({"label": label, **dict(summary)}, ensure_ascii=False)
    )


def _percentile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty values")
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    """Atomically write JSON-serializable rows to a JSONL file.

    Args:
        records: Rows to serialize in iteration order.
        output_path: Destination whose parent directories are created as needed.

    Output:
        Replaces the destination only after its temporary file is fully written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    tmp_path.replace(output_path)


def _first_text(
    example: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    default: str,
) -> str:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _first_identifier_text(
    example: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    default: str,
) -> str:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, float):
            return str(value)
    return default


def _has_valid_imports_shape(example: Mapping[str, Any]) -> bool:
    imports = example.get("imports")
    if imports is not None and not isinstance(imports, (str, list, tuple)):
        return False
    for key in ("context_lines", "preamble_context", "open_lines"):
        value = example.get(key)
        if value is not None and not isinstance(value, (str, list, tuple)):
            return False
    return True


def _extract_imports(example: Mapping[str, Any]) -> tuple[str, ...]:
    return _extract_preamble(example).imports


def _extract_preamble(example: Mapping[str, Any]) -> LeanPreamble:
    import_modules: list[str] = []
    context_lines: list[str] = []
    unknown_preamble_lines: list[str] = []

    def add_import(value: Any) -> None:
        module = str(value).strip().removeprefix("import ").strip()
        if module and module not in import_modules:
            import_modules.append(module)

    def add_context_line(value: Any) -> None:
        line = str(value).strip()
        if line:
            context_lines.append(line)

    def add_unknown_line(value: Any) -> None:
        line = str(value).strip()
        if line:
            unknown_preamble_lines.append(line)
            context_lines.append(line)

    def add_preamble_item(value: Any) -> None:
        line = str(value).strip()
        if not line:
            return
        if line.startswith("import "):
            add_import(line)
        elif _is_context_preamble_line(line):
            add_context_line(line)
        elif _looks_like_import_module(line):
            add_import(line)
        else:
            add_unknown_line(line)

    imports_value = example.get("imports")
    if isinstance(imports_value, str):
        for item in imports_value.splitlines():
            add_preamble_item(item)
    elif isinstance(imports_value, (list, tuple)):
        for item in imports_value:
            add_preamble_item(item)

    for key in ("context_lines", "preamble_context", "open_lines"):
        value = example.get(key)
        if isinstance(value, str):
            for line in value.splitlines():
                add_context_line(line)
        elif isinstance(value, (list, tuple)):
            for line in value:
                add_context_line(line)

    header = _first_text(example, ("header", "src_header", "import_block"), default="")
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("import "):
            add_import(line)
        elif _is_context_preamble_line(line):
            add_context_line(line)
        elif line:
            add_unknown_line(line)
    return LeanPreamble(
        imports=tuple(import_modules),
        context_lines=tuple(context_lines),
        unknown_preamble_lines=tuple(unknown_preamble_lines),
    )


def _is_context_preamble_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(
        (
            "open ",
            "open scoped ",
            "namespace ",
            "section",
            "variable ",
            "variables ",
            "universe ",
            "universes ",
            "noncomputable section",
            "set_option ",
            "local notation ",
            "local instance ",
            "notation ",
            "attribute ",
            "include ",
            "omit ",
        )
    )


def _looks_like_import_module(line: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9_']*(?:\.[A-Z][A-Za-z0-9_']*)*", line))


def _normalize_statement_for_compare(statement: str) -> str:
    statement = _strip_markdown_fence(statement).strip()
    return re.sub(r"\s+", " ", statement)


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _looks_like_full_declaration(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("theorem ") or stripped.startswith("lemma ")


def _indent_proof(proof: str) -> str:
    lines = proof.splitlines() or [proof]
    return "\n".join(
        line if not line.strip() else f"  {line}" for line in lines
    )


def parse_args() -> argparse.Namespace:
    """Parse dataset, filtering, validation, and output CLI options.

    Returns:
        The preparation command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Prepare Lean-Workbook training data and miniF2F benchmark data."
    )
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--train_dataset_config", default=None)
    parser.add_argument(
        "--train_data_kind",
        default=None,
        help="Optional schema kind override, e.g. lean-workbook for a local file.",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--train_output", default=None)
    parser.add_argument("--train_rejected_output", default=None)
    parser.add_argument("--validation_output", default=None)
    parser.add_argument("--grpo_train_output", default=None)
    parser.add_argument("--grpo_validation_output", default=None)
    parser.add_argument("--validation_ratio", type=float, default=0.02)
    parser.add_argument("--benchmark_dataset_name", default=None)
    parser.add_argument("--benchmark_dataset_config", default=None)
    parser.add_argument(
        "--benchmark_data_kind",
        default=None,
        help="Optional schema kind override, e.g. minif2f for a local file.",
    )
    parser.add_argument("--benchmark_split", default="test")
    parser.add_argument("--benchmark_sample_size", type=int, default=None)
    parser.add_argument("--benchmark_output", default=None)
    parser.add_argument("--benchmark_rejected_output", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Deprecated alias for --max_seq_length.",
    )
    parser.add_argument("--filter_overlength", action="store_true")
    parser.add_argument("--min_completion_tokens", type=int, default=1)
    parser.add_argument(
        "--verify_with_pantograph",
        action="store_true",
        help="Filter generic normalized records through warm Pantograph workers.",
    )
    parser.add_argument(
        "--lean_project_path",
        default=str(Path(__file__).resolve().parents[2] / "lean_project"),
    )
    parser.add_argument("--lean_timeout", type=int, default=120)
    parser.add_argument("--verification_workers", type=int, default=1)
    parser.add_argument(
        "--pantograph_imports",
        default="Mathlib",
        help="Comma-separated imports used to start the validation server.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for sampling and train/validation splitting. "
            "When omitted, each run uses a fresh random order."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Prepare requested SFT, GRPO, and/or benchmark JSONL artifacts."""
    args = parse_args()
    if args.max_length is not None:
        print("WARNING: --max_length is deprecated; use --max_seq_length")
        args.max_seq_length = args.max_length
    if not args.train_output and not args.benchmark_output and not args.grpo_train_output:
        raise ValueError(
            "provide --train_output, --grpo_train_output, and/or --benchmark_output"
        )

    if args.train_output:
        if not args.train_dataset_name:
            raise ValueError("--train_dataset_name is required with --train_output")
        train_dataset = load_dataset_source(
            args.train_dataset_name,
            split=args.train_split,
            config_name=args.train_dataset_config,
            revision=args.dataset_revision,
        )
        train_dataset = sample_dataset(train_dataset, None, args.seed)
        train_kind = normalize_dataset_name(
            args.train_data_kind or args.train_dataset_name
        )
        train_records = normalize_records(
            train_dataset,
            dataset_kind=train_kind,
            source_name=args.train_dataset_name,
            require_proof=True,
            limit=args.train_sample_size,
        )
        pantograph_imports = tuple(
            item.strip() for item in args.pantograph_imports.split(",") if item.strip()
        )
        if args.verify_with_pantograph and should_validate_dataset_with_pantograph(
            train_kind
        ):
            before_count = len(train_records)
            train_records = validate_records_with_pantograph(
                train_records,
                lean_project_path=args.lean_project_path,
                imports=pantograph_imports,
                timeout=args.lean_timeout,
                require_proof=True,
                num_workers=args.verification_workers,
                rejected_output_path=(
                    Path(args.train_rejected_output)
                    if args.train_rejected_output
                    else None
                ),
            )
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "train",
                        "before": before_count,
                        "after": len(train_records),
                    },
                    ensure_ascii=False,
                )
            )
        elif args.verify_with_pantograph:
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "train",
                        "data_kind": train_kind,
                        "skipped": True,
                        "reason": "specialized normalizer does not require Lean compile validation",
                    },
                    ensure_ascii=False,
                )
            )
        train_records = filter_records_by_token_length(
            train_records,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            max_seq_length=args.max_seq_length,
            min_completion_tokens=args.min_completion_tokens,
            enabled=args.filter_overlength,
        )
        print_token_length_summary(
            "train_text_before_validation_split",
            token_length_summary(
                (record.training_text() for record in train_records),
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                max_length=args.max_seq_length,
            ),
        )
        validation_records: list[NormalizedExample] = []
        if args.validation_output:
            train_records, validation_records = split_training_validation_records(
                train_records,
                validation_ratio=args.validation_ratio,
                seed=None if args.seed is None else args.seed + 1,
            )
        write_jsonl(
            (record.to_sft_training_record() for record in train_records),
            Path(args.train_output),
        )
        print(f"wrote {len(train_records)} training records to {args.train_output}")
        if args.validation_output:
            write_jsonl(
                (record.to_sft_training_record() for record in validation_records),
                Path(args.validation_output),
            )
            print(
                f"wrote {len(validation_records)} in-training validation records "
                f"to {args.validation_output}"
            )

    if args.grpo_train_output:
        if not args.train_dataset_name:
            raise ValueError("--train_dataset_name is required with --grpo_train_output")
        train_kind = normalize_dataset_name(
            args.train_data_kind or args.train_dataset_name
        )
        if train_kind != "lean-workbook":
            raise ValueError(
                "GRPO preparation currently supports only Lean-Workbook; "
                f"got {args.train_dataset_name!r} ({train_kind})"
            )
        grpo_dataset = load_dataset_source(
            args.train_dataset_name,
            split=args.train_split,
            config_name=args.train_dataset_config,
            revision=args.dataset_revision,
        )
        grpo_dataset = sample_dataset(grpo_dataset, None, args.seed)
        grpo_records = normalize_records(
            grpo_dataset,
            dataset_kind=train_kind,
            source_name=args.train_dataset_name,
            require_proof=True,
            limit=args.train_sample_size,
        )
        if args.verify_with_pantograph:
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "grpo_train",
                        "data_kind": train_kind,
                        "skipped": True,
                        "reason": "Lean-Workbook uses specialized normalization for GRPO preparation",
                    },
                    ensure_ascii=False,
                )
            )
        grpo_records = filter_grpo_records_by_prompt_length(
            grpo_records,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            max_seq_length=args.max_seq_length,
            enabled=args.filter_overlength,
        )
        print_token_length_summary(
            "grpo_prompt_before_validation_split",
            token_length_summary(
                (record.generation_prompt() for record in grpo_records),
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                max_length=args.max_seq_length,
            ),
        )
        grpo_validation_records: list[NormalizedExample] = []
        if args.grpo_validation_output:
            grpo_records, grpo_validation_records = split_training_validation_records(
                grpo_records,
                validation_ratio=args.validation_ratio,
                seed=None if args.seed is None else args.seed + 1,
            )
        write_jsonl(
            (record.to_grpo_training_record() for record in grpo_records),
            Path(args.grpo_train_output),
        )
        print(f"wrote {len(grpo_records)} GRPO training records to {args.grpo_train_output}")
        if args.grpo_validation_output:
            write_jsonl(
                (
                    record.to_grpo_training_record()
                    for record in grpo_validation_records
                ),
                Path(args.grpo_validation_output),
            )
            print(
                f"wrote {len(grpo_validation_records)} GRPO validation records "
                f"to {args.grpo_validation_output}"
            )

    if args.benchmark_output:
        if not args.benchmark_dataset_name:
            raise ValueError("--benchmark_dataset_name is required with --benchmark_output")
        benchmark_dataset = load_dataset_source(
            args.benchmark_dataset_name,
            split=args.benchmark_split,
            config_name=args.benchmark_dataset_config,
            revision=args.dataset_revision,
        )
        benchmark_dataset = sample_dataset(benchmark_dataset, None, args.seed)
        benchmark_kind = normalize_dataset_name(
            args.benchmark_data_kind or args.benchmark_dataset_name
        )
        benchmark_records = normalize_records(
            benchmark_dataset,
            dataset_kind=benchmark_kind,
            source_name=args.benchmark_dataset_name,
            require_proof=False,
            limit=args.benchmark_sample_size,
        )
        pantograph_imports = tuple(
            item.strip() for item in args.pantograph_imports.split(",") if item.strip()
        )
        if args.verify_with_pantograph and should_validate_dataset_with_pantograph(
            benchmark_kind
        ):
            before_count = len(benchmark_records)
            benchmark_records = validate_records_with_pantograph(
                benchmark_records,
                lean_project_path=args.lean_project_path,
                imports=pantograph_imports,
                timeout=args.lean_timeout,
                require_proof=False,
                num_workers=args.verification_workers,
                rejected_output_path=(
                    Path(args.benchmark_rejected_output)
                    if args.benchmark_rejected_output
                    else None
                ),
            )
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "benchmark",
                        "before": before_count,
                        "after": len(benchmark_records),
                    },
                    ensure_ascii=False,
                )
            )
        elif args.verify_with_pantograph:
            print(
                "PANTOGRAPH_VALIDATION "
                + json.dumps(
                    {
                        "label": "benchmark",
                        "data_kind": benchmark_kind,
                        "skipped": True,
                        "reason": "specialized normalizer does not require Lean compile validation",
                    },
                    ensure_ascii=False,
                )
            )
        print_token_length_summary(
            "benchmark_prompt",
            token_length_summary(
                (record.generation_prompt() for record in benchmark_records),
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                max_length=args.max_seq_length,
            ),
        )
        write_jsonl(
            (record.to_benchmark_prompt_record() for record in benchmark_records),
            Path(args.benchmark_output),
        )
        print(f"wrote {len(benchmark_records)} benchmark records to {args.benchmark_output}")


if __name__ == "__main__":
    main()



