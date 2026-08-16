"""Classify Lean/Pantograph diagnostics and retrieve local declarations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Iterable

from .index import (
    DeclarationCandidate,
    LocalMathlibDeclarationIndex,
    infer_math_domain,
)


class CompilerErrorCategory(str, Enum):
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    INVALID_FIELD_NOTATION = "invalid_field_notation"
    MISSING_IMPORT = "missing_import"
    AMBIGUOUS_NAMESPACE = "ambiguous_namespace"
    APPLICATION_TYPE_MISMATCH = "application_type_mismatch"
    UNKNOWN_CONSTANT = "unknown_constant"
    PARSER_OR_OLD_SYNTAX = "parser_or_old_syntax"
    OTHER = "other"


@dataclass(frozen=True)
class CompilerErrorAnalysis:
    category: CompilerErrorCategory
    diagnostic: str
    missing_name: str | None = None
    receiver_expression: str | None = None
    receiver_type: str | None = None
    expected_type: str | None = None
    actual_type: str | None = None
    missing_import: str | None = None
    current_imports: tuple[str, ...] = ()

    def prompt_record(self) -> dict[str, object]:
        record = asdict(self)
        record["category"] = self.category.value
        return record


@dataclass(frozen=True)
class RetrievedCompilerContext:
    analysis: CompilerErrorAnalysis
    candidates: tuple[DeclarationCandidate, ...]

    def prompt_record(self) -> dict[str, object]:
        return {
            "analysis": self.analysis.prompt_record(),
            "candidate_declarations": [
                candidate.prompt_record() for candidate in self.candidates
            ],
        }


_NAME = r"[A-Za-z_][A-Za-z0-9_'.]*"


def _first_group(patterns: Iterable[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return next(
                (value.strip() for value in match.groups() if value),
                None,
            )
    return None


def classify_compiler_error(
    diagnostic: str,
    *,
    current_imports: Iterable[str] = (),
) -> CompilerErrorAnalysis:
    """Extract the actionable API/name/type fragments from Lean diagnostics."""

    text = str(diagnostic or "").strip()
    lowered = text.casefold()
    missing_name = _first_group(
        (
            rf"unknown identifier\s+[`'\"]?({_NAME})",
            rf"unknown constant\s+[`'\"]?({_NAME})",
            rf"invalid field notation[\s\S]*?field\s+[`'\"]?({_NAME})",
            rf"cannot resolve field\s+[`'\"]?({_NAME})",
            rf"ambiguous(?: identifier| namespace)?\s+[`'\"]?({_NAME})",
        ),
        text,
    )
    receiver_expression = _first_group(
        (
            r"invalid field notation[\s\S]*?type of\s*\n?\s*([^\n]+)",
            r"invalid field notation[\s\S]*?the expression\s*\n?\s*([^\n]+)",
        ),
        text,
    )
    receiver_type = _first_group(
        (
            r"invalid field notation[\s\S]*?(?:has type|receiver type:)\s*\n?\s*([^\n]+)",
            r"invalid field notation[\s\S]*?\bis\s+([^\n;]+);?\s*(?:cannot resolve field|the field)",
            r"function expected at[\s\S]*?term has type\s*\n?\s*([^\n]+)",
        ),
        text,
    )
    expected_type = _first_group(
        (
            r"expected type\s*\n?\s*([^\n]+)",
            r"is expected to have type\s*\n?\s*([^\n]+)",
            r"application type mismatch[\s\S]*?but is expected to have type\s*\n?\s*([^\n]+)",
        ),
        text,
    )
    actual_type = _first_group(
        (
            r"application type mismatch[\s\S]*?has type\s*\n?\s*([^\n]+)",
            r"argument[\s\S]*?has type\s*\n?\s*([^\n]+)",
        ),
        text,
    )
    missing_import = _first_group(
        (
            rf"unknown module\s+[`'\"]?({_NAME})",
            rf"failed to import(?: module)?\s+[`'\"]?({_NAME})",
            rf"missing import\s+[`'\"]?({_NAME})",
            rf"module\s+[`'\"]?({_NAME})[`'\"]?\s+was not found",
        ),
        text,
    )

    if "invalid field notation" in lowered or "cannot resolve field" in lowered:
        category = CompilerErrorCategory.INVALID_FIELD_NOTATION
    elif missing_import or "unknown module" in lowered or "failed to import" in lowered:
        category = CompilerErrorCategory.MISSING_IMPORT
    elif "ambiguous" in lowered and ("namespace" in lowered or "identifier" in lowered):
        category = CompilerErrorCategory.AMBIGUOUS_NAMESPACE
    elif "application type mismatch" in lowered or "type mismatch" in lowered:
        category = CompilerErrorCategory.APPLICATION_TYPE_MISMATCH
    elif "unknown constant" in lowered:
        category = CompilerErrorCategory.UNKNOWN_CONSTANT
    elif "unknown identifier" in lowered:
        category = CompilerErrorCategory.UNKNOWN_IDENTIFIER
    elif any(
        marker in lowered
        for marker in (
            "parser error",
            "unexpected token",
            "unexpected end of input",
            "unexpected syntax",
            "invalid syntax",
            "deprecated syntax",
            "unknown parser",
        )
    ):
        category = CompilerErrorCategory.PARSER_OR_OLD_SYNTAX
    else:
        category = CompilerErrorCategory.OTHER

    return CompilerErrorAnalysis(
        category=category,
        diagnostic=text,
        missing_name=missing_name,
        receiver_expression=receiver_expression,
        receiver_type=receiver_type,
        expected_type=expected_type,
        actual_type=actual_type,
        missing_import=missing_import,
        current_imports=tuple(dict.fromkeys(str(item) for item in current_imports)),
    )


class CompilerErrorRetriever:
    """Turn one compiler error into 3--10 environment-pinned declarations."""

    def __init__(self, index: LocalMathlibDeclarationIndex) -> None:
        self.index = index

    def retrieve(
        self,
        diagnostic: str,
        *,
        current_imports: Iterable[str] = (),
        statement: str = "",
        informal_statement: str = "",
        limit: int = 10,
    ) -> RetrievedCompilerContext:
        limit = max(3, min(10, limit))
        analysis = classify_compiler_error(
            diagnostic,
            current_imports=current_imports,
        )
        primary_query_names: list[str] = []
        if analysis.missing_name:
            primary_query_names.append(analysis.missing_name)
        # Lean often emits a useful fully-qualified candidate after the primary
        # error.  It is still checked against the local index before prompting.
        for name in re.findall(
            rf"(?:candidate|declaration|field)\s+[`'\"]?({_NAME})",
            analysis.diagnostic,
            flags=re.IGNORECASE,
        ):
            if name not in primary_query_names:
                primary_query_names.append(name)
        if not primary_query_names and analysis.receiver_expression:
            field_match = re.search(rf"\.({_NAME})\b", statement)
            if field_match:
                primary_query_names.append(field_match.group(1))
        fallback_query_names: list[str] = []
        if len(primary_query_names) < 3:
            fallback_text = "\n".join((statement, informal_statement))
            ignored = {
                "theorem", "lemma", "nat", "int", "real", "prop", "type",
                "true", "false", "let", "fun", "forall", "exists", "if",
                "then", "else", "and", "or", "not", "mathlib",
            }
            identifiers = re.findall(_NAME, fallback_text)
            for name in reversed(identifiers):
                short = name.rsplit(".", 1)[-1]
                if short.casefold() in ignored or len(short) < 3:
                    continue
                if (
                    name not in primary_query_names
                    and name not in fallback_query_names
                ):
                    fallback_query_names.append(name)
                if len(fallback_query_names) >= 6:
                    break

        parameter_type = analysis.receiver_type or analysis.expected_type
        if not parameter_type:
            binder = re.search(
                r"[({]\s*[A-Za-z_][A-Za-z0-9_']*"
                r"(?:\s+[A-Za-z_][A-Za-z0-9_']*)*\s*:\s*([^)}]+)[)}]",
                statement,
            )
            if binder:
                parameter_type = binder.group(1).strip()
        domain = infer_math_domain(f"{informal_statement}\n{statement}")
        merged: dict[tuple[str, str, int], DeclarationCandidate] = {}
        # The unavailable compiler name is authoritative.  Context/prose
        # fallback terms may fill an undersized result but must never crowd the
        # primary name's candidates out of the 3--10 declaration window.
        for query in primary_query_names[:4]:
            for candidate in self.index.search(
                query,
                parameter_type=parameter_type,
                domain=domain,
                limit=limit,
            ):
                declaration = candidate.declaration
                key = (
                    declaration.full_name,
                    declaration.module,
                    declaration.source_line,
                )
                previous = merged.get(key)
                if previous is None or candidate.score > previous.score:
                    merged[key] = candidate
        candidates = sorted(
            merged.values(), key=lambda item: item.score, reverse=True
        )[:limit]
        if len(candidates) < limit:
            primary_keys = {
                (
                    item.declaration.full_name,
                    item.declaration.module,
                    item.declaration.source_line,
                )
                for item in candidates
            }
            fallback: list[DeclarationCandidate] = []
            for query in fallback_query_names:
                fallback.extend(
                    self.index.search(
                        query,
                        parameter_type=parameter_type,
                        domain=domain,
                        limit=limit,
                    )
                )
            fallback.sort(key=lambda item: item.score, reverse=True)
            for candidate in fallback:
                declaration = candidate.declaration
                key = (
                    declaration.full_name,
                    declaration.module,
                    declaration.source_line,
                )
                if key in primary_keys:
                    continue
                candidates.append(candidate)
                primary_keys.add(key)
                if len(candidates) >= limit:
                    break
        return RetrievedCompilerContext(
            analysis=analysis,
            candidates=tuple(candidates),
        )


__all__ = [
    "CompilerErrorAnalysis",
    "CompilerErrorCategory",
    "CompilerErrorRetriever",
    "RetrievedCompilerContext",
    "classify_compiler_error",
]
