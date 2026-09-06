"""Pantograph-backed reward functions for GRPO training."""

from __future__ import annotations

import atexit
import json
import time
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.data.training import proof_length_metrics
from lean_prover.lean_training.data.preparation import (
    contains_forbidden_proof_token,
)
from lean_prover.lean_training.evaluation.assembler import assemble_grpo_completion
from lean_prover.lean_training.grpo_pipeline.config import GRPOTrainConfig
from lean_prover.lean_training.verification.pantograph import (
    PantographCheckResult,
    PantographTheoremVerifier,
)


class PantographRewardFunction:
    """Compile generated Lean proofs and convert verification outcomes to rewards."""

    # TRL names custom callables through ``__name__`` during trainer setup.
    __name__ = "pantograph_reward"

    def __init__(self, config: GRPOTrainConfig) -> None:
        self.config = config
        self.imports = tuple(
            item.strip() for item in config.pantograph_imports.split(",") if item.strip()
        )
        self.verifier: PantographTheoremVerifier | None = None
        self.reward_log_path = (
            Path(config.reward_log_file).expanduser()
            if config.reward_log_file
            else None
        )
        if self.reward_log_path:
            self.reward_log_path.parent.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        """Score a trainer batch of completions in input order."""

        rewards: list[float] = []
        for index, completion in enumerate(completions):
            row = self._row_kwargs(kwargs, index)
            rewards.append(self.score_completion(completion, row, index))
        return rewards

    def score_completion(
        self,
        completion: Any,
        row: dict[str, Any],
        index: int,
    ) -> float:
        """Validate, compile, log, and score one generated proof."""

        problem_id = str(row.get(self.config.id_field, f"row/{index}"))
        statement = str(row.get(self.config.statement_field, "") or "")
        proof = extract_completion_text(completion)
        generated_lengths = proof_length_metrics(proof)
        reference_tokens = coerce_positive_int(
            row.get("reference_proof_length_tokens")
        )
        log_row: dict[str, Any] = {
            "problem_id": problem_id,
            "reward_index": index,
            "statement_hash": row.get("statement_hash"),
            "generated_proof": proof,
            "generated_proof_structure": classify_proof_structure(proof),
            "generated_proof_length_tokens": generated_lengths["tokens"],
            "generated_proof_length_characters": generated_lengths["characters"],
            "generated_proof_length_lines": generated_lengths["lines"],
            "reference_proof_length_tokens": reference_tokens,
            "reference_proof_length_characters": row.get(
                "reference_proof_length_characters"
            ),
            "reference_proof_length_lines": row.get("reference_proof_length_lines"),
            "success": False,
            "reward": self.config.failure_reward,
            "reward_components": {
                "compile": 0.0,
                "format": 0.0,
                "brevity": 0.0,
            },
            "reason": "",
            "diagnostics": "",
        }
        start = time.monotonic()
        try:
            if not proof.strip():
                log_row["reason"] = "empty_completion"
                return self._finish(log_row, start)
            if contains_forbidden_proof_token(proof):
                log_row["reason"] = "forbidden_proof_token"
                return self._finish(log_row, start)
            log_row["reward_components"]["format"] = self.config.format_reward
            assembly_row = dict(row)
            assembly_row.setdefault("id", problem_id)
            assembly_row.setdefault("lean_statement", statement)
            assembly_row.setdefault("imports", self.imports)
            attempt_value = row.get("attempt_index")
            attempt_index = int(attempt_value) if isinstance(attempt_value, int) and attempt_value >= 0 else index
            assembled = assemble_grpo_completion(
                assembly_row,
                proof,
                problem_index=index,
                attempt_index=attempt_index,
            )
            source = assembled.source
            result = self._verifier().check_source(
                source,
                timeout=self.config.lean_timeout,
                reject_forbidden=True,
            )
            log_row.update(
                {
                    "lean_code": source,
                    "success": result.success,
                    "diagnostics": result.diagnostics,
                    "compile_messages": list(result.messages),
                    "compile_errors": list(result.errors),
                    "compile_warnings": list(result.warnings),
                    "verification_seconds": result.check_seconds,
                    "timed_out": result.timed_out,
                    **extract_compile_feedback(result),
                }
            )
            if result.success:
                log_row["reward_components"][
                    "compile"
                ] = self.config.compile_success_reward
                generated_tokens = generated_lengths["tokens"]
                if (
                    reference_tokens is not None
                    and generated_tokens < reference_tokens
                ):
                    reduction_ratio = (
                        reference_tokens - generated_tokens
                    ) / reference_tokens
                    log_row["reward_components"]["brevity"] = (
                        self.config.brevity_reward * reduction_ratio
                    )
                    log_row["brevity_reduction_ratio"] = reduction_ratio
                log_row["reward"] = sum(log_row["reward_components"].values())
                log_row["reason"] = "compiled"
            else:
                log_row["reward"] = self.config.format_reward
                log_row["reason"] = result.error_type or "compile_failed"
            return self._finish(log_row, start)
        except ValueError as error:
            log_row["reason"] = "invalid_proof_format"
            log_row["diagnostics"] = str(error)
            return self._finish(log_row, start)
        except Exception as error:
            log_row["reason"] = "reward_error"
            log_row["diagnostics"] = repr(error)
            return self._finish(log_row, start)

    @staticmethod
    def _row_kwargs(kwargs: dict[str, Any], index: int) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)) and index < len(value):
                row[key] = value[index]
            else:
                row[key] = value
        return row

    def _verifier(self) -> PantographTheoremVerifier:
        if self.verifier is None:
            self.verifier = PantographTheoremVerifier(
                self.config.lean_project_path,
                imports=self.imports,
                timeout=self.config.lean_timeout,
            )
            self.verifier.warmup(timeout=self.config.lean_timeout)
        return self.verifier

    def _finish(self, log_row: dict[str, Any], start: float) -> float:
        log_row["reward_seconds"] = round(time.monotonic() - start, 4)
        reward = float(log_row["reward"])
        self._write_log(log_row)
        return reward

    def _write_log(self, row: dict[str, Any]) -> None:
        if self.reward_log_path is None:
            return
        with self.reward_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        if self.verifier is not None:
            self.verifier.close()
            self.verifier = None


def extract_completion_text(completion: Any) -> str:
    """Normalize string, message, or message-list completions to plain text."""

    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "").strip()
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content") or last.get("text") or "").strip()
        return str(last).strip()
    return str(completion or "").strip()


def classify_proof_structure(proof: str) -> str:
    """Classify the outer shape of a generated Lean proof for reward auditing."""

    stripped = proof.strip()
    if not stripped:
        return "empty"
    if stripped.startswith(("theorem ", "lemma ")):
        return "full_declaration"
    if stripped.startswith("by"):
        return "tactic"
    if stripped.startswith(("fun ", "fun\n", "λ")):
        return "lambda"
    if stripped.startswith("calc"):
        return "calc"
    if stripped.startswith("match"):
        return "match"
    return "term"


def extract_compile_feedback(result: PantographCheckResult) -> dict[str, Any]:
    """Extract stable error fields immediately after Pantograph compilation."""

    errors = [str(item).strip() for item in result.errors if str(item).strip()]
    warnings = [str(item).strip() for item in result.warnings if str(item).strip()]
    primary_error = errors[0] if errors else ""
    if not primary_error and not result.success:
        primary_error = result.diagnostics.strip()
    return {
        "compile_error_type": result.error_type,
        "compile_error_count": len(errors),
        "compile_warning_count": len(warnings),
        "compile_primary_error": primary_error,
    }


def coerce_positive_int(value: Any) -> int | None:
    """Return a positive integer metadata value or ``None`` when unavailable."""

    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def coerce_string_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a scalar or iterable context value into non-empty strings."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(line.strip() for line in value.splitlines() if line.strip())
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()
