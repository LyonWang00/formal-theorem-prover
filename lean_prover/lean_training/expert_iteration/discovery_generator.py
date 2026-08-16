"""Discovery generation adapter over the existing benchmark generators."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lean_prover.lean_training.evaluation.benchmark import (
    TransformersGenerator,
    VllmGenerator,
    extract_proof_body,
)
from lean_prover.lean_training.data.preparation import normalize_proof_for_assembly

from .config import GenerationConfig
from .datasets import assert_prompt_has_no_reference_proof
from .schemas import (
    DataRole,
    DiscoveryStatementState,
    GenerationRecord,
    StatementRecord,
    stable_hash,
)
from .utils import append_jsonl, count_jsonl, iter_jsonl


@dataclass(frozen=True)
class GenerationRun:
    count: int
    output_path: str

    def __len__(self) -> int:
        return self.count


class GeneratorBackend(Protocol):
    backend_name: str

    def generate_batch(
        self,
        prompts: list[str],
        *,
        k: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> list[dict[str, Any]]: ...


def create_generator_backend(
    base_model: str,
    adapter_path: str | None,
    config: GenerationConfig,
) -> GeneratorBackend:
    """Create the existing Transformers/vLLM generator without reimplementing inference."""

    backend = config.backend
    if backend in {"auto", "vllm"}:
        try:
            return VllmGenerator(
                base_model,
                adapter_path=adapter_path,
                max_model_len=config.max_model_len,
                gpu_memory_utilization=config.gpu_memory_utilization,
                enforce_eager=config.enforce_eager,
                load_in_4bit=config.load_in_4bit,
                kv_cache_memory_bytes=config.kv_cache_memory_bytes,
            )
        except RuntimeError:
            if backend == "vllm":
                raise
    return TransformersGenerator(
        base_model,
        adapter_path=adapter_path,
        load_in_4bit=config.load_in_4bit,
        generation_contract_path=config.generation_contract_path,
        expected_generation_contract_sha256=config.generation_contract_sha256,
    )


def generate_candidates(
    statements: list[StatementRecord],
    states: dict[str, DiscoveryStatementState],
    *,
    iteration: int,
    checkpoint: str,
    generator: GeneratorBackend,
    config: GenerationConfig,
    sampling_budget: dict[str, int],
    zero_success_backoff_after_rounds: int,
    output_path: str | Path,
    seed: int,
    force: bool = False,
) -> GenerationRun:
    """Generate idempotent candidates, resuming from stable generation IDs."""

    path = Path(output_path)
    if force and path.exists():
        path.unlink()
    completed_ids = {
        str(row["generation_id"])
        for row in iter_jsonl(path)
        if row.get("generation_id")
    }
    grouped: dict[int, list[tuple[int, StatementRecord]]] = defaultdict(list)
    for statement_index, statement in enumerate(statements):
        state = states[statement.statement_id]
        key = "audit" if state.current_bucket.value == "solved_easy" else state.current_bucket.value
        count = sampling_budget.get(key, config.samples_per_statement)
        if state.consecutive_zero_success_rounds >= zero_success_backoff_after_rounds:
            count = max(1, count // 2)
        missing = [
            sample_index
            for sample_index in range(count)
            if _generation_id(
                statement.statement_id,
                iteration,
                checkpoint,
                sample_index,
                max_new_tokens=config.max_new_tokens,
            )
            not in completed_ids
        ]
        if missing:
            grouped[count].append((statement_index, statement))
    for sample_count, group in grouped.items():
        for offset in range(0, len(group), config.batch_size):
            indexed_batch = group[offset : offset + config.batch_size]
            batch = [statement for _, statement in indexed_batch]
            # Use positions in the frozen input manifest, not positions in the
            # remaining-work list.  With batch_size=1 this makes an interrupted
            # and resumed run seed-identical to an uninterrupted run.
            prompt_offset = indexed_batch[0][0]
            batch_seed = seed + iteration * 1_000_000 + prompt_offset
            prompts = [statement.generation_prompt() for statement in batch]
            for statement, prompt in zip(batch, prompts, strict=True):
                assert_prompt_has_no_reference_proof(statement, prompt)
            rows = generator.generate_batch(
                prompts,
                k=sample_count,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=batch_seed,
            )
            for row in rows:
                statement = batch[int(row["problem_batch_index"])]
                sample_index = int(row["local_attempt_index"])
                generation_id = _generation_id(
                    statement.statement_id,
                    iteration,
                    checkpoint,
                    sample_index,
                    max_new_tokens=config.max_new_tokens,
                )
                if generation_id in completed_ids:
                    continue
                raw = str(row.get("raw_completion") or "")
                extracted = extract_proof_body(raw) or None
                normalized = None
                proof_format = None
                if extracted:
                    normalized, resolved_format = normalize_proof_for_assembly(extracted)
                    proof_format = resolved_format.value
                raw_finish_reason = str(row.get("finish_reason") or "").lower()
                finish_reason = (
                    raw_finish_reason
                    if raw_finish_reason in {"stop", "eos", "length", "abort", "error"}
                    else None
                )
                record = GenerationRecord(
                    generation_id=generation_id,
                    statement_id=statement.statement_id,
                    data_role=DataRole.DISCOVERY,
                    iteration=iteration,
                    checkpoint=checkpoint,
                    sample_index=sample_index,
                    prompt=statement.generation_prompt(),
                    raw_output=raw,
                    extracted_proof=extracted,
                    normalized_proof=normalized,
                    proof_format=proof_format,
                    finish_reason=finish_reason,
                    generation_seed=batch_seed,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_new_tokens=config.max_new_tokens,
                    metadata={
                        "backend": generator.backend_name,
                        "completion_tokens": row.get("completion_tokens"),
                        "generation_seconds": row.get("generation_problem_seconds"),
                        "finish_reason": finish_reason or row.get("finish_reason"),
                        "stop_reason": row.get("stop_reason"),
                        "eos_token_id": getattr(
                            getattr(generator, "tokenizer", None),
                            "eos_token_id",
                            None,
                        ),
                        "pad_token_id": getattr(
                            getattr(generator, "tokenizer", None),
                            "pad_token_id",
                            None,
                        ),
                        "stop_token_ids": list(
                            getattr(generator, "stop_token_ids", ()) or ()
                        ),
                        "prompt_format": "plain_text_lean_sections_v1",
                        "base_model_path": getattr(
                            generator, "model_name_or_path", None
                        ),
                        "adapter_path": getattr(generator, "adapter_path", None),
                        "generation_contract_path": getattr(
                            generator, "generation_contract_path", None
                        ),
                        "generation_contract_sha256": getattr(
                            generator, "generation_contract_sha256", None
                        ),
                        "model_load_strategy": getattr(
                            generator, "model_load_strategy", None
                        ),
                    },
                )
                append_jsonl(path, (record.model_dump(mode="json"),))
                completed_ids.add(generation_id)
    return GenerationRun(count=count_jsonl(path), output_path=str(path))


def _generation_id(
    statement_id: str,
    iteration: int,
    checkpoint: str,
    sample_index: int,
    *,
    max_new_tokens: int,
) -> str:
    return (
        "gen_"
        f"{stable_hash(statement_id, iteration, checkpoint, sample_index, max_new_tokens)[:24]}"
    )
