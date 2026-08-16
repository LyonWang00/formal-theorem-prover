"""Asynchronous proof generation and Lean verification benchmark pipeline.

The pipeline has four layers:

1. generation: vLLM when available, otherwise Transformers as a local fallback;
2. queues: one priority queue per Lean worker, with all attempts for one
   problem routed to the same worker;
3. verification: N Pantograph workers, each owning a queue and verifier;
4. storage: streaming JSONL attempt rows plus one JSONL row per problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lean_prover.lean_training.prepare_datasets import (
    cleanup_proof_body,
    compose_lean_theorem,
    contains_forbidden_proof_token,
)
from lean_prover.lean_training.verification_pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification_schema import (
    VerificationTask,
    make_verification_result,
)


@dataclass(frozen=True)
class BenchmarkRecord:
    problem_id: str
    prompt: str
    lean_statement: str
    imports: tuple[str, ...]
    context_lines: tuple[str, ...] = ()


@dataclass
class ProblemState:
    problem_id: str
    prompt: str
    expected_attempts: int
    results: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    canceled: bool = False


class JsonlStore:
    def __init__(self, output_dir: Path, *, resume: bool = False) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_path = output_dir / "attempts.jsonl"
        self.problems_path = output_dir / "problem_results.jsonl"
        self.success_attempts_path = output_dir / "success_attempts.jsonl"
        self.summary_path = output_dir / "summary.json"
        self.shards_dir = output_dir / "attempt_shards"
        if not resume:
            if self.shards_dir.exists():
                shutil.rmtree(self.shards_dir)
            archive_path = output_dir / "attempt_shards.zip"
            if archive_path.exists():
                archive_path.unlink()
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        self._attempts = self.attempts_path.open(mode, encoding="utf-8")
        self._problems = self.problems_path.open(mode, encoding="utf-8")
        self._success_attempts = self.success_attempts_path.open(
            mode, encoding="utf-8"
        )
        self._lock = threading.Lock()

    def write_attempt(self, result: Any) -> None:
        self.write_attempt_json(result.to_json())

    def write_attempt_json(self, result: Mapping[str, Any]) -> None:
        with self._lock:
            self._attempts.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            self._attempts.flush()
            shard_path = self.shard_path_for_result(result)
            with shard_path.open("a", encoding="utf-8") as shard:
                shard.write(json.dumps(result, ensure_ascii=False) + "\n")

    def write_problem(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            self._problems.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._problems.flush()

    def write_success_attempt(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            self._success_attempts.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._success_attempts.flush()

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def package_attempt_shards(self) -> Path:
        archive_path = self.output_dir / "attempt_shards.zip"
        with zipfile.ZipFile(
            archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(self.shards_dir.glob("*.jsonl")):
                archive.write(path, arcname=f"attempt_shards/{path.name}")
        return archive_path

    def shard_path_for_result(self, result: Mapping[str, Any]) -> Path:
        problem_index = int(result.get("problem_index", -1))
        problem_id = sanitize_filename(str(result.get("problem_id", "unknown")))
        return self.shards_dir / f"problem_{problem_index:05d}_{problem_id}.jsonl"

    def close(self) -> None:
        self._attempts.close()
        self._problems.close()
        self._success_attempts.close()


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:120] or "unknown"


def load_completed_problem_ids(output_dir: Path, expected_attempts: int) -> set[str]:
    return set(load_completed_problem_rows(output_dir, expected_attempts))


def load_completed_problem_rows(
    output_dir: Path, expected_attempts: int
) -> dict[str, dict[str, Any]]:
    path = output_dir / "problem_results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row.get("problem_id", ""))
            if row.get("success") or int(row.get("num_attempts_recorded", 0)) >= expected_attempts:
                completed[problem_id] = row
    return completed


def load_attempt_rows(output_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    path = output_dir / "attempts.jsonl"
    attempts: dict[str, dict[int, dict[str, Any]]] = {}
    if not path.exists():
        return attempts
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"WARNING: ignoring corrupt attempts.jsonl line {line_number}",
                    flush=True,
                )
                continue
            problem_id = str(row.get("problem_id", ""))
            attempt_index = int(row.get("attempt_index", -1))
            if problem_id and attempt_index >= 0:
                attempts.setdefault(problem_id, {})[attempt_index] = row
    return attempts


def file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def build_run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    project_path = Path(args.lean_project_path)
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "adapter_hash": file_sha256(Path(args.adapter_path) / "adapter_config.json")
        if args.adapter_path
        else None,
        "benchmark_file": args.benchmark_file,
        "benchmark_file_hash": file_sha256(args.benchmark_file),
        "lean_project_path": str(project_path),
        "lean_project_git_commit": git_commit(project_path),
        "pass_k": args.pass_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "generation_seed": args.generation_seed,
        "generation_backend": args.generation_backend,
        "generation_batch_size": args.generation_batch_size,
        "num_workers": args.num_workers,
        "imports": list(args.imports),
        "lean_timeout": args.lean_timeout,
        "code_git_commit": git_commit(repo_root),
        "early_stop_on_success": True,
    }


def validate_or_write_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = build_run_manifest(args)
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"previous": previous.get(key), "current": manifest.get(key)}
            for key in manifest
            if previous.get(key) != manifest.get(key)
        }
        if mismatches and not args.force_resume:
            raise ValueError(
                "refusing resume because run_manifest.json differs: "
                + json.dumps(mismatches, ensure_ascii=False)
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class TransformersGenerator:
    """Generate proof completions with a local Transformers model and optional LoRA adapter."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        adapter_path: str | None,
        load_in_4bit: bool,
    ) -> None:
        """Load the generation backend.

        Args:
            model_name_or_path: Hugging Face model ID or local model directory.
            adapter_path: Optional PEFT LoRA adapter directory.
            load_in_4bit: Whether to load the base model with 4-bit quantization.

        Output:
            Initializes the tokenizer and reusable causal language model.
        """
        self.backend_name = "transformers"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model
        self.model.eval()

    def generate(
        self,
        prompt: str,
        *,
        k: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> tuple[list[str], float]:
        """Generate multiple completions for one prompt.

        Args:
            prompt: Formatted Lean proof-generation prompt.
            k: Number of completions to return.
            max_new_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature; non-positive means deterministic decoding.
            top_p: Nucleus-sampling probability threshold.
            seed: Optional random seed for this call.

        Returns:
            The decoded completions and elapsed generation seconds.
        """
        start = time.monotonic()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": k,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        completions = [
            self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            for row in generated
        ]
        return completions, round(time.monotonic() - start, 4)

    def generate_batch(
        self,
        prompts: list[str],
        *,
        k: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate completions for several prompts through a compatibility loop.

        Args:
            prompts: Prompts to process in input order.
            k: Number of completions requested per prompt.
            max_new_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus-sampling probability threshold.
            seed: Optional base seed, offset once per prompt.

        Returns:
            Flat result rows containing indices, completion text, finish metadata,
            token counts, and generation timing.
        """

        batch_start = time.monotonic()
        rows: list[dict[str, Any]] = []
        for problem_batch_index, prompt in enumerate(prompts):
            completions, problem_seconds = self.generate(
                prompt,
                k=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=None if seed is None else seed + problem_batch_index,
            )
            for local_attempt_index, completion in enumerate(completions):
                completion_tokens = len(
                    self.tokenizer(completion, add_special_tokens=False)["input_ids"]
                )
                rows.append(
                    {
                        "problem_batch_index": problem_batch_index,
                        "local_attempt_index": local_attempt_index,
                        "raw_completion": completion,
                        "generation_problem_seconds": problem_seconds,
                        "completion_tokens": completion_tokens,
                        "finish_reason": None,
                        "hit_eos": self.tokenizer.eos_token in completion
                        if self.tokenizer.eos_token
                        else False,
                        "hit_max_new_tokens": completion_tokens >= max_new_tokens,
                    }
                )
        batch_seconds = round(time.monotonic() - batch_start, 4)
        for row in rows:
            row["generation_batch_seconds"] = batch_seconds
        return rows


class VllmGenerator:
    """Generate proof completions with vLLM and an optional LoRA adapter."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        adapter_path: str | None,
        max_model_len: int | None,
        gpu_memory_utilization: float,
    ) -> None:
        """Create the reusable vLLM engine.

        Args:
            model_name_or_path: Model ID or local base-model directory.
            adapter_path: Optional LoRA adapter directory.
            max_model_len: Optional maximum sequence length.
            gpu_memory_utilization: Fraction of GPU memory reserved by vLLM.

        Output:
            Initializes the vLLM engine and optional LoRA request.

        Raises:
            RuntimeError: If vLLM is unavailable or LoRA setup fails.
        """
        try:
            from vllm import LLM
        except ImportError as error:
            raise RuntimeError("vLLM is not installed") from error
        self.backend_name = "vllm"
        self.adapter_path = adapter_path
        self._lora_request = None
        kwargs: dict[str, Any] = {
            "model": model_name_or_path,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if adapter_path:
            try:
                from vllm.lora.request import LoRARequest

                kwargs["enable_lora"] = True
                self._lora_request = LoRARequest("adapter", 1, adapter_path)
            except Exception as error:
                raise RuntimeError(f"failed to configure vLLM LoRA: {error}") from error
        self.llm = LLM(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        k: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> tuple[list[str], float]:
        """Generate ``k`` completions for one prompt.

        Args:
            prompt: Formatted Lean proof-generation prompt.
            k: Number of completions to sample.
            max_new_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature; non-positive means greedy decoding.
            top_p: Nucleus-sampling probability threshold.
            seed: Optional sampling seed.

        Returns:
            The generated completion strings and elapsed seconds.
        """
        from vllm import SamplingParams

        start = time.monotonic()
        params_kwargs = {"n": k, "max_tokens": max_new_tokens}
        if temperature <= 0:
            params_kwargs["temperature"] = 0
        else:
            params_kwargs.update({"temperature": temperature, "top_p": top_p})
        if seed is not None:
            params_kwargs["seed"] = seed
        params = SamplingParams(**params_kwargs)
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = self.llm.generate([prompt], params, **kwargs)
        completions = [item.text for item in outputs[0].outputs]
        return completions, round(time.monotonic() - start, 4)

    def generate_batch(
        self,
        prompts: list[str],
        *,
        k: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate completions for a prompt batch in one vLLM request.

        Args:
            prompts: Prompts submitted together.
            k: Number of completions requested per prompt.
            max_new_tokens: Maximum new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus-sampling probability threshold.
            seed: Optional batch sampling seed.

        Returns:
            Flat result rows with indices, generated text, finish metadata,
            token counts, and batch/per-problem timing.
        """
        from vllm import SamplingParams

        start = time.monotonic()
        params_kwargs = {"n": k, "max_tokens": max_new_tokens}
        if temperature <= 0:
            params_kwargs["temperature"] = 0
        else:
            params_kwargs.update({"temperature": temperature, "top_p": top_p})
        if seed is not None:
            params_kwargs["seed"] = seed
        params = SamplingParams(**params_kwargs)
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = self.llm.generate(prompts, params, **kwargs)
        batch_seconds = round(time.monotonic() - start, 4)
        rows: list[dict[str, Any]] = []
        for problem_batch_index, request_output in enumerate(outputs):
            for local_attempt_index, output in enumerate(request_output.outputs):
                token_ids = getattr(output, "token_ids", None) or ()
                finish_reason = getattr(output, "finish_reason", None)
                rows.append(
                    {
                        "problem_batch_index": problem_batch_index,
                        "local_attempt_index": local_attempt_index,
                        "raw_completion": output.text,
                        "generation_batch_seconds": batch_seconds,
                        "generation_problem_seconds": None,
                        "completion_tokens": len(token_ids),
                        "finish_reason": finish_reason,
                        "hit_eos": str(finish_reason).lower() in {"stop", "eos"},
                        "hit_max_new_tokens": (
                            str(finish_reason).lower() == "length"
                            or len(token_ids) >= max_new_tokens
                        ),
                    }
                )
        per_problem_seconds = round(batch_seconds / max(1, len(prompts)), 4)
        for row in rows:
            row["generation_problem_seconds"] = per_problem_seconds
        return rows


def read_benchmark_records(path: str, limit: int | None) -> list[BenchmarkRecord]:
    """Load and validate proof-free benchmark records from JSONL.

    Args:
        path: Input JSONL file path.
        limit: Optional maximum number of records to load.

    Returns:
        Normalized records containing prompts, Lean statements, imports, and context.

    Raises:
        ValueError: If a row includes a reference proof or duplicates a problem ID.
    """
    records: list[BenchmarkRecord] = []
    seen_problem_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "proof" in row and str(row["proof"]).strip():
                raise ValueError(f"benchmark record {row.get('id')} contains proof")
            prompt = row.get("prompt")
            if not prompt:
                prompt = (
                    "### Informal statement\n"
                    f"{row.get('informal_statement', '').strip()}\n\n"
                    "### Lean statement\n"
                    f"{row['lean_statement'].strip()}\n\n"
                    "### Lean proof\n"
                )
            problem_id = str(row.get("id", f"problem-{len(records)}"))
            if problem_id in seen_problem_ids:
                raise ValueError(f"duplicate benchmark problem_id: {problem_id}")
            seen_problem_ids.add(problem_id)
            records.append(
                BenchmarkRecord(
                    problem_id=problem_id,
                    prompt=prompt,
                    lean_statement=str(row["lean_statement"]),
                    imports=tuple(
                        row.get("imports")
                        or (row.get("preamble") or {}).get("imports")
                        or ("Mathlib",)
                    ),
                    context_lines=tuple(
                        row.get("context_lines")
                        or (row.get("preamble") or {}).get("context_lines")
                        or ()
                    ),
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def extract_proof_body(completion: str) -> str:
    """Normalize a raw model completion into the Lean proof body to verify.

    Args:
        completion: Raw text returned by the generation backend.

    Returns:
        Cleaned proof text without trailing prompt sections or Markdown fences.
    """
    text = completion.strip()
    for marker in ("\n\n###", "\n### Informal", "\n### Lean statement"):
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos]
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    if "```" in text:
        text = text.split("```", 1)[0]
    return cleanup_proof_body(text)


def create_generator(args: argparse.Namespace):
    """Create the selected generator, falling back from vLLM in auto mode.

    Args:
        args: Parsed model, adapter, quantization, and vLLM settings.

    Returns:
        A configured ``VllmGenerator`` or ``TransformersGenerator``.

    Raises:
        RuntimeError: If vLLM was explicitly requested but cannot be initialized.
    """
    if args.generation_backend in {"vllm", "auto"}:
        try:
            return VllmGenerator(
                args.model_name_or_path,
                adapter_path=args.adapter_path,
                max_model_len=args.vllm_max_model_len,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            )
        except Exception as error:
            if args.generation_backend == "vllm":
                raise
            print(f"vLLM unavailable; falling back to Transformers: {error}")
    return TransformersGenerator(
        args.model_name_or_path,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
    )


def problem_summary_row(state: ProblemState) -> dict[str, Any]:
    """Build the persisted problem-level summary from accumulated attempts.

    Args:
        state: Current results and completion state for one problem.

    Returns:
        A JSON-serializable problem result row.
    """
    return {
        "problem_id": state.problem_id,
        "prompt": state.prompt,
        "success": state.success,
        "num_attempts_recorded": len(state.results),
        "attempts": sorted(state.results, key=lambda item: item["attempt_index"]),
    }


def problem_is_complete(state: ProblemState) -> bool:
    return len(state.results) >= state.expected_attempts


def pass_at_counts(states: Mapping[str, ProblemState], max_k: int) -> dict[str, float]:
    """Calculate empirical pass@1 through pass@``max_k``.

    Args:
        states: Problem states containing ordered attempt results.
        max_k: Largest attempt cutoff to calculate.

    Returns:
        A mapping from pass@k labels to solved-problem fractions.
    """
    values: dict[str, float] = {}
    total = len(states)
    for k in range(1, max_k + 1):
        successes = 0
        for state in states.values():
            ordered = sorted(state.results, key=lambda item: item["attempt_index"])
            if any(item.get("success") for item in ordered[:k]):
                successes += 1
        values[f"pass@{k}"] = successes / total if total else 0.0
    return values


def iter_generation_batches(
    records: list[BenchmarkRecord],
    *,
    completed_problem_ids: set[str],
    existing_attempt_rows: Mapping[str, Mapping[int, Mapping[str, Any]]],
    pass_k: int,
    batch_size: int,
) -> Iterable[list[tuple[int, BenchmarkRecord, list[int]]]]:
    """Yield batches containing only problems and attempts still needing generation.

    Args:
        records: Benchmark records in stable input order.
        completed_problem_ids: Problems already complete during resume.
        existing_attempt_rows: Existing attempts grouped by problem and attempt index.
        pass_k: Target attempt count per problem.
        batch_size: Maximum number of problems per generation batch.

    Yields:
        Lists of ``(problem_index, record, missing_attempt_indices)`` tuples.
    """
    batch: list[tuple[int, BenchmarkRecord, list[int]]] = []
    for problem_index, record in enumerate(records):
        if record.problem_id in completed_problem_ids:
            print(
                f"RESUME_SKIP problem_id={record.problem_id} problem_index={problem_index}",
                flush=True,
            )
            continue
        existing_attempts = existing_attempt_rows.get(record.problem_id, {})
        missing_attempt_indices = [
            attempt_index
            for attempt_index in range(pass_k)
            if attempt_index not in existing_attempts
        ]
        if not missing_attempt_indices:
            continue
        batch.append((problem_index, record, missing_attempt_indices))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def split_prompt_sections(prompt: str) -> dict[str, str]:
    sections = {
        "informal_statement": "",
        "lean_statement": "",
    }
    informal_marker = "### Informal statement"
    lean_marker = "### Lean statement"
    proof_marker = "### Lean proof"
    informal_pos = prompt.find(informal_marker)
    lean_pos = prompt.find(lean_marker)
    proof_pos = prompt.find(proof_marker)
    if informal_pos != -1 and lean_pos != -1:
        sections["informal_statement"] = prompt[
            informal_pos + len(informal_marker) : lean_pos
        ].strip()
    if lean_pos != -1:
        end = proof_pos if proof_pos != -1 else len(prompt)
        sections["lean_statement"] = prompt[lean_pos + len(lean_marker) : end].strip()
    return sections


def success_attempt_row(result: Mapping[str, Any]) -> dict[str, Any]:
    sections = split_prompt_sections(str(result.get("prompt", "")))
    return {
        "problem_id": result.get("problem_id"),
        "attempt_id": result.get("attempt_id"),
        "problem_index": result.get("problem_index"),
        "attempt_index": result.get("attempt_index"),
        "informal_statement": sections["informal_statement"],
        "lean_statement": sections["lean_statement"],
        "generated_proof": result.get("generated_proof"),
        "lean_code": result.get("lean_code"),
        "imports": result.get("imports"),
        "status": result.get("status"),
        "success": result.get("success"),
        "diagnostics": result.get("diagnostics"),
        "compile_messages": result.get("compile_messages"),
        "compile_errors": result.get("compile_errors"),
        "compile_warnings": result.get("compile_warnings"),
        "has_compile_errors": result.get("has_compile_errors"),
        "has_compile_warnings": result.get("has_compile_warnings"),
        "generation_seconds": result.get("generation_seconds"),
        "queue_wait_seconds": result.get("queue_wait_seconds"),
        "verification_seconds": result.get("verification_seconds"),
        "total_seconds": result.get("total_seconds"),
        "worker_id": result.get("worker_id"),
        "verifier_backend": result.get("verifier_backend"),
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run generation, Pantograph verification, persistence, and aggregation.

    Args:
        args: Parsed input/output paths, generation settings, resume options, and
            Pantograph worker configuration.

    Returns:
        Final summary with pass@k, attempt/error counts, timing, backend details,
        and the attempt-shard archive path.

    Side Effects:
        Writes benchmark artifacts under ``args.output_dir`` and starts model and
        Pantograph worker processes.
    """
    start = time.monotonic()
    records = read_benchmark_records(args.benchmark_file, args.num_benchmark_samples)
    output_dir = Path(args.output_dir)
    validate_or_write_manifest(args, output_dir)
    completed_problem_rows = (
        load_completed_problem_rows(output_dir, args.pass_k) if args.resume else {}
    )
    existing_attempt_rows = load_attempt_rows(output_dir) if args.resume else {}
    completed_problem_ids = set(completed_problem_rows)
    store = JsonlStore(output_dir, resume=args.resume)
    states = {
        record.problem_id: ProblemState(
            problem_id=record.problem_id,
            prompt=record.prompt,
            expected_attempts=args.pass_k,
        )
        for record in records
    }
    for problem_id, row in completed_problem_rows.items():
        if problem_id in states:
            states[problem_id].results = list(row.get("attempts") or [])
            states[problem_id].success = bool(row.get("success"))
            states[problem_id].canceled = bool(row.get("success"))
    for problem_id, attempts in existing_attempt_rows.items():
        if problem_id in states and problem_id not in completed_problem_ids:
            states[problem_id].results = list(attempts.values())
            if any(row.get("success") for row in attempts.values()):
                states[problem_id].success = True
                states[problem_id].canceled = True
                completed_problem_ids.add(problem_id)
    warmup_reports: list[dict[str, Any]] = []
    fatal_errors: list[str] = []
    recovered_worker_failures: list[str] = []
    result_counter = 0
    written_problem_ids: set[str] = set(completed_problem_ids)
    active_problem_ids = {
        record.problem_id
        for record in records
        if record.problem_id not in completed_problem_ids
    }
    if args.resume and not active_problem_ids:
        all_results = [result for state in states.values() for result in state.results]
        attempts_with_compile_errors = sum(
            1 for result in all_results if result.get("has_compile_errors")
        )
        attempts_with_compile_warnings = sum(
            1 for result in all_results if result.get("has_compile_warnings")
        )
        successful_attempts_with_warnings = sum(
            1
            for result in all_results
            if result.get("success") and result.get("has_compile_warnings")
        )
        archive_path = store.package_attempt_shards()
        successes = sum(1 for state in states.values() if state.success)
        pass_at = pass_at_counts(states, args.pass_k)
        summary = {
            "model_name_or_path": args.model_name_or_path,
            "adapter_path": args.adapter_path,
            "generation_backend": args.generation_backend,
            "verifier_backend": "pantograph",
            "num_workers": args.num_workers,
            "num_benchmark_samples": len(records),
            "pass_k": args.pass_k,
            "successes": successes,
            f"pass@{args.pass_k}": successes / len(records) if records else 0.0,
            "pass_at": pass_at,
            "early_stop_on_success": True,
            "queued_attempts": 0,
            "rejected_attempts": 0,
            "recorded_attempt_results": len(all_results),
            "attempts_with_compile_errors": attempts_with_compile_errors,
            "attempts_with_compile_warnings": attempts_with_compile_warnings,
            "successful_attempts_with_warnings": successful_attempts_with_warnings,
            "warmup_seconds": 0.0,
            "warmup_reports": [],
            "fatal_errors": [],
            "recovered_worker_failures": [],
            "generation_seconds": 0.0,
            "total_seconds": round(time.monotonic() - start, 4),
            "attempt_shards_archive": str(archive_path),
            "resume_skipped_problem_ids": sorted(completed_problem_ids),
        }
        for problem_id in sorted(completed_problem_ids):
            print(f"RESUME_SKIP problem_id={problem_id}", flush=True)
        store.write_summary(summary)
        store.close()
        return summary
    def record_result(result: dict[str, Any]) -> None:
        nonlocal result_counter
        problem_id = str(result["problem_id"])
        state = states[problem_id]
        state.results.append(result)
        result_counter += 1
        if bool(result.get("success")):
            state.success = True
            state.canceled = True
        store.write_attempt_json(result)
        if bool(result.get("success")):
            store.write_success_attempt(success_attempt_row(result))
        if problem_id not in written_problem_ids and problem_is_complete(state):
            row = problem_summary_row(state)
            store.write_problem(row)
            print("PROBLEM_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
            written_problem_ids.add(problem_id)

    warmup_start = time.monotonic()
    verification_pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=args.lean_project_path,
            imports=args.imports,
            timeout=args.lean_timeout,
            warmup_timeout=args.warmup_timeout,
            num_workers=args.num_workers,
            queue_maxsize=args.queue_maxsize,
            disable_warmup=args.disable_warmup,
            cancel_on_success=True,
        )
    )
    verification_pool.start()
    warmup_reports.extend(verification_pool.warmup_reports)
    fatal_errors.extend(verification_pool.fatal_errors)
    warmup_seconds = round(time.monotonic() - warmup_start, 4)
    if fatal_errors:
        verification_pool.close()
        summary = {
            "model_name_or_path": args.model_name_or_path,
            "adapter_path": args.adapter_path,
            "generation_backend": args.generation_backend,
            "verifier_backend": "pantograph",
            "num_workers": args.num_workers,
            "num_benchmark_samples": len(records),
            "pass_k": args.pass_k,
            "successes": 0,
            f"pass@{args.pass_k}": 0.0,
            "queued_attempts": 0,
            "rejected_attempts": 0,
            "recorded_attempt_results": result_counter,
            "attempts_with_compile_errors": 0,
            "attempts_with_compile_warnings": 0,
            "successful_attempts_with_warnings": 0,
            "warmup_seconds": warmup_seconds,
            "warmup_reports": sorted(
                warmup_reports, key=lambda item: item["worker_id"]
            ),
            "fatal_errors": list(fatal_errors),
            "recovered_worker_failures": list(recovered_worker_failures),
            "generation_seconds": 0.0,
            "total_seconds": round(time.monotonic() - start, 4),
        }
        summary["attempt_shards_archive"] = str(store.package_attempt_shards())
        store.write_summary(summary)
        store.close()
        return summary

    generator = create_generator(args)
    generation_start = time.monotonic()
    rejected_count = 0
    queued_count = 0
    for generation_batch_index, generation_batch in enumerate(
        iter_generation_batches(
            records,
            completed_problem_ids=completed_problem_ids,
            existing_attempt_rows=existing_attempt_rows,
            pass_k=args.pass_k,
            batch_size=args.generation_batch_size,
        )
    ):
        prompts = [record.prompt for _, record, _ in generation_batch]
        max_missing = max(len(missing) for _, _, missing in generation_batch)
        generated_rows = generator.generate_batch(
            prompts,
            k=max_missing,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=(
                args.generation_seed + generation_batch_index
                if args.generation_seed is not None
                else None
            ),
        )
        batch_lookup = {
            problem_batch_index: (problem_index, record, missing_attempt_indices)
            for problem_batch_index, (
                problem_index,
                record,
                missing_attempt_indices,
            ) in enumerate(generation_batch)
        }
        for generated in generated_rows:
            problem_batch_index = int(generated["problem_batch_index"])
            local_attempt_index = int(generated["local_attempt_index"])
            problem_index, record, missing_attempt_indices = batch_lookup[
                problem_batch_index
            ]
            if local_attempt_index >= len(missing_attempt_indices):
                continue
            attempt_index = missing_attempt_indices[local_attempt_index]
            raw_completion = str(generated["raw_completion"])
            proof = extract_proof_body(raw_completion)
            rejected_reason = None
            try:
                lean_code = compose_lean_theorem(record.lean_statement, proof)
            except ValueError as error:
                lean_code = ""
                rejected_reason = str(error)
            task = VerificationTask(
                priority=problem_index,
                problem_index=problem_index,
                attempt_index=attempt_index,
                problem_id=record.problem_id,
                prompt=record.prompt,
                generated_proof=proof,
                raw_completion=raw_completion,
                lean_code=lean_code,
                imports=record.imports or args.imports,
                context_lines=record.context_lines,
                enqueue_time=time.monotonic(),
                generation_seconds=generated.get("generation_problem_seconds"),
                payload={
                    "generation_batch_seconds": generated.get(
                        "generation_batch_seconds"
                    ),
                    "generation_problem_seconds": generated.get(
                        "generation_problem_seconds"
                    ),
                    "completion_tokens": generated.get("completion_tokens"),
                    "finish_reason": generated.get("finish_reason"),
                    "hit_eos": generated.get("hit_eos"),
                    "hit_max_new_tokens": generated.get("hit_max_new_tokens"),
                    "generation_batch_index": generation_batch_index,
                },
            )
            if rejected_reason or contains_forbidden_proof_token(proof):
                reason = rejected_reason or "generated proof contains sorry/admit"
                rejected_count += 1
                result = make_verification_result(
                    task,
                    worker_id=None,
                    status="rejected",
                    success=False,
                    diagnostics=reason,
                    verifier_backend="precheck",
                    verification_seconds=0.0,
                    rejected_reason=reason,
                    compile_messages=(reason,),
                    compile_errors=(reason,),
                )
                record_result(result.to_json())
                print(
                    f"rejected problem_id={record.problem_id} attempt_id="
                    f"({problem_index},{attempt_index}) reason={reason}"
                )
                continue
            verification_pool.submit(task)
            queued_count += 1
        for result in verification_pool.drain():
            record_result(result)
    generation_seconds = round(time.monotonic() - generation_start, 4)

    def all_active_problems_complete() -> bool:
        return all(problem_is_complete(states[problem_id]) for problem_id in active_problem_ids)

    while active_problem_ids and not all_active_problems_complete():
        for result in verification_pool.drain(block=True, timeout=1.0):
            record_result(result)

    verification_pool.finish()
    for result in verification_pool.drain_until_done():
        record_result(result)
    recovered_worker_failures.extend(verification_pool.recovered_worker_failures)
    fatal_errors.extend(verification_pool.fatal_errors)
    verification_pool.close()

    # Flush problem rows for all-rejected problems or problems not written due to
    # exact expected-count races.
    if store.problems_path.exists():
        with store.problems_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    written_problem_ids.add(json.loads(line)["problem_id"])
    for state in states.values():
        if state.problem_id not in written_problem_ids:
            store.write_problem(problem_summary_row(state))
            written_problem_ids.add(state.problem_id)

    successes = sum(1 for state in states.values() if state.success)
    all_results = [result for state in states.values() for result in state.results]
    attempts_with_compile_errors = sum(
        1 for result in all_results if result.get("has_compile_errors")
    )
    attempts_with_compile_warnings = sum(
        1 for result in all_results if result.get("has_compile_warnings")
    )
    successful_attempts_with_warnings = sum(
        1
        for result in all_results
        if result.get("success") and result.get("has_compile_warnings")
    )
    canceled_attempts = sum(1 for result in all_results if result.get("status") == "canceled")
    timeout_attempts = sum(1 for result in all_results if result.get("timed_out"))
    backend_error_attempts = sum(
        1
        for result in all_results
        if "pantograph" in str(result.get("verifier_backend", ""))
        and result.get("status") == "failed"
        and result.get("timed_out") is not True
        and not result.get("has_compile_errors")
    )
    max_token_truncated_attempts = sum(
        1 for result in all_results if result.get("hit_max_new_tokens")
    )
    pass_at = pass_at_counts(states, args.pass_k)
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "generation_backend": getattr(generator, "backend_name", args.generation_backend),
        "verifier_backend": "pantograph",
        "num_workers": args.num_workers,
        "generation_batch_size": args.generation_batch_size,
        "num_benchmark_samples": len(records),
        "pass_k": args.pass_k,
        "successes": successes,
        f"pass@{args.pass_k}": successes / len(records) if records else 0.0,
        "pass_at": pass_at,
        "early_stop_on_success": True,
        "queued_attempts": queued_count,
        "rejected_attempts": rejected_count,
        "verified_attempts": sum(
            1 for result in all_results if result.get("verifier_backend") == "pantograph"
        ),
        "canceled_attempts": canceled_attempts,
        "timeout_attempts": timeout_attempts,
        "backend_error_attempts": backend_error_attempts,
        "max_token_truncated_attempts": max_token_truncated_attempts,
        "recorded_attempt_results": result_counter,
        "attempts_with_compile_errors": attempts_with_compile_errors,
        "attempts_with_compile_warnings": attempts_with_compile_warnings,
        "successful_attempts_with_warnings": successful_attempts_with_warnings,
        "warmup_seconds": warmup_seconds,
        "warmup_reports": sorted(warmup_reports, key=lambda item: item["worker_id"]),
        "fatal_errors": list(fatal_errors),
        "recovered_worker_failures": list(recovered_worker_failures),
        "generation_seconds": generation_seconds,
        "total_seconds": round(time.monotonic() - start, 4),
    }
    summary["attempt_shards_archive"] = str(store.package_attempt_shards())
    store.write_summary(summary)
    store.close()
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI options and normalize the comma-separated imports value.

    Returns:
        Benchmark arguments with ``imports`` represented as a tuple of module names.
    """
    default_project = Path(__file__).resolve().parents[2] / "lean_project"
    parser = argparse.ArgumentParser(description="Async Lean SFT benchmark pipeline.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--benchmark_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_benchmark_samples", type=int, default=None)
    parser.add_argument("--pass_k", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--queue_maxsize", type=int, default=128)
    parser.add_argument(
        "--generation_backend",
        choices=("auto", "vllm", "transformers"),
        default="auto",
    )
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generation_seed", type=int, default=20260711)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--force_resume", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lean_timeout", type=int, default=180)
    parser.add_argument(
        "--warmup_timeout",
        type=int,
        default=900,
        help="Seconds allowed for per-worker dependency warmup before benchmark.",
    )
    parser.add_argument(
        "--disable_warmup",
        action="store_true",
        help="Skip per-worker dependency warmup.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to existing output files and skip problem-level results "
            "already marked complete in problem_results.jsonl."
        ),
    )
    parser.add_argument(
        "--lean_project_path",
        default=os.environ.get("LEAN_PROJECT_PATH", str(default_project)),
    )
    parser.add_argument("--imports", default="Mathlib")
    args = parser.parse_args()
    args.imports = tuple(item.strip() for item in args.imports.split(",") if item.strip())
    return args


def main() -> None:
    """Execute the CLI benchmark and print the final JSON summary."""
    summary = run_pipeline(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


