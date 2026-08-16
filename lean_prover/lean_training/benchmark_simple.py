"""Benchmark base and LoRA-adapted models on Lean proof generation.

The input benchmark file should be produced by ``prepare_datasets.py`` and must not
contain proof bodies. For each problem, the script samples k proof attempts,
compiles each completed theorem in the configured Lean project, and reports
pass@k: an example succeeds if any of its first k attempts compiles.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lean_prover.lean_training.prepare_datasets import (
    build_lean_source_with_preamble,
    cleanup_proof_body,
    compose_lean_theorem,
    contains_forbidden_proof_token,
)
from lean_prover.lean_training.pantograph_verifier import PantographTheoremVerifier


@dataclass(frozen=True)
class BenchmarkConfig:
    base_model_name_or_path: str
    benchmark_file: str
    output_dir: str
    adapter_path: str | None
    pass_k: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    load_in_4bit: bool
    lean_project_path: str
    lean_timeout: int
    imports: tuple[str, ...]


def parse_args() -> BenchmarkConfig:
    default_project = Path(__file__).resolve().parents[2] / "lean_project"
    parser = argparse.ArgumentParser(
        description="Generate Lean proofs and compute pass@k by compilation."
    )
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--benchmark_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--pass_k",
        default="1",
        help="Comma-separated k values, e.g. 1,4,8.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--lean_project_path",
        default=os.environ.get("LEAN_PROJECT_PATH", str(default_project)),
    )
    parser.add_argument("--lean_timeout", type=int, default=120)
    parser.add_argument(
        "--imports",
        default="Mathlib",
        help="Comma-separated default imports used when a benchmark record has none.",
    )
    args = parser.parse_args()
    pass_k = tuple(sorted({int(item) for item in args.pass_k.split(",") if item}))
    if not pass_k or min(pass_k) <= 0:
        raise ValueError("--pass_k must contain positive integers")
    imports = tuple(item.strip() for item in args.imports.split(",") if item.strip())
    return BenchmarkConfig(
        base_model_name_or_path=args.base_model_name_or_path,
        adapter_path=args.adapter_path,
        benchmark_file=args.benchmark_file,
        output_dir=args.output_dir,
        pass_k=pass_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=args.load_in_4bit,
        lean_project_path=args.lean_project_path,
        lean_timeout=args.lean_timeout,
        imports=imports,
    )


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "lean_statement" not in record:
                raise ValueError(f"line {line_number} lacks lean_statement")
            if "proof" in record and str(record["proof"]).strip():
                raise ValueError(
                    f"line {line_number} contains a proof body; "
                    "prepare benchmark data without proofs"
                )
            records.append(record)
    return records


def build_generation_prompt(record: Mapping[str, Any]) -> str:
    prompt = record.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    informal = str(record.get("informal_statement", "")).strip()
    statement = str(record["lean_statement"]).strip()
    return (
        "### Informal statement\n"
        f"{informal}\n\n"
        "### Lean statement\n"
        f"{statement}\n\n"
        "### Lean proof\n"
    )


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name_or_path: str, load_in_4bit: bool):
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
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def load_adapter_model(
    model_name_or_path: str,
    adapter_path: str,
    load_in_4bit: bool,
):
    from peft import PeftModel

    base_model = load_model(model_name_or_path, load_in_4bit)
    return PeftModel.from_pretrained(base_model, adapter_path)


def generate_proof_attempts(
    model,
    tokenizer,
    records: Iterable[Mapping[str, Any]],
    *,
    attempts_per_problem: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    model.eval()
    attempts: list[dict[str, Any]] = []
    for problem_index, record in enumerate(records, start=1):
        prompt = build_generation_prompt(record)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        for sample_index in range(attempts_per_problem):
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            raw_completion = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )
            proof = extract_proof_body(raw_completion)
            rejected_reason = None
            output_proof = proof
            if contains_forbidden_proof_token(proof):
                rejected_reason = "generated proof contains sorry/admit"
                output_proof = ""
            try:
                full_lean_code = compose_lean_theorem(
                    str(record["lean_statement"]),
                    output_proof,
                )
            except ValueError as error:
                rejected_reason = str(error)
                full_lean_code = ""
            attempts.append(
                {
                    "id": record.get("id", f"problem-{problem_index}"),
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "raw_completion": raw_completion,
                    "generated_proof": output_proof,
                    "rejected_reason": rejected_reason,
                    "lean_statement": record["lean_statement"],
                    "imports": record.get("imports")
                    or (record.get("preamble") or {}).get("imports")
                    or [],
                    "context_lines": record.get("context_lines")
                    or (record.get("preamble") or {}).get("context_lines")
                    or [],
                    "full_lean_code": full_lean_code,
                }
            )
    return attempts


def extract_proof_body(completion: str) -> str:
    text = completion.strip()
    text = _take_before_stop_marker(text)
    text = _strip_code_fence(text)
    if "###" in text:
        text = text.split("###", 1)[0].strip()
    if "```" in text:
        text = text.split("```", 1)[0].strip()
    return cleanup_proof_body(text)


def compile_attempts(
    attempts: list[dict[str, Any]],
    verifier: PantographTheoremVerifier,
    *,
    default_imports: tuple[str, ...],
) -> list[dict[str, Any]]:
    compiled = []
    for attempt in attempts:
        lean_code = str(attempt["full_lean_code"])
        imports = tuple(attempt.get("imports") or default_imports)
        context_lines = tuple(attempt.get("context_lines") or ())
        if attempt.get("rejected_reason"):
                result = {
                    "success": False,
                    "diagnostics": attempt["rejected_reason"],
                    "errors": [attempt["rejected_reason"]],
                    "warnings": [],
                    "messages": [attempt["rejected_reason"]],
                }
        elif contains_forbidden_proof_token(lean_code):
            result = {
                "success": False,
                "diagnostics": "rejected generated proof containing sorry/admit",
                "errors": ["rejected generated proof containing sorry/admit"],
                "warnings": [],
                "messages": ["rejected generated proof containing sorry/admit"],
            }
        else:
            verifier_imports = tuple(verifier.imports)
            mathlib_covers_attempt_imports = verifier_imports == ("Mathlib",)
            if imports != verifier_imports and not mathlib_covers_attempt_imports:
                result = {
                    "success": False,
                    "diagnostics": (
                        "attempt imports differ from the active Pantograph "
                        f"server imports: attempt={imports}, server={verifier_imports}"
                    ),
                    "check_seconds": 0.0,
                    "timed_out": False,
                    "errors": [
                        "attempt imports differ from the active Pantograph "
                        f"server imports: attempt={imports}, server={verifier_imports}"
                    ],
                    "warnings": [],
                    "messages": [
                        "attempt imports differ from the active Pantograph "
                        f"server imports: attempt={imports}, server={verifier_imports}"
                    ],
                }
            else:
                source = build_lean_source_with_preamble(
                    lean_code,
                    context_lines=context_lines,
                )
                result = verifier.check_source(
                    source,
                    timeout=verifier.timeout,
                ).to_json()
        compiled.append(
            {
                **attempt,
                "success": bool(result.get("success")),
                "diagnostics": result.get("diagnostics", ""),
                "check_seconds": result.get("check_seconds"),
                "timed_out": result.get("timed_out"),
                "compile_messages": result.get("messages", []),
                "compile_errors": result.get("errors", []),
                "compile_warnings": result.get("warnings", []),
                "has_compile_errors": bool(result.get("errors", [])),
                "has_compile_warnings": bool(result.get("warnings", [])),
                "verifier_backend": "pantograph",
            }
        )
    return compiled


def summarize_pass_at_k(
    compiled_attempts: list[dict[str, Any]],
    pass_k: tuple[int, ...],
) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for attempt in compiled_attempts:
        by_problem.setdefault(str(attempt["id"]), []).append(attempt)

    summary: dict[str, Any] = {
        "num_problems": len(by_problem),
        "num_attempts": len(compiled_attempts),
        "pass_at_k": {},
    }
    for k in pass_k:
        successes = 0
        for attempts in by_problem.values():
            ordered = sorted(attempts, key=lambda item: int(item["sample_index"]))
            if any(bool(item["success"]) for item in ordered[:k]):
                successes += 1
        summary["pass_at_k"][f"pass@{k}"] = (
            successes / len(by_problem) if by_problem else 0.0
        )
    return summary


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(record: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def benchmark_model(
    *,
    label: str,
    model,
    tokenizer,
    records: list[dict[str, Any]],
    config: BenchmarkConfig,
    verifier: PantographTheoremVerifier,
) -> dict[str, Any]:
    print(f"generating {label} attempts...")
    attempts = generate_proof_attempts(
        model,
        tokenizer,
        records,
        attempts_per_problem=max(config.pass_k),
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    print(f"compiling {label} attempts...")
    compiled = compile_attempts(
        attempts,
        verifier,
        default_imports=config.imports,
    )
    output_dir = Path(config.output_dir)
    write_jsonl(compiled, output_dir / f"{label}_attempts.jsonl")
    summary = summarize_pass_at_k(compiled, config.pass_k)
    summary.update(
        {
            "model_label": label,
            "benchmark_file": config.benchmark_file,
            "base_model_name_or_path": config.base_model_name_or_path,
            "adapter_path": config.adapter_path if label == "adapter" else None,
            "lean_project_path": config.lean_project_path,
        }
    )
    write_json(summary, output_dir / f"{label}_summary.json")
    return summary


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _take_before_stop_marker(text: str) -> str:
    markers = ("\n\n###", "\n### Informal", "\n### Lean statement")
    end = len(text)
    for marker in markers:
        position = text.find(marker)
        if position != -1:
            end = min(end, position)
    return text[:end]


def main() -> None:
    config = parse_args()
    records = read_jsonl(config.benchmark_file)
    verifier = PantographTheoremVerifier(
        config.lean_project_path,
        timeout=config.lean_timeout,
        imports=config.imports,
    )
    try:
        tokenizer = load_tokenizer(config.base_model_name_or_path)

        base_model = load_model(
            config.base_model_name_or_path,
            load_in_4bit=config.load_in_4bit,
        )
        base_summary = benchmark_model(
            label="base",
            model=base_model,
            tokenizer=tokenizer,
            records=records,
            config=config,
            verifier=verifier,
        )
        unload_model(base_model)

        summaries = {"base": base_summary}
        if config.adapter_path:
            adapter_model = load_adapter_model(
                config.base_model_name_or_path,
                config.adapter_path,
                load_in_4bit=config.load_in_4bit,
            )
            adapter_summary = benchmark_model(
                label="adapter",
                model=adapter_model,
                tokenizer=tokenizer,
                records=records,
                config=config,
                verifier=verifier,
            )
            unload_model(adapter_model)
            summaries["adapter"] = adapter_summary

        write_json(summaries, Path(config.output_dir) / "summary.json")
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    finally:
        verifier.close()


if __name__ == "__main__":
    main()



