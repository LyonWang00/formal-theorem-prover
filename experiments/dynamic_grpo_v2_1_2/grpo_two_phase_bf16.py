"""Hash-bound two-phase GRPO rollout for disjoint multi-GPU generation.

Phase A runs one generation-only process per frozen ID shard.  Phase B owns one
global two-worker Pantograph pool and verifies the frozen generation chunks.
The split prevents four GPU processes from accidentally starting eight Lean
workers while keeping every problem/attempt key resumable and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - local Windows smoke only
    fcntl = None  # type: ignore[assignment]

from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    compose_lean_theorem,
    contains_forbidden_proof_token,
    normalize_proof_for_assembly,
)
from lean_prover.lean_training.evaluation.benchmark import (
    create_generator,
    extract_proof_body,
)
from lean_prover.lean_training.evaluation.rollout import read_rollout_records
from transformers import AutoTokenizer
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
if (
    os.environ.get("PANTOGRAPH_POOL_SOCKET_DIR")
    and os.environ.get("PANTOGRAPH_POOL_CLIENT_MODE")
):
    from persistent_pantograph_service import (
        PersistentVerificationPool as VerificationPool,
    )
    from lean_prover.lean_training.verification.pool import VerificationPoolConfig
else:
    from lean_prover.lean_training.verification.pool import (
        VerificationPool,
        VerificationPoolConfig,
    )
from lean_prover.lean_training.verification.schema import (
    VerificationTask,
    make_verification_result,
)


ORIGIN = "compute_grpo"
DOMAIN = "slurm_node_local"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def verify_hash(path: str | Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: expected={expected} actual={actual}")


def generation_payload_sha256(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "generation_payload_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def chunk_set_sha256(chunks: Path) -> str:
    return hashlib.sha256(
        "".join(
            f"{path.name}:{sha256_file(path)}\n"
            for path in sorted(chunks.glob("*.jsonl"))
        ).encode("utf-8")
    ).hexdigest()


def plan_shards(count: int, shard_count: int) -> list[list[int]]:
    if count < 0 or shard_count <= 0:
        raise ValueError("invalid shard dimensions")
    shards = [[] for _ in range(shard_count)]
    for index in range(count):
        shards[index % shard_count].append(index)
    return shards


def generation_queue_windows(
    items: Sequence[Any], queue_depth: int
) -> Iterable[Sequence[Any]]:
    """Yield bounded request queues for vLLM's continuous scheduler.

    A queue may be larger than the historical 128-problem submission group.
    vLLM then admits waiting requests as active sequences finish, avoiding an
    idle tail after every 128-problem group while keeping restart loss bounded.
    """
    if queue_depth <= 0:
        raise ValueError("generation queue depth must be positive")
    for start in range(0, len(items), queue_depth):
        yield items[start : start + queue_depth]


def freeze_shards(args: argparse.Namespace) -> dict[str, Any]:
    verify_hash(args.data, args.data_sha256, "data")
    verify_hash(args.manifest, args.manifest_sha256, "manifest")
    verify_hash(args.audit, args.audit_sha256, "audit")
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("GRPO audit status is not PASS")
    audit_artifacts = audit.get("artifacts") or {}
    if audit_artifacts.get("data_sha256") != args.data_sha256:
        raise ValueError("GRPO audit is not bound to the requested data SHA")
    if audit_artifacts.get("manifest_sha256") != args.manifest_sha256:
        raise ValueError("GRPO audit is not bound to the requested manifest SHA")
    tokenizer = getattr(args, "_tokenizer_for_test", None)
    if tokenizer is None:
        tokenizer_root = Path(args.tokenizer_path).resolve()
        tokenizer_sums = tokenizer_root / "TOKENIZER_SHA256SUMS"
        verify_hash(
            tokenizer_sums,
            args.tokenizer_sums_sha256,
            "SFT tokenizer TOKENIZER_SHA256SUMS",
        )
        # Match the authoritative SFT vLLM contract exactly: vLLM receives this
        # explicit tokenizer path, and the offline freeze uses AutoTokenizer
        # from the same immutable directory.
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_root),
            trust_remote_code=False,
            local_files_only=True,
            padding_side="left",
        )
    records = read_rollout_records(args.data, rollout_kind="grpo")
    data_rows = list(iter_jsonl(args.data))
    manifest_rows = list(iter_jsonl(args.manifest))
    if len(records) != len(data_rows) or len(records) != len(manifest_rows):
        raise ValueError("data/manifest row count mismatch")
    if args.expected_count is not None and len(records) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} records, got {len(records)}")
    if int(audit.get("final_records", -1)) != len(records):
        raise ValueError("GRPO audit record count does not match the data")
    audit_gates = audit.get("full_audit") or {}
    required_zero_gates = (
        "manifest_data_projection_mismatches",
        "prompt_statement_semantic_mismatches",
        "noncanonical_prompts",
        "invalid_imports",
        "proof_bearing_field_leaks",
    )
    if any(int(audit_gates.get(key, -1)) != 0 for key in required_zero_gates):
        raise ValueError("GRPO audit contains a nonzero or missing semantic gate")
    problem_ids = [record.problem_id for record in records]
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError("problem IDs are not globally unique")
    for index, (record, data_row, manifest_row) in enumerate(
        zip(records, data_rows, manifest_rows)
    ):
        manifest_id = str(manifest_row.get("record_id") or manifest_row.get("id") or "")
        data_id = str(data_row.get("record_id") or data_row.get("id") or "")
        if manifest_id != record.problem_id or data_id != record.problem_id:
            raise ValueError(f"data/manifest problem ID mismatch at row {index}")
        if str(manifest_row.get("statement_hash") or "") != str(
            data_row.get("statement_hash") or ""
        ):
            raise ValueError(f"data/manifest stored statement hash mismatch at row {index}")
        if (
            str(manifest_row.get("lean_statement") or "").strip()
            != str(data_row.get("lean_statement") or "").strip()
            or str(manifest_row.get("lean_statement") or "").strip()
            != record.lean_statement
        ):
            raise ValueError(f"data/manifest Lean statement mismatch at row {index}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shard_indices = plan_shards(len(records), args.shards)
    shard_reports: list[dict[str, Any]] = []
    union: set[str] = set()
    prompt_token_counts: list[int] = []
    effective_token_counts: list[int] = []
    token_contract: dict[int, tuple[int, int]] = {}
    for global_index, record in enumerate(records):
        prompt_tokens = len(tokenizer(record.prompt, add_special_tokens=True)["input_ids"])
        effective_max_new_tokens = min(
            args.max_new_tokens,
            args.max_model_len - prompt_tokens,
        )
        if effective_max_new_tokens <= 0:
            raise ValueError(
                f"nonpositive generation budget at row {global_index}: "
                f"prompt_tokens={prompt_tokens} model_limit={args.max_model_len}"
            )
        prompt_token_counts.append(prompt_tokens)
        effective_token_counts.append(effective_max_new_tokens)
        token_contract[global_index] = (prompt_tokens, effective_max_new_tokens)
    for shard_index, indices in enumerate(shard_indices):
        rows = []
        for global_index in indices:
            record = records[global_index]
            prompt_sha256 = hashlib.sha256(record.prompt.encode("utf-8")).hexdigest()
            prompt_tokens, effective_max_new_tokens = token_contract[global_index]
            rows.append(
                {
                    "global_index": global_index,
                    "problem_id": record.problem_id,
                    "statement_hash": record.statement_hash,
                    "manifest_statement_hash": manifest_rows[global_index][
                        "statement_hash"
                    ],
                    "prompt_sha256": prompt_sha256,
                    "prompt_tokens": prompt_tokens,
                    "effective_max_new_tokens": effective_max_new_tokens,
                    "max_new_tokens_ceiling": args.max_new_tokens,
                    "max_model_len": args.max_model_len,
                    "generation_backend": "vllm",
                    "generation_batch_size": int(
                        getattr(args, "generation_batch_size", 128)
                    ),
                    "sampling_seed_base": int(getattr(args, "seed", 20260711)),
                    "temperature": float(getattr(args, "temperature", 1.0)),
                    "top_p": float(getattr(args, "top_p", 0.95)),
                }
            )
        ids = {row["problem_id"] for row in rows}
        if union & ids:
            raise ValueError("cross-shard problem ID overlap")
        union.update(ids)
        path = output / f"shard_{shard_index}.jsonl"
        atomic_write_text(path, "".join(canonical_json(row) + "\n" for row in rows))
        shard_reports.append(
            {
                "shard_index": shard_index,
                "path": path.name,
                "count": len(rows),
                "sha256": sha256_file(path),
            }
        )
    if union != set(problem_ids):
        raise ValueError("shard union does not equal the authoritative problem set")
    report = {
        "schema": "grpo_generation_shards_v1",
        "status": "frozen",
        "data": Path(args.data).name,
        "data_sha256": args.data_sha256,
        "manifest": Path(args.manifest).name,
        "manifest_sha256": args.manifest_sha256,
        "audit": Path(args.audit).name,
        "audit_sha256": args.audit_sha256,
        "problem_count": len(records),
        "attempts_per_problem": args.pass_k,
        "expected_attempt_count": len(records) * args.pass_k,
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_sums_sha256": args.tokenizer_sums_sha256,
        "max_model_len": args.max_model_len,
        "max_new_tokens_ceiling": args.max_new_tokens,
        "generation_backend": "vllm",
        "generation_batch_size": int(getattr(args, "generation_batch_size", 128)),
        "sampling_seed_base": int(getattr(args, "seed", 20260711)),
        "temperature": float(getattr(args, "temperature", 1.0)),
        "top_p": float(getattr(args, "top_p", 0.95)),
        "prompt_tokens_min": min(prompt_token_counts),
        "prompt_tokens_max": max(prompt_token_counts),
        "effective_max_new_tokens_min": min(effective_token_counts),
        "effective_max_new_tokens_max": max(effective_token_counts),
        "reduced_budget_problem_count": sum(
            count < args.max_new_tokens for count in effective_token_counts
        ),
        "assignment": "zero_based_authoritative_row_index_mod_shard_count",
        "shard_count": args.shards,
        "shards": shard_reports,
    }
    atomic_write_json(output / "SHARDS_FROZEN.json", report)
    return report


def _load_frozen_shard(
    path: str | Path, expected_sha256: str
) -> list[dict[str, Any]]:
    verify_hash(path, expected_sha256, "shard manifest")
    rows = list(iter_jsonl(path))
    indices = [int(row["global_index"]) for row in rows]
    ids = [str(row["problem_id"]) for row in rows]
    if len(set(indices)) != len(rows) or len(set(ids)) != len(rows):
        raise ValueError("shard manifest contains duplicate index or problem ID")
    return rows


def _validate_chunk(
    path: Path,
    manifest_row: Mapping[str, Any],
    *,
    pass_k: int,
    model_sums_sha256: str | None = None,
    expected_vllm_version: str | None = None,
    expected_bitsandbytes_version: str | None = None,
) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    expected_id = str(manifest_row["problem_id"])
    expected_index = int(manifest_row["global_index"])
    keys = set()
    for row in rows:
        if row.get("generation_backend") != "vllm":
            raise ValueError(f"chunk is not from the vLLM backend: {path}")
        if row.get("precision") != "bfloat16":
            raise ValueError(f"chunk quantization contract mismatch: {path}")
        if (
            expected_vllm_version is not None
            and row.get("vllm_version") != expected_vllm_version
        ):
            raise ValueError(f"chunk vLLM version mismatch: {path}")
        if (
            expected_bitsandbytes_version is not None
            and row.get("bitsandbytes_version") != expected_bitsandbytes_version
        ):
            raise ValueError(f"chunk bitsandbytes version mismatch: {path}")
        if int(row.get("generation_seed", -1)) != int(
            manifest_row["sampling_seed_base"]
        ) + expected_index:
            raise ValueError(f"chunk sampling seed mismatch: {path}")
        if row.get("problem_id") != expected_id:
            raise ValueError(f"chunk problem ID mismatch: {path}")
        if int(row.get("problem_index", -1)) != expected_index:
            raise ValueError(f"chunk problem index mismatch: {path}")
        key = (expected_id, int(row.get("attempt_index", -1)))
        keys.add(key)
        if row.get("receipt_origin") != ORIGIN or row.get("execution_domain") != DOMAIN:
            raise ValueError(f"chunk origin/domain mismatch: {path}")
        if str(row.get("statement_hash") or "") != str(manifest_row["statement_hash"]):
            raise ValueError(f"chunk statement hash mismatch: {path}")
        if hashlib.sha256(str(row.get("prompt") or "").encode("utf-8")).hexdigest() != str(
            manifest_row["prompt_sha256"]
        ):
            raise ValueError(f"chunk prompt hash mismatch: {path}")
        if row.get("generation_payload_sha256") != generation_payload_sha256(row):
            raise ValueError(f"chunk generation payload hash mismatch: {path}")
        if model_sums_sha256 is not None and row.get("model_sums_sha256") != model_sums_sha256:
            raise ValueError(f"chunk merged-model hash mismatch: {path}")
        prompt_tokens = int(row.get("prompt_tokens", -1))
        effective_max_new_tokens = int(row.get("effective_max_new_tokens", -1))
        if prompt_tokens != int(manifest_row["prompt_tokens"]):
            raise ValueError(f"chunk prompt token count mismatch: {path}")
        if effective_max_new_tokens != int(manifest_row["effective_max_new_tokens"]):
            raise ValueError(f"chunk generation budget mismatch: {path}")
        if (
            effective_max_new_tokens <= 0
            or effective_max_new_tokens > int(manifest_row["max_new_tokens_ceiling"])
            or prompt_tokens + effective_max_new_tokens > int(manifest_row["max_model_len"])
        ):
            raise ValueError(f"chunk token budget contract violation: {path}")
    expected_keys = {(expected_id, attempt) for attempt in range(pass_k)}
    if keys != expected_keys or len(rows) != pass_k:
        raise ValueError(f"chunk does not contain exactly pass@{pass_k}: {path}")
    return rows


def _generator_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        generation_backend=args.generation_backend,
        model_name_or_path=args.model,
        tokenizer_path=args.tokenizer_path,
        adapter_path=None,
        vllm_max_model_len=args.model_max_length,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_enforce_eager=True,
        vllm_kv_cache_memory_bytes=None,
        load_in_4bit=False,
        generation_contract_path=None,
        generation_contract_sha256=None,
    )


def quantization_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Bind Phase A to the audited SFT vLLM BnB-NF4 implementation."""
    import importlib.metadata
    import vllm

    installed = importlib.metadata.version("bitsandbytes")
    if args.expected_bitsandbytes_version and installed != args.expected_bitsandbytes_version:
        raise RuntimeError("bitsandbytes version mismatch")
    vllm_root = Path(vllm.__file__).resolve().parent
    loader = vllm_root / "model_executor/model_loader/bitsandbytes_loader.py"
    linear = vllm_root / "model_executor/layers/quantization/bitsandbytes.py"
    loader_sha256 = sha256_file(loader)
    linear_sha256 = sha256_file(linear)
    if args.expected_bnb_loader_sha256 and loader_sha256 != args.expected_bnb_loader_sha256:
        raise RuntimeError("vLLM bitsandbytes loader source hash mismatch")
    if args.expected_bnb_linear_sha256 and linear_sha256 != args.expected_bnb_linear_sha256:
        raise RuntimeError("vLLM bitsandbytes linear source hash mismatch")
    loader_text = loader.read_text(encoding="utf-8")
    linear_text = linear.read_text(encoding="utf-8")
    if not all(fragment in loader_text for fragment in ("compress_statistics=True", 'quant_type="nf4"')):
        raise RuntimeError("vLLM loader no longer proves NF4 double quantization")
    if not all(fragment in linear_text for fragment in ("x.to(torch.bfloat16)", "out.to(original_type)")):
        raise RuntimeError("vLLM BnB linear path no longer proves BF16 matmul")
    marker_sha256 = None
    if args.bitsandbytes_install_marker:
        marker = Path(args.bitsandbytes_install_marker).resolve()
        marker_sha256 = sha256_file(marker)
        if (
            args.expected_bitsandbytes_install_marker_sha256
            and marker_sha256 != args.expected_bitsandbytes_install_marker_sha256
        ):
            raise RuntimeError("bitsandbytes install marker hash mismatch")
    return {
        "backend": "bitsandbytes",
        "installed_version": installed,
        "install_marker_sha256": marker_sha256,
        "loader_source_sha256": loader_sha256,
        "linear_source_sha256": linear_sha256,
        "online_quant_type": "nf4",
        "compress_statistics": True,
        "double_quant_equivalent": True,
        "matmul_compute_dtype": "bfloat16",
    }


def generate_shard(args: argparse.Namespace) -> dict[str, Any]:
    verify_hash(args.data, args.data_sha256, "data")
    model_path = Path(args.model).resolve()
    model_sums = Path(args.model_sums).resolve()
    if model_sums.parent != model_path:
        raise ValueError("merged-model SHA256SUMS must live inside the model directory")
    verify_hash(model_sums, args.model_sums_sha256, "merged-model SHA256SUMS")
    manifest_rows = _load_frozen_shard(args.shard_manifest, args.shard_sha256)
    records = read_rollout_records(args.data, rollout_kind="grpo")
    data_rows = list(iter_jsonl(args.data))
    selected = []
    for row in manifest_rows:
        index = int(row["global_index"])
        record = records[index]
        if record.problem_id != row["problem_id"]:
            raise ValueError(f"problem ID drift at global index {index}")
        if record.statement_hash != row["statement_hash"]:
            raise ValueError(f"statement hash drift at global index {index}")
        if str(data_rows[index].get("statement_hash") or "") != str(
            row["manifest_statement_hash"]
        ):
            raise ValueError(f"stored statement hash drift at global index {index}")
        if hashlib.sha256(record.prompt.encode("utf-8")).hexdigest() != row["prompt_sha256"]:
            raise ValueError(f"prompt hash drift at global index {index}")
        if int(row["max_new_tokens_ceiling"]) != args.max_new_tokens:
            raise ValueError(f"generation ceiling mismatch at global index {index}")
        if int(row["max_model_len"]) != args.model_max_length:
            raise ValueError(f"model length mismatch at global index {index}")
        if row.get("generation_backend") != "vllm":
            raise ValueError(f"frozen backend mismatch at global index {index}")
        if int(row["generation_batch_size"]) != args.generation_batch_size:
            raise ValueError(f"generation batch size mismatch at global index {index}")
        if int(row["sampling_seed_base"]) != args.seed:
            raise ValueError(f"sampling seed mismatch at global index {index}")
        if float(row["temperature"]) != args.temperature or float(row["top_p"]) != args.top_p:
            raise ValueError(f"sampling parameter mismatch at global index {index}")
        selected.append((row, record))

    output = Path(args.output)
    chunks = output / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    completed: set[int] = set()
    finish_counts: Counter[str] = Counter()
    effective_token_caps: dict[int, int] = {}
    for row, _ in selected:
        index = int(row["global_index"])
        path = chunks / f"problem_{index:06d}.jsonl"
        if path.exists():
            existing_attempts = _validate_chunk(
                path,
                row,
                pass_k=args.pass_k,
                model_sums_sha256=args.model_sums_sha256,
                expected_vllm_version=args.expected_vllm_version,
                expected_bitsandbytes_version=args.expected_bitsandbytes_version,
            )
            for attempt in existing_attempts:
                finish_counts[str(attempt.get("finish_reason") or "unknown")] += 1
            if existing_attempts:
                effective_token_caps[index] = int(
                    existing_attempts[0]["effective_max_new_tokens"]
                )
            completed.add(index)
    if args.generation_backend != "vllm":
        raise ValueError("GRPO Phase A defaults to and requires vLLM")
    import importlib.metadata

    runtime_versions = {
        "vllm": importlib.metadata.version("vllm"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
    }
    quantization = {"backend": "none", "matmul_compute_dtype": "bfloat16"}
    if args.expected_vllm_version and runtime_versions["vllm"] != args.expected_vllm_version:
        raise ValueError(f"vLLM version mismatch: {runtime_versions['vllm']}")
    if (
        args.expected_bitsandbytes_version
        and runtime_versions["bitsandbytes"] != args.expected_bitsandbytes_version
    ):
        raise ValueError(
            f"bitsandbytes version mismatch: {runtime_versions['bitsandbytes']}"
        )
    if len(completed) == len(selected):
        generator = None
    else:
        generator = create_generator(_generator_args(args))

    started = time.monotonic()
    pending = [item for item in selected if int(item[0]["global_index"]) not in completed]
    queue_depth = int(
        getattr(args, "generation_queue_depth", None) or args.generation_batch_size
    )
    if queue_depth < args.generation_batch_size:
        raise ValueError("generation queue depth cannot be smaller than batch size")
    for batch in generation_queue_windows(pending, queue_depth):
        prompts: list[str] = []
        token_limits: list[int] = []
        seeds: list[int] = []
        tokenizer_count_rows: list[dict[str, int]] = []
        for manifest_row, record in batch:
            global_index = int(manifest_row["global_index"])
            prompt_tokens = len(
                generator.tokenizer(record.prompt, add_special_tokens=True)["input_ids"]
            )
            effective_max_new_tokens = min(
                args.max_new_tokens,
                args.model_max_length - prompt_tokens,
            )
            if prompt_tokens != int(manifest_row["prompt_tokens"]):
                raise ValueError(f"runtime tokenizer drift at global index {global_index}")
            if effective_max_new_tokens != int(manifest_row["effective_max_new_tokens"]):
                raise ValueError(f"runtime token budget drift at global index {global_index}")
            prompts.append(record.prompt)
            token_limits.append(effective_max_new_tokens)
            seeds.append(args.seed + global_index)
            effective_token_caps[global_index] = effective_max_new_tokens
            tokenizer_count_rows.append(
                {
                    "global_index": global_index,
                    "actual_prompt_tokens": prompt_tokens,
                    "frozen_prompt_tokens": int(manifest_row["prompt_tokens"]),
                }
            )
        print(
            canonical_json(
                {
                    "schema": "grpo_runtime_tokenizer_count_gate_v1",
                    "status": "PASS",
                    "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
                    "rows": tokenizer_count_rows,
                }
            ),
            flush=True,
        )
        generated_rows = generator.generate_batch(
            prompts,
            k=args.pass_k,
            max_new_tokens=token_limits,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=seeds,
        )
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for generated in generated_rows:
            grouped[int(generated["problem_batch_index"])].append(generated)
        if set(grouped) != set(range(len(batch))):
            raise ValueError("vLLM batch did not return every prompt")
        for batch_index, (manifest_row, record) in enumerate(batch):
            global_index = int(manifest_row["global_index"])
            prompt_tokens = int(manifest_row["prompt_tokens"])
            effective_max_new_tokens = int(manifest_row["effective_max_new_tokens"])
            generated_for_problem = sorted(
                grouped[batch_index], key=lambda row: int(row["local_attempt_index"])
            )
            if len(generated_for_problem) != args.pass_k:
                raise ValueError(f"vLLM pass@{args.pass_k} output mismatch")
            chunk_rows = []
            for generated in generated_for_problem:
                attempt_index = int(generated["local_attempt_index"])
                raw_completion = str(generated["raw_completion"])
                generated_proof = extract_proof_body(raw_completion)
                normalized_proof, proof_format = normalize_proof_for_assembly(
                    generated_proof
                )
                rejected_reason = None
                try:
                    lean_code = compose_lean_theorem(
                        record.lean_statement,
                        normalized_proof,
                        proof_format=proof_format,
                    )
                except ValueError as error:
                    lean_code = ""
                    rejected_reason = str(error)
                if contains_forbidden_proof_token(generated_proof):
                    rejected_reason = (
                        rejected_reason or "generated proof contains sorry/admit"
                    )
                result = {
                    "generation_backend": "vllm",
                    "generation_scheduler": "continuous_waiting_queue",
                    "generation_queue_depth": queue_depth,
                    "precision": "bfloat16",
                    "vllm_version": runtime_versions["vllm"],
                    "bitsandbytes_version": runtime_versions["bitsandbytes"],
                    "problem_id": record.problem_id,
                    "problem_index": global_index,
                    "attempt_id": f"{global_index},{attempt_index}",
                    "attempt_index": attempt_index,
                    "prompt": record.prompt,
                    "statement": record.lean_statement,
                    "statement_hash": record.statement_hash,
                    "imports": list(record.imports),
                    "context_lines": list(record.context_lines),
                    "raw_completion": raw_completion,
                    "generated_proof": generated_proof,
                    "normalized_proof": normalized_proof,
                    "proof_format": proof_format.value,
                    "lean_code": lean_code,
                    "rejected_reason": rejected_reason,
                    "assembler_version": ASSEMBLER_VERSION,
                    "normalization_version": NORMALIZATION_VERSION,
                    "generation_problem_seconds": generated.get(
                        "generation_problem_seconds"
                    ),
                    "generation_batch_seconds": generated.get(
                        "generation_batch_seconds"
                    ),
                    "completion_tokens": generated.get("completion_tokens"),
                    "finish_reason": generated.get("finish_reason"),
                    "hit_eos": generated.get("hit_eos"),
                    "hit_max_new_tokens": generated.get("hit_max_new_tokens"),
                    "prompt_tokens": prompt_tokens,
                    "requested_max_new_tokens": args.max_new_tokens,
                    "effective_max_new_tokens": effective_max_new_tokens,
                    "model_max_length": args.model_max_length,
                    "generation_seed": args.seed + global_index,
                    "model_sums_sha256": args.model_sums_sha256,
                    "receipt_origin": ORIGIN,
                    "execution_domain": DOMAIN,
                }
                result["generation_payload_sha256"] = generation_payload_sha256(
                    result
                )
                chunk_rows.append(result)
                finish_counts[str(result["finish_reason"] or "unknown")] += 1
            chunk_path = chunks / f"problem_{global_index:06d}.jsonl"
            atomic_write_text(
                chunk_path,
                "".join(canonical_json(row) + "\n" for row in chunk_rows),
            )
            _validate_chunk(
                chunk_path,
                manifest_row,
                pass_k=args.pass_k,
                model_sums_sha256=args.model_sums_sha256,
                expected_vllm_version=args.expected_vllm_version,
                expected_bitsandbytes_version=args.expected_bitsandbytes_version,
            )
            completed.add(global_index)
        atomic_write_json(
            output / "progress.json",
            {
                "status": "running",
                "shard_manifest_sha256": args.shard_sha256,
                "completed_problems": len(completed),
                "expected_problems": len(selected),
                "completed_attempts": len(completed) * args.pass_k,
                "finish_reason_counts": dict(sorted(finish_counts.items())),
                "elapsed_seconds": round(time.monotonic() - started, 4),
                "last_global_index": max(
                    int(manifest_row["global_index"]) for manifest_row, _ in batch
                ),
                "generation_backend": "vllm",
                "generation_scheduler": "continuous_waiting_queue",
                "generation_batch_size": args.generation_batch_size,
                "generation_queue_depth": queue_depth,
            },
        )
    report = {
        "schema": "grpo_generation_shard_complete_v1",
        "status": "complete",
        "shard_manifest": str(Path(args.shard_manifest).resolve()),
        "shard_manifest_sha256": args.shard_sha256,
        "data_sha256": args.data_sha256,
        "model": str(Path(args.model).resolve()),
        "model_sums_sha256": args.model_sums_sha256,
        "problem_count": len(selected),
        "attempt_count": len(selected) * args.pass_k,
        "pass_k": args.pass_k,
        "generation_backend": "vllm",
        "generation_scheduler": "continuous_waiting_queue",
        "generation_batch_size": args.generation_batch_size,
        "generation_queue_depth": queue_depth,
        "runtime_versions": runtime_versions,
        "quantization_audit": quantization,
        "max_new_tokens": args.max_new_tokens,
        "model_max_length": args.model_max_length,
        "effective_max_new_tokens_min": min(effective_token_caps.values(), default=0),
        "effective_max_new_tokens_max": max(effective_token_caps.values(), default=0),
        "context_capped_problem_count": sum(
            cap < args.max_new_tokens for cap in effective_token_caps.values()
        ),
        "seed": args.seed,
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "chunks_sha256": chunk_set_sha256(chunks),
        "receipt_origin": ORIGIN,
        "execution_domain": DOMAIN,
    }
    atomic_write_json(output / "GENERATION_COMPLETE.json", report)
    return report


def _generation_rows(
    frozen: Mapping[str, Any],
    frozen_root: Path,
    generation_root: Path,
    *,
    pass_k: int,
    model_sums_sha256: str,
) -> Iterable[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    for shard in frozen["shards"]:
        shard_path = Path(shard["path"])
        if not shard_path.is_absolute():
            shard_path = frozen_root / shard_path
        manifest_rows = _load_frozen_shard(shard_path, shard["sha256"])
        shard_root = generation_root / f"shard_{int(shard['shard_index'])}"
        complete = json.loads((shard_root / "GENERATION_COMPLETE.json").read_text(encoding="utf-8"))
        if complete["status"] != "complete" or complete["shard_manifest_sha256"] != shard["sha256"]:
            raise ValueError(f"generation shard {shard['shard_index']} is not complete")
        expected_shard_attempts = int(shard["count"]) * pass_k
        if (
            int(complete.get("problem_count", -1)) != int(shard["count"])
            or int(complete.get("attempt_count", -1)) != expected_shard_attempts
            or int(complete.get("pass_k", -1)) != pass_k
            or complete.get("data_sha256") != frozen["data_sha256"]
            or complete.get("model_sums_sha256") != model_sums_sha256
            or int(complete.get("max_new_tokens", -1))
            != int(frozen["max_new_tokens_ceiling"])
            or int(complete.get("model_max_length", -1))
            != int(frozen["max_model_len"])
            or complete.get("receipt_origin") != ORIGIN
            or complete.get("execution_domain") != DOMAIN
            or complete.get("generation_backend") != "vllm"
        ):
            raise ValueError(f"generation shard {shard['shard_index']} contract mismatch")
        chunks = shard_root / "chunks"
        if complete.get("chunks_sha256") != chunk_set_sha256(chunks):
            raise ValueError(f"generation shard {shard['shard_index']} chunk-chain mismatch")
        for manifest_row in manifest_rows:
            index = int(manifest_row["global_index"])
            chunk = shard_root / "chunks" / f"problem_{index:06d}.jsonl"
            for row in _validate_chunk(
                chunk,
                manifest_row,
                pass_k=pass_k,
                model_sums_sha256=model_sums_sha256,
                expected_vllm_version=str(complete["runtime_versions"]["vllm"]),
                expected_bitsandbytes_version=str(
                    complete["runtime_versions"]["bitsandbytes"]
                ),
            ):
                key = (str(row["problem_id"]), int(row["attempt_index"]))
                if key in seen:
                    raise ValueError(f"duplicate global generation key: {key}")
                seen.add(key)
                yield row
    expected = int(frozen["problem_count"]) * pass_k
    if len(seen) != expected:
        raise ValueError(f"expected {expected} generation keys, got {len(seen)}")


def _task_from_generation(row: Mapping[str, Any]) -> VerificationTask:
    payload = {
        "normalized_proof": row["normalized_proof"],
        "proof_format": row["proof_format"],
        "assembler_version": row["assembler_version"],
        "normalization_version": row["normalization_version"],
        "statement": row["statement"],
        "statement_hash": row["statement_hash"],
        "generation_batch_seconds": row.get("generation_batch_seconds"),
        "generation_problem_seconds": row.get("generation_problem_seconds"),
        "completion_tokens": row.get("completion_tokens"),
        "finish_reason": row.get("finish_reason"),
        "hit_eos": row.get("hit_eos"),
        "hit_max_new_tokens": row.get("hit_max_new_tokens"),
        "generation_seed": row.get("generation_seed"),
        "generation_payload_sha256": row["generation_payload_sha256"],
        "receipt_origin": ORIGIN,
        "execution_domain": DOMAIN,
    }
    task = VerificationTask(
        priority=int(row["problem_index"]),
        problem_index=int(row["problem_index"]),
        attempt_index=int(row["attempt_index"]),
        problem_id=str(row["problem_id"]),
        prompt=str(row["prompt"]),
        generated_proof=str(row["generated_proof"]),
        raw_completion=str(row["raw_completion"]),
        lean_code=str(row["lean_code"]),
        imports=tuple(row["imports"]),
        context_lines=tuple(row["context_lines"]),
        generation_seconds=row.get("generation_problem_seconds"),
        payload=payload,
    )
    if task.lean_code:
        source = build_labeled_lean_code(task, include_imports=True)
        payload["assembled_source"] = source
        payload["assembled_source_hash"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return task


def audit_generations(args: argparse.Namespace) -> dict[str, Any]:
    frozen_path = Path(args.frozen_shards)
    verify_hash(frozen_path, args.frozen_shards_sha256, "frozen shards")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    finish_counts: Counter[str] = Counter()
    attempt_count = 0
    for row in _generation_rows(
        frozen,
        frozen_path.parent,
        Path(args.generation_root),
        pass_k=int(frozen["attempts_per_problem"]),
        model_sums_sha256=args.model_sums_sha256,
    ):
        finish_counts[str(row.get("finish_reason") or "unknown")] += 1
        attempt_count += 1
    expected_attempts = int(frozen["expected_attempt_count"])
    if attempt_count != expected_attempts:
        raise ValueError(f"expected {expected_attempts} generated attempts, got {attempt_count}")
    report = {
        "schema": "grpo_two_phase_generation_audit_v1",
        "status": "PASS",
        "frozen_shards_sha256": args.frozen_shards_sha256,
        "model_sums_sha256": args.model_sums_sha256,
        "problem_count": int(frozen["problem_count"]),
        "attempt_count": attempt_count,
        "attempts_per_problem": int(frozen["attempts_per_problem"]),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "receipt_origin": ORIGIN,
        "execution_domain": DOMAIN,
    }
    atomic_write_json(args.output, report)
    return report


def _load_receipts(
    path: Path,
) -> tuple[
    set[tuple[str, int]],
    dict[str, dict[str, Any]],
    dict[tuple[str, int], str],
]:
    seen: set[tuple[str, int]] = set()
    generation_hashes: dict[tuple[str, int], str] = {}
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"attempts": set(), "successful_attempts": set(), "problem_index": None}
    )
    if not path.exists():
        return seen, stats, generation_hashes
    for row in iter_jsonl(path):
        if row.get("receipt_origin") != ORIGIN or row.get("execution_domain") != DOMAIN:
            raise ValueError("existing receipt origin/domain mismatch")
        key = (str(row["problem_id"]), int(row["attempt_index"]))
        if key in seen:
            raise ValueError(f"duplicate existing receipt key: {key}")
        generation_hash = str(row.get("generation_payload_sha256") or "")
        if len(generation_hash) != 64:
            raise ValueError(f"existing receipt lacks generation payload hash: {key}")
        seen.add(key)
        generation_hashes[key] = generation_hash
        state = stats[key[0]]
        state["problem_index"] = int(row["problem_index"])
        state["attempts"].add(key[1])
        if row.get("success"):
            state["successful_attempts"].add(key[1])
    return seen, stats, generation_hashes


def verify_generations(args: argparse.Namespace) -> dict[str, Any]:
    if fcntl is None:
        raise RuntimeError("Phase B verification requires POSIX file locking")
    frozen_path = Path(args.frozen_shards)
    verify_hash(frozen_path, args.frozen_shards_sha256, "frozen shards")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected_attempts = int(frozen["expected_attempt_count"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    receipts_path = output / "receipts.jsonl"
    seen, stats, receipt_generation_hashes = _load_receipts(receipts_path)
    receipts_handle = receipts_path.open("a", encoding="utf-8", newline="\n")
    checkpoint_lock_handle = (output / "checkpoint.lock").open("a+")
    checkpoint_mutex = threading.Lock()
    result_counts: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()
    verification_seconds: list[float] = []
    for row in iter_jsonl(receipts_path):
        result_counts["success" if row.get("success") else "failure"] += 1
        if row.get("timed_out"):
            result_counts["timeout"] += 1
        finish_counts[str(row.get("finish_reason") or "unknown")] += 1
        if row.get("verification_seconds") is not None:
            verification_seconds.append(float(row["verification_seconds"]))

    def record(result: Mapping[str, Any]) -> None:
        key = (str(result["problem_id"]), int(result["attempt_index"]))
        if key in seen:
            raise ValueError(f"verification pool emitted duplicate result: {key}")
        row = dict(result)
        if row.get("receipt_origin") != ORIGIN or row.get("execution_domain") != DOMAIN:
            raise ValueError("new receipt origin/domain mismatch")
        with checkpoint_mutex:
            fcntl.flock(checkpoint_lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                receipts_handle.write(canonical_json(row) + "\n")
                receipts_handle.flush()
                os.fsync(receipts_handle.fileno())
                seen.add(key)
                receipt_generation_hashes[key] = str(row["generation_payload_sha256"])
                state = stats[key[0]]
                state["problem_index"] = int(row["problem_index"])
                state["attempts"].add(key[1])
                if row.get("success"):
                    state["successful_attempts"].add(key[1])
                    result_counts["success"] += 1
                else:
                    result_counts["failure"] += 1
                if row.get("timed_out"):
                    result_counts["timeout"] += 1
                finish_counts[str(row.get("finish_reason") or "unknown")] += 1
                if row.get("verification_seconds") is not None:
                    verification_seconds.append(float(row["verification_seconds"]))
                atomic_write_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "receipt_count": len(seen),
                        "expected_receipt_count": expected_attempts,
                        "success_count": result_counts["success"],
                        "failure_count": result_counts["failure"],
                        "timeout_count": result_counts["timeout"],
                    },
                )
            finally:
                fcntl.flock(checkpoint_lock_handle.fileno(), fcntl.LOCK_UN)

    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=args.lean_project,
            imports=("Mathlib",),
            timeout=args.timeout,
            warmup_timeout=args.warmup_timeout,
            num_workers=4,
            queue_maxsize=args.queue_maxsize,
            task_spool_dir=args.task_spool_dir,
            cancel_on_success=False,
        )
    )
    fatal_errors: list[str] = []
    recovered_failures: list[str] = []
    started = time.monotonic()
    try:
        pool.start()
        batch: list[VerificationTask] = []

        def flush_batch() -> None:
            nonlocal batch
            if not batch:
                return
            run = pool.run_batch(batch, on_result=record)
            fatal_errors.extend(run.fatal_errors)
            recovered_failures.extend(run.recovered_worker_failures)
            batch = []
            if fatal_errors:
                raise RuntimeError(f"verification pool fatal errors: {fatal_errors}")

        generation_rows = list(
            _generation_rows(
                frozen,
                frozen_path.parent,
                Path(args.generation_root),
                pass_k=int(frozen["attempts_per_problem"]),
                model_sums_sha256=args.model_sums_sha256,
            )
        )
        # Frozen generation files are grouped by shard.  The verification pool
        # routes problem_index % num_workers, so consuming shard-by-shard sends
        # an entire batch to one worker and leaves the others idle.  A stable
        # problem/attempt ordering balances every batch without changing any
        # generation or receipt identity.
        generation_rows.sort(
            key=lambda row: (
                int(row["problem_index"]),
                int(row["attempt_index"]),
            )
        )
        for generation_row in generation_rows:
            key = (
                str(generation_row["problem_id"]),
                int(generation_row["attempt_index"]),
            )
            if key in seen:
                if receipt_generation_hashes[key] != generation_row["generation_payload_sha256"]:
                    raise ValueError(f"receipt/generation provenance mismatch: {key}")
                continue
            task = _task_from_generation(generation_row)
            rejected_reason = generation_row.get("rejected_reason")
            if rejected_reason:
                result = make_verification_result(
                    task,
                    worker_id=None,
                    status="rejected",
                    success=False,
                    diagnostics=str(rejected_reason),
                    verifier_backend="precheck",
                    verification_seconds=0.0,
                    rejected_reason=str(rejected_reason),
                    compile_messages=(str(rejected_reason),),
                    compile_errors=(str(rejected_reason),),
                )
                record(result.to_json())
            else:
                batch.append(task)
                if len(batch) >= args.verify_batch_size:
                    flush_batch()
        flush_batch()
    finally:
        warmups = list(pool.warmup_reports)
        fatal_errors.extend(item for item in pool.fatal_errors if item not in fatal_errors)
        recovered_failures.extend(
            item for item in pool.recovered_worker_failures if item not in recovered_failures
        )
        pool.close()
        receipts_handle.close()
        checkpoint_lock_handle.close()
    if fatal_errors:
        raise RuntimeError(f"verification failed closed: {fatal_errors}")
    if len(seen) != expected_attempts:
        raise ValueError(f"expected {expected_attempts} receipts, got {len(seen)}")

    pass_k = int(frozen["attempts_per_problem"])
    problem_rows = []
    histogram: Counter[str] = Counter()
    successful_problems = 0
    for problem_id, state in sorted(stats.items(), key=lambda item: item[1]["problem_index"]):
        attempts = set(state["attempts"])
        if attempts != set(range(pass_k)):
            raise ValueError(f"problem {problem_id} does not have exact pass@{pass_k}")
        successes = len(state["successful_attempts"])
        success = successes > 0
        successful_problems += int(success)
        histogram[f"{successes}/{pass_k}"] += 1
        problem_rows.append(
            {
                "problem_id": problem_id,
                "problem_index": state["problem_index"],
                "num_attempts_recorded": pass_k,
                "successful_attempts": successes,
                "success": success,
            }
        )
    if len(problem_rows) != int(frozen["problem_count"]):
        raise ValueError("problem result count mismatch")
    atomic_write_text(
        output / "problem_results.jsonl",
        "".join(canonical_json(row) + "\n" for row in problem_rows),
    )
    summary = {
        "schema": "grpo_two_phase_verification_summary_v1",
        "status": "complete",
        "frozen_shards_sha256": args.frozen_shards_sha256,
        "model_sums_sha256": args.model_sums_sha256,
        "problem_count": len(problem_rows),
        "attempt_count": len(seen),
        "successful_problem_count": successful_problems,
        f"pass@{pass_k}": successful_problems / len(problem_rows),
        "successful_attempt_count": result_counts["success"],
        "failed_attempt_count": result_counts["failure"],
        "timeout_count": result_counts["timeout"],
        "problem_attempt_success_histogram": dict(sorted(histogram.items())),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "warmup_reports": warmups,
        "fatal_errors": fatal_errors,
        "recovered_worker_failures": recovered_failures,
        "worker_count": 4,
        "timeout_seconds": args.timeout,
        "receipt_origin": ORIGIN,
        "execution_domain": DOMAIN,
        "receipts_sha256": sha256_file(receipts_path),
        "problem_results_sha256": sha256_file(output / "problem_results.jsonl"),
        "elapsed_seconds": round(time.monotonic() - started, 4),
    }
    if verification_seconds:
        ordered = sorted(verification_seconds)
        summary["verification_seconds_mean"] = sum(ordered) / len(ordered)
        middle = len(ordered) // 2
        summary["verification_seconds_median"] = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
    atomic_write_json(output / "summary.json", summary)
    return summary


def run_self_test() -> dict[str, Any]:
    assert [len(shard) for shard in plan_shards(0, 4)] == [0, 0, 0, 0]
    shards = plan_shards(9, 4)
    assert [len(shard) for shard in shards] == [3, 2, 2, 2]
    assert set().union(*(set(shard) for shard in shards)) == set(range(9))
    for left in range(4):
        for right in range(left + 1, 4):
            assert set(shards[left]).isdisjoint(shards[right])
    queue_items = list(range(13))
    queue_windows = [list(window) for window in generation_queue_windows(queue_items, 5)]
    assert [len(window) for window in queue_windows] == [5, 5, 3]
    assert [item for window in queue_windows for item in window] == queue_items
    try:
        list(generation_queue_windows(queue_items, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("zero queue depth must fail closed")
    with tempfile.TemporaryDirectory(prefix="grpo-two-phase-test-") as temporary:
        class ToyTokenizer:
            def __call__(self, prompt: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
                assert add_special_tokens is True
                return {"input_ids": list(range(len(prompt.split()) + 1))}

        root = Path(temporary)
        data = root / "data.jsonl"
        manifest = root / "manifest.jsonl"
        audit = root / "audit.json"
        toy_rows = [
            {
                "id": f"toy-{index}",
                "prompt": f"import Mathlib\n\ntheorem toy_{index} : True := by sorry\n",
                "lean_statement": f"theorem toy_{index} : True",
                "statement_hash": hashlib.sha256(
                    f"theorem toy_{index} : True".encode("utf-8")
                ).hexdigest(),
                "imports": ["Mathlib"],
            }
            for index in range(2)
        ]
        atomic_write_text(data, "".join(canonical_json(row) + "\n" for row in toy_rows))
        atomic_write_text(
            manifest,
            "".join(
                canonical_json(
                    {
                        "record_id": row["id"],
                        "statement_hash": row["statement_hash"],
                        "lean_statement": row["lean_statement"],
                    }
                )
                + "\n"
                for row in toy_rows
            ),
        )
        atomic_write_json(
            audit,
            {
                "status": "PASS",
                "final_records": 2,
                "artifacts": {
                    "data_sha256": sha256_file(data),
                    "manifest_sha256": sha256_file(manifest),
                },
                "full_audit": {
                    "manifest_data_projection_mismatches": 0,
                    "prompt_statement_semantic_mismatches": 0,
                    "noncanonical_prompts": 0,
                    "invalid_imports": 0,
                    "proof_bearing_field_leaks": 0,
                },
            },
        )
        frozen = freeze_shards(
            SimpleNamespace(
                data=str(data),
                data_sha256=sha256_file(data),
                manifest=str(manifest),
                manifest_sha256=sha256_file(manifest),
                audit=str(audit),
                audit_sha256=sha256_file(audit),
                output=str(root / "frozen"),
                shards=4,
                pass_k=4,
                expected_count=2,
                tokenizer_path="self-test-tokenizer",
                tokenizer_sums_sha256="0" * 64,
                max_new_tokens=2048,
                max_model_len=4096,
                _tokenizer_for_test=ToyTokenizer(),
            )
        )
        assert [item["count"] for item in frozen["shards"]] == [1, 1, 0, 0]
        manifest_row = next(
            iter_jsonl(root / "frozen" / frozen["shards"][0]["path"])
        )
        toy_record = read_rollout_records(data, rollout_kind="grpo")[0]
        chunk = root / "chunk.jsonl"
        chunk_rows = []
        for attempt in range(4):
            chunk_row = {
                "problem_id": manifest_row["problem_id"],
                "problem_index": manifest_row["global_index"],
                "attempt_index": attempt,
                "prompt": toy_record.prompt,
                "statement_hash": toy_record.statement_hash,
                "prompt_tokens": manifest_row["prompt_tokens"],
                "effective_max_new_tokens": manifest_row[
                    "effective_max_new_tokens"
                ],
                "model_sums_sha256": "0" * 64,
                "generation_backend": "vllm",
                "precision": "bfloat16",
                "vllm_version": "self-test",
                "bitsandbytes_version": "self-test",
                "generation_seed": 20260711 + int(manifest_row["global_index"]),
                "receipt_origin": ORIGIN,
                "execution_domain": DOMAIN,
            }
            chunk_row["generation_payload_sha256"] = generation_payload_sha256(chunk_row)
            chunk_rows.append(chunk_row)
        atomic_write_text(chunk, "".join(canonical_json(row) + "\n" for row in chunk_rows))
        assert len(
            _validate_chunk(
                chunk,
                manifest_row,
                pass_k=4,
                model_sums_sha256="0" * 64,
                expected_vllm_version="self-test",
                expected_bitsandbytes_version="self-test",
            )
        ) == 4
        generation_root = root / "generations"
        toy_records = read_rollout_records(data, rollout_kind="grpo")
        for shard in frozen["shards"]:
            shard_manifest = root / "frozen" / shard["path"]
            shard_rows = list(iter_jsonl(shard_manifest))
            shard_root = generation_root / f"shard_{shard['shard_index']}"
            shard_chunks = shard_root / "chunks"
            shard_chunks.mkdir(parents=True, exist_ok=True)
            for row in shard_rows:
                record = toy_records[int(row["global_index"])]
                rows = []
                for attempt in range(4):
                    generated = {
                        "problem_id": row["problem_id"],
                        "problem_index": row["global_index"],
                        "attempt_index": attempt,
                        "prompt": record.prompt,
                        "statement_hash": record.statement_hash,
                        "prompt_tokens": row["prompt_tokens"],
                        "effective_max_new_tokens": row[
                            "effective_max_new_tokens"
                        ],
                        "model_sums_sha256": "0" * 64,
                        "generation_backend": "vllm",
                        "precision": "bfloat16",
                        "vllm_version": "self-test",
                        "bitsandbytes_version": "self-test",
                        "generation_seed": 20260711 + int(row["global_index"]),
                        "receipt_origin": ORIGIN,
                        "execution_domain": DOMAIN,
                    }
                    generated["generation_payload_sha256"] = generation_payload_sha256(
                        generated
                    )
                    rows.append(generated)
                atomic_write_text(
                    shard_chunks / f"problem_{int(row['global_index']):06d}.jsonl",
                    "".join(canonical_json(item) + "\n" for item in rows),
                )
            atomic_write_json(
                shard_root / "GENERATION_COMPLETE.json",
                {
                    "status": "complete",
                    "shard_manifest_sha256": shard["sha256"],
                    "data_sha256": frozen["data_sha256"],
                    "model_sums_sha256": "0" * 64,
                    "problem_count": shard["count"],
                    "attempt_count": shard["count"] * 4,
                    "pass_k": 4,
                    "generation_backend": "vllm",
                    "generation_batch_size": 128,
                    "runtime_versions": {
                        "vllm": "self-test",
                        "bitsandbytes": "self-test",
                    },
                    "max_new_tokens": frozen["max_new_tokens_ceiling"],
                    "model_max_length": frozen["max_model_len"],
                    "chunks_sha256": chunk_set_sha256(shard_chunks),
                    "receipt_origin": ORIGIN,
                    "execution_domain": DOMAIN,
                },
            )
        frozen_path = root / "frozen" / "SHARDS_FROZEN.json"
        loaded_generations = list(
            _generation_rows(
                frozen,
                frozen_path.parent,
                generation_root,
                pass_k=4,
                model_sums_sha256="0" * 64,
            )
        )
        assert len(loaded_generations) == 8
        assert len(
            {
                (row["problem_id"], row["attempt_index"])
                for row in loaded_generations
            }
        ) == 8
        generation_audit = audit_generations(
            SimpleNamespace(
                frozen_shards=str(frozen_path),
                frozen_shards_sha256=sha256_file(frozen_path),
                generation_root=str(generation_root),
                model_sums_sha256="0" * 64,
                output=str(root / "GENERATION_AUDIT.json"),
            )
        )
        assert generation_audit["status"] == "PASS"
        assert generation_audit["problem_count"] == 2
        assert generation_audit["attempt_count"] == 8
        receipt = dict(loaded_generations[0])
        receipt["success"] = False
        receipt_path = root / "receipts.jsonl"
        atomic_write_text(receipt_path, canonical_json(receipt) + "\n")
        receipt_keys, _, receipt_hashes = _load_receipts(receipt_path)
        receipt_key = (str(receipt["problem_id"]), int(receipt["attempt_index"]))
        assert receipt_keys == {receipt_key}
        assert receipt_hashes[receipt_key] == receipt["generation_payload_sha256"]
    return {
        "status": "pass",
        "zero_record_shard_test": True,
        "nine_record_disjoint_shard_test": True,
        "continuous_queue_window_test": True,
        "two_record_freeze_hash_test": True,
        "four_attempt_atomic_chunk_resume_test": True,
        "phase_b_global_generation_key_test": True,
        "generation_audit_test": True,
        "receipt_generation_provenance_test": True,
        "origin": ORIGIN,
        "domain": DOMAIN,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-shards")
    freeze.add_argument("--data", required=True)
    freeze.add_argument("--data-sha256", required=True)
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--manifest-sha256", required=True)
    freeze.add_argument("--audit", required=True)
    freeze.add_argument("--audit-sha256", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--shards", type=int, default=4)
    freeze.add_argument("--pass-k", type=int, default=4)
    freeze.add_argument("--expected-count", type=int)
    freeze.add_argument("--tokenizer-path", required=True)
    freeze.add_argument("--tokenizer-sums-sha256", required=True)
    freeze.add_argument("--max-new-tokens", type=int, default=2048)
    freeze.add_argument("--max-model-len", type=int, default=4096)
    freeze.add_argument("--generation-batch-size", type=int, default=128)
    freeze.add_argument("--seed", type=int, default=20260711)
    freeze.add_argument("--temperature", type=float, default=1.0)
    freeze.add_argument("--top-p", type=float, default=0.95)

    generate = commands.add_parser("generate-shard")
    generate.add_argument("--data", required=True)
    generate.add_argument("--data-sha256", required=True)
    generate.add_argument("--shard-manifest", required=True)
    generate.add_argument("--shard-sha256", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--tokenizer-path", required=True)
    generate.add_argument("--model-sums", required=True)
    generate.add_argument("--model-sums-sha256", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--pass-k", type=int, default=4)
    generate.add_argument(
        "--generation-backend",
        choices=("vllm",),
        default="vllm",
    )
    generate.add_argument("--generation-batch-size", type=int, default=128)
    generate.add_argument(
        "--generation-queue-depth",
        type=int,
        default=None,
        help=(
            "Number of problems submitted to vLLM's waiting queue at once; "
            "defaults to generation-batch-size for backward compatibility."
        ),
    )
    generate.add_argument("--max-new-tokens", type=int, required=True)
    generate.add_argument("--model-max-length", type=int, default=4096)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    generate.add_argument("--expected-vllm-version", default=None)
    generate.add_argument("--expected-bitsandbytes-version", default=None)
    generate.add_argument("--bitsandbytes-install-marker", default=None)
    generate.add_argument(
        "--expected-bitsandbytes-install-marker-sha256", default=None
    )
    generate.add_argument("--expected-bnb-loader-sha256", default=None)
    generate.add_argument("--expected-bnb-linear-sha256", default=None)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int, default=20260711)

    verify = commands.add_parser("verify")
    verify.add_argument("--frozen-shards", required=True)
    verify.add_argument("--frozen-shards-sha256", required=True)
    verify.add_argument("--generation-root", required=True)
    verify.add_argument("--model-sums-sha256", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--lean-project", required=True)
    verify.add_argument("--task-spool-dir", required=True)
    verify.add_argument("--timeout", type=int, default=180)
    verify.add_argument("--warmup-timeout", type=int, default=1200)
    verify.add_argument("--queue-maxsize", type=int, default=256)
    verify.add_argument("--verify-batch-size", type=int, default=128)

    audit = commands.add_parser("audit-generations")
    audit.add_argument("--frozen-shards", required=True)
    audit.add_argument("--frozen-shards-sha256", required=True)
    audit.add_argument("--generation-root", required=True)
    audit.add_argument("--model-sums-sha256", required=True)
    audit.add_argument("--output", required=True)

    commands.add_parser("self-test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "freeze-shards":
        result = freeze_shards(args)
    elif args.command == "generate-shard":
        result = generate_shard(args)
    elif args.command == "verify":
        result = verify_generations(args)
    elif args.command == "audit-generations":
        result = audit_generations(args)
    else:
        result = run_self_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
