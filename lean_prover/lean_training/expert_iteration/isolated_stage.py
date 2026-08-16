"""GPU-stage subprocess entry point and coordinator-side runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .utils import append_jsonl, write_json_atomic


class IsolatedStageRunner:
    """Run GPU-heavy stages in close-fds subprocesses and persist lifecycle data."""

    def __init__(self, run_dir: str | Path, *, flashinfer_sampler: bool = True) -> None:
        self.runtime_dir = Path(run_dir) / "runtime"
        self.request_dir = self.runtime_dir / "isolated_requests"
        self.log_dir = self.runtime_dir / "isolated_logs"
        self.events_path = self.runtime_dir / "isolated_processes.jsonl"
        self.flashinfer_sampler = flashinfer_sampler
        self.active_process: subprocess.Popen | None = None

    def run(
        self,
        stage: str,
        payload: dict[str, Any],
        *,
        timeout: int,
        service_callback: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if self.active_process is not None:
            raise RuntimeError("another isolated GPU stage is already running")
        run_id = f"{stage}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        request_path = self.request_dir / f"{run_id}.request.json"
        result_path = self.request_dir / f"{run_id}.result.json"
        log_path = self.log_dir / f"{run_id}.log"
        write_json_atomic(request_path, {"stage": stage, "payload": payload})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        if stage == "training":
            environment["LEAN_TRAINING_SKIP_CUDA_ALLOCATOR_WARMUP"] = "1"
            environment.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True",
            )
            environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if stage == "generation" and self.flashinfer_sampler:
            environment["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
            expose_environment_cuda_toolkit(environment)
        command = [
            sys.executable,
            "-m",
            "lean_prover.lean_training.expert_iteration.isolated_stage",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        started_at = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                close_fds=True,
                start_new_session=os.name != "nt",
            )
            self.active_process = process
            self._record_event(
                {
                    "run_id": run_id,
                    "stage": stage,
                    "event": "started",
                    "pid": process.pid,
                    "close_fds": True,
                    "started_at": started_at,
                    "request_path": str(request_path),
                    "result_path": str(result_path),
                    "log_path": str(log_path),
                    "flashinfer_sampler": (
                        self.flashinfer_sampler if stage == "generation" else None
                    ),
                }
            )
            try:
                deadline = time.monotonic() + timeout
                while (return_code := process.poll()) is None:
                    if time.monotonic() >= deadline:
                        self._terminate_active()
                        self._record_event(
                            {
                                "run_id": run_id,
                                "stage": stage,
                                "event": "timeout",
                                "pid": process.pid,
                                "timeout_seconds": timeout,
                                "finished_at": time.time(),
                            }
                        )
                        raise TimeoutError(
                            f"isolated {stage} process exceeded {timeout}s; see {log_path}"
                        )
                    if service_callback is not None:
                        service_callback()
                    time.sleep(0.5)
            except BaseException:
                self._terminate_active()
                raise
            finally:
                self.active_process = None
        result = (
            json.loads(result_path.read_text(encoding="utf-8-sig"))
            if result_path.exists()
            else {}
        )
        self._record_event(
            {
                "run_id": run_id,
                "stage": stage,
                "event": "finished",
                "pid": process.pid,
                "return_code": return_code,
                "finished_at": time.time(),
                "duration_seconds": round(time.time() - started_at, 4),
            }
        )
        if return_code != 0 or not result.get("success"):
            error = result.get("error") or _tail(log_path, 80)
            raise RuntimeError(
                f"isolated {stage} process failed with code {return_code}: {error}"
            )
        return dict(result.get("result") or {})

    def close(self) -> None:
        self._terminate_active()

    def _terminate_active(self) -> None:
        process = self.active_process
        if process is None or process.poll() is not None:
            return
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=5)

    def _record_event(self, event: dict[str, Any]) -> None:
        append_jsonl(self.events_path, (event,))


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    stage = str(request["stage"])
    payload = dict(request["payload"])
    if stage == "generation":
        return _run_generation(payload)
    if stage == "training":
        return _run_training(payload)
    if stage == "probe":
        return {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "payload": payload,
            "open_file_descriptors": (
                len(list(Path("/proc/self/fd").iterdir()))
                if Path("/proc/self/fd").is_dir()
                else None
            ),
        }
    raise ValueError(f"unsupported isolated stage: {stage}")


def _run_generation(payload: dict[str, Any]) -> dict[str, Any]:
    from .config import ExpertIterationConfig
    from .discovery_generator import create_generator_backend, generate_candidates
    from .schemas import DiscoveryStatementState, StatementRecord

    config = ExpertIterationConfig.model_validate(payload["config"])
    selected = [StatementRecord.model_validate(row) for row in payload["selected"]]
    states = {
        key: DiscoveryStatementState.model_validate(value)
        for key, value in payload["statement_states"].items()
    }
    generation_config = config.discovery.generation
    generator = create_generator_backend(
        str(payload["base_model"]),
        payload.get("adapter_path"),
        generation_config,
    )
    generations = generate_candidates(
        selected,
        states,
        iteration=int(payload["iteration"]),
        checkpoint=str(payload["checkpoint"]),
        generator=generator,
        config=generation_config,
        sampling_budget=dict(payload.get("sampling_budget") or config.discovery.sampling_budget),
        zero_success_backoff_after_rounds=(
            config.discovery.max_consecutive_zero_success_rounds
        ),
        output_path=str(payload["output_path"]),
        seed=int(payload.get("generation_seed", config.seed)),
        force=bool(payload.get("force", False)),
    )
    return {
        "pid": os.getpid(),
        "candidates_generated": len(generations),
        "backend": generator.backend_name,
        "flashinfer_sampler_enabled": os.environ.get(
            "VLLM_USE_FLASHINFER_SAMPLER", "1"
        )
        == "1",
        "cuda_home": os.environ.get("CUDA_HOME"),
        "nvcc": os.environ.get("FLASHINFER_NVCC"),
        "base_model_path": str(payload["base_model"]),
        "adapter_path": payload.get("adapter_path"),
        "tokenizer_path": str(payload["base_model"]),
        "vllm_model_path": getattr(
            generator,
            "model_name_or_path",
            str(payload["base_model"]),
        ),
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
        "stop_token_ids": list(getattr(generator, "stop_token_ids", ()) or ()),
    }


def _run_training(payload: dict[str, Any]) -> dict[str, Any]:
    from .config import ExpertIterationConfig
    from .trainer_adapter import SFTTrainerAdapter

    config = ExpertIterationConfig.model_validate(payload["config"])
    metrics = SFTTrainerAdapter(config).train_iteration(**payload["arguments"])
    return {"pid": os.getpid(), "metrics": metrics}


def _tail(path: Path, count: int) -> str:
    if not path.exists():
        return "child produced no result or log"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def expose_environment_cuda_toolkit(environment: dict[str, str]) -> None:
    """Expose a version-matched CUDA toolkit already installed in this venv."""

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / version / "site-packages"
    candidates = [
        site_packages / "nvidia" / "cu13",
        Path("/usr/local/cuda"),
    ]
    cuda_home = next(
        (path for path in candidates if (path / "bin" / "nvcc").is_file()),
        None,
    )
    if cuda_home is None:
        return
    lib64 = cuda_home / "lib64"
    if not lib64.exists() and (cuda_home / "lib").is_dir():
        try:
            lib64.symlink_to("lib", target_is_directory=True)
        except FileExistsError:
            pass
    cudart_link = cuda_home / "lib" / "libcudart.so"
    if not cudart_link.exists():
        versioned_cudart = next(
            iter(sorted((cuda_home / "lib").glob("libcudart.so.*"))),
            None,
        )
        if versioned_cudart is not None:
            try:
                cudart_link.symlink_to(versioned_cudart.name)
            except FileExistsError:
                pass
    wsl_libcuda = Path("/usr/lib/wsl/lib/libcuda.so")
    if lib64.is_dir() and wsl_libcuda.is_file():
        stubs = lib64 / "stubs"
        stubs.mkdir(parents=True, exist_ok=True)
        stub = stubs / "libcuda.so"
        if not stub.exists():
            try:
                stub.symlink_to(wsl_libcuda)
            except FileExistsError:
                pass
    environment["CUDA_HOME"] = str(cuda_home)
    nvcc = _select_matching_nvcc(cuda_home)
    environment["FLASHINFER_NVCC"] = str(nvcc)
    environment["PATH"] = os.pathsep.join(
        [
            str(nvcc.parent),
            str(Path(sys.prefix) / "bin"),
            environment.get("PATH", ""),
        ]
    )
    library_dirs = [cuda_home / "lib", Path("/usr/lib/wsl/lib")]
    nvidia_root = site_packages / "nvidia"
    if nvidia_root.is_dir():
        library_dirs.extend(
            path for path in nvidia_root.glob("*/lib") if path.is_dir()
        )
    library_path = os.pathsep.join(str(path) for path in library_dirs if path.is_dir())
    for name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        existing = environment.get(name)
        environment[name] = (
            os.pathsep.join([library_path, existing]) if existing else library_path
        )


def _select_matching_nvcc(cuda_home: Path) -> Path:
    header = cuda_home / "include" / "cuda.h"
    header_match = re.search(
        r"^#define\s+CUDA_VERSION\s+(\d+)",
        header.read_text(encoding="utf-8", errors="ignore"),
        flags=re.MULTILINE,
    )
    header_version = None
    if header_match:
        encoded = int(header_match.group(1))
        header_version = (encoded // 1000, (encoded % 1000) // 10)
    candidates = [
        path / "bin" / "nvcc"
        for path in sorted(Path(sys.prefix).glob("flashinfer_cuda*/nvidia/cu*"))
    ]
    candidates.append(cuda_home / "bin" / "nvcc")
    for candidate in candidates:
        if not candidate.is_file() or not (
            candidate.parent.parent / "nvvm" / "bin" / "cicc"
        ).is_file():
            continue
        try:
            output = subprocess.run(
                [str(candidate), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        version_match = re.search(r"release\s+(\d+)\.(\d+)", output)
        if header_version is None or (
            version_match
            and (int(version_match.group(1)), int(version_match.group(2)))
            == header_version
        ):
            return candidate
    return cuda_home / "bin" / "nvcc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
    try:
        result = execute_request(request)
        write_json_atomic(args.result, {"success": True, "result": result})
    except BaseException as error:
        write_json_atomic(
            args.result,
            {
                "success": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
