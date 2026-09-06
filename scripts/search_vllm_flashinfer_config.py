from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = (
    "flashinfer",
    "FLASHINFER",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "Using FlashInfer",
    "top-p & top-k",
    "topk_topp",
    "SAMPLER",
    "sampler",
)

ENVIRONMENT_KEYS = (
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_ATTENTION_BACKEND",
    "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
    "NVCC_PREPEND_FLAGS",
)


def distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def module_info(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"installed": False, "path": None, "package_path": None}
    package_path = None
    if spec.origin and spec.origin != "namespace":
        package_path = str(Path(spec.origin).resolve().parent)
    elif spec.submodule_search_locations:
        package_path = str(Path(next(iter(spec.submodule_search_locations))).resolve())
    return {
        "installed": True,
        "path": spec.origin,
        "package_path": package_path,
    }


def source_matches(
    roots: list[Path],
    patterns: tuple[str, ...],
    max_matches: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if any(pattern in line for pattern in patterns):
                    matches.append(
                        {
                            "path": str(path),
                            "line": line_no,
                            "text": line.strip()[:260],
                        }
                    )
                    if len(matches) >= max_matches:
                        return matches
    return matches


def vllm_flashinfer_detection(vllm: dict[str, Any], flashinfer: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    ok = bool(vllm["installed"] and flashinfer["installed"])
    if not vllm["installed"]:
        problems.append("vllm module is not importable")
    if not flashinfer["installed"]:
        problems.append("flashinfer module is not importable")

    env_value = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    if env_value is not None and env_value.strip().lower() in {"0", "false", "no"}:
        ok = False
        problems.append("VLLM_USE_FLASHINFER_SAMPLER disables FlashInfer sampler")

    return {
        "ok": ok,
        "problems": problems,
        "flashinfer_sampler_env": env_value,
        "expected_runtime_signal": "Using FlashInfer for top-p & top-k sampling",
    }


def flashinfer_sampler_smoke(
    model: str,
    timeout: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> dict[str, Any]:
    code = (
        "import os\n"
        "os.environ.pop('VLLM_USE_FLASHINFER_SAMPLER', None)\n"
        "os.environ.pop('CCCL_DISABLE_CTK_COMPATIBILITY_CHECK', None)\n"
        "os.environ.pop('NVCC_PREPEND_FLAGS', None)\n"
        "from vllm import LLM, SamplingParams\n"
        f"llm = LLM(model={model!r}, trust_remote_code=True, "
        f"gpu_memory_utilization={gpu_memory_utilization!r}, "
        f"max_model_len={max_model_len!r})\n"
        "outs = llm.generate(['Complete this Lean proof:\\nexample : True := by\\n'], "
        "SamplingParams(max_tokens=8, temperature=0.0))\n"
        "print(outs[0].outputs[0].text)\n"
    )
    env = os.environ.copy()
    env.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
    env.pop("CCCL_DISABLE_CTK_COMPATIBILITY_CHECK", None)
    env.pop("NVCC_PREPEND_FLAGS", None)
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            env=env,
        )
        output = completed.stdout
        signal = "Using FlashInfer for top-p & top-k sampling"
        return {
            "ok": completed.returncode == 0 and signal in output,
            "returncode": completed.returncode,
            "seconds": round(time.monotonic() - start, 4),
            "detected_flashinfer_sampler": signal in output,
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        return {
            "ok": False,
            "returncode": None,
            "seconds": round(time.monotonic() - start, 4),
            "detected_flashinfer_sampler": False,
            "output_tail": output[-4000:] + "\nTIMEOUT",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect how the active Python environment exposes vLLM and FlashInfer."
    )
    parser.add_argument("--max_matches", type=int, default=120)
    parser.add_argument("--pattern", action="append", default=None)
    parser.add_argument("--run_smoke", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    vllm = module_info("vllm")
    flashinfer = module_info("flashinfer")
    roots = [
        Path(info["package_path"])
        for info in (vllm, flashinfer)
        if info.get("package_path")
    ]
    report: dict[str, Any] = {
        "python": {"executable": sys.executable, "version": sys.version},
        "versions": {
            "vllm": distribution_version("vllm"),
            "flashinfer_python": distribution_version("flashinfer-python"),
            "flashinfer_cubin": distribution_version("flashinfer-cubin"),
            "torch": distribution_version("torch"),
            "cuda_tile": distribution_version("cuda-tile"),
        },
        "paths": {
            "vllm": vllm,
            "flashinfer": flashinfer,
        },
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "vllm_flashinfer_detection": vllm_flashinfer_detection(vllm, flashinfer),
        "source_search": {
            "patterns": patterns,
            "matches": source_matches(roots, patterns, args.max_matches),
        },
        "smoke_test": {"skipped": True},
    }
    if args.run_smoke:
        report["smoke_test"] = flashinfer_sampler_smoke(
            model=args.model,
            timeout=args.timeout,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
