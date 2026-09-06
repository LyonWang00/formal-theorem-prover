from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def configure_runtime_paths() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    env_root = Path(sys.executable).resolve().parents[1]
    py_bin = env_root / "bin"
    home = Path.home()
    site_packages = (
        env_root
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    cuda_home = Path(os.environ.get("CUDA_HOME", site_packages / "nvidia/cu13"))
    nvidia_libs: list[str] = []
    nvidia_root = site_packages / "nvidia"
    if nvidia_root.exists():
        nvidia_libs = [str(path) for path in nvidia_root.glob("*/*") if path.name == "lib"]
    if (cuda_home / "lib").exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        cuda_bin = cuda_home / "bin"
        libcudart = cuda_home / "lib/libcudart.so.13"
        libcudart_link = cuda_home / "lib/libcudart.so"
        if libcudart.exists() and not libcudart_link.exists():
            libcudart_link.symlink_to(libcudart.name)
        lib64 = cuda_home / "lib64"
        if not lib64.exists():
            lib64.symlink_to(cuda_home / "lib", target_is_directory=True)
        nvidia_libs.insert(0, str(cuda_home / "lib"))
    else:
        cuda_bin = cuda_home / "bin"
    nvidia_libs.append("/usr/lib/wsl/lib")
    path_entries = [
        str(py_bin),
        str(cuda_bin),
        str(home / ".elan/bin"),
        "/opt/elan/bin",
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
        "/usr/lib/wsl/lib",
    ]
    os.environ["PATH"] = ":".join(
        entry for entry in path_entries if entry and Path(entry).exists()
    )
    for name in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        existing = os.environ.get(name, "")
        os.environ[name] = ":".join(nvidia_libs + ([existing] if existing else []))
    os.environ.pop("CCCL_DISABLE_CTK_COMPATIBILITY_CHECK", None)
    os.environ.pop("NVCC_PREPEND_FLAGS", None)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(project_root / ".cache/huggingface"))


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "seconds": round(time.monotonic() - start, 4),
            "output": completed.stdout,
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "ok": False,
            "returncode": None,
            "seconds": round(time.monotonic() - start, 4),
            "output": (error.stdout or "") + "\nTIMEOUT",
            "cmd": cmd,
        }


def check_pip() -> dict[str, Any]:
    return run_command([sys.executable, "-m", "pip", "check"], timeout=180)


def check_cuda_sonames() -> dict[str, Any]:
    names = ["libcuda.so", "libcudart.so.13", "libcudnn.so.9", "libcusparseLt.so.0"]
    results: dict[str, Any] = {}
    ok = True
    search_dirs = [
        path
        for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH")
        for path in os.environ.get(var, "").split(":")
        if path
    ]
    for name in names:
        candidates = [Path(directory) / name for directory in search_dirs]
        load_target = next((str(path) for path in candidates if path.exists()), name)
        try:
            ctypes.CDLL(load_target)
            results[name] = {"ok": True, "path": load_target}
        except OSError as error:
            ok = False
            results[name] = {"ok": False, "path": load_target, "error": str(error)}
    return {"ok": ok, "libraries": results}


def _module_info(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"installed": False, "path": None}
    return {
        "installed": True,
        "path": spec.origin,
        "package_path": (
            str(Path(spec.origin).resolve().parent)
            if spec.origin and spec.origin != "namespace"
            else None
        ),
    }


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _cuda_major_from_version(version: str | None) -> int | None:
    if not version:
        return None
    head = version.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def check_cuda_torch_flashinfer_consistency() -> dict[str, Any]:
    packages = {
        "torch": _distribution_version("torch"),
        "vllm": _distribution_version("vllm"),
        "flashinfer_python": _distribution_version("flashinfer-python"),
        "flashinfer_cubin": _distribution_version("flashinfer-cubin"),
        "cuda_tile": _distribution_version("cuda-tile"),
        "nvidia_cuda_runtime_cu13": _distribution_version("nvidia-cuda-runtime-cu13"),
        "nvidia_cuda_nvcc_cu13": _distribution_version("nvidia-cuda-nvcc-cu13"),
        "nvidia_cuda_nvrtc_cu13": _distribution_version("nvidia-cuda-nvrtc-cu13"),
        "nvidia_cuda_runtime_cu12": _distribution_version("nvidia-cuda-runtime-cu12"),
        "nvidia_cuda_nvcc_cu12": _distribution_version("nvidia-cuda-nvcc-cu12"),
        "nvidia_cuda_nvrtc_cu12": _distribution_version("nvidia-cuda-nvrtc-cu12"),
    }
    modules = {
        "torch": _module_info("torch"),
        "vllm": _module_info("vllm"),
        "flashinfer": _module_info("flashinfer"),
    }
    torch_info: dict[str, Any] = {"import_ok": False}
    ok = True
    problems: list[str] = []
    try:
        import torch

        torch_cuda_version = torch.version.cuda
        torch_info = {
            "import_ok": True,
            "version": torch.__version__,
            "cuda_version": torch_cuda_version,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "bf16_supported": (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
        }
        if not torch_cuda_version:
            ok = False
            problems.append("torch was installed without CUDA support")
        if not torch.cuda.is_available():
            ok = False
            problems.append("torch.cuda.is_available() is false")
    except Exception as error:
        ok = False
        torch_info = {"import_ok": False, "error": repr(error)}
        problems.append("torch import failed")

    if not modules["vllm"]["installed"]:
        ok = False
        problems.append("vllm module is not importable")
    if not modules["flashinfer"]["installed"]:
        ok = False
        problems.append("flashinfer module is not importable")
    if packages["flashinfer_python"] is None:
        ok = False
        problems.append("flashinfer-python distribution is not installed")

    torch_cuda_major = _cuda_major_from_version(torch_info.get("cuda_version"))
    installed_cuda_package_majors: set[int] = set()
    for name in packages:
        if name.endswith("_cu13") and packages[name]:
            installed_cuda_package_majors.add(13)
        if name.endswith("_cu12") and packages[name]:
            installed_cuda_package_majors.add(12)
    if torch_cuda_major and installed_cuda_package_majors:
        mismatches = sorted(
            major for major in installed_cuda_package_majors if major != torch_cuda_major
        )
        if mismatches:
            ok = False
            problems.append(
                "installed NVIDIA CUDA wheel major versions differ from torch CUDA "
                f"major {torch_cuda_major}: {mismatches}"
            )

    nvcc = shutil.which("nvcc")
    nvcc_result = (
        run_command([nvcc, "--version"], timeout=30) if nvcc else {"ok": False, "output": "nvcc not found"}
    )
    return {
        "ok": ok,
        "problems": problems,
        "packages": packages,
        "modules": modules,
        "torch": torch_info,
        "nvcc": nvcc_result,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_HOME",
                "LD_LIBRARY_PATH",
                "LIBRARY_PATH",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_ATTENTION_BACKEND",
                "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
                "NVCC_PREPEND_FLAGS",
            )
        },
    }


def check_vllm_flashinfer_detection(max_matches: int = 40) -> dict[str, Any]:
    vllm_info = _module_info("vllm")
    flashinfer_info = _module_info("flashinfer")
    ok = bool(vllm_info["installed"] and flashinfer_info["installed"])
    problems: list[str] = []
    if not vllm_info["installed"]:
        problems.append("vllm module is not importable")
    if not flashinfer_info["installed"]:
        problems.append("flashinfer module is not importable")

    env_value = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    if env_value is not None and env_value.strip().lower() in {"0", "false", "no"}:
        ok = False
        problems.append("VLLM_USE_FLASHINFER_SAMPLER disables FlashInfer sampler")

    roots = [Path(vllm_info["package_path"])] if vllm_info.get("package_path") else []
    patterns = (
        "flashinfer",
        "FLASHINFER",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "Using FlashInfer",
        "top-p & top-k",
        "sampler",
    )
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
                            "text": line.strip()[:240],
                        }
                    )
                    if len(matches) >= max_matches:
                        break
            if len(matches) >= max_matches:
                break
    if ok and not matches:
        problems.append("no FlashInfer-related references were found in vLLM source")
    return {
        "ok": ok,
        "problems": problems,
        "vllm": vllm_info,
        "flashinfer": flashinfer_info,
        "versions": {
            "vllm": _distribution_version("vllm"),
            "flashinfer_python": _distribution_version("flashinfer-python"),
            "flashinfer_cubin": _distribution_version("flashinfer-cubin"),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_ATTENTION_BACKEND",
                "CUDA_HOME",
            )
        },
        "source_matches": matches,
    }


def check_flashinfer_sampler() -> dict[str, Any]:
    code = (
        "from vllm import LLM, SamplingParams\n"
        "llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', "
        "trust_remote_code=True, gpu_memory_utilization=0.65, max_model_len=1024)\n"
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
            timeout=420,
            check=False,
            env=env,
        )
        output = completed.stdout
        return {
            "ok": completed.returncode == 0
            and "Using FlashInfer for top-p & top-k sampling" in output,
            "returncode": completed.returncode,
            "seconds": round(time.monotonic() - start, 4),
            "output": output,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "ok": False,
            "returncode": None,
            "seconds": round(time.monotonic() - start, 4),
            "output": (error.stdout or "") + "\nTIMEOUT",
        }


def check_pantograph_warmup(project_path: Path, timeout: int) -> dict[str, Any]:
    start = time.monotonic()
    try:
        from lean_prover.lean_training.verification.pantograph import (
            PantographTheoremVerifier,
        )

        verifier = PantographTheoremVerifier(
            project_path,
            imports=("Mathlib",),
            timeout=timeout,
        )
        try:
            result = verifier.warmup(timeout=timeout)
        finally:
            verifier.close()
        return {
            "ok": result.success,
            "seconds": round(time.monotonic() - start, 4),
            "server_startup_seconds": verifier.server_startup_seconds,
            "diagnostics": result.diagnostics,
        }
    except Exception as error:
        return {
            "ok": False,
            "seconds": round(time.monotonic() - start, 4),
            "diagnostics": repr(error),
        }


def check_mathlib_import(project_path: Path, timeout: int) -> dict[str, Any]:
    lake = shutil.which("lake")
    if lake is None:
        return {"ok": False, "output": "lake not found on PATH"}
    with tempfile.TemporaryDirectory(prefix="lean_mathlib_import_") as tmp:
        source = Path(tmp) / "ImportSmoke.lean"
        source.write_text("import Mathlib\n\nexample : True := by\n  trivial\n", encoding="utf-8")
        return run_command([lake, "env", "lean", str(source)], cwd=project_path, timeout=timeout)


def check_wsl_memory(min_memory_gb: float, min_swap_gb: float) -> dict[str, Any]:
    if "microsoft" not in platform.release().lower():
        return {"ok": True, "skipped": True, "reason": "not running under WSL"}
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.strip().split()[0])
    mem_gb = values.get("MemTotal", 0) / 1024 / 1024
    swap_gb = values.get("SwapTotal", 0) / 1024 / 1024
    return {
        "ok": mem_gb >= min_memory_gb and swap_gb >= min_swap_gb,
        "memory_gb": round(mem_gb, 2),
        "swap_gb": round(swap_gb, 2),
        "min_memory_gb": min_memory_gb,
        "min_swap_gb": min_swap_gb,
    }


def main() -> None:
    configure_runtime_paths()
    parser = argparse.ArgumentParser(description="Preflight checks for Lean SFT workflows.")
    parser.add_argument("--lean_project_path", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--min_wsl_memory_gb", type=float, default=12.0)
    parser.add_argument("--min_wsl_swap_gb", type=float, default=8.0)
    parser.add_argument("--skip_flashinfer_smoke", action="store_true")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    project_path = Path(args.lean_project_path).expanduser().resolve()
    checks = {
        "python": {"executable": sys.executable, "version": sys.version},
        "pip_check": check_pip(),
        "cuda_sonames": check_cuda_sonames(),
        "cuda_torch_flashinfer_consistency": check_cuda_torch_flashinfer_consistency(),
        "vllm_flashinfer_detection": check_vllm_flashinfer_detection(),
        "wsl_memory_swap": check_wsl_memory(
            args.min_wsl_memory_gb,
            args.min_wsl_swap_gb,
        ),
        "pantograph_warmup": check_pantograph_warmup(project_path, args.timeout),
        "mathlib_import": check_mathlib_import(project_path, args.timeout),
    }
    if args.skip_flashinfer_smoke:
        checks["flashinfer_sampler_smoke"] = {"ok": True, "skipped": True}
    else:
        checks["flashinfer_sampler_smoke"] = check_flashinfer_sampler()

    ok = all(
        value.get("ok", False)
        for key, value in checks.items()
        if isinstance(value, dict) and key != "python"
    )
    report = {"ok": ok, "checks": checks}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
