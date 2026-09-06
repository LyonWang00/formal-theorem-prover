from __future__ import annotations

import importlib

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}")
        print(f"capability={torch.cuda.get_device_capability(0)}")

    for name in ("vllm", "transformers", "peft", "trl", "bitsandbytes", "datasets"):
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            print(f"ok {name} {version}")
        except Exception as exc:
            print(f"bad {name} {exc!r}")


if __name__ == "__main__":
    main()

