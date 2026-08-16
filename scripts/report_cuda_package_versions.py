import importlib.metadata as metadata

PACKAGES = [
    "cuda-toolkit",
    "cuda-tile",
    "nvidia-cuda-tileiras",
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-crt",
    "nvidia-cuda-cccl",
    "nvidia-nvvm",
    "nvidia-cuda-nvrtc",
    "flashinfer-python",
]


def main() -> None:
    for package in PACKAGES:
        try:
            dist = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            print(f"{package}: not installed")
            continue
        requires = dist.requires or []
        print(f"{package}=={dist.version}")
        for req in requires:
            if "nvidia" in req or "cuda" in req:
                print(f"  requires: {req}")


if __name__ == "__main__":
    main()
