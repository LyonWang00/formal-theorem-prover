import importlib.metadata as metadata

PACKAGES = [
    "torch",
    "vllm",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "datasets",
    "accelerate",
    "pantograph",
]


def main() -> None:
    for package in PACKAGES:
        print(f"{package}=={metadata.version(package)}")


if __name__ == "__main__":
    main()
