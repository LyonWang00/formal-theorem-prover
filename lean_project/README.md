# Lean project

This directory is the Lake project used by the Python theorem-proving code.

- `lean-toolchain` pins Lean `4.29.1`.
- `lakefile.lean` declares the mathlib dependency.
- `lake-manifest.json` locks mathlib and transitive dependency commits.
- `LeanProject/` contains repository-owned Lean source files.

Python code should point `LEAN_PROJECT_PATH` to this directory. If the variable is not set, `lean_prover.backends.BackendConfig.from_environment()` defaults to the repository-local `lean_project/`.

Minimal Pantograph-backed check:

```python
from lean_prover.backends import BackendConfig, create_backend

backend = create_backend(BackendConfig.from_environment())
try:
    root = backend.start("∀ n : Nat, n + 0 = n")
    result = backend.apply_tactic(root, "simp")
    print(result.status)
finally:
    backend.close()
```

For Linux/WSL setup, prefer:

```bash
bash scripts/setup_project_environment.sh
```

The setup script installs Python dependencies, configures CUDA paths for vLLM/FlashInfer, prepares mathlib cache, and runs environment preflight checks.
