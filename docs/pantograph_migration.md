# Pantograph migration

## Compatibility decision

The project uses the complete official PyPantograph `v0.3.15` release stack:

| Component | Pinned revision |
| --- | --- |
| PyPantograph | `ffa7f243824d2762825abddb1e9f6e939ede761f` |
| Pantograph submodule | `842c0fe6e76b0771cc7f7939604c7a0b90e17433` |
| Lean | `leanprover/lean4:v4.29.1` |
| Mathlib | `5e932f97dd25535344f80f9dd8da3aab83df0fe6` (`v4.29.1`) |

The PyPantograph gitlink points to exactly the Pantograph revision above, and
that revision's `lean-toolchain` is exactly `v4.29.1`. We deliberately do not
pair the `0.3.15` Python wrapper with a later unreleased Pantograph commit.

## Architecture

`InteractiveProofBackend` is the boundary used by proof search. It exposes
session startup, goal inspection, tactic execution, state disposal, tactic
path recovery, and close. Search nodes carry only backend-neutral metadata:
`state_id`, goals, parent, tactic, depth, and completion status.

`PantographBackend` keeps a map of live states. Applying several tactics to
the same parent produces independent Pantograph states. A failed tactic
returns an error transition without deleting the parent or terminating the
server. Pruned nodes are sent to Pantograph's `goal.delete` endpoint.

`LeanCompiler` is intentionally separate and performs final verification in a
fresh process. Pantograph is the sole interactive proof backend.

## Runtime layout

`LEAN_PROJECT_PATH` selects the Lean project used by Pantograph and
`LeanCompiler`. A native Linux/WSL checkout uses its repository-local
`lean_project/`. Docker uses its internal `/opt/lean-project`.

When the setup script detects a Windows-mounted `/mnt/*` checkout, it creates
a Linux-native runtime copy under the current user's data directory and prints
the resulting path. This compatibility path avoids slow cross-filesystem
Mathlib access without embedding a machine-specific location in Python code.

## Changed files

- `lean_prover/proof_backend.py`: backend-neutral state and transition API.
- `lean_prover/pantograph_backend.py`: branching Pantograph implementation.
- `lean_prover/backend_factory.py`: Pantograph-default backend selection.
- `lean_prover/proof_search.py`: backend-neutral bounded search entry point.
- `scripts/install_pantograph.sh`: reproducible pinned installation.
- `scripts/setup_pantograph_runtime.sh`: ext4 runtime synchronization/build.
- `scripts/pantograph_smoke_test.py`: real end-to-end branching demonstration.
- `tests/test_pantograph_backend.py`: isolated lifecycle/branching tests.
- `tests/test_pantograph_integration.py`: real Mathlib integration test.

## Best-first and beam search integration

Future queue entries should store `ProofState` plus a model/search score.
Expand a node by calling `apply_tactic` repeatedly on that same state. Enqueue
each `OPEN` child independently, stop on `PROVED`, and call `release_state`
for every pruned child and exhausted parent. `tactic_path` reconstructs the
complete proof from any successful node, so search algorithms do not need to
store backend-native objects.

## Known limitations

- First Mathlib startup currently takes about one minute and uses substantial
  memory; keep one Pantograph session alive across search expansions.
- PyPantograph `v0.3.15` emits a Python 3.12 event-loop deprecation warning.
  It does not affect the tested protocol and is not patched locally.
- Windows-mounted WSL paths do not support all POSIX metadata operations and
  are slow for Mathlib. Prefer a native Linux/WSL checkout or Docker.
