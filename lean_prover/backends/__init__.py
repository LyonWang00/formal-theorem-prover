"""Shared Lean backend interfaces and Pantograph implementation."""

from .backend_factory import BackendConfig, create_backend
from .pantograph_backend import (
    PantographBackend,
    PantographOptions,
    PantographUnavailableError,
)
from .proof_backend import (
    InteractiveProofBackend,
    ProofState,
    TacticProposer,
    TacticTransition,
    TransitionStatus,
)

__all__ = [
    "BackendConfig",
    "InteractiveProofBackend",
    "PantographBackend",
    "PantographOptions",
    "PantographUnavailableError",
    "ProofState",
    "TacticProposer",
    "TacticTransition",
    "TransitionStatus",
    "create_backend",
]
