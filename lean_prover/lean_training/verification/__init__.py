"""Shared Pantograph verification services for training and evaluation."""

from .pantograph import PantographCheckResult, PantographTaskVerifier, PantographTheoremVerifier
from .cache import VerificationCache
from .pool import VerificationPool, VerificationPoolConfig, VerificationPoolRun, run_verification_pool
from .schema import VerificationResult, VerificationTask, VerificationWarmupReport

__all__ = [
    "PantographCheckResult",
    "PantographTaskVerifier",
    "PantographTheoremVerifier",
    "VerificationPool",
    "VerificationPoolConfig",
    "VerificationPoolRun",
    "VerificationResult",
    "VerificationTask",
    "VerificationWarmupReport",
    "run_verification_pool",
    "VerificationCache",
]
