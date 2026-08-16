"""Verified EI discovery-attempt proof-repair utilities."""

from .client import DeepSeekRepairClient, RepairClientConfig, RepairClientError
from .environment import load_environment_contract
from .parser import RepairParseError, normalize_repaired_proof, parse_repair_response, parse_repaired_proof
from .pipeline import DiscoveryAttemptRepairPipeline, RepairPipeline, RepairRunResult
from .prompting import REPAIR_SYSTEM_PROMPT, build_repair_prompt, prompt_sha256
from .schema import AggregatedAttempt, APICallAttempt, EnvironmentContract, RepairClientResponse, RepairData, RepairProposal, RepairRequest

__all__ = [
    "DeepSeekRepairClient",
    "DiscoveryAttemptRepairPipeline",
    "AggregatedAttempt",
    "APICallAttempt",
    "EnvironmentContract",
    "REPAIR_SYSTEM_PROMPT",
    "RepairClientConfig",
    "RepairClientError",
    "RepairData",
    "RepairClientResponse",
    "RepairProposal",
    "RepairParseError",
    "RepairPipeline",
    "RepairRequest",
    "RepairRunResult",
    "build_repair_prompt",
    "load_environment_contract",
    "normalize_repaired_proof",
    "parse_repair_response",
    "parse_repaired_proof",
    "prompt_sha256",
]
