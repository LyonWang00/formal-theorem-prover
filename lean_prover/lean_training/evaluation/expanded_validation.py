"""Contracts and statistics for the B2 expanded validation experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from math import comb
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable


SEED_DERIVATION_VERSION = "isolated_shard_batch_offset_v1"
PRIMARY_SEED = 20260730
REPLICATION_SEED = 20260803


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_statement(statement: str) -> str:
    without_block = re.sub(r"/-.*?-/", " ", statement, flags=re.DOTALL)
    without_line = re.sub(r"--[^\n]*", " ", without_block)
    return re.sub(r"\s+", " ", without_line).strip()


def normalized_statement_hash(row: dict[str, Any]) -> str:
    statement = str(row.get("lean_statement") or row.get("statement") or "")
    return sha256(normalized_statement(statement).encode("utf-8")).hexdigest()


def statement_id(row: dict[str, Any]) -> str:
    value = row.get("statement_id") or row.get("id") or row.get("record_id")
    if not value:
        raise ValueError("row has no statement identity")
    return str(value)


def stable_full500_ids(generation_rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(str(row["statement_id"]) for row in generation_rows)
    if len(counts) != 500 or set(counts.values()) != {4}:
        raise ValueError(
            f"first-round generations must contain 500 statements x 4; got "
            f"{len(counts)} statements and counts={sorted(set(counts.values()))}"
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for row in generation_rows:
        key = str(row["statement_id"])
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def subset_in_order(rows: list[dict[str, Any]], ordered_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {statement_id(row): row for row in rows}
    missing = [key for key in ordered_ids if key not in by_id]
    if missing:
        raise ValueError(f"dataset is missing {len(missing)} statement IDs: {missing[:3]}")
    return [by_id[key] for key in ordered_ids]


def lean_statement_from_prompt(prompt: str) -> str:
    marker = "### Lean statement\n"
    proof_marker = "\n\n### Lean proof"
    if marker not in prompt or proof_marker not in prompt:
        raise ValueError("generation prompt lacks fixed Lean statement/proof sections")
    return prompt.split(marker, 1)[1].split(proof_marker, 1)[0].strip()


def restore_full500_rows(
    discovery_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map legacy first-round IDs onto rebuilt verified_v2 rows by normalized statement."""

    ordered_legacy_ids = stable_full500_ids(generation_rows)
    prompt_statement_by_id: dict[str, str] = {}
    for generation in generation_rows:
        legacy_id = str(generation["statement_id"])
        statement = lean_statement_from_prompt(str(generation["prompt"]))
        previous = prompt_statement_by_id.setdefault(legacy_id, statement)
        if normalized_statement(previous) != normalized_statement(statement):
            raise ValueError(f"legacy statement ID maps to multiple prompts: {legacy_id}")
    rebuilt_by_hash: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in discovery_rows:
        rebuilt_by_hash[normalized_statement_hash(row)].append(row)
    restored: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {}
    for legacy_id in ordered_legacy_ids:
        statement = prompt_statement_by_id[legacy_id]
        key = sha256(normalized_statement(statement).encode("utf-8")).hexdigest()
        matches = rebuilt_by_hash.get(key, [])
        if len(matches) != 1:
            raise ValueError(
                f"legacy statement {legacy_id} has {len(matches)} verified_v2 hash matches"
            )
        source = dict(matches[0])
        rebuilt_id = statement_id(source)
        source["verified_v2_statement_id"] = rebuilt_id
        source["legacy_round1_statement_id"] = legacy_id
        source["statement_id"] = legacy_id
        restored.append(source)
        mapping[legacy_id] = {
            "verified_v2_statement_id": rebuilt_id,
            "normalized_statement_hash": key,
        }
    if len({row["verified_v2_statement_id"] for row in restored}) != 500:
        raise ValueError("legacy-to-verified_v2 mapping is not one-to-one")
    return restored, mapping


def assert_disjoint(
    left_name: str,
    left: list[dict[str, Any]],
    right_name: str,
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    left_ids = {statement_id(row) for row in left}
    right_ids = {statement_id(row) for row in right}
    left_hashes = {normalized_statement_hash(row) for row in left}
    right_hashes = {normalized_statement_hash(row) for row in right}
    shared_ids = sorted(left_ids & right_ids)
    shared_hashes = sorted(left_hashes & right_hashes)
    return {
        "left": left_name,
        "right": right_name,
        "shared_statement_ids": shared_ids,
        "shared_normalized_statement_hashes": shared_hashes,
        "valid": not shared_ids and not shared_hashes,
    }


def proof_free_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    marker = "### Lean proof"
    if marker not in prompt:
        raise ValueError(f"prompt lacks Lean proof marker: {statement_id(row)}")
    prefix, suffix = prompt.split(marker, 1)
    if suffix.strip():
        raise ValueError(f"prompt contains proof text after marker: {statement_id(row)}")
    reference = str(row.get("reference_proof") or row.get("proof") or "").strip()
    if reference and reference in prompt:
        raise ValueError(f"reference proof leaked into prompt: {statement_id(row)}")
    return prefix + marker + "\n"


def original_success_counts(
    generation_rows: list[dict[str, Any]], verification_rows: list[dict[str, Any]]
) -> dict[str, int]:
    generation_by_id = {str(row["generation_id"]): row for row in generation_rows}
    counts: defaultdict[str, int] = defaultdict(int)
    attempts: Counter[str] = Counter()
    for verification in verification_rows:
        generation = generation_by_id[str(verification["generation_id"])]
        key = str(generation["statement_id"])
        attempts[key] += 1
        counts[key] += int(bool(verification.get("verified")))
    if len(attempts) != 500 or set(attempts.values()) != {4}:
        raise ValueError("original M0 verification does not cover 500 x 4")
    return dict(counts)


def derived_seed(global_seed: int, *, local_batch_offset: int, iteration: int = -10) -> int:
    return global_seed + iteration * 1_000_000 + local_batch_offset


def seed_map(
    rows: list[dict[str, Any]], global_seed: int, *, generator_batch_size: int
) -> dict[str, int]:
    return {
        statement_id(row): derived_seed(
            global_seed,
            local_batch_offset=(index // generator_batch_size) * generator_batch_size,
        )
        for index, row in enumerate(rows)
    }


def success_counts(attempts: list[dict[str, Any]]) -> dict[str, int]:
    by_id: defaultdict[str, int] = defaultdict(int)
    attempt_counts: Counter[str] = Counter()
    for row in attempts:
        key = str(row["problem_id"])
        attempt_counts[key] += 1
        by_id[key] += int(bool(row.get("success")))
    if attempt_counts and set(attempt_counts.values()) != {4}:
        raise ValueError(f"each statement needs four attempts: {set(attempt_counts.values())}")
    return dict(by_id)


def metric_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    counts = success_counts(attempts)
    first = {
        str(row["problem_id"]): bool(row.get("success"))
        for row in attempts
        if int(row["attempt_index"]) == 0
    }
    first_two: defaultdict[str, bool] = defaultdict(bool)
    for row in attempts:
        if int(row["attempt_index"]) < 2:
            first_two[str(row["problem_id"])] |= bool(row.get("success"))
    solved = sorted(key for key, value in counts.items() if value > 0)
    verified = sum(counts.values())
    total = max(1, len(attempts))
    statements = max(1, len(counts))
    return {
        "statement_count": len(counts),
        "candidate_count": len(attempts),
        "verified_candidate_count": verified,
        "candidate_success_rate": verified / total,
        "pass_at_1": sum(first.values()) / statements,
        "pass_at_2": sum(first_two.values()) / statements,
        "pass_at_4": len(solved) / statements,
        "solved_statement_count": len(solved),
        "solved_statement_ids": solved,
        "success_at_4_distribution": {
            f"{index}_of_4": sum(value == index for value in counts.values())
            for index in range(5)
        },
        "success_count_by_statement": counts,
    }


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    return min(
        1.0,
        2.0 * sum(comb(discordant, value) for value in range(lower + 1))
        / (2**discordant),
    )


def paired_analysis(
    m0_attempts: list[dict[str, Any]],
    b2_attempts: list[dict[str, Any]],
    *,
    bootstrap_seed: int = PRIMARY_SEED,
    resamples: int = 10_000,
) -> dict[str, Any]:
    m0_counts = success_counts(m0_attempts)
    b2_counts = success_counts(b2_attempts)
    ids = sorted(set(m0_counts) & set(b2_counts))
    if len(ids) != len(m0_counts) or len(ids) != len(b2_counts):
        raise ValueError("paired analyses require identical statement IDs")
    m0_first = {
        str(row["problem_id"]): bool(row.get("success"))
        for row in m0_attempts
        if int(row["attempt_index"]) == 0
    }
    b2_first = {
        str(row["problem_id"]): bool(row.get("success"))
        for row in b2_attempts
        if int(row["attempt_index"]) == 0
    }
    deltas4 = [int(b2_counts[key] > 0) - int(m0_counts[key] > 0) for key in ids]
    deltas1 = [int(b2_first[key]) - int(m0_first[key]) for key in ids]
    rng = random.Random(bootstrap_seed)
    samples4: list[float] = []
    samples1: list[float] = []
    for _ in range(resamples):
        selected = [rng.randrange(len(ids)) for _ in ids]
        samples4.append(sum(deltas4[index] for index in selected) / len(ids))
        samples1.append(sum(deltas1[index] for index in selected) / len(ids))
    samples4.sort()
    samples1.sort()
    low = int(0.025 * (resamples - 1))
    high = int(0.975 * (resamples - 1))
    both = sum(m0_counts[key] > 0 and b2_counts[key] > 0 for key in ids)
    m0_only = sum(m0_counts[key] > 0 and b2_counts[key] == 0 for key in ids)
    b2_only = sum(m0_counts[key] == 0 and b2_counts[key] > 0 for key in ids)
    return {
        "statement_count": len(ids),
        "both_solved": both,
        "m0_only_solved": m0_only,
        "b2_only_solved": b2_only,
        "neither_solved": len(ids) - both - m0_only - b2_only,
        "discordant_pair_count": m0_only + b2_only,
        "delta_pass_at_1": sum(deltas1) / len(ids),
        "delta_pass_at_4": sum(deltas4) / len(ids),
        "pass_at_1_bootstrap_95_ci": [samples1[low], samples1[high]],
        "pass_at_4_bootstrap_95_ci": [samples4[low], samples4[high]],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "mcnemar_exact_two_sided_p": exact_mcnemar(m0_only, b2_only),
    }


def primary_gate(
    *,
    strict_m0_solved: int,
    strict_b2_solved: int,
    nonexpert_m0_solved: int,
    nonexpert_b2_solved: int,
    monitor_m0_solved: int,
    monitor_b2_solved: int,
) -> dict[str, Any]:
    checks = {
        "strict_unseen_not_lower": strict_b2_solved >= strict_m0_solved,
        "full500_nonexpert_not_lower_by_more_than_one": (
            nonexpert_b2_solved >= nonexpert_m0_solved - 1
        ),
        "monitor_not_lower_by_more_than_one": monitor_b2_solved >= monitor_m0_solved - 1,
    }
    return {"checks": checks, "run_replication": all(checks.values())}


def benchmark_gate(
    *,
    strict_primary_m0: int,
    strict_primary_b2: int,
    strict_replication_m0: int,
    strict_replication_b2: int,
    monitor_primary_m0: int,
    monitor_primary_b2: int,
    monitor_replication_m0: int,
    monitor_replication_b2: int,
) -> dict[str, Any]:
    checks = {
        "strict_primary_not_lower": strict_primary_b2 >= strict_primary_m0,
        "strict_replication_not_lower": strict_replication_b2 >= strict_replication_m0,
        "monitor_primary_not_lower_by_more_than_one": monitor_primary_b2 >= monitor_primary_m0 - 1,
        "monitor_replication_not_lower_by_more_than_one": (
            monitor_replication_b2 >= monitor_replication_m0 - 1
        ),
    }
    return {"checks": checks, "run_benchmark": all(checks.values())}
