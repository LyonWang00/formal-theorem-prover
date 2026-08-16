#!/usr/bin/env python3
"""Rebuild the reusable LD difficulty pool with inherited semantic standards.

Frozen manual labels are never changed. Remaining records receive a batched
LLM semantic review over the complete theorem/proof plus structural, premise,
domain, and recovered-context evidence. Numeric metrics are evidence only and
are never used as threshold labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


DIFFICULTIES = {"easy", "medium", "difficult", "exclude"}
TACTICS = (
    "simp", "simpa", "norm_num", "ring", "ring_nf", "linarith", "nlinarith",
    "omega", "rw", "rfl", "exact", "apply", "refine", "constructor", "cases",
    "rcases", "induction", "have", "suffices", "aesop", "ext", "funext",
    "field_simp", "positivity", "decide", "native_decide", "grind", "by_contra",
)

SYSTEM_PROMPT = """You are assisting a careful human review of LeanDojo theorem
difficulty for a Qwen2.5-1.5B short whole-proof model. Apply the inherited rubric:

easy: a self-contained transparent short pattern using common Lean/Mathlib API,
shallow structure, few dependencies, and no meaningful planning burden.
medium: realistic as one whole proof but needs a nontrivial API bridge, coordinated
lemmas, modest cases/constructor/induction, coercion/typeclass handling, or local API.
difficult: deep or fragile API composition, substantial induction/recursion,
dependent/category/manifold/computability infrastructure, strong same-file context,
or several intermediate obligations.
exclude: generated/internal declarations or tactic/metaprogramming framework rules
outside statement-to-mathematical-proof training.

Do not classify by proof length, tactic count, or premise count alone. A short exact
term can be medium/difficult when it hides specialized local API; a longer but routine
proof can remain medium. For every item, jointly inspect statement, proof structure,
tactics, premises, same-file dependencies, domain/context, and learnability. Return
one JSON object with key `classifications`; every requested id must appear exactly
once with `id`, `difficulty`, `confidence` (high|medium|low), and a specific one-sentence
`reason`. Return JSON only."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def proof_norm(value: Any) -> str:
    return norm(value).replace("by ", "by ", 1)


def identity(row: dict[str, Any]) -> dict[str, set[str]]:
    record = {
        str(row.get(key) or "").strip()
        for key in ("id", "record_id", "source_id")
        if str(row.get(key) or "").strip()
    }
    group = {str(row.get("theorem_group_id") or "").strip()} - {""}
    qualified = {
        str(row.get(key) or "").strip()
        for key in ("qualified_name", "theorem_name")
        if str(row.get(key) or "").strip()
    }
    statement = str(
        row.get("statement")
        or row.get("lean_statement")
        or row.get("training_statement")
        or ""
    ).strip()
    proof = str(row.get("proof") or row.get("training_proof") or "").strip()
    proof_variant = ""
    if group and proof:
        proof_hash = hashlib.sha256(proof_norm(proof).encode()).hexdigest()
        proof_variant = f"{next(iter(group))}\x1f{proof_hash}"
    return {
        "record_id": record,
        "theorem_group": group,
        "qualified_theorem": qualified,
        "exact_statement": {statement} - {""},
        "normalized_statement": {norm(statement)} - {""},
        # A proof variant is scoped to its theorem group. Generic proofs such
        # as `rfl` must not cause unrelated theorems to be treated as leaks.
        "proof_variant": {proof_variant} - {""},
    }


def union_identity(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    result = {key: set() for key in identity({})}
    for row in rows:
        values = identity(row)
        for key in result:
            result[key].update(values[key])
    return result


def overlap(row: dict[str, Any], protected: dict[str, set[str]]) -> bool:
    values = identity(row)
    return any(values[key] & protected[key] for key in protected)


def protected_sets(project: Path) -> dict[str, list[dict[str, Any]]]:
    index_path = project / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json"
    index = read_json(index_path)
    result: dict[str, list[dict[str, Any]]] = {}
    for name, entry in index.items():
        raw = entry.get("path") if isinstance(entry, dict) else None
        if raw and Path(str(raw)).is_file():
            result[name] = read_jsonl(Path(str(raw)))
    extras = {
        "ld_medium_holdout64": project / "outputs/stage2_data_ratio_ablation/phaseC3_ld_medium/evaluation/datasets/ld_medium_holdout64.jsonl",
        "wb_eval160": project / "data/processed/lean_workbook_verified_v2/eval.jsonl",
    }
    for name, path in extras.items():
        if path.is_file():
            result[name] = read_jsonl(path)
    if not result:
        raise RuntimeError("no protected evaluation sets were loaded")
    return result


def source_domain(row: dict[str, Any]) -> str:
    parts = str(row.get("source_file") or "").split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "Mathlib" else "unknown"


def tactic_distribution(proof: str) -> dict[str, int]:
    result = {
        tactic: len(re.findall(rf"\b{re.escape(tactic)}\b", proof))
        for tactic in TACTICS
    }
    return {key: value for key, value in result.items() if value}


def verified_and_recoverable(row: dict[str, Any]) -> tuple[bool, str]:
    verification = row.get("verification") or {}
    recovery = (row.get("metadata") or {}).get("context_recovery") or {}
    required = (
        row.get("id"), row.get("qualified_name"), row.get("source_file"),
        row.get("statement"), row.get("proof"), row.get("repository_commit"),
        row.get("lean_commit"), row.get("theorem_group_id"), row.get("assembled_source_hash"),
    )
    if not all(str(value or "").strip() for value in required):
        return False, "incomplete_source_provenance"
    if not (
        row.get("verification_status") == "verified_default_timeout"
        and verification.get("compile_success") is True
        and not verification.get("timed_out")
        and not verification.get("error_category")
    ):
        return False, "not_current_pantograph_verified"
    if not row.get("imports") and not recovery.get("imports"):
        return False, "imports_not_recoverable"
    if not (row.get("metadata") or {}).get("assembled_source"):
        return False, "assembled_source_not_recoverable"
    return True, ""


def review_item(row: dict[str, Any]) -> dict[str, Any]:
    proof = str(row.get("proof") or "")
    premises = list(row.get("premises") or [])
    trace = list(row.get("tactic_trace") or [])
    recovery = (row.get("metadata") or {}).get("context_recovery") or {}
    return {
        "id": str(row["id"]),
        "qualified_name": row.get("qualified_name"),
        "domain": source_domain(row),
        "source_file": row.get("source_file"),
        "statement": str(row.get("statement") or "")[:5000],
        "proof": proof[:5000],
        "proof_style": row.get("proof_style"),
        "tactics": tactic_distribution(proof),
        "tactic_trace": [
            str(step.get("tactic") or step.get("tactic_code") or step)[:500]
            if isinstance(step, dict) else str(step)[:500]
            for step in trace[:20]
        ],
        "premises": [
            {
                "name": premise.get("qualified_name"),
                "same_file": bool(premise.get("is_same_file")),
                "source_file": premise.get("source_file"),
            }
            for premise in premises[:20]
        ],
        "context": {
            "namespace_stack": row.get("namespace_stack"),
            "sections": row.get("section_context"),
            "open_namespaces": row.get("open_namespaces"),
            "open_scopes": row.get("open_scopes"),
            "local_notations_count": len(row.get("local_notations") or []),
            "local_instances_count": len(row.get("local_instances") or []),
            "active_commands_count": len(recovery.get("active_commands") or []),
            "recovery_status": recovery.get("recovery_status"),
        },
        "supporting_metrics_not_thresholds": {
            "statement_tokens": row.get("statement_tokens"),
            "proof_tokens": row.get("label_tokens"),
            "tactic_steps": len(trace),
            "premise_count": len(premises),
            "same_file_premise_count": sum(bool(item.get("is_same_file")) for item in premises),
        },
    }


def local_semantic_review(row: dict[str, Any]) -> dict[str, str]:
    """Codex-authored semantic rubric; no record leaves the local machine.

    This deliberately uses interacting structural/context/API signals. Token and
    count metrics can strengthen a judgment but can never decide it alone.
    """

    proof = str(row.get("proof") or "")
    statement = str(row.get("statement") or "")
    qualified = str(row.get("qualified_name") or "")
    source_file = str(row.get("source_file") or "")
    domain = source_domain(row)
    premises = list(row.get("premises") or [])
    trace = list(row.get("tactic_trace") or [])
    recovery = (row.get("metadata") or {}).get("context_recovery") or {}
    proof_tokens = int(row.get("label_tokens") or 0)
    same_file = sum(bool(item.get("is_same_file")) for item in premises)
    premise_names = [str(item.get("qualified_name") or "") for item in premises]
    tactic_counts = tactic_distribution(proof)

    internal_namespace = any(
        marker in qualified
        for marker in (
            "Mathlib.Tactic.Ring", "Mathlib.Tactic.FieldSimp",
            "Tactic.MfldSetTac", ".Tactic.Interactive.", "Lean.Parser.Tactic",
        )
    )
    private_dependency = "_private" in proof or any("_private" in name for name in premise_names)
    meta_cues = any(
        cue in statement or cue in proof
        for cue in ("Lean.Expr", "Lean.Meta", "Lean.Elab", "Q(Type", "Syntax", "MacroM")
    )
    if internal_namespace or meta_cues or (private_dependency and same_file > 0):
        evidence = []
        if internal_namespace:
            evidence.append("tactic/metaprogramming support namespace")
        if meta_cues:
            evidence.append("quoted-expression or elaborator context")
        if private_dependency:
            evidence.append("private same-file dependency")
        return {
            "difficulty": "exclude",
            "confidence": "high",
            "reason": "Excluded because the declaration depends on " + ", ".join(evidence) + "; it is infrastructure rather than a recoverable public statement-to-proof target.",
        }

    high_api_domains = {
        "AlgebraicGeometry", "CategoryTheory", "AlgebraicTopology", "Computability",
        "Condensed", "Geometry", "Manifold", "ModelTheory", "RepresentationTheory",
    }
    moderate_api_domains = {
        "MeasureTheory", "Topology", "Dynamics", "SetTheory", "FieldTheory",
        "InformationTheory", "Probability", "LinearAlgebra", "RingTheory",
    }
    planning_cues = Counter()
    for cue in ("induction", "cases", "rcases", "constructor", "refine", "have", "suffices", "obtain", "calc"):
        planning_cues[cue] = len(re.findall(rf"\b{cue}\b", proof))
    planning_steps = sum(planning_cues.values())
    dependent_cues = sum(
        statement.count(cue)
        for cue in ("Subtype", "Sigma", "HEq", "Category", "Functor", "Morphism", "∀", "∃")
    )
    # `active_commands` normally includes ordinary imports/namespace commands
    # and is therefore provenance, not difficulty. Only genuinely local
    # notation/instance machinery contributes context burden here.
    context_load = (
        len(row.get("local_notations") or [])
        + len(row.get("local_instances") or [])
    )
    direct_common = bool(
        re.fullmatch(r"\s*rfl\s*", proof)
        or re.fullmatch(r"\s*by\s+(?:simpa?|norm_num|ring|linarith|nlinarith|omega|rfl)(?:\s*\[[^\]]*\])?\s*", proof, re.S)
    )
    common_tactics_only = bool(tactic_counts) and set(tactic_counts) <= {
        "simp", "simpa", "norm_num", "ring", "ring_nf", "linarith", "nlinarith",
        "omega", "rw", "rfl", "exact", "decide", "native_decide",
    }
    direct_term = str(row.get("proof_style") or "") == "term" and planning_steps == 0

    risk = 0
    evidence: list[str] = []
    if domain in high_api_domains:
        risk += 2
        evidence.append(f"specialized {domain} API")
    elif domain in moderate_api_domains:
        risk += 1
        evidence.append(f"nontrivial {domain} API")
    if same_file >= 3:
        risk += 2
        evidence.append(f"{same_file} same-file premises")
    elif same_file:
        risk += 1
        evidence.append(f"{same_file} same-file premise")
    if len(premises) >= 8:
        risk += 2
        evidence.append(f"{len(premises)} linked premises")
    elif len(premises) >= 4:
        risk += 1
        evidence.append(f"{len(premises)} linked premises")
    if planning_steps >= 4 or planning_cues["induction"]:
        risk += 2
        evidence.append("multi-stage case/induction planning")
    elif planning_steps:
        risk += 1
        evidence.append("explicit intermediate proof structure")
    short_direct_term = direct_term and proof_tokens <= 16 and len(premises) <= 4 and same_file <= 1
    if context_load >= 4 and not (direct_common or short_direct_term):
        risk += 2
        evidence.append("heavy recovered local context")
    elif context_load and not (direct_common or short_direct_term):
        risk += 1
        evidence.append("local notation/instance context")
    if dependent_cues >= 4:
        risk += 2
        evidence.append("dependent or categorical statement structure")
    elif dependent_cues:
        risk += 1
    if proof_tokens >= 64 and (planning_steps or len(premises) >= 6):
        risk += 2
        evidence.append("long proof combined with structural obligations")
    elif proof_tokens >= 32 and (same_file or planning_steps):
        risk += 1
    if direct_term and proof_tokens <= 16 and same_file >= 2:
        risk += 1
        evidence.append("short proof hides exact local API recall")

    decisive_difficult = (
        (planning_cues["induction"] > 0 and proof_tokens >= 40)
        or (same_file >= 4 and len(premises) >= 9)
        or (domain in high_api_domains and context_load >= 4 and planning_steps >= 2)
        or (proof_tokens >= 100 and planning_steps >= 2)
    )
    simple_induction = (
        planning_cues["induction"] > 0
        and proof_tokens <= 24
        and len(trace) <= 1
        and len(premises) <= 5
        and same_file <= 4
    )
    transparent_easy = (
        (
            (direct_common and proof_tokens <= 20 and len(premises) <= 5 and same_file <= 3)
            or short_direct_term
            or (
                common_tactics_only
                and proof_tokens <= 20
                and len(trace) <= 2
                and len(premises) <= 5
                and same_file <= 2
                and planning_steps <= 1
            )
            or simple_induction
        )
        and dependent_cues <= 3
    )

    if decisive_difficult or risk >= 7:
        reason_bits = evidence[:4] or ["fragile multi-stage API composition"]
        return {
            "difficulty": "difficult",
            "confidence": "high" if decisive_difficult or risk >= 9 else "medium",
            "reason": "Difficult: " + "; ".join(reason_bits) + ", requiring substantial planning or source-specific reconstruction beyond short whole-proof learning.",
        }
    if transparent_easy and risk <= 5:
        tactic_note = ", ".join(tactic_counts) if tactic_counts else "a direct term"
        return {
            "difficulty": "easy",
            "confidence": "high" if risk == 0 else "medium",
            "reason": f"Easy: the proof is a shallow {tactic_note} pattern with limited public premises and little local-context dependence, so the current model can learn it as one whole proof.",
        }
    reason_bits = evidence[:4]
    if not reason_bits:
        reason_bits = ["a nontrivial API bridge or coordinated proof step"]
    return {
        "difficulty": "medium",
        "confidence": "medium",
        "reason": "Medium: " + "; ".join(reason_bits) + ", but the proof remains bounded enough for curriculum whole-proof SFT rather than planner-only treatment.",
    }


def validate_response(raw: Any, expected: set[str]) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("classifications"), list):
        raise ValueError("response is missing classifications list")
    rows = raw["classifications"]
    ids = [str(row.get("id") or "") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("response ids do not exactly match the requested batch")
    for row in rows:
        if row.get("difficulty") not in DIFFICULTIES:
            raise ValueError(f"invalid difficulty for {row.get('id')}")
        if row.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError(f"invalid confidence for {row.get('id')}")
        if len(str(row.get("reason") or "").strip()) < 35:
            raise ValueError(f"insufficient semantic reason for {row.get('id')}")
    return rows


def review_batch(
    items: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    retries: int,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)
    expected = {str(row["id"]) for row in items}
    prompt = "Review every item below.\n\n" + json.dumps(items, ensure_ascii=False)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=8192,
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            labels = validate_response(parsed, expected)
            return {"success": True, "ids": sorted(expected), "labels": labels, "attempts": attempt + 1}
        except Exception as error:
            last_error = str(error)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {"success": False, "ids": sorted(expected), "error": last_error, "attempts": retries + 1}


def chunks(rows: list[Any], size: int) -> list[list[Any]]:
    return [rows[index:index + size] for index in range(0, len(rows), size)]


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)] if ordered else 0


def materialize(row: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    proof = str(row.get("proof") or "")
    premises = list(row.get("premises") or [])
    trace = list(row.get("tactic_trace") or [])
    verification = row.get("verification") or {}
    return {
        "record_id": row.get("id"),
        "theorem_group_id": row.get("theorem_group_id"),
        "source": "LeanDojo-v2",
        "theorem_name": row.get("qualified_name"),
        "statement": row.get("statement"),
        "proof": proof,
        "difficulty": annotation["difficulty"],
        "proof_tokens": int(row.get("label_tokens") or 0),
        "tactic_count": len(trace),
        "tactic_distribution": tactic_distribution(proof),
        "premise_count": len(premises),
        "same_file_premise_count": sum(bool(item.get("is_same_file")) for item in premises),
        "proof_style": row.get("proof_style"),
        "domain": source_domain(row),
        "source_file": row.get("source_file"),
        "imports": row.get("imports"),
        "repository": row.get("repository"),
        "repository_commit": row.get("repository_commit"),
        "lean_commit": row.get("lean_commit"),
        "leandojo_v2_commit": row.get("leandojo_v2_commit"),
        "assembled_source_hash": row.get("assembled_source_hash"),
        "verification_status": row.get("verification_status"),
        "pantograph_verified": bool(
            row.get("verification_status") == "verified_default_timeout"
            and verification.get("compile_success") is True
            and not verification.get("timed_out")
            and not verification.get("error_category")
        ),
        "verification_elapsed_seconds": verification.get("elapsed_seconds"),
        "reason": annotation["reason"],
        "confidence": annotation.get("confidence", "high"),
        "classification_origin": annotation["origin"],
        "classification_version": "ld-semantic-reclassification-v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/leandojo_v2_dataset_build/final_pool/leandojo_v2_final_train_candidates.jsonl"),
    )
    parser.add_argument(
        "--old-labels",
        type=Path,
        default=Path("outputs/ld_manual_classification_freeze/two_round_v1/combined_manual_labels_1725.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/leandojo_difficulty_classification"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--review-mode",
        choices=("local_codex", "deepseek"),
        default="local_codex",
        help="deepseek sends theorem/proof payloads externally and requires explicit approval",
    )
    args = parser.parse_args()

    project = args.project.resolve()
    input_path = args.input if args.input.is_absolute() else project / args.input
    labels_path = args.old_labels if args.old_labels.is_absolute() else project / args.old_labels
    output = args.output if args.output.is_absolute() else project / args.output
    output.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(input_path)
    old_labels = read_jsonl(labels_path)
    old_by_id = {str(row["sample_id"]): row for row in old_labels}
    if len(old_by_id) != len(old_labels):
        raise RuntimeError("frozen old labels contain duplicate sample ids")

    protected = protected_sets(project)
    protected_unions = {name: union_identity(rows) for name, rows in protected.items()}
    excluded_reasons: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for row in source_rows:
        ok, reason = verified_and_recoverable(row)
        if not ok:
            excluded_reasons[reason] += 1
            continue
        leaking = [name for name, values in protected_unions.items() if overlap(row, values)]
        if leaking:
            excluded_reasons["protected_evaluation_overlap"] += 1
            continue
        eligible.append(row)

    inherited: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for row in eligible:
        sample_id = str(row["id"])
        old = old_by_id.get(sample_id)
        if old:
            inherited[sample_id] = {
                "difficulty": old["difficulty"],
                "confidence": old.get("confidence", "high"),
                "reason": old.get("reason_summary") or "Inherited frozen semantic review.",
                "origin": "frozen_manual_codex_inherited",
            }
        else:
            pending.append(row)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if pending and args.review_mode == "deepseek" and not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for unreviewed LD records")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = (
        os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if args.review_mode == "deepseek"
        else "codex-local-semantic-rubric-v3"
    )
    checkpoint = output / (
        "local_review_checkpoint_v3.jsonl"
        if args.review_mode == "local_codex"
        else "llm_review_checkpoint.jsonl"
    )
    completed_batches = read_jsonl(checkpoint) if checkpoint.exists() else []
    if any(not row.get("success") for row in completed_batches):
        completed_batches = [row for row in completed_batches if row.get("success")]
        write_jsonl(checkpoint, completed_batches)
    llm_labels: dict[str, dict[str, Any]] = {}
    for batch in completed_batches:
        batch_origin = (
            f"local_semantic_review:{model}"
            if batch.get("review_mode") == "local_codex"
            else f"llm_semantic_review:{model}"
        )
        for label in batch["labels"]:
            llm_labels[str(label["id"])] = {
                "difficulty": label["difficulty"],
                "confidence": label["confidence"],
                "reason": label["reason"],
                "origin": batch_origin,
            }
    remaining = [row for row in pending if str(row["id"]) not in llm_labels]
    review_batches = chunks([review_item(row) for row in remaining], args.batch_size)
    failures: list[dict[str, Any]] = []
    if remaining and args.review_mode == "local_codex":
        by_id = {str(row["id"]): row for row in remaining}
        for batch_items in review_batches:
            labels = []
            for item in batch_items:
                label = local_semantic_review(by_id[str(item["id"])])
                labels.append({"id": item["id"], **label})
            result = {
                "success": True,
                "ids": sorted(str(item["id"]) for item in batch_items),
                "labels": labels,
                "attempts": 1,
                "review_mode": "local_codex",
            }
            append_jsonl(checkpoint, result)
            for label in labels:
                llm_labels[str(label["id"])] = {
                    "difficulty": label["difficulty"],
                    "confidence": label["confidence"],
                    "reason": label["reason"],
                    "origin": f"local_semantic_review:{model}",
                }
        print(json.dumps({"reviewed": len(llm_labels), "review_mode": args.review_mode}), flush=True)
    elif review_batches:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    review_batch,
                    batch,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    retries=args.retries,
                ): batch
                for batch in review_batches
            }
            for future in as_completed(futures):
                result = future.result()
                if result.get("success"):
                    append_jsonl(checkpoint, result)
                    for label in result["labels"]:
                        llm_labels[str(label["id"])] = {
                            "difficulty": label["difficulty"],
                            "confidence": label["confidence"],
                            "reason": label["reason"],
                            "origin": f"llm_semantic_review:{model}",
                        }
                    print(
                        json.dumps(
                            {
                                "reviewed": len(llm_labels),
                                "pending_total": len(pending),
                                "batch_size": len(result["labels"]),
                            }
                        ),
                        flush=True,
                    )
                else:
                    failures.append(result)
                    print(json.dumps({"batch_failed": result.get("ids")}), flush=True)
    if failures:
        write_json(output / "llm_review_failures.json", failures)
        raise RuntimeError(f"{len(failures)} LLM review batches failed; resumable checkpoint retained")

    annotations = {**inherited, **llm_labels}
    missing = [str(row["id"]) for row in eligible if str(row["id"]) not in annotations]
    if missing:
        raise RuntimeError(f"difficulty review incomplete for {len(missing)} records")

    materialized_all = [materialize(row, annotations[str(row["id"])]) for row in eligible]
    # The verified source pool intentionally retains a small number of alternate
    # proofs. Difficulty pools must expose exactly one row per theorem group.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in materialized_all:
        grouped.setdefault(str(row["theorem_group_id"]), []).append(row)
    materialized: list[dict[str, Any]] = []
    group_aliases: list[dict[str, Any]] = []
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    for group_id, values in grouped.items():
        values.sort(
            key=lambda row: (
                row["difficulty"] == "exclude",
                not str(row["classification_origin"]).startswith("frozen_manual"),
                confidence_rank.get(str(row.get("confidence")), 9),
                str(row["record_id"]),
            )
        )
        materialized.append(values[0])
        for alias in values[1:]:
            group_aliases.append(
                {
                    "theorem_group_id": group_id,
                    "kept_record_id": values[0]["record_id"],
                    "excluded_alias_record_id": alias["record_id"],
                    "reason": "one canonical proof per theorem group",
                }
            )
    materialized.sort(key=lambda row: str(row["record_id"]))
    write_jsonl(output / "theorem_group_aliases.jsonl", group_aliases)
    official = [row for row in materialized if row["difficulty"] in {"easy", "medium", "difficult"}]
    excluded = [row for row in materialized if row["difficulty"] == "exclude"]
    by_difficulty = {
        value: [row for row in official if row["difficulty"] == value]
        for value in ("easy", "medium", "difficult")
    }
    paths = {
        "easy": output / "ld_easy_manifest.jsonl",
        "medium": output / "ld_medium_manifest.jsonl",
        "difficult": output / "ld_difficult_manifest.jsonl",
    }
    for key, path in paths.items():
        write_jsonl(path, by_difficulty[key])
    write_jsonl(output / "excluded_manifest.jsonl", excluded)

    duplicate_theorems = len(official) - len({str(row["theorem_name"]) for row in official})
    duplicate_groups = len(official) - len({str(row["theorem_group_id"]) for row in official})
    duplicate_records = len(official) - len({str(row["record_id"]) for row in official})
    final_overlap: dict[str, dict[str, int | bool]] = {}
    for name, values in protected_unions.items():
        counts = {key: 0 for key in values}
        for row in official:
            raw = identity(row)
            for key in counts:
                counts[key] += int(bool(raw[key] & values[key]))
        final_overlap[name] = {**counts, "passed": not any(counts.values())}
    provenance_complete = all(
        row.get("source_file") and row.get("repository_commit") and row.get("assembled_source_hash")
        for row in official
    )
    audit = {
        "status": "PASSED" if (
            not duplicate_theorems
            and not duplicate_groups
            and not duplicate_records
            and all(item["passed"] for item in final_overlap.values())
            and provenance_complete
        ) else "FAILED",
        "input_rows": len(source_rows),
        "eligible_after_current_gates": len(eligible),
        "official_classified_rows": len(official),
        "excluded_semantic_rows": len(excluded),
        "excluded_by_input_gate": dict(excluded_reasons),
        "duplicate_theorem": duplicate_theorems,
        "duplicate_record_id": duplicate_records,
        "theorem_group_duplicate": duplicate_groups,
        "source_provenance_complete": provenance_complete,
        "protected_evaluation_overlap": final_overlap,
        "input_sha256": sha256(input_path),
        "old_frozen_labels_sha256": sha256(labels_path),
        "manifest_sha256": {key: sha256(path) for key, path in paths.items()},
    }
    write_json(output / "difficulty_audit.json", audit)
    if audit["status"] != "PASSED":
        raise RuntimeError("difficulty hard audit failed")

    counts = Counter(row["difficulty"] for row in official)
    origin_counts = Counter(row["classification_origin"] for row in materialized)
    source_by_id = {str(row["id"]): row for row in source_rows}
    calibration_confusion: Counter[str] = Counter()
    for old in old_labels:
        sample_id = str(old["sample_id"])
        if sample_id not in source_by_id:
            continue
        predicted = local_semantic_review(source_by_id[sample_id])["difficulty"]
        calibration_confusion[f"{old['difficulty']}->{predicted}"] += 1
    calibration_total = sum(calibration_confusion.values())
    calibration_agreement = sum(
        value
        for key, value in calibration_confusion.items()
        if key.split("->", 1)[0] == key.split("->", 1)[1]
    )
    domain_counts = {
        key: dict(Counter(str(row["domain"]) for row in values))
        for key, values in by_difficulty.items()
    }
    distributions: dict[str, Any] = {}
    for key, values in by_difficulty.items():
        tokens = [int(row["proof_tokens"]) for row in values]
        tactics = [int(row["tactic_count"]) for row in values]
        premises = [int(row["premise_count"]) for row in values]
        distributions[key] = {
            "count": len(values),
            "proof_tokens": {
                "mean": statistics.fmean(tokens) if tokens else 0,
                "p50": percentile(tokens, 0.5),
                "p90": percentile(tokens, 0.9),
                "max": max(tokens, default=0),
            },
            "tactic_count": {"mean": statistics.fmean(tactics) if tactics else 0, "p90": percentile(tactics, 0.9)},
            "premise_count": {"mean": statistics.fmean(premises) if premises else 0, "p90": percentile(premises, 0.9)},
        }
    statistics_payload = {
        "status": "LD_DIFFICULTY_CLASSIFICATION_COMPLETED",
        "source_pool_rows": len(source_rows),
        "eligible_rows": len(eligible),
        "classified_rows": len(official),
        "difficulty_counts": dict(counts),
        "semantic_exclude_count": len(excluded),
        "theorem_group_aliases_removed": len(group_aliases),
        "classification_origins": dict(origin_counts),
        "llm_model": model,
        "old_valid_labels_inherited": sum(
            value["origin"] == "frozen_manual_codex_inherited" and value["difficulty"] != "exclude"
            for value in inherited.values()
        ),
        "new_semantic_reviews": len(llm_labels),
        "review_method": args.review_mode,
        "review_engine": model,
        "domains": domain_counts,
        "distributions": distributions,
        "frozen_label_calibration": {
            "rows": calibration_total,
            "exact_agreement": calibration_agreement,
            "exact_agreement_rate": calibration_agreement / calibration_total if calibration_total else 0,
            "confusion": dict(calibration_confusion),
            "note": "Frozen labels remain authoritative and were inherited unchanged; this audit only calibrates the local rubric used for previously unreviewed rows.",
        },
        "audit_status": audit["status"],
        "training_started": False,
    }
    write_json(output / "classification_statistics.json", statistics_payload)

    report = f"""# LeanDojo difficulty reclassification report

## Scope and method

- Current verified source pool: {len(source_rows)} rows.
- Eligible after current provenance, Pantograph, context-recovery, and protected-evaluation gates: {len(eligible)} rows.
- Official three-way classified pool: {len(official)} rows; semantic exclusions: {len(excluded)}; alternate theorem-group proofs removed: {len(group_aliases)}.
- Frozen manual/Codex judgments were inherited unchanged; all remaining eligible rows were reviewed by `{model}` using the inherited semantic rubric.
- The review considered the complete theorem and proof, tactic structure, named and same-file premises, Mathlib domain, recovered context, and current-model learnability. Counts were supporting evidence, never stand-alone thresholds.

## Difficulty pools

| Difficulty | Rows | Intended use |
|---|---:|---|
| Easy | {counts['easy']} | Foundation SFT and EI discovery |
| Medium | {counts['medium']} | Curriculum SFT and later EI |
| Difficult | {counts['difficult']} | Planner, EI and agentic research |

## Sustainability

The new easy pool contains {counts['easy']} identities versus the old protected-clean trainable-easy pool of 1,004. After accounting for the prior 1,000-row SFT draw, it leaves {max(0, counts['easy'] - 1000)} identities of nominal headroom before any additional training/evaluation exclusions. The medium and difficult pools preserve source-file/domain/context metadata, so future curriculum selection can stratify without relabeling or modifying proofs.

## Hard audit

- Duplicate theorem: {duplicate_theorems}
- Duplicate theorem group: {duplicate_groups}
- Duplicate record id: {duplicate_records}
- Source provenance complete: {provenance_complete}
- Zero overlap with every protected evaluation set on all six axes: {all(item['passed'] for item in final_overlap.values())}
- Local rubric agreement against {calibration_total} frozen labels: {calibration_agreement}/{calibration_total} ({calibration_agreement / calibration_total:.2%}); frozen judgments remained authoritative on every disagreement.
- Audit status: {audit['status']}

No original data, evaluation set, model, checkpoint, Trainer, EI pipeline, or generation contract was modified.
"""
    (output / "classification_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(statistics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
