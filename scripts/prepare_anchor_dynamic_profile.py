"""Select a deterministic, stratified 3x300 anchor profile without proof leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from lean_prover.lean_training.expert_iteration.schemas import stable_statement_id


def _hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _length_bin(tokens: int) -> str:
    if tokens <= 20:
        return "xs"
    if tokens <= 35:
        return "short"
    if tokens <= 60:
        return "medium"
    return "long"


def _round_robin(rows: list[dict], *, count: int, seed: int) -> list[dict]:
    strata: dict[tuple[str, str, str], deque[dict]] = defaultdict(deque)
    for row in rows:
        key = (str(row["category"]), str(row["tactic_signature"]), _length_bin(int(row["proof_tokens"])))
        strata[key].append(row)
    for key, values in list(strata.items()):
        strata[key] = deque(sorted(values, key=lambda row: _hash(seed, str(row["record_id"]))))
    keys = sorted(strata, key=lambda key: _hash(seed, "|".join(key)))
    selected: list[dict] = []
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            values = strata[key]
            if values and len(selected) < count:
                selected.append(values.popleft())
            if values:
                next_keys.append(key)
        keys = next_keys
    if len(selected) != count:
        raise ValueError(f"requested {count} rows but selected {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--static", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--per-bucket", type=int, default=300)
    args = parser.parse_args()

    anchors = [json.loads(line) for line in args.anchor.read_text(encoding="utf-8").splitlines() if line.strip()]
    static = [json.loads(line) for line in args.static.read_text(encoding="utf-8").splitlines() if line.strip()]
    anchor_by_id = {str(row.get("record_id") or row.get("id")): row for row in anchors}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in static:
        groups[str(row["static_bucket"])].append(row)
    expected = ("static_easy", "static_medium", "static_hard")
    selected = []
    for offset, bucket in enumerate(expected):
        selected.extend(_round_robin(groups[bucket], count=args.per_bucket, seed=args.seed + offset))
    selected.sort(key=lambda row: _hash(args.seed, str(row["record_id"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "anchor_dynamic_profile_selected.jsonl"
    map_rows = []
    with profile_path.open("w", encoding="utf-8") as handle:
        for feature in selected:
            original = anchor_by_id[str(feature["record_id"])]
            source = str(original.get("source") or original.get("source_dataset") or "unknown")
            source_id = str(original.get("id") or original.get("record_id"))
            statement = str(original.get("lean_statement") or original.get("statement"))
            evaluation_id = stable_statement_id(source, source_id, statement)
            metadata = dict(original.get("metadata") or {})
            metadata.update({
                "anchor_record_id": feature["record_id"],
                "anchor_static_bucket": feature["static_bucket"],
                "anchor_static_score": feature["static_score"],
                "anchor_tactic_signature": feature["tactic_signature"],
                "anchor_proof_tokens": feature["proof_tokens"],
            })
            prepared = {
                "id": source_id,
                "source": source,
                "data_role": "discovery",
                "category": feature["category"],
                "imports": original.get("imports") or ["Mathlib"],
                "namespace": original.get("namespace"),
                "context": original.get("context"),
                "context_lines": original.get("context_lines") or [],
                "informal_statement": original.get("informal_statement") or "",
                "lean_statement": statement,
                "proof": None,
                "completion": None,
                "reference_proof": None,
                "has_reference_proof": False,
                "metadata": metadata,
            }
            handle.write(json.dumps(prepared, ensure_ascii=False) + "\n")
            map_rows.append({**feature, "evaluation_statement_id": evaluation_id})
    with (args.output_dir / "anchor_dynamic_profile_map.jsonl").open("w", encoding="utf-8") as handle:
        for row in map_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    model_files = sorted(path for path in args.model.rglob("*") if path.is_file() and path.name.endswith((".safetensors", ".json", ".model")))
    model_manifest = {str(path.relative_to(args.model)): _sha256(path) for path in model_files}
    sampling = {"temperature": 0.8, "top_p": 0.95, "do_sample": True, "max_new_tokens": 256, "samples_per_statement": 4, "seed": args.seed}
    manifest = {
        "seed": args.seed,
        "selection_algorithm": "round_robin(category,tactic_signature,proof_length_bin)+sha256_tiebreak",
        "static_bucket_population": {key: len(groups[key]) for key in expected},
        "selected_bucket_counts": dict(Counter(row["static_bucket"] for row in selected)),
        "selected_category_counts": dict(Counter(row["category"] for row in selected)),
        "selected_signature_counts": dict(Counter(row["tactic_signature"] for row in selected)),
        "profile_sha256": _sha256(profile_path),
        "sampling": sampling,
        "sampling_sha256": hashlib.sha256(json.dumps(sampling, sort_keys=True).encode()).hexdigest(),
        "checkpoint_path": str(args.model),
        "checkpoint_file_sha256": model_manifest,
        "checkpoint_manifest_sha256": hashlib.sha256(json.dumps(model_manifest, sort_keys=True).encode()).hexdigest(),
    }
    (args.output_dir / "anchor_dynamic_profile_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
