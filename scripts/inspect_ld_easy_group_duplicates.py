import collections
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


project = Path(".")
root = project / "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion"
pool = {row["id"]: row for row in read_jsonl(root / "frozen_expansion_pool.jsonl")}
labels = [
    row
    for row in read_jsonl(root / "expansion_labels.jsonl")
    if row.get("difficulty") == "easy"
    and row.get("trainable_for_short_whole_proof") is True
]
old = read_jsonl(
    project / "outputs/ld_length_difficulty_pipeline/difficulty/easy_pool.jsonl"
)
rows = old + [pool[row["sample_id"]] for row in labels]
groups: dict[str, list[dict]] = collections.defaultdict(list)
for row in rows:
    group = str(row.get("theorem_group_id") or row.get("id"))
    groups[group].append(
        {
            "id": row.get("id"),
            "full_name": row.get("full_name"),
            "statement": row.get("statement"),
        }
    )
print(
    json.dumps(
        {group: members for group, members in groups.items() if len(members) > 1},
        ensure_ascii=False,
        indent=2,
    )
)
