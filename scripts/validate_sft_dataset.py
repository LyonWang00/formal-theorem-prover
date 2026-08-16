#!/usr/bin/env python3
"""Load one prepared SFT dataset and enforce its Pantograph attestation contract."""

from __future__ import annotations

import argparse
import json

from lean_prover.lean_training.data.training import load_prepared_dataset
from lean_prover.lean_training.sft_pipeline.trainer import (
    validate_pantograph_attestation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    dataset = load_prepared_dataset(args.dataset)
    validate_pantograph_attestation(dataset, name="sft_train")
    expert_sources = (
        dataset["expert_source"]
        if "expert_source" in dataset.column_names
        else []
    )
    print(
        json.dumps(
            {
                "rows": len(dataset),
                "columns": dataset.column_names,
                "current_expert_rows": sum(
                    value == "current_expert" for value in expert_sources
                ),
                "all_pantograph_verified": all(dataset["pantograph_verified"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
