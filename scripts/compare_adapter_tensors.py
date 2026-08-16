"""Compare PEFT adapter tensor values independently of safetensors metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left = load_file(str(args.left))
    right = load_file(str(args.right))
    if left.keys() != right.keys():
        raise SystemExit(
            f"tensor keys differ: left_only={sorted(left.keys() - right.keys())}, "
            f"right_only={sorted(right.keys() - left.keys())}"
        )

    differing = 0
    max_absolute_difference = 0.0
    for key in left:
        difference = left[key] != right[key]
        differing += int(difference.sum())
        max_absolute_difference = max(
            max_absolute_difference,
            float((left[key] - right[key]).abs().max()),
        )
    print(
        {
            "tensor_count": len(left),
            "differing_elements": differing,
            "max_absolute_difference": max_absolute_difference,
        }
    )


if __name__ == "__main__":
    main()
