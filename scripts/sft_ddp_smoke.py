#!/usr/bin/env python3
"""Fast Accelerate/DDP smoke test with auditable record ownership."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import gather_object
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyRecordDataset(Dataset):
    """Synthetic, deterministic rows used only to exercise DDP plumbing."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        value = float(index + 1)
        return {
            "features": torch.tensor([value], dtype=torch.float32),
            "target": torch.tensor([2.0 * value + 1.0], dtype=torch.float32),
            "record_id": f"smoke-{index:04d}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-optimizer-steps", type=int, default=2)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit a one-process CPU run for CI; production smoke tests use CUDA.",
    )
    return parser.parse_args()


def gpu_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"allocated": None, "reserved": None, "max_allocated": None}
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    args = parse_args()
    if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.max_optimizer_steps <= 0:
        raise ValueError("max optimizer steps must be positive")

    accelerator = Accelerator(
        cpu=args.allow_cpu and not torch.cuda.is_available(),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    if accelerator.device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for CI")

    world_size = accelerator.num_processes
    dataset_size = (
        args.per_device_batch_size
        * args.gradient_accumulation_steps
        * args.max_optimizer_steps
        * world_size
    )
    dataset = TinyRecordDataset(dataset_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        drop_last=False,
    )
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    optimizer_steps = 0
    local_record_ids: list[str] = []
    model.train()
    for batch in dataloader:
        local_record_ids.extend(str(item) for item in batch["record_id"])
        with accelerator.accumulate(model):
            prediction = model(batch["features"])
            loss = torch.nn.functional.mse_loss(prediction, batch["target"])
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            optimizer_steps += 1
            if optimizer_steps >= args.max_optimizer_steps:
                break

    rank_report = {
        "rank": int(accelerator.process_index),
        "local_rank": int(accelerator.local_process_index),
        "record_ids": local_record_ids,
        "optimizer_steps": optimizer_steps,
        "gpu_memory_bytes": gpu_memory(accelerator.device),
    }
    print("SFT_DDP_SMOKE_RANK " + json.dumps(rank_report, ensure_ascii=False))
    gathered = gather_object([rank_report])
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        gathered = sorted(gathered, key=lambda item: item["rank"])
        all_ids = [record_id for item in gathered for record_id in item["record_ids"]]
        counts = Counter(all_ids)
        duplicates = sorted(record_id for record_id, count in counts.items() if count > 1)
        report = {
            "world_size": int(world_size),
            "global_rank": int(accelerator.process_index),
            "local_rank": int(accelerator.local_process_index),
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": (
                args.per_device_batch_size
                * args.gradient_accumulation_steps
                * world_size
            ),
            "optimizer_steps": optimizer_steps,
            "records_by_rank": {
                str(item["rank"]): item["record_ids"] for item in gathered
            },
            "duplicate_draw_check": {
                "passed": not duplicates,
                "duplicate_record_ids": duplicates,
                "total_draws": len(all_ids),
                "unique_draws": len(counts),
            },
            "gpu_memory_by_rank_bytes": {
                str(item["rank"]): item["gpu_memory_bytes"] for item in gathered
            },
            "rank_optimizer_steps": {
                str(item["rank"]): item["optimizer_steps"] for item in gathered
            },
        }
        if duplicates:
            raise RuntimeError(f"DDP smoke test drew duplicate records: {duplicates}")
        if any(
            item["optimizer_steps"] != args.max_optimizer_steps for item in gathered
        ):
            raise RuntimeError("not every rank completed the requested optimizer steps")
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print("SFT_DDP_SMOKE " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
