from __future__ import annotations

import pytest

from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.distributed import (
    DistributedContext,
    global_batch_size,
    resolve_qlora_device_map,
    validate_no_duplicate_train_sharding,
)
from lean_prover.lean_training.sft_pipeline.trainer import save_training_config


class FakeState:
    distributed_type = "MULTI_GPU"
    device = "cuda:1"

    def wait_for_everyone(self) -> None:
        pass


def context(*, world_size: int, local_rank: int = 0) -> DistributedContext:
    return DistributedContext(
        state=FakeState(),  # type: ignore[arg-type]
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        is_main_process=local_rank == 0,
    )


def test_local_rank_map_binds_one_gpu_per_process() -> None:
    assert resolve_qlora_device_map(
        "local_rank", context(world_size=4, local_rank=3)
    ) == {"": 3}


@pytest.mark.parametrize("unsafe", ["auto", "cuda:0", "cuda:2", "0"])
def test_distributed_map_rejects_auto_and_fixed_cuda(unsafe: str) -> None:
    with pytest.raises(ValueError):
        resolve_qlora_device_map(unsafe, context(world_size=2, local_rank=1))


def test_single_process_legacy_auto_is_preserved() -> None:
    assert resolve_qlora_device_map("auto", context(world_size=1)) == "auto"


def test_global_batch_is_derived_without_changing_components() -> None:
    assert global_batch_size(
        per_device_batch_size=2, gradient_accumulation_steps=8, world_size=4
    ) == 64


def test_training_config_is_written_only_by_main_process(tmp_path) -> None:
    config = SFTTrainConfig(
        model_name_or_path="model",
        train_file="train.jsonl",
        output_dir=str(tmp_path / "output"),
    )
    save_training_config(config, context(world_size=2, local_rank=1))
    assert not (tmp_path / "output").exists()

    save_training_config(config, context(world_size=2, local_rank=0))
    assert (tmp_path / "output" / "training_config.json").is_file()


def test_no_duplicate_sharding_requires_equal_rank_microbatches() -> None:
    assert validate_no_duplicate_train_sharding(
        dataset_size=16,
        per_device_batch_size=2,
        world_size=4,
        drop_last=False,
    ) == 2
    with pytest.raises(ValueError, match="Refusing to replay or discard records"):
        validate_no_duplicate_train_sharding(
            dataset_size=13,
            per_device_batch_size=2,
            world_size=4,
            drop_last=False,
        )
