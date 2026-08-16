# SFT Accelerate/DDP 安全审计

审计范围：标准入口 `lean_prover.lean_training.sft`、其 `sft_pipeline` 和
`modeling` 调用链，以及所有直接复用 `build_trainer` 的 SFT/消融脚本。
GRPO、推理和 Pantograph 验证进程不属于本报告的训练入口。

## 结论

标准 SFT 入口现在可由 `accelerate launch` 以 1/2/4 个进程启动。每个进程
只持有本地 GPU 上的一份 QLoRA 模型；Trainer 负责 DDP、梯度同步、优化器和
checkpoint 内部保存。项目自定义的配置、tokenizer 和诊断文件只由 rank 0
写入。Accelerate 的 `even_batches` 被关闭，避免为补齐跨 rank 批次而重复抽取
头部样本。

本次没有更改训练数据、formatter、tokenizer、loss、LoRA 配置、学习率、
per-device batch、梯度累积、epoch、评估或 checkpoint 周期。

## 逐项审计

| 位置 | 发现与风险原因 | 是否需要修改 | 处理结果 |
|---|---|---:|---|
| `lean_prover/lean_training/sft_pipeline/config.py:50`、`trainer.py:235` | programmatic 配置的 legacy 默认仍为 `device_map=auto`；若每个 DDP 进程直接沿用，会与“一 rank 一卡”冲突。 | 标准入口需要 | 为不改变旧消融合同，dataclass 默认保留；标准 CLI 和新启动器明确使用 `local_rank`，分布式硬门拒绝 `auto`。|
| `lean_prover/lean_training/sft_pipeline/distributed.py:65-94` | DDP 下需要拒绝 `auto`、`cuda:0`、`cuda:x` 等不安全映射。 | 是 | 新增硬门；DDP 仅允许本地 rank 映射，单进程可显式保留 legacy `auto`。|
| `lean_prover/lean_training/modeling/quantization.py:18-53` | 4-bit 模型加载直接消费 `device_map`；若是 `auto` 会产生跨卡模型。 | 是 | 接受已解析的 `{ "": local_rank }`，模型结构和量化配置不变。|
| `lean_prover/lean_training/sft_pipeline/trainer.py:686-718, 799-809, 948-966, 1098-1110` | `training_config.json`、EOS 合同、tokenization diagnostics、tokenizer 和最终状态原来可能由每个进程同时写。 | 是 | 所有项目级写盘显式加 main-process 门；训练前后有 barrier。Trainer 自身 checkpoint 仍使用原保存策略。|
| `lean_prover/lean_training/sft_pipeline/trainer.py:903-907` | Accelerate 默认 `even_batches=True` 会在不能整除 world size 时复制样本；LoRA DDP 也不需要搜索未使用参数。 | 是 | 设 `even_batches=False`；多进程时 `ddp_find_unused_parameters=False`。|
| `lean_prover/lean_training/sft_pipeline/distributed.py` 的 `validate_no_duplicate_train_sharding` | `even_batches=False` 时，训练微批次数若不能整除 world size，各 rank 步数不同，反向同步可能挂起。 | 是 | 启动前硬失败；不擅自复制或丢弃记录，由实验负责人调整数据量或原有 batch 参数。|
| `lean_prover/lean_training/sft_pipeline/trainer.py:89-118` | `WeightedRandomSampler(replacement=True)` 会产生重复样本。这里的重复是既有 source-weighted 实验策略，不是 DDP 偶发重复。各 rank 的 DataLoader 仍由 Accelerate 分片。 | 否（实验合同） | 保留原 sampler；报告中明确该模式允许策略性重复。|
| `lean_prover/lean_training/sft_pipeline/trainer.py:121-172` | fixed-manifest sampler 在各 rank 生成同一全局 permutation；必须依靠 Accelerate 的 batch shard 分配，且不能补齐复制。 | 是 | 保留 sampler 和 seed，使用 `even_batches=False`。|
| `scripts/linux_sft_train_workflow.sh:150` | 旧工作流仍用普通 `python -m ...sft`，只会启动单进程；不是多 GPU DDP 入口。 | 多卡时是 | 保留作为旧单卡工作流；多卡改用 `scripts/run_sft_ddp.sh`，避免改变既有数据准备合同。|
| `lean_prover/lean_training/expert_iteration/trainer_adapter.py:114-116, 200-207` | 适配器自行保存 tokenizer/state，merge 阶段还用 `device_map=auto`；若整个 orchestrator 被 `accelerate launch` 多开会重复写盘/合并。 | 是（若要 DDP） | 本次不改；禁止用新启动器直接启动该 orchestrator。它需要单独的 rank-0 编排设计。|
| `scripts/train_expert_sft_ablation.py:68-70` | 消融脚本自行保存文件，没有 rank-0 门。 | 是（若要 DDP） | 本次不改实验脚本；继续单进程运行。|
| `scripts/train_expert_sft_b_ablation.py:113-125` | 除重复保存外还会重命名 checkpoint，多 rank 运行会竞争。 | 是（若要 DDP） | 本次不改；继续单进程运行。|
| `scripts/train_wb_ld_small_sft_arm.py:172, 197-242` | 显式 `device_map=auto`，并自行复制 checkpoint、写 trace；其 sampler trace 假定单进程看到完整抽样序列。 | 是（若要 DDP） | 本次不改该冻结消融合同；继续单进程运行。|

## 未发现的模式

在上述 SFT 执行图中没有发现 `device="auto"`（非 `device_map`）、
`model.to("cuda")`、`model.cuda()`、固定 `cuda:0/cuda:x`、手写
`DistributedSampler` 或手写每-rank DataLoader。GPU 选择集中在 QLoRA
`from_pretrained(..., device_map=...)`，现已由分布式硬门统一解析。

## 运行边界

- 新 DDP 启动器只支持标准 `lean_prover.lean_training.sft`。
- 不得用该启动器直接运行上述实验性消融/orchestrator 脚本。
- `source_weighted_replacement` 的重复抽样是已有训练策略；smoke test 的
  duplicate 检查使用无放回的极小顺序数据，检查的是 DDP 分片重复。
- 2/4 卡 smoke 必须在相应数量的可见 GPU 上实际执行；启动器会在进程启动前
  检查可见卡数，不会静默降级。
