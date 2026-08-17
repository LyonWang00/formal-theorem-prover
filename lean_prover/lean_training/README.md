# Lean 证明模型训练框架

`lean_training` 负责 Lean 证明数据准备、LoRA/QLoRA 监督微调（SFT）、GRPO 强化学习、Pantograph 验证以及统一离线评估。目录按“共享能力”和“训练流程”分层：数据、模型构建、验证与评估由 SFT 和 GRPO 共用；两种训练方法只保留各自特有的配置和训练逻辑。

## 目录结构

```text
lean_training/
├── data/                       # 共享数据层
│   ├── adapters/              # 数据源及角色专用 adapter
│   │   ├── __init__.py        # 注册表与统一导出
│   │   ├── common.py          # 共享解析和 verified-scope 门禁
│   │   ├── numinamath.py      # NuminaMath SFT + GRPO
│   │   ├── kimina.py          # Kimina GRPO
│   │   ├── minif2f.py         # miniF2F benchmark
│   │   ├── lean_workbook.py   # Lean-Workbook SFT/轨迹重建
│   │   └── leandojo.py        # LeanDojo SFT/轨迹重建
│   ├── preparation.py          # 通用读取、标准化、过滤、切分、Pantograph 校验与 JSONL 导出
│   ├── training.py             # SFT/GRPO/评估格式生成、proof 长度元数据与跨阶段去重
│   └── cli.py                  # 数据准备命令的参数解析与输出编排
├── modeling/                   # SFT 与 GRPO 共用的模型构建层
│   ├── tokenizer.py            # tokenizer 加载和 padding 配置
│   ├── quantization.py         # QLoRA 计算精度与 4-bit 模型加载
│   ├── lora.py                 # 相互独立的 SFT/GRPO PEFT LoRA 配置入口
│   └── runtime.py              # 随机种子和依赖版本信息
├── verification/              # 共享 Lean/Pantograph 验证层
│   ├── schema.py               # 验证任务、结果与预热报告的数据结构
│   ├── pantograph.py           # 单个持久化 Pantograph 验证器
│   ├── pool.py                 # 多进程验证池、磁盘任务引用与有界队列
│   └── cache.py                # 按 Lean 环境隔离的 SQLite 验证缓存
├── runtime/                   # 硬件感知的运行时策略层
│   ├── config.py               # runtime YAML 配置模型
│   ├── profile.py              # laptop/server profile 解析
│   ├── lifecycle.py            # vLLM/Pantograph/Trainer 生命周期
│   ├── resource_manager.py     # orchestrator 使用的统一资源入口
│   ├── memory_monitor.py       # RAM/GPU 采样和 OOM 前保护
│   └── engine.py               # sequential 与未来 pipeline/async 接口
├── sft_pipeline/               # LoRA/QLoRA SFT 专属流程
│   ├── config.py               # SFTTrainConfig
│   └── trainer.py              # 参数解析、数据检查、SFTTrainer 构建与训练入口
├── grpo_pipeline/              # GRPO 专属流程
│   ├── config.py               # GRPOTrainConfig
│   ├── rewards.py              # 编译、结构、简短性奖励及 Pantograph 错误提取
│   └── trainer.py              # 参数解析、数据检查、GRPOTrainer 构建与训练入口
├── evaluation/                 # Prover 的 benchmark 与训练后 rollout
│   ├── benchmark.py            # miniF2F 生成、并行验证、pass@k 与结果汇总
│   └── rollout.py              # SFT/GRPO final_data 的统一 rollout 门面
├── expert_iteration/           # 可恢复的多轮 SFT 专家迭代
│   ├── config.py               # YAML/JSON 配置与跨字段校验
│   ├── schemas.py              # 数据角色、Bank、生成/验证与轮次状态 Schema
│   ├── datasets.py             # 角色隔离、重叠报告和 prompt 泄漏保护
│   ├── discovery_pool.py       # new/frontier/unsolved/audit 确定性调度
│   ├── discovery_generator.py  # 复用 benchmark 生成后端的候选生成
│   ├── discovery_verifier.py   # 复用 Pantograph 验证池与环境缓存
│   ├── banks.py                # 幂等 Proof Bank / Failure Bank
│   ├── train_dataset_builder.py# 累计专家数据、token mix 与门类温和平衡
│   ├── trainer_adapter.py      # 复用现有 QLoRA SFTTrainer
│   ├── evaluation_adapter.py   # monitor 与最终 benchmark 适配器
│   └── orchestrator.py         # 阶段状态机、恢复、停止与 checkpoint 选择
├── sft.py                      # 兼容 CLI，转发到 sft_pipeline.trainer
└── grpo.py                     # 兼容 CLI，转发到 grpo_pipeline.trainer
```

`benchmark_pantograph.py` 仅用于 Pantograph/Lean 环境诊断和性能测试；`check_data_leakage.py` 用于检查训练、验证和基准数据之间的定理泄漏。它们不是正式训练入口。

## 模块边界与数据流

```text
原始/verified 数据
  └─ data.adapters → data.preparation → Pantograph
       ├─ SFT manifest（statement + proof + provenance）
       │    └─ data.training → prompt + completion ── sft_pipeline
       └─ GRPO manifest（statement-only + provenance）
            └─ data.training → prompt + lean_statement ── grpo_pipeline
                                                        └─ rewards ── verification

基座模型 / SFT adapter / GRPO adapter
  └─ evaluation.benchmark ── verification ── pass@k 与明细结果
```

- `data/adapters/` 处理数据源字段和角色契约；`data/preparation.py` 只处理通用读取、调度、过滤、切分和 Lean 校验；`data/training.py` 是唯一的 SFT、GRPO 与生成评估记录格式及跨文件去重实现。
- SFT 去重键为 `(normalized lean_statement, normalized proof)`，保留同题不同证明；GRPO 去重键仅为 normalized `lean_statement`。
- `modeling/tokenizer.py` 和 `modeling/quantization.py` 由两种训练共享；`modeling/lora.py` 提供独立的 `build_sft_lora_config` 与 `build_grpo_lora_config`，两边可分别演进超参数。
- `verification` 是唯一的 Pantograph 编译验证实现，既供数据校验和 GRPO reward 使用，也供离线 benchmark 使用。
- `evaluation` 与训练方法无关，通过可选的 LoRA adapter 路径评估基座、SFT 或 GRPO 模型。
- 顶层同名脚本只用于保留已有命令和外部导入兼容性，不包含第二份业务实现。新代码应优先导入包内规范路径。

现阶段仍保留一条 legacy SFT 路径：`data/verified_builder.py` 为
Lean-Workbook/LeanDojo 生成带完整 `LeanDataRecord` 与环境 attestation 的富记录。
这些富记录属于审计/迁移层，不应直接成为长期训练格式。新 `data.cli` 会同时
写出两个文件：富 `sft_manifest` (`SFTGeneralData`) sidecar，以及只含 `prompt`、`completion`
的 trainer-facing JSONL。SFT trainer 默认把两者逐行绑定校验；legacy 富训练行
仍可作为过渡输入，但新数据不得再向训练行添加 provenance、hash、验证回执或
重复的 `text`/`proof` 字段。

## 规范导入路径

```python
from lean_prover.lean_training.data.preparation import normalize_records
from lean_prover.lean_training.data.adapters import get_adapter
from lean_prover.lean_training.data.training import build_grpo_training_record
from lean_prover.lean_training.modeling.quantization import build_qlora_model
from lean_prover.lean_training.modeling.lora import build_grpo_lora_config
from lean_prover.lean_training.verification.pool import VerificationPool
from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.grpo_pipeline.rewards import PantographRewardFunction
from lean_prover.lean_training.evaluation.benchmark import run_pipeline
```

顶层路径（如 `lean_prover.lean_training.sft`）继续作为 CLI 和旧代码兼容入口，但不建议新模块依赖这些转发层。

## 数据约定

SFT trainer-facing 记录固定为两个字段：

- `prompt`：模型输入的定理陈述和上下文。
- `completion`：监督训练使用的证明体。

`lean_statement`、规范化 `proof`、imports/context、来源、去重哈希和 Pantograph
验证事实只保存在同序的 `sft_manifest` sidecar。需要加权采样的专家迭代训练
允许第三个 trainer-only 字段 `sample_weight`。`text = prompt + completion`、
重复的 `proof` 字段以及逐行 provenance 都禁止写入新训练文件。

SFT 训练时的 validation 用于计算 `eval_loss`，因此同样采用极简
`prompt`/`completion` 投影并配套独立 manifest。SFT 的生成式评估数据由
benchmark/evaluation 流程生成，只包含提示、定理与环境字段，不包含目标 proof。
不要把这两种“评估”数据混用。

GRPO 训练、GRPO validation 和 benchmark 记录都是无目标证明的提示记录：

- `prompt`：生成提示。
- `lean_statement`：Pantograph 编译时使用的原始定理陈述。
- `id`：奖励日志、验证结果和断点续跑使用的稳定标识。
- `imports`、`context_lines`：可选的 Lean 前置环境。
- GRPO 记录不包含、也不会读取 `proof`、`reference_proof`、`completion`、`text` 或任何由参考证明派生的长度/hash 字段。奖励只依据模型本轮生成的证明及 Pantograph 结果计算。

GRPO reward 记录生成证明的长度与外层结构。Pantograph 编译后会保存错误类型、首个错误、错误/警告数量；编译成功时获得 compile reward，格式有效时获得 format reward，且生成证明短于参考证明时按缩短比例获得 brevity reward。简短性奖励只对编译成功的证明生效，避免用空答案或无效短答案刷分。

### SFT 与 GRPO 数据重叠

是否排除 SFT 训练阶段已经见过的定理取决于实验目标，不应硬编码：

- 衡量 GRPO 带来的新增泛化能力时，建议排除 SFT train 与 validation 中出现过的全部 `statement_hash`。
- 在同一批题上继续强化已学习策略时，可以允许重叠，但结果不能解释为未见题泛化提升。

使用可重复的 `--grpo_exclude_sft_file` 参数启用排除。过滤按规范化定理的 `statement_hash` 进行，而不是按可能变化的记录 ID 进行。

如果 SFT 与 GRPO 都从完全相同的数据源和 split 取全量记录，那么排除 SFT train 与 validation 后不会剩下 GRPO 数据；此时应先为 GRPO 选择独立数据源或独立 split，而不是强行启用排除。例如：

```bash
python -m lean_prover.lean_training.data.cli \
  --train_dataset_name lean_prover/Dataset/verified_data/kimina_verified_success.jsonl \
  --train_data_kind kimina-grpo \
  --grpo_train_output outputs/data/lean_workbook_grpo.jsonl \
  --grpo_validation_output outputs/data/lean_workbook_grpo_validation.jsonl \
  --grpo_exclude_sft_file outputs/data/lean_workbook_train.jsonl \
  --grpo_exclude_sft_file outputs/data/lean_workbook_validation.jsonl
```

## 常用命令

以下命令保留原有顶层入口，现由兼容层转发到新包结构。

分别准备 SFT、GRPO 和基准数据（示例拆开执行，避免混用数据角色）：

```bash
python -m lean_prover.lean_training.data.cli \
  --train_dataset_name InternLM/Lean-Workbook \
  --train_output outputs/data/lean_workbook_train.jsonl \
  --validation_output outputs/data/lean_workbook_validation.jsonl \
  --validation_ratio 0.02

python -m lean_prover.lean_training.data.cli \
  --train_dataset_name lean_prover/Dataset/verified_data/kimina_verified_success.jsonl \
  --train_data_kind kimina-grpo \
  --grpo_train_output outputs/data/kimina_grpo.jsonl \
  --grpo_validation_output outputs/data/kimina_grpo_validation.jsonl \
  --validation_ratio 0.02

python -m lean_prover.lean_training.data.cli \
  --benchmark_dataset_name path/to/minif2f/test.jsonl \
  --benchmark_output outputs/data/minif2f_benchmark.jsonl
```

运行 SFT：

```bash
python -m lean_prover.lean_training.sft \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file outputs/data/lean_workbook_train.jsonl \
  --validation_file outputs/data/lean_workbook_validation.jsonl \
  --output_dir outputs/runs/qwen2_5_0_5b_lean_sft
```

运行 GRPO（输入记录须包含 `prompt`、`lean_statement` 和 `id`）：

```bash
python -m lean_prover.lean_training.grpo \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file outputs/data/kimina_grpo.jsonl \
  --output_dir outputs/runs/qwen2_5_0_5b_lean_grpo \
  --lean_project_path lean_project \
  --pantograph_imports Mathlib \
  --compile_success_reward 1.0 \
  --format_reward 0.1 \
  --brevity_reward 0.1
```

检查数据泄漏：

```bash
python -m lean_prover.lean_training.check_data_leakage \
  --train_file outputs/data/lean_workbook_train.jsonl \
  --validation_file outputs/data/lean_workbook_validation.jsonl \
  --benchmark_file outputs/data/minif2f_benchmark.jsonl
```

统一评估基座模型或 LoRA adapter：

```bash
python -m lean_prover.lean_training.evaluation.benchmark \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/runs/qwen2_5_0_5b_lean_sft \
  --benchmark_file lean_prover/Dataset/final_data/minif2f_data.jsonl \
  --output_dir lean_prover/Dataset/experiment_result \
  --generation_backend vllm \
  --pass_k 32 \
  --lean_project_path lean_project \
  --imports Mathlib
```

不传 `--adapter_path` 时评估基座模型；传入 SFT 或 GRPO 产生的 adapter 时使用同一套 benchmark，避免训练方法之间出现评估口径分叉。

## 评估输出

- `attempts.jsonl`：全部生成与验证尝试。
- `problem_results.jsonl`：每个问题的聚合结果。
- `success_attempts.jsonl`：编译成功的证明、诊断和耗时。
- `summary.json`：pass@k、后端、worker 数量、耗时和错误统计。
- `attempt_shards/`：支持恢复和调试的问题级尝试分片。

Lean warning 会保留在输出中；仅 Pantograph error、超时、拒绝或基础设施异常视为失败。

## 多轮 Expert Iteration

该流程只做“生成候选 → Pantograph 验证 → 累计专家 SFT”，不会调用 GRPO/DPO。它复用现有 SFT、生成、验证和 benchmark 实现，并严格区分五种角色：

| 角色 | 用途 | 可进入 train/Bank |
| --- | --- | --- |
| `train` | 初始 SFT anchor | 进入 train，不进入 Bank |
| `eval` | 每轮固定、有 target 的 `eval_loss` | 禁止 |
| `discovery` | 无 target 的候选探索池 | 仅验证成功 proof 可进入 Proof Bank 和后续 train |
| `monitor` | 可选固定生成式 Pass@k | 禁止 |
| `benchmark` | 最终 miniF2F 生成式评估 | 禁止，默认只在迭代结束后运行 |

先复制并修改 [示例配置](../../configs/expert_iteration.example.yaml)，确保各数据文件按 statement ID、规范化 statement hash 和源 ID 均不重叠。`fixed_anchor` 会先把 SFT0 adapter 合并为固定基线；`continue_adapter` 则逐轮继续加载前一 adapter。

完整多轮运行：

```bash
python run_expert_iteration.py --config configs/expert_iteration.example.yaml
```

资源生命周期由顶层 `runtime` 配置统一决定，orchestrator 不再固定 worker 数量、队列大小或资源常驻方式。默认 `laptop` profile 面向 8 GB GPU：vLLM 只在生成 batch 内存在；生成结束后才启动两个 Pantograph worker，验证阶段结束即关闭；训练前会强制关闭所有验证资源。`server` profile 保留 vLLM/Pantograph 常驻和 `pipeline` 模式接口，但当前版本仍使用顺序实现，不包含在线 reward 或 GRPO async pipeline。

```yaml
runtime:
  profile: laptop
  generation_verification_mode: sequential
  generation: {mode: isolated, batch_size: 100, persistent: false}
  verification: {lifecycle: stage, workers: 2, persistent: false}
  memory: {streaming: true, max_queue_size: 32}
  cache: {backend: sqlite}
  debug: {save_full_source_on_failure_only: true}
  memory_guard: {enabled: true, warning_ratio: 0.85, critical_ratio: 0.95}
```

生成结果逐条追加到 `generations.jsonl`，中断后按稳定 generation ID 跳过已完成项。验证结果同样分批追加；cache 位于 `verification_cache.sqlite`，查询不会加载全量记录，且 Lean/mathlib、assembler 或 normalization 身份变化会自动 cache miss。验证任务完整内容先写入 `runtime/verification_tasks/`，多进程队列只传任务 ID、文件路径和调度元数据；任务与结果队列都受 `runtime.memory.max_queue_size` 限制。

每个阶段和 GPU 子进程等待期间会把协调器 RSS、系统内存、可用内存、GPU 显存与活动资源写入 `runtime/memory.jsonl`，生命周期事件写入 `runtime/lifecycle.jsonl`。达到 warning 阈值只记录告警；达到 critical 阈值时会停止当前生成/验证资源、保留已 flush 的 JSONL、回收 Python/CUDA cache，并以可恢复错误退出当前运行，而不是直接杀死整个实验。

成功验证明细只保留 generation ID、proof、status 和编译时间；失败明细额外保留 prompt、raw output、assembled source 和完整错误。worker 继续记录 PID、心跳、Pantograph server 启动时间和重启次数；单 worker 异常时只重启该 worker并重试当前磁盘任务。每轮验证前仍会重新计算 Lean、mathlib、imports 与验证配置指纹，指纹变化时由 runtime 层重建验证池。

vLLM discovery 生成和 QLoRA 训练默认由独立子进程执行。子进程通过磁盘 JSON 请求/结果与主协调器通信，并以 `close_fds=True` 启动，不继承 Pantograph 的队列或管道；阶段结束后操作系统回收完整 CUDA context。运行信息分别写入 `runtime/pantograph_pool.json`、`runtime/isolated_processes.jsonl`、`runtime/lifecycle.jsonl`、`runtime/memory.jsonl` 和 `runtime/isolated_logs/`。设置 `execution.isolate_gpu_stages: false` 可退回进程内执行，低显存环境不建议关闭隔离。

`execution.flashinfer_sampler: true` 会在 vLLM 生成子进程中设置 `VLLM_USE_FLASHINFER_SAMPLER=1`，并自动暴露当前 Python 环境中与 PyTorch CUDA 主版本匹配的 `nvcc`、CUDA headers 和库路径。它不会替换现有 torch、vLLM 或 FlashInfer 版本。

在 8 GB 等低显存 GPU 上，可同时设置 `discovery.generation.load_in_4bit: true` 和 `discovery.generation.enforce_eager: true`：前者让 vLLM 使用已有 BitsAndBytes 进行动态 4-bit 权重量化，后者避免 CUDA graph/TorchInductor 编译峰值；两者都不关闭 FlashInfer sampler。

如果 WSL 或共享 GPU 上的自动 KV-cache 预算仍不稳定，可用 `discovery.generation.kv_cache_memory_bytes` 指定正整数预算；默认 `null`，由 vLLM 按 `gpu_memory_utilization` 自动计算。

一轮运行或从中断阶段继续：

```bash
python run_expert_iteration.py --config configs/expert_iteration.example.yaml --iteration 0
python run_expert_iteration.py --config configs/expert_iteration.example.yaml \
  --iteration 0 --stage verify_discovery --resume
python run_expert_iteration.py --config configs/expert_iteration.example.yaml \
  --iteration 0 --stage build_train_dataset --resume
```

只运行可选 monitor 或最终 benchmark：

```bash
python run_expert_iteration.py --config configs/expert_iteration.example.yaml \
  --iteration 1 --stage monitor --resume
python run_expert_iteration.py --config configs/expert_iteration.example.yaml \
  --stage benchmark --resume
```

在启动 GPU 模型前检查路径、配置比例、角色重叠和候选规模：

```bash
python run_expert_iteration.py --config configs/expert_iteration.example.yaml --dry-run
```

小规模真实流程可从 [smoke 配置](../../configs/expert_iteration.smoke.example.yaml) 开始：它使用 10 道 discovery、每题 2 个候选、2 个 Pantograph worker、单轮且跳过训练，benchmark 默认关闭。单独验证 5 道 miniF2F：

```bash
python run_expert_iteration.py --config configs/expert_iteration.smoke.example.yaml
python -m lean_prover.lean_training.evaluation.benchmark \
  --model_name_or_path path/to/model \
  --adapter_path path/to/adapter \
  --benchmark_file lean_prover/Dataset/final_data/minif2f_data.jsonl \
  --output_dir lean_prover/Dataset/experiment_result \
  --num_benchmark_samples 5 --pass_k 2 --num_workers 2
```

每轮保存在 `iteration_NNN/` 下，包括独立状态、discovery 明细、训练 manifest、固定 eval 指标、可选 monitor、checkpoint、机器可读 `metrics.json` 和 `summary.md`。全局 Bank 使用稳定 ID 幂等写入；若 Lean/Mathlib/import 环境 hash 变化，历史 proof 会标记为 `needs_reverification`，不会直接进入下一轮训练。

## 扩展原则

- 新的数据源适配、normalization 和通用过滤写入 `data/preparation.py`；训练格式拼接写入 `data/training.py`，不要在 trainer 中复制。
- 新的 tokenizer 与量化公共能力写入 `modeling`；SFT/GRPO 的 LoRA 构建和训练超参数保持独立。
- 新的编译检查写入 `verification`；GRPO 的奖励组合和分值策略保留在 `grpo_pipeline/rewards.py`。
- 新的训练算法建立独立 pipeline 包，并复用 `data`、`modeling`、`verification` 和 `evaluation`。
- GRPO 奖励实验应先放入测试目录或单独实验分支；稳定的分项奖励再迁入 `grpo_pipeline/rewards.py`，避免形成第二套训练入口。
