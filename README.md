# Formal Theorem Prover

本项目围绕 Lean 4、Mathlib、Pantograph、Planner 和 LoRA 微调构建形式化定理证明流程。当前已经基本完成的部分是：

- `Planner`：定理分解、blueprint 生成与 statement 级检查。
- `lean_training`：数据处理、QLoRA SFT、vLLM 批量生成、Pantograph 评估。
- `backends`：共享的 Pantograph 后端接口。

后续计划包括独立的 `Prover` 部分和强化学习部分。强化学习相关代码建议继续放在 `lean_prover/lean_training` 下；`Prover` 文件夹等设计稳定后再新建。

## 项目结构

```text
formal-theorem-prover/
  lean_project/                 Lean/Lake 项目，固定 Lean 和 mathlib 版本
  lean_prover/
    __init__.py                 轻量包入口
    proof_search.py             小型 BFS tactic-search baseline
    backends/                   Pantograph 后端接口和实现
    Planner/                    blueprint planning 与 statement 检查
    lean_training/              数据处理、SFT、benchmark、verification pool
  scripts/                      环境配置、preflight、WSL/Linux 工作流
  tests/                        单元测试和 smoke tests
  requirements_lean_env.txt     lean_env 环境的 Python 依赖锁定文件
```

`lean_prover` 根目录保持轻量。稳定的后端入口从 `lean_prover.__init__` 导出；Planner、训练和评估相关功能应分别从 `lean_prover.Planner`、`lean_prover.lean_training`、`lean_prover.backends` 导入。

## 环境与版本

推荐环境是 Linux/WSL2 + conda 环境 `lean_env`。

当前测试过的核心版本：

- Python `3.12`
- Lean `4.29.1`
- Mathlib `v4.29.1`
- Pantograph / PyPantograph `0.3.15`
- PyTorch `2.11.0`
- CUDA package family `13.0`
- vLLM `0.22.1`
- FlashInfer `0.6.11.post2`
- Transformers `5.13.1`
- TRL `1.8.0`
- PEFT `0.19.1`
- bitsandbytes `0.49.2`

版本锁定文件：

- `requirements_lean_env.txt`：Python 依赖。
- `lean_project/lean-toolchain`：Lean 版本。
- `lean_project/lakefile.lean` 和 `lean_project/lake-manifest.json`：mathlib 和 Lake 依赖。

## 一键环境配置

如果用户已经安装 Miniforge 或 Mambaforge、Git，以及提供 `lake` 命令的 Elan/Lean，可以直接使用 one-command setup。脚本会在安装依赖前检查这些前置命令：

```bash
bash scripts/setup_project_environment.sh
```

常用自定义参数：

```bash
ENV_NAME=lean_env \
PYTHON_VERSION=3.12 \
PANTOGRAPH_VERSION=0.3.15 \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts/setup_project_environment.sh
```

脚本会优先从当前 `PATH` 中的 `conda` 自动获取安装根目录，并兼容常见的 Miniforge、Mambaforge、Miniconda 和 `/opt/conda` 布局。如果 conda 位于其他位置，可显式设置 `CONDA_ROOT=/path/to/conda`。项目根目录默认根据脚本自身位置确定；只有从项目目录外调用或使用特殊目录布局时才需要设置 `LEAN_PROVER_ROOT=/path/to/formal-theorem-prover`。

默认版本为 `0.3.15`，并优先从项目内的 `.vendor/PyPantograph` 安装；如果该目录不存在，或通过 `PANTOGRAPH_VERSION` 选择了其他版本，则从官方 PyPantograph 仓库的对应版本标签安装。也可以通过 `PANTOGRAPH_SOURCE` 指定其他本地源码目录或兼容 pip 的 Git URL。

该脚本会：

1. 创建或复用 conda 环境 `lean_env`。
2. 自动 source `scripts/lean_env_runtime.sh` 并检查 `git`、`lake`。
3. 安装 `requirements_lean_env.txt`。
4. 显式安装 Pantograph / PyPantograph `0.3.15`，并检查实际安装版本；可通过 `PANTOGRAPH_VERSION` 覆盖版本。
5. 依赖安装后再次 source runtime，使全新环境中的 CUDA、vLLM、FlashInfer 和 Pantograph 路径立即生效。
6. 默认执行 `lake exe cache get` 准备 mathlib 缓存；如需跳过，可设置 `RUN_LAKE_CACHE=0`。
7. 默认执行 `scripts/preflight_lean_env.py`，其中包括 Pantograph warmup 和 Mathlib import；如需跳过，可设置 `RUN_PREFLIGHT=0`。

因此，对已经具备上述前置命令的用户，推荐优先使用 one-command setup。手动配置只在需要调试依赖、切换 CUDA/vLLM/Pantograph 版本或使用非标准 conda 路径时才需要。

## 如何 source runtime 环境

`scripts/lean_env_runtime.sh` 的作用是为 vLLM、FlashInfer 和 Pantograph 设置运行时环境。它会设置：

- `CUDA_HOME`
- Linux-only `PATH`
- `LD_LIBRARY_PATH`
- `LIBRARY_PATH`
- HuggingFace cache 相关变量

如果你已经用 one-command setup 完成环境配置，后续每次打开新终端时运行：

```bash
cd "$HOME/projects/formal-theorem-prover"
CONDA_ROOT="$(conda info --base)"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate lean_env
export PY="$(command -v python)"
export LINUX_PROJECT="$(pwd)"
source scripts/lean_env_runtime.sh
```

如果 conda 没有加入 `PATH`，请使用实际安装根目录初始化 conda；`PY` 始终可以在激活环境后动态获取：

```bash
CONDA_ROOT="/path/to/conda"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate lean_env
export PY="$(command -v python)"
export LINUX_PROJECT="$(pwd)"
source scripts/lean_env_runtime.sh
```

训练和评估 workflow 脚本已经会自动 source `scripts/lean_env_runtime.sh`，例如：

```bash
bash scripts/linux_sft_train_workflow.sh
bash scripts/linux_benchmark_adapter_workflow.sh
```

但如果你直接运行 Python 模块，例如直接调用 `evaluation.benchmark`，建议先手动 source runtime 环境。

## Preflight 检查

正式训练或评估前建议运行：

```bash
python scripts/preflight_lean_env.py \
  --lean_project_path lean_project \
  --timeout 1200
```

检查内容包括：

- `pip check`
- CUDA soname
- FlashInfer sampler smoke
- Pantograph warmup
- Mathlib import
- WSL memory/swap

## Backends

后端代码位于 `lean_prover/backends`。

| 文件 | 作用 |
| --- | --- |
| `proof_backend.py` | 后端无关的 proof state、tactic transition 和 proposer 协议。 |
| `pantograph_backend.py` | 常驻 Pantograph server 封装。 |
| `backend_factory.py` | 基于环境变量创建后端。 |

最小示例：

```python
from lean_prover.backends import BackendConfig, create_backend

backend = create_backend(BackendConfig.from_environment())
try:
    root = backend.start("forall n : Nat, n + 0 = n")
    result = backend.apply_tactic(root, "simp")
    print(result.status)
finally:
    backend.close()
```

当前项目统一以 Pantograph 作为主要 Lean 验证后端；旧的直接 Lean 子进程编译路径已经移除。

## Planner

`lean_prover/Planner` 负责定理规划和 statement 级检查。

主要职责：

- 表示 theorem problem 和 blueprint。
- 校验 blueprint schema 和依赖结构。
- 使用 Pantograph 检查 theorem/lemma statement。
- Planner 节点不包含 proof body。

详细说明见 `lean_prover/Planner/README.md`。

## Lean Training

`lean_prover/lean_training` 包含当前的 SFT 和 benchmark 流程。

| 文件 | 作用 |
| --- | --- |
| `data/cli.py` | 通过专用 adapter 准备 SFT、GRPO 和 miniF2F benchmark 数据。 |
| `check_data_leakage.py` | 检查 train、validation、benchmark 之间的 statement 泄露。 |
| `sft.py` | 使用 TRL `SFTTrainer` 进行 QLoRA SFT。 |
| `evaluation/benchmark.py` | Prover 的 miniF2F 批量生成、Pantograph pass@k 验证与报告。 |
| `evaluation/rollout.py` | Prover 的 SFT/GRPO 训练数据 rollout 与统一报告。 |
| `verification/pantograph.py` | 单个常驻 Pantograph verifier worker。 |
| `verification/pool.py` | 多进程 verifier 调度。 |
| `verification/schema.py` | verification task/result 数据结构。 |

更详细的训练和评估命令见 `lean_prover/lean_training/README.md`。

## 常用工作流

准备数据：

```bash
python -m lean_prover.lean_training.data.cli \
  --train_dataset_name InternLM/Lean-Workbook \
  --train_output outputs/data/lean_workbook_train.jsonl \
  --benchmark_dataset_name path/to/minif2f/test.jsonl \
  --benchmark_output outputs/data/minif2f_benchmark.jsonl
```

检查数据泄露：

```bash
python -m lean_prover.lean_training.check_data_leakage \
  --train_file outputs/data/lean_workbook_train.jsonl \
  --benchmark_file outputs/data/minif2f_benchmark.jsonl
```

训练：

```bash
python -m lean_prover.lean_training.sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_file outputs/data/lean_workbook_train.jsonl \
  --output_dir outputs/runs/qwen2_5_0_5b_lean_sft
```

评估：

```bash
python -m lean_prover.lean_training.evaluation.benchmark \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/runs/qwen2_5_0_5b_lean_sft \
  --benchmark_file lean_prover/Dataset/final_data/minif2f_data.jsonl \
  --output_dir lean_prover/Dataset/experiment_result \
  --generation_backend vllm \
  --generation_batch_size 4 \
  --pass_k 32 \
  --num_workers 2
```

## WSL/Linux workflow 脚本

常用脚本：

- `scripts/setup_project_environment.sh`：一键环境配置。
- `scripts/lean_env_runtime.sh`：CUDA/vLLM/Pantograph runtime 设置。
- `scripts/linux_sft_train_workflow.sh`：Linux/WSL SFT 工作流。
- `scripts/linux_benchmark_adapter_workflow.sh`：Linux/WSL adapter benchmark 工作流。
- `scripts/preflight_lean_env.py`：正式训练/评估前的环境检查。

在 WSL 中建议把项目放在 Linux 原生文件系统中，例如 `$HOME/projects/formal-theorem-prover`。尽量避免在 `/mnt/<drive-letter>` 下进行大量 Lean/mathlib 编译和缓存读写。

## 测试

运行 Python 测试：

```bash
python -m pytest -q
```

运行 Pantograph smoke test：

```bash
python -m scripts.pantograph_smoke_test
```

## 开发约定

- 共享 Lean/Pantograph 后端代码放在 `lean_prover/backends`。
- Planner 相关代码放在 `lean_prover/Planner`。
- SFT、benchmark 和未来 RL 代码放在 `lean_prover/lean_training`。
- 不再新增零散的根目录 `lean_prover/*.py`，除非它是稳定的包级工具。
- 未来如果需要 `lean_prover/Prover`，应在 Prover 设计稳定后再创建。
