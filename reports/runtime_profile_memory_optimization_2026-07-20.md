# Expert Iteration Runtime Profile 与内存优化验收报告

日期：2026-07-20  
测试环境：RTX 4060 Laptop 8 GB、WSL Ubuntu 24.04、Lean 4.29.1、Qwen2.5-1.5B clean M0  
测试输出：`outputs/qwen25_1_5b_runtime_laptop_smoke50`

## 1. 当前 runtime 架构

专家迭代状态机、Lean 验证语义、Proof Bank/Failure Bank 语义和 reward 逻辑保持不变。新增 `lean_training/runtime` 作为资源策略层：配置先解析为不可变 `RuntimeProfile`，orchestrator 只通过 `ResourceManager` 和 `GenerationVerificationEngine` 请求生成、验证及训练资源，不再在 `run()` 开始时固定预热 Pantograph。

当前 discovery 执行仍为顺序模式：生成 batch 流式写入 JSONL，释放 vLLM 后启动 Pantograph，分批验证并流式写入结果，阶段结束按 profile 决定是否关闭 worker。`pipeline` 和 `async` 具有稳定方法接口，但本次没有实现在线 reward、GRPO 或真正异步流水线。

## 2. Profile 设计与差异

| 策略 | laptop | server |
|---|---:|---:|
| generation mode | isolated | persistent 接口 |
| statement batch | 100 | 512 |
| Pantograph 生命周期 | stage | persistent |
| Pantograph workers | 2 | 16 |
| generation/verification | sequential | pipeline 接口，当前顺序 fallback |
| queue maxsize | 32 | 1000 |
| JSONL streaming | 开启 | 开启 |
| cache | SQLite | SQLite |
| full source | 仅失败 | 仅失败 |

配置可覆盖 profile 默认值，但资源数量、生命周期、队列和 cache 决策均由 runtime 层解析，orchestrator 中没有 `if laptop/if server` 分支。

## 3. 生命周期变化

旧实现会在整个任务开始时启动一组 Pantograph worker，并在 vLLM 生成期间维持常驻。新 laptop 流程为：

```text
PRECHECK Pantograph -> stop
PRECHECK vLLM -> stop -> PRECHECK Pantograph -> stop
Discovery vLLM -> stop
Discovery Pantograph -> stop
若训练阈值满足：确认验证资源已关闭 -> QLoRA -> stop
```

50题 smoke 实际启动两次 generation 子进程（precheck、main）和三次两-worker Pantograph pool（固定/reference precheck、generation precheck 验证、main 验证），worker restart 为 0。此前同规模 smoke 使用一个跨阶段常驻 pool；新模式以约两次额外 Pantograph/mathlib 预热换取低峰值和阶段隔离。主 discovery 中 vLLM 与 Pantograph 编译没有重叠。

最后修正后，只有真正调用训练 adapter 时才记录 `trainer_start`；本次 24 条训练 proof 低于阈值 50，没有启动 QLoRA 子进程。

## 4. 内存优化

- `generations.jsonl` 每生成一条即 append/flush，不再保留整轮 `by_id` record 字典或反复重写全文件。
- verification 以 `max_queue_size` 为上限分批执行并逐条 append；主流程不返回整轮 verification list。
- `verification_cache.sqlite` 使用 WAL、按单 proof 查询，并用 environment/assembler/normalization identity 自动 cache miss。
- 完整 `VerificationTask` 先写入 `runtime/verification_tasks/`；任务队列只传 ID、路径、priority、attempt 和 enqueue time。任务队列与结果队列均有界。
- Bank 更新使用 `generation_index.sqlite` 做磁盘 join，不再构建整轮 generation dict、verification list、successful list 或 failed list。
- Failure Bank 只在显式 query 时读取记录；插入采用幂等 ID 集合与流式 append，不长期保存所有 raw failure 对象。
- 成功 verification 不保存 assembled source/context/debug log；失败记录保留 source、prompt、raw output 和完整错误。成功任务文件自动删除，失败任务文件保留。
- `MemoryMonitor` 每阶段及 GPU 子进程等待期间记录 coordinator RSS、系统内存、GPU 显存和活动资源。
- critical memory 会协作式停止当前资源、保留已 flush JSONL，执行 GC 与 CUDA cache 回收并抛出可恢复错误；不会直接 kill 整个实验。

## 5. RAM/GPU 实测

优化前没有同口径的 `memory.jsonl`，只能引用任务给出的历史现象：系统 RAM 经常达到 95%～100%。因此不能把该历史观察当作严格可复现实验基线。

新 laptop 50题 smoke 共记录 71 个采样点：

- 系统 RAM 峰值：24.9859%。
- coordinator RSS 峰值：907,739,136 bytes，约 865.7 MiB。
- GPU 显存采样峰值：5,504 MiB。
- warning：0；critical：0；OOM：0。
- 运行前 WSL 可用 RAM 约 14 GiB、GPU 10 MiB。
- 运行结束无 expert iteration、Pantograph、EngineCore/vLLM 或 GPU compute 残留进程。

实测满足 RAM 峰值小于 85% 的验收目标。历史 95%～100% 与本次 24.99% 的差异方向明确，但由于旧实现没有监控日志，报告不宣称这是严格 A/B 数字。

## 6. 50题 smoke 结果

- 总耗时：547.6 秒。
- generation：200/200，generation ID 200/200 唯一。
- verification：200/200，关联 generation ID 200/200 唯一。
- 编译成功候选：54/200，成功率 27%。
- 解题：24/50，Pass@4 为 48%。
- Proof Bank：41；Failure Bank：146。
- 两个 Pantograph worker 均参与；restart 0/0。
- SQLite cache：180 个唯一 proof/environment 项；少于 200 是因为重复 proof 共享 cache key。
- 成功 verification 中 full source：0；失败 verification 中 full source：146/146。
- 主运行加 precheck 共保留 165 个失败任务文件；成功任务均已清理。
- 每题最多一个训练 proof 后得到 24 条，低于 `minimum_new_proofs_to_train=50`，训练正确跳过，没有伪造 M1。

模型生成与验证成功率和改造前同一 clean M0 smoke 一致，说明 runtime 改造没有改变专家迭代算法或 Lean 判断结果。

## 7. 中断恢复与测试

新增测试分别模拟 generation 和 verification 在写入一部分 JSONL 后中断；resume 保留已完成行、补齐剩余行，稳定 ID 无重复。原有两轮 orchestration resume 测试继续验证 Proof Bank/Failure Bank 幂等。

Server mock 验证 profile 切换、persistent 生命周期和 pipeline 接口；OOM 测试验证 warning 只记录、critical 先调用清理再抛出可恢复异常；queue 测试确认 IPC 中没有 `lean_code` 等完整 source。

最终 Linux pytest：98 passed、1 skipped、0 failed。`compileall` 与 `git diff --check` 通过；项目环境未安装 ruff，因此没有运行 ruff。

## 8. 修改文件

新增：

- `lean_prover/lean_training/runtime/{__init__,config,profile,lifecycle,resource_manager,memory_monitor,engine}.py`
- `lean_prover/lean_training/verification/cache.py`
- `configs/runtime.server.example.yaml`
- `scripts/summarize_runtime_profile.py`

修改：

- `lean_prover/lean_training/expert_iteration/{config,orchestrator,discovery_generator,discovery_verifier,isolated_stage,utils,banks}.py`
- `lean_prover/lean_training/verification/{__init__,schema,pool}.py`
- `lean_prover/lean_training/README.md`
- 四个 expert iteration 示例/运行配置
- `tests/test_expert_iteration.py`

## 9. 未来 pipeline/GRPO 扩展

`GenerationVerificationEngine` 暴露 `run_sequential()`、`run_pipeline()` 和 `run_async()`。当前 `run_pipeline()` 明确使用顺序 fallback，`run_async()` 明确抛出 `NotImplementedError`。未来服务器 pipeline 可以在该层加入有界 producer/consumer，不需要修改状态机；未来 GRPO 在线 reward 可实现 async engine，并继续复用 SQLite cache、磁盘任务引用、memory guard 和 ResourceManager。此次没有引入 Ray、分布式调度或 GRPO 实现。
