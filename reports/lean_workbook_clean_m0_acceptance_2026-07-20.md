# Lean Workbook clean M0 与专家迭代验收报告

日期：2026-07-20  
Linux 项目：`/home/lean/projects/formal-theorem-prover`  
目标环境：Lean 4.29.1；mathlib `5e932f97dd25535344f80f9dd8da3aab83df0fe6`；environment hash `46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a`。

## 结论

大型专家迭代的发现与验证阶段可以恢复运行，但专家 QLoRA 必须继续受 `minimum_new_proofs_to_train=50` 硬门槛保护。本次 50 题 smoke 得到 41 条 verified Proof Bank 记录、解出 24 题；按每题最多选一个训练 proof 后只有 24 条，因此训练被正确跳过，没有伪造 M1/M2。

## 1. 原始数据 schema 与旧预处理丢失点

- Hugging Face 原始 parquet 共 25,214 个 tactic-step 行、13,517 个唯一题目。
- 原始字段只有 `answer/formal_statement/id/natural_language_statement/state_after/state_before/status/tactic`。没有 source module、imports、namespace、section、open scope、local notation、local attributes 或原文件 preceding context；变量和 hypothesis 只能从 proof state 恢复。
- 轨迹内部连续性检查为 0 个 transition mismatch，13,517 条完整轨迹可按行序重组。
- 旧丢失发生于 `data/preparation.py::sample_dataset -> normalize_records`：先对 tactic-step 行 shuffle，再按 ID 保留第一行，导致多步 proof 随机只剩一步；同时未过滤 `status=disproved`。

证据：`data/processed/lean_workbook_verified_v2/audit/raw_schema_report.json`。

## 2. 全量验证与 quarantine

- 唯一记录：13,517。
- 原始状态 proved：10,434；disproved：3,083。
- 在目标 Lean/Pantograph 环境中 proof verified：7,581，proved 候选验证率 72.6567%。
- quarantine：5,936。
- 可用于当前环境的 verified：7,581。
- 明确不可进入训练的 source-disproved：3,083。
- 缺上下文、抽取/语法/elaboration/tactic/unsolved/timeout 等无法安全自动修复：2,852。
- 明确版本不兼容：1。

错误 taxonomy：

| 类型 | 数量 |
|---|---:|
| source_status_disproved | 3,083 |
| missing_local_variable | 1,320 |
| reference_syntax_error | 704 |
| reference_tactic_failure | 277 |
| reference_unsolved_goals | 196 |
| reference_elaboration_error | 192 |
| missing_import | 131 |
| timeout | 23 |
| missing_hypothesis | 6 |
| source_extraction_corruption | 3 |
| version_incompatible_identifier | 1 |

证据：`audit/audit_summary.json`、`audit/error_taxonomy.json`、`audit/error_examples/`。

## 3. verified_v2 数据契约

- train 3,000；eval 160；discovery 700；verified reserve 3,718；monitor 32；benchmark 96。
- 所有角色 statement overlap 为 0。
- train/eval：`statement_verified=true`、`proof_verified=true`、`pantograph_verified=true`，未验证数为 0。
- discovery：700/700 statement 与 reference proof 均已验证，但训练/生成 prompt 不包含 reference proof。
- monitor/benchmark 是 statement-only，永不进入 Proof Bank。
- attestation 绑定 assembled source hash、environment hash、assembler v2、normalization v2。
- 3 条超过 tokenizer 1024 上限的 verified 记录只从训练候选中排除，仍保留在验证审计中。
- Lean Workbook 本身未做盲目版本替换；monitor/benchmark 只使用三个显式、版本化的 minif2f migration rule。

## 4. Golden 与 proof 分布

- Golden reference：train/eval/verified-extra 各 10，30/30 通过；statement 30/30 唯一、proof 30/30 唯一；覆盖 simp 6、norm_num 2、linarith 19、多步 9。
- 最终 train：2,521 个唯一 normalized proof，479 个重复记录；单 tactic 1,200/3,000=40%，多步 1,800。
- 相同 proof 上限 50；top-10/top-20/top-50 proof coverage 分别为 7.70%/9.97%/13.63%。
- proof token：p50 12、p90 32、p95 42、p99 63、max 269、mean 15.48。

## 5. 旧 M0 与旧 generation

- 旧 M0 保留并标记为 `legacy_unverified_data_m0`，不作为正式 anchor。
- 原始 1000 discovery 与 384 benchmark 文件 hash 重验证前后相同。
- 修复后 discovery：19/1000=1.90%；benchmark：2/384=0.5208%；合计 21/1384=1.5173%。
- 旧输出仍有 discovery length 715/1000、benchmark length 276/384，证明旧 M0 的 EOS/重复问题真实存在。

## 6. clean M0 训练

- 从原始 `models/Qwen2.5-1.5B-Instruct` 开始，不继承旧 M0。
- verified train/eval：3,000/160；所有 attestation 有效。
- QLoRA：NF4 double quant、BF16、r=32、alpha=64、dropout=0.05、LR=5e-5、batch=1、grad accumulation=16、1 epoch、max length=1024。
- requested packing=true；completion-only loss 下 effective packing=false。
- 188 steps，训练 2,634 秒；train loss 0.8113；eval loss 0.3515；eval mean token accuracy 0.9003。
- 117,677 个有效 completion label token；0 truncation、0 all-label-ignored、0 prompt-prefix mismatch、0 missing labeled EOS、0 unattested。
- adapter 已以 BF16 `safe_merge=true` 合并为 clean merged anchor，并记录输入权重 hash。

## 7. Checkpoint 功能门禁

- 固定 10 prompt、greedy、temperature=0。
- base 与 adapter：10/10 输出不同，确认 LoRA 实际生效。
- base+adapter 与 merged：首 token 10/10 一致，完整 token 8/10 一致。
- Transformers merged 与 vLLM merged：首 token 10/10 一致，完整 token 2/10 一致；BF16 后端数值路径造成后续 greedy 分叉，但实际绝对路径均指向 clean merged anchor，未回退 base。
- vLLM 日志确认 FlashInfer top-p/top-k sampler 实际启用。

## 8. 训练集记忆与 EOS/stop 门禁

固定 seed=42 的 50 条 verified train，输入已删除 proof/reference target，4 候选：

- Pass@1：4/50=8%。
- Pass@4：13/50=26%。
- extraction：200/200=100%。
- stop/eos：191/200=95.5%；length：9/200=4.5%。
- repetitive：9/200=4.5%；multiple proof：1/200=0.5%。
- tokenizer EOS 与 vLLM stop token 都是 151645。

门槛 `extraction>=90%`、`length<=30%` 通过。

## 9. Discovery bootstrap

从 700 条 reference-verified discovery 中选择 prompt 最短的 30 条；reference proof 只用于选择凭据，不写入模型输入。4 候选：

- Pass@1：5/30=16.67%。
- Pass@4：16/30=53.33%。
- extraction：120/120=100%。
- stop/eos：116/120=96.67%；length：4/120=3.33%。
- repetitive：3/120=2.5%；multiple proof：0。
- 双 Pantograph worker 无重启、无 fatal。

## 10. 50 题专家迭代 smoke

- precheck：direct Lean 3/3、Pantograph 3/3、reference round-trip 30/30、checkpoint/prompt/generation smoke 全部通过。
- 50 statement × 4 candidate = 200；54 个 successful candidate，24/50 题至少一个 proof 成功。
- candidate success rate 27%；success@4 48%。
- Proof Bank 41；Failure Bank 146。
- error distribution：elaboration 111、tactic 13、unsolved goals 22；无 term-mode assembler 错误。
- 两个常驻 worker 均被使用，restart count 均为 0；server startup 42.86 秒，跨 precheck 与正式轮次复用。
- 当轮每题最多一个训练 proof 后为 24，低于 50；记录 `current proofs 24 < minimum 50` 并跳过专家 QLoRA。

## 11. 资源、状态机与测试

- vLLM 与训练阶段使用隔离 GPU 子进程；Pantograph worker 由协调器提前创建并常驻。
- worker 使用 spawn、心跳、PID、启动耗时、重启计数、有界队列、当前任务重试；已修复重启边界下旧 generation 已完成 result 被丢弃导致主进程无限等待的问题，并增加 300 秒无结果进展保护。
- 主协调器结束后 expert/Pantograph/EngineCore/benchmark 进程为 0，GPU compute process 为 0。
- 状态词汇已包含 DATA_AUDIT、DATA_BUILD、PRECHECK、TRAIN_INITIAL、CHECKPOINT_VERIFY、TRAIN_MEMORIZATION、DISCOVERY_BOOTSTRAP、SELECT_POOL、GENERATE、VERIFY、UPDATE_BANKS、BUILD_DATASET、TRAIN_EXPERT、EVAL、MONITOR、FINALIZE、BENCHMARK；旧序列化值保留为别名。
- Linux 全量 pytest：89 passed、1 skipped、0 failed。
- 未删除旧 M0、旧 generation 或 Recovery-24.04；未生成伪 M1/M2。

## 12. 主要新增/修改文件

- 数据：`lean_prover/lean_training/data/contracts.py`、`audit.py`、`lean_workbook.py`、`verified_builder.py`。
- 验证：`lean_prover/lean_training/verification/pool.py`、`schema.py`、`pantograph.py`。
- 流程：`lean_prover/lean_training/evaluation/benchmark.py`、`expert_iteration/precheck.py`、`schemas.py` 及既有 expert coordinator/isolated stage。
- 脚本：`build_verified_lean_workbook.py`、`verify_statement_datasets.py`、`verify_golden_reference.py`、`reverify_existing_generations.py`、`mark_legacy_m0.py`、`merge_lora_adapter.py`、`checkpoint_equivalence.py`、`build_proof_free_gate_dataset.py`。
- 配置：`configs/expert_iteration.clean_m0_smoke50.yaml`。
- 测试：`tests/test_lean_training_verified_data.py`、`test_lean_training_benchmark_storage.py`、`test_expert_iteration.py`。

## 13. 放行建议

允许恢复大型专家迭代的发现/验证阶段。保持 clean merged M0 为 fixed anchor、双常驻 Pantograph worker、隔离 vLLM/QLoRA 子进程、FlashInfer sampler 和 50-proof 训练阈值。若大型首轮仍不足 50 个题级训练 proof，应按配置扩大 discovery pool 或停止，不应降低验证或训练门槛。
