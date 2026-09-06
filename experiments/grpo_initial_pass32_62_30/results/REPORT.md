# GRPO vs SFT epoch-2：miniF2F-test pass@32 评估报告

## 结论摘要

本报告仅覆盖 miniF2F-test 的244题；每题固定生成32次。GRPO与SFT使用相同的vLLM采样合同；GRPO证明由绑定imports/open/statement/proof的Lean CLI逐采样校验，SFT沿用此前冻结的Pantograph收据。

## Pass@k

| k | GRPO（无偏估计） | SFT（无偏估计） | GRPO实际前缀 | SFT实际前缀 | 前缀差值 | 配对bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37.86% | 26.75% | 91/244 (37.30%) | 68/244 (27.87%) | +9.43 pp | [+4.51, +14.34] pp |
| 4 | 49.04% | 37.15% | 122/244 (50.00%) | 94/244 (38.52%) | +11.48 pp | [+6.97, +15.98] pp |
| 8 | 53.75% | 41.40% | 132/244 (54.10%) | 101/244 (41.39%) | +12.70 pp | [+8.20, +17.62] pp |
| 16 | 58.18% | 45.29% | 145/244 (59.43%) | 112/244 (45.90%) | +13.52 pp | [+8.61, +18.85] pp |
| 32 | 62.30% | 48.77% | 152/244 (62.30%) | 119/244 (48.77%) | +13.52 pp | [+9.02, +18.44] pp |

## 生成与编译质量

| 指标 | GRPO | SFT epoch-2 |
|---|---:|---:|
| 编译成功尝试 | 2956/7808 (37.86%) | 2089/7808 (26.75%) |
| 题内精确重复尝试率 | 42.07% | 20.84% |
| 题内成对碰撞率 | 21.90% | 8.26% |
| 出现重复的题数 | 197/244 | 137/244 |
| 严格幻觉API尝试率 | 811/7808 (10.39%) | 1706/7808 (21.85%) |
| import/namespace前缀污染 | 0 | 0 |
| 达到2048 token上限 | 194 | 234 |

严格幻觉API只计Lean明确报告的unknown identifier/constant/declaration/tactic/field；错误投影、类型不匹配、策略失败等另列，不混入该指标。

## 错误类型（尝试数）

```json
{
  "GRPO": {
    "hallucinated_api_strict": 811,
    "invalid_projection": 14,
    "other_lean_failure": 1109,
    "success": 2956,
    "syntax_or_parser": 284,
    "tactic_or_unsolved_goal": 2107,
    "timeout": 20,
    "type_or_elaboration": 507
  },
  "SFT": {
    "hallucinated_api_strict": 1706,
    "invalid_projection": 31,
    "other_lean_failure": 530,
    "success": 2089,
    "syntax_or_parser": 372,
    "tactic_or_unsolved_goal": 2284,
    "timeout": 8,
    "type_or_elaboration": 788
  }
}
```

## 资源

```json
{
  "samples": 584,
  "per_gpu": {
    "0": {
      "memory_used_mib_mean": 27699.739726027397,
      "memory_used_mib_peak": 29692.0,
      "utilization_mean_pct": 39.50684931506849,
      "utilization_peak_pct": 57.0
    },
    "1": {
      "memory_used_mib_mean": 26886.31506849315,
      "memory_used_mib_peak": 29692.0,
      "utilization_mean_pct": 45.25342465753425,
      "utilization_peak_pct": 94.0
    },
    "2": {
      "memory_used_mib_mean": 28106.45205479452,
      "memory_used_mib_peak": 29692.0,
      "utilization_mean_pct": 39.23287671232877,
      "utilization_peak_pct": 61.0
    },
    "3": {
      "memory_used_mib_mean": 26072.904109589042,
      "memory_used_mib_peak": 29692.0,
      "utilization_mean_pct": 38.02739726027397,
      "utilization_peak_pct": 62.0
    }
  }
}
```

## 口径说明

- 主要 pass@k 同时给出标准无偏估计和冻结样本顺序的实际前缀命中率；pass@32两者相同。
- 重复生成率按同一道题32个规范化proof中的精确重复计算；另给成对碰撞率，避免单一高频模板被低估。
- 所有含sorry/admit的结果均按失败处理。
- 本版修复了旧直接编译回退遗漏imports/open前缀的问题；按题分组仅复用Mathlib加载，每个采样由namespace与到达哨兵隔离，哨兵缺失时自动单条复核。旧v1统计已作废。
