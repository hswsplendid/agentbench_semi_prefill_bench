# AgentBench Semi-Prefill Compression Benchmark

在 vLLM 场景下测试 32k 上下文压缩（cw=32000, thr=30000），测量 semi-prefill 开销。

## 背景

Agent 多轮对话会累积大量 context token。上下文压缩（Context Compression）可以在不显著丢失信息的前提下控制 token 增长。但压缩后 prefix 不再可直接复用 KV cache，需要重新 prefill 一段窗口（**semi-prefill**），这成为在线瓶颈。

本 Benchmark 的目标是：
1. 在 AgentBench 任务上触发 32k 上下文压缩，并记录压缩比诊断指标
2. 记录完整的 ABC 段落（压缩前/后的 token 分布）
3. 测量 semi-prefill 的频率、长度和时延占比
4. 分析 agent workload 特征

## 压缩架构（ABC 三段式）

```
压缩前：[A - System Prefix] + [B1 - 完整历史]                    + [C1 - 最近保留段]
压缩后：[A - System Prefix] + [B2 - 压缩摘要] + [B2_ack(可选)]  + [C1 - 最近保留段]
```

| 段落 | 含义 | 说明 |
|------|------|------|
| **A** | System Prefix | 系统提示词，固定不变 |
| **B1** | 完整历史 | 压缩前的全部对话历史 |
| **B2** | 压缩摘要 | 由 LLM 生成的结构化摘要，替代 B1 |
| **C1** | 最近保留段 | 最近 N 条消息原样保留 |

## 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `CONTEXT_WINDOW` | 32000 | 总上下文 token 预算 |
| `RESERVE_TOKENS` | 2000 | 预留给响应的 token |
| 压缩阈值 | 30000 | cw - reserve，超过即触发压缩 |
| `KEEP_RECENT_TOKENS_BUDGET` | 2800 | C1 段目标长度（2000-3000） |
| `SUMMARY_MAX_TOKENS` | 1024 | 摘要最大 token |
| `TARGET_INITIAL_TOKENS` | 20000 | 初始 context 膨胀目标，低于阈值，避免开局压缩 |
| `MAX_TOKENS` | 3072 | 每次 LLM decode 上限 |
| LTP `max_turns` | 40 | 默认 LTP 对话轮数 |

### 压缩触发条件

```
shouldCompact(contextTokens, contextWindow, reserveTokens)
  => contextTokens > contextWindow - reserveTokens
  => contextTokens > 32000 - 2000 = 30000
```

当 total prompt tokens > 30000 时触发压缩。

### 预期触发形态

- 初始 prompt 约 21k tokens，低于 30k 阈值，不在第 1 轮压缩
- Agent/host prompt 要求更长的结构化 reasoning 与解释，使上下文随 turn 增长
- 目标是在 25-40 轮内至少触发一次压缩，典型触发率约 1/25 到 1/40，低于 1/5 或 1/6
- C1 保留预算为 2800 tokens，实际保留段目标范围为 2000-3000 tokens

## 数据集改造

LTP（Lateral Thinking Puzzle）谜题原始 system prompt 只有 ~330 tokens，不足以触发 32k 压缩。通过 `data_loader.py` 注入可压缩的 research reference 作为 B1，但保持初始上下文低于压缩阈值；随后通过更长的 agent/host decode 让对话历史自然增长并触发压缩。

```python
# 填充参考信息块
EXPANSION_FILLER = """
--- Reference Information Block {idx} ---
This is a reference document containing contextual information for the task.
{lines}
--- End of Block {idx} ---
"""
```

每个块包含 8 行确定性的 filler 文本，持续添加直到达到 `TARGET_INITIAL_TOKENS`。reference 会被拆成多轮 user/assistant chunk，避免单条超长消息被 C1 整体保留而无法压缩。

## 快速开始

### 前置条件

```bash
# vLLM 或 proxy 已运行
curl http://localhost:6003/v1/models   # proxy
curl http://localhost:8005/v1/models   # 或直接 vLLM

# 依赖
pip install openai transformers openpyxl
```

### 运行测试

```bash
# 压缩模式，1 个 LTP 样本（smoke test）
python run.py --mode compressed --tasks ltp --max-samples 1

# 同时运行 baseline + compressed
python run.py --mode both --tasks ltp --max-samples 4

# 指定 puzzle IDs
python run.py --mode compressed --puzzle-ids 0 1 2 3

# 自定义参数
python run.py --mode compressed \
    --context-window 32000 \
    --reserve-tokens 2000 \
    --keep-recent-tokens 2800 \
    --initial-tokens 20000 \
    --max-turns 40 \
    --max-tokens 3072 \
    --max-samples 2

# 仅分析已有结果
python run.py --analyze-only --mode compressed
python run.py --analyze-only --mode compressed --output-json
```

### 断点续传

程序在每个 turn 后自动保存 checkpoint 到 `results/{mode}/checkpoints/`。
重新运行相同参数会自动从断点恢复。

```bash
# 中断后重新运行，自动恢复
python run.py --mode compressed --puzzle-ids 0 1 2
# 输出: [RESUME] from turn 12/25, compressions=8
```

### 单独运行分析

```bash
python analyze.py --mode compressed
python analyze.py --mode both --json
```

## 输出文件

运行后在 `results/{mode}/` 下生成：

```
results/{mode}/
├── agentbench_results/
│   ├── results.jsonl          # 每样本的结果摘要
│   └── checkpoints/
│       └── puzzle_0.json      # 断点信息（完成后删除）
├── prompt_logs/
│   └── puzzle_0.jsonl         # 每次 query + 每次压缩的完整 messages
├── abc_segments/
│   └── puzzle_0_abc.json      # 每次压缩的 ABC 段落分析
├── traces/
│   └── puzzle_0.json          # 每轮对话 trace（agent/host 响应）
└── timing/
    └── puzzle_0.json           # 每次请求的时延记录（含分类）
```

### results.jsonl 格式

```json
{
  "id": 0,
  "mode": "compressed",
  "solved": true,
  "total_turns": 25,
  "final_token_count": 7832,
  "compression_count": 19,
  "game_progress": 0.75,
  "compressions": [{...}]
}
```

### abc_segments 格式

```json
{
  "turn": 6,
  "A_tokens": 330,
  "B1_tokens": 28950,
  "B2_tokens": 4822,
  "B2_to_B1_ratio": 0.1666,
  "C1_tokens_before": 0,
  "C1_tokens_after": 2588,
  "summary_generation_time_s": 15.3,
  "abc_segments": {
    "before": {"A": [...], "B1": [...], "C1": [...], ...},
    "after": {"A": [...], "B2": [...], "C1": [...], ...}
  }
}
```

### timing 格式

```json
{
  "turn": 0,
  "step": 0,
  "classification": "full_prefill",
  "prompt_tokens": 29100,
  "output_tokens": 256,
  "ttft_ms": 523.5,
  "total_ms": 1200.3,
  "decode_ms": 676.8
}
```

## 指标说明

### (a) Agent Workload Stats

| 指标 | 说明 |
|------|------|
| turns_per_task | 每个任务的总轮数 |
| new_tokens_per_step | 每步（LLM 调用）新增的 output token 数 |
| tool_calls_per_sample | 每个样本的总工具调用次数 |
| context_length | 每次请求的 prompt token 总量分布 |
| context_churn_ratio | output_tokens / prompt_tokens |
| semi_prefill_count | Semi-prefill 触发次数 |
| semi_prefill_tokens | 每次 semi-prefill 的 token 长度 |

### (b) Semi-prefill Ratio Analysis

一次性 agent 执行的时间分解：

```
                     prefill        semi-prefill     decode
                     ────────       ────────────     ──────
通常只有 1 次          ↑              反复发生         单步开销小
（首次请求）                          主要瓶颈
```

| 指标 | 说明 |
|------|------|
| prefill_pct | Full prefill 时间占总端到端的百分比 |
| semi_prefill_pct | 所有 semi-prefill 合计时间占比 |
| decode_pct | 所有增量 decode 合计时间占比 |
| summary_gen | LLM 生成压缩摘要的额外开销 |

### 计算公式

```
context_churn_ratio = output_tokens / prompt_tokens
semi_prefill_pct = sum(semi_prefill_total_ms) / total_e2e_ms * 100
B2_to_B1_ratio = B2_tokens / B1_tokens  (diagnostic only)
compression_saving = (B1 - B2) / B1 * 100%
```

## 代码结构

```
agentbench_semi_prefill_bench/
├── config.py              # 所有可调参数
├── data_loader.py         # LTP 数据加载 + context 膨胀
├── agent.py               # Agent 包装器（压缩 + ABC 记录 + checkpoint）
├── bench_handler.py       # Semi-prefill 处理器（流式调用 + 计时分类）
├── analyze.py             # 统计分析
├── run.py                 # 主入口（LTP + AgentBench 任务编排）
├── run_llama_ltp_prompts.py  # LTP 谜题 prompt
└── README.md              # 本文档
```
