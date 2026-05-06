"""
Configuration for AgentBench Semi-Prefill Compression Benchmark.

Centralizes all tuneable parameters and paths.
"""

import os
from pathlib import Path

# ============================================================
# Project paths
# ============================================================
BENCH_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = BENCH_ROOT / "results"
AGENTBENCH_ROOT = Path("/root/agentbench/AgentBench")
COMPRESSOR_ROOT = Path("/root/bfcl_compression_bench")

# ============================================================
# Model configuration
# ============================================================
MODEL_REGISTRY = {
    "Qwen3-30B-A3B": {
        "model_path": "/root/share/models/Qwen3-30B-A3B",
        "tokenizer_path": "/root/share/models/Qwen3-30B-A3B",
    },
    "Qwen3-235B-A22B": {
        "model_path": "/root/share/models/Qwen3-235B-A22B",
        "tokenizer_path": "/root/share/models/Qwen3-235B-A22B",
    },
    "Llama-3.3-70B-Instruct": {
        "model_path": "/root/share/models/Llama-3.3-70B-Instruct",
        "tokenizer_path": "/root/share/models/Llama-3.3-70B-Instruct",
    },
    "GLM-4-9B-0414": {
        "model_path": "/root/share/models/GLM-4-9B-0414",
        "tokenizer_path": "/root/share/models/GLM-4-9B-0414",
    },
}

DEFAULT_MODEL = "Llama-3.3-70B-Instruct"

# ============================================================
# vLLM / Proxy endpoints
# ============================================================
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8005/v1")
PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:6003/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# ============================================================
# Compression settings (32k target: cw=32000, threshold=30000)
# ============================================================

COMPRESS_MODE = "token_budget"
CONTEXT_WINDOW = 32000       # cw: total context budget
RESERVE_TOKENS = 2000        # reserve for response
KEEP_RECENT_TOKENS_BUDGET = 2800  # C1 segment target: 2000-3000
SUMMARY_MAX_TOKENS = 1024

# Effective threshold: CONTEXT_WINDOW - RESERVE_TOKENS = 30000
# Compression triggers when total_tokens > 30000
# B2/B1 is recorded as a diagnostic metric, not as a pass/fail gate.

QUALITY_GUARD_ENABLED = False
QUALITY_GUARD_MAX_RETRIES = 0
USE_STRUCTURED_INSTRUCTIONS = True
PRESERVED_RECENT_TURNS = 1

# ============================================================
# Dataset expansion
# ============================================================

# Target initial token count (A + B1). Keep this safely below the 30000
# compression threshold so compaction is triggered by dialogue growth, not at
# turn 1.
TARGET_INITIAL_TOKENS = 20000

# Repeat filler to bloat initial context
EXPANSION_FILLER = (
    "\n\n--- Reference Information Block {idx} ---\n"
    "This is a reference document containing contextual information for the task.\n"
    "The following are example interactions from a similar task domain:\n"
    "{lines}\n"
    "--- End of Block {idx} ---"
)

EXPANSION_LINES_PER_BLOCK = 8
EXPANSION_LINE_TEMPLATE = (
    "Query {n}: The system processed request #{n} with parameters "
    "[alpha={alpha}, beta={beta}, gamma={gamma}] resulting in status {status}. "
    "Execution path: step_{a} -> step_{b} -> step_{c} (duration: {dur}ms). "
    "Memory consumed: {mem}MB, throughput: {tp} req/s."
)

# ============================================================
# Task categories
# ============================================================

TASK_CATEGORIES = {
    "ltp": {
        "task_name": "ltp-std",
        "description": "Lateral Thinking Puzzles (logic reasoning, no Docker needed)",
        "data_file": AGENTBENCH_ROOT / "data" / "lateralthinkingpuzzle" / "standard.xlsx",
        "max_turns": 25,
        "standalone": True,   # can run without AgentBench task server
    },
    "os": {
        "task_name": "os-std",
        "description": "Operating System interaction (Docker required)",
        "data_file": AGENTBENCH_ROOT / "data" / "os_interaction" / "data",
        "max_turns": 8,
        "standalone": False,
    },
    "db": {
        "task_name": "dbbench-std",
        "description": "Database SQL query generation (Docker required)",
        "data_file": AGENTBENCH_ROOT / "data" / "dbbench",
        "max_turns": 15,
        "standalone": False,
    },
}

DEFAULT_TASKS = ["ltp"]

# ============================================================
# Output directories
# ============================================================

def get_result_dir(mode: str) -> Path:
    return RESULTS_DIR / mode / "agentbench_results"

def get_prompt_log_dir(mode: str) -> Path:
    return RESULTS_DIR / mode / "prompt_logs"

def get_trace_dir(mode: str) -> Path:
    return RESULTS_DIR / mode / "traces"

def get_abc_dir(mode: str) -> Path:
    return RESULTS_DIR / mode / "abc_segments"

def get_timing_dir(mode: str) -> Path:
    return RESULTS_DIR / mode / "timing"

def get_checkpoint_dir(mode: str) -> Path:
    return RESULTS_DIR / mode / "checkpoints"

def get_analysis_dir() -> Path:
    return RESULTS_DIR / "analysis"

# ============================================================
# Model-specific settings
# ============================================================
TEMPERATURE = 0.0
MAX_TOKENS = 3072
STREAM_MAX_RETRIES = 2
STREAM_RETRY_BACKOFF_S = 1.0

# ============================================================
# Batch and resume
# ============================================================
BATCH_SIZE = 4
SWAP_WARN_THRESHOLD_MB = 7000
