"""
Semi-prefill benchmark handler for AgentBench compressed samples.

Manages the full lifecycle of a sample run:
- Initialization / resume from checkpoint
- Turn-by-turn execution with streaming LLM calls
- Compression + ABC segment recording
- Timing classification: full_prefill / semi_prefill / incremental
- Incremental log persistence

Based on bfcl_sp_bench_llama/bench_handler.py pattern.
"""

import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from transformers import AutoTokenizer

BENCH_ROOT = Path(__file__).parent.resolve()
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
sys.path.append(str(Path(__file__).parent.parent / "bfcl_compression_bench"))
from compressor import ContextCompressor, estimate_tokens, estimate_message_tokens, should_compact

import config as CFG


class AgentBenchSemiPrefillHandler:
    """Orchestrates a single sample's execution with full instrumentation.

    Handles:
    - LLM streaming with timing classification
    - Compression at token budget thresholds
    - ABC segment recording (before/after each compression)
    - Checkpoint/resume (within-sample)
    - Prompt logging (every query + every compression)
    """

    def __init__(self, preset: Optional[dict] = None):
        if preset is None:
            preset = {
                "context_window": CFG.CONTEXT_WINDOW,
                "reserve_tokens": CFG.RESERVE_TOKENS,
                "keep_recent_tokens": CFG.KEEP_RECENT_TOKENS_BUDGET,
                "summary_max_tokens": CFG.SUMMARY_MAX_TOKENS,
                "threshold": CFG.CONTEXT_WINDOW - CFG.RESERVE_TOKENS,
            }
        self.preset = preset

        model_info = CFG.MODEL_REGISTRY[CFG.DEFAULT_MODEL]
        self.model_path = model_info["model_path"]
        self.tokenizer_path = model_info.get("tokenizer_path", model_info["model_path"])

        api_base = CFG.PROXY_URL or CFG.VLLM_API_BASE
        self.client = OpenAI(base_url=api_base, api_key=CFG.VLLM_API_KEY)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True,
        )

        self.compressor = ContextCompressor(
            api_base=api_base,
            model_name=self.model_path,
            api_key=CFG.VLLM_API_KEY,
            summary_max_tokens=preset["summary_max_tokens"],
            context_window=preset["context_window"],
            reserve_tokens=preset["reserve_tokens"],
            keep_recent_tokens=preset["keep_recent_tokens"],
            quality_guard_enabled=CFG.QUALITY_GUARD_ENABLED,
            quality_guard_max_retries=CFG.QUALITY_GUARD_MAX_RETRIES,
            use_structured_instructions=CFG.USE_STRUCTURED_INSTRUCTIONS,
            preserved_recent_turns=CFG.PRESERVED_RECENT_TURNS,
        )

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _dirs(self, mode: str) -> dict:
        return {
            "result": CFG.get_result_dir(mode),
            "prompt_log": CFG.get_prompt_log_dir(mode),
            "trace": CFG.get_trace_dir(mode),
            "abc": CFG.get_abc_dir(mode),
            "timing": CFG.get_timing_dir(mode),
            "checkpoint": CFG.get_checkpoint_dir(mode),
        }

    def _ensure_dirs(self, dirs: dict):
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(self, mode: str, sample_id: str) -> Optional[dict]:
        path = CFG.get_checkpoint_dir(mode) / f"{sample_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_checkpoint(self, mode: str, state: dict, sample_id: str):
        payload = {
            "sample_id": sample_id,
            "turn_idx": state["turn_idx"],
            "step_idx": state["step_idx"],
            "messages": state["messages"],
            "timing_log": state["timing_log"],
            "abc_snapshots": state["abc_snapshots"],
            "prompt_snapshots": state["prompt_snapshots"],
            "turn_traces": state["turn_traces"],
            "previous_summary": state.get("previous_summary"),
            "last_was_compression": state.get("last_was_compression", False),
            "compression_count": state.get("compression_count", 0),
            "saved_at": time.time(),
        }
        path = CFG.get_checkpoint_dir(mode) / f"{sample_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    def clear_checkpoint(self, mode: str, sample_id: str):
        path = CFG.get_checkpoint_dir(mode) / f"{sample_id}.json"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def _message_text(self, msg: dict) -> str:
        parts = [str(msg.get("role", ""))]
        if msg.get("content"):
            parts.append(str(msg["content"]))
        if msg.get("tool_calls"):
            parts.append(json.dumps(msg["tool_calls"], ensure_ascii=False))
        if msg.get("tool_call_id"):
            parts.append(str(msg["tool_call_id"]))
        return "\n".join(parts)

    def message_tokens(self, messages: list[dict]) -> int:
        if not messages:
            return 0
        text = "\n".join(self._message_text(m) for m in messages)
        return len(self.tokenizer.tokenize(text))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_step(self, state: dict) -> str:
        if state["turn_idx"] == 0 and state["step_idx"] == 0:
            return "full_prefill"
        if state.get("last_was_compression"):
            return "semi_prefill"
        return "incremental"

    # ------------------------------------------------------------------
    # ABC segment splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_abc_before(messages: list[dict]) -> Tuple[list, list, list]:
        if not messages:
            return [], [], []
        if messages[0].get("role") == "system":
            return [messages[0]], list(messages[1:]), []
        return [], list(messages), []

    @staticmethod
    def _split_abc_after(compressed: list[dict]) -> Tuple[list, list, list]:
        if not compressed:
            return [], [], []
        if compressed[0].get("role") == "system":
            a = [compressed[0]]
            rest = list(compressed[1:])
        else:
            a = []
            rest = list(compressed)

        b2 = []
        idx = 0
        while idx < len(rest):
            msg = rest[idx]
            content = str(msg.get("content", ""))
            if msg.get("role") == "user" and "[Previous conversation summary]" in content:
                b2.append(msg)
                idx += 1
                continue
            if msg.get("role") == "assistant" and "context from the previous conversation" in content:
                b2.append(msg)
                idx += 1
                continue
            break
        return a, b2, rest[idx:]

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def _maybe_compress(self, state: dict) -> bool:
        """Run compression if threshold exceeded. Returns True if compressed."""
        total_tokens = self.message_tokens(state["messages"])
        if not should_compact(total_tokens, self.preset["context_window"], self.preset["reserve_tokens"]):
            return False

        before_messages = deepcopy(state["messages"])
        before_tokens = total_tokens

        compressed_messages, info = self.compressor.compress(
            before_messages,
            keep_recent_turns=1,
            previous_summary=state.get("previous_summary"),
            use_token_budget=True,
        )

        if info is None:
            return False

        state["messages"] = compressed_messages

        # Extract previous_summary for iterative compression
        for msg in compressed_messages:
            if msg.get("role") == "user" and "[Previous conversation summary]" in msg.get("content", ""):
                state["previous_summary"] = msg["content"]
                break

        # ABC breakdown
        a_before, b1_before, c1_before = self._split_abc_before(before_messages)
        a_after, b2_after, c1_after = self._split_abc_after(compressed_messages)

        # Refine: match C1 length in before
        keep_len = len(c1_after)
        if keep_len > 0 and len(b1_before) >= keep_len:
            c1_before = b1_before[-keep_len:]
            b1_before = b1_before[:-keep_len]

        post_tokens = self.message_tokens(compressed_messages)

        a_tokens = self.message_tokens(a_before)
        b1_tokens = self.message_tokens(b1_before)
        b2_tokens = self.message_tokens(b2_after)
        c1_before_tokens = self.message_tokens(c1_before)
        c1_after_tokens = self.message_tokens(c1_after)

        abc_event = {
            "turn": state["turn_idx"],
            "step": state["step_idx"],
            "pre_prompt_tokens": before_tokens,
            "post_prompt_tokens": post_tokens,
            "A_tokens": a_tokens,
            "B1_tokens": b1_tokens,
            "B2_tokens": b2_tokens,
            "B2_to_B1_ratio": round(b2_tokens / max(b1_tokens, 1), 4),
            "C1_tokens_before": c1_before_tokens,
            "C1_tokens_after": c1_after_tokens,
            "C2_tokens": b1_tokens,
            "summary_generation_time_s": info.get("summary_generation_time_s"),
            "abc_segments": {
                "before": {
                    "A": a_before,
                    "B1": b1_before,
                    "C1": c1_before,
                    "A_tokens": a_tokens,
                    "B1_tokens": b1_tokens,
                    "C1_tokens": c1_before_tokens,
                },
                "after": {
                    "A": a_after,
                    "B2": b2_after,
                    "C1": c1_after,
                    "A_tokens": a_tokens,
                    "B2_tokens": b2_tokens,
                    "C1_tokens": c1_after_tokens,
                },
            },
        }

        state["abc_snapshots"].append(abc_event)
        state["prompt_snapshots"].append({
            "type": "compression",
            "turn": state["turn_idx"],
            "pre_prompt_tokens": before_tokens,
            "post_prompt_tokens": post_tokens,
            "B2_plus_C1_tokens": b2_tokens + c1_after_tokens,
            "C1_tokens_after": c1_after_tokens,
            "summary_generation_time_s": abc_event["summary_generation_time_s"],
        })
        state["last_was_compression"] = True
        state["compression_count"] += 1
        return True

    # ------------------------------------------------------------------
    # Streaming LLM call
    # ------------------------------------------------------------------

    def _stream_query(self, state: dict) -> Tuple[dict, dict]:
        """Streaming LLM call with timing classification.

        Returns (response_dict, timing_dict).
        """
        self._maybe_compress(state)

        messages = deepcopy(state["messages"])
        classification = self._classify_step(state)
        prompt_tokens = self.message_tokens(messages)

        # Log query prompt
        self._append_prompt_log(
            state,
            {
                "type": "query",
                "turn": state["turn_idx"],
                "step": state["step_idx"],
                "classification": classification,
                "prompt_tokens": prompt_tokens,
                "messages": messages,
            },
        )
        state["prompt_snapshots"].append({
            "type": "query",
            "turn": state["turn_idx"],
            "step": state["step_idx"],
            "classification": classification,
            "prompt_tokens": prompt_tokens,
            "message_count": len(messages),
        })

        request = {
            "model": self.model_path,
            "messages": messages,
            "temperature": CFG.TEMPERATURE,
            "max_tokens": CFG.MAX_TOKENS,
            "stream": True,
        }

        max_attempts = CFG.STREAM_MAX_RETRIES + 1
        backoff_s = CFG.STREAM_RETRY_BACKOFF_S

        for attempt in range(max_attempts):
            t0 = time.perf_counter()
            ttft_ms = None
            content_parts = []
            usage = None
            stream = None

            try:
                try:
                    stream = self.client.chat.completions.create(
                        **request,
                        stream_options={"include_usage": True},
                    )
                except TypeError:
                    stream = self.client.chat.completions.create(**request)

                for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta is None:
                        continue
                    if getattr(delta, "content", None):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        content_parts.append(delta.content)

                total_ms = (time.perf_counter() - t0) * 1000.0
                if ttft_ms is None:
                    ttft_ms = total_ms

                content = "".join(content_parts)
                output_est = len(self.tokenizer.tokenize(content)) if content else 0
                input_tokens = prompt_tokens
                if usage:
                    input_tokens = getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens
                    output_est = getattr(usage, "completion_tokens", output_est) or output_est

                timing = {
                    "turn": state["turn_idx"],
                    "step": state["step_idx"],
                    "classification": classification,
                    "prompt_tokens": input_tokens,
                    "output_tokens": output_est,
                    "ttft_ms": round(ttft_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "decode_ms": round(max(total_ms - ttft_ms, 0.0), 2),
                }
                if attempt:
                    timing["stream_retries"] = attempt

                response = {
                    "content": content,
                    "input_tokens": input_tokens,
                    "output_tokens": output_est,
                }
                return response, timing

            except Exception as exc:
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()

                error_msg = str(exc)
                if "maximum context length" in error_msg.lower():
                    # Emergency compression
                    compressed = self._maybe_compress(state)
                    if compressed:
                        request["messages"] = state["messages"]
                        messages = state["messages"]
                        continue

                if attempt >= max_attempts - 1:
                    # Non-stream fallback
                    request_ns = dict(request)
                    request_ns["stream"] = False
                    t0 = time.perf_counter()
                    resp = self.client.chat.completions.create(**request_ns)
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    content = resp.choices[0].message.content or ""

                    timing = {
                        "turn": state["turn_idx"],
                        "step": state["step_idx"],
                        "classification": classification,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": len(self.tokenizer.tokenize(content)),
                        "ttft_ms": round(total_ms, 2),
                        "total_ms": round(total_ms, 2),
                        "decode_ms": 0.0,
                        "stream_retries": attempt + 1,
                        "fallback": "non_stream",
                    }
                    response = {"content": content, "input_tokens": prompt_tokens,
                               "output_tokens": len(self.tokenizer.tokenize(content))}
                    return response, timing

                if backoff_s > 0:
                    time.sleep(backoff_s)

        return {"content": ""}, {}

    # ------------------------------------------------------------------
    # Prompt logging
    # ------------------------------------------------------------------

    def _append_prompt_log(self, state: dict, entry: dict):
        """Append entry to prompt_log file and state."""
        sample_id = state.get("sample_id", "unknown")
        path = CFG.get_prompt_log_dir(state.get("mode", "compressed")) / f"{sample_id}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Persist all logs
    # ------------------------------------------------------------------

    def _persist_sample_logs(self, state: dict):
        sample_id = state["sample_id"]
        mode = state["mode"]

        with open(CFG.get_timing_dir(mode) / f"{sample_id}.json", "w", encoding="utf-8") as f:
            json.dump(state["timing_log"], f, indent=2, ensure_ascii=False)
        with open(CFG.get_abc_dir(mode) / f"{sample_id}_abc.json", "w", encoding="utf-8") as f:
            json.dump(state["abc_snapshots"], f, indent=2, ensure_ascii=False)
        with open(CFG.get_trace_dir(mode) / f"{sample_id}.json", "w", encoding="utf-8") as f:
            json.dump({
                "sample_id": sample_id,
                "turns": state["turn_traces"],
                "total_turns": state["turn_idx"],
                "steps": len(state["timing_log"]),
                "compressions": len(state["abc_snapshots"]),
                "compression_count": state.get("compression_count", 0),
                "resume_count": state.get("resume_count", 0),
            }, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Initialize state
    # ------------------------------------------------------------------

    def init_state(self, sample_id: str, mode: str,
                   checkpoint: Optional[dict] = None) -> dict:
        if checkpoint is not None:
            state = {
                "sample_id": sample_id,
                "mode": mode,
                "turn_idx": checkpoint["turn_idx"],
                "step_idx": checkpoint["step_idx"],
                "messages": checkpoint["messages"],
                "timing_log": checkpoint["timing_log"],
                "abc_snapshots": checkpoint["abc_snapshots"],
                "prompt_snapshots": checkpoint["prompt_snapshots"],
                "turn_traces": checkpoint.get("turn_traces", []),
                "previous_summary": checkpoint.get("previous_summary"),
                "last_was_compression": checkpoint.get("last_was_compression", False),
                "compression_count": checkpoint.get("compression_count", 0),
                "resume_count": checkpoint.get("resume_count", 0) + 1,
            }
            return state

        return {
            "sample_id": sample_id,
            "mode": mode,
            "turn_idx": 0,
            "step_idx": 0,
            "messages": [],
            "timing_log": [],
            "abc_snapshots": [],
            "prompt_snapshots": [],
            "turn_traces": [],
            "previous_summary": None,
            "last_was_compression": False,
            "compression_count": 0,
            "resume_count": 0,
        }
