"""
Compression-aware agent with ABC segment recording and checkpoint/resume.

Records for every inference call:
  - query: full messages sent to LLM
  - compression: pre/post messages + ABC breakdown
  - timing: ttft_ms, decode_ms, total_ms, classification

Supports LTP standalone mode and AgentBench task-client mode.
"""

import json
import os
import re
import sys
import time
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# Agent interface (mirror AgentBench AgentClient to avoid heavy imports)
class AgentClient:
    def __init__(self, *args, **kwargs):
        pass

    def inference(self, history: List[dict]) -> str:
        raise NotImplementedError()


# Import compressor without shadowing this package's config.py.
BENCH_ROOT = Path(__file__).parent.resolve()
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
sys.path.append(str(Path(__file__).parent.parent / "bfcl_compression_bench"))
from compressor import ContextCompressor, estimate_tokens, estimate_message_tokens, should_compact

import config as CFG


class BenchmarkAgent(AgentClient):
    """Agent with compression, ABC recording, checkpoint, and timing.

    Thread-local state ensures correct metrics across parallel sample runs.
    """

    _tls = threading.local()

    def __init__(
        self,
        mode: str,          # "baseline" or "compressed"
        api_base: str,
        model_name: str,
        tokenizer,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = CFG.MAX_TOKENS,
        result_dir: Path = None,
        prompt_log_dir: Path = None,
        trace_dir: Path = None,
        abc_dir: Path = None,
        timing_dir: Path = None,
        checkpoint_dir: Path = None,
        context_window: int = CFG.CONTEXT_WINDOW,
        reserve_tokens: int = CFG.RESERVE_TOKENS,
        keep_recent_tokens: int = CFG.KEEP_RECENT_TOKENS_BUDGET,
        summary_max_tokens: int = CFG.SUMMARY_MAX_TOKENS,
    ):
        super().__init__()
        self.mode = mode
        self.api_base = api_base
        self.model_name = model_name
        self.api_key = api_key
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.summary_max_tokens = summary_max_tokens

        # Directories
        self.result_dir = Path(result_dir) if result_dir else CFG.get_result_dir(mode)
        self.prompt_log_dir = Path(prompt_log_dir) if prompt_log_dir else CFG.get_prompt_log_dir(mode)
        self.trace_dir = Path(trace_dir) if trace_dir else CFG.get_trace_dir(mode)
        self.abc_dir = Path(abc_dir) if abc_dir else CFG.get_abc_dir(mode)
        self.timing_dir = Path(timing_dir) if timing_dir else CFG.get_timing_dir(mode)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CFG.get_checkpoint_dir(mode)

        for d in [self.result_dir, self.prompt_log_dir, self.trace_dir,
                  self.abc_dir, self.timing_dir, self.checkpoint_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # LLM client
        self._llm = OpenAI(base_url=api_base, api_key=api_key)

        # Compressor
        if mode == "compressed":
            self._compressor = ContextCompressor(
                api_base=api_base,
                model_name=model_name,
                api_key=api_key,
                summary_max_tokens=summary_max_tokens,
                context_window=context_window,
                reserve_tokens=reserve_tokens,
                keep_recent_tokens=keep_recent_tokens,
                quality_guard_enabled=CFG.QUALITY_GUARD_ENABLED,
                quality_guard_max_retries=CFG.QUALITY_GUARD_MAX_RETRIES,
                use_structured_instructions=CFG.USE_STRUCTURED_INSTRUCTIONS,
                preserved_recent_turns=CFG.PRESERVED_RECENT_TURNS,
            )
        else:
            self._compressor = None

    # ------------------------------------------------------------------
    # Thread-local state
    # ------------------------------------------------------------------

    def _get_tls(self):
        tls = self._tls
        if not hasattr(tls, "initialized"):
            tls.initialized = False
        return tls

    def _init_sample_state(self):
        tls = self._tls
        tls.initialized = True
        tls.turn_idx = 0
        tls.step_idx = 0
        tls.previous_summary = None
        tls.prompt_entries = []      # log entries (query + compression)
        tls.timing_log = []           # per-request timing
        tls.abc_snapshots = []        # per-compression ABC breakdown
        tls.turn_traces = []          # per-turn summary
        tls.compression_count = 0
        tls.cumulative_tool_calls = 0
        tls.last_was_compression = False

    def _ensure_initialized(self):
        tls = self._get_tls()
        if not tls.initialized:
            self._init_sample_state()

    # ------------------------------------------------------------------
    # Checkpoint / Resume
    # ------------------------------------------------------------------

    def save_checkpoint(self, sample_id: str, state: dict):
        payload = {
            "sample_id": sample_id,
            "turn_idx": state["turn_idx"],
            "step_idx": state.get("step_idx", 0),
            "messages": state["messages"],
            "previous_summary": state.get("previous_summary"),
            "prompt_entries": state["prompt_entries"],
            "timing_log": state["timing_log"],
            "abc_snapshots": state["abc_snapshots"],
            "turn_traces": state["turn_traces"],
            "compression_count": state["compression_count"],
            "cumulative_tool_calls": state.get("cumulative_tool_calls", 0),
            "last_was_compression": state.get("last_was_compression", False),
            "saved_at": time.time(),
        }
        path = self.checkpoint_dir / f"{sample_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    def load_checkpoint(self, sample_id: str) -> Optional[dict]:
        path = self.checkpoint_dir / f"{sample_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def clear_checkpoint(self, sample_id: str):
        path = self.checkpoint_dir / f"{sample_id}.json"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Persist logs
    # ------------------------------------------------------------------

    def _persist_logs(self, sample_id: str, state: dict):
        # Prompt log in the same per-turn JSONL shape as the reference
        # semi_prefill_bench logs.
        log_file = self.prompt_log_dir / f"{sample_id}.jsonl"
        with open(log_file, "w", encoding="utf-8") as f:
            for turn in state.get("turn_traces", []):
                entry = {
                    "turn": turn.get("turn"),
                    "compressed_this_turn": turn.get("compressed_this_turn", False),
                    "input_tokens_est": turn.get("input_tokens_est", turn.get("tokens_at_query")),
                    "agent_prompt": turn.get("agent_prompt", []),
                    "agent_response": turn.get("agent_response", turn.get("agent", "")),
                    "host_prompt": turn.get("host_prompt", []),
                    "host_response": turn.get("host_response", turn.get("host", "")),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # ABC segments
        abc_file = self.abc_dir / f"{sample_id}_abc.json"
        with open(abc_file, "w", encoding="utf-8") as f:
            json.dump(state["abc_snapshots"], f, indent=2, ensure_ascii=False)

        # Timing
        timing_file = self.timing_dir / f"{sample_id}.json"
        with open(timing_file, "w", encoding="utf-8") as f:
            json.dump(state["timing_log"], f, indent=2, ensure_ascii=False)

        # Trace
        trace_file = self.trace_dir / f"{sample_id}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump({
                "sample_id": sample_id,
                "turns": state["turn_traces"],
                "total_turns": state["turn_idx"],
                "steps": len(state["timing_log"]),
                "compressions": len(state["abc_snapshots"]),
                "tool_calls": state.get("cumulative_tool_calls", 0),
            }, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def message_tokens(self, messages: list[dict]) -> int:
        if not messages:
            return 0
        text = "\n".join(self._message_text(m) for m in messages)
        return len(self.tokenizer.tokenize(text))

    def _message_text(self, msg: dict) -> str:
        parts = [str(msg.get("role", ""))]
        if msg.get("content"):
            parts.append(str(msg["content"]))
        if msg.get("tool_calls"):
            parts.append(json.dumps(msg["tool_calls"], ensure_ascii=False))
        if msg.get("tool_call_id"):
            parts.append(str(msg["tool_call_id"]))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Step classification
    # ------------------------------------------------------------------

    def _classify_step(self, state: dict) -> str:
        if state["turn_idx"] == 0 and state["step_idx"] == 0:
            return "full_prefill"
        if state.get("last_was_compression"):
            return "semi_prefill"
        return "incremental"

    # ------------------------------------------------------------------
    # ABC segment extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _split_abc_before(messages: list[dict]) -> Tuple[list, list, list]:
        """Split messages before compression: A=system, B1=history, C1=recent."""
        if not messages:
            return [], [], []
        if messages[0].get("role") == "system":
            return [messages[0]], list(messages[1:]), []
        return [], list(messages), []

    @staticmethod
    def _split_abc_after(compressed: list[dict]) -> Tuple[list, list, list]:
        """Split messages after compression: A=system, B2=summary, C1=recent."""
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

    def _maybe_compress(self, state: dict) -> Optional[dict]:
        """Apply compression if token budget exceeded. Returns compression_event or None."""
        if self.mode != "compressed" or self._compressor is None:
            return None

        total_tokens = self.message_tokens(state["messages"])
        if not should_compact(total_tokens, self.context_window, self.reserve_tokens):
            return None

        before_messages = deepcopy(state["messages"])
        before_tokens = total_tokens

        compressed_messages, info = self._compressor.compress(
            before_messages,
            keep_recent_turns=1,
            previous_summary=state.get("previous_summary"),
            use_token_budget=True,
        )

        if info is None:
            return None

        state["messages"] = compressed_messages

        # Update previous_summary for iterative compression
        for msg in compressed_messages:
            if msg.get("role") == "user" and "[Previous conversation summary]" in msg.get("content", ""):
                state["previous_summary"] = msg["content"]
                break

        # ABC breakdown
        a_before, b1_before, c1_before = self._split_abc_before(before_messages)
        a_after, b2_after, c1_after = self._split_abc_after(compressed_messages)

        # Refine: extract C1 from B1 for before
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
        c2_tokens = b1_tokens  # C2 = full history being summarized

        compression_event = {
            "turn": state["turn_idx"],
            "step": state["step_idx"],
            "compression_info": info,
            "pre_prompt_tokens": before_tokens,
            "post_prompt_tokens": post_tokens,
            "A_tokens": a_tokens,
            "B1_tokens": b1_tokens,
            "B2_tokens": b2_tokens,
            "B2_to_B1_ratio": round(b2_tokens / max(b1_tokens, 1), 4),
            "C1_tokens_before": c1_before_tokens,
            "C1_tokens_after": c1_after_tokens,
            "C2_tokens": c2_tokens,
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

        state["abc_snapshots"].append(compression_event)
        state["compression_count"] += 1
        state["last_was_compression"] = True

        # Log compression entry
        comp_entry = {
            "type": "compression",
            "test_id": state.get("sample_id", "unknown"),
            "turn": state["turn_idx"],
            "pre_message_count": len(before_messages),
            "post_message_count": len(compressed_messages),
            "pre_tokens_est": before_tokens,
            "post_tokens_est": post_tokens,
            "compression_info": info,
            "abc_segments": compression_event["abc_segments"],
            "compression_event_summary": {
                k: compression_event[k]
                for k in ["A_tokens", "B1_tokens", "B2_tokens",
                           "B2_to_B1_ratio", "C1_tokens_before",
                           "C1_tokens_after", "C2_tokens",
                           "summary_generation_time_s"]
            },
        }
        state["prompt_entries"].append(comp_entry)

        return compression_event

    # ------------------------------------------------------------------
    # Streaming inference with timing
    # ------------------------------------------------------------------

    def _stream_llm(self, messages: list[dict], state: dict) -> Tuple[str, dict]:
        """Stream LLM call with timing. Returns (content, timing_dict)."""
        classification = self._classify_step(state)
        prompt_tokens = self.message_tokens(messages)

        # Log query
        query_entry = {
            "type": "query",
            "test_id": state.get("sample_id", "unknown"),
            "turn": state["turn_idx"],
            "step": state["step_idx"],
            "classification": classification,
            "messages": deepcopy(messages),
            "message_count": len(messages),
            "prompt_tokens": prompt_tokens,
            "compressed_this_step": state.get("last_was_compression", False),
        }
        state["prompt_entries"].append(query_entry)

        tokenizer_safety_margin = 1024
        available_decode_tokens = max(
            256, self.context_window - prompt_tokens - tokenizer_safety_margin
        )
        request_max_tokens = min(self.max_tokens, available_decode_tokens)

        request = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": request_max_tokens,
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
                    stream = self._llm.chat.completions.create(
                        **request,
                        stream_options={"include_usage": True},
                    )
                except TypeError:
                    stream = self._llm.chat.completions.create(**request)

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
                output_text = content
                output_est_tokens = len(self.tokenizer.tokenize(output_text)) if output_text else 0
                input_tokens = prompt_tokens
                if usage:
                    input_tokens = getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens
                    output_est_tokens = getattr(usage, "completion_tokens", output_est_tokens) or output_est_tokens

                timing = {
                    "turn": state["turn_idx"],
                    "step": state["step_idx"],
                    "classification": classification,
                    "prompt_tokens": input_tokens,
                    "requested_max_tokens": request_max_tokens,
                    "output_tokens": output_est_tokens,
                    "ttft_ms": round(ttft_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "decode_ms": round(max(total_ms - ttft_ms, 0.0), 2),
                }
                if attempt:
                    timing["stream_retries"] = attempt

                state["timing_log"].append(timing)
                return content, timing

            except Exception as exc:
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()

                # On context overflow, try compressing and retrying
                error_msg = str(exc)
                is_overflow = (
                    "maximum context length" in error_msg
                    or "requested" in error_msg
                )
                if is_overflow and self._compressor is not None:
                    self._maybe_compress(state)
                    request["messages"] = state["messages"]
                    messages = state["messages"]
                    prompt_tokens = self.message_tokens(messages)
                    available_decode_tokens = max(
                        256, self.context_window - prompt_tokens - tokenizer_safety_margin
                    )
                    request_max_tokens = min(self.max_tokens, available_decode_tokens)
                    request["max_tokens"] = request_max_tokens
                    continue

                if attempt >= max_attempts - 1:
                    # Fallback: non-stream
                    request_ns = dict(request)
                    request_ns["stream"] = False
                    t0 = time.perf_counter()
                    resp = self._llm.chat.completions.create(**request_ns)
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    content = resp.choices[0].message.content or ""
                    timing = {
                        "turn": state["turn_idx"],
                        "step": state["step_idx"],
                        "classification": classification,
                        "prompt_tokens": prompt_tokens,
                        "requested_max_tokens": request_max_tokens,
                        "requested_max_tokens": request_max_tokens,
                        "output_tokens": len(self.tokenizer.tokenize(content)),
                        "ttft_ms": round(total_ms, 2),
                        "total_ms": round(total_ms, 2),
                        "decode_ms": 0.0,
                        "stream_retries": attempt + 1,
                        "fallback": "non_stream",
                    }
                    state["timing_log"].append(timing)
                    return content, timing

                if backoff_s > 0:
                    time.sleep(backoff_s)

        return "", {}

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def inference(self, history: List[dict]) -> str:
        """Called by AgentBench task runner (AgentClient interface)."""
        self._ensure_initialized()
        tls = self._tls

        # Convert history to OpenAI format
        messages = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "agent":
                role = "assistant"
            messages.append({"role": role, "content": content})

        # Update turn counter
        tls.turn_idx = sum(
            1 for m in history
            if m.get("role") == "user"
            and not str(m.get("content", "")).startswith("<tool_response>")
        )

        # Apply compression
        self._maybe_compress({"messages": messages, "turn_idx": tls.turn_idx,
                              "step_idx": tls.step_idx,
                              "last_was_compression": False,
                              "sample_id": str(id(threading.current_thread()))})

        # Stream call
        content, timing = self._stream_llm(messages, {"turn_idx": tls.turn_idx,
                                                       "step_idx": tls.step_idx})
        return content

    # ------------------------------------------------------------------
    # Standalone LTP inference (used by run.py for LTP task)
    # ------------------------------------------------------------------

    def run_ltp_turn(
        self,
        agent_messages: list[dict],
        host_system: str,
        state: dict,
    ) -> Tuple[str, str]:
        """Run one LTP turn: agent query → host response.

        Returns (agent_response, host_response).
        """
        # Compression before agent query
        compression_event = self._maybe_compress(state)

        # Agent query
        query_messages = state["messages"]
        agent_response, timing = self._stream_llm(query_messages, state)

        state["messages"].append({"role": "assistant", "content": agent_response})

        # Host response
        host_messages = [
            {"role": "system", "content": host_system},
            {"role": "user", "content": agent_response},
        ]
        host_response, host_timing = self._stream_llm(host_messages, state)

        state["messages"].append({"role": "user", "content": host_response})

        # Record turn trace
        turn_entry = {
            "turn": state["turn_idx"],
            "agent": agent_response[:500],
            "host": host_response[:200],
            "tokens_est": self.message_tokens(query_messages),
            "compressed_this_turn": compression_event is not None,
            "timing_agent": timing,
            "timing_host": host_timing,
        }
        if compression_event:
            turn_entry["compression_summary"] = {
                k: compression_event[k]
                for k in ["pre_prompt_tokens", "post_prompt_tokens",
                           "A_tokens", "B1_tokens", "B2_tokens",
                           "B2_to_B1_ratio", "C1_tokens_after",
                           "summary_generation_time_s"]
                if k in compression_event
            }
        state["turn_traces"].append(turn_entry)

        if state.get("last_was_compression"):
            state["last_was_compression"] = False

        state["turn_idx"] += 1

        return agent_response, host_response

    # ------------------------------------------------------------------
    # Sample lifecycle
    # ------------------------------------------------------------------

    def init_sample(self, sample_id: str, initial_messages: Optional[list[dict]] = None):
        """Initialize or resume sample state."""
        self._init_sample_state()
        tls = self._tls
        tls.sample_id = sample_id
        tls.step_idx = 0
        tls.turn_idx = 0

        ckpt = self.load_checkpoint(sample_id)
        if ckpt:
            tls.turn_idx = ckpt["turn_idx"]
            tls.step_idx = ckpt.get("step_idx", 0)
            tls.previous_summary = ckpt.get("previous_summary")
            tls.prompt_entries = ckpt.get("prompt_entries", [])
            tls.timing_log = ckpt.get("timing_log", [])
            tls.abc_snapshots = ckpt.get("abc_snapshots", [])
            tls.turn_traces = ckpt.get("turn_traces", [])
            tls.compression_count = ckpt.get("compression_count", 0)
            tls.cumulative_tool_calls = ckpt.get("cumulative_tool_calls", 0)
            tls.last_was_compression = ckpt.get("last_was_compression", False)
            tls.resume_count = ckpt.get("resume_count", 0) + 1

            if "messages" in ckpt:
                tls.sample_messages = ckpt["messages"]
        else:
            tls.resume_count = 0
            tls.sample_messages = deepcopy(initial_messages or [])

    def finalize_sample(self, sample_id: str):
        """Save all logs and clear checkpoint."""
        tls = self._tls
        state = {
            "sample_id": sample_id,
            "turn_idx": tls.turn_idx,
            "step_idx": tls.step_idx,
            "messages": tls.sample_messages,
            "previous_summary": tls.previous_summary,
            "prompt_entries": tls.prompt_entries,
            "timing_log": tls.timing_log,
            "abc_snapshots": tls.abc_snapshots,
            "turn_traces": tls.turn_traces,
            "compression_count": tls.compression_count,
            "cumulative_tool_calls": tls.cumulative_tool_calls,
            "last_was_compression": tls.last_was_compression,
        }
        self._persist_logs(sample_id, state)
        self.clear_checkpoint(sample_id)

    def get_sample_stats(self) -> dict:
        tls = self._tls
        if not tls.initialized:
            return {}
        return {
            "total_turns": tls.turn_idx,
            "total_steps": len(tls.timing_log),
            "compression_count": tls.compression_count,
            "abc_snapshots": tls.abc_snapshots,
            "resume_count": getattr(tls, "resume_count", 0),
        }
