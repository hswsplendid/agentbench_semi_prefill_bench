#!/usr/bin/env python3
"""
AgentBench Semi-Prefill Compression Benchmark — Main CLI.

Orchestrates standalone LTP (Lateral Thinking Puzzle) testing with:
- Baseline (no compression) and Compressed (context compression) modes
- Full prompt recording (ABC segments, trace, prompt log)
- Semi-prefill timing classification
- Checkpoint/resume (within and across samples)
- Workload statistics and semi-prefill ratio analysis

Usage:
    # LTP smoke test (1 sample, compressed)
    python run.py --mode compressed --tasks ltp --max-samples 1

    # LTP batch test
    python run.py --mode both --tasks ltp --max-samples 4

    # With custom model
    python run.py --mode compressed --model Qwen3-235B-A22B --tasks ltp

    # Analyze only
    python run.py --analyze-only --mode compressed
"""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

BENCH_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(BENCH_ROOT))
sys.path.append(str(Path("/root/bfcl_compression_bench")))

from openai import OpenAI
from transformers import AutoTokenizer

from compressor import should_compact

import config as CFG
from data_loader import load_ltp_puzzles, build_initial_messages, build_research_document, estimate_initial_token_distribution
from agent import BenchmarkAgent
from analyze import run_analysis
from run_llama_ltp_prompts import AGENT_SYSTEM_PROMPT, HOST_SYSTEM_PROMPT


# ============================================================
# Health checks
# ============================================================

def check_health(api_base: str = None, verbose: bool = True) -> tuple:
    url = api_base or CFG.PROXY_URL
    import urllib.request
    ok = True
    lines = []
    try:
        resp = urllib.request.urlopen(f"{url}/models", timeout=15)
        data = json.loads(resp.read())
        models = [m["id"] for m in data.get("data", [])]
        lines.append(f"  API {url}: OK ({', '.join(models[:2])})")
    except Exception as e:
        lines.append(f"  API {url}: FAIL ({e})")
        ok = False

    try:
        r = subprocess.run(["free", "-m"], capture_output=True, text=True)
        for l in r.stdout.splitlines():
            if "Swap" in l:
                parts = l.split()
                swap_mb = int(parts[2]) if len(parts) >= 3 else 0
                lines.append(f"  Swap used: {swap_mb}MB")
                if swap_mb > CFG.SWAP_WARN_THRESHOLD_MB:
                    lines.append(f"  *** SWAP WARNING ***")
    except:
        pass

    report = "\n".join(lines)
    if verbose:
        print(report)
    return ok, report


# ============================================================
# LTP puzzle runner
# ============================================================

AGENT_LONG_OUTPUT_CONTROL = """Benchmark control instruction for this turn:
- Continue the lateral-thinking puzzle from the latest host answer.
- Target 1800-2200 tokens for this turn if the remaining context allows it.
- Use exactly these sections: Known Facts, Remaining Mysteries, Reasoning, Question Strategy, My Question.
- Under Known Facts and Remaining Mysteries, write 5 detailed bullets each when evidence exists.
- Under Reasoning, write at least 3 substantial paragraphs covering surface observations, leading hypothesis, alternative hypothesis, and why the next question is high value.
- Ask exactly one yes/no question at the end. Do not give a final answer yet."""


def run_ltp_puzzle(
    puzzle: dict,
    mode: str,
    agent: BenchmarkAgent,
    tokenizer,
    max_turns: int = 25,
) -> dict:
    """Run a single LTP puzzle, return result entry.

    Records:
    - Full prompt log (query messages + compression events)
    - ABC segments (before/after each compression)
    - Timing log (ttft/decode/total + classification)
    - Trace (per-turn agent/host responses)
    - Checkpoint (per-turn for resume)
    """
    pid = puzzle["id"]
    story = puzzle["story"]
    answer = puzzle["answer"]
    answer_keys = puzzle.get("answer_keys", "")

    print(f"\n  Puzzle {pid}: {story[:80]}...")

    # Build system prompts
    agent_sys = AGENT_SYSTEM_PROMPT.format(story=story, max_turns=max_turns)
    host_sys = HOST_SYSTEM_PROMPT.format(
        story=story, answer=answer, answer_keys=answer_keys,
    )

    # Use research-document-injected messages from data_loader
    # (or fall back to bare system prompt if puzzle has no pre-built messages)
    if "messages" in puzzle and len(puzzle["messages"]) > 1:
        initial_messages = deepcopy(puzzle["messages"])
        research_tokens = sum(
            len(tokenizer.tokenize(m.get("content", "")))
            for m in initial_messages[1:]  # skip system
        )
        print(f"    [INIT] Using research-injected messages: "
              f"A={len(tokenizer.tokenize(initial_messages[0].get('content','')))} tok, "
              f"B1_research={research_tokens} tok")
    else:
        # Fallback: bare system prompt
        sys_content = puzzle.get("system_prompt", agent_sys)
        if CFG.TARGET_INITIAL_TOKENS > 0:
            sys_tokens = len(tokenizer.tokenize(sys_content))
            if sys_tokens < CFG.TARGET_INITIAL_TOKENS:
                from data_loader import build_initial_messages
                initial_messages = build_initial_messages(
                    sys_content,
                    target_initial_tokens=CFG.TARGET_INITIAL_TOKENS,
                    tokenizer=tokenizer,
                    include_research=True,
                )
            else:
                initial_messages = [{"role": "system", "content": sys_content}]
        else:
            initial_messages = [{"role": "system", "content": sys_content}]

    # Initialize agent state
    sample_id = f"puzzle_{pid}"
    agent.init_sample(sample_id, initial_messages)
    tls = agent._tls

    start_turn = tls.turn_idx  # may be >0 if resumed from checkpoint

    if start_turn == 0:
        # Fresh start — use initial messages
        tls.sample_messages = deepcopy(initial_messages)
        tls.previous_summary = False
        tls.compression_count = 0
    else:
        print(f"    [RESUME] from turn {start_turn}/{max_turns}, "
              f"messages={len(tls.sample_messages)}, compressions={tls.compression_count}")

    solved = False
    total_turns = start_turn

    for turn in range(start_turn, max_turns):
        total_turns = turn + 1
        tls.turn_idx = turn

        # Build state dict for agent methods
        state = {
            "messages": tls.sample_messages,
            "turn_idx": tls.turn_idx,
            "step_idx": 0,
            "previous_summary": tls.previous_summary,
            "prompt_entries": tls.prompt_entries,
            "timing_log": tls.timing_log,
            "abc_snapshots": tls.abc_snapshots,
            "turn_traces": tls.turn_traces,
            "compression_count": tls.compression_count,
            "last_was_compression": tls.last_was_compression,
            "sample_id": sample_id,
        }

        # ---- Compression (before agent query) ----
        if mode == "compressed" and agent._compressor is not None:
            total_tok = agent.message_tokens(state["messages"])
            do_compress = should_compact(
                total_tok, CFG.CONTEXT_WINDOW, CFG.RESERVE_TOKENS
            )
            print(f"    Turn {turn+1}/{max_turns} [tok={total_tok}, thr={CFG.CONTEXT_WINDOW - CFG.RESERVE_TOKENS}, compress={do_compress}]", end="", flush=True)

            if do_compress and turn >= 2:
                compression_event = agent._maybe_compress(state)
                if compression_event:
                    tls.sample_messages = state["messages"]
                    tls.previous_summary = state["previous_summary"]
                    tls.last_was_compression = True
                    print(f" [COMPRESS] {compression_event.get('pre_prompt_tokens')}->{compression_event.get('post_prompt_tokens')} "
                          f"B2/B1={compression_event.get('B2_to_B1_ratio'):.3f} "
                          f"C1={compression_event.get('C1_tokens_after')}", end="")
        else:
            print(f"    Turn {turn+1}/{max_turns}", end="", flush=True)

        # ---- Agent query ----
        query_messages = tls.sample_messages + [
            {"role": "user", "content": AGENT_LONG_OUTPUT_CONTROL}
        ]
        agent_prompt_for_log = deepcopy(query_messages)
        agent_response, timing = agent._stream_llm(query_messages, state)

        tls.sample_messages = state.get("messages", tls.sample_messages)
        tls.sample_messages.append({"role": "assistant", "content": agent_response})

        print(f" → {agent_response[:80]}...")

        # ---- Host response ----
        host_messages = [
            {"role": "system", "content": host_sys},
            {"role": "user", "content": agent_response},
        ]
        host_prompt_for_log = deepcopy(host_messages)
        host_response, host_timing = agent._stream_llm(host_messages, state)
        tls.sample_messages.append({"role": "user", "content": host_response})

        print(f"    Host: {host_response[:80]}")

        # ---- Record turn trace ----
        query_tokens = agent.message_tokens(query_messages)
        turn_entry = {
            "turn": turn,
            "agent": agent_response,
            "host": host_response,
            "tokens_at_query": query_tokens,
            "compressed_this_turn": state.get("last_was_compression", False),
            "agent_prompt": agent_prompt_for_log,
            "agent_response": agent_response,
            "host_prompt": host_prompt_for_log,
            "host_response": host_response,
            "input_tokens_est": query_tokens,
            "timing_agent": timing,
            "timing_host": host_timing,
        }
        tls.turn_traces.append(turn_entry)

        # ---- Sync back state ----
        tls.prompt_entries = state.get("prompt_entries", tls.prompt_entries)
        tls.timing_log = state.get("timing_log", tls.timing_log)
        tls.abc_snapshots = state.get("abc_snapshots", tls.abc_snapshots)
        tls.compression_count = state.get("compression_count", tls.compression_count)

        if state.get("last_was_compression"):
            tls.last_was_compression = False

        # ---- Save checkpoint + incremental logs ----
        agent.save_checkpoint(sample_id, {
            "turn_idx": turn + 1,
            "step_idx": 0,
            "messages": tls.sample_messages,
            "previous_summary": tls.previous_summary,
            "prompt_entries": tls.prompt_entries,
            "timing_log": tls.timing_log,
            "abc_snapshots": tls.abc_snapshots,
            "turn_traces": tls.turn_traces,
            "compression_count": tls.compression_count,
            "last_was_compression": False,
            "sample_id": sample_id,
        })

        # Incremental log saves
        agent._persist_logs(sample_id, {
            "sample_id": sample_id,
            "messages": tls.sample_messages,
            "turn_idx": turn + 1,
            "step_idx": 0,
            "previous_summary": tls.previous_summary,
            "prompt_entries": tls.prompt_entries,
            "timing_log": tls.timing_log,
            "abc_snapshots": tls.abc_snapshots,
            "turn_traces": tls.turn_traces,
            "compression_count": tls.compression_count,
            "last_was_compression": False,
        })

    # ---- Evaluate game progress (simple key match) ----
    gp_score, key_results = 0.0, []
    if answer_keys:
        import re
        key_lines = [l.strip() for l in answer_keys.strip().split("\n")
                     if re.match(r"^(\d+[.\):]|[-*•])\s+", l.strip())]
        if not key_lines:
            key_lines = [l.strip() for l in answer_keys.strip().split("\n") if l.strip()]
        agent_text = "\n".join(
            t.get("agent", "") for t in tls.turn_traces[-10:]
        ).lower()
        deduced_count = 0
        for k in key_lines:
            deduced = any(
                w.lower() in agent_text for w in k.split() if len(w) > 3
            )
            key_results.append({"key": k, "deduced": deduced})
            if deduced:
                deduced_count += 1
        gp_score = round(deduced_count / max(len(key_lines), 1), 4)
        solved = gp_score >= 0.6

    final_tokens = agent.message_tokens(tls.sample_messages)

    result = {
        "id": pid,
        "mode": mode,
        "story": story[:200],
        "solved": solved,
        "total_turns": total_turns,
        "final_token_count": final_tokens,
        "compression_count": tls.compression_count,
        "compressions": tls.abc_snapshots,
        "game_progress": gp_score,
        "key_results": key_results,
    }

    print(f"    Result: GP={gp_score:.2f} solved={solved} turns={total_turns} "
          f"compressions={tls.compression_count} final_tokens={final_tokens}")

    agent.finalize_sample(sample_id)
    return result


# ============================================================
# Result persistence
# ============================================================

def save_result(mode: str, result: dict):
    result_dir = CFG.get_result_dir(mode)
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "results.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_results(mode: str) -> dict:
    path = CFG.get_result_dir(mode) / "results.jsonl"
    if not path.exists():
        return {}
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                results[e["id"]] = e
            except:
                pass
    return results


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AgentBench Semi-Prefill Compression Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", default="compressed",
                        choices=["baseline", "compressed", "both"],
                        help="Test mode (default: compressed)")
    parser.add_argument("--tasks", nargs="+", default=["ltp"],
                        choices=list(CFG.TASK_CATEGORIES.keys()),
                        help="Tasks to run (default: ltp)")
    parser.add_argument("--model", default=CFG.DEFAULT_MODEL,
                        help=f"Model key (default: {CFG.DEFAULT_MODEL})")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per task (0=all)")
    parser.add_argument("--max-turns", type=int,
                        default=CFG.TASK_CATEGORIES["ltp"]["max_turns"],
                        help=f"Max turns per LTP puzzle (default: {CFG.TASK_CATEGORIES['ltp']['max_turns']})")
    parser.add_argument("--puzzle-ids", nargs="*", type=int, default=None,
                        help="Specific LTP puzzle IDs to run")
    parser.add_argument("--initial-tokens", type=int, default=CFG.TARGET_INITIAL_TOKENS,
                        help=f"Target initial token count (default: {CFG.TARGET_INITIAL_TOKENS})")
    parser.add_argument("--context-window", type=int, default=CFG.CONTEXT_WINDOW,
                        help=f"Context window (default: {CFG.CONTEXT_WINDOW})")
    parser.add_argument("--reserve-tokens", type=int, default=CFG.RESERVE_TOKENS,
                        help=f"Reserve tokens (default: {CFG.RESERVE_TOKENS})")
    parser.add_argument("--keep-recent-tokens", type=int, default=CFG.KEEP_RECENT_TOKENS_BUDGET,
                        help=f"Keep recent tokens budget (default: {CFG.KEEP_RECENT_TOKENS_BUDGET})")
    parser.add_argument("--summary-max-tokens", type=int, default=CFG.SUMMARY_MAX_TOKENS,
                        help=f"Summary max tokens (default: {CFG.SUMMARY_MAX_TOKENS})")
    parser.add_argument("--max-tokens", type=int, default=CFG.MAX_TOKENS,
                        help=f"Max decode tokens per LLM call (default: {CFG.MAX_TOKENS})")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Optional root directory for benchmark outputs")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results")
    parser.add_argument("--output-json", action="store_true",
                        help="Output analysis to JSON files")
    return parser.parse_args()


def main():
    args = parse_args()

    fixed_ltp_turns = CFG.TASK_CATEGORIES["ltp"]["max_turns"]
    if args.max_turns != fixed_ltp_turns:
        print(f"[CONFIG] Forcing LTP max_turns to {fixed_ltp_turns} (requested {args.max_turns})")
        args.max_turns = fixed_ltp_turns

    # Override config with CLI args
    CFG.CONTEXT_WINDOW = args.context_window
    CFG.RESERVE_TOKENS = args.reserve_tokens
    CFG.KEEP_RECENT_TOKENS_BUDGET = args.keep_recent_tokens
    CFG.SUMMARY_MAX_TOKENS = args.summary_max_tokens
    CFG.TARGET_INITIAL_TOKENS = args.initial_tokens
    CFG.MAX_TOKENS = args.max_tokens
    CFG.TASK_CATEGORIES["ltp"]["max_turns"] = args.max_turns
    if args.results_root:
        CFG.RESULTS_DIR = Path(args.results_root).resolve()

    print("=" * 70)
    print("  AgentBench Semi-Prefill Compression Benchmark")
    print(f"  Mode: {args.mode} | Model: {args.model} | Tasks: {args.tasks}")
    print(f"  Compression: cw={CFG.CONTEXT_WINDOW} thr={CFG.CONTEXT_WINDOW - CFG.RESERVE_TOKENS} "
          f"C1_budget={CFG.KEEP_RECENT_TOKENS_BUDGET} summary_max={CFG.SUMMARY_MAX_TOKENS}")
    print(f"  Initial tokens target: {CFG.TARGET_INITIAL_TOKENS}")
    print(f"  Max turns: {args.max_turns} | Decode max tokens: {CFG.MAX_TOKENS}")
    print(f"  Results root: {CFG.RESULTS_DIR}")
    print("=" * 70)

    # Analyze-only mode
    if args.analyze_only:
        modes = ["baseline", "compressed"] if args.mode == "both" else [args.mode]
        for m in modes:
            run_analysis(m, output_json=args.output_json)
        return

    # Health check
    model_info = CFG.MODEL_REGISTRY[args.model]
    api_base = CFG.PROXY_URL or CFG.VLLM_API_BASE
    print("\n[Pre-flight health check]")
    ok, report = check_health(api_base)
    if not ok:
        print("WARNING: API may not be ready. Continuing anyway...")

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_info['tokenizer_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_info["tokenizer_path"], trust_remote_code=True,
    )

    modes = ["baseline", "compressed"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n{'#'*60}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'#'*60}")

        # Create agent
        agent = BenchmarkAgent(
            mode=mode,
            api_base=api_base,
            model_name=model_info["model_path"],
            tokenizer=tokenizer,
            api_key=CFG.VLLM_API_KEY,
            max_tokens=CFG.MAX_TOKENS,
            context_window=CFG.CONTEXT_WINDOW,
            reserve_tokens=CFG.RESERVE_TOKENS,
            keep_recent_tokens=CFG.KEEP_RECENT_TOKENS_BUDGET,
            summary_max_tokens=CFG.SUMMARY_MAX_TOKENS,
        )

        for task_key in args.tasks:
            task_info = CFG.TASK_CATEGORIES[task_key]
            print(f"\n  Task: {task_key} ({task_info['description']})")

            if task_key == "ltp":
                run_ltp_puzzles_for_mode(
                    agent, mode, tokenizer, args,
                )
            else:
                print(f"  Task {task_key} not yet supported in standalone mode")
                print(f"  (requires AgentBench Docker task server)")

    # Final analysis
    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")
    for mode in modes:
        run_analysis(mode, output_json=args.output_json)

    print("\nDone!")


def run_ltp_puzzles_for_mode(agent, mode, tokenizer, args):
    """Run all specified LTP puzzles for a given mode."""
    # Determine puzzle IDs
    target_ids = args.puzzle_ids
    if target_ids is None:
        target_ids = list(range(min(args.max_samples or 8, 50)))

    if args.max_samples and args.max_samples > 0:
        target_ids = target_ids[:args.max_samples]

    # Load puzzles
    puzzles = load_ltp_puzzles(
        limit=max(target_ids) + 1 if target_ids else None,
        target_initial_tokens=args.initial_tokens,
        tokenizer=tokenizer,
    )
    puzzle_map = {p["id"]: p for p in puzzles}

    # Check existing results for resume
    existing = load_results(mode)
    remaining = []
    for pid in target_ids:
        if pid not in existing:
            remaining.append(pid)
        elif mode == "compressed":
            entry = existing[pid]
            if entry.get("compression_count", 0) == 0:
                print(f"  Puzzle {pid}: re-running (no compression in previous run)")
                remaining.append(pid)

    if not remaining:
        print("  All puzzles already completed!")
        return

    print(f"  Running {len(remaining)} puzzles: {remaining}")

    for pid in remaining:
        if pid not in puzzle_map:
            print(f"  Puzzle {pid} not found!")
            continue

        try:
            result = run_ltp_puzzle(
                puzzle_map[pid], mode, agent, tokenizer,
                max_turns=args.max_turns,
            )
            save_result(mode, result)

            # Validation check for compressed mode
            if mode == "compressed":
                comps = result.get("compressions", [])
                if comps:
                    for c in comps:
                        ratio = c.get("B2_to_B1_ratio", 1.0)
                        c1 = c.get("C1_tokens_after", 0)
                        print(f"    ABC: B2/B1={ratio:.4f} (diagnostic), "
                              f"C1={c1} ({2000<=c1<=3000 and 'PASS' or 'RANGE'})")
                else:
                    print(f"    WARNING: No compression triggered for puzzle {pid}!")
        except Exception as e:
            print(f"  ERROR running puzzle {pid}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
