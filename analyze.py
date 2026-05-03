"""
AgentBench Compression Semi-Prefill Analysis.

Analyzes collected benchmark data to produce:

(a) Agent Workload Statistics:
  - Turns per task
  - New tokens per turn
  - Tool call count
  - Total context length distribution
  - Context churn ratio
  - Semi-prefill trigger count & token length

(b) Semi-prefill Ratio Analysis:
  - prefill / semi-prefill / decode time proportions
  - Semi-prefill count and token length per sample
  - Summary generation overhead

Usage:
  python analyze.py --mode compressed             # Analyze compressed results
  python analyze.py --mode baseline               # Analyze baseline results
  python analyze.py --mode both                   # Compare both modes
  python analyze.py --mode compressed --json      # Output JSON
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import config as CFG


def load_timing(mode: str, sample_id: str) -> list[dict]:
    path = CFG.get_timing_dir(mode) / f"{sample_id}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trace(mode: str, sample_id: str) -> dict:
    path = CFG.get_trace_dir(mode) / f"{sample_id}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_abc(mode: str, sample_id: str) -> list[dict]:
    path = CFG.get_abc_dir(mode) / f"{sample_id}_abc.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sample_ids(mode: str) -> list[str]:
    """Discover sample IDs from timing directory."""
    timing_dir = CFG.get_timing_dir(mode)
    if not timing_dir.exists():
        return []
    ids = []
    for f in timing_dir.iterdir():
        if f.suffix == ".json" and not f.name.endswith("_abc.json"):
            ids.append(f.stem)
    return sorted(ids)


def analyze_workload_stats(mode: str, sample_ids: list[str]) -> dict:
    """Compute (a) agent workload statistics."""
    all_turns = []
    all_new_tokens = []
    all_tool_calls = []
    all_context_lengths = []
    all_churn_ratios = []
    semi_prefill_events = []
    compression_events = []

    per_sample = {}

    for sid in sample_ids:
        timing = load_timing(mode, sid)
        trace = load_trace(mode, sid)
        abc = load_abc(mode, sid)

        if not timing:
            continue

        turns = trace.get("total_turns", 0)
        steps = len(timing)
        compressions = len(abc)
        tool_calls = trace.get("tool_calls", 0)

        # Per-turn new tokens (output tokens)
        turn_new_tokens = [t.get("output_tokens", 0) for t in timing]

        # Context lengths (prompt tokens per step)
        context_lengths = [t.get("prompt_tokens", 0) for t in timing]

        # Context churn ratio: new_tokens / total_context
        churn_ratios = []
        for t in timing:
            pt = t.get("prompt_tokens", 1)
            ot = t.get("output_tokens", 0)
            churn_ratios.append(ot / max(pt, 1))

        # Semi-prefill events
        semi_events = [t for t in timing if t.get("classification") == "semi_prefill"]
        full_prefill_events = [t for t in timing if t.get("classification") == "full_prefill"]
        incremental_events = [t for t in timing if t.get("classification") == "incremental"]

        # Compression events from ABC
        for c in abc:
            compression_events.append({
                "sample_id": sid,
                "turn": c.get("turn"),
                "B1_tokens": c.get("B1_tokens"),
                "B2_tokens": c.get("B2_tokens"),
                "B2_to_B1_ratio": c.get("B2_to_B1_ratio"),
                "C1_tokens_after": c.get("C1_tokens_after"),
                "summary_time_s": c.get("summary_generation_time_s"),
            })

        for s in semi_events:
            semi_prefill_events.append({
                "sample_id": sid,
                "turn": s.get("turn"),
                "prompt_tokens": s.get("prompt_tokens"),
                "ttft_ms": s.get("ttft_ms"),
                "total_ms": s.get("total_ms"),
            })

        all_turns.append(turns)
        all_new_tokens.extend(turn_new_tokens)
        all_tool_calls.append(tool_calls)
        all_context_lengths.extend(context_lengths)
        all_churn_ratios.extend(churn_ratios)

        per_sample[sid] = {
            "turns": turns,
            "steps": steps,
            "compressions": compressions,
            "tool_calls": tool_calls,
            "full_prefill_count": len(full_prefill_events),
            "semi_prefill_count": len(semi_events),
            "incremental_count": len(incremental_events),
            "avg_context_length": sum(context_lengths) / max(len(context_lengths), 1),
            "max_context_length": max(context_lengths) if context_lengths else 0,
            "total_new_tokens": sum(turn_new_tokens),
            "context_churn_ratio": sum(churn_ratios) / max(len(churn_ratios), 1),
        }

    stats = {
        "mode": mode,
        "num_samples": len(sample_ids),
        "num_completed": len(per_sample),
        "workload": {
            "turns_per_task": {
                "min": min(all_turns) if all_turns else 0,
                "max": max(all_turns) if all_turns else 0,
                "avg": round(sum(all_turns) / max(len(all_turns), 1), 2),
                "values": all_turns,
            },
            "new_tokens_per_step": {
                "min": min(all_new_tokens) if all_new_tokens else 0,
                "max": max(all_new_tokens) if all_new_tokens else 0,
                "avg": round(sum(all_new_tokens) / max(len(all_new_tokens), 1), 1),
            },
            "tool_calls_per_sample": {
                "min": min(all_tool_calls) if all_tool_calls else 0,
                "max": max(all_tool_calls) if all_tool_calls else 0,
                "avg": round(sum(all_tool_calls) / max(len(all_tool_calls), 1), 1),
            },
            "context_length": {
                "min": min(all_context_lengths) if all_context_lengths else 0,
                "max": max(all_context_lengths) if all_context_lengths else 0,
                "avg": round(sum(all_context_lengths) / max(len(all_context_lengths), 1), 1),
            },
            "context_churn_ratio": {
                "min": round(min(all_churn_ratios), 4) if all_churn_ratios else 0,
                "max": round(max(all_churn_ratios), 4) if all_churn_ratios else 0,
                "avg": round(sum(all_churn_ratios) / max(len(all_churn_ratios), 1), 4),
            },
        },
        "compression": {
            "total_events": len(compression_events),
            "per_sample_avg": round(len(compression_events) / max(len(per_sample), 1), 1),
            "B2_to_B1_ratios": [c["B2_to_B1_ratio"] for c in compression_events],
            "C1_token_range": {
                "min": min((c.get("C1_tokens_after", 0) for c in compression_events), default=0),
                "max": max((c.get("C1_tokens_after", 0) for c in compression_events), default=0),
                "avg": round(sum(c.get("C1_tokens_after", 0) for c in compression_events)
                           / max(len(compression_events), 1), 0),
            },
            "summary_time_s": {
                "min": round(min((c.get("summary_time_s", 0) or 0 for c in compression_events), default=0), 1),
                "max": round(max((c.get("summary_time_s", 0) or 0 for c in compression_events), default=0), 1),
                "avg": round(sum(c.get("summary_time_s", 0) or 0 for c in compression_events)
                           / max(len(compression_events), 1), 1),
            },
        },
        "semi_prefill": {
            "total_events": len(semi_prefill_events),
            "per_sample_avg": round(len(semi_prefill_events) / max(len(per_sample), 1), 1),
            "tokens_per_event": {
                "min": min((s.get("prompt_tokens", 0) for s in semi_prefill_events), default=0),
                "max": max((s.get("prompt_tokens", 0) for s in semi_prefill_events), default=0),
                "avg": round(sum(s.get("prompt_tokens", 0) for s in semi_prefill_events)
                           / max(len(semi_prefill_events), 1), 0),
            },
        },
        "per_sample": per_sample,
    }

    # Aggregate compression ratio stats
    ratios = stats["compression"]["B2_to_B1_ratios"]
    if ratios:
        stats["compression"]["B2_to_B1_ratio_stats"] = {
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "avg": round(sum(ratios) / len(ratios), 4),
            "median": round(sorted(ratios)[len(ratios)//2], 4),
        }

    return stats


def analyze_semi_prefill_ratio(mode: str, sample_ids: list[str]) -> dict:
    """Compute (b) semi-prefill time ratio analysis.

    Breaks total end-to-end time into prefill / semi-prefill / decode proportions.
    """
    all_classifications = {"full_prefill": [], "semi_prefill": [], "incremental": []}
    total_time_ms = 0
    summary_generation_time_s = 0

    per_sample_ratio = {}

    for sid in sample_ids:
        timing = load_timing(mode, sid)
        abc = load_abc(mode, sid)

        if not timing:
            continue

        sample_class = {"full_prefill": 0.0, "semi_prefill": 0.0, "incremental": 0.0,
                        "summary_gen": 0.0}
        sample_total = 0.0

        for t in timing:
            cls = t.get("classification", "incremental")
            ms = t.get("total_ms", 0)
            sample_class[cls] = sample_class.get(cls, 0) + ms
            sample_total += ms
            all_classifications[cls].append(ms)

        # Summary generation time
        for c in abc:
            st = c.get("summary_generation_time_s", 0) or 0
            sample_class["summary_gen"] += st * 1000  # convert to ms
            summary_generation_time_s += st

        total_time_ms += sample_total

        if sample_total > 0:
            per_sample_ratio[sid] = {
                "total_time_s": round(sample_total / 1000, 2),
                "prefill_pct": round(sample_class["full_prefill"] / sample_total * 100, 1),
                "semi_prefill_pct": round(sample_class["semi_prefill"] / sample_total * 100, 1),
                "decode_pct": round(sample_class["incremental"] / sample_total * 100, 1),
                "summary_gen_pct": round(sample_class["summary_gen"] / sample_total * 100, 1),
                "prefill_ms": round(sample_class["full_prefill"], 1),
                "semi_prefill_ms": round(sample_class["semi_prefill"], 1),
                "decode_ms": round(sample_class["incremental"], 1),
                "summary_gen_ms": round(sample_class["summary_gen"], 1),
            }

    # Aggregate
    aggregate = {}
    for cls, values in all_classifications.items():
        if values:
            aggregate[cls] = {
                "total_ms": round(sum(values), 1),
                "count": len(values),
                "avg_ms": round(sum(values) / len(values), 2),
                "max_ms": round(max(values), 2),
            }
        else:
            aggregate[cls] = {"total_ms": 0, "count": 0, "avg_ms": 0, "max_ms": 0}

    overall = {}
    if total_time_ms > 0:
        overall = {
            "total_e2e_time_s": round(total_time_ms / 1000, 2),
            "summary_generation_total_s": round(summary_generation_time_s, 2),
            "prefill_pct": round(aggregate["full_prefill"]["total_ms"] / total_time_ms * 100, 1),
            "semi_prefill_pct": round(aggregate["semi_prefill"]["total_ms"] / total_time_ms * 100, 1),
            "decode_pct": round(aggregate["incremental"]["total_ms"] / total_time_ms * 100, 1),
        }

    return {
        "mode": mode,
        "num_samples": len(sample_ids),
        "classifications": aggregate,
        "overall_ratio": overall,
        "per_sample_ratio": per_sample_ratio,
    }


def print_workload_report(stats: dict):
    """Pretty-print workload statistics."""
    print(f"\n{'='*70}")
    print(f"  AGENT WORKLOAD STATISTICS — {stats['mode'].upper()}")
    print(f"{'='*70}")
    print(f"  Samples completed: {stats['num_completed']}/{stats['num_samples']}")

    wl = stats["workload"]
    print(f"\n  --- Turn Statistics ---")
    print(f"  Turns per task:   min={wl['turns_per_task']['min']}  max={wl['turns_per_task']['max']}  avg={wl['turns_per_task']['avg']}")
    print(f"  New tokens/step:  min={wl['new_tokens_per_step']['min']}  max={wl['new_tokens_per_step']['max']}  avg={wl['new_tokens_per_step']['avg']}")
    print(f"  Tool calls/sample:min={wl['tool_calls_per_sample']['min']}  max={wl['tool_calls_per_sample']['max']}  avg={wl['tool_calls_per_sample']['avg']}")
    print(f"  Context length:   min={wl['context_length']['min']}  max={wl['context_length']['max']}  avg={wl['context_length']['avg']}")
    print(f"  Churn ratio:      min={wl['context_churn_ratio']['min']}  max={wl['context_churn_ratio']['max']}  avg={wl['context_churn_ratio']['avg']}")

    cp = stats["compression"]
    print(f"\n  --- Compression ---")
    print(f"  Total events: {cp['total_events']} ({cp['per_sample_avg']}/sample)")
    if "B2_to_B1_ratio_stats" in cp:
        rs = cp["B2_to_B1_ratio_stats"]
        print(f"  B2/B1 ratio:      min={rs['min']:.4f}  max={rs['max']:.4f}  avg={rs['avg']:.4f}  median={rs['median']:.4f}  (diagnostic)")
    print(f"  C1 tokens:        min={cp['C1_token_range']['min']}  max={cp['C1_token_range']['max']}  avg={cp['C1_token_range']['avg']}")
    print(f"  Summary time:     min={cp['summary_time_s']['min']}s  max={cp['summary_time_s']['max']}s  avg={cp['summary_time_s']['avg']}s")

    sp = stats["semi_prefill"]
    print(f"\n  --- Semi-Prefill ---")
    print(f"  Total events: {sp['total_events']} ({sp['per_sample_avg']}/sample)")
    print(f"  Tokens/event:     min={sp['tokens_per_event']['min']}  max={sp['tokens_per_event']['max']}  avg={sp['tokens_per_event']['avg']}")


def print_semi_prefill_report(ratio: dict):
    """Pretty-print semi-prefill ratio analysis."""
    print(f"\n{'='*70}")
    print(f"  SEMI-PREFILL RATIO ANALYSIS — {ratio['mode'].upper()}")
    print(f"{'='*70}")

    ov = ratio["overall_ratio"]
    if ov:
        print(f"  Total E2E time:  {ov['total_e2e_time_s']:.1f}s")
        print(f"  Summary gen:     {ov['summary_generation_total_s']:.1f}s")
        print(f"\n  Time breakdown:")
        print(f"    Prefill:        {ov['prefill_pct']:.1f}%  ({ratio['classifications']['full_prefill']['total_ms']:.0f}ms)")
        print(f"    Semi-prefill:   {ov['semi_prefill_pct']:.1f}%  ({ratio['classifications']['semi_prefill']['total_ms']:.0f}ms, {ratio['classifications']['semi_prefill']['count']} events)")
        print(f"    Decode:         {ov['decode_pct']:.1f}%  ({ratio['classifications']['incremental']['total_ms']:.0f}ms, {ratio['classifications']['incremental']['count']} events)")

    print(f"\n  Classification details:")
    for cls in ["full_prefill", "semi_prefill", "incremental"]:
        d = ratio["classifications"][cls]
        print(f"    {cls:16s} count={d['count']:3d}  total={d['total_ms']:8.0f}ms  avg={d['avg_ms']:7.1f}ms  max={d['max_ms']:7.1f}ms")

    print(f"\n  Per-sample ratios:")
    for sid, r in sorted(ratio["per_sample_ratio"].items()):
        print(f"    {sid:20s}  total={r['total_time_s']:6.1f}s  prefill={r['prefill_pct']:5.1f}%  semi={r['semi_prefill_pct']:5.1f}%  decode={r['decode_pct']:5.1f}%")


def run_analysis(mode: str, output_json: bool = False):
    """Main analysis entry point."""
    sample_ids = list_sample_ids(mode)

    if not sample_ids:
        print(f"No results found for mode '{mode}' in {CFG.get_timing_dir(mode)}")
        return

    print(f"Analyzing {len(sample_ids)} samples for mode '{mode}'...")

    workload = analyze_workload_stats(mode, sample_ids)
    semi_ratio = analyze_semi_prefill_ratio(mode, sample_ids)

    if output_json:
        output_dir = CFG.get_analysis_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"workload_{mode}.json", "w") as f:
            json.dump(workload, f, indent=2, ensure_ascii=False)
        with open(output_dir / f"semi_prefill_ratio_{mode}.json", "w") as f:
            json.dump(semi_ratio, f, indent=2, ensure_ascii=False)
        print(f"JSON output saved to {output_dir}")

    print_workload_report(workload)
    print_semi_prefill_report(semi_ratio)

    return workload, semi_ratio


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgentBench Semi-Prefill Analysis")
    parser.add_argument("--mode", default="compressed",
                        choices=["baseline", "compressed", "both"],
                        help="Results mode to analyze")
    parser.add_argument("--json", action="store_true",
                        help="Also output JSON files")
    args = parser.parse_args()

    modes = ["baseline", "compressed"] if args.mode == "both" else [args.mode]
    for m in modes:
        run_analysis(m, output_json=args.json)
