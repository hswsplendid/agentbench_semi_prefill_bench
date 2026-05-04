#!/usr/bin/env python3
"""
Semi-Prefill Overhead Analysis for cw32000 (No Prefix Caching).
Generates charts and JSON stats for the MD report.
Adapted from /root/semi_prefill_bench/analyze_cw8000.py
"""
import json
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "/root/agentbench_semi_prefill_bench/results/compressed"
OUT_DIR  = os.path.join(DATA_DIR, "..", "analysis")
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Hardware parameters (measured from T0 data: 21.5k tok → 14.7s TTFT)
# r_pf ≈ 14652ms / 21690tok ≈ 0.675 ms/tok
# r_dec ≈ (77544-14652)ms / 782tok ≈ 80 ms/tok
R_PF  = 0.68   # ms/token prefill  (measured ~1500 tok/s)
R_DEC = 75.0   # ms/token decode   (measured ~13 tok/s)
C_FIXED = 50   # ms per request fixed overhead

PUZZLE_IDS = [0, 1, 2]  # Puzzle 3 is incomplete

# ── Helpers ─────────────────────────────────────────────────────────
def load_puzzle(pid):
    trace  = json.load(open(f"{DATA_DIR}/traces/puzzle_{pid}.json"))
    timing = json.load(open(f"{DATA_DIR}/timing/puzzle_{pid}.json"))
    abc    = json.load(open(f"{DATA_DIR}/abc_segments/puzzle_{pid}_abc.json"))
    log    = [json.loads(l) for l in
              open(f"{DATA_DIR}/prompt_logs/puzzle_{pid}.jsonl").read().strip().splitlines()]
    return trace, timing, abc, log

def prefill_cost(n_tokens):
    return C_FIXED + n_tokens * R_PF

def decode_cost(n_tokens):
    return n_tokens * R_DEC

# ── (a) Load and structure data ─────────────────────────────────────
all_stats = {}

for pid in PUZZLE_IDS:
    trace, timing, abc, log = load_puzzle(pid)

    # Map timing to turns: each turn has agent_step + host_step
    agent_steps = [t for t in timing if t['step'] == 0]  # agent calls
    host_steps = [t for t in timing if t['step'] == 1]   # host calls (approximate)

    # Use prompt_logs for per-turn data (most reliable)
    turns = log
    n_turns = len(turns)

    # Per-turn token deltas (non-compression turns)
    deltas = []
    for i in range(1, n_turns):
        if not turns[i].get('compressed_this_turn', False):
            prev_tok = turns[i-1]['input_tokens_est']
            cur_tok  = turns[i]['input_tokens_est']
            if cur_tok > prev_tok:
                deltas.append(cur_tok - prev_tok)

    # Context lengths over turns
    ctx_lengths = [t['input_tokens_est'] for t in turns]

    # Context churn ratio
    churn_ratios = []
    for i, t in enumerate(turns):
        if t.get('compressed_this_turn', False):
            # Compression turn: B2+C1 is re-prefilled (whole context after compression)
            for c in abc:
                if c.get('turn') == t['turn']:
                    total_post = c['A_tokens'] + c['B2_tokens'] + c['C1_tokens_after']
                    churn = (c['B2_tokens'] + c['C1_tokens_after']) / total_post
                    churn_ratios.append(churn)
                    break
        elif i > 0:
            delta = t['input_tokens_est'] - turns[i-1]['input_tokens_est']
            if delta < 0:
                delta = t['input_tokens_est']
            churn = delta / max(t['input_tokens_est'], 1)
            churn_ratios.append(churn)
        else:
            churn_ratios.append(1.0)

    # Semi-prefill stats from ABC
    sp_events = []
    for c in abc:
        b2_c1 = c['B2_tokens'] + c['C1_tokens_after']
        sp_time = prefill_cost(b2_c1)
        sp_events.append({
            'turn': c['turn'],
            'A': c['A_tokens'],
            'B1': c['B1_tokens'],
            'B2': c['B2_tokens'],
            'C1': c['C1_tokens_after'],
            'B2_C1': b2_c1,
            'B2_B1_ratio': c.get('B2_to_B1_ratio', 0),
            'semi_prefill_ms': sp_time,
            'summary_time_s': c.get('summary_generation_time_s', 0) or 0,
            'pre_tokens': c.get('pre_prompt_tokens', 0),
            'post_tokens': c.get('post_prompt_tokens', 0),
        })

    # Total output (agent + host chars per turn)
    total_agent_chars = sum(len(t.get('agent_response', '')) for t in turns)
    total_host_chars = sum(len(t.get('host_response', '')) for t in turns)

    # Agent/host output tokens from timing
    agent_output_tok = sum(t.get('output_tokens', 0) for t in agent_steps if t.get('output_tokens'))
    host_output_tok = sum(t.get('output_tokens', 0) for t in host_steps if t.get('output_tokens'))

    stats = {
        'puzzle_id': pid,
        'n_turns': n_turns,
        'n_compressions': len(abc),
        'compression_rate': len(abc) / n_turns if n_turns > 0 else 0,
        'delta_median': statistics.median(deltas) if deltas else 0,
        'delta_mean': sum(deltas)/len(deltas) if deltas else 0,
        'delta_min': min(deltas) if deltas else 0,
        'delta_max': max(deltas) if deltas else 0,
        'ctx_lengths': ctx_lengths,
        'ctx_max': max(ctx_lengths),
        'ctx_min': min(ctx_lengths),
        'churn_ratios': churn_ratios,
        'churn_mean': sum(churn_ratios)/len(churn_ratios) if churn_ratios else 0,
        'sp_events': sp_events,
        'total_agent_chars': total_agent_chars,
        'total_host_chars': total_host_chars,
        'agent_output_tok': agent_output_tok,
        'host_output_tok': host_output_tok,
        'turns': turns,
        'timing': timing,
        'abc': abc,
    }
    all_stats[pid] = stats


# ── (b) Phase Breakdown (No Prefix Caching model) ───────────────────
# Key difference from cw8000: without prefix caching, EVERY turn does
# a full prefill of the current context. Compression reduces context
# from ~30k to ~5k, making subsequent prefill much cheaper.

phase_data = {}

for pid in PUZZLE_IDS:
    s = all_stats[pid]
    turns = s['turns']
    abc_map = {c.get('turn'): c for c in s['abc']}

    full_prefill_ms = 0.0
    semi_prefill_ms = 0.0
    incremental_prefill_ms = 0.0
    decode_ms = 0.0

    turn_details = []

    # Map each turn to timing data
    agent_steps = {}
    host_steps = {}
    for t in s['timing']:
        key = t['turn']
        if t['step'] == 0:
            agent_steps[key] = t
        else:
            if key not in host_steps:
                host_steps[key] = t

    for i, t in enumerate(turns):
        turn_num = t['turn']
        ag = agent_steps.get(turn_num, {})
        hs = host_steps.get(turn_num, {})

        agent_out = ag.get('output_tokens', 0)
        host_out = hs.get('output_tokens', 0)
        turn_decode = decode_cost(agent_out) + decode_cost(host_out)
        decode_ms += turn_decode

        is_compressed = t.get('compressed_this_turn', False)

        if turn_num == 0:
            # Full prefill: all tokens
            ctx_tok = t['input_tokens_est']
            pf = prefill_cost(ctx_tok)
            full_prefill_ms += pf
            turn_details.append({
                'turn': turn_num,
                'class': 'full_prefill',
                'prefill_tokens': ctx_tok,
                'prefill_ms': pf,
                'decode_ms': turn_decode,
                'total_ms': pf + turn_decode,
            })
        elif is_compressed:
            # Compression turn: prefill is (B2+C1+A) = post_compression tokens
            c = abc_map.get(turn_num, {})
            post_tok = c.get('post_prompt_tokens', t['input_tokens_est'])
            sp_tokens = post_tok
            sp = prefill_cost(sp_tokens)
            semi_prefill_ms += sp
            turn_details.append({
                'turn': turn_num,
                'class': 'semi_prefill',
                'prefill_tokens': sp_tokens,
                'A_cached': c.get('A_tokens', 0),
                'B2': c.get('B2_tokens', 0),
                'C1': c.get('C1_tokens_after', 0),
                'prefill_ms': sp,
                'decode_ms': turn_decode,
                'total_ms': sp + turn_decode,
            })
        else:
            # No prefix cache: full context prefill
            ctx_tok = t['input_tokens_est']
            pf = prefill_cost(ctx_tok)
            incremental_prefill_ms += pf
            turn_details.append({
                'turn': turn_num,
                'class': 'incremental',
                'prefill_tokens': ctx_tok,
                'prefill_ms': pf,
                'decode_ms': turn_decode,
                'total_ms': pf + turn_decode,
            })

    total_ms = full_prefill_ms + incremental_prefill_ms + semi_prefill_ms + decode_ms
    phase_data[pid] = {
        'full_prefill_ms': full_prefill_ms,
        'incremental_prefill_ms': incremental_prefill_ms,
        'semi_prefill_ms': semi_prefill_ms,
        'decode_ms': decode_ms,
        'total_ms': total_ms,
        'full_prefill_pct': full_prefill_ms / total_ms * 100 if total_ms > 0 else 0,
        'incremental_prefill_pct': incremental_prefill_ms / total_ms * 100 if total_ms > 0 else 0,
        'semi_prefill_pct': semi_prefill_ms / total_ms * 100 if total_ms > 0 else 0,
        'decode_pct': decode_ms / total_ms * 100 if total_ms > 0 else 0,
        'turn_details': turn_details,
    }

# ── Save JSON results ──────────────────────────────────────────────
json_out = {
    'config': {
        'r_pf': R_PF,
        'r_dec': R_DEC,
        'c_fixed': C_FIXED,
        'data_dir': DATA_DIR,
        'prefix_caching': False,
    },
    'workload_stats': {pid: {k: v for k, v in s.items()
                              if k not in ('timing', 'abc', 'turns', 'ctx_lengths', 'churn_ratios')}
                        for pid, s in all_stats.items()},
    'phase_breakdown': {pid: {k: v for k, v in p.items() if k != 'turn_details'}
                         for pid, p in phase_data.items()},
}
for pid in PUZZLE_IDS:
    json_out['workload_stats'][pid]['sp_events'] = all_stats[pid]['sp_events']

with open(f"{OUT_DIR}/analysis_results.json", 'w') as f:
    json.dump(json_out, f, indent=2, default=str)

print("JSON results saved to", f"{OUT_DIR}/analysis_results.json")

# ── CHARTS ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'full_prefill': '#2196F3',
    'incremental': '#4CAF50',
    'semi_prefill': '#FF5722',
    'decode': '#9C27B0',
    'cached': '#E0E0E0',
    'summary_decode': '#F44336',
}

# ── Chart 1: Context Length Over Turns ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
for ax, pid in zip(axes, PUZZLE_IDS):
    s = all_stats[pid]
    turns_x = list(range(len(s['ctx_lengths'])))
    ctx = s['ctx_lengths']
    colors = []
    for t in s['turns']:
        if t['turn'] == 0:
            colors.append(COLORS['full_prefill'])
        elif t.get('compressed_this_turn', False):
            colors.append(COLORS['semi_prefill'])
        else:
            colors.append(COLORS['incremental'])

    ax.bar(turns_x, ctx, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=30000, color='red', linestyle='--', alpha=0.7, label='Threshold (30000)')
    ax.set_xlabel('Turn')
    ax.set_title(f'Puzzle {pid} ({s["n_turns"]} turns, {s["n_compressions"]} comp.)')
    if ax == axes[0]:
        ax.set_ylabel('Input Tokens')
    ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Context Length Over Turns (cw32000, No Prefix Caching)', fontsize=14, y=1.02)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['full_prefill'], label='Full Prefill (T0)'),
    Patch(facecolor=COLORS['incremental'], label='No Cache Re-Prefill'),
    Patch(facecolor=COLORS['semi_prefill'], label='Post-Compression'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/context_length_over_turns.png")
plt.close()
print("Chart 1: context_length_over_turns.png")

# ── Chart 2: Phase Breakdown Stacked Bar ────────────────────────────
# Using theoretical model with No Prefix Caching
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(PUZZLE_IDS))
width = 0.5

fp_vals = [phase_data[pid]['full_prefill_ms'] / 1000 for pid in PUZZLE_IDS]
ip_vals = [phase_data[pid]['incremental_prefill_ms'] / 1000 for pid in PUZZLE_IDS]
sp_vals = [phase_data[pid]['semi_prefill_ms'] / 1000 for pid in PUZZLE_IDS]
dc_vals = [phase_data[pid]['decode_ms'] / 1000 for pid in PUZZLE_IDS]

fp_arr = np.array(fp_vals)
ip_arr = np.array(ip_vals)
sp_arr = np.array(sp_vals)
dc_arr = np.array(dc_vals)

ip_bottom = fp_arr
sp_bottom = fp_arr + ip_arr
dc_bottom = fp_arr + ip_arr + sp_arr

ax.bar(x, fp_arr, width, label='Full Prefill (T0)', color=COLORS['full_prefill'])
ax.bar(x, ip_arr, width, bottom=ip_bottom, label='No-Cache Prefill (Incr. class)', color=COLORS['incremental'])
ax.bar(x, sp_arr, width, bottom=sp_bottom, label='Semi-Prefill (post-compression)', color=COLORS['semi_prefill'])
ax.bar(x, dc_arr, width, bottom=dc_bottom, label='Decode', color=COLORS['decode'])

ax.set_xlabel('Puzzle')
ax.set_ylabel('Theoretical Latency (seconds)')
ax.set_title('Phase Breakdown: Prefill / Semi-Prefill / Decode (No Prefix Caching)')
ax.set_xticks(x)
ax.set_xticklabels([f'Puzzle {pid}' for pid in PUZZLE_IDS])
ax.legend(fontsize=9)

# Add percentage labels
for i, pid in enumerate(PUZZLE_IDS):
    p = phase_data[pid]
    total = p['total_ms'] / 1000
    y_positions = [
        fp_vals[i] / 2,
        fp_vals[i] + ip_vals[i] / 2,
        fp_vals[i] + ip_vals[i] + sp_vals[i] / 2,
        fp_vals[i] + ip_vals[i] + sp_vals[i] + dc_vals[i] / 2,
    ]
    pcts = [p['full_prefill_pct'], p['incremental_prefill_pct'],
            p['semi_prefill_pct'], p['decode_pct']]
    for j, (y, pct) in enumerate(zip(y_positions, pcts)):
        if pct > 3:
            ax.text(i, y, f'{pct:.1f}%', ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/phase_breakdown_stacked.png")
plt.close()
print("Chart 2: phase_breakdown_stacked.png")

# ── Chart 3: Semi-Prefill Token Composition ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, pid in zip(axes, PUZZLE_IDS):
    s = all_stats[pid]
    events = s['sp_events']
    if not events:
        ax.text(0.5, 0.5, 'No compressions', transform=ax.transAxes, ha='center')
        continue
    turns_x = [e['turn'] for e in events]
    a_vals  = [e['A'] for e in events]
    b2_vals = [e['B2'] for e in events]
    c1_vals = [e['C1'] for e in events]

    xi = np.arange(len(events))
    w = 0.6
    ax.bar(xi, a_vals, w, label='A (system, cache hit)', color=COLORS['cached'], edgecolor='gray')
    ax.bar(xi, b2_vals, w, bottom=a_vals, label='B2 (summary)', color='#FF9800')
    ax.bar(xi, c1_vals, w, bottom=[a_vals[j]+b2_vals[j] for j in range(len(events))],
           label='C1 (recent)', color=COLORS['semi_prefill'])

    # Add B2/B1 ratio labels
    for j, e in enumerate(events):
        ratio = e['B2_B1_ratio']
        ax.text(j, a_vals[j] + b2_vals[j] + c1_vals[j] + 200,
                f'B2/B1={ratio:.1%}', ha='center', fontsize=8, color='#555')

    ax.set_xticks(xi)
    ax.set_xticklabels([f'T{t}' for t in turns_x])
    ax.set_xlabel('Compression Turn')
    ax.set_title(f'Puzzle {pid} ({len(events)} compressions)')
    if ax == axes[0]:
        ax.set_ylabel('Tokens')
    ax.legend(fontsize=8)

fig.suptitle('Semi-Prefill Token Composition: A (cached) + B2 (summary) + C1 (recent)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/semi_prefill_composition.png")
plt.close()
print("Chart 3: semi_prefill_composition.png")

# ── Chart 4: Per-Turn Latency (No Prefix Cache, Theoretical) ────────
for pid in PUZZLE_IDS:
    p = phase_data[pid]
    details = p['turn_details']

    fig, ax = plt.subplots(figsize=(14, 5))
    turns_x = [d['turn'] for d in details]
    pf_ms   = [d['prefill_ms'] / 1000 for d in details]
    dc_ms   = [d['decode_ms'] / 1000 for d in details]

    bar_colors_pf = []
    for d in details:
        if d['class'] == 'full_prefill':
            bar_colors_pf.append(COLORS['full_prefill'])
        elif d['class'] == 'semi_prefill':
            bar_colors_pf.append(COLORS['semi_prefill'])
        else:
            bar_colors_pf.append(COLORS['incremental'])

    xi = np.arange(len(details))
    w = 0.6
    ax.bar(xi, pf_ms, w, label='Prefill (full ctx, no cache)', color=bar_colors_pf, edgecolor='white', linewidth=0.5)
    ax.bar(xi, dc_ms, w, bottom=pf_ms, label='Decode', color=COLORS['decode'], alpha=0.7)

    ax.set_xticks(xi)
    ax.set_xticklabels([f'T{t}' for t in turns_x], fontsize=8, rotation=45)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title(f'Puzzle {pid}: Per-Turn Latency (No Prefix Caching, Theory Model)')

    legend_elements = [
        Patch(facecolor=COLORS['full_prefill'], label='Full Prefill'),
        Patch(facecolor=COLORS['incremental'], label='No-Cache Prefill'),
        Patch(facecolor=COLORS['semi_prefill'], label='Post-Compression Prefill'),
        Patch(facecolor=COLORS['decode'], alpha=0.7, label='Decode'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/per_turn_latency_P{pid}.png")
    plt.close()
    print(f"Chart 4-P{pid}: per_turn_latency_P{pid}.png")

# ── Chart 5: Pie Charts ────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, pid in zip(axes[:3], PUZZLE_IDS):
    p = phase_data[pid]
    sizes = [p['full_prefill_ms'], p['incremental_prefill_ms'],
             p['semi_prefill_ms'], p['decode_ms']]
    labels = ['Full Prefill', 'No-Cache Prefill', 'Semi-Prefill', 'Decode']
    colors = [COLORS['full_prefill'], COLORS['incremental'],
              COLORS['semi_prefill'], COLORS['decode']]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sz, lb, cl = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(sz, labels=lb, colors=cl, autopct='%1.1f%%',
                                           textprops={'fontsize': 9}, pctdistance=0.75)
        for t in autotexts:
            t.set_fontsize(8)
    ax.set_title(f'Puzzle {pid}', fontsize=12)

# Aggregate
agg_fp = sum(phase_data[pid]['full_prefill_ms'] for pid in PUZZLE_IDS)
agg_ip = sum(phase_data[pid]['incremental_prefill_ms'] for pid in PUZZLE_IDS)
agg_sp = sum(phase_data[pid]['semi_prefill_ms'] for pid in PUZZLE_IDS)
agg_dc = sum(phase_data[pid]['decode_ms'] for pid in PUZZLE_IDS)
agg_total = agg_fp + agg_ip + agg_sp + agg_dc

sizes = [agg_fp, agg_ip, agg_sp, agg_dc]
labels = ['Full Prefill', 'No-Cache Prefill', 'Semi-Prefill', 'Decode']
colors = [COLORS['full_prefill'], COLORS['incremental'],
          COLORS['semi_prefill'], COLORS['decode']]
sz, lb, cl = zip(*[(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0])
wedges, texts, autotexts = axes[3].pie(sz, labels=lb, colors=cl, autopct='%1.1f%%',
                                        textprops={'fontsize': 9}, pctdistance=0.75)
for t in autotexts:
    t.set_fontsize(8)
axes[3].set_title('Aggregate', fontsize=12)

fig.suptitle('Latency Phase Distribution (No Prefix Caching, Theory Model)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/phase_pie_charts.png")
plt.close()
print("Chart 5: phase_pie_charts.png")

# ── Chart 6: Semi-Prefill Spikes ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

for pid in PUZZLE_IDS:
    s = all_stats[pid]
    p = phase_data[pid]
    details = p['turn_details']

    turns_x = []
    pf_costs = []
    for d in details:
        turns_x.append(d['turn'])
        pf_costs.append(d['prefill_ms'] / 1000)

    ax.plot(turns_x, pf_costs, marker='o', markersize=4, label=f'P{pid}')

    # Mark compression turns
    for d in details:
        if d['class'] == 'semi_prefill':
            ax.plot(d['turn'], d['prefill_ms'] / 1000, marker='^', markersize=10,
                    color=COLORS['semi_prefill'], zorder=5)

# Baseline: prefill cost at compressed context (~5k tok)
bl_prefill_s = prefill_cost(5000) / 1000
ax.axhline(y=bl_prefill_s, color='gray', linestyle='--', alpha=0.7,
           label=f'Post-compression baseline (~5k tok, {bl_prefill_s:.1f}s)')

ax.set_xlabel('Turn')
ax.set_ylabel('Prefill Latency (seconds)')
ax.set_title('Per-Turn Prefill Cost (No Prefix Cache): Compression Reduces Prefill 6-8x')
ax.legend(fontsize=9)
from matplotlib.lines import Line2D
custom = [Line2D([0], [0], marker='^', color=COLORS['semi_prefill'], linestyle='None',
                 markersize=10, label='Compression turn')]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + custom, fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/semi_prefill_spikes.png")
plt.close()
print("Chart 6: semi_prefill_spikes.png")

# ── Chart 7: Context Churn Ratio ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for pid in PUZZLE_IDS:
    s = all_stats[pid]
    turns_x = []
    churns = []
    for i, t in enumerate(s['turns']):
        if t.get('compressed_this_turn', False):
            for c in s['abc']:
                if c.get('turn') == t['turn']:
                    total = c['A_tokens'] + c['B2_tokens'] + c['C1_tokens_after']
                    churns.append((c['B2_tokens'] + c['C1_tokens_after']) / total)
                    break
        elif i > 0:
            delta = t['input_tokens_est'] - s['turns'][i-1]['input_tokens_est']
            if delta < 0:
                delta = 0
            churns.append(delta / max(t['input_tokens_est'], 1))
        else:
            churns.append(1.0)
        turns_x.append(t['turn'])

    ax.plot(turns_x, churns, marker='o', markersize=4, label=f'P{pid}', alpha=0.8)

    # Mark compression turns
    for i, t in enumerate(s['turns']):
        if t.get('compressed_this_turn', False):
            ax.plot(t['turn'], churns[i], marker='^', markersize=10,
                    color=COLORS['semi_prefill'], zorder=5)

ax.set_xlabel('Turn')
ax.set_ylabel('Context Churn Ratio')
ax.set_title('Context Churn Ratio per Turn (No Prefix Cache: every turn is full re-prefill)')
ax.set_ylim(0, 1.1)
handles, labels_leg = ax.get_legend_handles_labels()
custom = [Line2D([0], [0], marker='^', color=COLORS['semi_prefill'], linestyle='None',
                 markersize=10, label='Compression turn')]
ax.legend(handles=handles + custom, fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/context_churn_ratio.png")
plt.close()
print("Chart 7: context_churn_ratio.png")

# ── Chart 8: Summary Generation Time per Compression Event ──────────
fig, ax = plt.subplots(figsize=(12, 5))
for pid in PUZZLE_IDS:
    s = all_stats[pid]
    events = s['sp_events']
    if not events:
        continue
    turns_x = [e['turn'] for e in events]
    times_s = [e['summary_time_s'] for e in events]
    b2_vals = [e['B2'] for e in events]

    ax.bar(np.arange(len(events)) + PUZZLE_IDS.index(pid) * 0.3, times_s,
           0.25, label=f'P{pid}', alpha=0.85)

ax.set_xlabel('Compression Event')
ax.set_ylabel('Summary Generation Time (seconds)')
ax.set_title('Summary Generation Time per Compression Event (Sync Mode)')
ax.legend()
ax.axhline(y=sum(e['summary_time_s'] for pid in PUZZLE_IDS
                  for e in all_stats[pid]['sp_events']) /
             max(sum(1 for pid in PUZZLE_IDS for e in all_stats[pid]['sp_events']), 1),
           color='red', linestyle='--', alpha=0.7, label='Average')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/summary_generation_time.png")
plt.close()
print("Chart 8: summary_generation_time.png")

# ── Chart 9: Prefill-Only Bar (No Decode, measured data) ──────────
# Using measured timing data (not theoretical model)
prefill_only_data = {}

for pid in PUZZLE_IDS:
    s = all_stats[pid]
    timing = s['timing']

    fp_ms = 0.0  # full prefill
    ncpf_ms = 0.0  # no-cache re-prefill (incremental classification)
    sp_ms = 0.0  # post-compression prefill (semi_prefill classification)

    for t in timing:
        cls = t.get('classification', 'incremental')
        # Prefill time ≈ ttft_ms (first token time is dominated by prefill)
        pf_time = t.get('ttft_ms', 0)
        if cls == 'full_prefill':
            fp_ms += pf_time
        elif cls == 'semi_prefill':
            sp_ms += pf_time
        else:
            ncpf_ms += pf_time

    total_pf = fp_ms + ncpf_ms + sp_ms
    prefill_only_data[pid] = {
        'fp_ms': fp_ms, 'ncpf_ms': ncpf_ms, 'sp_ms': sp_ms,
        'total_pf_ms': total_pf,
        'fp_pct': fp_ms / total_pf * 100 if total_pf > 0 else 0,
        'ncpf_pct': ncpf_ms / total_pf * 100 if total_pf > 0 else 0,
        'sp_pct': sp_ms / total_pf * 100 if total_pf > 0 else 0,
    }

# Aggregate
agg_pf = {
    'fp_ms': sum(prefill_only_data[pid]['fp_ms'] for pid in PUZZLE_IDS),
    'ncpf_ms': sum(prefill_only_data[pid]['ncpf_ms'] for pid in PUZZLE_IDS),
    'sp_ms': sum(prefill_only_data[pid]['sp_ms'] for pid in PUZZLE_IDS),
}
agg_pf['total_pf_ms'] = agg_pf['fp_ms'] + agg_pf['ncpf_ms'] + agg_pf['sp_ms']
for k in ['fp', 'ncpf', 'sp']:
    agg_pf[f'{k}_pct'] = agg_pf[f'{k}_ms'] / agg_pf['total_pf_ms'] * 100 if agg_pf['total_pf_ms'] > 0 else 0

# ---- Bar chart ----
fig, ax = plt.subplots(figsize=(10, 6))
labels = [f'Puzzle {p}' for p in PUZZLE_IDS] + ['Aggregate']

fp_vals = np.array([prefill_only_data[p]['fp_ms'] for p in PUZZLE_IDS] + [agg_pf['fp_ms']])
ncpf_vals = np.array([prefill_only_data[p]['ncpf_ms'] for p in PUZZLE_IDS] + [agg_pf['ncpf_ms']])
sp_vals = np.array([prefill_only_data[p]['sp_ms'] for p in PUZZLE_IDS] + [agg_pf['sp_ms']])

x = np.arange(len(labels))
w = 0.5

b1 = ax.bar(x, fp_vals, w, label='Full Prefill (T0 only)', color='#4CAF50', edgecolor='white')
b2 = ax.bar(x, ncpf_vals, w, bottom=fp_vals, label='No-Cache Re-Prefill (every turn)', color='#2196F3', edgecolor='white')
b3 = ax.bar(x, sp_vals, w, bottom=fp_vals + ncpf_vals, label='Post-Compression Prefill', color='#FF9800', edgecolor='white')

# Annotate percentages
all_data = [prefill_only_data[p] for p in PUZZLE_IDS] + [agg_pf]
for i, d in enumerate(all_data):
    total = d['total_pf_ms'] if isinstance(d, dict) else d
    if isinstance(d, dict):
        total = d['total_pf_ms']
    else:
        total = d
    if total == 0:
        continue
    # Full Prefill %
    if fp_vals[i] > 0:
        ax.text(i, fp_vals[i]/2, f"{fp_vals[i]/total*100:.1f}%", ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    # No-Cache %
    if ncpf_vals[i] > 0:
        ax.text(i, fp_vals[i] + ncpf_vals[i]/2, f"{ncpf_vals[i]/total*100:.1f}%", ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    # Semi-Prefill %
    if sp_vals[i] > 0:
        ax.text(i, fp_vals[i] + ncpf_vals[i] + sp_vals[i]/2, f"{sp_vals[i]/total*100:.1f}%", ha='center', va='center', fontsize=9, color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Prefill Latency (ms) — Measured TTFT')
ax.set_title('Prefill-Only Breakdown (No Prefix Cache, Measured TTFT)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/prefill_only_bar.png")
plt.close()
print("Chart 9: prefill_only_bar.png")

# ---- Pie charts ----
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
pf_colors = ['#4CAF50', '#2196F3', '#FF9800']
pf_labels = ['Full Prefill\n(T0 only)', 'No-Cache Re-Prefill\n(every turn)', 'Post-Comp.\nPrefill']

for ax, pid in zip(axes[:3], PUZZLE_IDS):
    d = prefill_only_data[pid]
    sizes = [d['fp_ms'], d['ncpf_ms'], d['sp_ms']]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, pf_labels, pf_colors) if s > 0]
    if non_zero:
        sz, lb, cl = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(sz, labels=lb, colors=cl, autopct='%1.1f%%',
                                           textprops={'fontsize': 8}, pctdistance=0.7)
        for t in autotexts:
            t.set_fontsize(7)
    ax.set_title(f'Puzzle {pid}\n({d["total_pf_ms"]/1000:.1f}s prefill total)', fontsize=11)

# Aggregate pie
d = agg_pf
sizes = [d['fp_ms'], d['ncpf_ms'], d['sp_ms']]
sz, lb, cl = zip(*[(s, l, c) for s, l, c in zip(sizes, pf_labels, pf_colors) if s > 0])
wedges, texts, autotexts = axes[3].pie(sz, labels=lb, colors=cl, autopct='%1.1f%%',
                                        textprops={'fontsize': 8}, pctdistance=0.7)
for t in autotexts:
    t.set_fontsize(7)
axes[3].set_title(f'Aggregate\n({agg_pf["total_pf_ms"]/1000:.1f}s prefill total)', fontsize=11)

fig.suptitle('Prefill-Only Latency Distribution (No Prefix Cache, Measured TTFT)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/prefill_only_pie.png")
plt.close()
print("Chart 10: prefill_only_pie.png")

# ── Print Summary ───────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'Puzzle':>8} {'Turns':>6} {'Comp':>5} {'Rate':>7} {'delta_med':>9} {'delta_avg':>9} {'B2+C1 avg':>10} {'SumTime avg':>12}")
for pid in PUZZLE_IDS:
    s = all_stats[pid]
    b2c1_avg = sum(e['B2_C1'] for e in s['sp_events']) / max(len(s['sp_events']), 1)
    st_avg = sum(e['summary_time_s'] for e in s['sp_events']) / max(len(s['sp_events']), 1)
    print(f"  P{pid:>5} {s['n_turns']:>6} {s['n_compressions']:>5} {s['compression_rate']:>6.1%} "
          f"{s['delta_median']:>9.0f} {s['delta_mean']:>9.0f} {b2c1_avg:>10.0f} {st_avg:>11.1f}s")

print(f"\nPhase Breakdown (theoretical, No Prefix Cache):")
print(f"{'Puzzle':>8} {'Total(s)':>10} {'FullPF%':>8} {'NoCachePF%':>10} {'SemiPF%':>8} {'Decode%':>8}")
for pid in PUZZLE_IDS:
    p = phase_data[pid]
    print(f"  P{pid:>5} {p['total_ms']/1000:>10.0f} {p['full_prefill_pct']:>7.1f}% "
          f"{p['incremental_prefill_pct']:>9.1f}% {p['semi_prefill_pct']:>7.1f}% {p['decode_pct']:>7.1f}%")

total_all = sum(phase_data[pid]['total_ms'] for pid in PUZZLE_IDS)
print(f"  {'Agg':>5} {total_all/1000:>10.0f} {agg_fp/total_all*100:>7.1f}% "
      f"{agg_ip/total_all*100:>9.1f}% {agg_sp/total_all*100:>7.1f}% {agg_dc/total_all*100:>7.1f}%")

print(f"\nSemi-Prefill details:")
print(f"  {'Puzzle':>8} {'Event':>6} {'Turn':>5} {'PreTok':>7} {'PostTok':>7} {'B2':>6} {'C1':>6} {'B2+C1':>7} {'SP(s)':>7} {'SumTime(s)':>10}")
for pid in PUZZLE_IDS:
    s = all_stats[pid]
    for j, e in enumerate(s['sp_events']):
        sp_s = prefill_cost(e['B2_C1']) / 1000
        print(f"  P{pid:>5} #{j+1:>4} T{e['turn']:>3} {e['pre_tokens']:>7} {e['post_tokens']:>7} "
              f"{e['B2']:>6} {e['C1']:>6} {e['B2_C1']:>7} {sp_s:>6.1f}s {e['summary_time_s']:>9.1f}s")

print(f"\nDone. All charts saved to: {CHART_DIR}")
