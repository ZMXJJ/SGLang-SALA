import os
import json
import glob
import random

PATHS = {
    "gpqa": "/user/linbiyuan/opencompass_results/outputs/MiniCPM-SALA-FixBOS/20260213_015619-sglang_minicpm4_think_sft_dense_fix_bos-gpqa_openai_simple_evals_fulldetail_gen_5aeece-jobid175275/predictions/MiniCPM-SALA-FixBOS-minicpm4-think-sft-sglang-dense-fixbos/",
    "ruler32k": "/user/linbiyuan/opencompass_results/outputs/MiniCPM-SALA-FixBOS/20260220_132254-sglang_minicpm4_think_sft_mix_flash-ruler_32k_noinst_1000_gen_f0949a-jobid182587/predictions/MiniCPM-SALA-FixBOS-minicpm4-think-sft-sglang-mix-flash/",
    "ruler64k": "/user/linbiyuan/opencompass_results/outputs/MiniCPM-SALA-FixBOS/20260220_144857-sglang_minicpm4_think_sft_dense-ruler_64k_noinst_1000_gen_f0949a-jobid182611/predictions/MiniCPM-SALA-FixBOS-minicpm4-think-sft-sglang-dense/",
    "ruler128k": "/user/linbiyuan/opencompass_results/outputs/MiniCPM-SALA-FixBOS/20260220_145201-sglang_minicpm4_think_sft_dense-ruler_128k_noinst_1000_gen_f0949a-jobid182612/predictions/MiniCPM-SALA-FixBOS-minicpm4-think-sft-sglang-dense/"
}

OUTPUT_DIR = "data"
FILES = {
    "public": os.path.join(OUTPUT_DIR, "perf_public_set.jsonl"),
    "private": os.path.join(OUTPUT_DIR, "perf_private_set.jsonl"),
    "public_source": os.path.join(OUTPUT_DIR, "perf_public_set_source.jsonl"),
    "private_source": os.path.join(OUTPUT_DIR, "perf_private_set_source.jsonl"),
}

SEED = 42

# 目标：每个集合 150 题（5个任务 * 30/任务），并排除 vt
PER_SPLIT_PER_TASK = 30
TOTAL_PER_TASK = PER_SPLIT_PER_TASK * 2  # public + private
EXCLUDED_TASKS = {"vt"}

# 过滤掉历史上“打满输出”的样本（来自 opencompass 预测的 res_length）
# 81929 这种通常意味着模型崩溃/复读直到 max_new_tokens 被截断，强烈影响评测耗时
CAP_HIT_THRESHOLD = 80000
DEFAULT_MAX_OUTPUT_LEN = 32768  # 统一将历史输出长度限制在 32k 以内
MAX_OUTPUT_LEN_BY_TASK = {
    # 如需对某些任务更严格，可在这里单独覆写（默认所有任务 <= DEFAULT_MAX_OUTPUT_LEN）
    "mcq": 32768,
}

def get_percentile(data, p):
    if not data: return 0
    s = sorted(data)
    idx = int((len(s) - 1) * p / 100.0)
    return s[idx]

def get_mean(data):
    if not data: return 0
    return sum(data) / len(data)

def get_quantile_indices(n: int, k: int):
    """Return k indices evenly covering [0, n-1]."""
    if k <= 0 or n <= 0:
        return []
    if k == 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]

def ks_distance(a, b):
    """Two-sample KS distance (max CDF gap)."""
    if not a or not b:
        return 0.0
    a = sorted(a)
    b = sorted(b)
    i = j = 0
    na = len(a)
    nb = len(b)
    d = 0.0
    while i < na and j < nb:
        x = a[i] if a[i] <= b[j] else b[j]
        while i < na and a[i] <= x:
            i += 1
        while j < nb and b[j] <= x:
            j += 1
        fa = i / na
        fb = j / nb
        d = max(d, abs(fa - fb))
    # tail
    d = max(d, abs(1.0 - j / nb), abs(i / na - 1.0))
    return float(d)

def infer_ctx_from_path_key(path_key: str):
    if "32k" in path_key:
        return "32k"
    if "64k" in path_key:
        return "64k"
    if "128k" in path_key:
        return "128k"
    return None

def infer_subtask_from_filename(jf: str):
    """e.g. ruler_niah_single_1_32k_0.json -> ruler_niah_single_1"""
    name = os.path.splitext(os.path.basename(jf))[0]
    parts = name.split("_")
    if len(parts) >= 3 and parts[-2] in {"32k", "64k", "128k"}:
        return "_".join(parts[:-2])
    return name

def get_generic_task_name(original_task, filename):
    filename = os.path.basename(filename).lower()
    
    if "gpqa" in original_task or "gpqa" in filename:
        return "mcq"
    
    if "niah" in filename:
        return "niah"
    elif "cwe" in filename:
        return "cwe"
    elif "fwe" in filename:
        return "fwe"
    elif "qa" in filename:
        return "qa"
    elif "vt" in filename:
        return "vt"
        
    if "ruler" in original_task:
        return "lcx"
        
    return "unknown"

def calculate_stats(items, task_type):
    if not items:
        return {"count": 0}
        
    lengths = [item['input_length'] for item in items if item.get('input_length') is not None]
    out_lengths = [item.get('output_length', 0) for item in items]
    cap_hits = [1 for x in out_lengths if x is not None and x >= CAP_HIT_THRESHOLD]
    
    stats = {
        "count": len(items),
        "input_len_mean": get_mean(lengths),
        "input_len_median": get_percentile(lengths, 50),
        "input_len_p90": get_percentile(lengths, 90),
        "input_len_min": min(lengths) if lengths else 0,
        "input_len_max": max(lengths) if lengths else 0,
        
        "output_len_mean": get_mean(out_lengths),
        "output_len_median": get_percentile(out_lengths, 50),
        "output_len_p90": get_percentile(out_lengths, 90),
        "output_len_p95": get_percentile(out_lengths, 95),
        "output_len_p99": get_percentile(out_lengths, 99),
        "output_len_min": min(out_lengths) if out_lengths else 0,
        "output_len_max": max(out_lengths) if out_lengths else 0,
        "cap_hit_count": sum(cap_hits),
        "cap_hit_rate": (sum(cap_hits) / len(out_lengths)) if out_lengths else 0,
    }
    
    if task_type == "mcq":
        golds = [item['gold'] for item in items if item.get('gold')]
        dist = {}
        for g in golds:
            g = str(g).upper().strip()
            dist[g] = dist.get(g, 0) + 1
        stats["gold_dist"] = dist
        
    return stats

def process_item_raw(item, task_name, jf, path_key: str):
    prompt_content = item.get('origin_prompt')
    input_length = item.get('all_input_length') or item.get('input_length')
    output_length = item.get('res_length', 0)
    
    final_prompt = prompt_content
    if isinstance(prompt_content, list):
        if len(prompt_content) == 1:
            if 'prompt' in prompt_content[0]:
                final_prompt = prompt_content[0]['prompt']
                if input_length is None:
                    input_length = prompt_content[0].get('input_length')
    if input_length is None and isinstance(prompt_content, list) and prompt_content:
        input_length = prompt_content[-1].get('input_length') if prompt_content else 0
            
    generic_task = get_generic_task_name(task_name, jf)
    ctx = infer_ctx_from_path_key(path_key)
    subtask = infer_subtask_from_filename(jf)

    return {
        "prompt": final_prompt,
        "input_length": input_length,
        "output_length": output_length,
        "task": generic_task,
        "gold": item.get('gold'),
        "source": item.get('origin_id'),
        "ctx": ctx,
        "subtask": subtask,
        "path_key": path_key,
    }

def generate_meta_info(public_items, private_items):
    lines = [
        "# Dataset Meta Info", 
        "", 
        f"**Random Seed:** {SEED}",
        f"**Per Split / Task:** {PER_SPLIT_PER_TASK}",
        f"**Excluded Tasks:** {sorted(list(EXCLUDED_TASKS))}",
        f"**CAP_HIT_THRESHOLD:** {CAP_HIT_THRESHOLD}",
        f"**DEFAULT_MAX_OUTPUT_LEN:** {DEFAULT_MAX_OUTPUT_LEN}",
        f"**MAX_OUTPUT_LEN_BY_TASK:** {json.dumps(MAX_OUTPUT_LEN_BY_TASK, sort_keys=True)}",
        f"**Distribution Plot:** {os.path.join(OUTPUT_DIR, 'io_length_dist.png')}",
        "",
        "This document compares the distribution of the Public and Private datasets.", 
        ""
    ]
    
    tasks = set(x['task'] for x in public_items + private_items)
    
    for task in sorted(list(tasks)):
        lines.append(f"## Task: {task}")
        
        pub_task_items = [x for x in public_items if x['task'] == task]
        pri_task_items = [x for x in private_items if x['task'] == task]
        
        pub_stats = calculate_stats(pub_task_items, task)
        pri_stats = calculate_stats(pri_task_items, task)
        
        lines.append("| Metric | Public Set | Private Set |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| Count | {pub_stats['count']} | {pri_stats['count']} |")
        lines.append(f"| Input Len (Mean) | {pub_stats.get('input_len_mean', 0):.2f} | {pri_stats.get('input_len_mean', 0):.2f} |")
        lines.append(f"| Input Len (Median) | {pub_stats.get('input_len_median', 0):.0f} | {pri_stats.get('input_len_median', 0):.0f} |")
        lines.append(f"| Input Len (P90) | {pub_stats.get('input_len_p90', 0):.0f} | {pri_stats.get('input_len_p90', 0):.0f} |")
        lines.append(f"| Output Len (Mean) | {pub_stats.get('output_len_mean', 0):.2f} | {pri_stats.get('output_len_mean', 0):.2f} |")
        lines.append(f"| Output Len (Median) | {pub_stats.get('output_len_median', 0):.0f} | {pri_stats.get('output_len_median', 0):.0f} |")
        lines.append(f"| Output Len (P90) | {pub_stats.get('output_len_p90', 0):.0f} | {pri_stats.get('output_len_p90', 0):.0f} |")
        lines.append(f"| Output Len (P95) | {pub_stats.get('output_len_p95', 0):.0f} | {pri_stats.get('output_len_p95', 0):.0f} |")
        lines.append(f"| Output Len (P99) | {pub_stats.get('output_len_p99', 0):.0f} | {pri_stats.get('output_len_p99', 0):.0f} |")
        lines.append(f"| Output Len (Range) | {pub_stats.get('output_len_min', 0)} - {pub_stats.get('output_len_max', 0)} | {pri_stats.get('output_len_min', 0)} - {pri_stats.get('output_len_max', 0)} |")
        lines.append(f"| Input Len (Range) | {pub_stats.get('input_len_min', 0)} - {pub_stats.get('input_len_max', 0)} | {pri_stats.get('input_len_min', 0)} - {pri_stats.get('input_len_max', 0)} |")
        lines.append(f"| CAP Hit (Count/Rate) | {pub_stats.get('cap_hit_count', 0)}/{pub_stats.get('cap_hit_rate', 0):.2%} | {pri_stats.get('cap_hit_count', 0)}/{pri_stats.get('cap_hit_rate', 0):.2%} |")

        # ctx distribution (rough check)
        def _ctx_dist(items):
            d = {}
            for x in items:
                k = x.get("ctx") or "none"
                d[k] = d.get(k, 0) + 1
            return d
        pub_ctx = json.dumps(_ctx_dist(pub_task_items), sort_keys=True)
        pri_ctx = json.dumps(_ctx_dist(pri_task_items), sort_keys=True)
        lines.append(f"| Ctx Dist | {pub_ctx} | {pri_ctx} |")

        # KS distance for input/output length
        pub_in = [x.get("input_length", 0) or 0 for x in pub_task_items]
        pri_in = [x.get("input_length", 0) or 0 for x in pri_task_items]
        pub_out = [x.get("output_length", 0) or 0 for x in pub_task_items]
        pri_out = [x.get("output_length", 0) or 0 for x in pri_task_items]
        lines.append(
            f"| KS Dist (In/Out) | {ks_distance(pub_in, pri_in):.3f} / {ks_distance(pub_out, pri_out):.3f} | {ks_distance(pub_in, pri_in):.3f} / {ks_distance(pub_out, pri_out):.3f} |"
        )
        
        if "gold_dist" in pub_stats:
            pub_dist = json.dumps(pub_stats["gold_dist"], sort_keys=True)
            pri_dist = json.dumps(pri_stats.get("gold_dist", {}), sort_keys=True)
            lines.append(f"| Answer Dist | {pub_dist} | {pri_dist} |")
            
        lines.append("")
        
    meta_path = os.path.join(OUTPUT_DIR, "meta_info.md")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated {meta_path}")

def save_distribution_plot(public_items, private_items, out_path: str):
    """Save an overview plot of input/output token length distributions."""
    import math
    from collections import defaultdict

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def _by_task(items):
        d = defaultdict(list)
        for x in items:
            d[x["task"]].append(x)
        return d

    pub_by = _by_task(public_items)
    pri_by = _by_task(private_items)

    tasks = sorted(set(pub_by.keys()) | set(pri_by.keys()))
    # Put overall first
    rows = ["__all__"] + tasks

    # bins
    in_bins = np.logspace(2, math.log10(140000), 26)  # ~100 .. 140k
    out_bins = np.logspace(0, math.log10(DEFAULT_MAX_OUTPUT_LEN), 26)  # 1 .. 32k

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=2,
        figsize=(12, 2.6 * len(rows)),
        constrained_layout=True,
    )

    def _lengths(items, key):
        arr = []
        for x in items:
            v = x.get(key, 0) or 0
            if v <= 0:
                v = 1
            arr.append(v)
        return arr

    for r, name in enumerate(rows):
        if name == "__all__":
            pub_items = public_items
            pri_items = private_items
            title_prefix = "ALL"
        else:
            pub_items = pub_by.get(name, [])
            pri_items = pri_by.get(name, [])
            title_prefix = name

        pub_in = _lengths(pub_items, "input_length")
        pri_in = _lengths(pri_items, "input_length")
        pub_out = _lengths(pub_items, "output_length")
        pri_out = _lengths(pri_items, "output_length")

        ax_in = axes[r][0]
        ax_out = axes[r][1]

        ax_in.hist(pub_in, bins=in_bins, histtype="step", linewidth=1.8, label="public")
        ax_in.hist(pri_in, bins=in_bins, histtype="step", linewidth=1.8, label="private")
        ax_in.set_xscale("log")
        ax_in.set_ylabel("count")
        ax_in.set_title(f"{title_prefix} input (KS={ks_distance(pub_in, pri_in):.3f})")

        ax_out.hist(pub_out, bins=out_bins, histtype="step", linewidth=1.8, label="public")
        ax_out.hist(pri_out, bins=out_bins, histtype="step", linewidth=1.8, label="private")
        ax_out.set_xscale("log")
        ax_out.set_title(f"{title_prefix} output (KS={ks_distance(pub_out, pri_out):.3f})")

        if r == 0:
            ax_in.legend(loc="upper right")
            ax_out.legend(loc="upper right")

    axes[0][0].set_xlabel("tokens (log)")
    axes[0][1].set_xlabel("tokens (log)")
    for r in range(1, len(rows)):
        axes[r][0].set_xlabel("tokens (log)")
        axes[r][1].set_xlabel("tokens (log)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved distribution plot to {out_path}")

def save_boxplot_plot(public_items, private_items, out_path: str):
    """Save boxplots for input/output token lengths (public vs private)."""
    from collections import defaultdict

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _by_task(items):
        d = defaultdict(list)
        for x in items:
            d[x["task"]].append(x)
        return d

    pub_by = _by_task(public_items)
    pri_by = _by_task(private_items)

    tasks = sorted(set(pub_by.keys()) | set(pri_by.keys()))
    # Prefer a stable, human-friendly order when possible
    preferred = ["cwe", "fwe", "mcq", "niah", "qa"]
    if all(t in tasks for t in preferred):
        tasks = preferred

    def _lengths(items, key):
        arr = []
        for x in items:
            v = x.get(key, 0) or 0
            if v <= 0:
                v = 1
            arr.append(v)
        return arr

    # Build data arrays per task
    pub_in = [_lengths(pub_by.get(t, []), "input_length") for t in tasks]
    pri_in = [_lengths(pri_by.get(t, []), "input_length") for t in tasks]
    pub_out = [_lengths(pub_by.get(t, []), "output_length") for t in tasks]
    pri_out = [_lengths(pri_by.get(t, []), "output_length") for t in tasks]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 8),
        constrained_layout=True,
        sharex=True,
    )

    def _draw(ax, data_pub, data_pri, title, y_max=None):
        n = len(tasks)
        centers = list(range(1, n + 1))
        offset = 0.18
        pos_pub = [c - offset for c in centers]
        pos_pri = [c + offset for c in centers]

        bp_pub = ax.boxplot(
            data_pub,
            positions=pos_pub,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#1f77b4", linewidth=1.6),
        )
        bp_pri = ax.boxplot(
            data_pri,
            positions=pos_pri,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#ff7f0e", linewidth=1.6),
        )

        for b in bp_pub["boxes"]:
            b.set(facecolor="#cfe8ff", edgecolor="#1f77b4", linewidth=1.4)
        for b in bp_pri["boxes"]:
            b.set(facecolor="#ffe0c7", edgecolor="#ff7f0e", linewidth=1.4)

        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_title(title)

        if y_max is not None:
            ax.set_ylim(1, y_max)

        # Legend (use the first box artists)
        ax.legend(
            [bp_pub["boxes"][0], bp_pri["boxes"][0]],
            ["public", "private"],
            loc="upper right",
            frameon=True,
        )

        ax.set_xticks(centers)
        ax.set_xticklabels(tasks)

    _draw(axes[0], pub_in, pri_in, "Input length boxplot (log scale)", y_max=140000)
    _draw(axes[1], pub_out, pri_out, f"Output length boxplot (log scale, capped <= {DEFAULT_MAX_OUTPUT_LEN})", y_max=DEFAULT_MAX_OUTPUT_LEN)

    axes[0].set_ylabel("tokens")
    axes[1].set_ylabel("tokens")
    axes[1].set_xlabel("task")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved boxplot to {out_path}")

def filter_items_for_task(task: str, items):
    """Filter out likely time-bombs and invalid samples."""
    max_out = MAX_OUTPUT_LEN_BY_TASK.get(task, DEFAULT_MAX_OUTPUT_LEN)
    filtered = []
    for it in items:
        # require gold for scoring
        if task == "mcq":
            g = it.get("gold")
            if not g:
                continue
            g = str(g).strip().upper()
            if g not in {"A", "B", "C", "D"}:
                continue
        else:
            if it.get("gold") is None:
                continue

        out_len = it.get("output_length", 0) or 0
        # hard drop cap-hit
        if out_len >= CAP_HIT_THRESHOLD:
            continue
        # task-specific max out filter
        if max_out is not None and out_len > max_out:
            continue
        # mcq 输入长度异常样本通常是格式/解析问题，会引入分布偏差且可能拖慢评测
        if task == "mcq":
            in_len = it.get("input_length", 0) or 0
            if in_len > 1200:
                continue
        filtered.append(it)
    return filtered

def balanced_sample(items, k, seed, sort_key):
    """Pick k items covering the distribution by quantiles."""
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    items_sorted = sorted(items, key=sort_key)
    idxs = get_quantile_indices(len(items_sorted), k)
    # Ensure uniqueness (quantile rounding may duplicate indices when k is large)
    picked = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            picked.append(items_sorted[idx])
            seen.add(idx)
    # Fill if needed
    if len(picked) < k:
        rnd = random.Random(seed)
        candidates = [x for i, x in enumerate(items_sorted) if i not in seen]
        rnd.shuffle(candidates)
        picked.extend(candidates[:k - len(picked)])
    return picked[:k]

def pick_min_range_window(items, k: int, seed: int, value_key):
    """
    Pick a contiguous window of size k (after sorting by value_key) that minimizes
    the range (max-min). Useful to narrow output_length variance for perf datasets.
    """
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)

    items_sorted = sorted(items, key=value_key)
    vals = [value_key(x) for x in items_sorted]
    best_range = None
    best_starts = []
    for i in range(0, len(items_sorted) - k + 1):
        r = vals[i + k - 1] - vals[i]
        if best_range is None or r < best_range:
            best_range = r
            best_starts = [i]
        elif r == best_range:
            best_starts.append(i)

    if not best_starts:
        return items_sorted[:k]

    rnd = random.Random(seed)
    start = rnd.choice(best_starts)
    return items_sorted[start:start + k]

def pick_min_output_band_by_gold(by_gold: dict, needs: dict, seed: int):
    """
    Find the narrowest output_length band [lo, hi] such that for every gold letter g,
    there are at least needs[g] items within the band. Then pick exactly needs[g] items
    per letter, preferring those closer to the band center.
    """
    letters = [g for g in sorted(needs.keys()) if needs.get(g, 0) > 0]
    if not letters:
        return {k: [] for k in needs.keys()}

    all_entries = []
    for g in letters:
        for it in by_gold.get(g, []):
            out_len = it.get("output_length", 0) or 0
            all_entries.append((out_len, g, it))
    if not all_entries:
        return None
    all_entries.sort(key=lambda x: x[0])

    counts = {g: 0 for g in letters}
    missing = len(letters)
    l = 0
    best = None  # (range, l, r)

    for r in range(len(all_entries)):
        out_r, g_r, _ = all_entries[r]
        if g_r in counts:
            before = counts[g_r]
            counts[g_r] = before + 1
            if before < needs[g_r] and counts[g_r] >= needs[g_r]:
                missing -= 1

        while missing == 0 and l <= r:
            out_l, g_l, _ = all_entries[l]
            band = out_r - out_l
            if best is None or band < best[0]:
                best = (band, l, r)

            # try shrink from left
            if g_l in counts:
                before = counts[g_l]
                counts[g_l] = before - 1
                if before >= needs[g_l] and counts[g_l] < needs[g_l]:
                    missing += 1
            l += 1

    if best is None:
        return None

    _, bl, br = best
    lo = all_entries[bl][0]
    hi = all_entries[br][0]
    center = (lo + hi) / 2.0
    rnd = random.Random(seed)

    candidates = {g: [] for g in letters}
    for out_len, g, it in all_entries[bl:br + 1]:
        candidates[g].append((out_len, it))

    picked = {}
    for g in letters:
        need = needs[g]
        cands = candidates.get(g, [])
        if len(cands) < need:
            return None
        scored = []
        for out_len, it in cands:
            scored.append((abs(out_len - center), out_len, rnd.random(), it))
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        picked[g] = [x[3] for x in scored[:need]]

    return picked

def split_pairs_evenly(items, seed, sort_key):
    """Split items into two sets with matched distribution via pairing."""
    items_sorted = sorted(items, key=sort_key)
    pub, pri = [], []
    rnd = random.Random(seed)
    for i in range(0, len(items_sorted), 2):
        a = items_sorted[i]
        b = items_sorted[i + 1] if i + 1 < len(items_sorted) else None
        if b is None:
            (pub if rnd.random() < 0.5 else pri).append(a)
            continue
        if rnd.random() < 0.5:
            pub.append(a); pri.append(b)
        else:
            pub.append(b); pri.append(a)
    return pub, pri

def allocate_split_quotas(groups, per_split_total: int, seed: int):
    """
    Allocate per-split quotas across groups, ensuring each group can provide
    2*quota samples (so public/private stay perfectly matched).
    """
    keys = sorted(groups.keys())
    if not keys or per_split_total <= 0:
        return {k: 0 for k in keys}

    rnd = random.Random(seed)

    base = per_split_total // len(keys)
    rem = per_split_total % len(keys)
    quotas = {k: base for k in keys}
    for k in keys[:rem]:
        quotas[k] += 1

    # capacity per group (per split) is floor(len(group)/2)
    deficits = 0
    for k in keys:
        cap = len(groups[k]) // 2
        if quotas[k] > cap:
            deficits += quotas[k] - cap
            quotas[k] = cap

    if deficits > 0:
        # redistribute to groups with remaining capacity
        candidates = keys[:]
        rnd.shuffle(candidates)
        for k in candidates:
            if deficits <= 0:
                break
            cap = len(groups[k]) // 2
            room = cap - quotas[k]
            if room <= 0:
                continue
            take = min(room, deficits)
            quotas[k] += take
            deficits -= take

    # Ensure sum(quotas) == per_split_total if possible
    # If still short (not enough capacity overall), we will just return the max feasible.
    return quotas

def extract():
    all_items = []
    
    for path_key, path in PATHS.items():
        print(f"Processing {path_key} from {path}")
        json_files = glob.glob(os.path.join(path, "*.json"))
        
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for key in data:
                            item = data[key]
                            if isinstance(item, dict) and 'origin_prompt' in item:
                                item['origin_id'] = f"{os.path.basename(jf)}_{key}"
                                item['source_file'] = jf
                                all_items.append(process_item_raw(item, path_key, jf, path_key))
                    elif isinstance(data, list):
                         for i, item in enumerate(data):
                            if isinstance(item, dict) and 'origin_prompt' in item:
                                item['origin_id'] = f"{os.path.basename(jf)}_{i}"
                                item['source_file'] = jf
                                all_items.append(process_item_raw(item, path_key, jf, path_key))
            except Exception as e:
                print(f"Error reading {jf}: {e}")
    
    from collections import defaultdict
    items_by_task = defaultdict(list)
    for item in all_items:
        if item["task"] in EXCLUDED_TASKS:
            continue
        items_by_task[item['task']].append(item)
        
    data_split = {
        "public": [],
        "private": []
    }
    
    for task, items in items_by_task.items():
        items = filter_items_for_task(task, items)
        print(f"Task {task}: kept {len(items)} items after filter")
        for item in items:
            if item['input_length'] is None: item['input_length'] = 0
            if item['output_length'] is None: item['output_length'] = 0

        if task == "mcq":
            # balance by gold letter, enforce exact 30/30 if capacity allows
            by_gold = defaultdict(list)
            for it in items:
                g = str(it.get("gold", "")).strip().upper()
                by_gold[g].append(it)
            letters = ["A", "B", "C", "D"]
            for g in letters:
                by_gold.setdefault(g, [])

            # initial per-split quotas: as even as possible
            base = PER_SPLIT_PER_TASK // len(letters)
            rem = PER_SPLIT_PER_TASK % len(letters)
            per_split_quota = {g: base for g in letters}
            for g in letters[:rem]:
                per_split_quota[g] += 1

            # adjust by capacity (need 2*q per letter)
            deficits = 0
            for g in letters:
                cap = len(by_gold[g]) // 2
                if per_split_quota[g] > cap:
                    deficits += per_split_quota[g] - cap
                    per_split_quota[g] = cap
            if deficits > 0:
                for g in letters:
                    if deficits <= 0:
                        break
                    cap = len(by_gold[g]) // 2
                    room = cap - per_split_quota[g]
                    take = min(room, deficits)
                    per_split_quota[g] += take
                    deficits -= take

            # sample and split per letter
            pub, pri = [], []
            needs = {g: 2 * per_split_quota[g] for g in letters if per_split_quota[g] > 0}
            band_picked = pick_min_output_band_by_gold(by_gold, needs, seed=SEED)

            if band_picked is None:
                # fallback: per-letter narrow window
                for g in letters:
                    g_items = by_gold[g]
                    need = 2 * per_split_quota[g]
                    if need <= 0:
                        continue
                    window = pick_min_range_window(
                        g_items,
                        need,
                        seed=SEED + ord(g),
                        value_key=lambda x: (x.get("output_length", 0) or 0),
                    )
                    picked = sorted(window, key=lambda x: (x.get("output_length", 0) or 0, x.get("input_length", 0) or 0))
                    pub.extend(picked[0::2])
                    pri.extend(picked[1::2])
            else:
                for g in letters:
                    picked = band_picked.get(g, [])
                    if not picked:
                        continue
                    picked = sorted(picked, key=lambda x: (x.get("output_length", 0) or 0, x.get("input_length", 0) or 0))
                    pub.extend(picked[0::2])
                    pri.extend(picked[1::2])

        else:
            # 先按 ctx 均衡（32k/64k/128k），再在 ctx 内按 subtask 均衡（niah/qa）
            ctx_groups = defaultdict(list)
            for it in items:
                ctx = it.get("ctx") or "none"
                ctx_groups[ctx].append(it)

            ctx_keys = [k for k in ["32k", "64k", "128k"] if k in ctx_groups]
            if not ctx_keys:
                ctx_keys = sorted(ctx_groups.keys())

            base = PER_SPLIT_PER_TASK // len(ctx_keys)
            rem = PER_SPLIT_PER_TASK % len(ctx_keys)
            ctx_quota = {k: base for k in ctx_keys}
            for k in ctx_keys[:rem]:
                ctx_quota[k] += 1

            rnd = random.Random(SEED)
            pub, pri = [], []

            for ctx in ctx_keys:
                per_split_ctx = ctx_quota.get(ctx, 0)
                if per_split_ctx <= 0:
                    continue
                ctx_items = ctx_groups[ctx]

                if task in {"niah", "qa"}:
                    sub_groups = defaultdict(list)
                    for it in ctx_items:
                        sub_groups[it.get("subtask") or "none"].append(it)
                    sub_quota = allocate_split_quotas(sub_groups, per_split_ctx, SEED)
                    for sub, q in sorted(sub_quota.items()):
                        if q <= 0:
                            continue
                        g_items = sub_groups[sub]
                        rnd.shuffle(g_items)
                        need = 2 * q
                        picked = balanced_sample(
                            g_items,
                            need,
                            seed=SEED,
                            sort_key=lambda x: (x["input_length"], x["output_length"]),
                        )
                        picked = sorted(picked, key=lambda x: (x["input_length"], x["output_length"]))
                        pub.extend(picked[0::2])
                        pri.extend(picked[1::2])
                else:
                    rnd.shuffle(ctx_items)
                    need = 2 * per_split_ctx
                    picked = balanced_sample(
                        ctx_items,
                        need,
                        seed=SEED,
                        sort_key=lambda x: (x["input_length"], x["output_length"]),
                    )
                    picked = sorted(picked, key=lambda x: (x["input_length"], x["output_length"]))
                    pub.extend(picked[0::2])
                    pri.extend(picked[1::2])

        # enforce exact per-split size if possible
        pub = sorted(pub, key=lambda x: (x["input_length"], x["output_length"]))[:PER_SPLIT_PER_TASK]
        pri = sorted(pri, key=lambda x: (x["input_length"], x["output_length"]))[:PER_SPLIT_PER_TASK]

        data_split["public"].extend(pub)
        data_split["private"].extend(pri)

    generate_meta_info(data_split["public"], data_split["private"])
    save_distribution_plot(
        data_split["public"],
        data_split["private"],
        os.path.join(OUTPUT_DIR, "io_length_dist.png"),
    )
    save_boxplot_plot(
        data_split["public"],
        data_split["private"],
        os.path.join(OUTPUT_DIR, "io_boxplot.png"),
    )

    for split_name, items in data_split.items():
        source_path = FILES[f"{split_name}_source"]
        clean_path = FILES[split_name]
        
        with open(source_path, 'w', encoding='utf-8') as f_src, \
             open(clean_path, 'w', encoding='utf-8') as f_clean:
            
            for item in items:
                f_src.write(json.dumps(item, ensure_ascii=False) + "\n")
                clean_item = item.copy()
                if 'source' in clean_item:
                    del clean_item['source']
                f_clean.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(items)} items to {source_path} and {clean_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    extract()
