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

OUTPUT_DIR = "acc_datasets"
FILES = {
    "public": os.path.join(OUTPUT_DIR, "public_set.jsonl"),
    "private": os.path.join(OUTPUT_DIR, "private_set.jsonl"),
    "public_source": os.path.join(OUTPUT_DIR, "public_set_source.jsonl"),
    "private_source": os.path.join(OUTPUT_DIR, "private_set_source.jsonl"),
}

SAMPLES_PER_TASK = 60
SEED = 42

def get_percentile(data, p):
    if not data: return 0
    s = sorted(data)
    idx = int((len(s) - 1) * p / 100.0)
    return s[idx]

def get_mean(data):
    if not data: return 0
    return sum(data) / len(data)

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
    
    stats = {
        "count": len(items),
        "input_len_mean": get_mean(lengths),
        "input_len_median": get_percentile(lengths, 50),
        "input_len_min": min(lengths) if lengths else 0,
        "input_len_max": max(lengths) if lengths else 0,
        
        "output_len_mean": get_mean(out_lengths),
        "output_len_median": get_percentile(out_lengths, 50),
        "output_len_p90": get_percentile(out_lengths, 90),
        "output_len_p99": get_percentile(out_lengths, 99),
        "output_len_min": min(out_lengths) if out_lengths else 0,
        "output_len_max": max(out_lengths) if out_lengths else 0,
    }
    
    if task_type == "mcq":
        golds = [item['gold'] for item in items if item.get('gold')]
        dist = {}
        for g in golds:
            g = str(g).upper().strip()
            dist[g] = dist.get(g, 0) + 1
        stats["gold_dist"] = dist
        
    return stats

def process_item_raw(item, task_name, jf):
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

    return {
        "prompt": final_prompt,
        "input_length": input_length,
        "output_length": output_length,
        "task": generic_task,
        "gold": item.get('gold'),
        "source": item.get('origin_id')
    }

def generate_meta_info(public_items, private_items):
    lines = [
        "# Dataset Meta Info", 
        "", 
        f"**Random Seed:** {SEED}",
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
        lines.append(f"| Output Len (Mean) | {pub_stats.get('output_len_mean', 0):.2f} | {pri_stats.get('output_len_mean', 0):.2f} |")
        lines.append(f"| Output Len (Median) | {pub_stats.get('output_len_median', 0):.0f} | {pri_stats.get('output_len_median', 0):.0f} |")
        lines.append(f"| Output Len (P90) | {pub_stats.get('output_len_p90', 0):.0f} | {pri_stats.get('output_len_p90', 0):.0f} |")
        lines.append(f"| Output Len (P99) | {pub_stats.get('output_len_p99', 0):.0f} | {pri_stats.get('output_len_p99', 0):.0f} |")
        lines.append(f"| Output Len (Range) | {pub_stats.get('output_len_min', 0)} - {pub_stats.get('output_len_max', 0)} | {pri_stats.get('output_len_min', 0)} - {pri_stats.get('output_len_max', 0)} |")
        lines.append(f"| Input Len (Range) | {pub_stats.get('input_len_min', 0)} - {pub_stats.get('input_len_max', 0)} | {pri_stats.get('input_len_min', 0)} - {pri_stats.get('input_len_max', 0)} |")
        
        if "gold_dist" in pub_stats:
            pub_dist = json.dumps(pub_stats["gold_dist"], sort_keys=True)
            pri_dist = json.dumps(pri_stats.get("gold_dist", {}), sort_keys=True)
            lines.append(f"| Answer Dist | {pub_dist} | {pri_dist} |")
            
        lines.append("")
        
    with open("acc_datasets/meta_info.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Generated acc_datasets/meta_info.md")

def extract():
    all_items = []
    
    for task_name, path in PATHS.items():
        print(f"Processing {task_name} from {path}")
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
                                all_items.append(process_item_raw(item, task_name, jf))
                    elif isinstance(data, list):
                         for i, item in enumerate(data):
                            if isinstance(item, dict) and 'origin_prompt' in item:
                                item['origin_id'] = f"{os.path.basename(jf)}_{i}"
                                item['source_file'] = jf
                                all_items.append(process_item_raw(item, task_name, jf))
            except Exception as e:
                print(f"Error reading {jf}: {e}")
    
    from collections import defaultdict
    items_by_task = defaultdict(list)
    for item in all_items:
        items_by_task[item['task']].append(item)
        
    data_split = {
        "public": [],
        "private": []
    }
    
    for task, items in items_by_task.items():
        print(f"Task {task}: found {len(items)} items")
        for item in items:
            if item['input_length'] is None: item['input_length'] = 0
            if item['output_length'] is None: item['output_length'] = 0
            
        random.seed(SEED)
        random.shuffle(items)
        
        items.sort(key=lambda x: (x['input_length'], x['output_length']))
        
        if len(items) > SAMPLES_PER_TASK:
            step = len(items) / SAMPLES_PER_TASK
            sampled = [items[int(i * step)] for i in range(SAMPLES_PER_TASK)]
        else:
            sampled = items
            
        for i, item in enumerate(sampled):
            if i % 2 == 0:
                data_split["public"].append(item)
            else:
                data_split["private"].append(item)

    generate_meta_info(data_split["public"], data_split["private"])

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
