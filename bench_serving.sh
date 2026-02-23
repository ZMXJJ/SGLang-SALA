#!/bin/bash
# ============================================================
# benchmark_duration 评测脚本（基于 sglang.bench_serving）
#
# 使用 sglang 官方 bench_serving 工具，在 3 档并发度下
# 分别跑完所有评测请求，记录 Benchmark Duration。
#
# 用法（容器内由 entrypoint.sh 调用）:
#   bash /app/bench_serving.sh <API_BASE> <EVAL_DATA> [MAX_SAMPLES]
#
# 参数:
#   API_BASE     - SGLang API 地址，如 http://127.0.0.1:30000
#   EVAL_DATA    - 评测数据集路径（原始 JSONL）
#   MAX_SAMPLES  - 最多使用的数据条数（可选）
#
# 输出:
#   最后一行输出 JSON: {"S1": xx.xx, "S8": xx.xx, "Smax": xx.xx}
# ============================================================
set -e

API_BASE="${1:?用法: bash bench_serving.sh <API_BASE> <EVAL_DATA> [MAX_SAMPLES]}"
EVAL_DATA="${2:?用法: bash bench_serving.sh <API_BASE> <EVAL_DATA> [MAX_SAMPLES]}"
MAX_SAMPLES="${3:-}"

# 解析 host 和 port
HOST=$(echo "${API_BASE}" | sed -E 's|https?://||' | cut -d: -f1)
PORT=$(echo "${API_BASE}" | sed -E 's|https?://||' | cut -d: -f2)

echo "[bench_serving] API: ${API_BASE} (host=${HOST}, port=${PORT})"
echo "[bench_serving] 数据集: ${EVAL_DATA}"

# ============================================================
# Step 1: 转换数据集格式
# speed_eval.jsonl -> custom 格式（conversations）
# ============================================================
CONVERTED_DATA="/tmp/bench_eval_data.jsonl"

python3 -c "
import json
import sys

input_file = '${EVAL_DATA}'
output_file = '${CONVERTED_DATA}'
max_samples = ${MAX_SAMPLES:-0} or None

data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

if max_samples and len(data) > max_samples:
    data = data[:max_samples]
    print(f'[bench_serving] 数据截断至 {max_samples} 条')

with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        # bench_serving custom 格式: conversations 数组，至少 2 轮
        converted = {
            'conversations': [
                {'role': 'user', 'content': item['input']},
                {'role': 'assistant', 'content': item.get('model_response', 'placeholder')}
            ]
        }
        f.write(json.dumps(converted, ensure_ascii=False) + '\n')

print(f'[bench_serving] 转换完成: {len(data)} 条 -> {output_file}')
"

NUM_PROMPTS=$(wc -l < "${CONVERTED_DATA}")
echo "[bench_serving] 总请求数: ${NUM_PROMPTS}"

# ============================================================
# Step 2: 在 3 档并发度下分别运行 bench_serving
# ============================================================
RESULT_JSON="{}"

# 定义 3 档测试: S1, S8, Smax(不限并发)
BENCH_CONFIGS="S1:1 S8:8 Smax:"

for CONFIG in ${BENCH_CONFIGS}; do
    LABEL=$(echo "${CONFIG}" | cut -d: -f1)
    CONC=$(echo "${CONFIG}" | cut -d: -f2)

    echo ""
    echo "────────────────────────────────────────────────────────────"
    if [ -n "${CONC}" ]; then
        echo "  [${LABEL}] 开始测试 - 并发度: ${CONC}, 共 ${NUM_PROMPTS} 条请求"
    else
        echo "  [${LABEL}] 开始测试 - 无并发上限, 共 ${NUM_PROMPTS} 条请求"
    fi
    echo "────────────────────────────────────────────────────────────"

    # 构建 bench_serving 命令
    BENCH_CMD="python3 -m sglang.bench_serving \
        --backend sglang \
        --host ${HOST} \
        --port ${PORT} \
        --dataset-name custom \
        --dataset-path ${CONVERTED_DATA} \
        --num-prompts ${NUM_PROMPTS} \
        --flush-cache"

    # 有并发限制时加 --max-concurrency，否则不加（全部请求同时发送）
    if [ -n "${CONC}" ]; then
        BENCH_CMD="${BENCH_CMD} --max-concurrency ${CONC}"
    fi

    BENCH_OUTPUT=$(eval ${BENCH_CMD} 2>&1) || true

    echo "${BENCH_OUTPUT}"

    # 提取 Benchmark duration (s): 行的值
    DURATION=$(echo "${BENCH_OUTPUT}" | grep -oP 'Benchmark duration \(s\):\s+\K[0-9.]+' || echo "0")

    echo "  [${LABEL}] Benchmark duration: ${DURATION}s"

    # 追加到结果 JSON
    RESULT_JSON=$(python3 -c "
import json
result = json.loads('${RESULT_JSON}')
result['${LABEL}'] = float('${DURATION}')
print(json.dumps(result))
")
done

# ============================================================
# Step 3: 输出汇总
# ============================================================
echo ""
echo "============================================================"
echo "  Benchmark Duration 汇总"
echo "============================================================"

python3 -c "
import json
result = json.loads('${RESULT_JSON}')
for key in ['S1', 'S8', 'Smax']:
    print(f'  {key:>4s}: {result.get(key, 0):>10.2f}s')
"

echo "============================================================"

# 最后一行输出 JSON（供 entrypoint.sh 捕获）
echo "${RESULT_JSON}"
