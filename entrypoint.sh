#!/bin/bash
# ============================================================
#  评测 Docker 入口脚本
#
# 流程:
#   1. 启动 SGLang 推理服务（后台）
#   2. 等待服务就绪
#   3. 运行 gpqa_eval.py 获取 acc
#   4. 运行 bench_serving.sh（基于 sglang.bench_serving）获取 benchmark_duration
#   5. 组装 JSON 结果输出到 stdout
#   6. 关闭 SGLang 服务
#
# 环境变量（后端传入）:
#   MODEL_PATH   - 模型路径（必须）
#   PORT         - SGLang 服务端口（默认 30000）
#   DATA_DIR     - GPQA 数据集目录（默认 /data/gpqa）
#   EVAL_DATA    - bench_serving 数据集路径（默认 /data/eval_data_128.jsonl）
#   RECORD_ID    - 提交记录ID
#   USER_ID      - 用户ID
#   TASK_ID      - 任务ID
#   GPQA_MAX_SAMPLES  - GPQA 每子集最多评测题数（可选，默认全部）
#   BENCH_MAX_SAMPLES - bench_serving 最多数据条数（可选，默认全部）
# ============================================================
set -e

# ---- 参数 ----
MODEL_PATH="${MODEL_PATH:?环境变量 MODEL_PATH 未设置}"
PORT="${PORT:-30000}"
DATA_DIR="${DATA_DIR:-/data/gpqa}"
EVAL_DATA="${EVAL_DATA:-/data/eval_data_128.jsonl}"
RECORD_ID="${RECORD_ID:-}"
USER_ID="${USER_ID:-}"
TASK_ID="${TASK_ID:-}"
GPQA_MAX_SAMPLES="${GPQA_MAX_SAMPLES:-}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-}"

API_BASE="http://127.0.0.1:${PORT}"
VENV_DIR="/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env"

# ---- 激活虚拟环境 ----
echo "[entrypoint] 激活虚拟环境: ${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# ---- 函数: 输出最终 JSON ----
output_result() {
    local state="$1"
    local error_msg="$2"
    local acc="$3"
    local bench_json="$4"

    python3 -c "
import json, sys

error_msg = sys.argv[1]
acc = float(sys.argv[2])
bench_json = sys.argv[3]
state = sys.argv[4]

# 解析 benchmark_duration 各档数据
try:
    bench = json.loads(bench_json)
except Exception:
    bench = {'S1': 0, 'S8': 0, 'Smax': 0}

output = {
    'record_id': '${RECORD_ID}',
    'user_id': '${USER_ID}',
    'result': {
        'error_msg': error_msg,
        'score': {
            'acc': acc,
            'benchmark_duration': {
                'S1': bench.get('S1', 0),
                'S8': bench.get('S8', 0),
                'Smax': bench.get('Smax', 0)
            }
        },
        'sort_by': 'acc'
    },
    'state': state,
    'task_id': '${TASK_ID}'
}
print(json.dumps(output, ensure_ascii=False, indent=2))
" "$error_msg" "$acc" "$bench_json" "$state"
}

# ---- 函数: 清理 SGLang 进程 ----
cleanup() {
    if [ -n "${SGLANG_PID}" ] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "[entrypoint] 正在关闭 SGLang 服务 (PID: ${SGLANG_PID}) ..."
        kill "${SGLANG_PID}" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
        echo "[entrypoint] SGLang 服务已关闭"
    fi
}
trap cleanup EXIT

# ============================================================
# Step 1: 启动 SGLang 推理服务
# ============================================================
echo "[entrypoint] 启动 SGLang 服务 ..."
echo "  模型路径: ${MODEL_PATH}"
echo "  端口:     ${PORT}"

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 128 \
    --skip-server-warmup \
    --port "${PORT}" \
    --dense-as-sparse \
    --enable-metrics &

SGLANG_PID=$!
echo "[entrypoint] SGLang 进程 PID: ${SGLANG_PID}"

# ============================================================
# Step 2: 等待服务就绪
# ============================================================
echo "[entrypoint] 等待服务就绪 ..."
MAX_RETRIES=120
RETRY_COUNT=0

while true; do
    # 1) /health 必须返回 200
    if curl -sf "${API_BASE}/health" > /dev/null 2>&1; then
        # 2) /v1/models 也要可访问且返回可解析 JSON（避免服务刚起但模型未就绪）
        if curl -sf "${API_BASE}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'data' in d and len(d['data'])>0" 2>/dev/null; then
            break
        fi
    fi

    # 检查进程是否意外退出
    if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "[entrypoint] [ERROR] SGLang 进程已退出"
        output_result "0" "SGLang 服务启动失败" "0" '{"S1":0,"S8":0,"Smax":0}'
        exit 1
    fi

    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "[entrypoint] [ERROR] 等待超时 (${MAX_RETRIES} 次重试)"
        output_result "0" "SGLang 服务启动超时" "0" '{"S1":0,"S8":0,"Smax":0}'
        exit 1
    fi
    echo "  等待中... (${RETRY_COUNT}/${MAX_RETRIES})"
done
echo "[entrypoint] SGLang 服务已就绪！"

# ============================================================
# Step 3: 运行 GPQA 评测
# ============================================================
echo "[entrypoint] 开始 GPQA 评测 ..."
ACC=0
EVAL_ERROR=""

EVAL_OUTPUT_DIR="/tmp/gpqa_outputs"

EVAL_CMD="python3 /app/gpqa_eval.py \
    --api_base ${API_BASE} \
    --data_dir ${DATA_DIR} \
    --subsets diamond \
    --output_dir ${EVAL_OUTPUT_DIR} \
    --concurrency 8 \
    --max_tokens 16384"

if [ -n "${GPQA_MAX_SAMPLES}" ]; then
    EVAL_CMD="${EVAL_CMD} --max_samples ${GPQA_MAX_SAMPLES}"
    echo "[entrypoint] GPQA 每个子集最多评测 ${GPQA_MAX_SAMPLES} 题"
fi

if eval ${EVAL_CMD}; then

    # 从 summary.json 提取 overall_accuracy
    if [ -f "${EVAL_OUTPUT_DIR}/summary.json" ]; then
        ACC=$(python3 -c "
import json
with open('${EVAL_OUTPUT_DIR}/summary.json') as f:
    data = json.load(f)
print(data.get('overall_accuracy', 0))
")
        echo "[entrypoint] GPQA 评测完成，overall accuracy: ${ACC}"
    else
        EVAL_ERROR="评测完成但未找到 summary.json"
        echo "[entrypoint] [WARN] ${EVAL_ERROR}"
    fi
else
    EVAL_ERROR="GPQA 评测脚本执行失败"
    echo "[entrypoint] [ERROR] ${EVAL_ERROR}"
fi

# ============================================================
# Step 4: 运行 benchmark_duration (sglang.bench_serving)
# ============================================================
echo "[entrypoint] 运行 benchmark_duration (sglang.bench_serving) ..."
BENCHMARK_DURATION='{"S1":0,"S8":0,"Smax":0}'

BENCH_ARGS="${API_BASE} ${EVAL_DATA}"
if [ -n "${BENCH_MAX_SAMPLES}" ]; then
    BENCH_ARGS="${BENCH_ARGS} ${BENCH_MAX_SAMPLES}"
    echo "[entrypoint] benchmark_duration 最多使用 ${BENCH_MAX_SAMPLES} 条数据"
fi

BENCH_OUTPUT=$(bash /app/bench_serving.sh ${BENCH_ARGS} 2>&1) || true

echo "${BENCH_OUTPUT}"

# 取最后一行作为 benchmark_duration JSON
BENCH_RESULT=$(echo "${BENCH_OUTPUT}" | tail -1)

BENCH_ERROR=""
if [ -n "${BENCH_RESULT}" ] && echo "${BENCH_RESULT}" | python3 -c "import json,sys; json.loads(sys.stdin.read())" 2>/dev/null; then
    BENCHMARK_DURATION="${BENCH_RESULT}"
    echo "[entrypoint] benchmark_duration: ${BENCHMARK_DURATION}"

    # 检查是否有档位失败（值为 0）
    FAILED_LEVELS=$(python3 -c "
import json
result = json.loads('${BENCHMARK_DURATION}')
failed = [k for k in ['S1', 'S8', 'Smax'] if result.get(k, 0) == 0]
if failed:
    print(','.join(failed))
")
    if [ -n "${FAILED_LEVELS}" ]; then
        BENCH_ERROR="benchmark_duration 部分失败: ${FAILED_LEVELS} 值为 0"
        echo "[entrypoint] [WARN] ${BENCH_ERROR}"
    fi
else
    BENCH_ERROR="benchmark_duration 获取失败，脚本执行异常"
    echo "[entrypoint] [ERROR] ${BENCH_ERROR}"
fi

# ============================================================
# Step 5: 输出最终结果
# ============================================================
echo ""
echo "============================================"
echo " 最终评测结果"
echo "============================================"

# 合并错误信息
FINAL_ERROR=""
if [ -n "${EVAL_ERROR}" ] && [ -n "${BENCH_ERROR}" ]; then
    FINAL_ERROR="${EVAL_ERROR}; ${BENCH_ERROR}"
elif [ -n "${EVAL_ERROR}" ]; then
    FINAL_ERROR="${EVAL_ERROR}"
elif [ -n "${BENCH_ERROR}" ]; then
    FINAL_ERROR="${BENCH_ERROR}"
fi

if [ -n "${FINAL_ERROR}" ]; then
    output_result "0" "${FINAL_ERROR}" "${ACC}" "${BENCHMARK_DURATION}"
else
    output_result "1" "" "${ACC}" "${BENCHMARK_DURATION}"
fi
