#!/bin/bash
# ============================================================
#  评测 Docker 入口脚本
#
# 流程:
#   1. 启动 SGLang 推理服务（后台）
#   2. 等待服务就绪
#   3. 运行 eval_model.py 获取 acc
#   4. 运行 bench_serving.sh（基于 sglang.bench_serving）获取 benchmark_duration
#   5. 组装 JSON 结果输出到 stdout
#   6. 关闭 SGLang 服务
#
# 环境变量（后端传入）:
#   MODEL_PATH         - 模型路径（必须）
#   PORT               - SGLang 服务端口（默认 30000）
#   PERF_DATA          - 准确率评测数据集路径（JSONL，默认 /data/perf_public_set.jsonl；切私有集传 /data/perf_private_set.jsonl）
#   SPEED_DATA         - 速度评测数据集路径（JSONL，默认 /data/speed_eval.jsonl）
#   RECORD_ID          - 提交记录ID
#   USER_ID            - 用户ID
#   TASK_ID            - 任务ID
#   PERF_MAX_SAMPLES   - 准确率评测最多样本数（可选，默认全部）
#   SPEED_MAX_SAMPLES  - 速度评测最多数据条数（可选，默认全部）
#   SGL_KERNEL_WHEEL   - 选手提交的 sgl-kernel wheel 路径（可选，优先级高于 INPUT_URL）
#   INPUT_URL          - 选手提交文件预签名下载直链（可选，作为 wheel 来源；未传 SGL_KERNEL_WHEEL 时使用）
#   SCORE_SYNC_URL     - 分数同步回调地址基址（可选；POST {SCORE_SYNC_URL}/score/sync）
#   SCORE_SYNC_REQUIRED- 分数同步是否强依赖（可选；默认 0；为 1 时回调失败会置 state=0）
#   DISABLE_CUDA_GRAPH - 是否禁用 CUDA Graph（可选；默认 1，避免 flashinfer+graph 在高并发下不稳定）
# ============================================================
set -e

# ---- 参数 ----
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
MODEL_PATH="${MODEL_PATH:?环境变量 MODEL_PATH 未设置}"
PORT="${PORT:-30000}"
PERF_DATA="${PERF_DATA:-/data/perf_public_set.jsonl}"
SPEED_DATA="${SPEED_DATA:-/data/speed_eval.jsonl}"
RECORD_ID="${RECORD_ID:-}"
USER_ID="${USER_ID:-}"
TASK_ID="${TASK_ID:-}"
PERF_MAX_SAMPLES="${PERF_MAX_SAMPLES:-}"
SPEED_MAX_SAMPLES="${SPEED_MAX_SAMPLES:-}"
GPU_PER_WORKER="${GPU_PER_WORKER:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
SGL_KERNEL_WHEEL="${SGL_KERNEL_WHEEL:-}"
INPUT_URL="${INPUT_URL:-}"
SCORE_SYNC_URL="${SCORE_SYNC_URL:-}"
SCORE_SYNC_REQUIRED="${SCORE_SYNC_REQUIRED:-0}"
SCORE_SYNC_RETRIES="${SCORE_SYNC_RETRIES:-3}"
SCORE_SYNC_CONNECT_TIMEOUT="${SCORE_SYNC_CONNECT_TIMEOUT:-5}"
SCORE_SYNC_TIMEOUT="${SCORE_SYNC_TIMEOUT:-20}"

CUDA_GRAPH_FLAG=""
if [ "${DISABLE_CUDA_GRAPH}" = "1" ] || [ "${DISABLE_CUDA_GRAPH}" = "true" ]; then
    CUDA_GRAPH_FLAG="--disable-cuda-graph"
fi

API_BASE="http://127.0.0.1:${PORT}"
VENV_DIR="/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env"

# ---- 激活虚拟环境 ----
echo "[entrypoint] 激活虚拟环境: ${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# ---- 函数: 生成最终 JSON（compact 或 pretty）----
make_output_json() {
    local state="$1"
    local error_msg="$2"
    local acc="$3"
    local bench_json="$4"
    local pretty="$5" # "1" or "0"

    python3 - "$error_msg" "$acc" "$bench_json" "$state" "$RECORD_ID" "$USER_ID" "$TASK_ID" "$pretty" <<'PY'
import json
import sys

error_msg = sys.argv[1]
acc = float(sys.argv[2])
bench_json = sys.argv[3]
state = sys.argv[4]
record_id = sys.argv[5]
user_id = sys.argv[6]
task_id = sys.argv[7]
pretty = sys.argv[8] == "1"

try:
    bench = json.loads(bench_json)
except Exception:
    bench = {"S1": 0, "S8": 0, "Smax": 0}

output = {
    "record_id": record_id,
    "user_id": user_id,
    "result": {
        "error_msg": error_msg,
        "score": {
            "acc": acc,
            "benchmark_duration": {
                "S1": bench.get("S1", 0),
                "S8": bench.get("S8", 0),
                "Smax": bench.get("Smax", 0),
            },
        },
        "sort_by": "acc",
    },
    "state": state,
    "task_id": task_id,
}

if pretty:
    print(json.dumps(output, ensure_ascii=False, indent=2))
else:
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
PY
}

# ---- 函数: 分数同步回调（best-effort，不输出到 stdout）----
score_sync() {
    local payload_json="$1"
    if [ -z "${SCORE_SYNC_URL}" ]; then
        return 0
    fi

    python3 - "$payload_json" <<'PY'
import json
import os
import sys
import time
from urllib import request, error

base = (os.environ.get("SCORE_SYNC_URL") or "").strip()
if not base:
    sys.exit(0)

url = base.rstrip("/") + "/score/sync"
payload = json.loads(sys.argv[1])

retries = int(os.environ.get("SCORE_SYNC_RETRIES") or "3")
connect_timeout = float(os.environ.get("SCORE_SYNC_CONNECT_TIMEOUT") or "5")
timeout = float(os.environ.get("SCORE_SYNC_TIMEOUT") or "20")

data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
headers = {"Content-Type": "application/json"}

last_err = None
for i in range(1, retries + 1):
    try:
        req = request.Request(url=url, data=data, headers=headers, method="POST")
        with request.urlopen(req, timeout=connect_timeout + timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.status

        if status == 200:
            try:
                obj = json.loads(body)
            except Exception:
                obj = None

            if isinstance(obj, dict) and obj.get("resultcode") == "0000":
                print(f"[entrypoint] 分数同步成功: {url}", file=sys.stderr)
                sys.exit(0)

            last_err = f"HTTP 200 但返回不符合预期: {body[:300]}"
        else:
            last_err = f"HTTP {status}: {body[:300]}"
    except Exception as e:
        last_err = repr(e)

    print(
        f"[entrypoint] [WARN] 分数同步失败 (attempt {i}/{retries}): {last_err}",
        file=sys.stderr,
    )
    if i < retries:
        time.sleep(min(2** (i - 1), 8))

sys.exit(1)
PY
}

# ---- 函数: 输出最终 JSON ----
output_result() {
    local state="$1"
    local error_msg="$2"
    local acc="$3"
    local bench_json="$4"

    local payload_compact
    payload_compact="$(make_output_json "$state" "$error_msg" "$acc" "$bench_json" "0")"

    local sync_failed=0
    if [ -n "${SCORE_SYNC_URL}" ]; then
        if ! score_sync "${payload_compact}"; then
            sync_failed=1
        fi
    fi

    if [ "${sync_failed}" = "1" ] && [ "${SCORE_SYNC_REQUIRED}" = "1" ]; then
        local sync_err="score_sync 回调失败"
        state="0"
        if [ -n "${error_msg}" ]; then
            error_msg="${error_msg}; ${sync_err}"
        else
            error_msg="${sync_err}"
        fi
    elif [ "${sync_failed}" = "1" ]; then
        echo "[entrypoint] [WARN] score_sync 回调失败（不影响评测 state，可通过平台侧重试）" >&2
    fi

    make_output_json "$state" "$error_msg" "$acc" "$bench_json" "1"
}

# ---- 函数: 安装选手提交的 sgl-kernel wheel（可选）----
SUBMISSION_ERROR=""
install_submission_sgl_kernel() {
    SUBMISSION_ERROR=""
    local wheel_path=""

    if [ -n "${SGL_KERNEL_WHEEL}" ]; then
        wheel_path="${SGL_KERNEL_WHEEL}"
        echo "[entrypoint] 检测到选手 wheel（SGL_KERNEL_WHEEL）: ${wheel_path}" >&2
    elif [ -n "${INPUT_URL}" ]; then
        wheel_path="/tmp/sgl_kernel_submission.whl"
        echo "[entrypoint] 未提供 SGL_KERNEL_WHEEL，尝试从 INPUT_URL 下载 wheel ..." >&2
        echo "  INPUT_URL: ${INPUT_URL}" >&2
        if ! curl -fL --retry 3 --connect-timeout 10 --max-time 300 -o "${wheel_path}" "${INPUT_URL}" 1>&2; then
            SUBMISSION_ERROR="选手 wheel 下载失败（INPUT_URL）"
            return 1
        fi
        echo "[entrypoint] wheel 已下载到: ${wheel_path}" >&2
    else
        echo "[entrypoint] 未提供选手 wheel（SGL_KERNEL_WHEEL/INPUT_URL 均为空），使用镜像内置 sgl-kernel" >&2
        return 0
    fi

    if [ ! -f "${wheel_path}" ]; then
        SUBMISSION_ERROR="选手 wheel 不存在: ${wheel_path}"
        return 1
    fi

    echo "[entrypoint] 校验 wheel 内容（仅允许 sgl-kernel）..." >&2
    if ! python3 - "${wheel_path}" 1>&2 <<'PY'
import hashlib
import os
import re
import sys
import zipfile

wheel_path = sys.argv[1]

def norm_dist_name(name: str) -> str:
    # PEP 427-ish normalization for path prefixes
    return re.sub(r"[-_.]+", "_", name).lower()

with open(wheel_path, "rb") as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

try:
    zf = zipfile.ZipFile(wheel_path)
except Exception as e:
    print(f"[entrypoint] [ERROR] wheel 不是有效 zip: {e}", file=sys.stderr)
    sys.exit(1)

names = zf.namelist()
meta_candidates = sorted([n for n in names if n.endswith(".dist-info/METADATA")])
if not meta_candidates:
    print("[entrypoint] [ERROR] wheel 缺少 .dist-info/METADATA", file=sys.stderr)
    sys.exit(1)

meta_path = meta_candidates[0]
meta = zf.read(meta_path).decode("utf-8", errors="replace")
dist_name = None
for line in meta.splitlines():
    if line.lower().startswith("name:"):
        dist_name = line.split(":", 1)[1].strip()
        break

if not dist_name:
    print("[entrypoint] [ERROR] METADATA 缺少 Name 字段", file=sys.stderr)
    sys.exit(1)

normalized = norm_dist_name(dist_name)
if normalized != "sgl_kernel":
    print(
        f"[entrypoint] [ERROR] wheel 分发名不匹配：期望 sgl-kernel/sgl_kernel，实际 {dist_name!r}",
        file=sys.stderr,
    )
    sys.exit(1)

allowed_prefixes = (
    "sgl_kernel/",
    "sgl_kernel.libs/",
)
allowed_dist_dirs = re.compile(r"^sgl_kernel-[^/]+\\.(dist-info|data)/")

bad = []
has_pth = False
for n in names:
    if n.endswith("/"):
        continue
    if n.endswith(".pth"):
        has_pth = True
    if n.startswith(allowed_prefixes):
        continue
    if allowed_dist_dirs.match(n):
        continue
    bad.append(n)
    if len(bad) >= 30:
        break

if has_pth:
    print("[entrypoint] [ERROR] wheel 包含 .pth 文件（不允许）", file=sys.stderr)
    sys.exit(1)

if bad:
    print("[entrypoint] [ERROR] wheel 包含不允许的文件/目录（示例前若干条）:", file=sys.stderr)
    for item in bad:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)

print(
    f"[entrypoint] wheel 校验通过: dist={dist_name}, sha256={sha256[:16]}..., file={os.path.basename(wheel_path)}",
    file=sys.stderr,
)
PY
    then
        SUBMISSION_ERROR="选手 wheel 校验失败（仅允许 sgl-kernel 内容）"
        return 1
    fi

    echo "[entrypoint] 安装选手 sgl-kernel wheel ..." >&2
    pip uninstall -y sgl-kernel 1>/dev/null 2>&1 || true
    if ! pip install --no-deps --force-reinstall "${wheel_path}" 1>&2; then
        SUBMISSION_ERROR="选手 wheel 安装失败（pip install）"
        return 1
    fi

    python3 - 1>&2 <<'PY'
import importlib.metadata as md
import sgl_kernel

try:
    ver = md.version("sgl-kernel")
except Exception:
    ver = None

print(f"[entrypoint] sgl_kernel import OK: file={sgl_kernel.__file__}, version={ver}")
PY

    return 0
}

# ---- 函数: 清理 SGLang 进程 ----
cleanup() {
    if [ -n "${SGLANG_PID}" ] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "[entrypoint] 正在关闭 SGLang 服务 (PID: ${SGLANG_PID}) ..." >&2
        kill "${SGLANG_PID}" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
        echo "[entrypoint] SGLang 服务已关闭" >&2
    fi
}
trap cleanup EXIT

# ============================================================
# Step 0: 安装选手提交物（可选）
# ============================================================
echo "[entrypoint] 检查并安装选手 sgl-kernel 提交物（如有）..."
if ! install_submission_sgl_kernel; then
    echo "[entrypoint] [ERROR] ${SUBMISSION_ERROR}" >&2
    output_result "0" "${SUBMISSION_ERROR}" "0" '{"S1":0,"S8":0,"Smax":0}'
    exit 1
fi

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
    --tp-size "${GPU_PER_WORKER}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    ${CUDA_GRAPH_FLAG} \
    --skip-server-warmup \
    --port "${PORT}" \
    --dense-as-sparse \
    --enable-metrics 1>&2 &

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
# Step 3: 运行 模型评测 (eval_model.py)
# ============================================================
echo "[entrypoint] 开始 模型评测 ..."
ACC=0
EVAL_ERROR=""

EVAL_CMD="python3 /app/eval_model.py \
    --api_base ${API_BASE} \
    --model_path ${MODEL_PATH} \
    --data_path ${PERF_DATA} \
    --concurrency 8"

if [ -n "${PERF_MAX_SAMPLES}" ]; then
    EVAL_CMD="${EVAL_CMD} --num_samples ${PERF_MAX_SAMPLES}"
    echo "[entrypoint] 准确率评测最多评测 ${PERF_MAX_SAMPLES} 条"
fi

if eval ${EVAL_CMD}; then

    # 找到最新的输出目录下的 summary.json
    SUMMARY_FILE=$(ls -t outputs/*/summary.json 2>/dev/null | head -1)
    if [ -f "${SUMMARY_FILE}" ]; then
        ACC=$(python3 -c "
import json
with open('${SUMMARY_FILE}') as f:
    data = json.load(f)
print(data.get('overall_accuracy', 0))
")
        echo "[entrypoint] 评测完成，overall accuracy: ${ACC}"
    else
        EVAL_ERROR="评测完成但未找到 summary.json"
        echo "[entrypoint] [WARN] ${EVAL_ERROR}"
    fi
else
    EVAL_ERROR="评测脚本执行失败"
    echo "[entrypoint] [ERROR] ${EVAL_ERROR}"
fi

# ============================================================
# Step 4: 运行 benchmark_duration (sglang.bench_serving)
# ============================================================
echo "[entrypoint] 运行 benchmark_duration (sglang.bench_serving) ..."
BENCHMARK_DURATION='{"S1":0,"S8":0,"Smax":0}'

BENCH_ARGS="${API_BASE} ${SPEED_DATA}"
if [ -n "${SPEED_MAX_SAMPLES}" ]; then
    BENCH_ARGS="${BENCH_ARGS} ${SPEED_MAX_SAMPLES}"
    echo "[entrypoint] 速度评测最多使用 ${SPEED_MAX_SAMPLES} 条数据"
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
