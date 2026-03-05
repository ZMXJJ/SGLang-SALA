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
# 状态回调 (POST {SCORE_SYNC_URL}/eval/callback):
#   10  PENDING      - 任务已接收
#   20  PREPARING    - 正在准备环境
#   30  DOWNLOADING  - 正在下载/安装选手提交内容
#   40  INFERENCING  - 正在执行推理评测
#   200 SUCCESS      - 评测完成
#   -1  FAILED       - 任务失败（附带 error_msg）
#
# 环境变量（后端传入）:
#   MODEL_PATH         - 模型路径（必须）
#   PORT               - SGLang 服务端口（默认 30000）
#   PERF_DATA          - 准确率评测数据集路径（JSONL，默认 /data/perf_public_set.jsonl；切私有集传 /data/perf_private_set.jsonl）
#   RECORD_ID          - 提交记录ID
#   USER_ID            - 用户ID
#   TASK_ID            - 任务ID
#   PERF_MAX_SAMPLES   - 准确率评测最多样本数（可选，默认全部）
#   SGLANG_KERNEL_WHEEL- 选手提交的 sgl-kernel wheel 路径（可选）
#   SUBMISSION_TAR     - 选手提交的扩展 tar 包路径（可选；解压后若包含 run.py 则会在启动 SGLang 前执行）
#   SUBMISSION_RUN_REL_PATH - 指定要执行的 run.py 相对路径（可选；用于 tar 内存在多个 run.py 的情况）
#   SGLANG_SERVER_ARGS - SGLang 服务启动参数（可选；用于调整部署参数）
#                        默认包含评测/加速相关参数（示例见 README）
#   SPEED_DATA_S1      - 速度评测 S1(并发=1) 数据集路径（为保证评测一致性，本脚本内固定写死；外部传入会被忽略）
#   SPEED_DATA_S8      - 速度评测 S8(并发=8) 数据集路径（为保证评测一致性，本脚本内固定写死；外部传入会被忽略）
#   SPEED_DATA_SMAX    - 速度评测 Smax(不设并发上限) 数据集路径（为保证评测一致性，本脚本内固定写死；外部传入会被忽略）
#   SCORE_SYNC_URL     - 回调地址基址（可选）
#                        状态回调: POST {SCORE_SYNC_URL}/eval/callback
#                        分数同步: POST {SCORE_SYNC_URL}/score/sync
#   SCORE_SYNC_REQUIRED- 分数同步是否强依赖（可选；默认 0；为 1 时回调失败会置 state=0）
#   速度评测数据（可选，未提供则跳过速度评测，仅输出 acc）:
#     - /data/speed_bench_c1.jsonl   (S1 并发=1)
#     - /data/speed_bench_c8.jsonl   (S8 并发=8)
#     - /data/speed_bench_cunlimited.jsonl (Smax 不设并发上限)
# ============================================================
set -e

# ---- 参数 ----
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
MODEL_PATH="${MODEL_PATH:?环境变量 MODEL_PATH 未设置}"
PORT="${PORT:-30000}"
PERF_DATA="${PERF_DATA:-/data/perf_public_set.jsonl}"
RECORD_ID="${RECORD_ID:-}"
USER_ID="${USER_ID:-}"
TASK_ID="${TASK_ID:-}"
export RECORD_ID USER_ID TASK_ID
PERF_MAX_SAMPLES="${PERF_MAX_SAMPLES:-}"
SGLANG_KERNEL_WHEEL="${SGLANG_KERNEL_WHEEL:-}"
SUBMISSION_TAR="${SUBMISSION_TAR:-}"
SUBMISSION_RUN_REL_PATH="${SUBMISSION_RUN_REL_PATH:-}"
SGLANG_SERVER_ARGS_DEFAULT="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse"
SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-${SGLANG_SERVER_ARGS_DEFAULT}}"

# ---- 速度评测数据集固定写死（防止外部注入篡改）----
if [ -n "${SPEED_DATA_S1:-}" ] && [ "${SPEED_DATA_S1}" != "/data/speed_bench_c1.jsonl" ]; then
    echo "[entrypoint] [WARN] 外部设置 SPEED_DATA_S1=${SPEED_DATA_S1} 将被忽略，使用固定数据集 /data/speed_bench_c1.jsonl" >&2
fi
if [ -n "${SPEED_DATA_S8:-}" ] && [ "${SPEED_DATA_S8}" != "/data/speed_bench_c8.jsonl" ]; then
    echo "[entrypoint] [WARN] 外部设置 SPEED_DATA_S8=${SPEED_DATA_S8} 将被忽略，使用固定数据集 /data/speed_bench_c8.jsonl" >&2
fi
if [ -n "${SPEED_DATA_SMAX:-}" ] && [ "${SPEED_DATA_SMAX}" != "/data/speed_bench_cunlimited.jsonl" ]; then
    echo "[entrypoint] [WARN] 外部设置 SPEED_DATA_SMAX=${SPEED_DATA_SMAX} 将被忽略，使用固定数据集 /data/speed_bench_cunlimited.jsonl" >&2
fi
if [ -n "${SPEED_DATA:-}" ]; then
    echo "[entrypoint] [WARN] 检测到旧变量 SPEED_DATA=${SPEED_DATA}；该变量不再生效，将使用固定速度评测数据集路径" >&2
fi

SPEED_DATA_S1="/data/speed_bench_c1.jsonl"
SPEED_DATA_S8="/data/speed_bench_c8.jsonl"
SPEED_DATA_SMAX="/data/speed_bench_cunlimited.jsonl"
export SPEED_DATA_S1 SPEED_DATA_S8 SPEED_DATA_SMAX
SCORE_SYNC_URL="${SCORE_SYNC_URL:-}"
SCORE_SYNC_REQUIRED="${SCORE_SYNC_REQUIRED:-0}"
SCORE_SYNC_RETRIES="${SCORE_SYNC_RETRIES:-3}"
SCORE_SYNC_CONNECT_TIMEOUT="${SCORE_SYNC_CONNECT_TIMEOUT:-5}"
SCORE_SYNC_TIMEOUT="${SCORE_SYNC_TIMEOUT:-20}"

normalize_server_args() {
    # 兼容用户习惯的下划线写法（如 --dense_as_sparse），统一转换为 CLI 的连字符写法
    SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS//--dense_as_sparse/--dense-as-sparse}"
    SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS//--disable_radix_cache/--disable-radix-cache}"
    SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS//--chunked_prefill_size/--chunked-prefill-size}"
    SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS//--skip_server_warmup/--skip-server-warmup}"
    SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS//--attention_backend/--attention-backend}"
}
normalize_server_args

API_BASE="http://127.0.0.1:${PORT}"
VENV_DIR="/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env"

# ---- 日志持久化（尽早创建，即使后续崩溃也能追溯）----
LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR_NAME="${RECORD_ID:-no_record}_${TASK_ID:-no_task}_${USER_ID:-no_user}"
LOG_DIR="outputs/${LOG_DIR_NAME}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/entrypoint_logs_${LOG_TIMESTAMP}"
mkdir -p "${LOG_DIR}" 2>/dev/null || true
LOG_FILE="${LOG_DIR}/entrypoint.log"
export LOG_DIR LOG_FILE

# 立即写入任务元信息
python3 - "${LOG_DIR}/metadata.json" <<'PYMETA' 2>/dev/null || true
import json, os, sys, time
meta = {
    "record_id": os.environ.get("RECORD_ID", ""),
    "user_id": os.environ.get("USER_ID", ""),
    "task_id": os.environ.get("TASK_ID", ""),
    "model_path": os.environ.get("MODEL_PATH", ""),
    "port": os.environ.get("PORT", "30000"),
    "perf_data": os.environ.get("PERF_DATA", ""),
    "sglang_server_args": os.environ.get("SGLANG_SERVER_ARGS", ""),
    "sglang_kernel_wheel": os.environ.get("SGLANG_KERNEL_WHEEL", ""),
    "submission_tar": os.environ.get("SUBMISSION_TAR", ""),
    "submission_run_rel_path": os.environ.get("SUBMISSION_RUN_REL_PATH", ""),
    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "start_epoch": int(time.time()),
}
with open(sys.argv[1], "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
PYMETA

echo "[entrypoint] 日志目录: ${LOG_DIR}" >&2

# 将 stdout/stderr 同步写入日志文件（不影响原有输出行为）
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

# ---- 函数: 状态回调（best-effort, POST {SCORE_SYNC_URL}/eval/callback）----
status_callback() {
    local status_code="$1"
    local message="$2"
    if [ -z "${SCORE_SYNC_URL}" ]; then
        return 0
    fi
    local url="${SCORE_SYNC_URL%/}/eval/callback"
    local payload
    payload=$(python3 -c "
import json, sys
print(json.dumps({
    'record_id': sys.argv[1],
    'user_id': sys.argv[2],
    'status_code': int(sys.argv[3]),
    'task_id': sys.argv[4],
    'message': sys.argv[5]
}, ensure_ascii=False, separators=(',',':')))" \
        "${RECORD_ID}" "${USER_ID}" "${status_code}" "${TASK_ID}" "${message}") || true

    if [ -n "${payload}" ]; then
        curl -s -o /dev/null \
            --connect-timeout "${SCORE_SYNC_CONNECT_TIMEOUT}" \
            --max-time "${SCORE_SYNC_TIMEOUT}" \
            -X POST "${url}" \
            -H 'Content-Type: application/json' \
            -d "${payload}" 2>/dev/null || true
    fi
    echo "[entrypoint] 状态回调: status_code=${status_code}, message=${message}" >&2
}

# ---- 状态 10: PENDING ----
status_callback 10 "任务已接收，等待处理"

# ---- 状态 20: PREPARING ----
status_callback 20 "正在准备环境，激活虚拟环境"

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

    local url="${SCORE_SYNC_URL%/}/score/sync"
    local retries="${SCORE_SYNC_RETRIES}"
    local connect_timeout="${SCORE_SYNC_CONNECT_TIMEOUT}"
    local timeout="${SCORE_SYNC_TIMEOUT}"
    local last_err=""

    for i in $(seq 1 "${retries}"); do
        local http_code=""
        local body=""
        http_code=$(curl -s -o /tmp/_score_sync_body -w '%{http_code}' \
            --connect-timeout "${connect_timeout}" \
            --max-time "${timeout}" \
            -X POST "${url}" \
            -H 'Content-Type: application/json' \
            -d "${payload_json}") || true
        body=$(cat /tmp/_score_sync_body 2>/dev/null || echo "")

        if [ "${http_code}" = "200" ]; then
            local resultcode
            resultcode=$(echo "${body}" | python3 -c "import json,sys; obj=json.load(sys.stdin); print(obj.get('resultcode',''))" 2>/dev/null || echo "")
            if [ "${resultcode}" = "0000" ]; then
                echo "[entrypoint] 分数同步成功: ${url}" >&2
                return 0
            fi
            last_err="HTTP 200 但返回不符合预期: ${body:0:300}"
        else
            last_err="HTTP ${http_code}: ${body:0:300}"
        fi

        echo "[entrypoint] [WARN] 分数同步失败 (attempt ${i}/${retries}): ${last_err}" >&2
        if [ "${i}" -lt "${retries}" ]; then
            local wait_sec=$(( 2 ** (i - 1) ))
            [ "${wait_sec}" -gt 8 ] && wait_sec=8
            sleep "${wait_sec}"
        fi
    done

    return 1
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
SUBMISSION_TAR_EXTRACT_DIR=""
install_submission_sgl_kernel() {
    SUBMISSION_ERROR=""
    local wheel_path=""

    if [ -z "${SGLANG_KERNEL_WHEEL}" ]; then
        echo "[entrypoint] 未提供选手 wheel（SGLANG_KERNEL_WHEEL 为空），使用镜像内置 sgl-kernel" >&2
        return 0
    fi

    local submission_path="${SGLANG_KERNEL_WHEEL}"
    echo "[entrypoint] 检测到选手提交文件（SGLANG_KERNEL_WHEEL）: ${submission_path}" >&2

    if [ ! -f "${submission_path}" ]; then
        SUBMISSION_ERROR="选手提交文件不存在: ${submission_path}"
        return 1
    fi

    # 如果是 tar.gz / tgz 压缩包，先解压再查找 .whl 文件
    if echo "${submission_path}" | grep -qiE '\.(tar\.gz|tgz)$'; then
        echo "[entrypoint] 检测到 tar.gz 压缩包，开始解压 ..." >&2
        local extract_dir="/tmp/sgl_kernel_extract_$$"
        rm -rf "${extract_dir}"
        mkdir -p "${extract_dir}"

        if ! tar xzf "${submission_path}" -C "${extract_dir}" 2>&1; then
            SUBMISSION_ERROR="tar.gz 解压失败: ${submission_path}"
            return 1
        fi

        echo "[entrypoint] 解压完成，查找 .whl 文件 ..." >&2
        local found_wheels=()
        while IFS= read -r -d '' f; do
            found_wheels+=("$f")
        done < <(find "${extract_dir}" -type f -name '*.whl' -print0 2>/dev/null)

        if [ ${#found_wheels[@]} -eq 0 ]; then
            SUBMISSION_ERROR="tar.gz 中未找到 .whl 文件"
            echo "[entrypoint] [ERROR] ${SUBMISSION_ERROR}，解压目录内容:" >&2
            ls -lR "${extract_dir}" >&2 || true
            return 1
        fi

        if [ ${#found_wheels[@]} -gt 1 ]; then
            echo "[entrypoint] [WARN] tar.gz 中包含多个 .whl 文件，使用第一个:" >&2
            for w in "${found_wheels[@]}"; do
                echo "  - ${w}" >&2
            done
        fi

        wheel_path="${found_wheels[0]}"
        echo "[entrypoint] 从压缩包中提取 wheel: ${wheel_path}" >&2
    else
        wheel_path="${submission_path}"
    fi

    echo "[entrypoint] 校验 wheel 内容（仅允许 sgl-kernel）..." >&2
    local canonical_name=""
    if ! canonical_name="$(python3 - "${wheel_path}" <<'PY'
import hashlib
import os
import re
import sys
import zipfile

wheel_path = sys.argv[1]

def norm_dist_name(name: str) -> str:
    # PEP 427-ish normalization for path prefixes
    return re.sub(r"[-_.]+", "_", name).lower()

h = hashlib.sha256()
with open(wheel_path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
sha256 = h.hexdigest()

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
version = None
for line in meta.splitlines():
    low = line.lower()
    if low.startswith("name:"):
        dist_name = line.split(":", 1)[1].strip()
    elif low.startswith("version:"):
        version = line.split(":", 1)[1].strip()
    if dist_name and version:
        break

if not dist_name:
    print("[entrypoint] [ERROR] METADATA 缺少 Name 字段", file=sys.stderr)
    sys.exit(1)
if not version:
    version = "0"

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
allowed_dist_dirs = re.compile(r"^sgl_kernel-[^/]+?\.(dist-info|data)/")

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

# 读取 wheel tag（用于生成“规范 wheel 文件名”，避免挂载/下载后被重命名导致 pip 安装失败）
wheel_tag = None
wheel_meta_candidates = sorted([n for n in names if n.endswith(".dist-info/WHEEL")])
if wheel_meta_candidates:
    wheel_txt = zf.read(wheel_meta_candidates[0]).decode("utf-8", errors="replace")
    for line in wheel_txt.splitlines():
        if line.startswith("Tag:"):
            wheel_tag = line.split(":", 1)[1].strip()
            break
if not wheel_tag:
    wheel_tag = "py3-none-any"

parts = wheel_tag.split("-")
if len(parts) != 3:
    wheel_tag = "py3-none-any"
    parts = wheel_tag.split("-")
py_tag, abi_tag, plat_tag = parts

ver_fn = re.sub(r"[^0-9A-Za-z.]+", "_", version)
canonical = f"sgl_kernel-{ver_fn}-{py_tag}-{abi_tag}-{plat_tag}.whl"

print(
    f"[entrypoint] wheel 校验通过: dist={dist_name}, version={version}, tag={wheel_tag}, sha256={sha256[:16]}..., file={os.path.basename(wheel_path)}",
    file=sys.stderr,
)
print(canonical)
PY
)"; then
        SUBMISSION_ERROR="选手 wheel 校验失败（仅允许 sgl-kernel 内容）"
        return 1
    fi

    # pip 对 wheel 文件名有格式要求；为避免被重命名导致安装失败，这里复制到一个规范文件名再安装
    local wheel_canon_path="/tmp/${canonical_name}"
    if ! cp -f "${wheel_path}" "${wheel_canon_path}" 1>&2; then
        SUBMISSION_ERROR="选手 wheel 复制失败（生成规范文件名）"
        return 1
    fi
    wheel_path="${wheel_canon_path}"
    echo "[entrypoint] wheel 已规范化为: ${wheel_path}" >&2

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

# ---- 函数: 解压并执行选手提交的 run.py（可选）----
run_submission_tar_hook() {
    SUBMISSION_ERROR=""

    if [ -z "${SUBMISSION_TAR}" ]; then
        return 0
    fi

    echo "[entrypoint] 检测到选手提交 tar（SUBMISSION_TAR）: ${SUBMISSION_TAR}" >&2

    if [ ! -f "${SUBMISSION_TAR}" ]; then
        SUBMISSION_ERROR="选手提交 tar 不存在: ${SUBMISSION_TAR}"
        return 1
    fi

    local extract_dir=""
    extract_dir="$(mktemp -d /tmp/soar_submission_XXXXXX 2>/dev/null || true)"
    if [ -z "${extract_dir}" ]; then
        extract_dir="/tmp/soar_submission_$$"
        rm -rf "${extract_dir}" 2>/dev/null || true
        mkdir -p "${extract_dir}" 2>/dev/null || true
    fi
    SUBMISSION_TAR_EXTRACT_DIR="${extract_dir}"

    echo "[entrypoint] 解压选手 tar 到: ${extract_dir}" >&2
    if ! python3 - "${SUBMISSION_TAR}" "${extract_dir}" <<'PY' 1>&2; then
import os
import sys
import tarfile

tar_path = sys.argv[1]
dest_dir = sys.argv[2]

os.makedirs(dest_dir, exist_ok=True)

def is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_directory, abs_target]) == abs_directory

try:
    tf = tarfile.open(tar_path, "r:*")
except Exception as e:
    print(f"[entrypoint] [ERROR] 无法打开 tar: {tar_path} ({e})", file=sys.stderr)
    sys.exit(1)

with tf:
    members = tf.getmembers()
    for m in members:
        name = m.name or ""
        if name.startswith("./"):
            name = name[2:]
        norm = os.path.normpath(name)
        if norm in ("", ".", "/"):
            continue
        if os.path.isabs(norm) or any(part == ".." for part in norm.split(os.sep)):
            print(f"[entrypoint] [ERROR] tar 内含不安全路径: {m.name}", file=sys.stderr)
            sys.exit(1)
        if m.issym() or m.islnk():
            print(f"[entrypoint] [ERROR] tar 内含 link（不允许）: {m.name}", file=sys.stderr)
            sys.exit(1)
        if m.ischr() or m.isblk() or m.isfifo():
            print(f"[entrypoint] [ERROR] tar 内含特殊文件（不允许）: {m.name}", file=sys.stderr)
            sys.exit(1)

        target_path = os.path.join(dest_dir, norm)
        if not is_within_directory(dest_dir, target_path):
            print(f"[entrypoint] [ERROR] tar 路径穿越: {m.name}", file=sys.stderr)
            sys.exit(1)

    tf.extractall(dest_dir)

print(f"[entrypoint] tar 解压完成: {dest_dir}", file=sys.stderr)
PY
        SUBMISSION_ERROR="选手提交 tar 解压失败: ${SUBMISSION_TAR}"
        return 1
    fi

    local run_py=""
    if [ -n "${SUBMISSION_RUN_REL_PATH}" ]; then
        if [[ "${SUBMISSION_RUN_REL_PATH}" = /* ]] || [[ "${SUBMISSION_RUN_REL_PATH}" == *".."* ]]; then
            SUBMISSION_ERROR="SUBMISSION_RUN_REL_PATH 不合法（必须为相对路径且不能包含 .. ）: ${SUBMISSION_RUN_REL_PATH}"
            return 1
        fi
        run_py="${extract_dir}/${SUBMISSION_RUN_REL_PATH}"
        if [ ! -f "${run_py}" ]; then
            SUBMISSION_ERROR="SUBMISSION_RUN_REL_PATH 指定的脚本不存在: ${SUBMISSION_RUN_REL_PATH}"
            return 1
        fi
    elif [ -f "${extract_dir}/run.py" ]; then
        run_py="${extract_dir}/run.py"
    else
        run_py="$(python3 - "${extract_dir}" <<'PY'
import os
import sys

root = sys.argv[1]

try:
    root_run = os.path.join(root, "run.py")
    if os.path.isfile(root_run):
        print(root_run)
        raise SystemExit(0)

    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 过滤掉常见无关目录，避免误选/加速遍历
        dirnames[:] = [
            d for d in dirnames
            if d not in ("__pycache__", ".git", ".svn") and not d.startswith(".")
        ]
        if "run.py" in filenames:
            candidates.append(os.path.join(dirpath, "run.py"))

    if not candidates:
        print("")
        raise SystemExit(0)

    if len(candidates) > 1:
        print("__MULTIPLE__")
        print(
            "[entrypoint] [ERROR] tar 中发现多个 run.py，请通过 SUBMISSION_RUN_REL_PATH 指定要执行的脚本（相对解压根目录）:",
            file=sys.stderr,
        )
        for p in sorted(candidates)[:50]:
            rel = os.path.relpath(p, root)
            print(f"  - {rel}", file=sys.stderr)
        raise SystemExit(0)

    print(candidates[0])
except Exception as e:
    print("__ERROR__")
    print(f"[entrypoint] [ERROR] 扫描 run.py 失败: {e}", file=sys.stderr)
    raise SystemExit(0)
PY
)"
    fi

    if [ "${run_py}" = "__ERROR__" ]; then
        SUBMISSION_ERROR="扫描 run.py 失败（请检查 SUBMISSION_TAR 内容）"
        return 1
    fi
    if [ "${run_py}" = "__MULTIPLE__" ]; then
        SUBMISSION_ERROR="tar 中发现多个 run.py，请使用 SUBMISSION_RUN_REL_PATH 指定"
        return 1
    fi

    if [ -z "${run_py}" ]; then
        echo "[entrypoint] tar 解压完成，但未发现 run.py，跳过选手预处理步骤" >&2
        return 0
    fi

    echo "[entrypoint] 运行选手预处理脚本: ${run_py}" >&2

    local result_json="${extract_dir}/soar_result.json"
    export SOAR_SUBMISSION_TAR="${SUBMISSION_TAR}"
    export SOAR_SUBMISSION_DIR="${extract_dir}"
    export SOAR_ORIG_MODEL_PATH="${MODEL_PATH}"
    export SOAR_MODEL_PATH="${MODEL_PATH}"
    export SOAR_SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS}"
    export SOAR_RESULT_JSON="${result_json}"
    export SOAR_LOG_DIR="${LOG_DIR}"
    export SOAR_LOG_FILE="${LOG_FILE}"

    local run_dir=""
    run_dir="$(dirname "${run_py}")"
    if ! (cd "${run_dir}" && python3 "$(basename "${run_py}")"); then
        SUBMISSION_ERROR="选手预处理脚本执行失败: ${run_py}"
        return 1
    fi

    if [ ! -f "${result_json}" ]; then
        echo "[entrypoint] [submission] run.py 未生成结果文件（SOAR_RESULT_JSON=${result_json}），保持默认 MODEL_PATH/SGLANG_SERVER_ARGS" >&2
        return 0
    fi

    if ! python3 -c "import json,sys; json.load(open(sys.argv[1], 'r', encoding='utf-8'))" "${result_json}" 2>/dev/null; then
        SUBMISSION_ERROR="选手预处理脚本输出的结果 JSON 无法解析: ${result_json}"
        return 1
    fi

    local new_model_path=""
    local new_server_args=""
    local new_server_args_append=""
    readarray -t _SUBMISSION_OUT < <(python3 - "${result_json}" <<'PY'
import json
import sys

p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    obj = json.load(f) or {}

def get_str(key: str) -> str:
    v = obj.get(key, "")
    if v is None:
        return ""
    if not isinstance(v, str):
        v = str(v)
    return v.replace("\\n", " ").strip()

print(get_str("model_path"))
print(get_str("sglang_server_args"))
print(get_str("sglang_server_args_append"))
PY
)
    new_model_path="${_SUBMISSION_OUT[0]:-}"
    new_server_args="${_SUBMISSION_OUT[1]:-}"
    new_server_args_append="${_SUBMISSION_OUT[2]:-}"

    if [ -n "${new_model_path}" ]; then
        if [ -d "${new_model_path}" ] || [ -f "${new_model_path}" ]; then
            MODEL_PATH="${new_model_path}"
            export MODEL_PATH
            echo "[entrypoint] [submission] MODEL_PATH 已更新为: ${MODEL_PATH}" >&2
        else
            echo "[entrypoint] [WARN] run.py 返回 model_path 但路径不存在: ${new_model_path}，忽略" >&2
        fi
    fi

    if [ -n "${new_server_args}" ]; then
        SGLANG_SERVER_ARGS="${new_server_args}"
        echo "[entrypoint] [submission] SGLANG_SERVER_ARGS 已被 run.py 覆盖" >&2
    elif [ -n "${new_server_args_append}" ]; then
        SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS} ${new_server_args_append}"
        echo "[entrypoint] [submission] SGLANG_SERVER_ARGS 已追加 run.py 参数" >&2
    fi
    normalize_server_args

    echo "[entrypoint] [submission] 最终 SGLANG_SERVER_ARGS: ${SGLANG_SERVER_ARGS}" >&2
    return 0
}

# ---- 函数: 清理 SGLang 进程 ----
cleanup() {
    if [ -n "${SGLANG_PID:-}" ] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "[entrypoint] 正在关闭 SGLang 服务 (PID: ${SGLANG_PID}) ..." >&2
        kill "${SGLANG_PID}" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
        echo "[entrypoint] SGLang 服务已关闭" >&2
    fi
}

# ---- 函数: 退出处理（记录退出状态 + 清理）----
on_exit() {
    local exit_code=$?
    if [ ${exit_code} -ne 0 ] && [ -d "${LOG_DIR:-}" ]; then
        echo "[entrypoint] [FATAL] 脚本异常退出, exit_code=${exit_code}, LINENO=${BASH_LINENO[0]:-unknown}" >&2
        python3 - "${LOG_DIR}/exit_status.json" "${exit_code}" <<'PYEXIT' 2>/dev/null || true
import json, os, sys, time
with open(sys.argv[1], "w") as f:
    json.dump({
        "exit_code": int(sys.argv[2]),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_epoch": int(time.time()),
        "record_id": os.environ.get("RECORD_ID", ""),
        "task_id": os.environ.get("TASK_ID", ""),
    }, f, ensure_ascii=False, indent=2)
PYEXIT
    fi
    cleanup
    if [ -n "${SUBMISSION_TAR_EXTRACT_DIR:-}" ] && [ -d "${SUBMISSION_TAR_EXTRACT_DIR}" ]; then
        rm -rf "${SUBMISSION_TAR_EXTRACT_DIR}" 2>/dev/null || true
    fi
}

# ---- 函数: 检查 SGLang 服务是否可用 ----
sglang_is_ready() {
    # /health + /v1/models 都可用，且 /v1/models 返回可解析 JSON
    if ! curl -sf "${API_BASE}/health" > /dev/null 2>&1; then
        return 1
    fi
    if ! curl -sf "${API_BASE}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'data' in d and len(d['data'])>0" 2>/dev/null; then
        return 1
    fi
    return 0
}

# ---- 函数: 重启 SGLang 服务并等待就绪 ----
restart_sglang_server() {
    echo "[entrypoint] 尝试重启 SGLang 服务 ..." >&2
    cleanup

    python3 -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --trust-remote-code \
        --port "${PORT}" \
        ${SGLANG_SERVER_ARGS} \
        --tp-size 1 \
        --max-running-requests 32 \
        --enable-metrics 1>&2 &

    SGLANG_PID=$!
    echo "[entrypoint] SGLang 进程 PID: ${SGLANG_PID}" >&2

    local max_retries=120
    local retry_count=0
    while true; do
        if sglang_is_ready; then
            echo "[entrypoint] SGLang 服务已就绪！" >&2
            return 0
        fi

        if [ -n "${SGLANG_PID}" ] && ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
            echo "[entrypoint] [ERROR] 重启后 SGLang 进程已退出" >&2
            return 1
        fi

        sleep 5
        retry_count=$((retry_count + 1))
        if [ "${retry_count}" -ge "${max_retries}" ]; then
            echo "[entrypoint] [ERROR] 重启等待超时 (${max_retries} 次重试)" >&2
            return 1
        fi
    done
}
trap on_exit EXIT

# ============================================================
# Step 0: 安装选手提交内容（可选）
# ============================================================
# ---- 状态 30: DOWNLOADING ----
status_callback 30 "正在下载并安装选手提交内容"

echo "[entrypoint] 检查并安装选手 sgl-kernel 提交内容（如有）..."
if ! install_submission_sgl_kernel; then
    echo "[entrypoint] [ERROR] ${SUBMISSION_ERROR}" >&2
    status_callback -1 "任务失败: ${SUBMISSION_ERROR}"
    output_result "0" "${SUBMISSION_ERROR}" "0" '{"S1":0,"S8":0,"Smax":0}'
    exit 1
fi

echo "[entrypoint] 检查并运行选手预处理脚本（如有）..."
if ! run_submission_tar_hook; then
    echo "[entrypoint] [ERROR] ${SUBMISSION_ERROR}" >&2
    status_callback -1 "任务失败: ${SUBMISSION_ERROR}"
    output_result "0" "${SUBMISSION_ERROR}" "0" '{"S1":0,"S8":0,"Smax":0}'
    exit 1
fi

# ============================================================
# Step 1: 启动 SGLang 推理服务
# ============================================================
echo "[entrypoint] 启动 SGLang 服务 ..."
echo "  模型路径: ${MODEL_PATH}"
echo "  端口:     ${PORT}"

if echo " ${SGLANG_SERVER_ARGS} " | grep -qE ' --tp-size | --max-running-requests '; then
    echo "[entrypoint] [WARN] 检测到 SGLANG_SERVER_ARGS 包含 --tp-size/--max-running-requests；将被入口脚本固定值覆盖（tp=1, max_running_requests=32）" >&2
fi

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --port "${PORT}" \
    ${SGLANG_SERVER_ARGS} \
    --tp-size 1 \
    --max-running-requests 32 \
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
        status_callback -1 "任务失败: SGLang 服务启动失败"
        output_result "0" "SGLang 服务启动失败" "0" '{"S1":0,"S8":0,"Smax":0}'
        exit 1
    fi

    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "[entrypoint] [ERROR] 等待超时 (${MAX_RETRIES} 次重试)"
        status_callback -1 "任务失败: SGLang 服务启动超时"
        output_result "0" "SGLang 服务启动超时" "0" '{"S1":0,"S8":0,"Smax":0}'
        exit 1
    fi
    echo "  等待中... (${RETRY_COUNT}/${MAX_RETRIES})"
done
echo "[entrypoint] SGLang 服务已就绪！"

# ---- 状态 40: INFERENCING ----
status_callback 40 "SGLang 服务已就绪，正在执行推理评测"

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
# Step 4: 速度评测（已跳过，仅测正确性）
# ============================================================
echo "[entrypoint] [SKIP] 速度评测已跳过，仅验证正确性"
BENCHMARK_DURATION='{"S1":0,"S8":0,"Smax":0}'
BENCH_ERROR=""

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
    status_callback -1 "任务失败: ${FINAL_ERROR}"
    output_result "0" "${FINAL_ERROR}" "${ACC}" "${BENCHMARK_DURATION}"
else
    status_callback 200 "评测完成"
    output_result "1" "" "${ACC}" "${BENCHMARK_DURATION}"
fi
