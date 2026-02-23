# ============================================================
# 模型综合评测 Docker 镜像 (v2.0.3)
# 基于 NVIDIA CUDA 12.9.1 (devel) + MiniCPM-SALA
#
# 评测内容:
#   1. 模型准确率 acc（eval_model.py）
#   2. benchmark_duration (S1/S8/Smax 三档并发)
#
# 构建:
#   docker build -t soar_eval .
#
# 运行（示例）:
#   docker run --rm --gpus '"device=0"' \
#     -v /path/to/model:/home/user/linbiyuan/models/MiniCPM-SALA \
#     -v /path/to/perf_public_set.jsonl:/data/perf_public_set.jsonl \
#     -v /path/to/speed_eval.jsonl:/data/speed_eval.jsonl \
#     -e MODEL_PATH=/home/user/linbiyuan/models/MiniCPM-SALA \
#     -e RECORD_ID=xxx \
#     -e USER_ID=xxx \
#     -e TASK_ID=xxx \
#     soar_eval
# ============================================================

FROM modelbest-registry.cn-beijing.cr.aliyuncs.com/public/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common wget curl git \
        build-essential cmake libnuma1 libnuma-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-full python3.10-dev python3.10-venv \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# ---- 安装 uv (Python 包管理器) ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

# ---- 复制 SGLang-MiniCPM-SALA 离线包 ----
COPY SGLang-MiniCPM-SALA /opt/SGLang-MiniCPM-SALA

# ---- 编译安装 SGLang-MiniCPM-SALA 环境 ----
RUN cd /opt/SGLang-MiniCPM-SALA \
    && bash install.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# ---- 从 sglang-minicpm-dev 源码重编译 sgl-kernel（含 sm_100a + sm_120a 支持）----
COPY sgl-kernel /tmp/sgl-kernel
RUN . /opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/bin/activate \
    && export TORCH_CUDA_ARCH_LIST="12.0;12.0a" \
    && export CMAKE_BUILD_PARALLEL_LEVEL=8 \
    && export CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=8 -DENABLE_BELOW_SM90=OFF" \
    && pip uninstall -y sgl-kernel \
    && pip install "scikit-build-core>=0.10" "cmake>=3.26,<4" \
    && cd /tmp/sgl-kernel \
    && pip install . --no-deps --no-build-isolation \
    && rm -rf /tmp/sgl-kernel

# ---- 复制评测脚本 ----
WORKDIR /app
COPY eval_model.py /app/eval_model.py
COPY entrypoint.sh /app/entrypoint.sh
COPY bench_serving.sh /app/bench_serving.sh
RUN chmod +x /app/entrypoint.sh /app/bench_serving.sh

# ---- 默认环境变量 ----
ENV MODEL_PATH=/home/user/linbiyuan/models/MiniCPM-SALA \
    PORT=30000 \
    PERF_DATA=/data/perf_public_set.jsonl \
    SPEED_DATA=/data/speed_eval.jsonl \
    RECORD_ID="" \
    USER_ID="" \
    TASK_ID=""

# ---- 暴露端口 ----
EXPOSE 30000

# ---- 入口 ----
ENTRYPOINT ["/app/entrypoint.sh"]
