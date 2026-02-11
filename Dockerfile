# ============================================================
# GPQA 评测 Docker 镜像
# 基于 NVIDIA CUDA 12.9.1 + MiniCPM-SALA
#
# 构建:
#   docker build -t soar_eval .
#
# 运行:
#   docker run --gpus all \
#     -v /home/user/model/stage2_rc2_job_165691_iter_4400:/model/stage2_rc2_job_165691_iter_4400 \
#     -v /path/to/gpqa_data:/data/gpqa \
#     -e MODEL_PATH=/model/MiniCPM-SALA \
#     -e PORT=30000 \
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
        build-essential cmake libnuma1 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-full python3.10-dev python3.10-venv \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# ---- 安装 uv (Python 包管理器) ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

# ---- 复制 SGLang-MiniCPM-SALA 离线包 ----
COPY SGLang-MiniCPM-SALA /opt/SGLang-MiniCPM-SALA

# ---- 运行 install.sh 编译安装 SGLang-MiniCPM-SALA 环境 ----
RUN cd /opt/SGLang-MiniCPM-SALA \
    && bash install.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# ---- 复制评测脚本 ----
WORKDIR /app
COPY gpqa_eval.py /app/gpqa_eval.py
COPY entrypoint.sh /app/entrypoint.sh
COPY bench_serving.sh /app/bench_serving.sh
RUN chmod +x /app/entrypoint.sh /app/bench_serving.sh

# ---- 默认环境变量 ----
ENV MODEL_PATH=/model/stage2_rc2_job_165691_iter_4400 \
    PORT=30000 \
    DATA_DIR=/data/gpqa \
    EVAL_DATA=/data/eval_data_128.jsonl \
    RECORD_ID="" \
    USER_ID="" \
    TASK_ID=""

# ---- 暴露端口 ----
EXPOSE 30000

# ---- 入口 ----
ENTRYPOINT ["/app/entrypoint.sh"]

