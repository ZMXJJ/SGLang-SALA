#!/bin/bash

# vllm服务启动脚本
# 用于启动MiniCPM4-8B模型服务

# 设置环境变量 - 使用GPU 4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 模型路径
MODEL_PATH="/home/linbiyuan/models/MiniCPM4-8B"

# 服务配置
HOST="0.0.0.0"  # 监听所有网络接口，允许远程访问
PORT=8000       # 服务端口，可根据需要修改

# vllm启动参数
echo "正在启动vllm服务..."
echo "模型路径: $MODEL_PATH"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "监听地址: $HOST:$PORT"
echo "-----------------------------------"

# 启动vllm服务
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --served-model-name "MiniCPM4-8B" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192

# 参数说明:
# --tensor-parallel-size 4: 使用4个GPU进行张量并行
# --gpu-memory-utilization 0.95: GPU显存使用率95%
# --max-model-len 8192: 最大序列长度

