# MiniCPM-SALA 推理环境搭建流程

## 环境要求

- CUDA 12.x 或更高版本
- `gcc` / `g++` 编译器
- `uv` 包管理器（脚本会自动检测）

## 快速开始

### 安装

```bash
cd SGLang-MiniCPM-SALA

# 指定 PyPI 镜像源
bash install.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

安装脚本会自动完成以下步骤：

1. 创建 `sglang_minicpm_sala_env` 虚拟环境（Python 3.10）
2. 安装 MiniCPM-SALA (SGLang fork) 及所有 Python 依赖
3. 编译安装 infllm_v2 CUDA 扩展
4. 编译安装 sparse_kernel CUDA 扩展
5. 安装 tilelang
6. 安装 flash-linear-attention

### 使用

```bash
# 激活环境
source sglang_minicpm_sala_env/bin/activate

# 启动推理服务（将 MODEL_PATH 替换为实际模型路径）
MODEL_PATH=/path/to/your/model

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse
```

| 参数 | 说明 |
|------|------|
| `--trust-remote-code` | 允许加载模型自带的自定义代码 |
| `--disable-radix-cache` | 禁用 RadixAttention 前缀缓存 |
| `--attention-backend minicpm_flashinfer` | 使用 MiniCPM FlashInfer 注意力后端 |
| `--chunked-prefill-size 8192` | chunked prefill 大小 |
| `--max-running-requests 32` | 最大并发推理请求数 |
| `--skip-server-warmup` | 跳过服务预热 |
| `--port 31111` | 服务端口 |
| `--dense-as-sparse` | 使用 dense-as-sparse 模式 |

## 目录结构

```
SGLang-MiniCPM-SALA/
├── README.md                       # 本文件
├── install.sh                      # 一键安装脚本（uv + venv）
├── verify.sh                       # 环境验证脚本
├── .gitignore
└── packages/                       # 自定义包源码
    ├── sglang-minicpm/python/      # MiniCPM-SALA (SGLang 定制 fork)
    ├── infllmv2_cuda_impl/         # InfLLM v2 CUDA 实现
    ├── sparse_kernel/              # sparse_kernel_extension CUDA 扩展
    └── python/                     # Python 3.10 standalone（离线安装用）
```

## 手动安装

如果一键脚本不满足需求，可以分步执行：

```bash
# 0. 确保 uv 可用
pip install uv

# 1. 创建虚拟环境
uv venv --python 3.10 sglang_minicpm_sala_env
source sglang_minicpm_sala_env/bin/activate

# 2. 安装 MiniCPM-SALA 及所有依赖
uv pip install --upgrade pip setuptools wheel
uv pip install -e ./packages/sglang-minicpm/python

# 3. 编译安装 CUDA 扩展（需 nvcc，耗时较长）
cd packages/infllmv2_cuda_impl && python setup.py install && cd ../..
cd packages/sparse_kernel && python setup.py install && cd ../..

# 4. 安装额外依赖
uv pip install tilelang flash-linear-attention
```

## 定制包说明

| 包名 | 来源 | 说明 |
|------|------|------|
| `sglang` | [MiniCPM-SALA](https://github.com/GitHubstart0916/sglang-minicpm/tree/mixed_minicpm_cudagraph)（基于 [sgl-project/sglang](https://github.com/sgl-project/sglang) 修改） | **不可用公版 sglang 替代** |
| `infllm_v2` | [OpenBMB/infllmv2_cuda_impl](https://github.com/OpenBMB/infllmv2_cuda_impl) | CUDA 扩展，需编译 |
| `sparse_kernel_extension` | - | 单文件 CUDA 扩展，需编译 |

## 验证安装

```bash
bash verify.sh
```

## Q&A

**Q: CUDA 扩展编译失败怎么办？**

- 确保系统安装了 CUDA 12 以上（`nvcc --version` 检查）
- 确保 `gcc` / `g++` 可用
- 如果 `CXX` 环境变量被设为 `clang++ -pthread`，手动 `export CXX=g++`

**Q: 离线环境怎么安装？**

- 将 Python 3.10 standalone tarball 放到 `packages/python/` 目录下，脚本会自动检测使用
