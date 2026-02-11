#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/minicpm_sala_env"
PACKAGES_DIR="${SCRIPT_DIR}/packages"

# PyPI 镜像：优先命令行参数，其次环境变量，默认官方源
if [ -n "$1" ]; then
    export UV_INDEX_URL="$1"
elif [ -z "${UV_INDEX_URL}" ]; then
    export UV_INDEX_URL="https://pypi.org/simple"
fi

echo "============================================"
echo " MiniCPM-SALA 安装 (uv)"
echo "============================================"
echo "PyPI 镜像: ${UV_INDEX_URL}"

# ---- 创建 venv ----
if [ -d "${VENV_DIR}" ]; then
    echo "[1/6] venv 已存在，跳过"
else
    echo "[1/6] 创建虚拟环境 (Python 3.10)..."
    if timeout 30 uv venv --python 3.10 "${VENV_DIR}"; then
        :
    else
        echo "  [!] uv 未能自动获取 Python 3.10，尝试本地 fallback..."
        rm -rf "${VENV_DIR}"
        BUNDLED_TARBALL="${PACKAGES_DIR}/python/cpython-3.10.19+20260203-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        BUNDLED_PYTHON_DIR="${SCRIPT_DIR}/.python-3.10"
        if [ ! -d "${BUNDLED_PYTHON_DIR}" ] && [ -f "${BUNDLED_TARBALL}" ]; then
            echo "  -> 解压内置 Python 3.10..."
            mkdir -p "${BUNDLED_PYTHON_DIR}"
            tar -xzf "${BUNDLED_TARBALL}" -C "${BUNDLED_PYTHON_DIR}" --strip-components=1
        fi
        if [ -x "${BUNDLED_PYTHON_DIR}/bin/python3.10" ]; then
            echo "  -> 使用内置 Python: ${BUNDLED_PYTHON_DIR}/bin/python3.10"
            uv venv --python "${BUNDLED_PYTHON_DIR}/bin/python3.10" "${VENV_DIR}"
        else
            LOCAL_PY310=$(command -v python3.10 2>/dev/null || true)
            if [ -n "${LOCAL_PY310}" ]; then
                echo "  -> 使用系统 Python: ${LOCAL_PY310}"
                uv venv --python "${LOCAL_PY310}" "${VENV_DIR}"
            else
                echo "错误: 无法获取 Python 3.10"
                exit 1
            fi
        fi
    fi
fi

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:$PATH"
echo "Python: $(python --version)"

# ---- 编译环境准备 ----
# 修正编译器（python-build-standalone 会设 CXX="clang++ -pthread"，不兼容 CMake）
if command -v g++ &> /dev/null; then
    export CC=gcc CXX=g++
fi
# 确保 nvcc 在 PATH 中（pip install 不继承 uv 的环境）
if [ -z "${CUDA_HOME}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME="/usr/local/cuda"
    fi
fi
if [ -n "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

# ---- 安装 sglang ----
echo "[2/6] 安装 MiniCPM-SALA (sglang)..."
uv pip install "cmake>=3.26"
uv pip install --upgrade pip setuptools wheel
uv pip install -e "${PACKAGES_DIR}/sglang-minicpm/python[all]"

# ---- 安装 infllm_v2 ----
echo "[3/6] 编译 infllm_v2..."
cd "${PACKAGES_DIR}/infllmv2_cuda_impl"
python setup.py install

# ---- 安装 sparse_kernel ----
echo "[4/6] 编译 sparse_kernel..."
cd "${PACKAGES_DIR}/sparse_kernel"
python setup.py install

# ---- 安装 tilelang ----
echo "[5/6] 安装 tilelang..."
uv pip install tilelang

# ---- 安装 flash-linear-attention ----
echo "[6/6] 安装 flash-linear-attention..."
uv pip install flash-linear-attention

# ---- 验证 ----
echo ""
echo "============================================"
echo " 安装完成！"
echo "============================================"
echo "激活环境: source ${VENV_DIR}/bin/activate"
