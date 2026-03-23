#!/bin/bash
# ============================================================
# SGLang MiniCPM 环境验证脚本
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/sglang_minicpm_sala_env"

# 激活 venv（如果未激活）
if [ -z "${VIRTUAL_ENV}" ]; then
    if [ -d "${VENV_DIR}" ]; then
        export VIRTUAL_ENV="${VENV_DIR}"
        export PATH="${VENV_DIR}/bin:$PATH"
    else
        echo "错误: 未找到 sglang_minicpm_sala_env 环境，请先运行 bash install.sh"
        exit 1
    fi
fi

PASS=0
FAIL=0

check() {
    local name="$1"
    local cmd="$2"
    printf "  %-30s" "${name}"
    if result=$(python -c "${cmd}" 2>&1); then
        echo -e "\033[32m✓\033[0m  ${result}"
        ((PASS++))
    else
        echo -e "\033[31m✗\033[0m  ${result}"
        ((FAIL++))
    fi
}

echo ""
echo "============================================"
echo " SGLang MiniCPM 环境验证"
echo "============================================"
echo ""
echo "Python: $(python --version 2>&1)"
echo "路径:   $(which python)"
echo ""

check "sglang"                   "import sglang; print(sglang.__version__)"
check "infllm_v2"                "import torch; import infllm_v2; print('OK')"
check "sparse_kernel_extension"  "import torch; import sparse_kernel_extension; print('OK')"
check "torch"                    "import torch; print(f'{torch.__version__}, CUDA {torch.version.cuda}')"
check "torch.cuda"               "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print(f'{torch.cuda.device_count()} GPU(s)')"

echo ""
echo "--------------------------------------------"
echo "  通过: ${PASS}  失败: ${FAIL}"
echo "--------------------------------------------"

if [ "${FAIL}" -gt 0 ]; then
    echo ""
    echo "存在失败项，请检查安装日志。"
    exit 1
fi
