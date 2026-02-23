# MiniCPM-SALA 推理评测 Docker 镜像

本项目提供 **MiniCPM-SALA** 模型的自动化推理评测 Docker 镜像，用于评估模型在综合数据集上的准确率（`acc`）和不同并发度下的推理速度（`benchmark_duration`）。

## 目录结构

```
SOAR/
├── SGLang-MiniCPM-SALA/            # SGLang 推理框架（含 MiniCPM FlashInfer attention 优化）
│   ├── install.sh                  # 环境安装脚本
│   ├── packages/                   # 离线依赖包
│   └── ...
├── data/                           # 评测数据集
│   ├── perf_public_set.jsonl       # 公开集（acc 评测）
│   ├── perf_private_set.jsonl      # 私有集（acc 评测）
│   ├── speed_eval.jsonl            # 速度评测数据（benchmark_duration）
│   ├── perf_public_set_source.jsonl    # 公开集（溯源版，可选）
│   ├── perf_private_set_source.jsonl   # 私有集（溯源版，可选）
│   └── meta_info.md                # 数据集说明
├── Dockerfile                      # Docker 镜像构建文件
├── entrypoint.sh                   # 容器入口脚本（编排整体评测流程）
├── eval_model.py                   # 模型准确率评测脚本
├── bench_serving.sh                # Benchmark Duration 评测脚本
└── README.md                       # 本说明文档
```

---

## 1. SGLang 环境说明

### 1.1 SGLang-MiniCPM-SALA

`SGLang-MiniCPM-SALA/` 是针对 MiniCPM 模型优化的 SGLang 推理框架，包含：

- **MiniCPM FlashInfer attention backend**：专为 MiniCPM 架构优化的 attention 计算后端
- **离线安装包**：所有 Python 依赖已打包，支持无网络环境下安装
- **虚拟环境**：安装后自动创建 `sglang_minicpm_sala_env` Python 虚拟环境

### 1.2 环境安装（Docker 构建时自动完成）

镜像构建过程中，Dockerfile 会自动执行以下步骤：

```bash
cd /opt/SGLang-MiniCPM-SALA
bash install.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

`install.sh` 会：
1. 使用 Python 3.10 创建虚拟环境 `sglang_minicpm_sala_env`
2. 安装 SGLang 及所有依赖（FlashInfer、PyTorch 等）
3. 编译 MiniCPM 专用的 CUDA kernel

### 1.3 SGLang 服务启动参数

容器运行时，`entrypoint.sh` 使用以下参数启动 SGLang 推理服务：

```bash
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --tp-size "${GPU_PER_WORKER}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --disable-cuda-graph \
    --skip-server-warmup \
    --port "${PORT}" \
    --dense-as-sparse \
    --enable-metrics
```

**参数说明**：

| 参数 | 值 | 说明 |
|------|------|------|
| `--model-path` | 由 `MODEL_PATH` 环境变量指定 | 模型权重路径 |
| `--trust-remote-code` | - | 允许执行模型仓库中的自定义代码 |
| `--disable-radix-cache` | - | 禁用 Radix Cache（前缀缓存），评测规则要求 |
| `--attention-backend` | `minicpm_flashinfer` | 使用 MiniCPM 专用 FlashInfer attention 后端（Blackwell / sm_120 推荐） |
| `--chunked-prefill-size` | `8192` | Prefill 阶段每次处理的最大 token 数 |
| `--tp-size` | 由 `GPU_PER_WORKER` 环境变量指定 | 张量并行大小（单机多卡） |
| `--max-running-requests` | 由 `MAX_RUNNING_REQUESTS` 环境变量指定 | 服务端最大并发请求数 |
| `--disable-cuda-graph` | 由 `DISABLE_CUDA_GRAPH=1` 启用 | 禁用 CUDA Graph（用于规避 flashinfer 在高并发下的不稳定问题） |
| `--skip-server-warmup` | - | 跳过服务预热，加快启动速度 |
| `--port` | 由 `PORT` 环境变量指定（默认 30000） | HTTP 服务端口 |
| `--dense-as-sparse` | - | MiniCPM 模型专用的稀疏注意力优化 |
| `--enable-metrics` | - | 开启 Prometheus 指标采集 |

---

## 2. 评测脚本说明

### 2.1 eval_model.py — 准确率评测

**功能**：评测模型在综合数据集上的准确率，支持多种任务类型。

**支持的任务类型**：

| 任务类型 | 说明 | 评分方式 |
|---------|------|---------|
| `mcq` | 多选题（来源 GPQA 等） | 提取 `ANSWER: X` 精确匹配 |
| `niah` | Needle in a Haystack | 包含匹配（命中任一答案即满分） |
| `cwe` | 关键词提取（Closed-world） | 多关键词覆盖率 |
| `fwe` | 关键词提取（Free-world） | 多关键词覆盖率 |
| `qa` | 问答 | 包含匹配 |
| `vt` | Variable Tracking | 包含匹配 |

**主要参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_base` | `http://127.0.0.1:30000` | SGLang API 地址 |
| `--model_path` | - | 模型路径（用于加载 tokenizer） |
| `--data_path` | `/data/perf_public_set.jsonl` | 评测数据集路径（容器内默认挂载到 `/data`） |
| `--max_seq_len` | `262144` | 最大序列长度 |
| `--concurrency` | `8` | 并发请求数 |
| `--num_samples` | 全部 | 最多评测样本数（调试用） |

**输出文件**：
- `predictions.jsonl`：每条样本的详细评测记录
- `summary.json`：汇总结果（`overall_accuracy`）

### 2.2 bench_serving.sh — Benchmark Duration 评测

**功能**：使用 SGLang 官方 `sglang.bench_serving` 工具，测量模型在不同并发度下的推理速度。

**评测规则**：

- **主要指标**：`benchmark_duration`（秒），跑完所有评测请求的总墙钟时间，越短越好
- **测试配置**：在 3 档并发度下分别测试：

| 档位 | 并发设置 | 说明 |
|------|----------|------|
| `S1` | `--max-concurrency 1` | 无并发，串行执行。反映单条请求的 Prefill + Decode 端到端时间 |
| `S8` | `--max-concurrency 8` | 低并发 |
| `Smax` | 不设 `--max-concurrency` | 无并发上限，全部请求同时发送 |

- 每档测试前清除 prefix cache（`--flush-cache`）
- 使用 `speed_eval.jsonl` 数据集（默认 128 条真实评测数据）

**工作流程**：

1. 将 `speed_eval.jsonl` 转换为 `sglang.bench_serving` 的 `custom` 格式（`conversations` 数组）
2. 依次在 S1 → S8 → Smax 三档并发度下运行 `sglang.bench_serving`
3. 从输出中提取 `Benchmark duration (s):` 的值
4. 输出汇总 JSON：`{"S1": xx.xx, "S8": xx.xx, "Smax": xx.xx}`

### 2.3 entrypoint.sh — 容器入口脚本

**功能**：编排整个评测流程，是容器的 `ENTRYPOINT`。

**执行流程**：

```
1. 激活 sglang_minicpm_sala_env 虚拟环境
2. 后台启动 SGLang 推理服务
3. 等待服务就绪（轮询 /health + /v1/models，最多等 10 分钟）
4. 运行 eval_model.py，评测准确率，获取 acc
5. 运行 bench_serving.sh，在 3 档并发度下测量推理速度，获取 benchmark_duration
6. 组装最终 JSON 结果输出到 stdout
7. 关闭 SGLang 服务，容器退出
```

**错误处理**：
- SGLang 启动失败或超时 → `state=0`，输出错误信息
- 评测失败 → `state=0`，`error_msg` 包含错误描述
- Benchmark 失败（某档位 duration=0）→ `state=0`，`error_msg` 包含 benchmark 错误
- 评测和 benchmark 同时失败 → 错误信息用 `;` 连接

---

## 3. Docker 镜像

### 3.1 基础镜像

```
modelbest-registry.cn-beijing.cr.aliyuncs.com/public/cuda:12.9.1-cudnn-devel-ubuntu24.04
```

### 3.2 构建

```bash
cd /path/to/SOAR
docker build -t soar_eval .
```

### 3.3 推送到远程仓库

```bash
docker tag soar_eval:latest modelbest-registry.cn-beijing.cr.aliyuncs.com/openbmb/sglang_test:2.0.4
docker push modelbest-registry.cn-beijing.cr.aliyuncs.com/openbmb/sglang_test:2.0.4
```

---

## 4. 运行方式

### 4.1 基本运行

```bash
docker run --gpus all \
  -v /path/to/model:/home/user/linbiyuan/models/MiniCPM-SALA \
  -v /path/to/data:/data \
  -e MODEL_PATH=/home/user/linbiyuan/models/MiniCPM-SALA \
  -e RECORD_ID=record_001 \
  -e USER_ID=user_001 \
  -e TASK_ID=task_001 \
  soar_eval:latest
```

### 4.2 快速测试（限制评测样本数）

```bash
docker run --gpus all \
  -v /path/to/model:/home/user/linbiyuan/models/MiniCPM-SALA \
  -v /path/to/data:/data \
  -e MODEL_PATH=/home/user/linbiyuan/models/MiniCPM-SALA \
  -e RECORD_ID=test_001 \
  -e USER_ID=test_user \
  -e TASK_ID=test_task \
  -e PERF_MAX_SAMPLES=5 \
  -e SPEED_MAX_SAMPLES=5 \
  soar_eval:latest
```

### 4.3 指定 GPU

```bash
# 全部 GPU
docker run --gpus all ...

# 指定单卡
docker run --gpus '"device=0"' ...

# 指定多卡
docker run --gpus '"device=0,1"' ...
```

---

## 5. 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `MODEL_PATH` | 是 | - | 容器内模型路径 |
| `PORT` | 否 | `30000` | SGLang 服务端口 |
| `PERF_DATA` | 否 | `/data/perf_public_set.jsonl` | 准确率评测数据集路径（JSONL）；切私有集可传 `/data/perf_private_set.jsonl` |
| `SPEED_DATA` | 否 | `/data/speed_eval.jsonl` | 速度评测数据集路径（JSONL） |
| `PERF_MAX_SAMPLES` | 否 | 空 | 准确率评测最多样本数（调试用，不传则跑全量） |
| `SPEED_MAX_SAMPLES` | 否 | 空 | 速度评测最多数据条数（调试用，不传则跑全量） |
| `GPU_PER_WORKER` | 否 | `1` | `--tp-size`，单机多卡时设置 |
| `MAX_RUNNING_REQUESTS` | 否 | `32` | `--max-running-requests`，建议在全量 benchmark（128 条）时设为 `128` 以避免排队 |
| `DISABLE_CUDA_GRAPH` | 否 | `1` | 为 `1` 时禁用 CUDA Graph（更稳；若想尝试更快可设为 `0`） |
| `RECORD_ID` | 是 | 空 | 提交记录 ID |
| `USER_ID` | 是 | 空 | 用户 ID |
| `TASK_ID` | 是 | 空 | 任务 ID |
| `SGL_KERNEL_WHEEL` | 否 | 空 | 选手提交的 `sgl-kernel` wheel 路径（优先级高于 `INPUT_URL`） |
| `INPUT_URL` | 否 | 空 | 选手提交文件的预签名下载直链（作为 wheel 来源；未传 `SGL_KERNEL_WHEEL` 时使用） |
| `SCORE_SYNC_URL` | 否 | 空 | 分数同步回调基址（POST `{SCORE_SYNC_URL}/score/sync`） |
| `SCORE_SYNC_REQUIRED` | 否 | `0` | 为 `1` 时回调失败会置 `state=0`；为 `0` 时仅 stderr 警告，评测仍按实际结果输出 |
| `SCORE_SYNC_RETRIES` | 否 | `3` | 分数同步重试次数 |
| `SCORE_SYNC_CONNECT_TIMEOUT` | 否 | `5` | 分数同步连接超时（秒） |
| `SCORE_SYNC_TIMEOUT` | 否 | `20` | 分数同步总超时（秒） |

---

## 6. 挂载卷

| 宿主机路径 | 容器路径 | 说明 |
|-----------|---------|------|
| 模型目录 | `/home/user/linbiyuan/models/MiniCPM-SALA` | 模型权重文件 |
| 数据目录 | `/data` | 包含 `perf_public_set.jsonl` / `perf_private_set.jsonl` / `speed_eval.jsonl` |
| 选手 wheel（可选） | 例如 `/submission/sgl-kernel.whl` | 配合 `SGL_KERNEL_WHEEL` 使用 |

### 数据集格式

**acc 评测数据集**（例如 `perf_public_set.jsonl` / `perf_private_set.jsonl`）：每行一个 JSON 对象，包含以下字段：

```json
{
  "prompt": "完整的 prompt 文本",
  "input_length": 1234,
  "output_length": 567,
  "task": "mcq",
  "gold": "B"
}
```

**速度评测数据集**（`speed_eval.jsonl`）：每行一个 JSON 对象，包含以下字段：

```json
{
  "index": 0,
  "question": "...",
  "input": "完整的 prompt 文本（用于推理）",
  "model_response": "参考回复",
  "prompt_tokens": 1234,
  "completion_tokens": 567
}
```

---

## 6.1 比赛提交流程（推荐：官方 evaluator 镜像 + wheel）

本评测镜像支持在**启动 SGLang 之前**，通过 `pip` 安装选手提交的 **`sgl-kernel` wheel**，从而允许选手在**不修改评测脚本/规则**的前提下优化推理性能（加速但不影响 `acc`）。

**优先级**：`SGL_KERNEL_WHEEL` > `INPUT_URL`。

### 方式 A：挂载 wheel（推荐，离线也可用）

```bash
docker run --gpus all \
  -v /path/to/model:/home/user/linbiyuan/models/MiniCPM-SALA:ro \
  -v /path/to/data:/data:ro \
  -v /path/to/sgl-kernel.whl:/submission/sgl-kernel.whl:ro \
  -e MODEL_PATH=/home/user/linbiyuan/models/MiniCPM-SALA \
  -e PERF_DATA=/data/perf_public_set.jsonl \
  -e SPEED_DATA=/data/speed_eval.jsonl \
  -e SGL_KERNEL_WHEEL=/submission/sgl-kernel.whl \
  -e RECORD_ID=record_001 \
  -e USER_ID=user_001 \
  -e TASK_ID=task_001 \
  modelbest-registry.cn-beijing.cr.aliyuncs.com/openbmb/sglang_test:2.0.4
```

### 方式 B：使用预签名链接下载 wheel（需要容器可联网）

```bash
docker run ... \
  -e INPUT_URL="https://example.com/presigned-download.whl" \
  modelbest-registry.cn-beijing.cr.aliyuncs.com/openbmb/sglang_test:2.0.4
```

### wheel 安全约束（评测侧强制）

- wheel 必须是 `sgl-kernel`（wheel 内 `METADATA Name` 为 `sgl-kernel` / `sgl_kernel`）
- wheel 只允许包含 `sgl_kernel/` 以及 `sgl_kernel-*.dist-info/`、`sgl_kernel-*.data/`、`sgl_kernel.libs/` 等必要文件；禁止 `.pth` 与其它顶层包
- 安装采用 `pip install --no-deps --force-reinstall`，避免选手替换 Torch / FlashInfer / SGLang 等依赖（保证公平与可复现）

---

## 7. 输出格式

容器执行完毕后，**stdout 最后输出一段 JSON**。

### 成功示例

```json
{
  "record_id": "record_001",
  "user_id": "user_001",
  "result": {
    "error_msg": "",
    "score": {
      "acc": 30.0,
      "benchmark_duration": {
        "S1": 1234.56,
        "S8": 456.78,
        "Smax": 198.32
      }
    },
    "sort_by": "acc"
  },
  "state": "1",
  "task_id": "task_001"
}
```

### 失败示例

```json
{
  "record_id": "record_001",
  "user_id": "user_001",
  "result": {
    "error_msg": "benchmark_duration 部分失败: S8,Smax 值为 0",
    "score": {
      "acc": 30.0,
      "benchmark_duration": {
        "S1": 1234.56,
        "S8": 0,
        "Smax": 0
      }
    },
    "sort_by": "acc"
  },
  "state": "0",
  "task_id": "task_001"
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `state` | string | `"1"` 全部成功，`"0"` 有失败（评测或 benchmark 任一失败即为 0） |
| `error_msg` | string | 成功时为空字符串；失败时包含错误描述，多个错误用 `;` 连接 |
| `acc` | float | 综合评测准确率（百分制），失败时为 0 |
| `benchmark_duration.S1` | float | 无并发（串行）跑完所有请求的墙钟时间（秒） |
| `benchmark_duration.S8` | float | 8 并发下的墙钟时间（秒） |
| `benchmark_duration.Smax` | float | 无并发上限下的墙钟时间（秒） |
| `sort_by` | string | 排序字段，固定为 `"acc"` |

**可能的 `error_msg` 值**：
- `评测脚本执行失败`
- `评测完成但未找到 summary.json`
- `benchmark_duration 部分失败: S8,Smax 值为 0`
- `benchmark_duration 获取失败，脚本执行异常`
- `SGLang 服务启动失败`
- `SGLang 服务启动超时`
- 多个错误会用 `;` 连接

---

## 8. 预估耗时

| 配置 | 预计耗时 |
|------|---------|
| 快速测试（`PERF_MAX_SAMPLES=5, SPEED_MAX_SAMPLES=10`） | 约 12-15 分钟 |
| 全量运行（180 题 acc + benchmark 128 条 × 3 档） | 约 4-6 小时 |

**快速测试耗时拆解**（`PERF_MAX_SAMPLES=5, SPEED_MAX_SAMPLES=10`）：
- SGLang 启动（加载模型）：约 2-3 分钟
- acc 评测（5 题，并发 8）：约 2-3 分钟
- Benchmark（S1 + S8 + Smax）：约 8 分钟

建议容器超时设为 **12 小时**。

---

## 9. 后端集成示例（Python）

```python
import subprocess
import json

record_id = "record_001"
user_id = "user_001"
task_id = "task_001"

result = subprocess.run(
    [
        "docker", "run", "--rm", "--gpus", '"device=0"',
        "-v", "/path/to/model:/home/user/linbiyuan/models/MiniCPM-SALA",
        "-v", "/path/to/data:/data",
        "-e", f"MODEL_PATH=/home/user/linbiyuan/models/MiniCPM-SALA",
        "-e", f"RECORD_ID={record_id}",
        "-e", f"USER_ID={user_id}",
        "-e", f"TASK_ID={task_id}",
        "modelbest-registry.cn-beijing.cr.aliyuncs.com/openbmb/sglang_test:2.0.4",
    ],
    capture_output=True,
    text=True,
    timeout=43200,  # 12 小时超时
)

# 从 stdout 提取最后的 JSON 块
lines = result.stdout.strip().split("\n")
json_start = None
for i, line in enumerate(lines):
    if line.strip() == "{":
        json_start = i
if json_start is not None:
    json_str = "\n".join(lines[json_start:])
    output = json.loads(json_str)
    print(json.dumps(output, indent=2, ensure_ascii=False))
```

---

## 10. 已知问题与注意事项

1. **`minicpm_flashinfer` + CUDA Graph 在高并发下可能不稳定**：在某些高并发/稀疏注意力场景下，FlashInfer 可能在 CUDA Graph replay 阶段触发内部错误。当前镜像默认 `DISABLE_CUDA_GRAPH=1`（即启动时追加 `--disable-cuda-graph`）以优先保证稳定性；如需尝试更高性能可将 `DISABLE_CUDA_GRAPH=0`（风险自担）。

2. **`--disable-radix-cache`**：评测规则要求关闭前缀缓存，这会导致评测速度略慢（每条请求都需要完整 prefill），但不影响评测结果的正确性。

3. **`MAX_RUNNING_REQUESTS` 建议与 benchmark 规模匹配**：全量速度评测默认 128 条请求，建议将 `MAX_RUNNING_REQUESTS=128`，避免服务端并发上限导致排队从而影响 `Smax` 指标。

4. **答案提取**：MCQ 任务使用 `re.search()` 提取第一个 `ANSWER: X` 匹配，与 OpenCompass 官方 `GPQA_Simple_Eval_postprocess` 行为一致。
