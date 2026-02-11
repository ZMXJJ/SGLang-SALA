# MiniCPM-SALA 推理评测 Docker 镜像

本项目提供 **MiniCPM-SALA** 模型的自动化推理评测 Docker 镜像，用于评估模型在 GPQA 学术基准上的准确率（`acc`）和不同并发度下的推理速度（`benchmark_duration`）。

## 目录结构

```
SORA-MiniCPM-Eval/
├── SGLang-MiniCPM-SALA/            # SGLang 推理框架（含 MiniCPM FlashInfer attention 优化）
│   ├── install.sh                  # 环境安装脚本
│   ├── packages/                   # 离线依赖包
│   ├── README.md                   # SGLang 环境说明文档
│   └── ...
├── Dockerfile                      # Docker 镜像构建文件
├── entrypoint.sh                   # 容器入口脚本（编排整体评测流程）
├── gpqa_eval.py                    # GPQA 评测脚本
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
    --max-running-requests 128 \
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
| `--attention-backend` | `minicpm_flashinfer` | 使用 MiniCPM 专用 FlashInfer attention 后端 |
| `--chunked-prefill-size` | `8192` | Prefill 阶段每次处理的最大 token 数 |
| `--max-running-requests` | `128` | 服务端最大并发请求数 |
| `--skip-server-warmup` | - | 跳过服务预热，加快启动速度 |
| `--port` | 由 `PORT` 环境变量指定（默认 30000） | HTTP 服务端口 |
| `--dense-as-sparse` | - | MiniCPM 模型专用的稀疏注意力优化 |
| `--enable-metrics` | - | 开启 Prometheus 指标采集 |

---

## 2. 评测脚本说明

### 2.1 gpqa_eval.py — GPQA 准确率评测

**功能**：评测模型在 GPQA（Graduate-Level Google-Proof Q&A）数据集上的多选题准确率。

**评测逻辑**（与 OpenCompass `gpqa_openai_simple_evals` 一致）：

1. 从 CSV 文件加载 GPQA 数据集
2. 按 OpenCompass 的 shuffle 模式打乱选项顺序（`ABCD → BCDA → CDAB → DABC` 循环），确保评测可复现
3. 使用 OpenAI Simple Eval 风格的 prompt 模板：
   ```
   Answer the following multiple choice question. The last line of your
   response should be of the following format: 'ANSWER: $LETTER' (without
   quotes) where LETTER is one of ABCD. Think step by step before answering.
   ```
4. 通过 OpenAI 兼容 API（`/v1/chat/completions`）调用 SGLang 服务
5. 用正则 `(?i)ANSWER\s*:\s*([A-D])` 从回复中提取答案
6. 精确匹配计算 accuracy

**主要参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_base` | `http://127.0.0.1:30000` | SGLang API 地址 |
| `--data_dir` | `/data/gpqa` | GPQA CSV 数据集目录 |
| `--subsets` | `diamond` | 评测子集 |
| `--max_tokens` | `16384` | 最大生成 token 数 |
| `--temperature` | `0.0` | 采样温度（贪心解码） |
| `--concurrency` | `8` | 并发请求数 |
| `--max_samples` | 全部 | 每子集最多评测题数（调试用） |
| `--output_dir` | `/tmp/gpqa_outputs` | 结果输出目录 |

**输出文件**：
- `GPQA_diamond.json`：每道题的详细评测记录（题目、选项、模型回复、预测答案、是否正确）
- `summary.json`：汇总结果（overall accuracy）

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
- 使用 `eval_data_128.jsonl` 数据集（128 条真实评测数据）

**工作流程**：

1. 将 `eval_data_128.jsonl` 转换为 `sglang.bench_serving` 的 `custom` 格式（`conversations` 数组）
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
4. 运行 gpqa_eval.py，评测 GPQA diamond 子集，获取 acc
5. 运行 bench_serving.sh，在 3 档并发度下测量推理速度，获取 benchmark_duration
6. 组装最终 JSON 结果输出到 stdout
7. 关闭 SGLang 服务，容器退出
```

**错误处理**：
- SGLang 启动失败或超时 → `state=0`，输出错误信息
- GPQA 评测失败 → `state=0`，`error_msg` 包含 GPQA 错误
- Benchmark 失败（某档位 duration=0）→ `state=0`，`error_msg` 包含 benchmark 错误
- GPQA 和 benchmark 同时失败 → 错误信息用 `;` 连接

---

## 3. Docker 镜像

### 3.1 基础镜像

```
modelbest-registry.cn-beijing.cr.aliyuncs.com/public/cuda:12.9.1-cudnn-devel-ubuntu24.04
```

### 3.2 构建

```bash
cd /path/to/SORA-MiniCPM-Eval
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
  -v /path/to/model:/model/stage2_rc2_job_165691_iter_4400 \
  -v /path/to/gpqa_data:/data/gpqa \
  -v /path/to/eval_data_128.jsonl:/data/eval_data_128.jsonl \
  -e MODEL_PATH=/model/stage2_rc2_job_165691_iter_4400 \
  -e RECORD_ID=record_001 \
  -e USER_ID=user_001 \
  -e TASK_ID=task_001 \
  soar_eval:latest
```

### 4.2 快速测试（限制评测样本数）

```bash
docker run --gpus all \
  -v /path/to/model:/model/stage2_rc2_job_165691_iter_4400 \
  -v /path/to/gpqa_data:/data/gpqa \
  -v /path/to/eval_data_128.jsonl:/data/eval_data_128.jsonl \
  -e MODEL_PATH=/model/stage2_rc2_job_165691_iter_4400 \
  -e RECORD_ID=test_001 \
  -e USER_ID=test_user \
  -e TASK_ID=test_task \
  -e GPQA_MAX_SAMPLES=5 \
  -e BENCH_MAX_SAMPLES=5 \
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
| `MODEL_PATH` | 是 | `/model/stage2_rc2_job_165691_iter_4400` | 容器内模型路径 |
| `PORT` | 否 | `30000` | SGLang 服务端口 |
| `DATA_DIR` | 否 | `/data/gpqa` | GPQA 数据集目录（需挂载） |
| `EVAL_DATA` | 否 | `/data/eval_data_128.jsonl` | Benchmark 数据集路径（需挂载） |
| `RECORD_ID` | 是 | 空 | 提交记录 ID |
| `USER_ID` | 是 | 空 | 用户 ID |
| `TASK_ID` | 是 | 空 | 任务 ID |
| `GPQA_MAX_SAMPLES` | 否 | 不传则跑全量 | GPQA 每子集最多评测题数（调试用） |
| `BENCH_MAX_SAMPLES` | 否 | 不传则跑全量 128 条 | Benchmark 最多数据条数（调试用） |

---

## 6. 挂载卷

| 宿主机路径 | 容器路径 | 说明 |
|-----------|---------|------|
| 模型目录 | `/model/stage2_rc2_job_165691_iter_4400` | 模型权重文件 |
| GPQA 数据集目录 | `/data/gpqa` | 包含 `gpqa_diamond.csv` |
| Benchmark 数据集 | `/data/eval_data_128.jsonl` | 128 条评测数据（JSONL 格式） |

### 数据集格式

**GPQA 数据集**（`gpqa_diamond.csv`）：标准 GPQA CSV 格式，包含 Question、Correct Answer、Incorrect Answer 1/2/3 等列。

**Benchmark 数据集**（`eval_data_128.jsonl`）：每行一个 JSON 对象，包含以下字段：

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
| `state` | string | `"1"` 全部成功，`"0"` 有失败（GPQA 或 benchmark 任一失败即为 0） |
| `error_msg` | string | 成功时为空字符串；失败时包含错误描述，多个错误用 `;` 连接 |
| `acc` | float | GPQA diamond 子集准确率（百分制），失败时为 0 |
| `benchmark_duration.S1` | float | 无并发（串行）跑完所有请求的墙钟时间（秒） |
| `benchmark_duration.S8` | float | 8 并发下的墙钟时间（秒） |
| `benchmark_duration.Smax` | float | 无并发上限下的墙钟时间（秒） |
| `sort_by` | string | 排序字段，固定为 `"acc"` |

**可能的 `error_msg` 值**：
- `GPQA 评测脚本执行失败`
- `benchmark_duration 部分失败: S8,Smax 值为 0`
- `benchmark_duration 获取失败，脚本执行异常`
- `SGLang 服务启动失败`
- `SGLang 服务启动超时`
- 多个错误会用 `;` 连接

---

## 8. 预估耗时

| 配置 | 预计耗时 |
|------|---------|
| 快速测试（`GPQA_MAX_SAMPLES=5, BENCH_MAX_SAMPLES=10`） | 约 12-15 分钟 |
| 全量运行（GPQA diamond 198 题 + benchmark 128 条 × 3 档） | 约 4-6 小时 |

**快速测试耗时拆解**（`GPQA_MAX_SAMPLES=5, BENCH_MAX_SAMPLES=10`）：
- SGLang 启动（加载模型）：约 2-3 分钟
- GPQA 评测（5 题，并发 8）：约 2-3 分钟
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
        "-v", "/path/to/model:/model/stage2_rc2_job_165691_iter_4400",
        "-v", "/path/to/gpqa_data:/data/gpqa",
        "-v", "/path/to/eval_data_128.jsonl:/data/eval_data_128.jsonl",
        "-e", f"MODEL_PATH=/model/stage2_rc2_job_165691_iter_4400",
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

1. **`minicpm_flashinfer` CUDA Graph Bug**：当 `--chunked-prefill-size` 设置过大（如 16384）时，FlashInfer 的 `PrefillSplitQOKVIndptr` 可能在 CUDA Graph replay 阶段触发断言失败。当前已将该值固定为 8192 以规避此问题。

2. **`--disable-radix-cache`**：评测规则要求关闭前缀缓存，这会导致 GPQA 评测速度略慢（每条请求都需要完整 prefill），但不影响评测结果的正确性。

3. **`--max-running-requests 128`**：设置为 128 以匹配 benchmark 数据集大小（128 条），同时避免过高并发触发 FlashInfer 内部 bug。

4. **答案提取**：使用 `re.search()` 提取第一个 `ANSWER: X` 匹配，与 OpenCompass 官方 `GPQA_Simple_Eval_postprocess` 行为一致。
