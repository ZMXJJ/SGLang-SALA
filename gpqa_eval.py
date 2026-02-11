#!/usr/bin/env python3
"""
GPQA 独立评测脚本
评测逻辑复用自:
  gpqa_openai_simple_evals_fulldetail_gen_5aeece.py
  opencompass/datasets/gpqa.py
功能:
  1. 从 CSV 文件加载 GPQA 数据集（extended / main / diamond）
  2. 按 OpenCompass 的 shuffle 模式打乱选项
  3. 使用 OpenAI Simple Eval 风格的 prompt 调用 SGLang API
  4. 用正则提取预测答案，精确匹配计算 accuracy
  5. 输出每个子集的结果并保存详细评测记录
用法:
  python gpqa_eval.py [选项]
示例:
  python gpqa_eval.py --api_base http://127.0.0.1:30000 --model MiniCPM-SALA
  python gpqa_eval.py --subsets diamond --max_tokens 8192
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

# 1. Prompt 模板
ALIGN_PROMPT = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{question}\n\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}"
)

# 2. 数据加载
GPQA_SUBSETS = {
    "extended": "gpqa_extended.csv",
    "main": "gpqa_main.csv",
    "diamond": "gpqa_diamond.csv",
}

SHUFFLE_PATTERNS = ["ABCD", "BCDA", "CDAB", "DABC"]


def load_gpqa_dataset(data_dir: str, subset_name: str):
    """
    从 CSV 加载 GPQA 数据集

    CSV 结构:
      - row[7] : Question
      - row[8] : Correct Answer
      - row[9] : Incorrect Answer 1
      - row[10]: Incorrect Answer 2
      - row[11]: Incorrect Answer 3

    选项按 cnt % 4 决定的 shuffle_pattern 打乱，保证评测一致性。
    """
    csv_file = GPQA_SUBSETS[subset_name]
    csv_path = os.path.join(data_dir, csv_file)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    data = []
    cnt = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if row[7] == "Question":
                continue  # 跳过表头
            cnt += 1
            question = row[7]
            # 第一个选项是正确答案
            options = [row[8], row[9], row[10], row[11]]
            c = SHUFFLE_PATTERNS[cnt % 4]

            line = {"question": question}
            ground_truth = options[0]

            for i in range(4):
                line["ABCD"[i]] = options[ord(c[i]) - ord("A")]

            for i in range(4):
                if line["ABCD"[i]] == ground_truth:
                    line["answer"] = "ABCD"[i]
                    break

            data.append(line)

    return data


# 3. 答案后处理
ANSWER_PATTERN = re.compile(r"(?i)ANSWER\s*:\s*([A-D])")


def extract_answer(text: str):
    """从模型回复中提取 ANSWER: X 格式的答案。"""
    if text is None:
        return None
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None


# 4. 模型调用
def call_model(
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: int = 300,
):
    """通过 OpenAI 兼容 API 调用模型。"""
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return content, usage
    except requests.exceptions.Timeout:
        print(f"  [WARN] 请求超时 ({timeout}s)")
        return None, {}
    except Exception as e:
        print(f"  [ERROR] 请求失败: {e}")
        return None, {}


# 5. 评分
def evaluate(predictions, references):
    """计算 accuracy，返回结果和每条详细信息。"""
    if len(predictions) != len(references):
        return {"error": "predictions 和 references 长度不一致"}

    correct = 0
    count = 0
    details = []

    for pred, ref in zip(predictions, references):
        count += 1
        is_correct = pred == ref
        if is_correct:
            correct += 1
        details.append({"pred": pred, "answer": ref, "correct": is_correct})

    accuracy = 100.0 * correct / count if count > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": count, "details": details}


# 6. 单个子集评测流程
def eval_subset(
    subset_name: str,
    data: list,
    api_base: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    concurrency: int,
):
    """评测单个 GPQA 子集，支持并发请求。"""
    total = len(data)
    print(f"\n{'='*60}")
    print(f"  评测子集: GPQA_{subset_name} ({total} 题)")
    print(f"{'='*60}")

    # 构造所有 prompt
    prompts = []
    for item in data:
        prompt = ALIGN_PROMPT.format(
            question=item["question"],
            A=item["A"],
            B=item["B"],
            C=item["C"],
            D=item["D"],
        )
        prompts.append(prompt)

    # 并发调用模型
    raw_responses = [None] * total
    usages = [None] * total
    completed = 0

    def _infer(idx):
        content, usage = call_model(
            api_base, model, prompts[idx], max_tokens, temperature, timeout
        )
        return idx, content, usage

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_infer, i): i for i in range(total)}
        for future in as_completed(futures):
            idx, content, usage = future.result()
            raw_responses[idx] = content
            usages[idx] = usage
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  进度: {completed}/{total}")

    # 提取答案
    predictions = [extract_answer(resp) for resp in raw_responses]
    references = [item["answer"] for item in data]

    # 评分
    result = evaluate(predictions, references)

    # 构建详细记录
    eval_details = []
    for i, item in enumerate(data):
        eval_details.append(
            {
                "index": i,
                "question": item["question"][:100] + "..."
                if len(item["question"]) > 100
                else item["question"],
                "options": {
                    "A": item["A"][:80],
                    "B": item["B"][:80],
                    "C": item["C"][:80],
                    "D": item["D"][:80],
                },
                "ground_truth": references[i],
                "prediction": predictions[i],
                "correct": predictions[i] == references[i],
                "raw_response": raw_responses[i],
                "usage": usages[i],
            }
        )

    result["eval_details"] = eval_details
    return result


# 7. 主函数
def main():
    parser = argparse.ArgumentParser(
        description="GPQA 独立评测脚本（不依赖 OpenCompass）"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:30000",
        help="SGLang/vLLM API 地址 (default: http://127.0.0.1:30000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称，不指定则自动从 /v1/models 获取",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/user/zbw/opencompass/data/gpqa/dataset",
        help="GPQA CSV 数据目录",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["extended", "main", "diamond"],
        choices=["extended", "main", "diamond"],
        help="要评测的子集 (default: extended main diamond)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="最大生成 token 数 (default: 8192)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度 (default: 0.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="单次请求超时秒数 (default: 300)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="并发请求数 (default: 8)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个子集最多评测的题数，默认全部评测",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果输出目录，默认为 ./outputs/<timestamp>",
    )

    args = parser.parse_args()

    # 自动获取模型名称
    if args.model is None:
        try:
            resp = requests.get(f"{args.api_base}/v1/models", timeout=10)
            resp.raise_for_status()
            models = resp.json()["data"]
            args.model = models[0]["id"]
            print(f"自动检测到模型: {args.model}")
        except Exception as e:
            print(f"[ERROR] 无法自动获取模型名称: {e}")
            print("请使用 --model 参数手动指定")
            sys.exit(1)

    # 健康检查
    try:
        resp = requests.get(f"{args.api_base}/health", timeout=10)
        if resp.status_code == 200:
            print(f"服务健康检查通过: {args.api_base}")
        else:
            print(f"[WARN] 健康检查返回: {resp.status_code}")
    except Exception as e:
        print(f"[ERROR] 无法连接服务 {args.api_base}: {e}")
        sys.exit(1)

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, "outputs", timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n配置信息:")
    print(f"  API 地址:    {args.api_base}")
    print(f"  模型:        {args.model}")
    print(f"  数据目录:    {args.data_dir}")
    print(f"  评测子集:    {args.subsets}")
    print(f"  max_tokens:  {args.max_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  并发数:      {args.concurrency}")
    print(f"  输出目录:    {args.output_dir}")

    # 逐子集评测
    summary = {}
    total_start = time.time()

    for subset in args.subsets:
        print(f"\n加载数据: {subset} ...")
        data = load_gpqa_dataset(args.data_dir, subset)
        if args.max_samples is not None and len(data) > args.max_samples:
            data = data[: args.max_samples]
            print(f"  加载完成: {len(data)} 题 (已截断至 {args.max_samples})")
        else:
            print(f"  加载完成: {len(data)} 题")

        start_time = time.time()
        result = eval_subset(
            subset_name=subset,
            data=data,
            api_base=args.api_base,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            concurrency=args.concurrency,
        )
        elapsed = time.time() - start_time

        accuracy = result["accuracy"]
        correct = result["correct"]
        total = result["total"]
        summary[subset] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "elapsed_seconds": round(elapsed, 1),
        }

        print(f"\n  GPQA_{subset} 结果: {correct}/{total} = {accuracy:.2f}%")
        print(f"  耗时: {elapsed:.1f}s")

        # 保存详细结果
        detail_path = os.path.join(args.output_dir, f"GPQA_{subset}.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "subset": subset,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "elapsed_seconds": round(elapsed, 1),
                    "model": args.model,
                    "api_base": args.api_base,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "details": result["eval_details"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  详细结果已保存: {detail_path}")

    total_elapsed = time.time() - total_start

    # 汇总报告
    print(f"\n{'='*60}")
    print(f"  GPQA 评测汇总")
    print(f"{'='*60}")
    print(f"  模型: {args.model}")
    print(f"  总耗时: {total_elapsed:.1f}s")
    print(f"  {'-'*40}")

    for subset, info in summary.items():
        print(
            f"  GPQA_{subset:10s}: {info['correct']:3d}/{info['total']:3d} = {info['accuracy']:6.2f}%"
        )

    print(f"  {'-'*40}")

    # 计算总体 accuracy
    total_correct = sum(v["correct"] for v in summary.values())
    total_count = sum(v["total"] for v in summary.values())
    overall = 100.0 * total_correct / total_count if total_count > 0 else 0.0
    print(f"  {'Overall':16s}: {total_correct:3d}/{total_count:3d} = {overall:6.2f}%")
    print(f"{'='*60}")

    # 保存汇总
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "api_base": args.api_base,
                "timestamp": timestamp,
                "total_elapsed_seconds": round(total_elapsed, 1),
                "overall_accuracy": round(overall, 2),
                "overall_correct": total_correct,
                "overall_total": total_count,
                "subsets": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n汇总结果已保存: {summary_path}")
    print(f"详细结果目录: {args.output_dir}")


if __name__ == "__main__":
    main()
