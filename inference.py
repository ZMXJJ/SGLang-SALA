#!/usr/bin/env python3
"""
推理脚本：使用OpenAI SDK兼容的API对数据集进行推理
支持任何兼容OpenAI Chat Completions API的服务，包括vllm、ollama、lm-studio等
"""

import json
import os
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import requests
from threading import Lock


class APIInference:
    """使用OpenAI SDK兼容API进行推理的类"""
    
    def __init__(
        self,
        api_base: str,
        model_name: str = "default",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        max_workers: int = 8,
        retry: int = 3,
        timeout: int = 300
    ):
        """
        初始化推理器
        
        Args:
            api_base: API的base URL，如 "http://localhost:8000/v1"
            model_name: 模型名称
            api_key: API密钥
            temperature: 采样温度
            max_tokens: 最大生成token数
            max_workers: 并发worker数量
            retry: 重试次数
            timeout: 请求超时时间（秒）
        """
        self.api_base = api_base.rstrip('/')
        self.api_url = f"{self.api_base}/chat/completions"
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.retry = retry
        self.timeout = timeout
        
        self.success_count = 0
        self.fail_count = 0
        self.lock = Lock()
        
        # 定义prompt模板（参考opencompass配置）
        self.prompt_templates = {
            'math': {
                'system': None,
                'template': '{question}\n\nPut your final answer within a \\boxed{{}}.'
            },
            'code': {
                'system': 'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.',
                'template': '### Question:\n{question}\n\n### Answer: (use the provided format with backticks)\n\n'
            },
            'long_text_qa': {
                'system': None,
                'template': 'Please read the following text and answer the questions below.\n\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\n\nLet\'s think step by step.\n\nBased on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".'
            },
            'long_text_understanding': {
                'system': None,
                'template': 'Please read the following text and answer the questions below.\n\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\n\nLet\'s think step by step.\n\nBased on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".'
            }
        }
        
        print(f"初始化推理器:")
        print(f"  API URL: {self.api_url}")
        print(f"  模型: {model_name}")
        print(f"  并发数: {max_workers}")
        print(f"  温度: {temperature}")
        print(f"  最大token: {max_tokens}")
    
    def construct_prompt(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据题目类型构造prompt
        
        Args:
            item: 数据项
            
        Returns:
            包含system和user prompt的dict
        """
        item_type = item['type']
        question = item['question']
        context = item.get('context', '')
        
        # 获取对应类型的模板
        template_config = self.prompt_templates.get(item_type, {
            'system': None,
            'template': '{question}'
        })
        
        # 格式化prompt
        try:
            if context:
                user_prompt = template_config['template'].format(
                    question=question,
                    context=context
                )
            else:
                user_prompt = template_config['template'].format(
                    question=question
                )
        except KeyError:
            # 如果模板中没有context占位符，只使用question
            user_prompt = template_config['template'].format(
                question=question
            )
        
        return {
            'system': template_config['system'],
            'user': user_prompt
        }
    
    def call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        调用API
        
        Args:
            messages: 消息列表
            
        Returns:
            API响应
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        for attempt in range(self.retry):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'success': True,
                        'response': result['choices'][0]['message']['content'],
                        'usage': result.get('usage', {})
                    }
                else:
                    error_msg = f"API返回错误: {response.status_code} - {response.text}"
                    if attempt < self.retry - 1:
                        time.sleep(1 * (attempt + 1))  # 指数退避
                        continue
                    return {
                        'success': False,
                        'error': error_msg
                    }
            
            except requests.exceptions.Timeout:
                error_msg = f"请求超时 (>{self.timeout}s)"
                if attempt < self.retry - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return {
                    'success': False,
                    'error': error_msg
                }
            
            except Exception as e:
                error_msg = f"请求异常: {str(e)}"
                if attempt < self.retry - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return {
                    'success': False,
                    'error': error_msg
                }
        
        return {
            'success': False,
            'error': f"重试{self.retry}次后仍然失败"
        }
    
    def inference_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单个数据项进行推理
        
        Args:
            item: 数据项
            
        Returns:
            包含推理结果的数据项
        """
        # 构造prompt
        prompt_dict = self.construct_prompt(item)
        
        # 构造消息
        messages = []
        if prompt_dict['system']:
            messages.append({"role": "system", "content": prompt_dict['system']})
        messages.append({"role": "user", "content": prompt_dict['user']})
        
        # 调用API
        result = self.call_api(messages)
        
        # 更新统计
        with self.lock:
            if result['success']:
                self.success_count += 1
            else:
                self.fail_count += 1
        
        # 构造输出
        output = {
            'id': item['id'],
            'type': item['type'],
            'question': item['question'],
            'answer': item.get('answer', ''),
            'prediction': result.get('response', ''),
            'success': result['success']
        }
        
        if not result['success']:
            output['error'] = result.get('error', '')
        
        if 'usage' in result:
            output['usage'] = result['usage']
        
        return output
    
    def inference_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            data: 数据列表
            
        Returns:
            推理结果列表
        """
        results = []
        
        print(f"\n开始推理，共 {len(data)} 条数据...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.inference_single, item): item 
                for item in data
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(data), desc="推理进度") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        
        return results


def load_dataset(filepath: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    加载数据集
    
    Args:
        filepath: 数据集文件路径
        limit: 限制读取的数量（用于测试）
        
    Returns:
        数据列表
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            data.append(json.loads(line.strip()))
    return data


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    保存推理结果
    
    Args:
        results: 推理结果列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='使用OpenAI SDK兼容API对数据集进行推理')
    
    # 必需参数
    parser.add_argument('--api_base', type=str, required=True,
                        help='API的base URL，如 http://localhost:8000/v1')
    parser.add_argument('--input', type=str, default='dataset.jsonl',
                        help='输入数据集路径 (默认: dataset.jsonl)')
    parser.add_argument('--output', type=str, default='results.jsonl',
                        help='输出结果路径 (默认: results.jsonl)')
    
    # 可选参数
    parser.add_argument('--model_name', type=str, default='default',
                        help='模型名称 (默认: default)')
    parser.add_argument('--api_key', type=str, default='EMPTY',
                        help='API密钥 (默认: EMPTY)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='采样温度 (默认: 0.0)')
    parser.add_argument('--max_tokens', type=int, default=65536,
                        help='最大生成token数 (默认: 65536)')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='并发worker数量 (默认: 8)')
    parser.add_argument('--retry', type=int, default=3,
                        help='失败重试次数 (默认: 3)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='请求超时时间/秒 (默认: 300)')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制处理的数据量（用于测试）')
    parser.add_argument('--filter_type', type=str, default=None,
                        choices=['math', 'code', 'long_text_qa', 'long_text_understanding'],
                        help='只推理指定类型的数据')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 加载数据
    print(f"加载数据集: {args.input}")
    data = load_dataset(args.input, limit=args.limit)
    print(f"加载了 {len(data)} 条数据")
    
    # 按类型筛选
    if args.filter_type:
        data = [item for item in data if item['type'] == args.filter_type]
        print(f"筛选类型 '{args.filter_type}' 后剩余 {len(data)} 条数据")
    
    if len(data) == 0:
        print("没有数据需要推理")
        return
    
    # 统计数据类型
    type_count = {}
    for item in data:
        t = item['type']
        type_count[t] = type_count.get(t, 0) + 1
    
    print("\n数据类型分布:")
    for t, count in sorted(type_count.items()):
        print(f"  {t}: {count} 条")
    
    # 初始化推理器
    inferencer = APIInference(
        api_base=args.api_base,
        model_name=args.model_name,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        retry=args.retry,
        timeout=args.timeout
    )
    
    # 执行推理
    start_time = time.time()
    results = inferencer.inference_batch(data)
    end_time = time.time()
    
    # 保存结果
    save_results(results, args.output)
    
    # 统计信息
    elapsed_time = end_time - start_time
    print("\n" + "=" * 60)
    print("推理完成统计")
    print("=" * 60)
    print(f"总数据量: {len(data)}")
    print(f"成功: {inferencer.success_count}")
    print(f"失败: {inferencer.fail_count}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每条: {elapsed_time/len(data):.2f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    main()

