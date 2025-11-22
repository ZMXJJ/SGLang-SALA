#!/usr/bin/env python3
"""
vllm服务测试脚本
用于测试vllm服务是否正常运行
可以从本地或远程机器运行此脚本
"""

import requests
import json
import sys

# 配置服务地址
# 本地测试使用: "http://localhost:8000"
# 远程测试使用: "http://服务器IP:8000"
BASE_URL = "http://localhost:8000"  # 修改为实际的服务器地址

def test_health():
    """测试服务健康状态"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✓ 健康检查: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False

def test_models():
    """测试模型列表接口"""
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        print(f"✓ 模型列表: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"  可用模型: {json.dumps(models, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ 模型列表获取失败: {e}")
        return False

def test_completion():
    """测试文本生成接口"""
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "MiniCPM4-8B",
            "messages": [
                {"role": "user", "content": "你好，请介绍一下你自己。"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        print("正在测试文本生成...")
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        print(f"✓ 文本生成: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"  生成内容: {content}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ 文本生成失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print(f"测试vllm服务: {BASE_URL}")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("健康检查", test_health),
        ("模型列表", test_models),
        ("文本生成", test_completion)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n[测试] {name}")
        result = test_func()
        results.append((name, result))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    # 返回退出码
    all_passed = all(result for _, result in results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
        print(f"使用指定的服务地址: {BASE_URL}")
    main()

