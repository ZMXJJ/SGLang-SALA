#!/usr/bin/env python3
"""
评分脚本：对推理结果进行评分
参考opencompass的评分方式：
- 数学题: MATHEvaluator (math_postprocess_v2)
- 代码题: 由于没有测试用例，跳过评分
- 长文本: first_option_postprocess + 精确匹配
"""

import json
import re
from typing import List, Dict, Any
from collections import defaultdict


class MathEvaluator:
    """数学题评分器，参考MATHEvaluator v2"""
    
    @staticmethod
    def extract_boxed_answer(text: str) -> str:
        """从文本中提取boxed答案"""
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1)
        return None
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """标准化答案"""
        if not text:
            return ""
        
        # 移除常见的单位和修饰词
        SUBSTITUTIONS = [
            ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
            (r'\ ', ''), (' ', ''), ('mbox', 'text'),
            (',\\text{and}', ','), ('\\text{and}', ','),
        ]
        
        REMOVED_EXPRESSIONS = [
            'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
            'hours', 'km', 'units', 'points', 'feet', 'minutes',
            'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters',
        ]
        
        result = text
        for before, after in SUBSTITUTIONS:
            result = result.replace(before, after)
        for expr in REMOVED_EXPRESSIONS:
            result = result.replace(expr, '')
        
        # 提取boxed内容
        result = re.sub(r'\\boxed\{(.*?)\}', r'\1', result)
        result = re.sub(r'\\text\{(.*?)\}', r'\1', result)
        result = re.sub(r'\\textbf\{(.*?)\}', r'\1', result)
        
        # 移除逗号（如果是纯数字）
        if result.replace(',', '').replace('.', '').replace('-', '').isdigit():
            result = result.replace(',', '')
        
        return result.strip()
    
    @staticmethod
    def postprocess(text: str) -> str:
        """后处理预测结果，参考math_postprocess_v2"""
        text = text.split('<|eot_id|>')[0]
        
        # 尝试提取boxed答案
        boxed_ans = MathEvaluator.extract_boxed_answer(text)
        if boxed_ans:
            return MathEvaluator.normalize_answer(boxed_ans)
        
        # 尝试从包含"final answer"或"answer is"的句子中提取
        for sentence in text.split('.'):
            if re.search('final answer|answer is', sentence.lower()):
                return MathEvaluator.normalize_answer(sentence)
        
        # 默认使用第一句
        first_sentence = text.split('.')[0]
        return MathEvaluator.normalize_answer(first_sentence)
    
    @staticmethod
    def is_equiv(pred: str, ref: str) -> bool:
        """判断预测答案和参考答案是否等价"""
        if not pred or not ref:
            return False
        
        # 标准化后比较
        pred_norm = MathEvaluator.normalize_answer(pred)
        ref_norm = MathEvaluator.normalize_answer(ref)
        
        if pred_norm == ref_norm:
            return True
        
        # 尝试作为数字比较
        try:
            pred_num = float(pred_norm)
            ref_num = float(ref_norm)
            return abs(pred_num - ref_num) < 1e-6
        except:
            pass
        
        return pred == ref


class OptionEvaluator:
    """选择题评分器，参考first_option_postprocess"""
    
    @staticmethod
    def extract_option(text: str, options: str = 'ABCD') -> str:
        """从文本中提取选项，参考first_option_postprocess"""
        if not text:
            return ''
        
        text = text.replace('\n\nAssistant:', '').strip().replace('**', '')
        
        # 定义匹配模式（中英文）
        patterns = [
            rf'答案是?\s*([{options}])',
            rf'答案是?\s*：\s*([{options}])',
            rf'答案是?\s*:\s*([{options}])',
            rf'答案应该?是\s*([{options}])',
            rf'答案为\s*([{options}])',
            rf'选择?\s*([{options}])',
            rf'故选?\s*([{options}])',
            rf'([{options}])\s?是正确的',
            rf'([{options}])\s?是正确答案',
            rf'所以\s?([{options}])[.。$]?$',
            rf'因此\s?([{options}])[。\.]?$',
            rf'(?i)ANSWER\s*:\s*([{options}])',
            rf'[Tt]he answer is:?\s+\(?([{options}])\)?',
            rf'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
            rf'[Tt]he correct answer is:?.*?\(([{options}])\)',
            rf'boxed\{{([{options}])\}}',
        ]
        
        # 尝试匹配
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                for i in options:
                    if i in match.group(0):
                        return i
        
        # 后备方案：找第一个大写字母选项
        for char in text:
            if char.isupper() and char in options:
                return char
        
        # 最后尝试：如果第一个字符是选项
        if text and text[0] in options:
            return text[0]
        
        return ''


def evaluate_results(results_path: str, output_path: str = None):
    """
    评分主函数
    
    Args:
        results_path: 推理结果文件路径
        output_path: 评分结果输出路径（可选）
    """
    print("=" * 60)
    print("开始评分")
    print("=" * 60)
    
    # 加载结果
    print(f"\n加载推理结果: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    
    print(f"加载了 {len(results)} 条结果")
    
    # 初始化统计
    stats = {
        'total': {'total': 0, 'correct': 0, 'success': 0},
        'by_type': defaultdict(lambda: {'total': 0, 'correct': 0, 'success': 0})
    }
    
    details = []
    
    # 逐条评分
    print("\n开始评分...")
    for item in results:
        item_type = item['type']
        prediction = item.get('prediction', '')
        answer = item.get('answer', '')
        success = item.get('success', False)
        
        # 更新统计
        stats['total']['total'] += 1
        stats['by_type'][item_type]['total'] += 1
        
        if success:
            stats['total']['success'] += 1
            stats['by_type'][item_type]['success'] += 1
        
        # 根据类型评分
        is_correct = False
        processed_pred = ''
        processed_ans = ''
        
        if item_type == 'math':
            # 数学题评分
            if success and answer:
                processed_pred = MathEvaluator.postprocess(prediction)
                processed_ans = answer
                is_correct = MathEvaluator.is_equiv(processed_pred, processed_ans)
        
        elif item_type == 'code':
            # 代码题跳过评分（没有测试用例）
            is_correct = None  # 标记为无法评分
        
        elif item_type in ['long_text_qa', 'long_text_understanding']:
            # 长文本选择题评分
            if success and answer:
                processed_pred = OptionEvaluator.extract_option(prediction)
                processed_ans = answer.strip().upper()
                is_correct = (processed_pred == processed_ans)
        
        # 更新正确数
        if is_correct:
            stats['total']['correct'] += 1
            stats['by_type'][item_type]['correct'] += 1
        
        # 记录详情
        detail = {
            'id': item['id'],
            'type': item_type,
            'success': success,
            'correct': is_correct,
            'prediction': prediction[:200] if len(prediction) > 200 else prediction,
            'answer': answer,
            'processed_pred': processed_pred,
            'processed_ans': processed_ans
        }
        details.append(detail)
    
    # 计算准确率
    print("\n" + "=" * 60)
    print("评分结果")
    print("=" * 60)
    
    # 总体统计
    total_count = stats['total']['total']
    total_success = stats['total']['success']
    total_correct = stats['total']['correct']
    
    print(f"\n总体统计:")
    print(f"  总题数: {total_count}")
    print(f"  推理成功: {total_success} ({total_success/total_count*100:.2f}%)")
    
    # 计算可评分题目的准确率（排除代码题）
    scorable_total = sum(
        stats['by_type'][t]['total'] 
        for t in stats['by_type'] 
        if t != 'code'
    )
    scorable_correct = sum(
        stats['by_type'][t]['correct'] 
        for t in stats['by_type'] 
        if t != 'code'
    )
    
    if scorable_total > 0:
        print(f"  答案正确: {scorable_correct}/{scorable_total} ({scorable_correct/scorable_total*100:.2f}%)")
    
    # 按类型统计
    print(f"\n按类型统计:")
    for item_type in sorted(stats['by_type'].keys()):
        type_stats = stats['by_type'][item_type]
        total = type_stats['total']
        success = type_stats['success']
        correct = type_stats['correct']
        
        print(f"\n  {item_type}:")
        print(f"    总数: {total}")
        print(f"    推理成功: {success} ({success/total*100:.2f}%)")
        
        if item_type == 'code':
            print(f"    准确率: N/A (代码题无法评分，需要测试用例)")
        else:
            if success > 0:
                acc = correct / success * 100
                print(f"    准确率: {correct}/{success} ({acc:.2f}%)")
            else:
                print(f"    准确率: 0/0 (无成功推理)")
    
    # 保存详细结果
    if output_path:
        result_data = {
            'summary': {
                'total': total_count,
                'success': total_success,
                'correct': scorable_correct,
                'scorable_total': scorable_total,
                'success_rate': f"{total_success/total_count*100:.2f}%",
                'accuracy': f"{scorable_correct/scorable_total*100:.2f}%" if scorable_total > 0 else "N/A"
            },
            'by_type': {}
        }
        
        for item_type, type_stats in stats['by_type'].items():
            result_data['by_type'][item_type] = {
                'total': type_stats['total'],
                'success': type_stats['success'],
                'correct': type_stats['correct'],
                'success_rate': f"{type_stats['success']/type_stats['total']*100:.2f}%",
                'accuracy': f"{type_stats['correct']/type_stats['success']*100:.2f}%" if type_stats['success'] > 0 and item_type != 'code' else "N/A"
            }
        
        result_data['details'] = details
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细评分结果已保存到: {output_path}")
    
    # 显示一些错误样例
    print("\n" + "=" * 60)
    print("错误样例（前5个）")
    print("=" * 60)
    
    error_count = 0
    for detail in details:
        if detail['correct'] == False and detail['type'] != 'code':
            error_count += 1
            if error_count <= 5:
                print(f"\n样例 {error_count}:")
                print(f"  ID: {detail['id']}")
                print(f"  类型: {detail['type']}")
                print(f"  预测: {detail['processed_pred']}")
                print(f"  答案: {detail['processed_ans']}")
                print(f"  原始预测: {detail['prediction'][:100]}...")
    
    if error_count == 0:
        print("\n  没有错误样例（全部正确或都是代码题）")
    
    print("\n" + "=" * 60)
    print("✓ 评分完成")
    print("=" * 60)
    
    return stats, details


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='评分脚本')
    parser.add_argument('--input', type=str, required=True,
                        help='推理结果文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='评分结果输出路径（JSON格式）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    evaluate_results(args.input, args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

