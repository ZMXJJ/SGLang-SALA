#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score Results Script
Evaluates the inference results against the ground truth in the dataset.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('scorer')

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def evaluate(results_path: str, ground_truth_path: str = None):
    logger = logging.getLogger('scorer')
    
    logger.info(f"Loading results from: {results_path}")
    results = load_jsonl(results_path)
    
    # If ground truth path is not provided, try to find 'answer' in results
    # (assuming results contain the original data)
    
    total = len(results)
    correct = 0
    missing_answer = 0
    
    logger.info(f"Evaluating {total} items...")
    
    for item in results:
        prediction = item.get('prediction', '')
        answer = item.get('answer', '')
        
        if not answer and ground_truth_path:
            # TODO: Implement lookup if answer is not in results
            pass
            
        if not answer:
            missing_answer += 1
            continue
            
        # Simple evaluation logic: check if answer is contained in prediction
        # or exact match after normalization
        
        # Normalize
        pred_norm = str(prediction).strip().lower()
        ans_norm = str(answer).strip().lower()
        
        is_correct = False
        
        # Strategy 1: Exact match
        if pred_norm == ans_norm:
            is_correct = True
        # Strategy 2: Answer in prediction (for longer generation)
        elif ans_norm in pred_norm:
            is_correct = True
        # Strategy 3: Boxed answer extraction (common in math datasets)
        elif "\\boxed{" in pred_norm:
            # Very simple extraction
            try:
                extracted = pred_norm.split("\\boxed{")[1].split("}")[0]
                if extracted == ans_norm:
                    is_correct = True
            except:
                pass
                
        if is_correct:
            correct += 1
            
    score = (correct / total) * 100 if total > 0 else 0
    
    logger.info("=" * 40)
    logger.info("Evaluation Results")
    logger.info("=" * 40)
    logger.info(f"Total: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Missing Answer: {missing_answer}")
    logger.info(f"Accuracy: {score:.2f}%")
    logger.info("=" * 40)
    
    return score

def main():
    parser = argparse.ArgumentParser(description="Score inference results.")
    parser.add_argument("results_file", help="Path to the results JSONL file")
    parser.add_argument("--ground-truth", help="Path to ground truth file (optional if answers are in results)")
    
    args = parser.parse_args()
    
    setup_logging()
    evaluate(args.results_file, args.ground_truth)

if __name__ == "__main__":
    main()
