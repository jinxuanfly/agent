#!/usr/bin/env python3
"""
多种子实验批量运行脚本
用于 Q1 论文增强计划的 P0-3 任务：5种子完整实验

使用方法:
    python run_multi_seed.py --max_train=200 --max_val=500

默认配置:
    - Agent1: GPT-4o-mini (gpt4om)
    - Agent2: Gemini (gemini)
    - Agent3: GPT-4o-mini (gpt4om)
    - Seeds: [42, 123, 456, 789, 1024]
"""

import subprocess
import sys
import os
import json
import time
import argparse
from pathlib import Path

# 配置
SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_PROVIDER1 = 'deepseek'
DEFAULT_PROVIDER2 = 'gemini'
DEFAULT_PROVIDER3 = 'gpt5.1'


def run_experiment(seed, max_train, max_val, provider1, provider2, provider3, skip_llm_inference=False):
    """运行单个seed的实验"""
    cmd = [
        sys.executable,
        'src/step4_hateful_memes/evaluate_with_llm.py',
        f'--max_train={max_train}',
        f'--max_val={max_val}',
        f'--provider1={provider1}',
        f'--provider2={provider2}',
        f'--provider3={provider3}',
        f'--seed={seed}',
    ]
    
    if skip_llm_inference:
        cmd.append('--skip_llm_inference')
    
    print(f"\n{'='*70}")
    print(f"运行实验: seed={seed}, train={max_train}, val={max_val}")
    print(f"Agent配置: {provider1} + {provider2} + {provider3}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n[错误] seed={seed} 实验失败 (耗时: {elapsed:.1f}s)")
        return False
    
    print(f"\n[完成] seed={seed} 实验成功 (耗时: {elapsed:.1f}s)")
    return True


def collect_results(seeds, provider1, provider2, provider3):
    """收集所有seed的结果"""
    result_dir = Path('results/hateful_memes')
    all_results = {}
    
    for seed in seeds:
        result_name = f'llm_{provider1}_{provider2}_{provider3}_seed{seed}'
        result_file = result_dir / f'evaluation_{result_name}.json'
        
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_results[seed] = data
        else:
            print(f"[警告] seed={seed} 的结果文件不存在: {result_file}")
    
    return all_results


def summarize_results(all_results, provider1, provider2, provider3):
    """汇总多种子结果，计算均值和标准差"""
    if not all_results:
        print("[错误] 没有可汇总的结果")
        return
    
    seeds = sorted(all_results.keys())
    methods = list(all_results[seeds[0]].keys())
    
    print(f"\n{'='*70}")
    print(f"多种子实验汇总 (Seeds: {seeds})")
    print(f"Agent配置: {provider1} + {provider2} + {provider3}")
    print(f"{'='*70}")
    print(f"\n{'方法':<25s} {'Accuracy均值':>12s} {'Accuracy标准差':>14s} {'Best':>8s}")
    print("-" * 65)
    
    summary = {}
    
    for method in methods:
        accuracies = []
        for seed in seeds:
            if method in all_results[seed]:
                acc = all_results[seed][method].get('accuracy', 0)
                accuracies.append(acc)
        
        if accuracies:
            import numpy as np
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            best_acc = np.max(accuracies)
            
            print(f"{method:<25s} {mean_acc:>12.2f} {std_acc:>14.2f} {best_acc:>8.2f}")
            summary[method] = {
                'mean': float(mean_acc),
                'std': float(std_acc),
                'best': float(best_acc),
                'values': [float(a) for a in accuracies],
            }
    
    # 保存汇总结果
    summary_file = Path('results/hateful_memes') / f'multi_seed_summary_{provider1}_{provider2}_{provider3}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'seeds': seeds,
            'providers': {'agent1': provider1, 'agent2': provider2, 'agent3': provider3},
            'summary': summary,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n汇总结果保存至: {summary_file}")
    
    # 关键指标对比
    if 'BestAgent' in summary and 'Uncertainty_Weighted_DS' in summary:
        best_agent_acc = summary['BestAgent']['mean']
        uw_ds_acc = summary['Uncertainty_Weighted_DS']['mean']
        gap = best_agent_acc - uw_ds_acc
        
        print(f"\n[关键指标]")
        print(f"  BestAgent均值: {best_agent_acc:.2f}%")
        print(f"  UW-DS均值: {uw_ds_acc:.2f}%")
        print(f"  差距: {gap:.2f}%")
        
        if gap <= 0:
            print(f"  ✅ 融合方法超越BestAgent!")
        elif gap < 5:
            print(f"  ⚠️ 差距较小 ({gap:.2f}%), 论文叙事可行")
        else:
            print(f"  ❌ 差距较大 ({gap:.2f}%), 需要改进")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='多种子实验批量运行')
    parser.add_argument('--max_train', type=int, default=200, help='训练样本数')
    parser.add_argument('--max_val', type=int, default=500, help='验证样本数')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1024',
                        help='种子列表，用逗号分隔')
    parser.add_argument('--provider1', type=str, default=DEFAULT_PROVIDER1,
                        help='Agent1模型')
    parser.add_argument('--provider2', type=str, default=DEFAULT_PROVIDER2,
                        help='Agent2模型')
    parser.add_argument('--provider3', type=str, default=DEFAULT_PROVIDER3,
                        help='Agent3模型')
    parser.add_argument('--skip_llm_inference', action='store_true',
                        help='跳过LLM推理（使用已有缓存）')
    parser.add_argument('--summary_only', action='store_true',
                        help='仅汇总已有结果，不运行实验')
    
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    print(f"多种子实验配置:")
    print(f"  Seeds: {seeds}")
    print(f"  Train/Val: {args.max_train}/{args.max_val}")
    print(f"  Agent1: {args.provider1}")
    print(f"  Agent2: {args.provider2}")
    print(f"  Agent3: {args.provider3}")
    print()
    
    if not args.summary_only:
        # 运行实验
        failed_seeds = []
        for seed in seeds:
            success = run_experiment(
                seed=seed,
                max_train=args.max_train,
                max_val=args.max_val,
                provider1=args.provider1,
                provider2=args.provider2,
                provider3=args.provider3,
                skip_llm_inference=args.skip_llm_inference,
            )
            if not success:
                failed_seeds.append(seed)
        
        if failed_seeds:
            print(f"\n[警告] 以下seed的实验失败: {failed_seeds}")
        else:
            print(f"\n[完成] 所有seed的实验成功!")
    
    # 汇总结果
    print("\n[汇总] 收集并分析结果...")
    all_results = collect_results(
        seeds=seeds,
        provider1=args.provider1,
        provider2=args.provider2,
        provider3=args.provider3,
    )
    
    if all_results:
        summary = summarize_results(
            all_results=all_results,
            provider1=args.provider1,
            provider2=args.provider2,
            provider3=args.provider3,
        )


if __name__ == '__main__':
    main()
