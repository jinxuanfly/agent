#!/usr/bin/env python3
"""
因果反思多种子批量运行脚本
==========================
对每个种子独立运行因果反思，汇总所有种子的均值和标准差。

使用方法：
    python src/step4_hateful_memes/run_causal_reflection_multi_seed.py --seeds=42,123,456,789,1024

注意：
- 使用 V2 模式（跳过文本消融，仅跨Agent证据交换，MAX_REFLECTIONS=1）
- 与已有 seed=42 的 v2 结果配置一致
"""

import subprocess
import sys
import os
import json
import time
import argparse
from pathlib import Path
import numpy as np

SEEDS = [42, 123, 456, 789, 1024]


def run_causal_reflection(seed, max_val=500, max_reflections=1, skip_ablation=True):
    """对单个种子运行因果反思"""
    cmd = [
        sys.executable,
        'src/step4_hateful_memes/evaluate_step5_causal_reflection.py',
        f'--max_val={max_val}',
        f'--seed={seed}',
        f'--max_reflections={max_reflections}',
    ]
    
    if skip_ablation:
        cmd.append('--skip_ablation')
    
    mode = "V2(跳过消融)" if skip_ablation else "完整"
    print(f"\n{'='*60}")
    print(f"因果反思: seed={seed}  {mode}  max_reflections={max_reflections}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n[失败] seed={seed} 运行失败 (耗时: {elapsed/60:.1f}分钟)")
        return False
    
    print(f"\n[完成] seed={seed} 运行成功 (耗时: {elapsed/60:.1f}分钟)")
    return True


def collect_and_summarize(seeds, result_dir='results/hateful_memes'):
    """收集所有种子的因果反思结果并汇总"""
    result_dir = Path(result_dir)
    
    # 先收集各种子 evaluation 中的 MajorityVoting 和 Uncertainty_Weighted_DS 作为 baseline
    eval_results = {}
    for seed in seeds:
        eval_file = result_dir / f'evaluation_llm_deepseek_gemini_gpt5.1_seed{seed}.json'
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_results[seed] = json.load(f)
    
    # 汇总因果反思结果
    print(f"\n{'='*70}")
    print(f"多种子因果反思汇总 ({len(seeds)} seeds: {seeds})")
    print(f"{'='*70}")
    
    print(f"\n{'方法':<30s} {'Accuracy均值':>12s} {'标准差':>14s} {'Best':>8s} {'Worst':>8s}")
    print("-" * 80)
    
    summary = {}
    
    # 1. MajorityVoting (来自各种子 evaluation)
    mv_accs = []
    for seed in seeds:
        if seed in eval_results and 'MajorityVoting' in eval_results[seed]:
            mv_accs.append(eval_results[seed]['MajorityVoting']['accuracy'])
    if mv_accs:
        print_row("MajorityVoting", mv_accs)
        summary['MajorityVoting'] = stats(mv_accs)
    
    # 2. Uncertainty_Weighted_DS (来自各种子 evaluation)
    uw_accs = []
    for seed in seeds:
        if seed in eval_results and 'Uncertainty_Weighted_DS' in eval_results[seed]:
            uw_accs.append(eval_results[seed]['Uncertainty_Weighted_DS']['accuracy'])
    if uw_accs:
        print_row("Uncertainty_Weighted_DS", uw_accs)
        summary['Uncertainty_Weighted_DS'] = stats(uw_accs)
    
    # 3. Causal Reflection (来自各种子 step5 结果)
    cr_accs = []
    for seed in seeds:
        # 尝试读取 step5_causal_reflection_v2_seed{seed}.json
        cr_file = result_dir / f'step5_causal_reflection_v2_seed{seed}.json'
        if not cr_file.exists():
            # 尝试旧格式
            cr_file = result_dir / f'step5_causal_reflection_v2.json'
            if seed == 42 and cr_file.exists():
                pass  # seed=42 的旧结果
            else:
                print(f"  [警告] seed={seed} 的因果反思结果不存在: {cr_file}")
                continue
        
        if cr_file.exists():
            with open(cr_file, 'r', encoding='utf-8') as f:
                cr_data = json.load(f)
            
            if 'full_sample' in cr_data and 'acc_reflection' in cr_data['full_sample']:
                acc = cr_data['full_sample']['acc_reflection']
                cr_accs.append(acc)
    
    if cr_accs:
        print_row("Causal_Reflection", cr_accs)
        summary['Causal_Reflection'] = stats(cr_accs)
    
    # 保存汇总
    summary_file = result_dir / 'causal_reflection_multi_seed_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'seeds': seeds,
            'summary': summary,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n汇总结果已保存到: {summary_file}")
    
    # 关键对比
    if 'MajorityVoting' in summary and 'Causal_Reflection' in summary:
        mv_mean = summary['MajorityVoting']['mean']
        cr_mean = summary['Causal_Reflection']['mean']
        cr_std = summary['Causal_Reflection']['std']
        delta = cr_mean - mv_mean
        print(f"\n[关键结论]")
        print(f"  Causal Reflection: {cr_mean:.2f}% ± {cr_std:.2f}%")
        print(f"  MajorityVoting:    {mv_mean:.2f}%")
        print(f"  提升:              +{delta:.2f}%")
    
    return summary


def print_row(name, accs):
    mean = np.mean(accs)
    std = np.std(accs)
    best = np.max(accs)
    worst = np.min(accs)
    print(f"{name:<30s} {mean:>12.2f} {std:>14.2f} {best:>8.2f} {worst:>8.2f}")


def stats(accs):
    return {
        'mean': float(np.mean(accs)),
        'std': float(np.std(accs)),
        'best': float(np.max(accs)),
        'worst': float(np.min(accs)),
        'values': [float(a) for a in accs],
    }


def main():
    parser = argparse.ArgumentParser(description='因果反思多种子批量运行')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1024',
                        help='种子列表，逗号分隔')
    parser.add_argument('--max_val', type=int, default=500, help='验证样本数')
    parser.add_argument('--max_reflections', type=int, default=1, help='最大反思轮数')
    parser.add_argument('--skip_ablation', action='store_true', default=True,
                        help='V2模式：跳过文本消融')
    parser.add_argument('--summary_only', action='store_true',
                        help='仅汇总已有结果，不运行实验')
    parser.add_argument('--skip_seeds', type=str, default='',
                        help='跳过的种子，逗号分隔（如 42）')
    
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    skip_seeds = [int(s.strip()) for s in args.skip_seeds.split(',')] if args.skip_seeds else []
    seeds = [s for s in seeds if s not in skip_seeds]
    
    print(f"因果反思多种子实验配置:")
    print(f"  种子: {seeds}")
    print(f"  验证样本: {args.max_val}")
    print(f"  最大反思轮数: {args.max_reflections}")
    print(f"  跳过文本消融: {args.skip_ablation}")
    print()
    
    if not args.summary_only:
        failed_seeds = []
        for seed in seeds:
            success = run_causal_reflection(
                seed=seed,
                max_val=args.max_val,
                max_reflections=args.max_reflections,
                skip_ablation=args.skip_ablation,
            )
            if not success:
                failed_seeds.append(seed)
        
        if failed_seeds:
            print(f"\n[警告] 以下种子运行失败: {failed_seeds}")
        else:
            print(f"\n[全部完成] 所有种子运行成功!")
    
    collect_and_summarize([s for s in SEEDS if s not in skip_seeds])


if __name__ == '__main__':
    main()