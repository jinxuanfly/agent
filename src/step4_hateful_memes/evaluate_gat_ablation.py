#!/usr/bin/env python3
"""
GAT共识层消融实验
================
验证GAT共识层在因果反思中是否有独特价值。

实验设计：
  A. MV-CR（当前）：MajorityVoting识别分歧 → 因果反思
  B. GAT-CR（GAT增强）：GAT共识后识别"深度分歧" → 因果反思
  C. 随机对照：随机选择等量样本 → 因果反思

核心问题：GAT共识能否筛选出"真正需要反思"的深度分歧样本？
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step2.gat_consensus import ConsensusEngine, GATConsensusLayer, DEVICE

# 配置
CHECKPOINT_DIR = 'checkpoints/hateful_memes'
RESULT_DIR = 'results/hateful_memes'
NUM_CLASSES = 2
EMBED_DIM = 256


def load_seed_data(seed, max_val=500):
    """加载某一种子的LLM推理缓存和标签"""
    seed_suffix = f'_seed{seed}'
    
    # 加载标签（从JSONL文件读取）
    DATA_DIR = 'data/Hateful_Memes/data'
    val_path = os.path.join(DATA_DIR, 'dev.jsonl')
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    val_data = val_data[:max_val]
    val_labels = torch.tensor([item['label'] for item in val_data])
    
    # 加载3个Agent的推理缓存
    all_beliefs = torch.zeros(max_val, 3, NUM_CLASSES)
    all_uncertainties = torch.zeros(max_val, 3)
    all_embs = torch.zeros(max_val, 3, EMBED_DIM)
    
    for i in range(3):
        path = os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}{seed_suffix}.pt')
        cache = torch.load(path, map_location='cpu')
        all_beliefs[:, i] = cache['beliefs'][:max_val]
        all_uncertainties[:, i] = cache['uncertainties'][:max_val]
        if 'embs' in cache:
            all_embs[:, i] = cache['embs'][:max_val]
    
    return all_beliefs, all_uncertainties, all_embs, val_labels


def compute_mv_predictions(all_beliefs):
    """计算MajorityVoting预测"""
    B = all_beliefs.shape[0]
    preds = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        stacked = torch.stack([all_beliefs[i, j] for j in range(3)])
        preds[i] = torch.mode(stacked.argmax(dim=1), 0).values.item()
    return preds


def identify_mv_disagreement(all_beliefs):
    """基于MV识别分歧样本"""
    B = all_beliefs.shape[0]
    disagreement_indices = []
    for i in range(B):
        preds = [all_beliefs[i, j].argmax().item() for j in range(3)]
        if len(set(preds)) > 1:
            disagreement_indices.append(i)
    return disagreement_indices


def identify_gat_disagreement(all_beliefs, all_uncertainties, all_embs, gat_engine, val_labels):
    """
    基于GAT共识识别深度分歧样本
    
    GAT共识后，如果3个Agent的信念仍然不一致（argmax不同），
    则认为是"深度分歧"——GAT无法解决的分歧。
    """
    B = all_beliefs.shape[0]
    gat_engine.layer.eval()
    
    gat_disagreement_indices = []
    gat_details = []  # 记录每个样本的GAT共识细节
    
    for idx in tqdm(range(B), desc="GAT共识分析"):
        agent_outputs = []
        for i in range(3):
            b_i = all_beliefs[idx, i]
            u_i = float(all_uncertainties[idx, i].item())
            emb_i = all_embs[idx, i]
            
            S = NUM_CLASSES / max(u_i, 1e-6)
            alpha_i = b_i * S + 1.0
            agent_outputs.append((alpha_i, b_i, u_i, emb_i))
        
        try:
            h = gat_engine.build_state(agent_outputs)
            h_final, n_iters, converged, energy_trace, attn_trace = \
                gat_engine.run(h, max_iters=10, tol=1e-4, verbose=False)
            
            outputs = gat_engine.extract_outputs(h_final)
            consensus_beliefs = torch.stack([o[1] for o in outputs])
            consensus_preds = consensus_beliefs.argmax(dim=1)
            
            original_preds = torch.tensor([all_beliefs[idx, j].argmax().item() for j in range(3)])
            
            # 深度分歧：GAT共识后Agent仍然不一致
            is_deep = len(set(consensus_preds.tolist())) > 1
            
            if is_deep:
                gat_disagreement_indices.append(idx)
            
            gat_details.append({
                'idx': idx,
                'original_preds': original_preds.tolist(),
                'consensus_preds': consensus_preds.tolist(),
                'original_disagreement': len(set(original_preds.tolist())) > 1,
                'gat_deep_disagreement': is_deep,
                'n_iters': n_iters,
                'converged': converged,
                'energy': energy_trace[-1] if energy_trace else 0,
            })
        except Exception as e:
            print(f"  [警告] GAT共识失败 idx={idx}: {e}")
            gat_details.append({
                'idx': idx,
                'error': str(e),
            })
    
    return gat_disagreement_indices, gat_details


def run_ablation(seed, max_val=500):
    """运行单个种子的GAT消融实验"""
    seed_suffix = f'_seed{seed}'
    
    print(f"\n{'='*60}")
    print(f"GAT消融实验: seed={seed}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    print(f"\n[1] 加载数据...")
    all_beliefs, all_uncertainties, all_embs, val_labels = load_seed_data(seed, max_val)
    
    # 2. 加载GAT模型
    print(f"\n[2] 加载GAT模型...")
    gat_model_path = os.path.join(CHECKPOINT_DIR, f'gat_consensus_llm{seed_suffix}.pt')
    
    if not os.path.exists(gat_model_path):
        print(f"  [错误] GAT模型不存在: {gat_model_path}")
        return None
    
    gat_layer = GATConsensusLayer(
        node_dim=EMBED_DIM + NUM_CLASSES + 1,
        hidden_dim=64,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    gat_layer.load_state_dict(torch.load(gat_model_path, map_location=DEVICE, weights_only=True))
    gat_layer.eval()
    
    gat_engine = ConsensusEngine(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, hidden_dim=64)
    gat_engine.layer = gat_layer
    
    # 3. 计算MV预测和分歧
    print(f"\n[3] 计算MV预测和分歧...")
    mv_preds = compute_mv_predictions(all_beliefs)
    mv_disagreement = identify_mv_disagreement(all_beliefs)
    mv_acc = (mv_preds == val_labels).float().mean().item()
    print(f"  MV准确率: {mv_acc*100:.2f}%")
    print(f"  MV分歧样本: {len(mv_disagreement)}/{max_val}")
    
    # 4. GAT共识分析
    print(f"\n[4] GAT共识分析...")
    gat_disagreement, gat_details = identify_gat_disagreement(
        all_beliefs, all_uncertainties, all_embs, gat_engine, val_labels
    )
    print(f"  GAT深度分歧样本: {len(gat_disagreement)}/{max_val}")
    
    # 5. 对比分析
    print(f"\n[5] 对比分析...")
    mv_set = set(mv_disagreement)
    gat_set = set(gat_disagreement)
    
    overlap = mv_set & gat_set
    mv_only = mv_set - gat_set
    gat_only = gat_set - mv_set
    
    print(f"  MV分歧: {len(mv_set)}")
    print(f"  GAT深度分歧: {len(gat_set)}")
    print(f"  重叠: {len(overlap)} ({len(overlap)/max(len(mv_set),1)*100:.1f}% of MV)")
    print(f"  MV独有(GAT已解决): {len(mv_only)}")
    print(f"  GAT独有(新发现): {len(gat_only)}")
    
    # 分析GAT已解决的样本（MV分歧但GAT共识后一致）
    if len(mv_only) > 0:
        resolved_labels = val_labels[list(mv_only)]
        resolved_mv_correct = (mv_preds[list(mv_only)] == resolved_labels).float().mean().item()
        print(f"    GAT已解决样本中MV正确率: {resolved_mv_correct*100:.2f}%")
    
    # 分析GAT深度分歧样本
    if len(gat_disagreement) > 0:
        deep_labels = val_labels[gat_disagreement]
        deep_mv_correct = (mv_preds[gat_disagreement] == deep_labels).float().mean().item()
        print(f"    GAT深度分歧中MV正确率: {deep_mv_correct*100:.2f}%")
    
    # 6. 加载因果反思结果做对比
    print(f"\n[6] 加载因果反思结果...")
    cr_file = os.path.join(RESULT_DIR, f'step5_causal_reflection_v2{seed_suffix}.json')
    if os.path.exists(cr_file):
        with open(cr_file, 'r', encoding='utf-8') as f:
            cr_data = json.load(f)
        cr_full_acc = cr_data['full_sample']['acc_reflection']
        cr_full_mv = cr_data['full_sample']['acc_mv']
        cr_delta = cr_data['full_sample']['delta']
        cr_correct_fixes = cr_data['results']['correct_fixes']
        cr_wrong_changes = cr_data['results']['wrong_changes']
        print(f"  CR全样本准确率: {cr_full_acc:.2f}%")
        print(f"  CR全样本MV: {cr_full_mv:.2f}%")
        print(f"  CR提升: {cr_delta:.2f}%")
        print(f"  CR正确修正/错误改变: {cr_correct_fixes}/{cr_wrong_changes}")
    else:
        cr_full_acc = None
        print(f"  [警告] 因果反思结果不存在")
    
    # 7. 模拟：如果只对GAT深度分歧做CR
    print(f"\n[7] 模拟GAT-CR效果...")
    # 对非深度分歧样本直接用MV，深度分歧样本假设CR修正
    # 这里用已有的CR结果来估算
    
    gat_cr_acc = None
    
    if cr_full_acc is not None:
        # 计算只在GAT深度分歧上做CR的理论效果
        # 非深度分歧样本：用MV
        non_deep_indices = list(set(range(max_val)) - gat_set)
        non_deep_correct = (mv_preds[non_deep_indices] == val_labels[non_deep_indices]).sum().item()
        
        # 深度分歧样本：假设CR能修正（使用CR实际结果中的修正率）
        # 从CR结果中读取实际修正数据
        cr_results_path = os.path.join(RESULT_DIR, f'causal_reflection_results{seed_suffix}.pt')
        if os.path.exists(cr_results_path):
            cr_results = torch.load(cr_results_path, map_location='cpu')
            final_preds = cr_results['final_predictions']
            cr_indices = cr_results.get('disagreement_indices', list(range(len(final_preds))))
            
            # 构建MV预测和CR修正后的预测
            full_mv_preds = mv_preds.clone()
            full_cr_preds = mv_preds.clone()
            for i, idx in enumerate(cr_indices):
                if final_preds[i] is not None:
                    full_cr_preds[idx] = final_preds[i]
            
            # 计算GAT-CR准确率
            gat_cr_correct = 0
            for idx in range(max_val):
                if idx in gat_set:
                    # 深度分歧：用CR修正后的预测
                    pred = full_cr_preds[idx].item()
                else:
                    # 非深度分歧：用MV
                    pred = mv_preds[idx].item()
                if pred == val_labels[idx].item():
                    gat_cr_correct += 1
            
            gat_cr_acc = gat_cr_correct / max_val * 100
            
            print(f"  GAT-CR模拟准确率: {gat_cr_acc:.2f}%")
            print(f"  MV准确率: {mv_acc*100:.2f}%")
            print(f"  GAT-CR vs MV: {gat_cr_acc - mv_acc*100:+.2f}%")
            print(f"  GAT-CR vs CR: {gat_cr_acc - cr_full_acc:+.2f}%")
            
            # 被GAT筛选掉的样本分析
            filtered_out = mv_set - gat_set
            if len(filtered_out) > 0:
                filtered_labels = val_labels[list(filtered_out)]
                filtered_mv_correct = (mv_preds[list(filtered_out)] == filtered_labels).float().mean().item()
                filtered_cr_correct = (full_cr_preds[list(filtered_out)] == filtered_labels).float().mean().item()
                print(f"\n  GAT筛选掉的样本分析 (n={len(filtered_out)}):")
                print(f"    MV正确率: {filtered_mv_correct*100:.2f}%")
                print(f"    CR修正后正确率: {filtered_cr_correct*100:.2f}%")
                print(f"    CR修正效果: {(filtered_cr_correct-filtered_mv_correct)*100:+.2f}%")
        else:
            print(f"  [警告] CR详细结果不存在，无法模拟")
    
    # 8. 保存结果
    result = {
        'seed': seed,
        'mv': {
            'accuracy': float(mv_acc * 100),
            'n_disagreement': len(mv_disagreement),
        },
        'gat': {
            'n_deep_disagreement': len(gat_disagreement),
            'n_overlap_mv': len(overlap),
            'n_gat_only': len(gat_only),
            'n_mv_only': len(mv_only),
        },
        'cr': {
            'full_accuracy': cr_full_acc,
            'delta': cr_delta if cr_full_acc else None,
        },
        'gat_cr_simulated': gat_cr_acc,
    }
    
    result_path = os.path.join(RESULT_DIR, f'gat_ablation{seed_suffix}.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {result_path}")
    
    # 保存详细分析
    detail_path = os.path.join(RESULT_DIR, f'gat_ablation_details{seed_suffix}.json')
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(gat_details, f, indent=2, ensure_ascii=False)
    print(f"详细分析已保存到: {detail_path}")
    
    return result


def summarize_all(seeds):
    """汇总所有种子的消融结果"""
    print(f"\n{'='*70}")
    print(f"GAT消融实验汇总 ({len(seeds)} seeds)")
    print(f"{'='*70}")
    
    results = {}
    for seed in seeds:
        path = os.path.join(RESULT_DIR, f'gat_ablation_seed{seed}.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                results[seed] = json.load(f)
    
    if not results:
        print("没有找到结果")
        return
    
    print(f"\n{'Seed':<8} {'MV Acc':>8} {'MV分歧':>8} {'GAT深度分歧':>12} {'重叠':>8} {'MV独有':>8} {'GAT独有':>8} {'CR Acc':>8} {'GAT-CR':>8}")
    print('-' * 90)
    
    mv_accs = []
    cr_accs = []
    gat_cr_accs = []
    mv_disagreements = []
    gat_disagreements = []
    overlaps = []
    mv_onlys = []
    gat_onlys = []
    
    for seed in seeds:
        if seed in results:
            r = results[seed]
            mv_acc = r['mv']['accuracy']
            cr_acc = r['cr']['full_accuracy'] or 0
            gat_cr = r.get('gat_cr_simulated') or 0
            mv_d = r['mv']['n_disagreement']
            gat_d = r['gat']['n_deep_disagreement']
            ov = r['gat']['n_overlap_mv']
            mo = r['gat']['n_mv_only']
            go = r['gat']['n_gat_only']
            
            mv_accs.append(mv_acc)
            cr_accs.append(cr_acc)
            gat_cr_accs.append(gat_cr)
            mv_disagreements.append(mv_d)
            gat_disagreements.append(gat_d)
            overlaps.append(ov)
            mv_onlys.append(mo)
            gat_onlys.append(go)
            
            print(f"{seed:<8} {mv_acc:>8.2f} {mv_d:>8} {gat_d:>12} {ov:>8} {mo:>8} {go:>8} {cr_acc:>8.2f} {gat_cr:>8.2f}")
    
    print('-' * 90)
    print(f"{'均值':<8} {np.mean(mv_accs):>8.2f} {np.mean(mv_disagreements):>8.0f} {np.mean(gat_disagreements):>12.0f} {np.mean(overlaps):>8.0f} {np.mean(mv_onlys):>8.0f} {np.mean(gat_onlys):>8.0f} {np.mean(cr_accs):>8.2f} {np.mean(gat_cr_accs):>8.2f}")
    
    print(f"\n[关键结论]")
    print(f"  MV分歧样本平均: {np.mean(mv_disagreements):.0f}")
    print(f"  GAT深度分歧平均: {np.mean(gat_disagreements):.0f}")
    print(f"  GAT可解决的分歧: {np.mean(mv_disagreements)-np.mean(gat_disagreements):.0f} ({100*(1-np.mean(gat_disagreements)/np.mean(mv_disagreements)):.1f}%)")
    print(f"  CR准确率: {np.mean(cr_accs):.2f}%")
    print(f"  GAT-CR模拟准确率: {np.mean(gat_cr_accs):.2f}%")
    print(f"  GAT-CR vs CR: {np.mean(gat_cr_accs)-np.mean(cr_accs):+.2f}%")
    
    # 保存汇总
    summary = {
        'seeds': seeds,
        'mv_acc': {'mean': float(np.mean(mv_accs)), 'std': float(np.std(mv_accs))},
        'cr_acc': {'mean': float(np.mean(cr_accs)), 'std': float(np.std(cr_accs))},
        'gat_cr_acc': {'mean': float(np.mean(gat_cr_accs)), 'std': float(np.std(gat_cr_accs))},
        'mv_disagreement': float(np.mean(mv_disagreements)),
        'gat_disagreement': float(np.mean(gat_disagreements)),
        'gat_resolved': float(np.mean(mv_disagreements) - np.mean(gat_disagreements)),
        'overlap': float(np.mean(overlaps)),
    }
    with open(os.path.join(RESULT_DIR, 'gat_ablation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存")


def main():
    parser = argparse.ArgumentParser(description='GAT共识层消融实验')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1024',
                        help='种子列表')
    parser.add_argument('--max_val', type=int, default=500, help='验证样本数')
    parser.add_argument('--summarize_only', action='store_true', help='仅汇总')
    parser.add_argument('--skip_seeds', type=str, default='', help='跳过的种子')
    
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    skip_seeds = [int(s.strip()) for s in args.skip_seeds.split(',')] if args.skip_seeds else []
    seeds = [s for s in seeds if s not in skip_seeds]
    
    if args.summarize_only:
        summarize_all(seeds)
        return
    
    for seed in seeds:
        run_ablation(seed, args.max_val)
    
    summarize_all(seeds)


if __name__ == '__main__':
    main()