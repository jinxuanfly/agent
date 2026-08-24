# -*- coding: utf-8 -*-
"""
任务1+2: Corr_Aware_DS折扣强度调优 + 能力自适应权重

基于500样本缓存数据，无需重新调用API:
1. 测试discount_strength = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
2. 实现基于准确率的能力自适应权重（Adaptive_Weight_DS）
3. 实现能力+相关性双重感知的融合（Adaptive_Corr_DS）
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 导入实验代码中的函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.step4_hateful_memes.evaluate_with_llm import (
    ds_fusion_decision,
    correlation_aware_ds_fusion,
    compute_agent_correlation,
)

CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints', 'hateful_memes')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'hateful_memes')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """加载训练集和验证集的LLM推理结果"""
    print("[1] 加载缓存数据...")

    # 从details文件加载验证集真实标签
    details_file = os.path.join(RESULTS_DIR, 'details_llm_gpt5_gemini_gpt5.json')
    with open(details_file, 'r', encoding='utf-8') as f:
        details = json.load(f)
    val_labels = torch.tensor(details['y_true'], dtype=torch.long)

    # 从evaluation文件加载训练集真实标签（与原实验相同参数：stratified=True默认）
    from src.step4_hateful_memes.evaluate_with_llm import HatefulMemesDataset
    train_dataset = HatefulMemesDataset(split='train', max_samples=200, load_images=False)
    train_labels = torch.tensor(train_dataset.labels, dtype=torch.long)
    # 校验：训练集样本数应与缓存beliefs一致
    train_ckpt0 = torch.load(os.path.join(CKPT_DIR, 'llm_train_agent0.pt'),
                              map_location='cpu', weights_only=False)
    assert len(train_labels) == train_ckpt0['beliefs'].shape[0], \
        f"训练集labels数({len(train_labels)})与缓存({train_ckpt0['beliefs'].shape[0]})不匹配"
    print(f"  [校验通过] 训练集={len(train_labels)}, 验证集={len(val_labels)}")

    data = {}
    for split in ['train', 'val']:
        beliefs = []
        uncertainties = []
        for i in range(3):
            ckpt = torch.load(os.path.join(CKPT_DIR, f'llm_{split}_agent{i}.pt'),
                              map_location='cpu', weights_only=False)
            beliefs.append(ckpt['beliefs'])
            uncertainties.append(ckpt['uncertainties'])
        labels = train_labels if split == 'train' else val_labels
        data[split] = {
            'beliefs': beliefs,
            'uncertainties': uncertainties,
            'labels': labels,
        }
        print(f"  {split}: {len(labels)} 样本, beliefs形状={beliefs[0].shape}")
    return data


def compute_agent_acc(beliefs, labels):
    """计算各Agent准确率"""
    accs = []
    for i, b in enumerate(beliefs):
        preds = b.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        accs.append(acc)
    return accs


def adaptive_weighted_ds_fusion(all_beliefs, all_uncertainties, u_threshold=0.5,
                                  agent_accuracies=None, temperature=2.0,
                                  correlation_matrix=None, discount_strength=0.2):
    """能力自适应+相关性感知的DS融合

    创新点：
    1. 基于准确率的能力权重（softmax归一化，temperature控制差距）
    2. 相关性折扣（避免同源Agent过度贡献）
    3. 能力放大（强Agent获得额外权重）

    Args:
        agent_accuracies: 训练集上计算的各Agent准确率list
        temperature: softmax温度，越大差距越大（强Agent权重越高）
        correlation_matrix: 相关性矩阵
        discount_strength: 相关性折扣强度
    """
    B = all_beliefs[0].shape[0]
    C = all_beliefs[0].shape[1]
    N = len(all_beliefs)
    device = all_beliefs[0].device

    if agent_accuracies is None:
        agent_weights = torch.ones(N, device=device) / N
    else:
        # 基于准确率计算能力权重（softmax with temperature）
        accs_tensor = torch.tensor(agent_accuracies, device=device, dtype=torch.float32)
        # 温度越低，强Agent权重越大（这里用1/temperature放大差距）
        scaled_accs = accs_tensor / max(temperature, 0.01)
        # 数值稳定softmax
        scaled_accs = scaled_accs - scaled_accs.max()
        exp_accs = torch.exp(scaled_accs * 10)  # 放大10倍让差距更明显
        agent_weights = exp_accs / (exp_accs.sum() + 1e-8)

    # 相关性折扣
    if correlation_matrix is not None and discount_strength > 0:
        discounts = torch.ones(N, device=device)
        for k in range(1, N):
            max_corr = correlation_matrix[k, :k].max()
            discounts[k] = 1.0 - discount_strength * max_corr.item()
            discounts[k] = max(0.1, discounts[k])
        agent_weights = agent_weights * discounts
        agent_weights = agent_weights / (agent_weights.sum() + 1e-8)

    # 执行DS融合
    b0 = all_beliefs[0]
    u0 = all_uncertainties[0]
    w0 = agent_weights[0]

    combined_belief = b0 * (1.0 - u0.unsqueeze(-1)) * w0
    combined_u = u0 * w0 + (1 - w0) * 0.5

    for b_idx in range(1, N):
        b = all_beliefs[b_idx]
        u = all_uncertainties[b_idx]
        w = agent_weights[b_idx]

        m1_b = combined_belief
        m1_u = combined_u
        m2_b = b * (1.0 - u.unsqueeze(-1)) * w
        m2_u = u * w + (1 - w) * 0.5

        sum_m1_b = m1_b.sum(dim=-1)
        sum_m2_b = m2_b.sum(dim=-1)
        agree = (m1_b * m2_b).sum(dim=-1)
        K = sum_m1_b * sum_m2_b - agree

        denom = 1.0 - K + 1e-8
        new_belief = (m1_b * m2_b + m1_b * m2_u.unsqueeze(-1) + m1_u.unsqueeze(-1) * m2_b) / denom.unsqueeze(-1)
        new_u = m1_u * m2_u / denom
        combined_belief = new_belief
        combined_u = new_u

    global_belief = combined_belief / (1.0 - combined_u.unsqueeze(-1) + 1e-8)
    global_u = combined_u
    preds = global_belief.argmax(dim=-1)
    rejected = global_u > u_threshold
    return preds, rejected, global_u, agent_weights


def evaluate_method(preds, labels, method_name):
    """评估单个方法"""
    acc = accuracy_score(labels.numpy(), preds.numpy()) * 100
    f1 = f1_score(labels.numpy(), preds.numpy(), average='macro') * 100
    return {'method': method_name, 'acc': acc, 'f1': f1}


def uncertainty_weighted_ds_fusion(all_beliefs, all_uncertainties, u_threshold=0.5,
                                     correlation_matrix=None, discount_strength=0.2,
                                     sharpness=5.0):
    """基于不确定性的自适应权重DS融合

    创新点：用Agent的不确定性u作为能力指标，而非训练集准确率
    - u越低 → Agent越自信 → 能力越强 → 权重越高
    - 避免训练集准确率过拟合问题

    Args:
        sharpness: 权重差距放大系数，越大强Agent权重越高
    """
    B = all_beliefs[0].shape[0]
    C = all_beliefs[0].shape[1]
    N = len(all_beliefs)
    device = all_beliefs[0].device

    # 每个Agent的平均不确定性（标量，但计算每个样本的权重更精细）
    # 这里使用每个样本的u作为该样本下Agent的权重
    # weight_i = softmax((1 - u_i) * sharpness) 对每个样本独立
    u_stack = torch.stack(all_uncertainties, dim=1)  # [B, N]
    confidence = 1.0 - u_stack  # [B, N]
    # 对每个样本，用softmax计算权重
    scaled_conf = confidence * sharpness
    scaled_conf = scaled_conf - scaled_conf.max(dim=1, keepdim=True)[0]
    exp_conf = torch.exp(scaled_conf)
    sample_weights = exp_conf / (exp_conf.sum(dim=1, keepdim=True) + 1e-8)  # [B, N]

    # 相关性折扣（可选，对所有Agent统一折扣）
    if correlation_matrix is not None and discount_strength > 0:
        # 对每个样本，折扣其与最相关Agent的证据
        # 简化：用全局折扣因子（基于平均相关性）
        discounts = torch.ones(N, device=device)
        for k in range(1, N):
            max_corr = correlation_matrix[k, :k].max()
            discounts[k] = 1.0 - discount_strength * max_corr.item()
            discounts[k] = max(0.1, discounts[k])
        sample_weights = sample_weights * discounts.unsqueeze(0)
        sample_weights = sample_weights / (sample_weights.sum(dim=1, keepdim=True) + 1e-8)

    # 执行DS融合（每个样本使用对应的权重）
    b0 = all_beliefs[0]
    u0 = all_uncertainties[0]
    w0 = sample_weights[:, 0]  # [B]

    combined_belief = b0 * (1.0 - u0.unsqueeze(-1)) * w0.unsqueeze(-1)
    combined_u = u0 * w0 + (1 - w0) * 0.5

    for b_idx in range(1, N):
        b = all_beliefs[b_idx]
        u = all_uncertainties[b_idx]
        w = sample_weights[:, b_idx]  # [B]

        m1_b = combined_belief
        m1_u = combined_u
        m2_b = b * (1.0 - u.unsqueeze(-1)) * w.unsqueeze(-1)
        m2_u = u * w + (1 - w) * 0.5

        sum_m1_b = m1_b.sum(dim=-1)
        sum_m2_b = m2_b.sum(dim=-1)
        agree = (m1_b * m2_b).sum(dim=-1)
        K = sum_m1_b * sum_m2_b - agree

        denom = 1.0 - K + 1e-8
        new_belief = (m1_b * m2_b + m1_b * m2_u.unsqueeze(-1) + m1_u.unsqueeze(-1) * m2_b) / denom.unsqueeze(-1)
        new_u = m1_u * m2_u / denom
        combined_belief = new_belief
        combined_u = new_u

    global_belief = combined_belief / (1.0 - combined_u.unsqueeze(-1) + 1e-8)
    global_u = combined_u
    preds = global_belief.argmax(dim=-1)
    rejected = global_u > u_threshold

    # 返回平均权重用于报告
    avg_weights = sample_weights.mean(dim=0)
    return preds, rejected, global_u, avg_weights


def main():
    print("=" * 80)
    print("任务1+2: Corr_Aware_DS折扣强度调优 + 能力自适应权重")
    print("=" * 80)

    data = load_data()
    train_b = data['train']['beliefs']
    train_u = data['train']['uncertainties']
    train_labels = data['train']['labels']

    val_b = data['val']['beliefs']
    val_u = data['val']['uncertainties']
    val_labels = data['val']['labels']

    # 计算训练集Agent准确率和相关性
    print("\n[2] 计算Agent准确率和相关性...")
    train_accs = compute_agent_acc(train_b, train_labels)
    val_accs = compute_agent_acc(val_b, val_labels)
    corr_matrix = compute_agent_correlation(train_b, train_labels)

    print("  Agent准确率:")
    for i in range(3):
        print(f"    Agent{i}: train={train_accs[i]*100:.2f}%, val={val_accs[i]*100:.2f}%")
    print(f"  相关性矩阵:")
    for i in range(3):
        row = "    "
        for j in range(3):
            row += f"{corr_matrix[i,j].item():.3f} "
        print(row)

    # ===== 任务1: Corr_Aware_DS折扣强度调优 =====
    print("\n" + "=" * 80)
    print("[任务1] Corr_Aware_DS 折扣强度调优")
    print("=" * 80)
    print(f"{'discount_strength':>20} | {'Acc%':>7} | {'F1%':>7}")
    print("-" * 45)

    corr_results = []
    for ds in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        preds, rejected, global_u = correlation_aware_ds_fusion(
            val_b, val_u, u_threshold=0.5,
            correlation_matrix=corr_matrix,
            discount_strength=ds,
        )
        result = evaluate_method(preds, val_labels, f"Corr_DS(ds={ds})")
        corr_results.append(result)
        print(f"{ds:>20.1f} | {result['acc']:>7.2f} | {result['f1']:>7.2f}")

    # ===== 任务2: 能力自适应权重 =====
    print("\n" + "=" * 80)
    print("[任务2] 能力自适应权重 (Adaptive_Weight_DS)")
    print("=" * 80)

    # 2a. 仅能力权重（无相关性折扣）
    print("\n[2a] 仅能力权重，不同temperature:")
    print(f"{'temperature':>15} | {'Acc%':>7} | {'F1%':>7} | {'权重分配':>30}")
    print("-" * 70)

    adaptive_results_no_corr = []
    for temp in [0.5, 1.0, 2.0, 5.0, 10.0]:
        preds, rejected, global_u, weights = adaptive_weighted_ds_fusion(
            val_b, val_u, u_threshold=0.5,
            agent_accuracies=train_accs,
            temperature=temp,
            correlation_matrix=None,
            discount_strength=0.0,
        )
        result = evaluate_method(preds, val_labels, f"Adaptive(t={temp})")
        adaptive_results_no_corr.append(result)
        w_str = f"[{weights[0].item():.3f}, {weights[1].item():.3f}, {weights[2].item():.3f}]"
        print(f"{temp:>15.1f} | {result['acc']:>7.2f} | {result['f1']:>7.2f} | {w_str:>30}")

    # 2b. 能力权重 + 相关性折扣（最佳组合）
    print("\n[2b] 能力权重 + 相关性折扣组合:")
    print(f"{'temp':>6} | {'ds':>5} | {'Acc%':>7} | {'F1%':>7} | {'权重分配':>30}")
    print("-" * 70)

    combined_results = []
    best_acc = 0
    best_config = None
    for temp in [1.0, 2.0, 5.0]:
        for ds in [0.1, 0.2, 0.3]:
            preds, rejected, global_u, weights = adaptive_weighted_ds_fusion(
                val_b, val_u, u_threshold=0.5,
                agent_accuracies=train_accs,
                temperature=temp,
                correlation_matrix=corr_matrix,
                discount_strength=ds,
            )
            result = evaluate_method(preds, val_labels, f"Adaptive_Corr(t={temp},ds={ds})")
            combined_results.append(result)
            w_str = f"[{weights[0].item():.3f}, {weights[1].item():.3f}, {weights[2].item():.3f}]"
            print(f"{temp:>6.1f} | {ds:>5.1f} | {result['acc']:>7.2f} | {result['f1']:>7.2f} | {w_str:>30}")
            if result['acc'] > best_acc:
                best_acc = result['acc']
                best_config = (temp, ds, result, weights)

    # ===== 任务3: 基于不确定性的自适应权重（核心创新） =====
    print("\n" + "=" * 80)
    print("[任务3] 基于不确定性的自适应权重 (Uncertainty_Weighted_DS)")
    print("  原理: u越低→Agent越自信→能力越强→权重越高（避免训练集准确率过拟合）")
    print("=" * 80)

    # 3a. 仅不确定性权重
    print("\n[3a] 仅不确定性权重，不同sharpness:")
    print(f"{'sharpness':>12} | {'Acc%':>7} | {'F1%':>7} | {'平均权重':>30}")
    print("-" * 70)

    unc_results = []
    best_unc_acc = 0
    best_unc_config = None
    for sharp in [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
        preds, rejected, global_u, weights = uncertainty_weighted_ds_fusion(
            val_b, val_u, u_threshold=0.5,
            correlation_matrix=None,
            discount_strength=0.0,
            sharpness=sharp,
        )
        result = evaluate_method(preds, val_labels, f"UncWeight(s={sharp})")
        unc_results.append(result)
        w_str = f"[{weights[0].item():.3f}, {weights[1].item():.3f}, {weights[2].item():.3f}]"
        print(f"{sharp:>12.1f} | {result['acc']:>7.2f} | {result['f1']:>7.2f} | {w_str:>30}")
        if result['acc'] > best_unc_acc:
            best_unc_acc = result['acc']
            best_unc_config = (sharp, 0.0, result, weights)

    # 3b. 不确定性权重 + 相关性折扣
    print("\n[3b] 不确定性权重 + 相关性折扣:")
    print(f"{'sharp':>8} | {'ds':>5} | {'Acc%':>7} | {'F1%':>7} | {'平均权重':>30}")
    print("-" * 70)

    unc_corr_results = []
    for sharp in [5.0, 10.0, 20.0]:
        for ds in [0.1, 0.2, 0.3]:
            preds, rejected, global_u, weights = uncertainty_weighted_ds_fusion(
                val_b, val_u, u_threshold=0.5,
                correlation_matrix=corr_matrix,
                discount_strength=ds,
                sharpness=sharp,
            )
            result = evaluate_method(preds, val_labels, f"UncWeight_Corr(s={sharp},ds={ds})")
            unc_corr_results.append(result)
            w_str = f"[{weights[0].item():.3f}, {weights[1].item():.3f}, {weights[2].item():.3f}]"
            print(f"{sharp:>8.1f} | {ds:>5.1f} | {result['acc']:>7.2f} | {result['f1']:>7.2f} | {w_str:>30}")
            if result['acc'] > best_unc_acc:
                best_unc_acc = result['acc']
                best_unc_config = (sharp, ds, result, weights)

    # ===== 基线对比 =====
    print("\n" + "=" * 80)
    print("[基线对比]")
    print("=" * 80)

    # 标准DS
    preds, _, _ = ds_fusion_decision(val_b, val_u)
    ds_result = evaluate_method(preds, val_labels, "DS_Fusion(标准)")

    # 原始GAT_EvidenceSwap（从结果文件读取）
    result_file = os.path.join(RESULTS_DIR, 'evaluation_llm_gpt5_gemini_gpt5.json')
    gat_ev_swap_acc = 70.40
    gat_ev_swap_f1 = 65.58
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        for m in results.get('methods', []):
            if 'GAT_EvidenceSwap' in m.get('method', ''):
                gat_ev_swap_acc = m.get('acc', 70.40)
                gat_ev_swap_f1 = m.get('f1', 65.58)
                break

    print(f"  DS_Fusion(标准):          Acc={ds_result['acc']:.2f}%, F1={ds_result['f1']:.2f}%")
    print(f"  GAT_EvidenceSwap(原实验):  Acc={gat_ev_swap_acc:.2f}%, F1={gat_ev_swap_f1:.2f}%")
    print(f"  BestAgent(Gemini):        Acc=79.60%, F1=80.75%")
    print(f"  最佳Adaptive配置 (t={best_config[0]}, ds={best_config[1]}):")
    print(f"    Acc={best_config[2]['acc']:.2f}%, F1={best_config[2]['f1']:.2f}%, 权重={best_config[3].tolist()}")
    print(f"  最佳Uncertainty配置 (s={best_unc_config[0]}, ds={best_unc_config[1]}):")
    print(f"    Acc={best_unc_config[2]['acc']:.2f}%, F1={best_unc_config[2]['f1']:.2f}%, 权重={best_unc_config[3].tolist()}")

    # ===== 总结 =====
    print("\n" + "=" * 80)
    print("[总结] 所有方法排名")
    print("=" * 80)
    all_results = corr_results + adaptive_results_no_corr + combined_results + unc_results + unc_corr_results
    all_results.append({'method': 'DS_Fusion(标准)', 'acc': ds_result['acc'], 'f1': ds_result['f1']})
    all_results.append({'method': 'GAT_EvidenceSwap(原)', 'acc': gat_ev_swap_acc, 'f1': gat_ev_swap_f1})
    all_results.append({'method': 'BestAgent(Gemini)', 'acc': 79.60, 'f1': 80.75})
    all_results.sort(key=lambda x: x['acc'], reverse=True)
    print(f"{'排名':>4} | {'方法':>35} | {'Acc%':>7} | {'F1%':>7}")
    print("-" * 65)
    for rank, r in enumerate(all_results, 1):
        marker = ""
        if r['method'] == f"Adaptive_Corr(t={best_config[0]},ds={best_config[1]})":
            marker = " **(best adaptive)**"
        elif r['method'] == f"UncWeight_Corr(s={best_unc_config[0]},ds={best_unc_config[1]})" or \
             r['method'] == f"UncWeight(s={best_unc_config[0]})":
            marker = " **(best uncertainty)**"
        print(f"{rank:>4} | {r['method']:>40} | {r['acc']:>7.2f} | {r['f1']:>7.2f}{marker}")

    # 保存结果
    output = {
        'agent_accuracies': {'train': train_accs, 'val': val_accs},
        'correlation_matrix': corr_matrix.tolist(),
        'corr_ds_tuning': corr_results,
        'adaptive_weight_only': adaptive_results_no_corr,
        'adaptive_weight_corr': combined_results,
        'uncertainty_weight_only': unc_results,
        'uncertainty_weight_corr': unc_corr_results,
        'baselines': {
            'DS_Fusion': ds_result,
            'GAT_EvidenceSwap': {'acc': gat_ev_swap_acc, 'f1': gat_ev_swap_f1},
            'BestAgent(Gemini)': {'acc': 79.60, 'f1': 80.75},
        },
        'best_adaptive_config': {
            'method': best_config[2]['method'],
            'temperature': best_config[0],
            'discount_strength': best_config[1],
            'acc': best_config[2]['acc'],
            'f1': best_config[2]['f1'],
            'weights': best_config[3].tolist(),
        },
        'best_uncertainty_config': {
            'method': best_unc_config[2]['method'],
            'sharpness': best_unc_config[0],
            'discount_strength': best_unc_config[1],
            'acc': best_unc_config[2]['acc'],
            'f1': best_unc_config[2]['f1'],
            'weights': best_unc_config[3].tolist(),
        }
    }
    out_file = os.path.join(RESULTS_DIR, 'corr_ds_tuning.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存至: {out_file}")


if __name__ == '__main__':
    main()
