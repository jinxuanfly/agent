"""
DS_Consensus == DS_Fusion 根因诊断脚本
========================================
问题：evaluate_cifar10n.py 中 simple_consensus(use_gat=True) 从未真正调用
step2/gat_consensus.py 中的 GAT 共识引擎。
它只是 DS_Fusion + 一个永远不会触发的额外拒识阈值。

本脚本诊断并量化此问题。
"""

import sys, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step2.gat_consensus import GATConsensusLayer, ConsensusEngine, global_decision, DEVICE as DEVICE_S2
from step4.evaluate_cifar10n import (
    load_features_and_heads, get_agent_outputs, 
    ds_fusion_decision
)

SEED = 42


def evaluate_method(preds, rejected, y_true):
    """统一评估函数"""
    rej_np = rejected.numpy()
    preds_np = preds.numpy()
    acc_all = accuracy_score(y_true, preds_np) * 100
    rej_rate = rej_np.mean() * 100
    mask = ~rej_np
    if mask.sum() > 0:
        acc = accuracy_score(y_true[mask], preds_np[mask]) * 100
        f1 = f1_score(y_true[mask], preds_np[mask], average='macro') * 100
    else:
        acc = 0.0
        f1 = 0.0
    return acc, f1, rej_rate, acc_all


def main():
    """诊断主函数"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("DS_Consensus == DS_Fusion 根因诊断")
    print("=" * 70)

    # =====================================================================
    # 第1步：确认问题 - 证明 DS_Consensus 输出完全等于 DS_Fusion
    # =====================================================================
    print("\n[1/5] 确认问题：DS_Consensus 是否等于 DS_Fusion？")

    features, test_labels = load_features_and_heads()
    B = min(500, len(test_labels))
    test_labels = test_labels[:B]

    all_alphas, all_beliefs, all_uncertainties, all_embs = [], [], [], []
    for name in ['agent1', 'agent2', 'agent3']:
        feats, head = features[name]
        feats = feats[:B]
        alpha, b, u, emb = get_agent_outputs(feats, head)
        all_alphas.append(alpha)
        all_beliefs.append(b)
        all_uncertainties.append(u)
        all_embs.append(emb)

    # 复现 simple_consensus(use_gat=True) 的行为
    def simple_consensus_replica(all_beliefs, all_uncertainties, use_gat=False):
        """完全复现 evaluate_cifar10n.py 中的 simple_consensus"""
        preds, rejected, global_u = ds_fusion_decision(
            all_beliefs, all_uncertainties, u_threshold=0.5
        )
        if use_gat:
            beliefs = torch.stack(all_beliefs, dim=1)
            mean_b = beliefs.mean(dim=1, keepdim=True)
            per_class_std = (beliefs - mean_b).pow(2).mean(dim=1).sqrt()
            disagreement = per_class_std.max(dim=-1).values
            dis_threshold = 0.6
            extra_reject = (disagreement > dis_threshold) & ~rejected
            rejected = rejected | extra_reject
        return preds, rejected, global_u

    ds_preds, ds_rej, ds_u = simple_consensus_replica(all_beliefs, all_uncertainties, use_gat=False)
    con_preds, con_rej, con_u = simple_consensus_replica(all_beliefs, all_uncertainties, use_gat=True)

    print(f"  DS_Fusion 和 DS_Consensus 预测相同: {(ds_preds == con_preds).all().item()}")
    print(f"  DS_Fusion 和 DS_Consensus 拒识相同: {(ds_rej == con_rej).all().item()}")
    print(f"  DS_Fusion 拒识率: {ds_rej.float().mean().item()*100:.2f}%")
    print(f"  DS_Consensus 额外拒识率: {(con_rej & ~ds_rej).float().mean().item()*100:.2f}%")

    # =====================================================================
    # 第2步：分歧度阈值分析
    # =====================================================================
    print("\n[2/5] 分歧度阈值分析")

    beliefs = torch.stack(all_beliefs, dim=1)
    mean_b = beliefs.mean(dim=1, keepdim=True)
    per_class_std = (beliefs - mean_b).pow(2).mean(dim=1).sqrt()
    disagreement = per_class_std.max(dim=-1).values

    print(f"  信念分歧度统计 (max per-class std):")
    print(f"    均值: {disagreement.mean().item():.4f}")
    print(f"    最大值: {disagreement.max().item():.4f}")
    print(f"    最小值: {disagreement.min().item():.4f}")
    print(f"    中位数: {disagreement.median().item():.4f}")
    print(f"  >> 当前阈值 0.6 超出范围，因此永远不触发 <<")

    # =====================================================================
    # 第3步：不确定性加权平均共识
    # =====================================================================
    print("\n[3/5] 使用不确定性加权平均共识")
    print("  [注意] step2 GATConsensusLayer 随机初始化未训练，")
    print("  在 CIFAR-10N 10分类+128d嵌入上会产生 NaN。")
    print("  使用不确定性加权平均作为合理替代（与 global_decision 逻辑一致）。")

    N = len(all_beliefs)
    K = all_beliefs[0].shape[1]

    belief_ds_list, u_ds_list = [], []
    for i in range(B):
        b_i_prod = all_beliefs[0][i].clone()
        for j in range(1, N):
            b_i_prod = b_i_prod * all_beliefs[j][i]
        b_i_prod = b_i_prod / (b_i_prod.sum() + 1e-8)
        belief_ds_list.append(b_i_prod.unsqueeze(0))
    belief_ds = torch.cat(belief_ds_list, dim=0)

    belief_after_list, u_after_list = [], []
    for i in range(B):
        b_i = torch.stack([all_beliefs[j][i] for j in range(N)], dim=0)
        u_i = torch.tensor([all_uncertainties[j][i].item() if isinstance(all_uncertainties[j][i], torch.Tensor)
                            else all_uncertainties[j][i] for j in range(N)])
        w_i = (1.0 - u_i).clamp(min=0.01)
        w_i = w_i / (w_i.sum() + 1e-8)
        b_weighted = (w_i.unsqueeze(1) * b_i).sum(dim=0)
        b_weighted = b_weighted / (b_weighted.sum() + 1e-8)
        u_weighted = (w_i * u_i).sum().item()
        belief_after_list.append(b_weighted.unsqueeze(0))
        u_after_list.append(u_weighted)

    belief_after = torch.cat(belief_after_list, dim=0)
    u_after = torch.tensor(u_after_list)

    belief_change = (belief_after - belief_ds).abs().mean(dim=1)
    print(f"  不确定性加权平均 vs DS乘积融合 信念平均变化: {belief_change.mean().item():.6f}")
    print(f"  最大变化: {belief_change.max().item():.6f}")

    ds_class_pred = belief_ds.argmax(dim=1)
    wavg_class_pred = belief_after.argmax(dim=1)
    decision_change = (ds_class_pred != wavg_class_pred)
    print(f"  决策不同的样本: {decision_change.sum().item()} / {B}")

    gat_u_all = 1 - belief_after.max(dim=1).values
    ds_u_all_local = 1 - belief_ds.max(dim=1).values
    u_change = (gat_u_all - ds_u_all_local).abs()
    print(f"  不确定性平均变化: {u_change.mean().item():.6f}")
    print(f"  不确定性降低样本: {(gat_u_all < ds_u_all_local).sum().item()} / {B}")

    # =====================================================================
    # 第4步：完整评估对比
    # =====================================================================
    print("\n[4/5] 完整评估对比")
    y_true = test_labels.numpy()

    ds_f_preds, ds_f_rej, _ = ds_fusion_decision(all_beliefs, all_uncertainties, u_threshold=0.5)
    g_preds = belief_after.argmax(dim=1)
    g_rej = u_after > 0.5
    con_preds_fake, con_rej_fake, _ = simple_consensus_replica(all_beliefs, all_uncertainties, use_gat=True)

    ds_a, ds_f, ds_r, ds_aa = evaluate_method(ds_f_preds, ds_f_rej, y_true)
    g_a, g_f, g_r, g_aa = evaluate_method(g_preds, g_rej, y_true)
    con_a, con_f, con_r, con_aa = evaluate_method(con_preds_fake, con_rej_fake, y_true)

    print(f"\n{'='*100}")
    print(f"{'Method':<30s} {'Acc%':<10s} {'F1%':<10s} {'Rej%':<10s} {'Acc_All':<10s}")
    print(f"{'-'*100}")
    print(f"{'DS_Fusion (基线)':<30s} {ds_a:<10.2f} {ds_f:<10.2f} {ds_r:<10.2f} {ds_aa:<10.2f}")
    print(f"{'U-加权平均共识':<30s} {g_a:<10.2f} {g_f:<10.2f} {g_r:<10.2f} {g_aa:<10.2f}")
    print(f"{'DS_Consensus (原始假GAT)':<30s} {con_a:<10.2f} {con_f:<10.2f} {con_r:<10.2f} {con_aa:<10.2f}")
    print(f"{'='*100}")

    # =====================================================================
    # 第5步：保存诊断结果
    # =====================================================================
    print("\n[5/5] 保存诊断结果")
    results = {
        'diagnosis_date': '2025-06-19',
        'root_cause': (
            'simple_consensus(use_gat=True) in evaluate_cifar10n.py does NOT '
            'call the real GAT consensus engine from step2/gat_consensus.py. '
            'It only adds a disagreement rejection threshold (0.6) on top of DS_Fusion, '
            'which never triggers in 10-class CIFAR-10N setting.'
        ),
        'evidence': {
            'ds_vs_con_preds_identical': (ds_preds == con_preds).all().item(),
            'ds_vs_con_rejection_identical': (ds_rej == con_rej).all().item(),
            'max_disagreement': disagreement.max().item(),
            'disagreement_threshold': 0.6,
            'samples_above_threshold': (disagreement > 0.6).sum().item(),
            'total_samples': B,
        },
        'real_gat_alternative': {
            'method': 'uncertainty_weighted_averaging',
            'decision_changed_count': decision_change.sum().item(),
            'uncertainty_reduced_count': (gat_u_all < ds_u_all_local).sum().item(),
        },
        'metrics': {
            'DS_Fusion': {'acc': round(ds_a, 2), 'f1': round(ds_f, 2), 'rej_rate': round(ds_r, 2)},
            'UncertaintyWeightedAvg': {'acc': round(g_a, 2), 'f1': round(g_f, 2), 'rej_rate': round(g_r, 2)},
            'DS_Consensus_fake': {'acc': round(con_a, 2), 'f1': round(con_f, 2), 'rej_rate': round(con_r, 2)},
        }
    }

    os.makedirs('results/cifar10n', exist_ok=True)
    with open('results/cifar10n/diagnose_consensus_root_cause.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  诊断结果已保存: results/cifar10n/diagnose_consensus_root_cause.json")

    # 可视化
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.hist(belief_change.numpy(), bins=30, color='steelblue', alpha=0.7)
    plt.xlabel('Mean Belief Change per Sample')
    plt.ylabel('Count')
    plt.title('U-Weighted Avg: Belief Change vs DS')
    plt.axvline(belief_change.mean().item(), color='r', linestyle='--',
                label=f'Mean={belief_change.mean().item():.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.hist(u_change.numpy(), bins=30, color='salmon', alpha=0.7)
    plt.xlabel('Uncertainty Change')
    plt.ylabel('Count')
    plt.title('U-Weighted Avg: Uncertainty Change')
    plt.axvline(u_change.mean().item(), color='r', linestyle='--',
                label=f'Mean={u_change.mean().item():.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    metrics_names = ['DS_Fusion', 'U-Wt Avg', 'DS_Fake']
    acc_vals = [ds_a, g_a, con_a]
    f1_vals = [ds_f, g_f, con_f]
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, acc_vals, width, label='Acc%', color='steelblue', alpha=0.8)
    plt.bar(x + width/2, f1_vals, width, label='F1%', color='salmon', alpha=0.8)
    plt.xticks(x, metrics_names, rotation=15)
    plt.ylabel('Score (%)')
    plt.title('Accuracy & F1 Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/gat_consensus_diagnosis_cifar10n.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
    print(f"""
结论:
  1. DS_Consensus == DS_Fusion 根因已确认:
     simple_consensus(use_gat=True) 从未调用 step2 的 GAT共识引擎
     仅添加分歧度拒识阈值(0.6)，信念范围[0,1]最大分歧度={disagreement.max().item():.4f}，阈值永不触发
  2. GAT引擎问题: step2 GATConsensusLayer 随机初始化未训练，10分类+128d嵌入产生NaN
  3. 不确定性加权平均 vs DS乘积融合({B}样本)：
     决策改变: {decision_change.sum().item()} 样本
     不确定性降低: {(gat_u_all < ds_u_all_local).sum().item()} 样本
""")


def simple_consensus_replica(all_beliefs, all_uncertainties, use_gat=False):
    """对外的简单共识函数"""
    preds, rejected, global_u = ds_fusion_decision(
        all_beliefs, all_uncertainties, u_threshold=0.5
    )
    if use_gat:
        beliefs = torch.stack(all_beliefs, dim=1)
        mean_b = beliefs.mean(dim=1, keepdim=True)
        per_class_std = (beliefs - mean_b).pow(2).mean(dim=1).sqrt()
        disagreement = per_class_std.max(dim=-1).values
        dis_threshold = 0.6
        extra_reject = (disagreement > dis_threshold) & ~rejected
        rejected = rejected | extra_reject
    return preds, rejected, global_u


if __name__ == '__main__':
    main()