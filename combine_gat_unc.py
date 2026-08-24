# -*- coding: utf-8 -*-
"""
GAT + Uncertainty_Weighted_DS 组合实验
=====================================
将两个有效创新点组合：
1. GAT_EvidenceSwap: GAT共识 + 证据交换（最佳融合方法 70.4%）
2. Uncertainty_Weighted_DS: 基于不确定性的自适应权重DS融合（核心创新 69.2%）

组合方案：
- 方案A (GAT_EvidSwap_UncDS): GAT共识 + 证据交换 + 不确定性加权DS最终融合
- 方案B (GAT_UncDS): GAT共识 + 不确定性加权DS（无证据交换）
- 方案C (GAT_EvidSwap_GATWeights): GAT共识 + 证据交换 + GAT学习的权重DS融合
- 方案D (UncDS_EvidSwap): 不确定性加权DS + 证据交换（无GAT）

利用500样本缓存，无需重新调用API。
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.step4_hateful_memes.evaluate_with_llm import (
    ds_fusion_decision,
    uncertainty_weighted_ds_fusion,
    HatefulMemesDataset,
    GATConsensusLayer,
    ConsensusEngine,
    DisagreementDeconstructor,
    CHECKPOINT_DIR,
    RESULT_DIR,
    NUM_CLASSES,
    DEVICE,
    U_THRESHOLD,
)


# =============================================================
# 加载缓存数据
# =============================================================

def load_cached_data():
    """加载训练集和验证集的缓存LLM推理结果"""
    print("[1] 加载缓存数据...")

    train_dataset = HatefulMemesDataset(split='train', max_samples=200, load_images=False)
    train_labels = torch.tensor(train_dataset.labels, dtype=torch.long)

    details_file = os.path.join(RESULT_DIR, 'details_llm_gpt5_gemini_gpt5.json')
    with open(details_file, 'r', encoding='utf-8') as f:
        details = json.load(f)
    val_labels = torch.tensor(details['y_true'], dtype=torch.long)

    data = {}
    for split in ['train', 'val']:
        beliefs, uncertainties, alphas, embs = [], [], [], []
        for i in range(3):
            ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f'llm_{split}_agent{i}.pt'),
                              map_location='cpu', weights_only=False)
            beliefs.append(ckpt['beliefs'])
            uncertainties.append(ckpt['uncertainties'])
            alphas.append(ckpt['alphas'])
            embs.append(ckpt['embs'])
        labels = train_labels if split == 'train' else val_labels
        data[split] = {
            'beliefs': beliefs, 'uncertainties': uncertainties,
            'alphas': alphas, 'embs': embs, 'labels': labels,
        }
        print(f"  {split}: {len(labels)} 样本, embs形状={embs[0].shape}")
    return data


# =============================================================
# GAT训练（参考主代码和消融实验）
# =============================================================

def train_gat(train_beliefs, train_uncertainties, train_embs, train_labels, n_epochs=50):
    """训练GAT共识层"""
    print(f"\n[2] 训练GAT共识层...")

    B_train = train_beliefs[0].shape[0]
    embed_dim = 256
    gat_node_dim = embed_dim + NUM_CLASSES + 1

    gat_layer = GATConsensusLayer(
        node_dim=gat_node_dim, hidden_dim=64, embed_dim=embed_dim, num_classes=NUM_CLASSES
    ).to(DEVICE)

    # 分歧样本训练
    train_preds = torch.stack([b.argmax(dim=1) for b in train_beliefs], dim=1)
    disagreement_mask = ~((train_preds[:, 0] == train_preds[:, 1]) &
                          (train_preds[:, 1] == train_preds[:, 2]))
    train_gat_indices = torch.where(disagreement_mask)[0]
    print(f"  分歧训练样本: {len(train_gat_indices)}")

    if len(train_gat_indices) < 5:
        train_gat_indices = torch.arange(B_train)

    optimizer = torch.optim.Adam(gat_layer.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    gat_layer.train()
    best_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = train_gat_indices[torch.randperm(len(train_gat_indices))]

        batch_size = 16
        for b_start in range(0, len(perm), batch_size):
            b_end = min(b_start + batch_size, len(perm))
            batch_idx = perm[b_start:b_end]

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, requires_grad=True, device=DEVICE)

            for b_idx_cpu in batch_idx.cpu().numpy():
                agent_outputs = []
                for i in range(3):
                    b_i = train_beliefs[i][b_idx_cpu:b_idx_cpu+1]
                    u_i = train_uncertainties[i][b_idx_cpu:b_idx_cpu+1]
                    emb_i = train_embs[i][b_idx_cpu:b_idx_cpu+1].to(DEVICE)
                    u_val = float(u_i.squeeze(-1).item())
                    S = NUM_CLASSES / max(u_val, 1e-6)
                    alpha_i = b_i[0] * S + 1.0
                    agent_outputs.append((alpha_i, b_i[0], u_val, emb_i[0]))

                engine_tmp = ConsensusEngine(embed_dim=embed_dim, num_classes=NUM_CLASSES, hidden_dim=64)
                engine_tmp.layer = gat_layer
                try:
                    h = engine_tmp.build_state(agent_outputs)
                    fusion_weights = gat_layer.forward_sample_weights(h)

                    true_label = train_labels[b_idx_cpu].unsqueeze(0).to(DEVICE)
                    correct_mask = torch.zeros(3, dtype=torch.bool, device=DEVICE)
                    for i in range(3):
                        if agent_outputs[i][1].argmax() == true_label[0]:
                            correct_mask[i] = True

                    target_weights = torch.zeros(3, dtype=torch.float32, device=DEVICE)
                    if correct_mask.sum() > 0:
                        target_weights[correct_mask] = 1.0 / correct_mask.sum()
                    else:
                        target_weights = torch.ones(3, device=DEVICE) / 3

                    loss = F.mse_loss(fusion_weights, target_weights)
                    total_loss = total_loss + loss
                except Exception:
                    continue

            if total_loss.item() > 0 or total_loss.requires_grad:
                total_loss = total_loss / len(batch_idx)
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: loss={avg_loss:.4f}")

    print(f"  GAT训练完成, 最佳loss={best_loss:.4f}")
    return gat_layer


# =============================================================
# GAT共识推理（产出调整后的beliefs和uncertainties）
# =============================================================

def gat_consensus_inference(gat_layer, val_beliefs, val_uncertainties, val_embs):
    """运行GAT共识层，产出调整后的beliefs和uncertainties

    Returns:
        gat_beliefs: [3] list of [B, C] 调整后的信念
        gat_uncertainties: [3] list of [B] 调整后的不确定性
        gat_fusion_weights: [B, 3] GAT学习的融合权重
    """
    print(f"\n[3] 运行GAT共识层...")
    gat_layer.eval()
    B_val = val_beliefs[0].shape[0]
    embed_dim = 256

    final_belief_list = []
    final_u_list = []
    gat_fusion_weights_list = []

    with torch.no_grad():
        for b_idx in range(B_val):
            agent_outputs = []
            for i in range(3):
                b_i = val_beliefs[i][b_idx]
                u_i = val_uncertainties[i][b_idx]
                emb_i = val_embs[i][b_idx].to(DEVICE)
                u_val = float(u_i.item())
                S = NUM_CLASSES / max(u_val, 1e-6)
                alpha_i = b_i * S + 1.0
                agent_outputs.append((alpha_i, b_i, u_val, emb_i))

            engine_tmp = ConsensusEngine(embed_dim=embed_dim, num_classes=NUM_CLASSES, hidden_dim=64)
            engine_tmp.layer = gat_layer
            try:
                h = engine_tmp.build_state(agent_outputs)
                fusion_weights = gat_layer.forward_sample_weights(h, hard_gate=False)
                original_beliefs = h[:, engine_tmp.embed_dim:engine_tmp.embed_dim+NUM_CLASSES]

                fused_belief = fusion_weights @ original_beliefs
                fused_belief = fused_belief / fused_belief.sum().clamp(min=1e-6)

                fs = []
                for i in range(3):
                    adjusted_belief = 0.8 * original_beliefs[i] + 0.2 * fused_belief
                    adjusted_belief = adjusted_belief / adjusted_belief.sum().clamp(min=1e-6)
                    fs.append(adjusted_belief)

                us = [val_uncertainties[i][b_idx].item() * 0.9 + 0.1 for i in range(3)]

                final_belief_list.append(torch.stack(fs, dim=0))
                final_u_list.append(torch.tensor(us))
                gat_fusion_weights_list.append(fusion_weights.cpu())
            except Exception as e:
                # 失败时使用原始值
                fs = [val_beliefs[i][b_idx] for i in range(3)]
                us = [val_uncertainties[i][b_idx].item() for i in range(3)]
                final_belief_list.append(torch.stack(fs, dim=0))
                final_u_list.append(torch.tensor(us))
                gat_fusion_weights_list.append(torch.ones(3) / 3)

    final_belief = torch.stack(final_belief_list, dim=0)  # [B, 3, C]
    final_uncertainty = torch.stack(final_u_list, dim=0)    # [B, 3]
    gat_fusion_weights = torch.stack(gat_fusion_weights_list, dim=0)  # [B, 3]

    print(f"  GAT共识完成, 平均u: {final_uncertainty.mean().item():.4f}")

    gat_beliefs = [final_belief[:, i] for i in range(3)]
    gat_uncertainties = [final_uncertainty[:, i] for i in range(3)]

    return gat_beliefs, gat_uncertainties, gat_fusion_weights


# =============================================================
# 证据交换（分歧解构 + 最佳agent→最差agent证据传递）
# =============================================================

def evidence_exchange(val_beliefs, val_uncertainties, val_alphas):
    """执行证据交换

    Returns:
        corrected_beliefs: [3] list of [B, C]
        corrected_uncertainties: [3] list of [B]
        conflict_types: list of str
    """
    print(f"\n[4] 运行分歧解构 + 证据交换...")
    B_val = val_beliefs[0].shape[0]

    all_beliefs = torch.stack(val_beliefs, dim=1)  # [B, 3, C]
    all_uncertainties = torch.stack(val_uncertainties, dim=1)  # [B, 3]
    all_alphas = torch.stack(val_alphas, dim=1).to(DEVICE)  # [B, 3, C]

    deconstructor = DisagreementDeconstructor(u_threshold=0.5, K_threshold=0.3)
    conflict_types, K_values = deconstructor.deconstruct_batch(all_beliefs, all_uncertainties)

    evidence_count = sum(1 for c in conflict_types if c == 'evidence_conflict')
    print(f"  分歧分布: 证据冲突={evidence_count}/{B_val}")

    corrected_alphas = all_alphas.clone()
    for b_idx in range(B_val):
        if conflict_types[b_idx] == 'evidence_conflict':
            best_agent = all_uncertainties[b_idx].argmin().item()
            worst_agent = all_uncertainties[b_idx].argmax().item()
            if best_agent != worst_agent:
                best_alpha = all_alphas[b_idx, best_agent]
                sender_evidence = (best_alpha - 1.0) * 2.0
                sender_evidence = F.relu(sender_evidence)
                corrected_alphas[b_idx, worst_agent] += sender_evidence

    corrected_beliefs = corrected_alphas / corrected_alphas.sum(dim=-1, keepdim=True)
    corrected_uncertainties = NUM_CLASSES / corrected_alphas.sum(dim=-1)

    corr_b_list = [corrected_beliefs[:, i] for i in range(3)]
    corr_u_list = [corrected_uncertainties[:, i] for i in range(3)]

    return corr_b_list, corr_u_list, conflict_types


# =============================================================
# 组合方法实现
# =============================================================

def method_A_gat_evidswap_uncds(corr_b_list, corr_u_list, sharpness=20.0):
    """方案A: GAT共识 + 证据交换 + 不确定性加权DS最终融合"""
    preds, rejected, u = uncertainty_weighted_ds_fusion(
        corr_b_list, corr_u_list, u_threshold=U_THRESHOLD,
        correlation_matrix=None, discount_strength=0.0,
        sharpness=sharpness,
    )
    return preds.cpu(), rejected.cpu(), u.cpu()


def method_B_gat_uncds(gat_beliefs, gat_uncertainties, sharpness=20.0):
    """方案B: GAT共识 + 不确定性加权DS（无证据交换）"""
    preds, rejected, u = uncertainty_weighted_ds_fusion(
        gat_beliefs, gat_uncertainties, u_threshold=U_THRESHOLD,
        correlation_matrix=None, discount_strength=0.0,
        sharpness=sharpness,
    )
    return preds.cpu(), rejected.cpu(), u.cpu()


def method_C_gat_evidswap_gatweights(corr_b_list, corr_u_list, gat_fusion_weights):
    """方案C: GAT共识 + 证据交换 + GAT学习的权重DS融合"""
    B = corr_b_list[0].shape[0]
    # 将每个样本的GAT权重作为agent_weights传入ds_fusion_decision
    # ds_fusion_decision目前接受标量权重，需要修改为接受per-sample权重
    # 这里直接实现带per-sample权重的DS融合
    preds = []
    rejected = []
    global_us = []

    for b_idx in range(B):
        weights = gat_fusion_weights[b_idx].to(DEVICE)  # [3]
        b_list = [corr_b_list[i][b_idx:b_idx+1].to(DEVICE) for i in range(3)]
        u_list = [corr_u_list[i][b_idx:b_idx+1].to(DEVICE) for i in range(3)]

        # 加权DS融合
        b0 = b_list[0]
        u0 = u_list[0]
        w0 = weights[0]

        combined_belief = b0 * (1.0 - u0) * w0
        combined_u = u0 * w0 + (1 - w0) * 0.5

        for i in range(1, 3):
            b = b_list[i]
            u = u_list[i]
            w = weights[i]

            m1_b = combined_belief
            m1_u = combined_u
            m2_b = b * (1.0 - u) * w
            m2_u = u * w + (1 - w) * 0.5

            sum_m1_b = m1_b.sum(dim=-1)
            sum_m2_b = m2_b.sum(dim=-1)
            agree = (m1_b * m2_b).sum(dim=-1)
            K = sum_m1_b * sum_m2_b - agree

            denom = 1.0 - K + 1e-8
            new_belief = (m1_b * m2_b + m1_b * m2_u + m1_u * m2_b) / denom
            new_u = m1_u * m2_u / denom
            combined_belief = new_belief
            combined_u = new_u

        global_belief = combined_belief / (1.0 - combined_u + 1e-8)
        preds.append(global_belief.argmax(dim=-1).item())
        rejected.append(combined_u.item() > U_THRESHOLD)
        global_us.append(combined_u.item())

    return torch.tensor(preds), torch.tensor(rejected), torch.tensor(global_us)


def method_D_uncds_evidswap(val_beliefs, val_uncertainties, val_alphas, sharpness=20.0):
    """方案D: 不确定性加权DS + 证据交换（无GAT）

    先用原始uncertainty-weighted DS识别分歧，然后证据交换，再用uncertainty-weighted DS
    """
    # 第一步：识别证据冲突
    deconstructor = DisagreementDeconstructor(u_threshold=0.5, K_threshold=0.3)
    all_beliefs = torch.stack(val_beliefs, dim=1)
    all_uncertainties = torch.stack(val_uncertainties, dim=1)
    conflict_types, _ = deconstructor.deconstruct_batch(all_beliefs, all_uncertainties)

    # 第二步：证据交换
    all_alphas = torch.stack(val_alphas, dim=1).to(DEVICE)
    B_val = all_beliefs.shape[0]
    corrected_alphas = all_alphas.clone()

    for b_idx in range(B_val):
        if conflict_types[b_idx] == 'evidence_conflict':
            best_agent = all_uncertainties[b_idx].argmin().item()
            worst_agent = all_uncertainties[b_idx].argmax().item()
            if best_agent != worst_agent:
                best_alpha = all_alphas[b_idx, best_agent]
                sender_evidence = (best_alpha - 1.0) * 2.0
                sender_evidence = F.relu(sender_evidence)
                corrected_alphas[b_idx, worst_agent] += sender_evidence

    corrected_beliefs = corrected_alphas / corrected_alphas.sum(dim=-1, keepdim=True)
    corrected_uncertainties = NUM_CLASSES / corrected_alphas.sum(dim=-1)

    corr_b_list = [corrected_beliefs[:, i] for i in range(3)]
    corr_u_list = [corrected_uncertainties[:, i] for i in range(3)]

    # 第三步：不确定性加权DS
    preds, rejected, u = uncertainty_weighted_ds_fusion(
        corr_b_list, corr_u_list, u_threshold=U_THRESHOLD,
        correlation_matrix=None, discount_strength=0.0,
        sharpness=sharpness,
    )
    return preds.cpu(), rejected.cpu(), u.cpu()


# =============================================================
# 主函数
# =============================================================

def main():
    print("=" * 80)
    print("GAT + Uncertainty_Weighted_DS 组合实验")
    print("=" * 80)

    # 加载数据
    data = load_cached_data()
    train_data = data['train']
    val_data = data['val']

    val_labels = val_data['labels']
    y_true = val_labels.numpy()
    B_val = len(val_labels)

    # 训练GAT
    gat_layer = train_gat(
        train_data['beliefs'], train_data['uncertainties'],
        train_data['embs'], train_data['labels'],
        n_epochs=50,
    )

    # GAT共识推理
    gat_beliefs, gat_uncertainties, gat_fusion_weights = gat_consensus_inference(
        gat_layer, val_data['beliefs'], val_data['uncertainties'], val_data['embs']
    )

    # 证据交换（基于GAT调整后的beliefs）
    corr_b_list, corr_u_list, conflict_types = evidence_exchange(
        gat_beliefs, gat_uncertainties, val_data['alphas']
    )

    # 运行所有方法
    print(f"\n[5] 运行组合方法...")

    methods = {}

    # 基线方法（用于对比）
    print(f"\n  --- 基线方法 ---")

    # 1. DS等权重
    ds_preds, ds_rej, ds_u = ds_fusion_decision(
        val_data['beliefs'], val_data['uncertainties'], u_threshold=U_THRESHOLD
    )
    methods['DS_Fusion(等权重)'] = ds_preds.cpu()

    # 2. Uncertainty_Weighted_DS
    unc_preds, unc_rej, unc_u = uncertainty_weighted_ds_fusion(
        val_data['beliefs'], val_data['uncertainties'], u_threshold=U_THRESHOLD,
        sharpness=20.0,
    )
    methods['Uncertainty_Weighted_DS'] = unc_preds.cpu()

    # 3. GAT_EvidenceSwap（原方法）
    evidswap_preds, evidswap_rej, evidswap_u = ds_fusion_decision(
        corr_b_list, corr_u_list, u_threshold=U_THRESHOLD
    )
    methods['GAT_EvidenceSwap'] = evidswap_preds.cpu()

    # 组合方法
    print(f"\n  --- 组合方法 ---")

    # 方案A: GAT + 证据交换 + 不确定性加权DS
    a_preds, a_rej, a_u = method_A_gat_evidswap_uncds(corr_b_list, corr_u_list, sharpness=20.0)
    methods['A_GAT_EvidSwap_UncDS(s=20)'] = a_preds

    # 方案B: GAT + 不确定性加权DS（无证据交换）
    b_preds, b_rej, b_u = method_B_gat_uncds(gat_beliefs, gat_uncertainties, sharpness=20.0)
    methods['B_GAT_UncDS(s=20)'] = b_preds

    # 方案C: GAT + 证据交换 + GAT权重DS
    c_preds, c_rej, c_u = method_C_gat_evidswap_gatweights(corr_b_list, corr_u_list, gat_fusion_weights)
    methods['C_GAT_EvidSwap_GATWeights'] = c_preds

    # 方案D: 不确定性加权DS + 证据交换（无GAT）
    d_preds, d_rej, d_u = method_D_uncds_evidswap(
        val_data['beliefs'], val_data['uncertainties'], val_data['alphas'], sharpness=20.0
    )
    methods['D_UncDS_EvidSwap(s=20)'] = d_preds

    # 评估所有方法
    print(f"\n[6] 评估结果:")
    print(f"{'='*80}")
    print(f"{'方法':<35s} {'Acc%':<10s} {'F1%':<10s} {'+vs DS':<10s} {'+vs UncDS':<10s}")
    print(f"{'-'*80}")

    results = {}
    ds_acc = accuracy_score(y_true, methods['DS_Fusion(等权重)']) * 100
    unc_acc = accuracy_score(y_true, methods['Uncertainty_Weighted_DS']) * 100

    for name, preds in methods.items():
        acc = accuracy_score(y_true, preds) * 100
        f1 = f1_score(y_true, preds, average='binary') * 100
        delta_ds = acc - ds_acc
        delta_unc = acc - unc_acc
        print(f"{name:<35s} {acc:<10.2f} {f1:<10.2f} {delta_ds:+.2f}     {delta_unc:+.2f}")
        results[name] = {
            'accuracy': acc,
            'f1': f1,
            'delta_vs_ds': delta_ds,
            'delta_vs_uncds': delta_unc,
        }

    print(f"{'='*80}")

    # 分歧样本分析
    print(f"\n[7] 分歧样本分析:")
    disagreement_mask = torch.tensor([c != 'none' for c in conflict_types], dtype=torch.bool)
    evidence_conflict_mask = torch.tensor([c == 'evidence_conflict' for c in conflict_types], dtype=torch.bool)

    print(f"  总样本: {B_val}, 分歧样本: {disagreement_mask.sum().item()}, 证据冲突: {evidence_conflict_mask.sum().item()}")
    print(f"\n  {'方法':<35s} {'全Acc%':<10s} {'分歧Acc%':<12s} {'证据冲突Acc%':<14s}")
    print(f"  {'-'*70}")

    for name, preds in methods.items():
        all_acc = accuracy_score(y_true, preds) * 100
        if disagreement_mask.sum() > 0:
            disagree_acc = accuracy_score(y_true[disagreement_mask], preds[disagreement_mask]) * 100
        else:
            disagree_acc = 0.0
        if evidence_conflict_mask.sum() > 0:
            evidence_acc = accuracy_score(y_true[evidence_conflict_mask], preds[evidence_conflict_mask]) * 100
        else:
            evidence_acc = 0.0
        print(f"  {name:<35s} {all_acc:<10.2f} {disagree_acc:<12.2f} {evidence_acc:<14.2f}")

    # 不同sharpness测试方案A
    print(f"\n[8] 方案A (GAT+EvidSwap+UncDS) 不同sharpness测试:")
    print(f"  {'sharpness':<15s} {'Acc%':<10s} {'F1%':<10s}")
    print(f"  {'-'*35}")
    sharpness_results = {}
    for s in [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
        s_preds, _, _ = method_A_gat_evidswap_uncds(corr_b_list, corr_u_list, sharpness=s)
        s_acc = accuracy_score(y_true, s_preds) * 100
        s_f1 = f1_score(y_true, s_preds, average='binary') * 100
        print(f"  {s:<15.1f} {s_acc:<10.2f} {s_f1:<10.2f}")
        sharpness_results[f's={s}'] = {'acc': s_acc, 'f1': s_f1}

    # 保存结果
    output = {
        '_metadata': {
            'experiment': 'GAT + Uncertainty_Weighted_DS 组合实验',
            'sample_size': B_val,
            'date': '2026-08-24',
            'description': '将GAT_EvidenceSwap和Uncertainty_Weighted_DS两个有效创新点组合',
        },
        'main_results': results,
        'sharpness_sensitivity_A': sharpness_results,
    }

    output_path = os.path.join(RESULT_DIR, 'combine_gat_unc_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {output_path}")

    # 总结
    print(f"\n{'='*80}")
    print(f"总结:")
    print(f"  最佳基线: GAT_EvidenceSwap = {results['GAT_EvidenceSwap']['accuracy']:.2f}%")
    best_combo = max(
        [(k, v) for k, v in results.items() if k.startswith(('A_', 'B_', 'C_', 'D_'))],
        key=lambda x: x[1]['accuracy']
    )
    print(f"  最佳组合: {best_combo[0]} = {best_combo[1]['accuracy']:.2f}%")
    delta = best_combo[1]['accuracy'] - results['GAT_EvidenceSwap']['accuracy']
    print(f"  提升: {delta:+.2f}% vs GAT_EvidenceSwap")
    print(f"  最佳组合 vs Uncertainty_Weighted_DS: {best_combo[1]['delta_vs_uncds']:+.2f}%")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
