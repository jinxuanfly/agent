# -*- coding: utf-8 -*-
"""
消融实验：验证Symbolic GAT和Uncertainty_Weighted_DS各自的贡献

简化版：直接调用主代码中的训练逻辑，对比symbolic vs random embedding
"""
import os, sys, io, json, time
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
    correlation_aware_ds_fusion,
    compute_agent_correlation,
    uncertainty_weighted_ds_fusion,
    HatefulMemesDataset,
    GATConsensusLayer,
    ConsensusEngine,
    DisagreementDeconstructor,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    DEVICE,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'hateful_memes')
RESULTS_ABLATION_DIR = os.path.join(RESULTS_DIR, 'ablation')
os.makedirs(RESULTS_ABLATION_DIR, exist_ok=True)


def load_cached_data():
    """加载训练集和验证集的缓存LLM推理结果"""
    print("[1] 加载缓存数据...")

    train_dataset = HatefulMemesDataset(split='train', max_samples=200, load_images=False)
    train_labels = torch.tensor(train_dataset.labels, dtype=torch.long)

    details_file = os.path.join(RESULTS_DIR, 'details_llm_gpt5_gemini_gpt5.json')
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


def compute_agent_acc(beliefs, labels):
    return [(b.argmax(dim=1) == labels).float().mean().item() for b in beliefs]


# =============================================================
# GAT训练函数（参考主代码）
# =============================================================

def train_gat(train_beliefs, train_uncertainties, train_embs, train_labels,
              embedding_type='symbolic', n_epochs=50):
    """训练GAT共识层

    Args:
        embedding_type: 'symbolic'（使用原embs）或 'random'（替换为随机投影）
    """
    print(f"\n  训练GAT ({embedding_type} embedding)...")

    B_train = train_beliefs[0].shape[0]
    embed_dim = 256
    gat_node_dim = embed_dim + NUM_CLASSES + 1

    # 替换embedding为随机投影
    if embedding_type == 'random':
        torch.manual_seed(42)
        proj_matrix = torch.randn(NUM_CLASSES, embed_dim) * 0.1
        train_embs_use = [train_beliefs[i] @ proj_matrix + torch.randn(B_train, embed_dim) * 0.01
                          for i in range(3)]
        print(f"    已用随机投影替换embedding")
    else:
        train_embs_use = train_embs

    gat_layer = GATConsensusLayer(
        node_dim=gat_node_dim, hidden_dim=64, embed_dim=embed_dim, num_classes=NUM_CLASSES
    ).to(DEVICE)

    # 训练数据准备
    train_preds = torch.stack([b.argmax(dim=1) for b in train_beliefs], dim=1)
    disagreement_mask = ~((train_preds[:, 0] == train_preds[:, 1]) &
                          (train_preds[:, 1] == train_preds[:, 2]))
    train_gat_indices = torch.where(disagreement_mask)[0]
    print(f"    分歧训练样本: {len(train_gat_indices)}")

    if len(train_gat_indices) < 5:
        train_gat_indices = torch.arange(B_train)

    optimizer = torch.optim.Adam(gat_layer.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    gat_layer.train()
    init_norm = sum(p.norm().item() for p in gat_layer.parameters())

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
                    emb_i = train_embs_use[i][b_idx_cpu:b_idx_cpu+1].to(DEVICE)
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
                except Exception as e:
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
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: loss={avg_loss:.4f}")

    final_norm = sum(p.norm().item() for p in gat_layer.parameters())
    print(f"    GAT训练完成: 参数变化={abs(final_norm - init_norm):.4f}, 最佳loss={best_loss:.4f}")

    return gat_layer, train_embs_use


def evaluate_gat_evidswap(gat_layer, val_beliefs, val_uncertainties, val_embs, val_labels,
                          embedding_type='symbolic'):
    """评估GAT_EvidenceSwap"""
    gat_layer.eval()
    B_val = val_beliefs[0].shape[0]
    embed_dim = 256

    if embedding_type == 'random':
        torch.manual_seed(42)
        proj_matrix = torch.randn(NUM_CLASSES, embed_dim) * 0.1
        val_embs_use = [val_beliefs[i] @ proj_matrix + torch.randn(B_val, embed_dim) * 0.01
                        for i in range(3)]
    else:
        val_embs_use = val_embs

    evidswap_preds = []

    with torch.no_grad():
        for b_idx in range(B_val):
            agent_outputs = []
            for i in range(3):
                b_i = val_beliefs[i][b_idx]
                u_i = val_uncertainties[i][b_idx]
                emb_i = val_embs_use[i][b_idx].to(DEVICE)
                u_val = float(u_i.item())
                S = NUM_CLASSES / max(u_val, 1e-6)
                alpha_i = b_i * S + 1.0
                agent_outputs.append((alpha_i, b_i, u_val, emb_i))

            engine_tmp = ConsensusEngine(embed_dim=embed_dim, num_classes=NUM_CLASSES, hidden_dim=64)
            engine_tmp.layer = gat_layer
            try:
                h = engine_tmp.build_state(agent_outputs)
                fusion_weights = gat_layer.forward_sample_weights(h)

                # 证据交换：用GAT权重加权融合，并替换最弱Agent
                agent_us = torch.tensor([val_uncertainties[i][b_idx].item() for i in range(3)])
                max_u_idx = agent_us.argmax().item()

                # 加权融合信念
                weighted_b = sum(val_beliefs[i][b_idx] * fusion_weights[i].item() for i in range(3))

                new_beliefs = [val_beliefs[i][b_idx].clone() for i in range(3)]
                new_uncertainties = [val_uncertainties[i][b_idx].clone() for i in range(3)]
                new_beliefs[max_u_idx] = weighted_b
                new_uncertainties[max_u_idx] = val_uncertainties[max_u_idx][b_idx] * 0.5

                preds, _, _ = ds_fusion_decision(
                    [b.unsqueeze(0) for b in new_beliefs],
                    [u.unsqueeze(0) for u in new_uncertainties],
                )
                evidswap_preds.append(preds[0].item())
            except Exception as e:
                preds, _, _ = ds_fusion_decision(
                    [val_beliefs[i][b_idx].unsqueeze(0) for i in range(3)],
                    [val_uncertainties[i][b_idx].unsqueeze(0) for i in range(3)],
                )
                evidswap_preds.append(preds[0].item())

    preds_tensor = torch.tensor(evidswap_preds)
    acc = accuracy_score(val_labels.numpy(), preds_tensor.numpy()) * 100
    f1 = f1_score(val_labels.numpy(), preds_tensor.numpy(), average='macro') * 100
    return {'acc': acc, 'f1': f1}


def main():
    print("=" * 80)
    print("消融实验：Symbolic GAT + Uncertainty_Weighted_DS")
    print("=" * 80)

    data = load_cached_data()

    train_b = data['train']['beliefs']
    train_u = data['train']['uncertainties']
    train_e = data['train']['embs']
    train_labels = data['train']['labels']

    val_b = data['val']['beliefs']
    val_u = data['val']['uncertainties']
    val_e = data['val']['embs']
    val_labels = data['val']['labels']

    train_accs = compute_agent_acc(train_b, train_labels)
    val_accs = compute_agent_acc(val_b, val_labels)
    corr_matrix = compute_agent_correlation(train_b, train_labels)

    print(f"\nAgent准确率: train={[f'{a*100:.1f}%' for a in train_accs]}, val={[f'{a*100:.1f}%' for a in val_accs]}")

    # =============================================================
    # 消融1: Symbolic GAT vs Random GAT vs No GAT
    # =============================================================
    print("\n" + "=" * 80)
    print("[消融1] Symbolic GAT vs Random GAT vs No GAT")
    print("=" * 80)

    ablation_gat = {}

    # 1a. No GAT（直接DS）
    print("\n[1a] No GAT基线...")
    preds, _, _ = ds_fusion_decision(val_b, val_u)
    acc = accuracy_score(val_labels.numpy(), preds.numpy()) * 100
    f1 = f1_score(val_labels.numpy(), preds.numpy(), average='macro') * 100
    ablation_gat['No_GAT(DS)'] = {'acc': acc, 'f1': f1}
    print(f"  No GAT (DS): Acc={acc:.2f}%, F1={f1:.2f}%")

    # 1b. Symbolic GAT
    print("\n[1b] Symbolic GAT...")
    gat_sym, _ = train_gat(train_b, train_u, train_e, train_labels, 'symbolic', n_epochs=50)
    result_sym = evaluate_gat_evidswap(gat_sym, val_b, val_u, val_e, val_labels, 'symbolic')
    ablation_gat['Symbolic_GAT_EvidSwap'] = result_sym
    print(f"  Symbolic GAT (EvidSwap): Acc={result_sym['acc']:.2f}%, F1={result_sym['f1']:.2f}%")

    # 1c. Random GAT
    print("\n[1c] Random GAT...")
    gat_rand, _ = train_gat(train_b, train_u, train_e, train_labels, 'random', n_epochs=50)
    result_rand = evaluate_gat_evidswap(gat_rand, val_b, val_u, val_e, val_labels, 'random')
    ablation_gat['Random_GAT_EvidSwap'] = result_rand
    print(f"  Random GAT (EvidSwap): Acc={result_rand['acc']:.2f}%, F1={result_rand['f1']:.2f}%")

    sym_acc = ablation_gat['Symbolic_GAT_EvidSwap']['acc']
    rand_acc = ablation_gat['Random_GAT_EvidSwap']['acc']
    nogat_acc = ablation_gat['No_GAT(DS)']['acc']
    print(f"\n  [Symbolic GAT贡献]")
    print(f"    vs No GAT:    {sym_acc:.2f}% - {nogat_acc:.2f}% = +{sym_acc - nogat_acc:.2f}%")
    print(f"    vs Random GAT: {sym_acc:.2f}% - {rand_acc:.2f}% = +{sym_acc - rand_acc:.2f}%")

    # =============================================================
    # 消融2: Uncertainty_Weighted_DS组件
    # =============================================================
    print("\n" + "=" * 80)
    print("[消融2] Uncertainty_Weighted_DS组件消融")
    print("=" * 80)

    ablation_uw = {}

    # 2a. DS等权重
    print("\n[2a] DS(等权重)...")
    preds, _, _ = ds_fusion_decision(val_b, val_u, agent_weights=None)
    acc = accuracy_score(val_labels.numpy(), preds.numpy()) * 100
    f1 = f1_score(val_labels.numpy(), preds.numpy(), average='macro') * 100
    ablation_uw['DS_等权重'] = {'acc': acc, 'f1': f1}
    print(f"  Acc={acc:.2f}%, F1={f1:.2f}%")

    # 2b. DS训练集准确率权重
    print("\n[2b] DS(训练集准确率权重)...")
    accs_tensor = torch.tensor(train_accs)
    acc_weights = accs_tensor / accs_tensor.sum()
    preds, _, _ = ds_fusion_decision(val_b, val_u, agent_weights=acc_weights)
    acc = accuracy_score(val_labels.numpy(), preds.numpy()) * 100
    f1 = f1_score(val_labels.numpy(), preds.numpy(), average='macro') * 100
    ablation_uw['DS_训练集准确率权重'] = {'acc': acc, 'f1': f1}
    print(f"  Acc={acc:.2f}%, F1={f1:.2f}%, weights={acc_weights.tolist()}")

    # 2c. Corr_Aware_DS(ds=0.5)
    print("\n[2c] Corr_Aware_DS(ds=0.5)...")
    preds, _, _ = correlation_aware_ds_fusion(val_b, val_u, correlation_matrix=corr_matrix,
                                               discount_strength=0.5)
    acc = accuracy_score(val_labels.numpy(), preds.numpy()) * 100
    f1 = f1_score(val_labels.numpy(), preds.numpy(), average='macro') * 100
    ablation_uw['Corr_Aware_DS(ds=0.5)'] = {'acc': acc, 'f1': f1}
    print(f"  Acc={acc:.2f}%, F1={f1:.2f}%")

    # 2d. Uncertainty_Weighted_DS(s=20)
    print("\n[2d] Uncertainty_Weighted_DS(s=20)...")
    preds, _, _ = uncertainty_weighted_ds_fusion(val_b, val_u, sharpness=20.0)
    acc = accuracy_score(val_labels.numpy(), preds.numpy()) * 100
    f1 = f1_score(val_labels.numpy(), preds.numpy(), average='macro') * 100
    ablation_uw['Uncertainty_Weighted_DS(s=20)'] = {'acc': acc, 'f1': f1}
    print(f"  Acc={acc:.2f}%, F1={f1:.2f}%")

    # 2e. Uncertainty_Weighted_DS(s=10)
    print("\n[2e] Uncertainty_Weighted_DS(s=10)...")
    preds, _, _ = uncertainty_weighted_ds_fusion(val_b, val_u, sharpness=10.0)
    acc = accuracy_score(val_labels.numpy(), preds.numpy()) * 100
    f1 = f1_score(val_labels.numpy(), preds.numpy(), average='macro') * 100
    ablation_uw['Uncertainty_Weighted_DS(s=10)'] = {'acc': acc, 'f1': f1}
    print(f"  Acc={acc:.2f}%, F1={f1:.2f}%")

    uw_acc = ablation_uw['Uncertainty_Weighted_DS(s=20)']['acc']
    eq_acc = ablation_uw['DS_等权重']['acc']
    tr_acc = ablation_uw['DS_训练集准确率权重']['acc']
    print(f"\n  [Uncertainty_Weighted贡献]")
    print(f"    vs 等权重DS:          {uw_acc:.2f}% - {eq_acc:.2f}% = +{uw_acc - eq_acc:.2f}%")
    print(f"    vs 训练集准确率权重DS: {uw_acc:.2f}% - {tr_acc:.2f}% = +{uw_acc - tr_acc:.2f}%")

    # =============================================================
    # 消融3: 分歧样本分析
    # =============================================================
    print("\n" + "=" * 80)
    print("[消融3] 分歧样本上的表现")
    print("=" * 80)

    val_preds_stack = torch.stack([b.argmax(dim=1) for b in val_b], dim=1)
    disagree_mask = ~((val_preds_stack[:, 0] == val_preds_stack[:, 1]) &
                       (val_preds_stack[:, 1] == val_preds_stack[:, 2]))
    n_disagree = disagree_mask.sum().item()
    print(f"  分歧样本数: {n_disagree}/{len(val_labels)}")

    ablation_disagree = {}
    print(f"\n  {'方法':<35} {'全样本Acc%':<12} {'分歧Acc%':<12}")
    print(f"  {'-'*60}")

    methods_to_eval = [
        ('DS_等权重', lambda: ds_fusion_decision(val_b, val_u, agent_weights=None)[0]),
        ('DS_训练集准确率权重', lambda: ds_fusion_decision(val_b, val_u, agent_weights=acc_weights)[0]),
        ('Corr_Aware_DS(ds=0.5)', lambda: correlation_aware_ds_fusion(
            val_b, val_u, correlation_matrix=corr_matrix, discount_strength=0.5)[0]),
        ('Uncertainty_Weighted_DS(s=20)', lambda: uncertainty_weighted_ds_fusion(
            val_b, val_u, sharpness=20.0)[0]),
        ('Uncertainty_Weighted_DS(s=10)', lambda: uncertainty_weighted_ds_fusion(
            val_b, val_u, sharpness=10.0)[0]),
    ]

    for name, func in methods_to_eval:
        preds = func()
        all_acc = accuracy_score(val_labels.numpy(), preds.numpy()) * 100
        if n_disagree > 0:
            dis_acc = accuracy_score(val_labels[disagree_mask].numpy(),
                                     preds[disagree_mask].numpy()) * 100
        else:
            dis_acc = 0.0
        ablation_disagree[name] = {'all_acc': all_acc, 'disagree_acc': dis_acc}
        print(f"  {name:<35} {all_acc:<12.2f} {dis_acc:<12.2f}")

    # =============================================================
    # 总结
    # =============================================================
    print("\n" + "=" * 80)
    print("[消融实验总结]")
    print("=" * 80)

    print("\n[消融1: Symbolic GAT贡献]")
    print(f"  {'方法':<30} {'Acc%':<10} {'F1%':<10}")
    print(f"  {'-'*50}")
    for m, r in sorted(ablation_gat.items(), key=lambda x: x[1]['acc'], reverse=True):
        print(f"  {m:<30} {r['acc']:<10.2f} {r['f1']:<10.2f}")

    print(f"\n[消融2: Uncertainty_Weighted_DS贡献]")
    print(f"  {'方法':<35} {'Acc%':<10} {'F1%':<10}")
    print(f"  {'-'*55}")
    for m, r in sorted(ablation_uw.items(), key=lambda x: x[1]['acc'], reverse=True):
        print(f"  {m:<35} {r['acc']:<10.2f} {r['f1']:<10.2f}")

    # 保存
    output = {
        'ablation1_symbolic_gat': ablation_gat,
        'ablation2_uncertainty_weighted': ablation_uw,
        'ablation3_disagreement_samples': ablation_disagree,
        'agent_accuracies': {'train': train_accs, 'val': val_accs},
        'correlation_matrix': corr_matrix.tolist(),
    }
    out_file = os.path.join(RESULTS_ABLATION_DIR, 'ablation_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存至: {out_file}")


if __name__ == '__main__':
    main()
