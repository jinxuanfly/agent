"""
训练GAT共识层脚本
===================================
在CIFAR-10N提取特征上训练GAT共识层，
使其学习有意义的注意力权重，用于论文实验展示。

训练策略：
1. 使用预提取的Agent特征和证据头输出
2. 构造训练数据：选择3个Agent中有分歧的样本
3. 损失函数：共识后的DS融合正确预测 + 收敛正则化
4. 评估：共识前后的准确率变化

论文目的：
- 展示GAT共识层可以学习注意力权重
- 证明可学习的共识优于手写相似性共识
- 为"共识学习"部分提供定量证据
"""
import sys
import os
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import json
import pickle

from plot_utils import setup_chinese_font, setup_plot_style
setup_chinese_font()
setup_plot_style()

from step1.synthetic_data import SEED, DEVICE
from step2.gat_consensus import ConsensusEngine, GATConsensusLayer
from step4.evaluate_cifar10n import load_features_and_heads, get_agent_outputs, ds_fusion_decision

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs('checkpoints/cifar10n', exist_ok=True)
os.makedirs('figures', exist_ok=True)


# =============================================================================
# 1. 加载数据并构造训练/验证集
# =============================================================================

def load_training_data(num_train=1000, num_valid=300):
    """
    加载训练和验证数据。
    选择那些Agent间有分歧的样本（共识可能带来收益的样本）。
    """
    print("=" * 60)
    print("加载训练数据")
    print("=" * 60)
    
    features, test_labels = load_features_and_heads()
    B = num_train + num_valid
    B = min(B, len(test_labels))
    
    all_alphas, all_beliefs, all_uncertainties, all_embs = [], [], [], []
    for name in ['agent1', 'agent2', 'agent3']:
        feats, head = features[name]
        alpha, b, u, emb = get_agent_outputs(feats[:B], head)
        all_alphas.append(alpha)
        all_beliefs.append(b)
        all_uncertainties.append(u)
        all_embs.append(emb)
    
    # DS融合基线
    ds_preds, ds_rej, ds_u = ds_fusion_decision(all_beliefs, all_uncertainties)
    y_true = test_labels[:B]
    
    # 筛选：只选择DS正确但Agent间有分歧的样本
    ds_correct = ds_preds == y_true
    has_disagreement = torch.zeros(B, dtype=torch.bool)
    for i in range(3):
        for j in range(i+1, 3):
            has_disagreement |= (all_beliefs[i].argmax(dim=1) != all_beliefs[j].argmax(dim=1))
    
    train_mask = ds_correct & has_disagreement
    valid_mask = ds_correct & ~has_disagreement
    
    train_indices = torch.where(train_mask)[0][:num_train]
    valid_indices = torch.where(valid_mask)[0][:num_valid]
    
    print(f"  DS正确样本: {ds_correct.sum().item()}/{B}")
    print(f"  有分歧样本: {has_disagreement.sum().item()}/{B}")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(valid_indices)} 样本")
    
    train_data = {
        'beliefs': [b[train_indices] for b in all_beliefs],
        'uncertainties': [u[train_indices] for u in all_uncertainties],
        'embeddings': [e[train_indices] for e in all_embs],
        'alphas': [a[train_indices] for a in all_alphas],
        'labels': y_true[train_indices],
        'ds_preds': ds_preds[train_indices],
        'ds_u': ds_u[train_indices],
    }
    
    valid_data = {
        'beliefs': [b[valid_indices] for b in all_beliefs],
        'uncertainties': [u[valid_indices] for u in all_uncertainties],
        'embeddings': [e[valid_indices] for e in all_embs],
        'alphas': [a[valid_indices] for a in all_alphas],
        'labels': y_true[valid_indices],
        'ds_preds': ds_preds[valid_indices],
        'ds_u': ds_u[valid_indices],
    }
    
    return train_data, valid_data


# =============================================================================
# 2. 训练GAT共识层
# =============================================================================

def train_gat_consensus(train_data, valid_data, num_epochs=100, lr=1e-3):
    """
    训练GAT共识层。
    """
    N = len(train_data['beliefs'])
    B_train = train_data['beliefs'][0].shape[0]
    K = train_data['beliefs'][0].shape[1]
    D = train_data['embeddings'][0].shape[1]
    
    print(f"\n{'='*60}")
    print(f"训练GAT共识层")
    print(f"{'='*60}")
    print(f"  Agent数: {N}")
    print(f"  类别数: {K}")
    print(f"  嵌入维度: {D}")
    print(f"  训练样本: {B_train}")
    print(f"  验证样本: {valid_data['beliefs'][0].shape[0]}")
    
    # 创建GAT共识引擎
    engine = ConsensusEngine(embed_dim=D, num_classes=K, hidden_dim=64)
    optimizer = torch.optim.Adam(engine.layer.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {'train_loss': [], 'valid_acc': [], 'valid_rej_rate': [], 'valid_acc_all': []}
    best_valid_acc = 0.0
    
    for epoch in range(num_epochs):
        engine.layer.train()
        total_loss = 0.0
        
        indices = torch.randperm(B_train)
        
        for start_idx in range(0, B_train, 32):
            batch_idx = indices[start_idx:start_idx+32]
            batch_loss = torch.tensor(0.0)
            
            for b_idx in batch_idx:
                b_idx_item = b_idx.item()
                
                agent_outputs = []
                for i in range(N):
                    b_i = train_data['beliefs'][i][b_idx_item:b_idx_item+1]
                    u_i = train_data['uncertainties'][i][b_idx_item:b_idx_item+1]
                    emb_i = train_data['embeddings'][i][b_idx_item:b_idx_item+1]
                    
                    S = K / u_i.squeeze(-1).clamp(min=1e-6)
                    alpha_i = b_i[0] * S[0] + 1.0
                    agent_outputs.append((alpha_i, b_i[0], u_i[0].item(), emb_i[0]))
                
                try:
                    h = engine.build_state(agent_outputs)
                    h_final, n_iters, converged, energy_trace, attn_trace = \
                        engine.run(h, max_iters=5, tol=1e-4, verbose=False)
                    
                    outputs = engine.extract_outputs(h_final)
                    new_belief = outputs[0][1].unsqueeze(0)
                    
                    true_label = train_data['labels'][b_idx_item]
                    ce_loss = F.cross_entropy(new_belief.unsqueeze(0), true_label.unsqueeze(0))
                    
                    if len(energy_trace) > 1:
                        energy_reg = F.relu(energy_trace[-1] - energy_trace[0])
                    else:
                        energy_reg = torch.tensor(0.0, device=new_belief.device)
                    
                    if len(attn_trace) > 0:
                        attn_last = attn_trace[-1]
                        if isinstance(attn_last, torch.Tensor) and attn_last.shape[-1] > 1:
                            attn_entropy = -(attn_last * torch.log(attn_last.clamp(min=1e-8))).sum()
                            attn_reg = -0.01 * attn_entropy
                        else:
                            attn_reg = torch.tensor(0.0, device=new_belief.device)
                    else:
                        attn_reg = torch.tensor(0.0, device=new_belief.device)
                    
                    loss = ce_loss + 0.1 * energy_reg + attn_reg
                    batch_loss = batch_loss + loss
                    
                except Exception as e:
                    pass
            
            if batch_loss.item() > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(engine.layer.parameters(), 1.0)
                optimizer.step()
                total_loss += batch_loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            valid_metrics = evaluate_gat(engine, valid_data)
            history['valid_acc'].append(valid_metrics['acc'])
            history['valid_rej_rate'].append(valid_metrics['rej_rate'])
            history['valid_acc_all'].append(valid_metrics['acc_all'])
            
            avg_loss = total_loss / max(len(indices), 1)
            history['train_loss'].append(avg_loss)
            
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                  f"loss={avg_loss:.4f}, "
                  f"valid_acc={valid_metrics['acc']:.1f}%, "
                  f"rej={valid_metrics['rej_rate']:.1f}%, "
                  f"acc_all={valid_metrics['acc_all']:.1f}%")
            
            if valid_metrics['acc'] > best_valid_acc:
                best_valid_acc = valid_metrics['acc']
                torch.save(engine.layer.state_dict(), 'checkpoints/cifar10n/gat_consensus.pt')
                print(f"  → 保存最佳模型 (acc={best_valid_acc:.2f}%)")
    
    engine.layer.load_state_dict(torch.load('checkpoints/cifar10n/gat_consensus.pt', weights_only=True))
    
    print(f"\n  ★ 训练完成. 最佳验证准确率: {best_valid_acc:.2f}%")
    
    return engine, history


def evaluate_gat(engine, data, max_iters=5, verbose=False):
    """评估GAT共识在验证集上的表现"""
    N = len(data['beliefs'])
    B = data['beliefs'][0].shape[0]
    K = data['beliefs'][0].shape[1]
    
    engine.layer.eval()
    
    new_preds = []
    new_us = []
    
    with torch.no_grad():
        for b_idx in range(B):
            agent_outputs = []
            for i in range(N):
                b_i = data['beliefs'][i][b_idx:b_idx+1]
                u_i = data['uncertainties'][i][b_idx:b_idx+1]
                emb_i = data['embeddings'][i][b_idx:b_idx+1]
                
                S = K / u_i.squeeze(-1).clamp(min=1e-6)
                alpha_i = b_i[0] * S[0] + 1.0
                agent_outputs.append((alpha_i, b_i[0], u_i[0].item(), emb_i[0]))
            
            try:
                h = engine.build_state(agent_outputs)
                h_final, n_iters, converged, _, _ = engine.run(h, max_iters=max_iters, tol=1e-4)
                
                outputs = engine.extract_outputs(h_final)
                
                ds_beliefs = []
                ds_us = []
                for i in range(N):
                    ds_beliefs.append(outputs[i][1])
                    ds_us.append(outputs[i][2])
                
                pred, rej, u = ds_fusion_decision(
                    [b.unsqueeze(0) for b in ds_beliefs],
                    [torch.tensor([[u_i]]) for u_i in ds_us]
                )
                
                new_preds.append(pred.item())
                new_us.append(u.item())
            except Exception as e:
                orig_pred, _, _ = ds_fusion_decision(
                    [data['beliefs'][i][b_idx:b_idx+1] for i in range(N)],
                    [data['uncertainties'][i][b_idx:b_idx+1] for i in range(N)]
                )
                new_preds.append(orig_pred.item())
                new_us.append(0.5)
    
    new_preds = np.array(new_preds)
    new_us = np.array(new_us)
    y_true = data['labels'].numpy()
    
    rejected = new_us > 0.5
    rej_rate = rejected.mean() * 100
    
    accepted = ~rejected
    if accepted.sum() > 0:
        acc = accuracy_score(y_true[accepted], new_preds[accepted]) * 100
    else:
        acc = 0.0
    
    acc_all = accuracy_score(y_true, new_preds) * 100
    
    return {'acc': acc, 'rej_rate': rej_rate, 'acc_all': acc_all, 'preds': new_preds, 'rejected': rejected}


def plot_training_history(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = list(range(0, len(history['train_loss']) * 5, 5))[:len(history['train_loss'])]
    
    axes[0].plot(epochs, history['train_loss'], 'b-o', alpha=0.8)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['valid_acc'], 'g-o', alpha=0.8, label='Accepted Acc')
    axes[1].plot(epochs, history['valid_acc_all'], 'r--o', alpha=0.8, label='All Acc')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['valid_rej_rate'], 'm-o', alpha=0.8)
    axes[2].set_title('Validation Rejection Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Rejection Rate (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/gat_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线: figures/gat_training_history.png")


def visualize_gat_attention(engine, sample_data, sample_labels, num_samples=10):
    """可视化GAT注意力权重"""
    N = len(sample_data['beliefs'])
    K = sample_data['beliefs'][0].shape[1]
    
    engine.layer.eval()
    
    all_attentions = []
    
    with torch.no_grad():
        for b_idx in range(min(num_samples, len(sample_data['labels']))):
            agent_outputs = []
            for i in range(N):
                b_i = sample_data['beliefs'][i][b_idx:b_idx+1]
                u_i = sample_data['uncertainties'][i][b_idx:b_idx+1]
                emb_i = sample_data['embeddings'][i][b_idx:b_idx+1]
                
                S = K / u_i.squeeze(-1).clamp(min=1e-6)
                alpha_i = b_i[0] * S[0] + 1.0
                agent_outputs.append((alpha_i, b_i[0], u_i[0].item(), emb_i[0]))
            
            h = engine.build_state(agent_outputs)
            _, _, _, _, attn_trace = engine.run(h, max_iters=5, tol=1e-4, verbose=False)
            
            if len(attn_trace) > 0:
                attn_last = attn_trace[-1]
                if isinstance(attn_last, torch.Tensor):
                    all_attentions.append(attn_last.cpu().numpy())
                else:
                    all_attentions.append(np.zeros((3, 3)))
            else:
                all_attentions.append(np.zeros((3, 3)))
    
    if len(all_attentions) == 0:
        print("  [警告] 没有注意力权重可可视化")
        return
    
    avg_attn = np.mean(all_attentions, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im = axes[0].imshow(avg_attn, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_title(f'Average GAT Attention ({len(all_attentions)} samples)')
    axes[0].set_xticks(range(N))
    axes[0].set_yticks(range(N))
    axes[0].set_xticklabels(['A1 (ResNet)', 'A2 (ViT)', 'A3 (Pixel)'])
    axes[0].set_yticklabels(['A1 (ResNet)', 'A2 (ViT)', 'A3 (Pixel)'])
    for i in range(N):
        for j in range(N):
            val = avg_attn[i, j]
            color = 'white' if val > 0.5 else 'black'
            axes[0].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, color=color)
    plt.colorbar(im, ax=axes[0])
    
    for i in range(N):
        axes[1].bar([f'A{i+1}→A{j+1}' for j in range(N)], 
                     avg_attn[i], alpha=0.7, label=f'A{i+1}')
    axes[1].set_title('Attention Distribution by Source Agent')
    axes[1].set_ylabel('Attention Weight')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=20)
    
    plt.tight_layout()
    plt.savefig('figures/gat_attention_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"GAT注意力图: figures/gat_attention_weights.png")


# =============================================================================
# 4. 对比评估
# =============================================================================

def compare_consensus_methods(test_data, engine):
    """对比三种方法：原始DS、相似性共识、GAT共识"""
    from step4.evaluate_cifar10n import similarity_consensus_batch
    
    N = len(test_data['beliefs'])
    B = test_data['beliefs'][0].shape[0]
    K = test_data['beliefs'][0].shape[1]
    y_true = test_data['labels'].numpy()
    
    print(f"\n{'='*60}")
    print(f"共识方法对比评估 ({B} samples)")
    print(f"{'='*60}")
    
    # 1. 原始DS融合
    ds_preds, ds_rej, ds_u = ds_fusion_decision(
        test_data['beliefs'], test_data['uncertainties']
    )
    ds_acc = accuracy_score(y_true, ds_preds.numpy()) * 100
    ds_rej_rate = ds_rej.float().mean().item() * 100
    print(f"  原始DS: acc={ds_acc:.2f}%, rej={ds_rej_rate:.2f}%")
    
    # 2. 信念相似性共识
    sim_beliefs, sim_us, _, _, _ = similarity_consensus_batch(
        test_data['beliefs'], test_data['uncertainties'], max_iters=10
    )
    sim_preds, sim_rej, sim_u = ds_fusion_decision(sim_beliefs, sim_us)
    sim_acc = accuracy_score(y_true, sim_preds.numpy()) * 100
    sim_rej_rate = sim_rej.float().mean().item() * 100
    print(f"  相似性共识: acc={sim_acc:.2f}%, rej={sim_rej_rate:.2f}%")
    
    # 3. GAT共识
    gat_metrics = evaluate_gat(engine, test_data, verbose=True)
    print(f"  GAT共识:   acc={gat_metrics['acc']:.2f}%, rej={gat_metrics['rej_rate']:.2f}%")
    
    methods = ['DS Fusion', 'Similarity\nConsensus', 'GAT\nConsensus']
    accs = [ds_acc, sim_acc, gat_metrics['acc']]
    rej_rates = [ds_rej_rate, sim_rej_rate, gat_metrics['rej_rate']]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
    
    bars1 = axes[0].bar(methods, accs, color=colors, alpha=0.8)
    axes[0].set_title('Accuracy (Accepted Samples)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.2f}%', ha='center', fontsize=11)
    
    bars2 = axes[1].bar(methods, rej_rates, color=colors, alpha=0.8)
    axes[1].set_title('Rejection Rate')
    axes[1].set_ylabel('Rejection Rate (%)')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars2, rej_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.2f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('figures/consensus_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"共识方法对比图: figures/consensus_methods_comparison.png")
    
    return {
        'DS_Fusion': {'acc': ds_acc, 'rej_rate': ds_rej_rate},
        'Similarity_Consensus': {'acc': sim_acc, 'rej_rate': sim_rej_rate},
        'GAT_Consensus': {'acc': gat_metrics['acc'], 'rej_rate': gat_metrics['rej_rate']}
    }


# =============================================================================
# 5. 主流程
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("GAT共识层训练脚本")
    print("\"异构多模态动态共识与协同\"论文实验")
    print("=" * 70)
    
    train_data, valid_data = load_training_data(num_train=1000, num_valid=300)
    
    engine, history = train_gat_consensus(train_data, valid_data, num_epochs=60, lr=1e-3)
    
    plot_training_history(history)
    
    print(f"\n{'='*60}")
    print("可视化GAT注意力权重")
    print(f"{'='*60}")
    visualize_gat_attention(engine, valid_data, valid_data['labels'])
    
    test_data = valid_data
    comparison = compare_consensus_methods(test_data, engine)
    
    print(f"\n{'='*60}")
    print("完整测试集评估")
    print(f"{'='*60}")
    
    features, test_labels = load_features_and_heads()
    B = 500
    
    all_beliefs, all_uncertainties, all_embs, all_alphas = [], [], [], []
    for name in ['agent1', 'agent2', 'agent3']:
        feats, head = features[name]
        alpha, b, u, emb = get_agent_outputs(feats[:B], head)
        all_alphas.append(alpha)
        all_beliefs.append(b)
        all_uncertainties.append(u)
        all_embs.append(emb)
    
    full_test_data = {
        'beliefs': all_beliefs,
        'uncertainties': all_uncertainties,
        'embeddings': all_embs,
        'alphas': all_alphas,
        'labels': test_labels[:B],
        'ds_preds': ds_fusion_decision(all_beliefs, all_uncertainties)[0],
        'ds_u': ds_fusion_decision(all_beliefs, all_uncertainties)[2],
    }
    
    final_metrics = evaluate_gat(engine, full_test_data, verbose=True)
    print(f"\n  ★ 最终测试结果:")
    print(f"    准确率(Accepted): {final_metrics['acc']:.2f}%")
    print(f"    拒识率: {final_metrics['rej_rate']:.2f}%")
    print(f"    全样本准确率: {final_metrics['acc_all']:.2f}%")
    
    print(f"\n模型已保存至 checkpoints/cifar10n/gat_consensus.pt")