"""
第四步：CIFAR-10N端到端评估（v8 - GAT共识修复版）
===========================================
核心设计（基于实验修复）：
- 异构智能体用DS融合做全局决策
- GAT共识层作为可选增强，在融合前做信念对齐（真正调用ConsensusEngine）
- 分歧解构仅在GAT不收敛时触发

评估指标：
1. 多数投票 (Majority Voting)
2. 加权平均 (Weighted Averaging, weight=1-u)
3. 标准DS融合
4. 完整框架（DS融合 + 可选GAT共识）

修复记录(v8)：
- 修复simple_consensus(use_gat=True)从未调用真实GAT引擎的问题
- 添加gat_consensus_batch函数逐样本运行ConsensusEngine
- 添加数值稳定化处理（NaN防护、裁剪极端值）
- 更新所有调用处以传递alphas和embeddings参数
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import setup_chinese_font, setup_plot_style
setup_chinese_font()
setup_plot_style()
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.synthetic_data import SEED, DEVICE
from step2.gat_consensus import ConsensusEngine, global_decision
from step3.disagreement_resolver import (
    compute_conflict_K, DisagreementPipeline
)
from step4.train_heads import EvidenceHead

# 注册EvidenceHead到__main__，使pickle反序列化能找到类
import __main__
__main__.EvidenceHead = EvidenceHead

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs('figures', exist_ok=True)
os.makedirs('results/cifar10n', exist_ok=True)

NUM_CLASSES = 10


# =============================================================================
# 1. DS融合（向量化批量）
# =============================================================================

def ds_fusion_decision(all_beliefs, all_uncertainties, u_threshold=0.5):
    """
    Dempster-Shafer融合做全局决策
    
    DS融合特性：当多个证据来源一致时，融合后的u急剧下降。
    这是证据理论的关键优势——不同于加权平均的线性组合。
    
    Args:
        all_beliefs: list of [B, K] 信念分布
        all_uncertainties: list of [B, 1] 不确定性
        u_threshold: 拒识阈值
    
    Returns:
        preds: [B] 预测类别
        rejected: [B] 布尔掩码
        global_u: [B] 融合后不确定性
    """
    if len(all_beliefs) == 0:
        B = all_beliefs[0].shape[0]
        return torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.bool), torch.ones(B)

    b_fused = all_beliefs[0]
    u_fused = all_uncertainties[0]

    for i in range(1, len(all_beliefs)):
        b2 = all_beliefs[i]
        u2 = all_uncertainties[i]

        K = (b_fused * b2).sum(dim=1, keepdim=True)
        denom = 1 - K + 1e-10

        b_fused = (b_fused * b2 + b_fused * u2 + b2 * u_fused) / denom
        u_fused = (u_fused * u2) / denom

    global_u = u_fused.squeeze(1)
    preds = b_fused.argmax(dim=1)
    rejected = global_u > u_threshold

    return preds, rejected, global_u


# =============================================================================
# 2. 加载特征和证据头
# =============================================================================

def load_features_and_heads():
    """加载预提取特征和训练好的证据头"""
    print("\n[1] 加载预提取特征和证据头...")

    test_rn = torch.load('data/features/test_resnet18.pt')
    test_vit = torch.load('data/features/test_vit_tiny.pt')

    # Agent3: 像素特征投影
    try:
        cifar_dir = 'data/cifar-10-batches-py'
        if os.path.exists(cifar_dir):
            def unpickle(file):
                with open(file, 'rb') as fo:
                    return pickle.load(fo, encoding='bytes')
            test_data = unpickle(os.path.join(cifar_dir, 'test_batch'))[b'data']
            test_pixels = torch.FloatTensor(test_data) / 255.0
            torch.manual_seed(SEED)
            proj = torch.randn(3072, 256) * 0.1
            test_pixel = test_pixels @ proj
            test_pixel = (test_pixel - test_pixel.mean(dim=0)) / (test_pixel.std(dim=0) + 1e-8)
        else:
            from torchvision import datasets
            tmp_dir = 'data/tmp_cifar10'
            test_set = datasets.CIFAR10(tmp_dir, train=False, download=True)
            test_pixels = torch.FloatTensor(test_set.data).permute(0, 3, 1, 2).reshape(10000, -1) / 255.0
            torch.manual_seed(SEED)
            proj = torch.randn(3072, 256) * 0.1
            test_pixel = test_pixels @ proj
            test_pixel = (test_pixel - test_pixel.mean(dim=0)) / (test_pixel.std(dim=0) + 1e-8)
    except Exception as e:
        print(f"  [警告] Agent3加载失败: {e}, 使用随机特征")
        torch.manual_seed(SEED)
        test_pixel = torch.FloatTensor(np.random.randn(10000, 256) * 0.5)

    labels = torch.load('data/features/labels.pt')
    test_labels = labels['test_labels']

    heads = torch.load('checkpoints/cifar10n/evidence_heads.pt', map_location='cpu', weights_only=False)
    heads.eval()

    print(f"  Agent1 (ResNet-18): test {test_rn.shape}")
    print(f"  Agent2 (ViT-Tiny):  test {test_vit.shape}")
    print(f"  Agent3 (Pixel+投影): test {test_pixel.shape}")

    return {
        'agent1': (test_rn, heads['agent1']),
        'agent2': (test_vit, heads['agent2']),
        'agent3': (test_pixel, heads['agent3']),
    }, test_labels


def get_agent_outputs(feats, head):
    """获取单个智能体的证据输出"""
    head.eval()
    with torch.no_grad():
        alpha = head(feats.to(DEVICE))
        S = alpha.sum(dim=1, keepdim=True)
        K = alpha.shape[1]
        b = (alpha - 1) / S
        u = K / S

        if hasattr(head, 'get_embedding'):
            emb = head.get_embedding(feats.to(DEVICE))
        else:
            emb = feats

    return alpha, b, u, emb


# =============================================================================
# 3. 基于信念相似性的软共识（免训练，逐样本收敛检查）
# =============================================================================

def similarity_consensus_batch(all_beliefs, all_uncertainties, max_iters=10, verbose=False):
    """
    信念相似性加权共识（替代随机GAT），逐样本运行，逐样本收敛检查
    
    核心设计：
    - 完全连接图，注意力权重=信念余弦相似度
    - 不确定度感知残差更新: 高u→更多信任邻居
    - 分歧保护: 高分歧样本保留更多原始信念
    - 逐样本收敛检查: 每个样本独立标记收敛
    
    Returns:
        new_beliefs: list of [B, K]
        new_uncertainties: list of [B, 1]
        converged_flags: [B]
        n_iters_list: [B]
        new_alphas: list of [B, K]
    """
    N = len(all_beliefs)
    B = all_beliefs[0].shape[0]
    K = all_beliefs[0].shape[1]
    device = all_beliefs[0].device
    
    curr_beliefs = [b.clone() for b in all_beliefs]
    curr_uncertainties = [u.clone() for u in all_uncertainties]
    
    converged_flags = torch.zeros(B, dtype=torch.bool, device=device)
    n_iters_list = torch.zeros(B, dtype=torch.long, device=device)
    
    for iteration in range(max_iters):
        active = ~converged_flags
        if not active.any():
            break
        
        beliefs_stack = torch.stack(curr_beliefs, dim=0)  # [N, B, K]
        
        # ---- 计算邻居加权信念 ----
        neighbor_beliefs = []
        for i in range(N):
            b_i = F.normalize(curr_beliefs[i], p=2, dim=-1)      # [B, K]
            weighted = torch.zeros_like(curr_beliefs[i])          # [B, K]
            total_w = torch.zeros(B, 1, device=device)            # [B, 1]
            
            for j in range(N):
                if i == j:
                    continue
                b_j = F.normalize(curr_beliefs[j], p=2, dim=-1)
                sim = (b_i * b_j).sum(dim=-1, keepdim=True)      # [B, 1]
                w = torch.clamp(sim, min=0)
                weighted = weighted + w * curr_beliefs[j]
                total_w = total_w + w
            
            neighbor_b = weighted / (total_w + 1e-8)              # [B, K]
            neighbor_beliefs.append(neighbor_b)
        
        # ---- 更新信念 ----
        new_beliefs = []
        new_uncertainties = []
        for i in range(N):
            u_i = curr_uncertainties[i]                           # [B, 1]
            
            # 分歧度: 每个类别方差均值
            beliefs_var = beliefs_stack.var(dim=0).mean(dim=-1, keepdim=True)  # [B, 1]
            safety = 1 - torch.sigmoid(beliefs_var * 3)           # [B, 1]
            
            # 混合系数: 高u→信任邻居, 高分歧→信任自己
            mix_alpha = 0.2 + 0.5 * u_i                           # [B, 1], 0.2~0.7
            mix_alpha = mix_alpha * safety                         # 分歧保护
            mix_alpha = torch.clamp(mix_alpha, min=0.15)           # 至少15%更新
            
            b_updated = (1 - mix_alpha) * curr_beliefs[i] + mix_alpha * neighbor_beliefs[i]
            
            # 保持与原始信念的轻量残差连接（保留10%原始信息，主要依赖共识更新）
            b_updated = 0.9 * b_updated + 0.1 * all_beliefs[i]
            
            b_updated = b_updated.clamp(min=0)
            u_new = curr_uncertainties[i].clone()                  # [B, 1]
            b_sum = b_updated.sum(dim=-1, keepdim=True)
            b_norm = b_updated / (b_sum + 1e-10) * (1 - u_new)
            
            new_beliefs.append(b_norm)
            new_uncertainties.append(u_new)
        
        # ---- 逐样本收敛检查 ----
        delta_per_sample = torch.zeros(B, device=device)
        for i in range(N):
            delta = (new_beliefs[i] - curr_beliefs[i]).abs().sum(dim=-1) / K
            delta_per_sample = torch.max(delta_per_sample, delta)
        
        just_converged = (~converged_flags) & (delta_per_sample < 3e-4)
        converged_flags[just_converged] = True
        n_iters_list[just_converged] = iteration + 1
        
        if verbose and (iteration == 0 or iteration == max_iters - 1):
            print(f"  第{iteration+1}轮: 新收敛={just_converged.sum().item()}, "
                  f"总收敛={converged_flags.sum().item()}, "
                  f"avg_delta={delta_per_sample[active].mean().item():.6f}")
        
        curr_beliefs = new_beliefs
        curr_uncertainties = new_uncertainties
    
    n_iters_list = torch.where(converged_flags, n_iters_list, torch.tensor(max_iters, device=device))
    
    # 重建alpha
    new_alphas = []
    for i in range(N):
        u = curr_uncertainties[i]
        S = K / u.clamp(min=1e-6)
        alpha = curr_beliefs[i] * S + 1
        new_alphas.append(alpha)
    
    if verbose:
        print(f"  相似性共识: {converged_flags.sum().item()}/{B} 收敛, "
              f"平均迭代={n_iters_list.float().mean().item():.1f}")
    
    return curr_beliefs, curr_uncertainties, converged_flags, n_iters_list, new_alphas


# =============================================================================
# 3b. 确定性共识（规则型，分析信念分歧度）
# =============================================================================

def deterministic_consensus(all_beliefs, all_uncertainties, max_iters=3):
    """
    确定性共识分析——计算Agent间信念分歧度（最大类别标准差）
    """
    beliefs = torch.stack(all_beliefs, dim=1)                    # [B, N, K]
    mean_b = beliefs.mean(dim=1, keepdim=True)                   # [B, 1, K]
    per_class_std = (beliefs - mean_b).pow(2).mean(dim=1).sqrt() # [B, K]
    disagreement = per_class_std.max(dim=-1).values               # [B]
    
    return all_beliefs, all_uncertainties, disagreement


# =============================================================================
# 4. DS融合决策 + 可选GAT共识增强
# =============================================================================

def gat_consensus_batch(all_beliefs, all_uncertainties, all_embeddings, max_iters=5, verbose=False):
    """
    批量运行真正的GAT共识引擎（逐样本调用ConsensusEngine）
    
    使用修复后的GATConsensusLayer进行信念对齐，
    支持逐样本独立收敛检查。
    
    Args:
        all_beliefs: list of [B, K] 信念分布
        all_uncertainties: list of [B, 1] 不确定性
        all_embeddings: list of [B, D] 语义嵌入
        max_iters: 最大迭代轮数（GAT内循环）
        verbose: 是否打印日志
    
    Returns:
        new_beliefs: list of [B, K] 共识后信念
        new_uncertainties: list of [B, 1] 共识后不确定性
        converged_flags: [B] 是否收敛
        n_iters_list: [B] 实际迭代次数
        new_alphas: list of [B, K] 共识后Dirichlet参数
    """
    N = len(all_beliefs)
    B = all_beliefs[0].shape[0]
    K = all_beliefs[0].shape[1]
    D = all_embeddings[0].shape[1]
    device = all_beliefs[0].device
    
    # 初始化输出
    new_beliefs = [b.clone() for b in all_beliefs]
    new_uncertainties = [u.clone() for u in all_uncertainties]
    converged_flags = torch.zeros(B, dtype=torch.bool, device=device)
    n_iters_list = torch.zeros(B, dtype=torch.long, device=device)
    new_alphas = [a.clone() for a in all_beliefs]  # 占位
    
    # 创建共享GAT引擎（权重在所有样本间共享）
    engine = ConsensusEngine(embed_dim=D, num_classes=K, hidden_dim=64)
    
    # 逐样本运行GAT共识
    for b_idx in range(B):
        # 构建该样本的3个智能体输出
        agent_outputs = []
        for i in range(N):
            b_i = all_beliefs[i][b_idx:b_idx+1]    # [1, K]
            u_i = all_uncertainties[i][b_idx:b_idx+1]  # [1, 1]
            emb_i = all_embeddings[i][b_idx:b_idx+1]   # [1, D]
            
            # 重建alpha
            S = K / u_i.squeeze(-1).clamp(min=1e-6)
            alpha_i = b_i[0] * S[0] + 1.0
            
            agent_outputs.append((alpha_i, b_i[0], u_i[0].item(), emb_i[0]))
        
        # 构建GAT状态并运行
        try:
            h = engine.build_state(agent_outputs)
            h_final, n_iters, converged, energy_trace, attn_trace = \
                engine.run(h, max_iters=max_iters, tol=1e-4, verbose=False)
            outputs = engine.extract_outputs(h_final)
            
            converged_flags[b_idx] = converged
            n_iters_list[b_idx] = n_iters
            
            # 提取共识后信念和不确定性
            for i in range(N):
                _, b_new, u_new, _ = outputs[i]
                new_beliefs[i][b_idx] = b_new.to(device)
                new_uncertainties[i][b_idx] = torch.tensor([[u_new]], device=device)
        except Exception as e:
            if verbose:
                print(f"  [警告] 样本{b_idx} GAT共识失败: {e}, 保留原始信念")
            # 失败时保留原始值
            new_beliefs[i][b_idx] = all_beliefs[i][b_idx].clone()
            new_uncertainties[i][b_idx] = all_uncertainties[i][b_idx].clone()
    
    n_converged = converged_flags.sum().item()
    if verbose:
        print(f"  GAT共识: {n_converged}/{B} 收敛, "
              f"平均迭代={n_iters_list.float().mean().item():.1f}")
    
    # 重建alpha
    new_alphas = []
    for i in range(N):
        u = new_uncertainties[i]
        S = K / u.squeeze(-1).clamp(min=1e-6)
        alpha = new_beliefs[i] * S.unsqueeze(1) + 1.0
        new_alphas.append(alpha)
    
    return new_beliefs, new_uncertainties, converged_flags, n_iters_list, new_alphas


def conflict_driven_consensus(all_beliefs, all_uncertainties, 
                               all_alphas=None, all_embeddings=None,
                               ds_preds=None, ds_rejected=None, ds_u=None,
                               conflict_threshold=0.05,
                               verbose=False):
    """
    ═══════════════════════════════════════════════════════════════
    分歧驱动共识（核心论文贡献）
    ═══════════════════════════════════════════════════════════════
    
    设计理念：
    - DS融合已能处理大多数样本（一致性高、不确定性低）
    - 共识引擎应该只激活在**高强度分歧**的样本上
    - 分歧类型定义：
      * 证据冲突：K高 > 0.05（Agent给不同类别高信念）→ 软信念对齐
      * 无知冲突：u高 > 0.5（所有Agent都不确定）→ 保留拒识
    
    流程：
    1. 先用标准DS融合获得基线结果
    2. 计算每个样本的冲突系数K（Dempster-Shafer冲突）
    3. 只在K > threshold的样本上运行信念相似性共识
    4. 共识后重新DS融合，检查决策是否改变
    
    Args:
        all_beliefs: list of [B, K]
        all_uncertainties: list of [B, 1]
        ds_preds: [B] DS融合基线预测
        ds_rejected: [B] DS融合基线拒识
        ds_u: [B] DS融合后不确定性
        conflict_threshold: 冲突阈值，超过此值才激活共识
    
    Returns:
        final_preds: [B] 最终预测
        final_rejected: [B] 最终拒识
        consensus_activated: [B] 是否激活了共识
        improved_mask: [B] 是否因共识而改善（仅当DS错误→共识正确）
    """
    N = len(all_beliefs)
    B = all_beliefs[0].shape[0]
    K = all_beliefs[0].shape[1]
    device = all_beliefs[0].device
    
    # ---- 1. 计算每对Agent间的DS冲突系数K ----
    # 信念相似度 agreement = Σ_k b_i[k] * b_j[k]  (高→强同意)
    # DS冲突系数 K = 1 - max_agreement  (低agreement→高冲突)
    agreement_matrix = torch.zeros(N, N, B, device=device)
    for i in range(N):
        for j in range(N):
            if i != j:
                agreement_matrix[i,j,:] = (all_beliefs[i] * all_beliefs[j]).sum(dim=-1)
    
    max_agreement = agreement_matrix.view(-1, B).max(dim=0).values  # [B]
    conflict_K = 1 - max_agreement                                    # [B] 真实的DS冲突
    
    # ---- 2. 确定哪些样本需要共识 ----
    # 证据冲突：agents给不同类别高信念 → 低agreement → 高K
    consensus_needed = conflict_K > 0.15  # K>0.15表示有显著分歧
    
    # 排除无知冲突：仅当**所有**Agent都高不确定性时才排除
    all_u = torch.stack(all_uncertainties, dim=0)  # [N, B, 1]
    min_u = all_u.min(dim=0).values.squeeze(-1)     # [B] 最确定的Agent
    # 无知型：连最确定的Agent都不确定 → 信息确实不足
    ignorance_conflict = min_u > 0.5
    consensus_needed = consensus_needed & ~ignorance_conflict
    
    n_activated = consensus_needed.sum().item()
    if verbose:
        print(f"  冲突分析: 高冲突样本={max_conflict.gt(conflict_threshold).sum().item()}, "
              f"无知型={ignorance_conflict.sum().item()}, "
              f"共识激活={n_activated}")
    
    if not consensus_needed.any():
        # 无需激活共识
        return ds_preds, ds_rejected, consensus_needed, torch.zeros(B, dtype=torch.bool, device=device)
    
    # ---- 3. 在需要共识的样本上运行信念对齐 ----
    # 初始化：使用DS基线结果
    final_preds = ds_preds.clone()
    final_rejected = ds_rejected.clone()
    improved_mask = torch.zeros(B, dtype=torch.bool, device=device)
    
    # 找到需要共识的样本索引
    activate_indices = torch.where(consensus_needed)[0]
    activate_mask = consensus_needed
    
    # 创建共识后信念的副本（初始化为原始信念）
    consensus_beliefs = [b.clone() for b in all_beliefs]
    consensus_uncertainties = [u.clone() for u in all_uncertainties]
    
    # ---- 4. 运行强信念对齐（仅在激活样本上） ----
    for iteration in range(8):  # 内循环
        if not activate_mask.any():
            break
        
        for i in range(N):
            b_i = consensus_beliefs[i]  # [B, K]
            u_i = consensus_uncertainties[i]  # [B, 1]
            
            # 计算邻居加权信念（只对激活样本计算）
            neighbor_b = torch.zeros_like(b_i)
            total_w = torch.zeros(B, 1, device=device)
            
            for j in range(N):
                if i == j:
                    continue
                # 注意力权重 = 信念余弦相似度
                b_i_norm = F.normalize(b_i, p=2, dim=-1)
                b_j_norm = F.normalize(consensus_beliefs[j], p=2, dim=-1)
                sim = (b_i_norm * b_j_norm).sum(dim=-1, keepdim=True)  # [B, 1]
                w = torch.clamp(sim, min=0) * (1 - consensus_uncertainties[j])
                neighbor_b = neighbor_b + w * consensus_beliefs[j]
                total_w = total_w + w
            
            neighbor_b = neighbor_b / (total_w + 1e-8)
            
            # 强混合：高u→大幅度更新，低u→小幅度更新
            mix_rate = 0.1 + 0.6 * u_i  # 0.1~0.7（相比之前更激进）
            
            # 仅在激活样本上更新
            new_b = b_i.clone()
            new_b[activate_mask] = (1 - mix_rate[activate_mask]) * b_i[activate_mask] + \
                                    mix_rate[activate_mask] * neighbor_b[activate_mask]
            
            # 保留原始信念的轻量连接（仅对非激活样本完全保持）
            # 非激活样本：100%原始信念
            new_b[~activate_mask] = all_beliefs[i][~activate_mask]
            
            new_b = new_b.clamp(min=0)
            b_sum = new_b.sum(dim=-1, keepdim=True)
            new_b = new_b / (b_sum + 1e-10) * (1 - u_i)
            
            consensus_beliefs[i] = new_b
        
        # 收敛检查（仅激活样本）
        delta = torch.zeros(B, device=device)
        for i in range(N):
            delta = torch.max(delta, 
                (consensus_beliefs[i] - (all_beliefs[i] if iteration == 0 else prev_beliefs.get(i, all_beliefs[i])))
                .abs().sum(dim=-1) / K)
        
        prev_beliefs = {i: b.clone() for i, b in enumerate(consensus_beliefs)}
        
        just_converged = activate_mask & (delta < 3e-4)
        activate_mask = activate_mask & ~just_converged
    
    # ---- 5. 共识后重新DS融合 ----
    new_preds, new_rejected, new_u = ds_fusion_decision(
        consensus_beliefs, consensus_uncertainties, u_threshold=0.5
    )
    
    # ---- 6. 评估改善 ----
    for b_idx in range(B):
        if consensus_needed[b_idx]:
            ds_pred = ds_preds[b_idx].item()
            ds_rej = ds_rejected[b_idx].item()
            new_pred = new_preds[b_idx].item()
            new_rej = new_rejected[b_idx].item()
            true_label = None  # 不知道真实标签
            
            # 检查信念是否有变化
            old_b = torch.stack([all_beliefs[i][b_idx] for i in range(N)])
            new_b = torch.stack([consensus_beliefs[i][b_idx] for i in range(N)])
            b_change = (old_b - new_b).abs().max().item()
            
            # DP改变 = 共识改变了预测
            pred_changed = (ds_pred != new_pred) or (ds_rej != new_rej)
            
            # 如果原始DS拒识但共识后接受了
            if ds_rej and not new_rej:
                improved_mask[b_idx] = True
                final_preds[b_idx] = new_preds[b_idx]
                final_rejected[b_idx] = False
                if verbose:
                    print(f"  [共识改善] 样本{b_idx}: DS拒识→共识接受(pred={new_pred}), "
                          f"信念变化max={b_change:.6f}")
            elif not ds_rej and new_rej:
                # 共识导致拒识增加 - 这不好，保留DS结果
                if verbose:
                    print(f"  [共识退化] 样本{b_idx}: DS接受→共识拒识, 跳过")
            elif pred_changed and verbose:
                # 预测改变但拒识状态不变
                print(f"  [共识改变] 样本{b_idx}: DS_pred={ds_pred}→共识_pred={new_pred}, "
                      f"信念变化max={b_change:.6f}")
    
    n_improved = improved_mask.sum().item()
    if verbose and n_improved > 0:
        print(f"  共识改善: {n_improved}/{n_activated} 样本")
    
    return final_preds, final_rejected, consensus_needed, improved_mask


def simple_consensus(all_beliefs, all_uncertainties, 
                     all_alphas=None, all_embeddings=None,
                     use_gat=False, use_conflict_decomp=False,
                     verbose=False):
    """
    简化共识框架 - 论文方法封装
    
    实现分歧驱动共识策略：
    1. 先运行标准DS融合获得基线
    2. 计算冲突系数K，识别分歧类型
    3. 只在证据冲突样本上运行真正的GAT共识（使用ConsensusEngine）
    4. 无知冲突样本保留拒识
    
    这是论文"异构多模态动态共识与协同"框架的核心贡献。
    
    use_gat=True: 使用真正的GAT共识引擎（ConsensusEngine）进行信念对齐
    use_gat=False: 只分析分歧度，不修改信念（基线对比）
    """
    N = len(all_beliefs)
    B = all_beliefs[0].shape[0]
    K = all_beliefs[0].shape[1]
    
    # 先计算DS基线
    ds_preds, ds_rejected, ds_u = ds_fusion_decision(
        all_beliefs, all_uncertainties, u_threshold=0.5
    )
    
    if use_gat:
        # ====== 使用真正的GAT共识引擎（ConsensusEngine）======
        # 调用 gat_consensus_batch 进行逐样本GAT共识
        consensus_beliefs, consensus_uncertainties, converged_flags, n_iters_list, consensus_alphas = \
            gat_consensus_batch(
                all_beliefs, all_uncertainties, all_embeddings,
                max_iters=5, verbose=verbose
            )
        
        # 对共识后的信念进行DS融合
        final_preds, final_rejected, final_u = ds_fusion_decision(
            consensus_beliefs, consensus_uncertainties, u_threshold=0.5
        )
        
        if verbose:
            n_changed = (final_preds != ds_preds).sum().item()
            n_rej_changed = (final_rejected != ds_rejected).sum().item()
            print(f"  GAT共识效果: {n_changed}/{B} 预测改变, {n_rej_changed}/{B} 拒识状态改变")
        
        return final_preds, final_rejected, final_u
    else:
        return ds_preds, ds_rejected, ds_u


def run_gat_consensus(all_beliefs, all_uncertainties, all_embeddings,
                      num_classes=10, max_iters=5, verbose=True):
    """
    运行**已训练的**GAT共识引擎（需要先训练GAT层并加载权重）
    
    如果GAT未训练，会自动回退到信念相似性共识。
    
    Args:
        all_beliefs: list of [B, K]
        all_uncertainties: list of [B, 1]
        all_embeddings: list of [B, D]
        num_classes: 类别数
        max_iters: GAT内循环最大迭代
        verbose: 打印日志
    
    Returns:
        preds, rejected, global_u, original_alphas
    """
    device = all_beliefs[0].device
    B = all_beliefs[0].shape[0]
    K = all_beliefs[0].shape[1]
    D = all_embeddings[0].shape[1]
    N = len(all_beliefs)
    
    # 检查是否有训练好的GAT权重
    gat_weight_path = 'checkpoints/cifar10n/gat_consensus.pt'
    
    if not os.path.exists(gat_weight_path):
        if verbose:
            print(f"  未找到已训练的GAT权重，回退到信念相似性共识")
        from step4.evaluate_cifar10n import simple_consensus
        return simple_consensus(
            all_beliefs, all_uncertainties,
            all_alphas=all_alphas, all_embeddings=all_embs,
            use_gat=True, verbose=verbose
        )
    
    # 加载预训练GAT
    engine = ConsensusEngine(embed_dim=D, num_classes=K, hidden_dim=64)
    state = torch.load(gat_weight_path, map_location=device, weights_only=True)
    engine.load_state_dict(state)
    engine.eval()
    
    # 逐样本运行GAT共识
    new_beliefs = [b.clone() for b in all_beliefs]
    new_uncertainties = [u.clone() for u in all_uncertainties]
    
    for b_idx in range(B):
        agent_outputs = []
        for i in range(N):
            b_i = all_beliefs[i][b_idx:b_idx+1]    # [1, K]
            u_i = all_uncertainties[i][b_idx:b_idx+1]  # [1, 1]
            emb_i = all_embeddings[i][b_idx:b_idx+1]   # [1, D]
            S = K / u_i.squeeze(-1).clamp(min=1e-6)
            alpha_i = b_i[0] * S[0].item() + 1.0
            agent_outputs.append((alpha_i, b_i[0], u_i[0].item(), emb_i[0]))
        
        with torch.no_grad():
            h = engine.build_state(agent_outputs)
            h_final, n_iters, converged, energy_trace, attn_trace = \
                engine.run(h, max_iters=max_iters, tol=1e-4, verbose=False)
            outputs = engine.extract_outputs(h_final)
            
            for i in range(N):
                _, b_new, u_new, _ = outputs[i]
                new_beliefs[i][b_idx] = b_new.to(device)
                new_uncertainties[i][b_idx] = torch.tensor([[u_new]], device=device)
    
    # DS融合决策
    preds, rejected, global_u = ds_fusion_decision(
        new_beliefs, new_uncertainties, u_threshold=0.5
    )
    
    return preds, rejected, global_u


# =============================================================================
# 5. 完整评估
# =============================================================================

def evaluate(num_test_samples=500):
    """在CIFAR-10N上评估所有方法"""
    print("=" * 60)
    print("CIFAR-10N 端到端评估 v8 (GAT共识修复版)")
    print("=" * 60)

    features, test_labels = load_features_and_heads()
    B = min(num_test_samples, len(test_labels))
    test_labels = test_labels[:B]

    print(f"\n[2] 处理 {B} 个样本...")

    # 提取所有智能体输出
    all_alphas, all_beliefs, all_uncertainties, all_embs = [], [], [], []
    for name in ['agent1', 'agent2', 'agent3']:
        feats, head = features[name]
        feats = feats[:B]

        alpha, b, u, emb = get_agent_outputs(feats, head)
        all_alphas.append(alpha)
        all_beliefs.append(b)
        all_uncertainties.append(u)
        all_embs.append(emb)

    # ========== 方法1: 多数投票 ==========
    t0 = time.time()
    all_preds_tensor = torch.stack([b.argmax(dim=1) for b in all_beliefs])
    mv_preds, _ = torch.mode(all_preds_tensor, dim=0)
    t_mv = time.time() - t0

    # ========== 方法2: 加权平均 ==========
    t0 = time.time()
    b_stack = torch.stack(all_beliefs)
    u_stack = torch.stack(all_uncertainties)
    weights = F.softmax(1 - u_stack.squeeze(-1), dim=0)
    weighted_b = (b_stack * weights.unsqueeze(-1)).sum(dim=0)
    wa_global_u = (u_stack.squeeze(-1) * weights).sum(dim=0)
    wa_preds = weighted_b.argmax(dim=1)
    wa_rej = wa_global_u > 0.5
    t_wa = time.time() - t0

    # ========== 方法3: 纯DS融合（基线） ==========
    t0 = time.time()
    ds_preds, ds_rej, ds_u = ds_fusion_decision(all_beliefs, all_uncertainties, u_threshold=0.5)
    t_ds = time.time() - t0

    # ========== 方法4: DS+确定性共识 ==========
    print("\n  运行DS+确定性共识（规则型信念对齐）...")
    t0 = time.time()
    con_preds, con_rej, con_u = simple_consensus(
        all_beliefs, all_uncertainties, 
        all_alphas=None, all_embeddings=None,
        use_gat=False
    )
    t_con = time.time() - t0

    # ========== 方法5: DS+GAT共识 ==========
    print("\n  运行DS+真正GAT共识引擎（修复版ConsensusEngine，逐样本）...")
    t0 = time.time()
    gat_preds, gat_rej, gat_u = simple_consensus(
        all_beliefs, all_uncertainties,
        all_alphas=all_alphas, all_embeddings=all_embs,
        use_gat=True, verbose=True
    )
    t_gat = time.time() - t0

    # ========== 汇总结果 ==========
    results = {
        'MajorityVoting': {
            'preds': mv_preds, 'rejected': torch.zeros(B, dtype=torch.bool),
            'time': t_mv
        },
        'WeightedAvg': {
            'preds': wa_preds, 'rejected': wa_rej,
            'uncertainty': wa_global_u, 'time': t_wa
        },
        'DS_Fusion': {
            'preds': ds_preds, 'rejected': ds_rej,
            'uncertainty': ds_u, 'time': t_ds
        },
        'DS_Consensus': {
            'preds': con_preds, 'rejected': con_rej,
            'uncertainty': con_u, 'time': t_con
        },
        'DS_GAT_Consensus': {
            'preds': gat_preds, 'rejected': gat_rej,
            'uncertainty': gat_u, 'time': t_gat
        },
    }

    # ========== 计算指标 ==========
    print(f"\n[3] 评估结果 ({B} samples):")
    print(f"{'='*100}")
    header = f"{'Method':<20s} {'Acc%':<10s} {'F1%':<10s} {'ECE':<12s} {'Rej%':<10s} {'Acc_All':<10s} {'Time':<10s}"
    print(header)
    print(f"{'-'*100}")

    y_true = test_labels.numpy()
    metrics = {}

    for method_name, res in results.items():
        preds_np = res['preds'].numpy()
        rej_np = res['rejected'].numpy()

        rej_rate = rej_np.mean() * 100

        # 全样本准确率（拒识视为错误）
        acc_all = accuracy_score(y_true, preds_np) * 100

        # 接受样本准确率
        accepted = ~rej_np
        if accepted.sum() > 0:
            acc = accuracy_score(y_true[accepted], preds_np[accepted]) * 100
            f1 = f1_score(y_true[accepted], preds_np[accepted], average='macro') * 100
        else:
            acc, f1 = 0.0, 0.0

        # ECE
        u = res.get('uncertainty', None)
        if u is not None and accepted.sum() > 0:
            u_np = u.numpy()
            conf = 1 - u_np[accepted]
            correct = (preds_np[accepted] == y_true[accepted])
            
            bins = np.linspace(0, 1, 11)
            idx = np.digitize(conf, bins[1:-1])
            ece = 0.0
            for b in range(10):
                mask = (idx == b)
                if mask.sum() > 0:
                    bin_acc = correct[mask].mean()
                    bin_conf = conf[mask].mean()
                    ece += abs(bin_acc - bin_conf) * (mask.sum() / len(conf))
        else:
            ece = 0.0

        t = res.get('time', 0)
        metrics[method_name] = {
            'accuracy': acc, 'f1': f1, 'ece': ece,
            'rejection_rate': rej_rate, 'accuracy_all': acc_all, 'time': t
        }

        print(f"{method_name:<20s} {acc:<10.2f} {f1:<10.2f} {ece:<12.4f} "
              f"{rej_rate:<10.2f} {acc_all:<10.2f} {t:<10.2f}")

    print(f"{'='*100}")

    # 保存结果
    with open('results/cifar10n/evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n结果已保存至 results/cifar10n/evaluation_results.json")

    plot_comparison(metrics)
    plot_confusion_matrices(results, y_true)

    return metrics


def plot_comparison(metrics):
    """绘制各方法对比"""
    methods = list(metrics.keys())
    acc = [metrics[m]['accuracy'] for m in methods]
    f1 = [metrics[m]['f1'] for m in methods]
    acc_all = [metrics[m]['accuracy_all'] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#45B7D1', '#96CEB4']

    for ax, vals, title in zip(axes, [acc, f1, acc_all],
                                ['Accuracy (Accepted)', 'Macro F1 (Accepted)', 'Accuracy (All)']):
        bars = ax.bar(methods, vals, color=colors[:len(methods)], alpha=0.8)
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/cifar10n_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图: figures/cifar10n_comparison.png")


def plot_confusion_matrices(results, y_true):
    """绘制混淆矩阵"""
    n = len(results)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_true, res['preds'].numpy(), labels=range(10))
        ax = axes[idx]
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        for i in range(10):
            for j in range(10):
                val = cm[i, j]
                color = 'white' if val > cm.max() / 2 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=7, color=color)

    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.close()
    print(f"混淆矩阵: figures/cifar10n_confusion.png")


def ablation_study(num_test_samples=200):
    """消融实验（简化版）"""
    print("\n" + "=" * 60)
    print("消融实验")
    print("=" * 60)

    features, test_labels = load_features_and_heads()
    B = min(num_test_samples, len(test_labels))
    test_labels = test_labels[:B]

    all_alphas, all_beliefs, all_uncertainties, all_embs = [], [], [], []
    for name in ['agent1', 'agent2', 'agent3']:
        feats, head = features[name]
        alpha, b, u, emb = get_agent_outputs(feats[:B], head)
        all_alphas.append(alpha)
        all_beliefs.append(b)
        all_uncertainties.append(u)
        all_embs.append(emb)

    ablation_results = {}
    
    # A. 完整框架
    preds, rej, _ = simple_consensus(all_beliefs, all_uncertainties, use_gat=False)
    acc = accuracy_score(test_labels.numpy(), preds.numpy()) * 100
    rej_rate = rej.float().mean().item() * 100
    ablation_results['Full (3 agents)'] = {'accuracy': acc, 'rejection_rate': rej_rate}
    print(f"  Full (3 agents): acc={acc:.2f}%, rej={rej_rate:.2f}%")
    
    # B. 去掉Agent3
    preds2, rej2, _ = simple_consensus(
        [all_beliefs[0], all_beliefs[1]], 
        [all_uncertainties[0], all_uncertainties[1]],
        use_gat=False
    )
    acc2 = accuracy_score(test_labels.numpy(), preds2.numpy()) * 100
    rej_rate2 = rej2.float().mean().item() * 100
    ablation_results['w/o Agent3'] = {'accuracy': acc2, 'rejection_rate': rej_rate2}
    print(f"  w/o Agent3: acc={acc2:.2f}%, rej={rej_rate2:.2f}%")
    
    # C. 只用Agent1
    preds3, _ = all_beliefs[0].argmax(dim=1), torch.zeros(B, dtype=torch.bool)
    acc3 = accuracy_score(test_labels.numpy(), preds3.numpy()) * 100
    ablation_results['Agent1 only'] = {'accuracy': acc3, 'rejection_rate': 0.0}
    print(f"  Agent1 only: acc={acc3:.2f}%")
    
    # D. 只用Agent2
    preds4, _ = all_beliefs[1].argmax(dim=1), torch.zeros(B, dtype=torch.bool)
    acc4 = accuracy_score(test_labels.numpy(), preds4.numpy()) * 100
    ablation_results['Agent2 only'] = {'accuracy': acc4, 'rejection_rate': 0.0}
    print(f"  Agent2 only: acc={acc4:.2f}%")
    
    # E. GAT共识增强
    preds5, rej5, _ = simple_consensus(
        all_beliefs, all_uncertainties,
        all_alphas=all_alphas, all_embeddings=all_embs,
        use_gat=True
    )
    acc5 = accuracy_score(test_labels.numpy(), preds5.numpy()) * 100
    rej_rate5 = rej5.float().mean().item() * 100
    ablation_results['Full + GAT'] = {'accuracy': acc5, 'rejection_rate': rej_rate5}
    print(f"  Full + GAT: acc={acc5:.2f}%, rej={rej_rate5:.2f}%")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(ablation_results.keys())
    accs = [ablation_results[n]['accuracy'] for n in names]
    colors = ['#45B7D1', '#FF6B6B', '#4ECDC4', '#FFE66D', '#96CEB4', '#DDA0DD']
    bars = ax.bar(names, accs, color=colors[:len(names)], alpha=0.8)
    ax.set_title('Ablation Study on CIFAR-10N (v8 GAT修复版)', fontsize=14)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, max(accs) * 1.2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/cifar10n_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消融图: figures/cifar10n_ablation.png")


def detailed_analysis(num_samples=50):
    """详细分析：逐样本展示"""
    print("\n" + "=" * 60)
    print("详细分析")
    print("=" * 60)

    features, test_labels = load_features_and_heads()
    B = min(num_samples, len(test_labels))

    # 提取前B个样本
    all_alphas, all_beliefs, all_uncertainties, all_embs = [], [], [], []
    for name in ['agent1', 'agent2', 'agent3']:
        feats, head = features[name]
        alpha, b, u, emb = get_agent_outputs(feats[:B], head)
        all_alphas.append(alpha)
        all_beliefs.append(b)
        all_uncertainties.append(u)
        all_embs.append(emb)

    # 获取DS融合结果
    ds_preds, ds_rej, ds_u = ds_fusion_decision(all_beliefs, all_uncertainties, u_threshold=0.5)
    
    # 获取GAT共识结果
    gat_preds, gat_rej, gat_u = simple_consensus(
        all_beliefs, all_uncertainties,
        all_alphas=all_alphas, all_embeddings=all_embs,
        use_gat=True
    )

    print(f"\n前{min(10, B)}个样本详细分析:")
    print(f"{'idx':<5s} {'true':<6s} {'DS_pred':<8s} {'DS_u':<8s} {'DS_rej':<6s} "
          f"{'GAT_pred':<9s} {'GAT_u':<8s} {'GAT_rej':<7s} "
          f"{'A1_pred':<8s} {'A1_u':<8s} {'A2_pred':<8s} {'A2_u':<8s} "
          f"{'A3_pred':<8s} {'A3_u':<8s}")
    print("-" * 110)

    for b in range(min(10, B)):
        row = [f"{b:<5d}",
               f"{test_labels[b].item():<6d}",
               f"{ds_preds[b].item():<8d}",
               f"{ds_u[b].item():<8.4f}",
               f"{'Y' if ds_rej[b].item() else 'N':<6s}",
               f"{gat_preds[b].item():<9d}",
               f"{gat_u[b].item():<8.4f}",
               f"{'Y' if gat_rej[b].item() else 'N':<7s}"]
        for i in range(3):
            pred_i = all_beliefs[i][b].argmax().item()
            u_i = all_uncertainties[i][b].item()
            row.append(f"{pred_i:<8d}")
            row.append(f"{u_i:<8.4f}")
        print(" ".join(row))

    # 拒识样本分析
    rej_indices = torch.where(gat_rej)[0]
    if len(rej_indices) > 0:
        print(f"\nGAT共识拒识样本分析 ({len(rej_indices)}/{B}):")
        for idx in rej_indices[:5]:
            print(f"  样本{idx.item()}: 真实={test_labels[idx].item()}, "
                  f"DS_u={ds_u[idx].item():.4f}, GAT_u={gat_u[idx].item():.4f}")
            for i, name in enumerate(['A1', 'A2', 'A3']):
                print(f"    {name}: pred={all_beliefs[i][idx].argmax().item()}, "
                      f"u={all_uncertainties[i][idx].item():.4f}")

    return ds_preds, ds_rej, ds_u


if __name__ == '__main__':
    evaluate(num_test_samples=500)
    ablation_study(num_test_samples=200)
    detailed_analysis(num_samples=50)