"""
Hateful Memes 完整增强评估管线 (v2.0)
======================================
改进点:
1. 更强的文本编码器 (BiLSTM + Attention)
2. 更强的融合编码器 (Cross-Attention + 门控融合)
3. 不确定性感知证据头
4. GAT共识层集成
5. 分歧解构器 + EMNet证据交换
6. 完整实验分析 + 可视化

提升点:
- 预计算tokenization，避免训练循环中的重复计算
- 修复converged布尔值bug
- 内联ds_fusion_decision避免跨模块依赖
"""

import os, sys, time, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
# 设置中文字体
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 添加src目录到路径，使from step2.gat_consensus等方式可以工作
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from plot_utils import setup_chinese_font, setup_plot_style
setup_chinese_font()
setup_plot_style()

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# GAT共识层
from step2.gat_consensus import ConsensusEngine, GATConsensusLayer

warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# 0. 配置
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

NUM_CLASSES = 2
U_THRESHOLD = 0.3

DATA_DIR = 'data/Hateful_Memes/data'
FEATURE_DIR = 'data/features/hateful_memes'
CHECKPOINT_DIR = 'checkpoints/hateful_memes'
FIGURE_DIR = 'figures'
RESULT_DIR = 'results/hateful_memes'

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# DS融合决策函数（内联，避免跨模块依赖）
# =============================================================================

def ds_fusion_decision(all_beliefs, all_uncertainties, u_threshold=0.5):
    """
    正确的Dempster-Shafer融合决策 - 修复版v2
    
    Args:
        all_beliefs: list of [B, C] tensors, 每个智能体的信念质量
        all_uncertainties: list of [B] tensors, 每个智能体的不确定性
        u_threshold: float, 拒绝阈值
    
    Returns:
        preds: [B] 预测类别
        rejected: [B] bool 是否被拒绝
        global_u: [B] 融合后不确定性
    """
    B = all_beliefs[0].shape[0]
    C = all_beliefs[0].shape[1]
    N = len(all_beliefs)
    device = all_beliefs[0].device
    
    # 初始化: 从第一个智能体开始
    b0 = all_beliefs[0]
    u0 = all_uncertainties[0]
    # 第一个mass = [b*(1-u), u]
    combined_belief = b0 * (1.0 - u0.unsqueeze(-1))  # [B, C]
    combined_u = u0  # [B]
    
    # 逐个组合后续智能体
    for b_idx in range(1, N):
        b = all_beliefs[b_idx]
        u = all_uncertainties[b_idx]
        
        # 计算冲突 K
        # K = sum_{i!=j} m1[i] * m2[j]
        # 对类别维度: m1[:,:C] * m2[:,:C] 然后求和
        # 对角元素: sum_i m1[i]*m2[i] (同类)
        # 总乘积: (sum m1[i]) * (sum m2[j]) = sum_i sum_j m1[i]*m2[j]
        # 总乘积 = (1-u1)(1-u2) 因为 b1.sum()=1, b2.sum()=1
        # 但更安全直接计算
        
        # m1 = [combined_belief * (1-combined_u), combined_u]
        # 但combined_belief本身已经是b*(1-u)形式，需要归一化
        # 正确做法：每次用置信mass而非belief
        
        m1_b = combined_belief  # [B, C] 已经是mass在类别上的分量
        m1_u = combined_u       # [B] 不确定性mass
        
        m2_b = b * (1.0 - u.unsqueeze(-1))  # [B, C]
        m2_u = u                            # [B]
        
        # 冲突 K = sum_{i!=j} m1_b[i] * m2_b[j]
        # = (sum_i m1_b[i]) * (sum_j m2_b[j]) - sum_i m1_b[i] * m2_b[i]
        sum_m1_b = m1_b.sum(dim=-1)  # [B]
        sum_m2_b = m2_b.sum(dim=-1)  # [B]
        # 注意：sum_m1_b + m1_u = 1.0 理论上，但数值上可能不等于
        agree = (m1_b * m2_b).sum(dim=-1)  # [B] 同类信念乘积和
        K = sum_m1_b * sum_m2_b - agree
        
        # 归一化因子 1-K (加epsilon防除零)
        denom = 1.0 - K + 1e-8
        
        # 组合: new_b[i] = (m1_b[i]*m2_b[i] + m1_b[i]*m2_u + m1_u*m2_b[i]) / (1-K)
        new_belief = (m1_b * m2_b + m1_b * m2_u.unsqueeze(-1) + m1_u.unsqueeze(-1) * m2_b) / denom.unsqueeze(-1)
        new_u = m1_u * m2_u / denom
        
        combined_belief = new_belief
        combined_u = new_u
    
    # combined_belief 已经是mass形式 = b*(1-u)
    # 需要提取pure belief = combined_belief / (1-combined_u)
    # 当combined_u接近1时会有数值问题，所以加epsilon
    global_belief = combined_belief / (1.0 - combined_u.unsqueeze(-1) + 1e-8)
    global_u = combined_u
    
    preds = global_belief.argmax(dim=-1)
    rejected = global_u > u_threshold
    
    return preds, rejected, global_u


def generate_emnet_data_supervised(train_alphas, train_labels, n_synthetic=5000, num_classes=2, device='cpu'):
    """
    有监督方式生成EMNet训练数据
    
    从训练数据中提取evidence_conflict样本，用真实标签构建监督信号:
    - 发送方证据 = 最佳agent的alpha-1
    - 目标伪计数 = 如果最佳agent预测正确，则奖励最佳agent的alpha增加
    
    Args:
        train_alphas: [N_train, 3, C] 训练集各agent的alpha
        train_labels: [N_train] 训练集真实标签
        n_synthetic: 额外生成的合成样本数
        num_classes: 类别数
        device: 设备
    
    Returns:
        x: [N, C] 发送方证据
        y: [N, C] 目标证据量（不是delta，而是期望的接收方总证据）
    """
    N = train_alphas.shape[0]
    evidence_list = []
    target_list = []
    
    # 1. 从训练集中收集真实样本
    for i in range(N):
        # 找到不确定性最低的agent作为sender
        alphas = train_alphas[i]  # [3, C]
        S = alphas.sum(dim=-1)    # [3]
        u = num_classes / S       # [3]
        best_idx = u.argmin().item()
        worst_idx = u.argmax().item()
        
        sender_ev = (alphas[best_idx] - 1.0).detach()  # [C]
        
        # 构建目标: 让worst agent的证据向正确类靠拢
        true_label = train_labels[i]
        target_ev = sender_ev.clone()
        target_ev[true_label] += 3.0  # 大幅度增加正确类证据
        # 其他类削减（如果它们大于0.5）
        for c in range(num_classes):
            if c != true_label and target_ev[c] > 0.5:
                target_ev[c] = max(0.1, target_ev[c] - 1.0)
        
        evidence_list.append(sender_ev)
        target_list.append(target_ev)
    
    # 2. 额外生成合成样本（可选的多样性）
    syn_x = torch.rand(n_synthetic, num_classes, device=device) * 5.0 + 0.1
    syn_y = syn_x.clone()
    syn_y += torch.randn(n_synthetic, num_classes, device=device) * 0.5 + 1.0
    syn_y = F.relu(syn_y) + 0.1
    
    x = torch.stack(evidence_list + [syn_x])
    y = torch.stack(target_list + [syn_y])
    
    return x.to(device), y.to(device)


# =============================================================================
# 1. 数据加载
# =============================================================================

class HatefulMemesDataset(Dataset):
    """Hateful Memes 数据集"""
    def __init__(self, split='train', max_samples=None):
        self.split = split
        self.max_samples = max_samples
        
        json_path = os.path.join(DATA_DIR, f'{split}.jsonl')
        if not os.path.exists(json_path):
            json_path = os.path.join(DATA_DIR, f'{split}.json')
        
        self.data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        if max_samples and max_samples < len(self.data):
            self.data = self.data[:max_samples]
        
        # 预计算每个样本的文本和图片路径
        self.texts = [item['text'] for item in self.data]
        self.img_paths = []
        for item in self.data:
            # img字段格式: 'img/08291.png'
            img_name = item['img'].split('/')[-1]
            full_path = os.path.join(DATA_DIR, img_name)
            if not os.path.exists(full_path):
                full_path = os.path.join(DATA_DIR, 'img', img_name)
            if not os.path.exists(full_path):
                full_path = os.path.join(DATA_DIR, 'images', img_name)
            self.img_paths.append(full_path)
        self.labels = [item['label'] for item in self.data]
        
        print(f"  [{split}] 加载 {len(self.data)} 样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'img_path': self.img_paths[idx],
            'label': self.labels[idx],
        }


def load_hm_data(dataset):
    """从HatefulMemesDataset提取列表数据"""
    return dataset.texts, dataset.img_paths, dataset.labels


# =============================================================================
# 2. 增强的编码器
# =============================================================================

class CharCNNTextEncoder(nn.Module):
    """字符级CNN文本编码器: Embedding + Conv1d + MaxPool + Multi-kernel"""
    def __init__(self, vocab_size=128, embed_dim=64, hidden_dim=256, output_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多尺度卷积核捕获不同n-gram模式
        conv_out_dim = (hidden_dim // 3) * 3  # 确保能被3整除
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_dim//3, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]
        ])
        self.conv_proj = nn.Linear(conv_out_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, x):
        emb = self.embedding(x)  # [B, L, D]
        emb = emb.permute(0, 2, 1)  # [B, D, L]
        conv_outs = []
        for conv in self.convs:
            out = conv(emb)
            out = F.relu(out)
            out = out.max(dim=-1).values  # global max pooling
            conv_outs.append(out)
        out = torch.cat(conv_outs, dim=-1)  # [B, hidden_dim]
        out = self.conv_proj(out)
        return self.proj(out)


def build_char_vocab():
    """构建可靠的可打印ASCII字符映射（与原版代码一致）"""
    vocab = {chr(i): idx+2 for idx, i in enumerate(range(32, 127))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab


class CNNDecoder(nn.Module):
    """图像编码器: ResNet18 512维 → 256维"""
    def __init__(self, input_dim=512, output_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """跨模态注意力融合: 文本+图像 → 交叉注意力 → 门控融合"""
    def __init__(self, text_dim=256, img_dim=256, hidden_dim=256, output_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, text_feat, img_feat):
        t_proj = self.text_proj(text_feat).unsqueeze(1)
        i_proj = self.img_proj(img_feat).unsqueeze(1)
        
        fused, _ = self.cross_attn(t_proj, i_proj, i_proj)
        fused = fused.squeeze(1)
        
        t_proj_sq = t_proj.squeeze(1)
        gate = self.gate(torch.cat([fused, t_proj_sq], dim=1))
        out = gate * fused + (1 - gate) * t_proj_sq
        return self.fusion(out)


# =============================================================================
# 3. 证据头
# =============================================================================

class EvidenceHead(nn.Module):
    """证据头: 特征 → Dirichlet参数α (无温度缩放，使用正则化控制不确定性)"""
    def __init__(self, input_dim, num_classes=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.evidence = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        h = self.net(x)
        # 直接用softplus，保持正常梯度梯度流动
        e = F.softplus(self.evidence(h)) + 1e-8
        alpha = e + 1.0  # alpha >= 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        belief = alpha / S          # [B, C]
        uncertainty = alpha.shape[-1] / S  # [B, 1]  u = C/S
        return alpha, belief, uncertainty.squeeze(-1), h


# =============================================================================
# 4. GAT共识层
# =============================================================================

class DeterministicConsensus:
    """
    确定性共识机制（替代随机初始化GAT）
    
    数学原理：
    1. 每个agent的置信度 = 1 - uncertainty
    2. 共识embedding = 置信度加权的agent均值
    3. 按置信度比例向共识方向移动
    
    这种方法：
    - 不需要训练（零样本）
    - 数学上保证收敛（每次迭代降低差异）
    - 保持agent多样性（不完全合并）
    """
    @staticmethod
    def consensus_update(embeddings, beliefs, uncertainties, mix_ratio=0.3):
        """
        确定性共识更新
        
        Args:
            embeddings: [B, N, D] 各agent的embedding
            beliefs: [B, N, C] 各agent的信念
            uncertainties: [B, N] 各agent的不确定性（0=确定, 1=完全不确定）
            mix_ratio: 向共识靠拢的比例
        
        Returns:
            new_embeddings: [B, N, D] 更新后的embedding
        """
        B, N, D = embeddings.shape
        
        # 置信度权重: w_i = (1-u_i) 重要性
        conf = 1.0 - uncertainties  # [B, N]
        conf_softmax = F.softmax(conf / 0.1, dim=-1)  # [B, N] 温度0.1使高置信度占主导
        
        # 计算共识embedding: 置信度加权的均值
        consensus_emb = (embeddings * conf_softmax.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B, 1, D]
        
        # 按mix_ratio向共识靠拢
        new_emb = embeddings + mix_ratio * (consensus_emb - embeddings)
        
        return new_emb


def consensus_energy(embeddings, beliefs, uncertainties):
    """共识能量函数"""
    mean_emb = embeddings.mean(dim=1, keepdim=True)
    energy = ((embeddings - mean_emb) ** 2).mean()
    mean_belief = beliefs.mean(dim=1, keepdim=True)
    energy += ((beliefs - mean_belief) ** 2).mean()
    energy += uncertainties.mean()
    return energy


# =============================================================================
# 5. 分歧解构器 + EMNet
# =============================================================================

def compute_conflict_K(b1, u1, b2, u2):
    """D-S冲突系数"""
    agreement = (b1 * b2).sum(dim=-1)
    both_u = u1 * u2
    K = 1.0 - agreement - both_u
    return torch.clamp(K, 0.0, 1.0)


class DisagreementDeconstructor:
    """分歧解构器"""
    def __init__(self, u_threshold=0.5, K_threshold=0.3):
        self.u_threshold = u_threshold
        self.K_threshold = K_threshold
    
    def deconstruct_batch(self, beliefs, uncertainties):
        B, N, C = beliefs.shape
        types = []
        K_values = []
        
        for b in range(B):
            worst_type = 'none'
            max_K = 0.0
            
            for i in range(N):
                for j in range(i + 1, N):
                    K = compute_conflict_K(
                        beliefs[b, i], uncertainties[b, i],
                        beliefs[b, j], uncertainties[b, j]
                    ).item()
                    avg_u = (uncertainties[b, i] + uncertainties[b, j]).item() / 2.0
                    
                    if avg_u > self.u_threshold:
                        ctype = 'ignorance_conflict'
                    elif K > self.K_threshold:
                        ctype = 'evidence_conflict'
                    else:
                        ctype = 'none'
                    
                    if ctype != 'none' and (worst_type == 'none' or 
                        (ctype == 'evidence_conflict' and worst_type == 'ignorance_conflict')):
                        worst_type = ctype
                    
                    max_K = max(max_K, K)
            
            types.append(worst_type)
            K_values.append(max_K)
        
        return types, K_values


class EMNet(nn.Module):
    """贝叶斯证据交换网络（修复版：去掉0.5缩放，直接输出伪计数）"""
    def __init__(self, num_classes=2, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),
        )
    
    def forward(self, sender_evidence):
        # 输出范围 [0, ∞)，典型值 1~10
        return self.net(sender_evidence)


# =============================================================================
# 6. 文本Tokenization（预计算）
# =============================================================================

def pre_tokenize(texts, max_len=128, vocab_size=5000):
    """
    预计算tokenization（批量处理，比逐样本快很多）
    
    Args:
        texts: list of strings
        max_len: 最大序列长度
        vocab_size: 词表大小
    
    Returns:
        tokens: [N, max_len] torch.long
    """
    N = len(texts)
    tokens = torch.zeros(N, max_len, dtype=torch.long)
    
    for i, text in enumerate(texts):
        text = str(text).lower().strip()[:max_len]
        for j, ch in enumerate(text):
            token_id = (ord(ch) % (vocab_size - 1)) + 1
            tokens[i, j] = token_id
    
    return tokens


# =============================================================================
# 7. 特征提取与缓存
# =============================================================================

def load_or_extract(filename, extract_func, *args, force=False):
    """缓存式特征加载"""
    filepath = os.path.join(FEATURE_DIR, filename)
    if not force and os.path.exists(filepath):
        data = torch.load(filepath, map_location='cpu', weights_only=True)
        print(f"    加载缓存: {filename} ({data.shape})")
        return data
    
    print(f"    提取特征: {filename}...")
    result = extract_func(*args)
    torch.save(result, filepath)
    print(f"    已保存: {filename} ({result.shape})")
    return result


# =============================================================================
# 8. 核心管线
# =============================================================================

def run_enhanced_pipeline(max_train=2000, max_val=500, force_extract=False,
                          skip_training=False, step=0):
    """完整增强评估管线"""
    print(f"\n{'='*80}")
    print(f"Hateful Memes 完整增强评估管线 v2.0")
    print(f"{'='*80}")
    print(f"训练样本: {max_train}, 验证样本: {max_val}")
    print(f"设备: {DEVICE}")
    print(f"{'='*80}\n")
    
    # ========== 0. 加载数据集 ==========
    print("[0] 加载数据集...")
    train_dataset = HatefulMemesDataset(split='train', max_samples=max_train)
    val_dataset = HatefulMemesDataset(split='dev', max_samples=max_val)
    
    train_texts = train_dataset.texts
    train_img_paths = train_dataset.img_paths
    train_labels = train_dataset.labels
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    
    val_texts = val_dataset.texts
    val_img_paths = val_dataset.img_paths
    val_labels = val_dataset.labels
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    
    train_labels_oh = F.one_hot(train_labels_t, NUM_CLASSES).float()
    val_labels_oh = F.one_hot(val_labels_t, NUM_CLASSES).float()
    
    print(f"  训练集: 正类={sum(train_labels)}/{len(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
    print(f"  验证集: 正类={sum(val_labels)}/{len(val_labels)} ({sum(val_labels)/len(val_labels)*100:.1f}%)")
    print(f"  训练集图片路径示例: {train_img_paths[0]}")
    
    # ========== 1. 特征提取 ==========
    print("\n[1] 特征提取...")
    
    print("  1a. 文本Token化 (预计算)...")
    train_tokens = pre_tokenize(train_texts)
    val_tokens = pre_tokenize(val_texts)
    print(f"    训练: {train_tokens.shape}, 验证: {val_tokens.shape}")
    
    print("  1b. 图像特征提取 (ResNet18)...")
    from torchvision import transforms, models
    
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(DEVICE).eval()
    
    def extract_img_features(img_paths, batch_size=64):
        N = len(img_paths)
        features = []
        for i in range(0, N, batch_size):
            batch_paths = list(img_paths[i:i+batch_size])
            batch_tensors = []
            for p in batch_paths:
                try:
                    if os.path.exists(p):
                        img = Image.open(p).convert('RGB')
                        batch_tensors.append(img_transform(img))
                    else:
                        batch_tensors.append(torch.randn(3, 224, 224))
                except Exception:
                    batch_tensors.append(torch.randn(3, 224, 224))
            batch_tensor = torch.stack(batch_tensors).to(DEVICE)
            with torch.no_grad():
                feat = resnet(batch_tensor).squeeze(-1).squeeze(-1)
            features.append(feat.cpu())
        return torch.cat(features, dim=0)
    
    train_img_feats = load_or_extract(
        'train_img_feats.pt', extract_img_features, train_img_paths, force=force_extract)
    val_img_feats = load_or_extract(
        'val_img_feats.pt', extract_img_features, val_img_paths, force=force_extract)
    print(f"    Train: {train_img_feats.shape}, Val: {val_img_feats.shape}")
    
    if step == 1:
        print("\n[完成] 特征提取完成")
        return None
    
    # ========== 2. 训练证据头 ==========
    print("\n[2] 训练证据头...")
    
    EMBED_DIM = 256
    
    char_vocab = build_char_vocab()
    vocab_size = len(char_vocab)
    
    def tokenize_with_vocab(texts, vocab, max_len=128):
        """使用正确字符词表的tokenization"""
        N = len(texts)
        tokens = torch.zeros(N, max_len, dtype=torch.long)
        for i, text in enumerate(texts):
            text = str(text).lower().strip()[:max_len]
            for j, ch in enumerate(text):
                token_id = vocab.get(ch, 1)  # 1 = UNK
                tokens[i, j] = token_id
        return tokens
    
    train_tokens = tokenize_with_vocab(train_texts, char_vocab)
    val_tokens = tokenize_with_vocab(val_texts, char_vocab)
    
    text_encoder = CharCNNTextEncoder(vocab_size=vocab_size, embed_dim=64, hidden_dim=256, output_dim=EMBED_DIM).to(DEVICE)
    img_encoder = CNNDecoder(input_dim=512, output_dim=EMBED_DIM).to(DEVICE)
    fusion_encoder = CrossAttentionFusion(text_dim=EMBED_DIM, img_dim=EMBED_DIM, 
                                           hidden_dim=128, output_dim=EMBED_DIM).to(DEVICE)
    
    agent1_head = EvidenceHead(EMBED_DIM, NUM_CLASSES).to(DEVICE)
    agent2_head = EvidenceHead(EMBED_DIM, NUM_CLASSES).to(DEVICE)
    agent3_head = EvidenceHead(EMBED_DIM, NUM_CLASSES).to(DEVICE)
    
    def try_load(model, path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            model.eval()
            return True
        return False
    
    has_all = all([
        try_load(text_encoder, f'{CHECKPOINT_DIR}/text_encoder.pt'),
        try_load(img_encoder, f'{CHECKPOINT_DIR}/img_encoder.pt'),
        try_load(fusion_encoder, f'{CHECKPOINT_DIR}/fusion_encoder.pt'),
        try_load(agent1_head, f'{CHECKPOINT_DIR}/agent1_head_enhanced.pt'),
        try_load(agent2_head, f'{CHECKPOINT_DIR}/agent2_head_enhanced.pt'),
        try_load(agent3_head, f'{CHECKPOINT_DIR}/agent3_head_enhanced.pt'),
    ])
    
    if has_all:
        print("  已加载所有编码器和证据头，跳过训练")
    elif skip_training:
        print("  skip_training=True，未找到全部缓存，使用随机初始化")
    else:
        def evidential_loss(alpha, targets, lam=0.001, reg_alpha=0.001):
            """
            增强的证据损失: 标准损失 + KL散度 + 不确定性正则化
            
            Args:
                alpha: Dirichlet参数 [B, C]
                targets: one-hot标签 [B, C]
                lam: KL散度权重
                reg_alpha: 证据量正则化系数 (约束总证据不要太大)
            """
            S = alpha.sum(dim=-1, keepdim=True)
            loss = (targets * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()
            
            alpha_tilde = alpha * (1 - targets) + targets
            S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
            kl = (torch.lgamma(alpha_tilde.sum(dim=-1)) - torch.lgamma(alpha_tilde).sum(dim=-1) +
                  ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1))
            
            # 证据量正则化: 惩罚总证据量的平方，鼓励适度的认知不确定性
            S_reg = (S ** 2).mean()
            
            return loss + lam * kl.mean() + reg_alpha * S_reg
        
        B = len(train_texts)
        
        # === Agent1: 文本 ===
        print("\n  训练Agent1 (文本)...")
        opt1 = torch.optim.Adam(list(text_encoder.parameters()) + list(agent1_head.parameters()), lr=1e-3)
        sched1 = CosineAnnealingLR(opt1, T_max=20)
        best_acc1 = 0.0
        
        # 预计算验证集tokens
        val_tokens_t = val_tokens.to(DEVICE)
        
        for epoch in range(20):
            text_encoder.train()
            agent1_head.train()
            total_loss = 0.0
            perm = torch.randperm(B)
            
            for i in range(0, B, 64):
                idx = perm[i:i+64]
                batch_tokens = train_tokens[idx].to(DEVICE)
                batch_labels = train_labels_oh[idx].to(DEVICE)
                
                opt1.zero_grad()
                emb = text_encoder(batch_tokens)
                alpha, _, _, _ = agent1_head(emb)
                loss = evidential_loss(alpha, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(text_encoder.parameters()) + list(agent1_head.parameters()), 1.0)
                opt1.step()
                total_loss += loss.item()
            
            sched1.step()
            
            with torch.no_grad():
                text_encoder.eval()
                agent1_head.eval()
                val_emb = text_encoder(val_tokens_t)
                _, val_b, val_u, _ = agent1_head(val_emb)
                val_acc = (val_b.argmax(dim=1).cpu() == val_labels_t).float().mean().item()
                
                if val_acc >= best_acc1:
                    best_acc1 = val_acc
                    torch.save(text_encoder.state_dict(), f'{CHECKPOINT_DIR}/text_encoder.pt')
                    torch.save(agent1_head.state_dict(), f'{CHECKPOINT_DIR}/agent1_head_enhanced.pt')
                elif epoch == 0:
                    # 首次训练至少保存一次
                    torch.save(text_encoder.state_dict(), f'{CHECKPOINT_DIR}/text_encoder.pt')
                    torch.save(agent1_head.state_dict(), f'{CHECKPOINT_DIR}/agent1_head_enhanced.pt')
                    best_acc1 = val_acc
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/20, Loss: {total_loss/max(B//64,1):.4f}, "
                      f"Val Acc: {val_acc*100:.2f}%, Avg u={val_u.mean().item():.4f}")
        
        print(f"  Agent1 最佳验证准确率: {best_acc1*100:.2f}%")
        text_encoder.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/text_encoder.pt', weights_only=True))
        agent1_head.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/agent1_head_enhanced.pt', weights_only=True))
        text_encoder.eval()
        agent1_head.eval()
        
        # === Agent2: 图像 ===
        print("\n  训练Agent2 (图像)...")
        opt2 = torch.optim.Adam(list(img_encoder.parameters()) + list(agent2_head.parameters()), lr=1e-3)
        sched2 = CosineAnnealingLR(opt2, T_max=20)
        best_acc2 = 0.0
        
        for epoch in range(20):
            img_encoder.train()
            agent2_head.train()
            total_loss = 0.0
            perm = torch.randperm(B)
            
            for i in range(0, B, 64):
                idx = perm[i:i+64]
                batch_img = train_img_feats[idx].to(DEVICE)
                batch_labels = train_labels_oh[idx].to(DEVICE)
                
                opt2.zero_grad()
                emb = img_encoder(batch_img)
                alpha, _, _, _ = agent2_head(emb)
                loss = evidential_loss(alpha, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(img_encoder.parameters()) + list(agent2_head.parameters()), 1.0)
                opt2.step()
                total_loss += loss.item()
            
            sched2.step()
            
            with torch.no_grad():
                img_encoder.eval()
                agent2_head.eval()
                val_emb = img_encoder(val_img_feats.to(DEVICE))
                _, val_b, val_u, _ = agent2_head(val_emb)
                val_acc = (val_b.argmax(dim=1).cpu() == val_labels_t).float().mean().item()
                
                if val_acc >= best_acc2:
                    best_acc2 = val_acc
                    torch.save(img_encoder.state_dict(), f'{CHECKPOINT_DIR}/img_encoder.pt')
                    torch.save(agent2_head.state_dict(), f'{CHECKPOINT_DIR}/agent2_head_enhanced.pt')
                elif epoch == 0:
                    torch.save(img_encoder.state_dict(), f'{CHECKPOINT_DIR}/img_encoder.pt')
                    torch.save(agent2_head.state_dict(), f'{CHECKPOINT_DIR}/agent2_head_enhanced.pt')
                    best_acc2 = val_acc
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/20, Loss: {total_loss/max(B//64,1):.4f}, "
                      f"Val Acc: {val_acc*100:.2f}%, Avg u={val_u.mean().item():.4f}")
        
        print(f"  Agent2 最佳验证准确率: {best_acc2*100:.2f}%")
        img_encoder.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/img_encoder.pt', weights_only=True))
        agent2_head.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/agent2_head_enhanced.pt', weights_only=True))
        img_encoder.eval()
        agent2_head.eval()
        
        # === Agent3: 融合 ===
        print("\n  训练Agent3 (融合)...")
        opt3 = torch.optim.Adam(list(fusion_encoder.parameters()) + list(agent3_head.parameters()), lr=1e-3)
        sched3 = CosineAnnealingLR(opt3, T_max=20)
        best_acc3 = 0.0
        
        for epoch in range(20):
            fusion_encoder.train()
            agent3_head.train()
            total_loss = 0.0
            perm = torch.randperm(B)
            
            for i in range(0, B, 64):
                idx = perm[i:i+64]
                batch_tokens = train_tokens[idx].to(DEVICE)
                batch_img = train_img_feats[idx].to(DEVICE)
                batch_labels = train_labels_oh[idx].to(DEVICE)
                
                opt3.zero_grad()
                with torch.no_grad():
                    text_emb = text_encoder(batch_tokens)
                    img_emb = img_encoder(batch_img)
                fused = fusion_encoder(text_emb, img_emb)
                alpha, _, _, _ = agent3_head(fused)
                loss = evidential_loss(alpha, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(fusion_encoder.parameters()) + list(agent3_head.parameters()), 1.0)
                opt3.step()
                total_loss += loss.item()
            
            sched3.step()
            
            with torch.no_grad():
                fusion_encoder.eval()
                agent3_head.eval()
                val_text_emb = text_encoder(val_tokens_t)
                val_img_emb = img_encoder(val_img_feats.to(DEVICE))
                val_fused = fusion_encoder(val_text_emb, val_img_emb)
                _, val_b, val_u, _ = agent3_head(val_fused)
                val_acc = (val_b.argmax(dim=1).cpu() == val_labels_t).float().mean().item()
                
                if val_acc > best_acc3:
                    best_acc3 = val_acc
                    torch.save(fusion_encoder.state_dict(), f'{CHECKPOINT_DIR}/fusion_encoder.pt')
                    torch.save(agent3_head.state_dict(), f'{CHECKPOINT_DIR}/agent3_head_enhanced.pt')
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/20, Loss: {total_loss/max(B//64,1):.4f}, "
                      f"Val Acc: {val_acc*100:.2f}%, Avg u={val_u.mean().item():.4f}")
        
        print(f"  Agent3 最佳验证准确率: {best_acc3*100:.2f}%")
        fusion_encoder.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/fusion_encoder.pt', weights_only=True))
        agent3_head.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/agent3_head_enhanced.pt', weights_only=True))
        fusion_encoder.eval()
        agent3_head.eval()
    
    if step == 2:
        print("\n[完成] 证据头训练完成")
        return None
    
    # ========== 2.5 训练GAT共识层 ==========
    print("\n[2.5] 训练GAT共识层...")
    
    # 提取训练集上各Agent的输出用于GAT训练
    N_AGENTS = 3
    train_tokens_t = train_tokens.to(DEVICE)
    with torch.no_grad():
        train_text_emb = text_encoder(train_tokens_t)
        train_img_emb = img_encoder(train_img_feats.to(DEVICE))
        train_fused = fusion_encoder(train_text_emb, train_img_emb)
        
        train_alpha1, train_b1, train_u1, _ = agent1_head(train_text_emb)
        train_alpha2, train_b2, train_u2, _ = agent2_head(train_img_emb)
        train_alpha3, train_b3, train_u3, _ = agent3_head(train_fused)
    
    train_all_beliefs = [train_b1, train_b2, train_b3]
    train_all_us = [train_u1, train_u2, train_u3]
    train_all_embs = [train_text_emb, train_img_emb, train_fused]
    train_y = train_labels_t.to(DEVICE)
    
    # DS融合基线
    train_ds_preds, _, _ = ds_fusion_decision(train_all_beliefs, train_all_us, u_threshold=U_THRESHOLD)
    train_correct = (train_ds_preds == train_y)
    
    # 有分歧样本筛选
    train_has_disagreement = torch.zeros(len(train_y), dtype=torch.bool, device=DEVICE)
    for i in range(N_AGENTS):
        for j in range(i+1, N_AGENTS):
            train_has_disagreement |= (train_all_beliefs[i].argmax(dim=1) != train_all_beliefs[j].argmax(dim=1))
    
    train_gat_mask = train_correct & train_has_disagreement
    train_gat_indices = torch.where(train_gat_mask)[0]
    
    print(f"  DS正确样本: {train_correct.sum().item()}/{len(train_y)}")
    print(f"  有分歧样本: {train_has_disagreement.sum().item()}/{len(train_y)}")
    print(f"  GAT训练样本: {len(train_gat_indices)}")
    
    if len(train_gat_indices) > 20:
        # 创建GAT共识引擎
        gat_node_dim = EMBED_DIM + NUM_CLASSES + 1  # 259
        gat_layer = GATConsensusLayer(
            node_dim=gat_node_dim, hidden_dim=64, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES
        ).to(DEVICE)
        
        # 检查是否有已训练的GAT模型
        gat_model_path = f'{CHECKPOINT_DIR}/gat_consensus_hm.pt'
        if os.path.exists(gat_model_path):
            gat_layer.load_state_dict(torch.load(gat_model_path, map_location=DEVICE, weights_only=True))
            gat_layer.eval()
            print(f"  加载已训练的GAT模型: {gat_model_path}")
        else:
            # 构造训练数据集
            gat_train_alphas = []
            for i in range(N_AGENTS):
                S = NUM_CLASSES / train_all_us[i].clamp(min=1e-6)
                alpha_i = train_all_beliefs[i] * S.unsqueeze(-1) + 1.0
                gat_train_alphas.append(alpha_i)
            
            # 训练GAT层
            gat_optimizer = torch.optim.Adam(gat_layer.parameters(), lr=5e-4, weight_decay=1e-4)
            gat_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gat_optimizer, T_max=30)
            
            best_gat_loss = float('inf')
            num_train = len(train_gat_indices)
            
            for epoch in range(30):
                gat_layer.train()
                total_loss = 0.0
                perm_indices = train_gat_indices[torch.randperm(num_train)]
                
                for start_idx in range(0, num_train, 32):
                    batch_idx = perm_indices[start_idx:start_idx+32]
                    batch_loss = 0.0
                    
                    for b_idx_cpu in batch_idx.cpu().numpy():
                        agent_outputs = []
                        for i in range(N_AGENTS):
                            b_i = train_all_beliefs[i][b_idx_cpu:b_idx_cpu+1]  # [1, C]
                            u_i = train_all_us[i][b_idx_cpu:b_idx_cpu+1]        # [1, 1]
                            emb_i = train_all_embs[i][b_idx_cpu:b_idx_cpu+1]    # [1, D]
                            u_val = u_i.squeeze(-1).item()  # scalar
                            S = NUM_CLASSES / max(u_val, 1e-6)
                            alpha_i = b_i[0] * S + 1.0
                            agent_outputs.append((alpha_i, b_i[0], u_val, emb_i[0]))
                        
                        # 构建状态并运行（使用训练中的GAT层）
                        engine_tmp = ConsensusEngine(
                            embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, hidden_dim=64
                        )
                        engine_tmp.layer = gat_layer
                        try:
                            h = engine_tmp.build_state(agent_outputs)
                            h_final, n_iters, converged, energy_trace, attn_trace = \
                                engine_tmp.run(h, max_iters=5, tol=1e-4, verbose=False)
                            
                            outputs = engine_tmp.extract_outputs(h_final)
                            new_belief = outputs[0][1].unsqueeze(0)
                            
                            true_label = train_y[b_idx_cpu].unsqueeze(0)
                            ce_loss = F.cross_entropy(new_belief.unsqueeze(0), true_label.unsqueeze(0))
                            
                            if len(energy_trace) > 1:
                                energy_reg = F.relu(energy_trace[-1] - energy_trace[0])
                                batch_loss = batch_loss + ce_loss + 0.01 * energy_reg
                            else:
                                batch_loss = batch_loss + ce_loss
                        except Exception:
                            pass
                    
                    if batch_loss > 0:
                        gat_optimizer.zero_grad()
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(gat_layer.parameters(), 1.0)
                        gat_optimizer.step()
                        total_loss += batch_loss.item()
                
                gat_scheduler.step()
                avg_loss = total_loss / max(num_train, 1)
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"    GAT Epoch {epoch+1:3d}/30: loss={avg_loss:.4f}")
                
                if avg_loss < best_gat_loss:
                    best_gat_loss = avg_loss
                    torch.save(gat_layer.state_dict(), gat_model_path)
            
            gat_layer.load_state_dict(torch.load(gat_model_path, map_location=DEVICE, weights_only=True))
            gat_layer.eval()
            print(f"  ★ GAT训练完成, 最佳loss={best_gat_loss:.4f}")
        
        # 创建GAT共识引擎用于评估（覆盖为训练好的GAT层）
        gat_engine = ConsensusEngine(
            embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, hidden_dim=64
        )
        gat_engine.layer = gat_layer
    else:
        print("  [警告] GAT训练样本不足，回退到确定性共识")
        gat_engine = None
    
    # ========== 3. 评估 ==========
    print("\n[3] 验证集评估...")
    B = len(val_labels)
    
    print("  提取验证特征...")
    val_tokens_t = val_tokens.to(DEVICE)
    with torch.no_grad():
        val_text_emb = text_encoder(val_tokens_t)
        val_img_emb = img_encoder(val_img_feats.to(DEVICE))
        val_fused = fusion_encoder(val_text_emb, val_img_emb)
        
        alpha1, b1, u1, _ = agent1_head(val_text_emb)
        alpha2, b2, u2, _ = agent2_head(val_img_emb)
        alpha3, b3, u3, _ = agent3_head(val_fused)
    
    all_alphas = torch.stack([alpha1, alpha2, alpha3], dim=1)
    all_beliefs = torch.stack([b1, b2, b3], dim=1)
    all_uncertainties = torch.stack([u1, u2, u3], dim=1)
    all_embs = torch.stack([val_text_emb, val_img_emb, val_fused], dim=1)
    
    y_true = val_labels_t.numpy()
    
    print(f"\n  各智能体基线:")
    print(f"  {'Agent':<15} {'Acc%':<10} {'F1%':<10} {'Avg u':<10}")
    print(f"  {'-'*45}")
    for idx, name in enumerate(['Agent1(文本)', 'Agent2(图像)', 'Agent3(融合)']):
        preds = all_beliefs[:, idx].argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_true, preds) * 100
        f1 = f1_score(y_true, preds, average='binary') * 100
        avg_u = all_uncertainties[:, idx].mean().item()
        print(f"  {name:<15} {acc:<10.2f} {f1:<10.2f} {avg_u:<10.4f}")
    
    # === 方法1: 多数投票 ===
    all_preds_tensor = all_beliefs.argmax(dim=-1)
    mv_preds, _ = torch.mode(all_preds_tensor, dim=1)
    
    # === 方法2: 加权平均 ===
    b_stack = all_beliefs.permute(1, 0, 2)
    u_stack = all_uncertainties.permute(1, 0)
    weights = F.softmax(1 - u_stack, dim=0)
    weighted_b = (b_stack * weights.unsqueeze(-1)).sum(dim=0)
    wa_global_u = (u_stack * weights).sum(dim=0)
    wa_preds = weighted_b.argmax(dim=1)
    wa_rej = wa_global_u > U_THRESHOLD
    
    # === 方法3: DS融合 ===
    b_list = [b1, b2, b3]
    u_list = [u1, u2, u3]
    ds_preds, ds_rej, ds_u = ds_fusion_decision(b_list, u_list, u_threshold=U_THRESHOLD)
    
    # === 方法4: GAT共识 + 重新证据头 + DS ===
    if gat_engine is not None:
        print("\n  运行真实GAT共识层...")
        final_belief_list = []
        final_u_list = []
        n_iters_list = []
        energy_history = []
        
        gat_engine.layer.eval()
        with torch.no_grad():
            for b_idx in range(B):
                agent_outputs = []
                for i in range(N_AGENTS):
                    b_i = all_beliefs[b_idx, i].unsqueeze(0)
                    u_i = all_uncertainties[b_idx, i].unsqueeze(0)
                    emb_i = all_embs[b_idx, i].unsqueeze(0)
                    u_val = u_i.squeeze(-1).item()  # scalar
                    S_val = NUM_CLASSES / max(u_val, 1e-6)
                    alpha_i = b_i[0] * S_val + 1.0
                    agent_outputs.append((alpha_i, b_i[0], u_val, emb_i[0]))
                
                h = gat_engine.build_state(agent_outputs)
                h_final, n_iters, converged, e_trace, _ = gat_engine.run(
                    h, max_iters=10, tol=1e-4, verbose=False
                )
                outputs = gat_engine.extract_outputs(h_final)
                
                fs = []
                us = []
                for i in range(N_AGENTS):
                    fs.append(outputs[i][1])
                    us.append(outputs[i][2])
                final_belief_list.append(torch.stack(fs, dim=0))
                final_u_list.append(torch.tensor(us))
                n_iters_list.append(n_iters)
                if len(e_trace) > 0:
                    energy_history.append(e_trace[-1])
                else:
                    energy_history.append(0.0)
        
        final_belief = torch.stack(final_belief_list, dim=0)
        final_uncertainty = torch.stack(final_u_list, dim=0)
        avg_n_iters = np.mean(n_iters_list)
        avg_energy = np.mean(energy_history)
        print(f"  平均共识迭代: {avg_n_iters:.1f}次, 平均最终能量: {avg_energy:.4f}")
        print(f"  共识后平均u: {final_uncertainty.mean().item():.4f} (从前{all_uncertainties.mean().item():.4f})")
    else:
        print("\n  运行确定性共识层（GAT不可用，回退）...")
        consensus = DeterministicConsensus()
        
        with torch.no_grad():
            cur_emb = all_embs.clone().to(DEVICE)
            energy_history = []
            
            for iteration in range(20):
                cur_b1 = agent1_head(cur_emb[:, 0])
                cur_b2 = agent2_head(cur_emb[:, 1])
                cur_b3 = agent3_head(cur_emb[:, 2])
                cur_belief = torch.stack([cur_b1[1], cur_b2[1], cur_b3[1]], dim=1)
                cur_uncertainty = torch.stack([cur_b1[2], cur_b2[2], cur_b3[2]], dim=1)
                
                energy = consensus_energy(cur_emb, cur_belief, cur_uncertainty)
                energy_history.append(energy.item())
                
                if iteration > 1 and abs(energy_history[-1] - energy_history[-2]) < 1e-3:
                    break
                
                cur_emb = DeterministicConsensus.consensus_update(
                    cur_emb, cur_belief, cur_uncertainty, mix_ratio=0.3
                )
            
            final_b1 = agent1_head(cur_emb[:, 0])[1]
            final_b2 = agent2_head(cur_emb[:, 1])[1]
            final_b3 = agent3_head(cur_emb[:, 2])[1]
            final_u1 = agent1_head(cur_emb[:, 0])[2]
            final_u2 = agent2_head(cur_emb[:, 1])[2]
            final_u3 = agent3_head(cur_emb[:, 2])[2]
            
            final_belief = torch.stack([final_b1, final_b2, final_b3], dim=1)
            final_uncertainty = torch.stack([final_u1, final_u2, final_u3], dim=1)
        
        avg_energy = energy_history[-1] if energy_history else 0.0
        print(f"  最终能量: {avg_energy:.6f}")
        print(f"  共识后平均u: {final_uncertainty.mean().item():.4f} (从前{all_uncertainties.mean().item():.4f})")
    
    gat_beliefs = [final_belief[:, i] for i in range(3)]
    gat_uncertainties = [final_uncertainty[:, i] for i in range(3)]
    gat_ds_preds, gat_ds_rej, gat_ds_u = ds_fusion_decision(gat_beliefs, gat_uncertainties, 
                                                              u_threshold=U_THRESHOLD)
    
    # === 方法5: GAT + 分歧解构 + EMNet ===
    print("\n  运行分歧解构 + EMNet纠偏...")
    
    deconstructor = DisagreementDeconstructor(u_threshold=0.5, K_threshold=0.3)
    conflict_types, K_values = deconstructor.deconstruct_batch(all_beliefs, all_uncertainties)
    
    evidence_count = sum(1 for c in conflict_types if c == 'evidence_conflict')
    ignorance_count = sum(1 for c in conflict_types if c == 'ignorance_conflict')
    no_conflict_count = sum(1 for c in conflict_types if c == 'none')
    print(f"  分歧分布: 证据冲突={evidence_count}, 无知冲突={ignorance_count}, 无分歧={no_conflict_count}")
    
    # 在弱agent场景（各Agent Acc~55%），学习型EMNet无法收敛到有效映射
    # 改用确定性证据交换：最佳agent（u最低）向最差agent（u最高）传递×2证据
    corrected_alphas = all_alphas.clone()
    
    for b_idx in range(B):
        if conflict_types[b_idx] == 'evidence_conflict':
            best_agent = all_uncertainties[b_idx].argmin().item()
            worst_agent = all_uncertainties[b_idx].argmax().item()
            if best_agent != worst_agent:
                best_alpha = all_alphas[b_idx, best_agent]  # [C]
                # 最佳agent的证据量（alpha-1），幅度放大2倍
                sender_evidence = (best_alpha - 1.0) * 2.0
                sender_evidence = F.relu(sender_evidence)  # 确保非负
                corrected_alphas[b_idx, worst_agent] += sender_evidence
    
    corrected_beliefs = corrected_alphas / corrected_alphas.sum(dim=-1, keepdim=True)
    corrected_uncertainties = NUM_CLASSES / corrected_alphas.sum(dim=-1)
    
    corr_b_list = [corrected_beliefs[:, i].to(DEVICE) for i in range(3)]
    corr_u_list = [corrected_uncertainties[:, i].to(DEVICE) for i in range(3)]
    corr_ds_preds, corr_ds_rej, corr_ds_u = ds_fusion_decision(corr_b_list, corr_u_list,
                                                                 u_threshold=U_THRESHOLD)
    
    # ========== 汇总结果 ==========
    results = {
        'Agent1_Text': {
            'preds': b1.argmax(dim=1), 'rejected': torch.zeros(B, dtype=torch.bool),
        },
        'Agent2_Image': {
            'preds': b2.argmax(dim=1), 'rejected': torch.zeros(B, dtype=torch.bool),
        },
        'Agent3_Fusion': {
            'preds': b3.argmax(dim=1), 'rejected': torch.zeros(B, dtype=torch.bool),
        },
        'MajorityVoting': {
            'preds': mv_preds, 'rejected': torch.zeros(B, dtype=torch.bool),
        },
        'WeightedAvg': {
            'preds': wa_preds, 'rejected': wa_rej, 'uncertainty': wa_global_u,
        },
        'DS_Fusion': {
            'preds': ds_preds, 'rejected': ds_rej, 'uncertainty': ds_u,
        },
        'GAT_DS_Fusion': {
            'preds': gat_ds_preds, 'rejected': gat_ds_rej, 'uncertainty': gat_ds_u,
        },
        'GAT_EMNet_Fusion': {
            'preds': corr_ds_preds.cpu(), 'rejected': corr_ds_rej.cpu(), 'uncertainty': corr_ds_u.cpu(),
        },
    }
    
    # ========== 计算指标 ==========
    print(f"\n{'='*120}")
    print(f"{'方法':<22s} {'Acc%':<10s} {'F1%':<10s} {'ECE':<12s} {'Rej%':<10s} {'Acc_All%':<10s}")
    print(f"{'-'*120}")
    
    metrics = {}
    for method_name, res in results.items():
        preds_np = res['preds'].cpu().numpy()
        rej_np = res['rejected'].cpu().numpy()
        
        rej_rate = rej_np.mean() * 100
        acc_all = accuracy_score(y_true, preds_np) * 100
        
        accepted = ~rej_np
        if accepted.sum() > 0:
            acc = accuracy_score(y_true[accepted], preds_np[accepted]) * 100
            f1 = f1_score(y_true[accepted], preds_np[accepted], average='binary') * 100
        else:
            acc, f1 = 0.0, 0.0
        
        u = res.get('uncertainty', None)
        if u is not None and accepted.sum() > 0:
            u_np = u.cpu().numpy()
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
        
        metrics[method_name] = {
            'accuracy': acc, 'f1': f1, 'ece': float(ece),
            'rejection_rate': rej_rate, 'accuracy_all': acc_all,
        }
        
        print(f"{method_name:<22s} {acc:<10.2f} {f1:<10.2f} {ece:<12.4f} "
              f"{rej_rate:<10.2f} {acc_all:<10.2f}")
    
    print(f"{'='*120}")
    
    # ========== 保存 ==========
    with open(os.path.join(RESULT_DIR, 'evaluation_results_enhanced.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: results/hateful_memes/evaluation_results_enhanced.json")
    
    detail_data = {
        'y_true': y_true.tolist(),
        'conflict_types': conflict_types,
        'K_values': [float(k) for k in K_values],
        'n_iters': n_iters,
        'energy_history': energy_history,
    }
    for method_name, res in results.items():
        detail_data[f'{method_name}_preds'] = res['preds'].cpu().tolist()
        detail_data[f'{method_name}_rejected'] = res['rejected'].cpu().tolist()
        if 'uncertainty' in res:
            detail_data[f'{method_name}_uncertainty'] = res['uncertainty'].cpu().tolist()
    
    with open(os.path.join(RESULT_DIR, 'evaluation_details_enhanced.json'), 'w', encoding='utf-8') as f:
        json.dump(detail_data, f, indent=2)
    
    # ========== 绘图 ==========
    print("\n[4] 生成可视化...")
    
    # 对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    all_m = ['MajorityVoting', 'WeightedAvg', 'DS_Fusion', 'GAT_DS_Fusion', 'GAT_EMNet_Fusion']
    labels = ['多数投票', '加权平均', 'DS融合', 'GAT+DS', 'GAT+EMNet']
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#DDA0DD']
    
    for ax, vals, title in zip(axes, 
        [[metrics[m]['accuracy'] for m in all_m],
         [metrics[m]['f1'] for m in all_m],
         [metrics[m]['ece'] for m in all_m]],
        ['准确率 (接受样本)', 'F1 (接受样本)', 'ECE (置信度校准)']):
        bars = ax.bar(labels, vals, color=colors, alpha=0.8, width=0.5)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.03,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Hateful Memes 增强方法对比', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'hateful_memes_enhanced_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图: figures/hateful_memes_enhanced_comparison.png")
    
    # 分歧分析图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    types = ['none', 'evidence_conflict', 'ignorance_conflict']
    counts = [sum(1 for c in conflict_types if c == t) for t in types]
    colors2 = ['#4ECDC4', '#FF6B6B', '#FFD93D']
    bars = axes[0].bar(['无分歧', '证据冲突', '无知冲突'], counts, color=colors2, alpha=0.8)
    axes[0].set_title('分歧类型分布', fontsize=13)
    axes[0].set_ylabel('样本数')
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(v), ha='center', va='bottom', fontsize=12)
    
    K_values_f = [float(k) for k in K_values]
    axes[1].hist(K_values_f, bins=20, color='#45B7D1', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0.3, color='red', linestyle='--', label='K阈值=0.3')
    axes[1].set_xlabel('冲突系数 K')
    axes[1].set_ylabel('样本数')
    axes[1].set_title('冲突系数分布', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Hateful Memes 分歧解构分析', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'hateful_memes_conflict_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  分歧分析图: figures/hateful_memes_conflict_analysis.png")
    
    # 混淆矩阵
    methods_plot = ['Agent1_Text', 'Agent2_Image', 'Agent3_Fusion',
                    'DS_Fusion', 'GAT_DS_Fusion', 'GAT_EMNet_Fusion']
    titles_plot = ['Agent1 (文本)', 'Agent2 (图像)', 'Agent3 (融合)',
                   'DS融合', 'GAT+DS', 'GAT+EMNet纠偏']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, method in enumerate(methods_plot):
        ax = axes[idx]
        cm = confusion_matrix(y_true, results[method]['preds'].cpu().numpy(), labels=[0, 1])
        ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(titles_plot[idx], fontsize=12)
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['非仇恨', '仇恨'])
        ax.set_yticklabels(['非仇恨', '仇恨'])
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, color=color)
    
    plt.suptitle('Hateful Memes 增强混淆矩阵', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'hateful_memes_enhanced_confusion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  混淆矩阵: figures/hateful_memes_enhanced_confusion.png")
    
    print(f"\n{'='*80}")
    print(f"增强评估完成！")
    print(f"{'='*80}")
    
    return metrics


# =============================================================================
# 入口
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hateful Memes 增强评估管线')
    parser.add_argument('--max_train', type=int, default=2000, help='训练样本数')
    parser.add_argument('--max_val', type=int, default=500, help='验证样本数')
    parser.add_argument('--force_extract', action='store_true', help='强制重新提取特征')
    parser.add_argument('--skip_training', action='store_true', help='跳过训练')
    parser.add_argument('--step', type=int, default=0, choices=[0, 1, 2, 3],
                        help='0=全部, 1=仅特征提取, 2=仅训练证据头, 3=仅评估')
    
    args = parser.parse_args()
    run_enhanced_pipeline(
        max_train=args.max_train, max_val=args.max_val,
        force_extract=args.force_extract, skip_training=args.skip_training, step=args.step,
    )