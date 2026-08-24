"""
训练CIFAR-10N证据头 (v5 - 修复版)
=================================
关键修复：
1. 修复CE基EDL损失：F.cross_entropy(p, target) 对概率p重复softmax → 改用 -y·log(p)
2. 扩大模型容量：hidden_dim=256, 3层Linear
3. 更长的训练 (patience=30)
4. 更高的学习率 (1e-3)

训练策略：在干净标签上训练，保留噪声标签供共识框架评估。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import json
import pickle
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cpu')

os.makedirs('checkpoints/cifar10n', exist_ok=True)
os.makedirs('results/cifar10n', exist_ok=True)

NUM_CLASSES = 10
BATCH_SIZE = 256
EPOCHS = 200


# =============================================================================
# 1. 数据加载
# =============================================================================

def load_pixel_features():
    """从CIFAR-10原始数据加载像素特征（降维到256d）。"""
    cifar_dir = 'data/cifar-10-batches-py'
    if not os.path.exists(cifar_dir):
        print("  [Agent3] 使用torchvision下载...")
        from torchvision import datasets
        tmp_dir = 'data/tmp_cifar10'
        train_set = datasets.CIFAR10(tmp_dir, train=True, download=True)
        test_set = datasets.CIFAR10(tmp_dir, train=False, download=True)
        train_pixels = torch.FloatTensor(train_set.data).permute(0,3,1,2).reshape(50000, -1) / 255.0
        test_pixels = torch.FloatTensor(test_set.data).permute(0,3,1,2).reshape(10000, -1) / 255.0
    else:
        print("  [Agent3] 从本地加载...")
        def unpickle(file):
            with open(file, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')
        train_data = []
        for i in range(1, 6):
            batch = unpickle(os.path.join(cifar_dir, f'data_batch_{i}'))
            train_data.append(batch[b'data'])
        train_data = np.concatenate(train_data)
        test_data = unpickle(os.path.join(cifar_dir, 'test_batch'))[b'data']
        train_pixels = torch.FloatTensor(train_data) / 255.0
        test_pixels = torch.FloatTensor(test_data) / 255.0

    torch.manual_seed(SEED)
    proj = torch.randn(3072, 256) * 0.1
    train_proj = train_pixels @ proj
    test_proj = test_pixels @ proj

    train_mean = train_proj.mean(dim=0, keepdim=True)
    train_std = train_proj.std(dim=0, keepdim=True) + 1e-8
    train_proj = (train_proj - train_mean) / train_std
    test_proj = (test_proj - train_mean) / train_std

    print(f"  [Agent3] 像素特征(投影256d): 训练{train_proj.shape}, 测试{test_proj.shape}")
    return train_proj, test_proj


def load_features():
    """加载特征和标签"""
    print("\n[1] 加载预提取特征...")

    train_rn = torch.load('data/features/train_resnet18.pt')
    test_rn = torch.load('data/features/test_resnet18.pt')
    train_vit = torch.load('data/features/train_vit_tiny.pt')
    test_vit = torch.load('data/features/test_vit_tiny.pt')

    labels = torch.load('data/features/labels.pt')
    train_labels = labels['train_labels']
    test_labels = labels['test_labels']

    print("  [策略] 使用干净标签训练证据头")

    train_pixel, test_pixel = load_pixel_features()

    print(f"  Agent1 (ResNet-18): 训练{train_rn.shape}, 测试{test_rn.shape}")
    print(f"  Agent2 (ViT-Tiny):  训练{train_vit.shape}, 测试{test_vit.shape}")
    print(f"  Agent3 (Pixel+投影): 训练{train_pixel.shape}, 测试{test_pixel.shape}")

    return {
        'agent1': (train_rn, test_rn, train_labels, test_labels),
        'agent2': (train_vit, test_vit, train_labels, test_labels),
        'agent3': (train_pixel, test_pixel, train_labels, test_labels),
    }, test_labels


# =============================================================================
# 2. 证据头模型 (扩大容量)
# =============================================================================

class EvidenceHead(nn.Module):
    """证据头: 特征 -> Dirichlet α = [α1, ..., αK]"""
    def __init__(self, input_dim, num_classes=10, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        logits = self.net(x)
        alpha = self.softplus(logits) + 1
        return alpha

    def get_embedding(self, x):
        """获取语义嵌入（倒数第二层输出）"""
        with torch.no_grad():
            h = self.net[0](x)      # Linear
            h = self.net[2](h)      # ReLU
            h = self.net[4](h)      # Linear
            h = self.net[6](h)      # ReLU
            h = self.net[8](h)      # Linear(hidden_dim//2)
            h = self.net[9](h)      # ReLU -> 128维嵌入
        return h


# =============================================================================
# 3. EDL损失 (CE基 + KL正则化, 修复版)
# =============================================================================

def edl_loss(alpha, target, num_classes=10, annealing_step=1.0):
    """
    证据深度学习损失 (修复版)
    - CE部分: -∑ y·log(α/S)  不使用F.cross_entropy避免重复softmax
    - KL部分: KL(Dir(α̃) || Dir(1))
    """
    K = num_classes
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S

    # CE损失: 手动计算
    target_one_hot = F.one_hot(target, K).float().to(alpha.device)
    ce_loss = -(target_one_hot * torch.log(p + 1e-10)).sum(dim=1).sum()

    # KL正则化
    alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)

    K_t = torch.tensor(K, dtype=alpha.dtype, device=alpha.device)
    kl_loss = (
        torch.lgamma(S_tilde)
        - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        - torch.lgamma(K_t)
        + (alpha_tilde - 1).mul(
            torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
        ).sum(dim=1, keepdim=True)
    )
    kl_loss = kl_loss.squeeze().sum()

    total_loss = ce_loss + annealing_step * kl_loss
    return total_loss


# =============================================================================
# 4. 训练函数
# =============================================================================

def train_head(model, train_feats, train_labels, test_feats, test_labels,
               agent_name, epochs=EPOCHS):
    """训练单个证据头"""
    print(f"\n{'='*50}")
    print(f"训练 {agent_name}")
    print(f"{'='*50}")

    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            alpha = model(feats)

            annealing = min(1.0, epoch / 20)
            loss = edl_loss(alpha, labels, NUM_CLASSES, annealing_step=annealing)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            _, pred = alpha.max(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = correct / total
        test_info = evaluate_head(model, test_feats, test_labels)

        if test_info['acc'] > best_acc:
            best_acc = test_info['acc']
            best_state = model.state_dict().copy()
            torch.save(model.state_dict(), f'checkpoints/cifar10n/{agent_name}_best.pt')
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  [早停] {epoch+1}轮, 最佳Acc: {best_acc:.2%}")
            break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(train_loader):.1f} "
                  f"| Train Acc: {train_acc:.2%} | Test Acc: {test_info['acc']:.2%} "
                  f"| Avg U: {test_info['avg_u']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  {agent_name} 完成! 最佳测试准确率: {best_acc:.2%}")

    return model, best_acc


def evaluate_head(model, feats, labels):
    """评估证据头"""
    model.eval()
    with torch.no_grad():
        alpha = model(feats)
        S = alpha.sum(dim=1, keepdim=True)
        K = alpha.shape[1]
        u = K / S
        _, pred = alpha.max(dim=1)
        acc = (pred == labels).float().mean().item()

    return {'acc': acc, 'avg_u': u.mean().item(), 'preds': pred, 'u': u, 'alpha': alpha}


# =============================================================================
# 5. 主流程
# =============================================================================

def train_all_heads():
    """训练所有智能体的证据头（在干净标签上）"""
    print("=" * 60)
    print("CIFAR-10N 证据头训练 v5 (修复CE损失+大容量)")
    print("=" * 60)

    features, test_labels = load_features()

    print("\n[2] 创建证据头...")
    heads = nn.ModuleDict({
        'agent1': EvidenceHead(input_dim=512, num_classes=10),
        'agent2': EvidenceHead(input_dim=192, num_classes=10),
        'agent3': EvidenceHead(input_dim=256, num_classes=10),
    })

    print("\n[3] 训练 (在干净标签上)...")
    for name in ['agent1', 'agent2', 'agent3']:
        train_feats, test_feats, train_labels_clean, test_labels_clean = features[name]
        train_head(heads[name], train_feats, train_labels_clean,
                   test_feats, test_labels_clean, name)

    # 保存模型
    print("\n[4] 保存模型...")
    torch.save(heads, 'checkpoints/cifar10n/evidence_heads.pt')
    print("  证据头已保存至 checkpoints/cifar10n/evidence_heads.pt")

    # 汇总
    print("\n" + "=" * 60)
    print("训练汇总 (干净标签)")
    print("=" * 60)
    for name in ['agent1', 'agent2', 'agent3']:
        info = evaluate_head(heads[name], features[name][1], test_labels)
        print(f"  {name:<10}: 测试Acc={info['acc']:.2%}, Avg U={info['avg_u']:.4f}")

    with open('results/cifar10n/head_results.json', 'w') as f:
        serializable = {}
        for name in ['agent1', 'agent2', 'agent3']:
            info = evaluate_head(heads[name], features[name][1], test_labels)
            serializable[name] = {'acc': info['acc'], 'avg_u': info['avg_u']}
        json.dump(serializable, f, indent=2)

    print("=" * 60)
    return heads


if __name__ == '__main__':
    train_all_heads()