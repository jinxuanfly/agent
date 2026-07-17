"""
第一步：合成数据与单智能体证据网络
====================================
生成二分类合成数据集（判断点(x,y)是否在圆内）。
设计3个异构智能体，每个智能体只看不同的特征视图。
为每个智能体训练MLP证据网络，输出Dirichlet参数。

修正(v3 - 2025.06.11):
- 放弃α放大因子（之前发现softplus logits太小导致α≈1）
- 改用改进损失：证据量损失鼓励S增大，使u落在0.1~0.3范围
- 使用更小的KL退火权重，防止α坍缩到1

修正(v4 - 新增硬样本):
- 增加 HardConflictCircleData 子类，用于生成"死锁"样本
- 在圆边界附近 (0.85 < r < 1.15) 构造 Agent1↔Agent2 极端对立
- 这些样本将导致内循环不收敛，用于测试分歧解构和纠偏

理论依据：
- Dirichlet分布是类别概率的共轭先验，α_c 表示第c类的证据量
- 总证据强度 S = Σα_c，认知不确定性 u = C/S（C为类别数）
- 信念质量 b_c = (α_c - 1) / S，满足 Σb_c + u = 1
- E-MSE损失：最小化预测概率与one-hot标签的均方误差
- KL正则化：惩罚α远离1，避免过拟合，惩罚项权重随训练逐步增加
- 证据量损失：鼓励S>10，防止退化为均匀分布
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# 1. 合成数据生成
# =============================================================================

class SyntheticCircleData:
    """
    生成二分类合成数据集：判断点(x,y)是否在单位圆内。
    圆形: x^2 + y^2 <= 1 为类别1（圆内），否则为类别0（圆外）
    
    三个异构智能体：
    - Agent1: 只看 x 坐标
    - Agent2: 只看 y 坐标  
    - Agent3: 看半径 r = sqrt(x^2 + y^2)（理想情况下可直接分类）
    
    通过添加噪声和标签翻转人为制造分歧。
    """
    
    def __init__(self, n_train=10000, n_test=2000, noise_level=0.15, flip_ratio=0.1):
        """
        Args:
            n_train: 训练样本数
            n_test: 测试样本数
            noise_level: 特征噪声级别（对Agent1和Agent2添加噪声）
            flip_ratio: 标签翻转比例（对Agent1和Agent2）
        """
        self.n_train = n_train
        self.n_test = n_test
        self.noise_level = noise_level
        self.flip_ratio = flip_ratio
        
        # 生成数据
        self._generate()
        
    def _generate(self):
        """生成训练和测试数据"""
        # 在 [-1.5, 1.5] x [-1.5, 1.5] 正方形内均匀采样
        n_total = self.n_train + self.n_test
        x = np.random.uniform(-1.5, 1.5, n_total)
        y = np.random.uniform(-1.5, 1.5, n_total)
        
        # 真实标签：是否在单位圆内
        r = np.sqrt(x**2 + y**2)
        labels = (r <= 1.0).astype(np.int64)
        
        # 为每个智能体构建特征视图
        # Agent1: 只看 x（含噪声）
        agent1_feat = x.copy().reshape(-1, 1)
        agent1_noise = np.random.normal(0, self.noise_level, agent1_feat.shape)
        agent1_feat = agent1_feat + agent1_noise
        
        # Agent2: 只看 y（含噪声）
        agent2_feat = y.copy().reshape(-1, 1)
        agent2_noise = np.random.normal(0, self.noise_level, agent2_feat.shape)
        agent2_feat = agent2_feat + agent2_noise
        
        # Agent3: 看半径 r（近乎完美的特征）
        agent3_feat = r.copy().reshape(-1, 1)
        agent3_noise = np.random.normal(0, self.noise_level * 0.2, agent3_feat.shape)
        agent3_feat = agent3_feat + agent3_noise
        
        # 为Agent1和Agent2制造标签翻转（分歧）
        agent1_labels = labels.copy()
        agent2_labels = labels.copy()
        agent3_labels = labels.copy()  # Agent3使用真实标签
        
        # 随机选择需要翻转的样本
        n_flip = int(n_total * self.flip_ratio)
        flip_idx1 = np.random.choice(n_total, n_flip, replace=False)
        flip_idx2 = np.random.choice(n_total, n_flip, replace=False)
        
        agent1_labels[flip_idx1] = 1 - agent1_labels[flip_idx1]
        agent2_labels[flip_idx2] = 1 - agent2_labels[flip_idx2]
        
        # 对Agent1和Agent2添加额外的特征噪声（更大的噪声使分歧更明显）
        extra_noise1 = np.random.normal(0, 0.1, agent1_feat.shape)
        agent1_feat = agent1_feat + extra_noise1 * (np.random.random(n_total) < 0.3).reshape(-1, 1)
        
        extra_noise2 = np.random.normal(0, 0.1, agent2_feat.shape)
        agent2_feat = agent2_feat + extra_noise2 * (np.random.random(n_total) < 0.3).reshape(-1, 1)
        
        # 分割训练/测试
        self.x_train = {
            'agent1': torch.FloatTensor(agent1_feat[:self.n_train]),
            'agent2': torch.FloatTensor(agent2_feat[:self.n_train]),
            'agent3': torch.FloatTensor(agent3_feat[:self.n_train]),
        }
        self.y_train = {
            'agent1': torch.LongTensor(agent1_labels[:self.n_train]),
            'agent2': torch.LongTensor(agent2_labels[:self.n_train]),
            'agent3': torch.LongTensor(agent3_labels[:self.n_train]),
        }
        
        self.x_test = {
            'agent1': torch.FloatTensor(agent1_feat[self.n_train:]),
            'agent2': torch.FloatTensor(agent2_feat[self.n_train:]),
            'agent3': torch.FloatTensor(agent3_feat[self.n_train:]),
        }
        self.y_test = {
            'agent1': torch.LongTensor(agent1_labels[self.n_train:]),
            'agent2': torch.LongTensor(agent2_labels[self.n_train:]),
            'agent3': torch.LongTensor(agent3_labels[self.n_train:]),
        }
        
        # 保存真实标签用于评估
        self.y_true = torch.LongTensor(labels[self.n_train:])
        self.r_test = r[self.n_train:]
        self.x_coords = x[self.n_train:]
        self.y_coords = y[self.n_train:]
        
        # 记录哪些样本被翻转
        self.flip_idx1 = flip_idx1[flip_idx1 >= self.n_train] - self.n_train
        self.flip_idx2 = flip_idx2[flip_idx2 >= self.n_train] - self.n_train
        
        # 保存原始坐标用于硬样本分析
        self.x_all = x
        self.y_all = y
        self.r_all = r
        
        print(f"数据生成完成:")
        print(f"  训练样本: {self.n_train}, 测试样本: {self.n_test}")
        print(f"  类别分布 (真实): 圆内={labels[:self.n_train].sum()}, 圆外={self.n_train - labels[:self.n_train].sum()}")
        print(f"  Agent1标签翻转: {n_flip} 样本")
        print(f"  Agent2标签翻转: {n_flip} 样本")


class HardConflictCircleData(SyntheticCircleData):
    """
    硬冲突数据生成器（继承自 SyntheticCircleData）
    
    特点：
    - 在圆边界附近 (0.85 < r < 1.15) 的样本中，强制Agent1和Agent2的标签相反
    - 使训练出的模型在这些样本上产生"高置信度冲突"
    - 这些样本将导致GAT内循环不收敛（或需极多轮次）
    
    设计原理：
    - Agent1看到x>0区间时"自信地"预测类别1
    - Agent2看到y>0区间时"自信地"预测类别0
    - 两者在边界区域（x≈0,y≈0附近）发生强烈冲突
    - 语义嵌入距离大，因为MLP对x和y的表示完全不同
    """
    
    def __init__(self, n_train=10000, n_test=2000, noise_level=0.15, 
                 flip_ratio=0.15, hard_ratio=0.2):
        """
        Args:
            hard_ratio: 硬样本占总样本的比例（针对边界附近区域）
        """
        self.hard_ratio = hard_ratio
        super().__init__(n_train, n_test, noise_level, flip_ratio)
    
    def _generate(self):
        """覆写生成方法，加入硬冲突样本"""
        n_total = self.n_train + self.n_test
        x = np.random.uniform(-1.5, 1.5, n_total)
        y = np.random.uniform(-1.5, 1.5, n_total)
        r = np.sqrt(x**2 + y**2)
        
        # 真实标签
        labels = (r <= 1.0).astype(np.int64)
        
        # 找到边界附近样本 (0.85 < r < 1.15)
        boundary_mask = (r > 0.85) & (r < 1.15)
        boundary_indices = np.where(boundary_mask)[0]
        n_boundary = len(boundary_indices)
        
        # 从中选择 hard_ratio 比例作为硬冲突样本
        n_hard = int(n_boundary * self.hard_ratio)
        hard_indices = np.random.choice(boundary_indices, n_hard, replace=False)
        
        print(f"  边界样本: {n_boundary}, 硬冲突样本: {n_hard}")
        
        # 为每个智能体构建特征视图
        agent1_feat = x.copy().reshape(-1, 1)
        agent1_noise = np.random.normal(0, self.noise_level, agent1_feat.shape)
        agent1_feat = agent1_feat + agent1_noise
        
        agent2_feat = y.copy().reshape(-1, 1)
        agent2_noise = np.random.normal(0, self.noise_level, agent2_feat.shape)
        agent2_feat = agent2_feat + agent2_noise
        
        agent3_feat = r.copy().reshape(-1, 1)
        agent3_noise = np.random.normal(0, self.noise_level * 0.2, agent3_feat.shape)
        agent3_feat = agent3_feat + agent3_noise
        
        # 标签初始化
        agent1_labels = labels.copy()
        agent2_labels = labels.copy()
        agent3_labels = labels.copy()
        
        # ---- 普通标签翻转（随机） ----
        n_flip_base = int(n_total * self.flip_ratio)
        # 从非硬样本中选择翻转
        non_hard_indices = np.setdiff1d(np.arange(n_total), hard_indices)
        n_flip = min(n_flip_base, len(non_hard_indices))
        flip_idx1 = np.random.choice(non_hard_indices, n_flip, replace=False)
        flip_idx2 = np.random.choice(non_hard_indices, n_flip, replace=False)
        
        agent1_labels[flip_idx1] = 1 - agent1_labels[flip_idx1]
        agent2_labels[flip_idx2] = 1 - agent2_labels[flip_idx2]
        
        # ---- 硬冲突处理：边界样本反向翻转 ----
        # 对硬样本，将Agent1和Agent2的标签设置为相反
        # 如果真实标签是1，则Agent1=0, Agent2=1（错一个）
        # 更极端：基于x和y的符号，让Agent1的决策与Agent2的决策相反
        for idx in hard_indices:
            if labels[idx] == 1:  # 真实在圆内
                agent1_labels[idx] = 0  # Agent1说是圆外
                agent2_labels[idx] = 1  # Agent2说是圆内（正确）
            else:  # 真实在圆外
                agent1_labels[idx] = 1  # Agent1说是圆内
                agent2_labels[idx] = 0  # Agent2说是圆外（正确）
        
        # ---- 硬样本额外噪声（使语义嵌入差异更大） ----
        for idx in hard_indices:
            if np.random.random() < 0.5:
                # 给Agent1的x特征加上大偏移
                agent1_feat[idx, 0] += np.random.choice([-0.8, 0.8]) * np.random.uniform(0.5, 1.0)
            else:
                # 给Agent2的y特征加上大偏移
                agent2_feat[idx, 0] += np.random.choice([-0.8, 0.8]) * np.random.uniform(0.5, 1.0)
        
        # 分割训练/测试
        self.x_train = {
            'agent1': torch.FloatTensor(agent1_feat[:self.n_train]),
            'agent2': torch.FloatTensor(agent2_feat[:self.n_train]),
            'agent3': torch.FloatTensor(agent3_feat[:self.n_train]),
        }
        self.y_train = {
            'agent1': torch.LongTensor(agent1_labels[:self.n_train]),
            'agent2': torch.LongTensor(agent2_labels[:self.n_train]),
            'agent3': torch.LongTensor(agent3_labels[:self.n_train]),
        }
        
        self.x_test = {
            'agent1': torch.FloatTensor(agent1_feat[self.n_train:]),
            'agent2': torch.FloatTensor(agent2_feat[self.n_train:]),
            'agent3': torch.FloatTensor(agent3_feat[self.n_train:]),
        }
        self.y_test = {
            'agent1': torch.LongTensor(agent1_labels[self.n_train:]),
            'agent2': torch.LongTensor(agent2_labels[self.n_train:]),
            'agent3': torch.LongTensor(agent3_labels[self.n_train:]),
        }
        
        self.y_true = torch.LongTensor(labels[self.n_train:])
        self.r_test = r[self.n_train:]
        self.x_coords = x[self.n_train:]
        self.y_coords = y[self.n_train:]
        
        # 记录硬样本在测试集中的索引
        self.hard_test_indices = hard_indices[hard_indices >= self.n_train] - self.n_train
        
        print(f"数据生成完成 (HardConflict模式):")
        print(f"  训练样本: {self.n_train}, 测试样本: {self.n_test}")
        print(f"  硬冲突训练样本: {(hard_indices < self.n_train).sum()}")
        print(f"  硬冲突测试样本: {len(self.hard_test_indices)}")
        print(f"  类别分布 (真实): 圆内={labels[:self.n_train].sum()}, 圆外={self.n_train - labels[:self.n_train].sum()}")


# =============================================================================
# 2. 证据网络 (Evidential MLP)
# =============================================================================

class EvidentialMLP(nn.Module):
    """
    证据MLP网络：输入特征，输出Dirichlet参数 α = [α_1, ..., α_C]
    
    网络结构：
    - 输入层 -> 隐藏层1 (ReLU) -> 隐藏层2 (ReLU) -> 输出层
    - 输出层使用softplus确保α > 0
    - 倒数第二层输出作为语义嵌入向量
    """
    
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        # 分类头：输出类别证据
        self.classifier = nn.Linear(embed_dim, output_dim)
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        前向传播，返回 (alpha, embedding)
        
        alpha = softplus(cls(emb)) + 1.0
        softplus确保α > 0，+1 提供最小先验证据（防止除零）
        """
        embedding = self.net(x)
        logits = self.classifier(embedding)
        alpha = torch.nn.functional.softplus(logits) + 1.0
        return alpha, embedding
    
    def get_output(self, x):
        """
        返回完整的认知状态 (alpha, b, u, embedding)
        
        Returns:
            alpha: Dirichlet参数 [batch_size, C]
            b: 信念质量 [batch_size, C]
            u: 认知不确定性 [batch_size, 1]
            embedding: 语义嵌入 [batch_size, embed_dim]
        """
        alpha, embedding = self.forward(x)
        S = alpha.sum(dim=1, keepdim=True)  # 总证据强度
        b = (alpha - 1.0) / S  # 信念质量
        u = self.output_dim / S  # 不确定性
        return alpha, b, u, embedding


# =============================================================================
# 3. 证据损失函数
# =============================================================================

class EvidentialLoss(nn.Module):
    """
    证据损失函数 = E-MSE + λ_kl * KL散度 + λ_S * 证据量损失
    
    修正(v3):
    - 降低KL退火速度（慢40轮才到满权重）
    - 降低KL最大权重（从1.0到0.5）
    - 加入证据量损失：鼓励S达到target_S（默认15）
      使u = C/S = 2/15 ≈ 0.133，达到合理水平
    """
    
    def __init__(self, annealing_step=40, kl_max_weight=0.5, target_S=15.0, S_weight=0.1):
        super().__init__()
        self.annealing_step = annealing_step
        self.kl_max_weight = kl_max_weight
        self.target_S = target_S
        self.S_weight = S_weight
        
    def forward(self, alpha, target, epoch=0):
        batch_size, num_classes = alpha.shape
        
        # One-hot编码
        y_onehot = torch.zeros_like(alpha)
        y_onehot.scatter_(1, target.unsqueeze(1), 1.0)
        
        # 总证据强度
        S = alpha.sum(dim=1, keepdim=True)
        
        # E-MSE损失
        # E[||p - y||²] = Σ (y_c - α_c/S)² + α_c(S-α_c)/(S²(S+1))
        p = alpha / S
        var = alpha * (S - alpha) / (S.pow(2) * (S + 1))
        mse = (y_onehot - p).pow(2).sum(dim=1)
        var_sum = var.sum(dim=1)
        loss_mse = (mse + var_sum).mean()
        
        # KL散度正则化
        alpha_tilde = y_onehot * alpha + (1 - y_onehot) * 1.0
        S_tilde = alpha_tilde.sum(dim=1)
        
        term1 = torch.lgamma(S_tilde)
        term2 = torch.lgamma(alpha_tilde).sum(dim=1)
        term3 = -torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
        term4 = ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde.unsqueeze(1)))).sum(dim=1)
        
        kl_div = term1 - term2 + term3 + term4
        
        # 退火（更慢）
        annealing_coef = min(1.0, epoch / self.annealing_step)
        loss_kl = kl_div.mean() * annealing_coef * self.kl_max_weight
        
        # 证据量损失：鼓励S达到目标值
        S_mean = S.mean()
        loss_S = torch.abs(S_mean - self.target_S) * self.S_weight
        
        total_loss = loss_mse + loss_kl + loss_S
        return total_loss, loss_mse, loss_kl, loss_S


# =============================================================================
# 4. 训练与评估函数
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, epoch, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_kl = 0
    total_S = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        alpha, _ = model(x_batch)
        loss, loss_mse, loss_kl, loss_S = criterion(alpha, y_batch, epoch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_kl += loss_kl.item()
        total_S += loss_S.item()
    
    n_batches = len(loader)
    return total_loss / n_batches, total_mse / n_batches, total_kl / n_batches, total_S / n_batches


def evaluate(model, x_test, y_test, device):
    """评估模型准确率并返回预测信息"""
    model.eval()
    with torch.no_grad():
        x_test = x_test.to(device)
        alpha, b, u, embedding = model.get_output(x_test)
        pred = b.argmax(dim=1)
        y_test = y_test.to(device)
        acc = (pred == y_test).float().mean().item()
        return {
            'accuracy': acc,
            'predictions': pred.cpu(),
            'alpha': alpha.cpu(),
            'b': b.cpu(),
            'u': u.cpu(),
            'embedding': embedding.cpu(),
        }


def compute_ece(probs, labels, n_bins=10):
    """计算期望校准误差 (ECE)"""
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) * (confidences < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_size = in_bin.sum().float()
            ece += (bin_size / len(labels)) * (bin_acc - bin_conf).abs()
    return ece.item()


def train_agent(agent_name, model, train_loader, test_x, test_y, 
                epochs=200, lr=1e-3, device=DEVICE):
    """训练单个智能体的证据网络"""
    print(f"\n{'='*50}")
    print(f"训练 {agent_name}")
    print(f"{'='*50}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = EvidentialLoss(annealing_step=40, kl_max_weight=0.5, target_S=15.0, S_weight=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5
    )
    
    history = {'loss': [], 'mse': [], 'kl': [], 'S': [], 'acc': []}
    
    for epoch in range(epochs):
        loss, mse, kl, S_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, device)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            eval_result = evaluate(model, test_x, test_y, device)
            acc = eval_result['accuracy']
            scheduler.step(loss)
            history['loss'].append(loss)
            history['mse'].append(mse)
            history['kl'].append(kl)
            history['S'].append(S_loss)
            history['acc'].append(acc)
            
            # 诊断信息
            S_mean = eval_result['alpha'].sum(dim=1).mean().item()
            u_mean = eval_result['u'].mean().item()
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f} | MSE: {mse:.4f} | KL: {kl:.4f} | "
                  f"Acc: {acc:.4f} | Mean S: {S_mean:.1f} | u: {u_mean:.4f}")
    
    # 最终评估
    eval_result = evaluate(model, test_x, test_y, device)
    print(f"\n{agent_name} 训练完成!")
    print(f"  最终准确率: {eval_result['accuracy']:.4f}")
    
    b = eval_result['b']
    test_y_cpu = test_y.cpu()
    ece = compute_ece(b, test_y_cpu)
    print(f"  ECE: {ece:.4f}")
    
    # Alpha/不确定性诊断
    S_mean = eval_result['alpha'].sum(dim=1).mean().item()
    u_mean = eval_result['u'].mean().item()
    print(f"  Mean S (总证据): {S_mean:.2f}, Mean u (不确定性): {u_mean:.4f}")
    
    plot_calibration_curve(b, test_y_cpu, agent_name)
    
    return model, eval_result, history


def plot_calibration_curve(probs, labels, agent_name):
    """绘制校准曲线"""
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()
    
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accs.append(accuracies[in_bin].mean().item())
            bin_confs.append(confidences[in_bin].mean().item())
            bin_sizes.append(in_bin.sum().item())
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_sizes.append(0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(bin_confs, bin_accs, 'o-', label=f'{agent_name}', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    for i, (conf, acc, size) in enumerate(zip(bin_confs, bin_accs, bin_sizes)):
        plt.bar(conf, acc, width=0.08, alpha=0.3, color='blue')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Calibration Curve - {agent_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/calibration_{agent_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  校准曲线已保存: figures/calibration_{agent_name}.png")


def visualize_predictions(data, agent_results, agent_names):
    """可视化每个智能体的预测结果"""
    n_agents = len(agent_names)
    fig, axes = plt.subplots(2, n_agents, figsize=(5 * n_agents, 10))
    if n_agents == 1:
        axes = axes.reshape(2, 1)
    
    x_coords = data.x_test['agent1'].numpy().flatten()
    y_coords = data.x_test['agent2'].numpy().flatten()
    
    for idx, name in enumerate(agent_names):
        result = agent_results[name]
        b = result['b'].numpy()
        u = result['u'].numpy().flatten()
        pred = result['predictions'].numpy()
        
        ax1 = axes[0, idx]
        scatter1 = ax1.scatter(x_coords, y_coords, c=pred, cmap='coolwarm', 
                               s=10, alpha=0.6, vmin=0, vmax=1)
        ax1.set_title(f'{name} - Predictions')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, ticks=[0, 1])
        circle = plt.Circle((0, 0), 1.0, fill=False, color='green', linestyle='--', linewidth=2)
        ax1.add_patch(circle)
        
        ax2 = axes[1, idx]
        scatter2 = ax2.scatter(x_coords, y_coords, c=u, cmap='YlOrRd', 
                               s=10, alpha=0.6, vmin=0, vmax=1)
        ax2.set_title(f'{name} - Uncertainty u')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2)
        circle = plt.Circle((0, 0), 1.0, fill=False, color='green', linestyle='--', linewidth=2)
        ax2.add_patch(circle)
    
    plt.tight_layout()
    plt.savefig('figures/agent_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测可视化已保存: figures/agent_predictions.png")


def print_divergence_analysis(data, agent_results, agent_names):
    """分析三个智能体之间的分歧情况"""
    n_test = len(data.y_true)
    predictions = {}
    uncertainties = {}
    
    for name in agent_names:
        result = agent_results[name]
        predictions[name] = result['predictions'].numpy()
        uncertainties[name] = result['u'].numpy().flatten()
    
    pred_matrix = np.stack([predictions[name] for name in agent_names], axis=1)
    all_agree = (pred_matrix.max(axis=1) == pred_matrix.min(axis=1))
    disagreement_idx = np.where(~all_agree)[0]
    
    print(f"\n{'='*50}")
    print(f"智能体分歧分析")
    print(f"{'='*50}")
    print(f"测试样本总数: {n_test}")
    print(f"三个智能体一致: {all_agree.sum()} ({all_agree.sum()/n_test*100:.1f}%)")
    print(f"存在分歧: {len(disagreement_idx)} ({len(disagreement_idx)/n_test*100:.1f}%)")
    
    if len(disagreement_idx) > 0:
        print(f"\n分歧样本示例 (前5个):")
        header_str = f"{'idx':<6} {'x':<8} {'y':<8} {'r':<8} {'真实':<6} "
        for name in agent_names:
            header_str += f"| {name:<12} u "
        print(header_str)
        print('-' * 90)
        
        for i in disagreement_idx[:5]:
            r_val = data.r_test[i]
            true_label = data.y_true[i].item()
            line = f"{i:<6} {data.x_test['agent1'][i].item():<8.3f} {data.x_test['agent2'][i].item():<8.3f} "
            line += f"{r_val:<8.3f} {true_label:<6} "
            for name in agent_names:
                line += f"| {int(predictions[name][i]):<12} {uncertainties[name][i]:.3f} "
            print(line)
    
    return disagreement_idx


# =============================================================================
# 5. 主函数
# =============================================================================

def main():
    print("=" * 60)
    print("第一步：合成数据生成与单智能体证据网络训练 (v3)")
    print("=" * 60)
    
    # 使用普通数据
    print("\n[1/5] 生成普通合成数据...")
    data = SyntheticCircleData(n_train=10000, n_test=2000, noise_level=0.15, flip_ratio=0.1)
    
    print("\n[2/5] 创建智能体证据网络...")
    agent_names = ['Agent1_x', 'Agent2_y', 'Agent3_r']
    models = {}
    
    models['Agent1_x'] = EvidentialMLP(input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32).to(DEVICE)
    models['Agent2_y'] = EvidentialMLP(input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32).to(DEVICE)
    models['Agent3_r'] = EvidentialMLP(input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32).to(DEVICE)
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {total_params} 参数")
    
    print("\n[3/5] 训练智能体...")
    agent_results = {}
    batch_size = 256
    n_epochs = 200
    
    name_to_key = {'Agent1_x': 'agent1', 'Agent2_y': 'agent2', 'Agent3_r': 'agent3'}
    
    for name in agent_names:
        model = models[name]
        data_key = name_to_key[name]
        
        train_dataset = TensorDataset(data.x_train[data_key], data.y_train[data_key])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model, eval_result, history = train_agent(
            name, model, train_loader, 
            data.x_test[data_key], data.y_test[data_key],
            epochs=n_epochs, lr=1e-3
        )
        agent_results[name] = eval_result
        torch.save(model.state_dict(), f'models/{name}_evidential.pth')
        print(f"  模型已保存: models/{name}_evidential.pth")
    
    print("\n[4/5] 综合分析...")
    print(f"\n{'='*50}")
    print(f"各智能体最终准确率")
    print(f"{'='*50}")
    for name in agent_names:
        result = agent_results[name]
        agent_key = {'Agent1_x': 'agent1', 'Agent2_y': 'agent2', 'Agent3_r': 'agent3'}[name]
        test_y = data.y_test[agent_key]
        ece = compute_ece(result['b'], test_y)
        S_values = result['alpha'].sum(dim=1)
        S_mean = S_values.mean().item()
        S_std = S_values.std().item()
        print(f"  {name:15s} | Acc: {result['accuracy']:.4f} | ECE: {ece:.4f} | "
              f"S: {S_mean:.1f}+/-{S_std:.1f} | u: {result['u'].mean():.4f}")
    
    print("\n[5/5] 生成可视化...")
    visualize_predictions(data, agent_results, agent_names)
    disagreement_idx = print_divergence_analysis(data, agent_results, agent_names)
    np.save('models/disagreement_samples.npy', disagreement_idx)
    print(f"\n分歧样本索引已保存: models/disagreement_samples.npy")
    
    print("\n" + "=" * 60)
    print("第一步完成！")
    print("=" * 60)
    
    return data, models, agent_results


if __name__ == '__main__':
    main()