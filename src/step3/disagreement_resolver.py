"""
第三步：分歧解构器 + 简单纠偏
==============================
功能组件：
1. D-S基本概率分配（BPA）转换
2. 冲突系数K计算
3. 分歧类型解构（证据冲突 vs 无知冲突）
4. 贝叶斯证据交换（EMNet）
5. 无知冲突处理（拒识）
6. 完整的分歧处理管线
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# 1. BPA转换与DS工具函数
# =============================================================================

def alpha_to_bpa(alpha):
    """
    将Dirichlet参数转换为D-S基本概率分配
    
    Args:
        alpha: [num_classes] Dirichlet超参数
    
    Returns:
        b: [num_classes] 信念质量（单点mass）
        u: 标量 不确定性（全集mass）
        e: [num_classes] 证据（alpha - 1）
    
    理论依据：
    - b_k = (α_k - 1) / S, where S = Σ(α_k - 1) + K = Σ α_k
    - u = K / S, where K = num_classes
    """
    K = alpha.shape[-1]
    S = alpha.sum()
    
    if S <= 0:
        b = torch.ones_like(alpha) / K
        u = torch.tensor(0.99, device=alpha.device)
        e = torch.zeros_like(alpha)
        return b, u, e
    
    e = alpha - 1.0  # 证据 = α - 1
    e = F.relu(e)    # 确保非负
    
    e_sum = e.sum()
    if e_sum > 0:
        b = e / S
        u = K / S
    else:
        # 无证据
        b = torch.ones_like(alpha) / K
        u = 0.99
        e = torch.zeros_like(alpha)
    
    u = torch.clamp(u, min=0.0, max=0.999)
    b = b / (b.sum() + 1e-8)
    
    return b, u, e


def compute_conflict_K(b1, u1, b2, u2):
    """
    计算两个智能体之间的D-S冲突系数K
    
    K = Σ_{i≠j} m₁(c_i) * m₂(c_j)
      = 1 - Σ_i m₁(c_i) * m₂(c_i) - m₁(Θ) * m₂(Θ)
    
    K ∈ [0, 1]，K越接近1表示冲突越大
    
    Returns:
        K: 冲突系数（Python float或标量Tensor）
    """
    agreement = (b1 * b2).sum()
    both_u = u1 * u2
    K = 1.0 - agreement - both_u
    # 使用torch.clamp处理Tensor，np.clip处理float
    if torch.is_tensor(K):
        K = torch.clamp(K, 0.0, 1.0)
    else:
        K = np.clip(K, 0.0, 1.0)
    return K


# =============================================================================
# 2. 分歧解构器
# =============================================================================

class DisagreementDeconstructor:
    """
    分歧解构器
    ----------
    功能：辨识分歧类型是"证据冲突"还是"无知冲突"
    
    判断规则：
    - 证据冲突（evidence_conflict）：K高（>K_th），u低（<u_th）
      → 智能体相信自己知道不同的事，有实质矛盾
    - 无知冲突（ignorance_conflict）：u高（>u_th），K可高可低
      → 智能体不知道自己在说什么，不一致来自不靠谱
    - 无分歧（none）：K低（<K_th），u低（<u_th）
      → 一致
    """
    
    def __init__(self, u_threshold=0.5, K_threshold=0.3):
        self.u_threshold = u_threshold
        self.K_threshold = K_threshold
    
    def deconstruct_pair(self, b1, u1, b2, u2):
        """
        解构一对智能体的分歧类型
        
        修正逻辑(v3 - 关键修复):
        - 训练好的证据网络u都很低(~0.001)，所以不能用u_threshold来区分
        - 分歧的本质由K值决定：
          * K高(>0.2) → 证据冲突：智能体自信但对同一问题有不同答案
          * K低(<0.1) → 无分歧：智能体基本一致
          * 中间(0.1-0.2) → 轻微分歧，可视为evidence_conflict
        - u高(>0.3) → 无知冲突：有人不确定
        - 使用更低的K阈值以适应训练好的低u模型
        
        Returns:
            conflict_type: str
                "evidence_conflict" | "ignorance_conflict" | "none"
            K: float 冲突系数
            details: dict 详细信息
        """
        K = compute_conflict_K(b1, u1, b2, u2)
        
        avg_u = (u1 + u2) / 2.0
        
        # 先检查无知（高不确定性）
        if avg_u > self.u_threshold:
            conflict_type = "ignorance_conflict"
        # 再检查证据冲突（低不确定性但高冲突）
        elif K > self.K_threshold:
            conflict_type = "evidence_conflict"
        else:
            conflict_type = "none"
        
        details = {
            'K': K,
            'avg_u': avg_u,
            'u1': u1,
            'u2': u2,
        }
        
        return conflict_type, K, details
    
    def deconstruct_group(self, beliefs, uncertainties):
        """
        解构智能体组的分歧（两两分析，取最严重类型）
        
        Args:
            beliefs: [n, C]
            uncertainties: [n]
        
        Returns:
            global_type: 最严重的分歧类型
            pair_results: [(i,j,type,K), ...]
        """
        n = beliefs.shape[0]
        pair_results = []
        
        types_priority = {'none': 0, 'ignorance_conflict': 1, 'evidence_conflict': 2}
        worst_type = 'none'
        
        for i in range(n):
            for j in range(i + 1, n):
                ctype, K, details = self.deconstruct_pair(
                    beliefs[i], uncertainties[i],
                    beliefs[j], uncertainties[j]
                )
                pair_results.append((i, j, ctype, K))
                
                if types_priority.get(ctype, 0) > types_priority.get(worst_type, 0):
                    worst_type = ctype
        
        return worst_type, pair_results


# =============================================================================
# 3. EMNet: 贝叶斯证据交换网络
# =============================================================================

class EMNet(nn.Module):
    """
    贝叶斯证据交换网络（Evidence Mapping Network）
    
    功能：当检测到证据冲突时，将发送方的证据映射为接收方的证据增量Δα
    本质：学习"如何从对方证据中提取有用信息"
    
    架构：
    - 输入：发送方证据 e_s = [e_s1, e_s2, ..., e_sC] ∈ ℝ^C
    - 输出：接收方证据增量 Δα_t ∈ ℝ^C
    - 通过残差连接实现"部分采纳"
    """
    
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),  # 确保输出为正
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, sender_evidence):
        """
        Args:
            sender_evidence: [batch, C] 发送方证据
        
        Returns:
            delta_alpha: [batch, C] 接收方应增加的α
        """
        delta = self.net(sender_evidence)  # [batch, C]
        # 限制增量大小，防止过度修正
        delta = delta * 0.5
        return delta


def train_emnet(emnet, train_loader, epochs=100, lr=0.001):
    """
    训练EMNet
    
    训练数据生成策略：
    - 当两个智能体证据冲突时，期望的Δα应使接收方转向正确类别
    - 构造输入：发送方证据 = max(0, α_s - 1)
    - 目标：接收方应增加的证据
    """
    optimizer = torch.optim.Adam(emnet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    emnet.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = emnet(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(emnet.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def generate_emnet_data(n_samples=10000, num_classes=2, device=DEVICE):
    """
    为EMNet生成合成训练数据
    
    模拟场景：
    - 发送方有较强证据（低u）
    - 接收方证据不足或错误
    - 目标是让接收方吸收发送方的部分证据
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # 生成发送方证据（有信息量）
    e_s = torch.rand(n_samples, num_classes, device=device) * 10 + 1
    # 接收方目标：吸收发送方证据的一定比例
    mixing_ratio = torch.rand(n_samples, 1, device=device) * 0.6 + 0.1  # 10%-70%
    target = e_s * mixing_ratio
    
    # 添加噪声
    noise = torch.randn(n_samples, num_classes, device=device) * 0.3
    target = F.relu(target + noise)
    
    return e_s, target


# =============================================================================
# 4. 无知冲突处理
# =============================================================================

class IgnoranceHandler:
    """
    无知冲突处理
    -----------
    策略：
    1. 拒识（reject）：标记为无法决策
    2. 主动信息请求：在现实系统中触发，这里记录日志
    3. 记录不确定性来源
    """
    
    def __init__(self, log_dir=None):
        self.rejection_log = []
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def handle(self, sample_id, agent_outputs, details):
        """
        处理无知冲突
        
        Returns:
            decision: -1 (拒识)
            metadata: dict
        """
        self.rejection_log.append({
            'sample_id': sample_id,
            'agent_uncertainties': [u.item() if hasattr(u, 'item') else u 
                                   for _, _, u, _ in agent_outputs],
            'details': details,
        })
        
        print(f"  [拒识] 样本 {sample_id}: 无知冲突, "
              f"平均u = {details.get('avg_u', 0):.3f}")
        
        return -1


# =============================================================================
# 5. 完整分歧处理管线
# =============================================================================

class DisagreementPipeline:
    """
    完整分歧处理管线
    
    流程：
    收到共识失败信号 →
        1. 分歧解构
        2a. 证据冲突 → EMNet证据交换 → 重入共识
        2b. 无知冲突 → 拒识
        3. 记录日志
    """
    
    def __init__(self, num_classes=2, u_threshold=0.5, K_threshold=0.3, 
                 emnet_hidden=16, log_dir=None):
        self.num_classes = num_classes
        self.u_threshold = u_threshold
        self.K_threshold = K_threshold
        
        self.deconstructor = DisagreementDeconstructor(u_threshold, K_threshold)
        self.emnet = EMNet(input_dim=num_classes, hidden_dim=emnet_hidden, 
                          output_dim=num_classes)
        self.ignorance_handler = IgnoranceHandler(log_dir)
        
        self.emnet_trained = False
        self.train_log = []
    
    def train_emnet_offline(self, n_samples=10000, epochs=100, lr=0.001):
        """离线训练EMNet"""
        print("\n[EMNet] 生成训练数据...")
        e_s, targets = generate_emnet_data(n_samples, self.num_classes)
        dataset = TensorDataset(e_s, targets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        print(f"[EMNet] 训练 {epochs} 轮次...")
        losses = train_emnet(self.emnet, loader, epochs, lr)
        
        self.emnet_trained = True
        self.train_log = losses
        
        print(f"[EMNet] 训练完成，最终损失: {losses[-1]:.6f}")
        return losses
    
    def evidence_exchange(self, alpha_s, alpha_t):
        """
        证据交换：将发送方的证据映射后提供给接收方
        
        Args:
            alpha_s: [C] 发送方Dirichlet参数
            alpha_t: [C] 接收方Dirichlet参数
        
        Returns:
            alpha_t_new: [C] 更新后的接收方参数
        """
        if not self.emnet_trained:
            # 回退到简单复制
            e_s = F.relu(alpha_s - 1.0)
            delta = e_s * 0.3
        else:
            with torch.no_grad():
                e_s = F.relu(alpha_s - 1.0).unsqueeze(0)
                delta = self.emnet(e_s).squeeze(0)
        
        alpha_t_new = alpha_t + delta
        return alpha_t_new
    
    def process(self, agent_outputs, sample_id=None):
        """
        处理分歧
        
        Args:
            agent_outputs: [(alpha, b, u, emb), ...] 列表
            sample_id: 样本标识
        
        Returns:
            conflict_type: str
            decision: int 或 -1（拒识）
            new_alpha_list: [(alpha, b, u, emb), ...] 或 None
            metadata: dict
        """
        n = len(agent_outputs)
        C = agent_outputs[0][0].shape[0]
        
        # 提取信念和不确定性
        beliefs = torch.stack([b for _, b, _, _ in agent_outputs])
        uncertainties = torch.tensor([u.item() if hasattr(u, 'item') else float(u) 
                                       for _, _, u, _ in agent_outputs])
        
        # 1. 分歧解构
        conflict_type, pair_results = self.deconstructor.deconstruct_group(
            beliefs, uncertainties
        )
        
        metadata = {
            'conflict_type': conflict_type,
            'pair_results': pair_results,
            'sample_id': sample_id,
        }
        
        print(f"\n  分歧分析: 类型={conflict_type}")
        for i, j, ctype, K in pair_results[:3]:  # 只显示前3对
            print(f"    Agent{i+1}-Agent{j+1}: K={K:.3f}, type={ctype}")
        
        if conflict_type == 'evidence_conflict':
            # 证据冲突 → EMNet交换
            return self._handle_evidence_conflict(agent_outputs, metadata)
        elif conflict_type == 'ignorance_conflict':
            # 无知冲突 → 拒识
            decision = self.ignorance_handler.handle(sample_id, agent_outputs, metadata)
            return conflict_type, decision, None, metadata
        else:
            # 无分歧（不应到达这里）
            return conflict_type, None, agent_outputs, metadata
    
    def _handle_evidence_conflict(self, agent_outputs, metadata):
        """
        处理证据冲突：两两交换证据
        
        策略：
        1. 以不确定性最低的智能体为基准（权威）
        2. 其他智能体吸收基准的证据
        3. 返回更新后的α
        """
        n = len(agent_outputs)
        uncertainties = [u.item() if hasattr(u, 'item') else float(u) 
                        for _, _, u, _ in agent_outputs]
        
        # 找最可信的（低u）
        best_idx = int(np.argmin(uncertainties))
        best_alpha = agent_outputs[best_idx][0]
        
        print(f"  证据交换: 以Agent{best_idx+1}为基准 "
              f"(u={uncertainties[best_idx]:.4f})")
        
        new_agent_outputs = []
        for i, (alpha, b, u, emb) in enumerate(agent_outputs):
            if i == best_idx:
                # 基准不变
                new_alpha = alpha
            else:
                # 吸收基准证据
                new_alpha = self.evidence_exchange(best_alpha, alpha)
            
            # 重新计算BPA
            b_new, u_new, e_new = alpha_to_bpa(new_alpha)
            new_agent_outputs.append((new_alpha, b_new, u_new, emb))
        
        return ('evidence_conflict', None, new_agent_outputs, 
                {**metadata, 'best_idx': best_idx})


# =============================================================================
# 6. 合成数据测试
# =============================================================================

def test_on_synthetic():
    """在合成数据上测试分歧解构器"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'step1'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'step2'))
    from synthetic_data import SyntheticCircleData, EvidentialMLP, evaluate
    from gat_consensus import ConsensusEngine, global_decision
    
    print("=" * 60)
    print("第三步测试：分歧解构器 + 简单纠偏")
    print("=" * 60)
    
    print("\n[1/5] 加载数据和模型...")
    data = SyntheticCircleData(n_train=10000, n_test=2000)
    
    name_to_key = {'Agent1_x': 'agent1', 'Agent2_y': 'agent2', 'Agent3_r': 'agent3'}
    agent_names = ['Agent1_x', 'Agent2_y', 'Agent3_r']
    
    model_base = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), '..', 'models')
    
    # 检查使用哪种模型（large=128维 vs standard=64维）
    hidden_dim_check = 128  # 默认large
    test_path = os.path.join(model_base, f'{agent_names[0]}_evidential.pth')
    chk = torch.load(test_path, map_location='cpu', weights_only=True)
    if 'net.0.weight' in chk and chk['net.0.weight'].shape[0] == 64:
        hidden_dim_check = 64
    elif 'net.0.weight' in chk and chk['net.0.weight'].shape[0] == 128:
        hidden_dim_check = 128
    
    models = {}
    for name in agent_names:
        model = EvidentialMLP(input_dim=1, hidden_dim=hidden_dim_check, output_dim=2, embed_dim=32)
        path = os.path.join(model_base, f'{name}_evidential.pth')
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        models[name] = model
    
    # [2/5] 构造特定分歧样本
    print("\n[2/5] 构造刻意分歧样本...")
    
    # 找证据冲突样本和ignorance样本
    conflict_samples = []
    ignorance_samples = []
    normal_samples = []
    
    for idx in range(2000):
        agent_outputs = []
        for name in agent_names:
            data_key = name_to_key[name]
            x = data.x_test[data_key][idx:idx + 1].to(DEVICE)
            model = models[name]
            alpha, b, u, emb = model.get_output(x)
            agent_outputs.append((alpha[0], b[0], u[0], emb[0]))
        
        beliefs = torch.stack([b for _, b, _, _ in agent_outputs])
        uncertainties = torch.tensor([u.item() if hasattr(u, 'item') else float(u) 
                                       for _, _, u, _ in agent_outputs])
        
        decon = DisagreementDeconstructor(u_threshold=0.5, K_threshold=0.3)
        ctype, _ = decon.deconstruct_group(beliefs, uncertainties)
        
        if ctype == 'evidence_conflict' and len(conflict_samples) < 3:
            conflict_samples.append((idx, agent_outputs))
        elif ctype == 'ignorance_conflict' and len(ignorance_samples) < 3:
            ignorance_samples.append((idx, agent_outputs))
        elif ctype == 'none' and len(normal_samples) < 3:
            normal_samples.append((idx, agent_outputs))
        
        if len(conflict_samples) >= 3 and len(ignorance_samples) >= 3 and len(normal_samples) >= 3:
            break
    
    print(f"  找到: 证据冲突={len(conflict_samples)}, 无知冲突={len(ignorance_samples)}, "
          f"正常={len(normal_samples)}")
    
    # [3/5] 测试分歧解构器
    print("\n[3/5] 测试分歧解构器...")
    
    decon = DisagreementDeconstructor(u_threshold=0.5, K_threshold=0.3)
    
    for label, samples in [("证据冲突", conflict_samples), 
                           ("无知冲突", ignorance_samples),
                           ("正常", normal_samples)]:
        print(f"\n  --- {label} ---")
        for idx, agent_outputs in samples[:2]:
            true_label = data.y_true[idx].item()
            beliefs = torch.stack([b for _, b, _, _ in agent_outputs])
            uncertainties = torch.tensor([u.item() if hasattr(u, 'item') else float(u) 
                                           for _, _, u, _ in agent_outputs])
            
            ctype, pairs = decon.deconstruct_group(beliefs, uncertainties)
            print(f"  样本{idx} (真实={true_label}): 分类={ctype}")
            for i, j, ctype_ij, K in pairs:
                print(f"    A{i+1}-A{j+1}: K={K:.3f}, "
                      f"u={uncertainties[i]:.3f}/{uncertainties[j]:.3f}")
    
    # [4/5] 训练EMNet并测试证据交换
    print("\n[4/5] 训练&测试EMNet...")
    pipeline = DisagreementPipeline(num_classes=2, u_threshold=0.5, K_threshold=0.3)
    pipeline.train_emnet_offline(n_samples=5000, epochs=50, lr=0.01)
    
    # 测试证据交换效果
    print("\n  --- 证据交换效果测试 ---")
    for idx, agent_outputs in conflict_samples[:2]:
        print(f"\n  样本{idx} (真实={data.y_true[idx].item()}):")
        
        # 原始输出
        for i, name in enumerate(agent_names):
            _, b, u, _ = agent_outputs[i]
            pred = b.argmax().item()
            print(f"    原始 A{i+1}({name}): pred={pred}, "
                  f"b=[{b[0]:.3f},{b[1]:.3f}], u={u.item():.4f}")
        
        # 分歧处理
        ctype, decision, new_outputs, meta = pipeline.process(agent_outputs, idx)
        
        if new_outputs is not None:
            print(f"    证据交换后:")
            for i, name in enumerate(agent_names):
                _, b_new, u_new, _ = new_outputs[i]
                pred_new = b_new.argmax().item()
                print(f"    A{i+1}({name}): pred={pred_new}, "
                      f"b=[{b_new[0]:.3f},{b_new[1]:.3f}], u={u_new:.4f}")
            
            # 共识前检查是否能融合正确
            engine = ConsensusEngine(embed_dim=32, num_classes=2)
            h_state = engine.build_state(new_outputs)
            h_final, n_iters, converged, energy_trace, attn_trace = \
                engine.run(h_state, max_iters=20, tol=1e-4, verbose=False)
            final_outputs = engine.extract_outputs(h_final)
            decision2, global_b, global_u, _ = global_decision(final_outputs)
            
            print(f"    共识后: u={global_u:.4f}, 决策={decision2}")
            if decision2 == data.y_true[idx].item():
                print(f"    → [OK] 成功修正!")
            else:
                print(f"    → [FAIL] 仍错误")
    
    # [5/5] 完整管线端到端测试
    print("\n[5/5] 端到端管线测试...")
    
    n_test = 200
    correct_before = 0
    correct_after = 0
    rejected = 0
    evidence_conflict_count = 0
    ignorance_count = 0
    
    for idx in range(min(2000, n_test * 10)):
        if correct_before + correct_after + rejected >= n_test * 3:
            break
        
        agent_outputs = []
        for name in agent_names:
            data_key = name_to_key[name]
            x = data.x_test[data_key][idx:idx + 1].to(DEVICE)
            model = models[name]
            alpha, b, u, emb = model.get_output(x)
            agent_outputs.append((alpha[0], b[0], u[0], emb[0]))
        
        true_label = data.y_true[idx].item()
        
        # 原始多数投票
        votes = [b.argmax().item() for _, b, _, _ in agent_outputs]
        majority = max(set(votes), key=votes.count)
        if majority == true_label:
            correct_before += 1
        
        # 分歧处理
        ctype, decision, new_outputs, meta = pipeline.process(agent_outputs, idx)
        
        if ctype == 'evidence_conflict':
            evidence_conflict_count += 1
            if new_outputs is not None:
                engine = ConsensusEngine(embed_dim=32, num_classes=2)
                h_state = engine.build_state(new_outputs)
                h_final, n_iters, converged, _, _ = \
                    engine.run(h_state, max_iters=20, tol=1e-4, verbose=False)
                final_outputs = engine.extract_outputs(h_final)
                decision, _, _, _ = global_decision(final_outputs)
                if decision == true_label:
                    correct_after += 1
                if decision == -1:
                    rejected += 1
            else:
                if true_label == -1 or decision == -1:  # 拒识算对
                    pass
        elif ctype == 'ignorance_conflict':
            ignorance_count += 1
            rejected += 1
        
    total = evidence_conflict_count + ignorance_count
    print(f"\n  测试统计:")
    print(f"  原始多数投票: {correct_before} 正确")
    print(f"  证据冲突: {evidence_conflict_count}, 无知冲突: {ignorance_count}")
    print(f"  纠偏后正确: {correct_after}, 拒识: {rejected}")
    if total > 0:
        print(f"  纠偏成功率: {correct_after/max(total,1)*100:.1f}%")
    
    print(f"\n{'=' * 60}")
    print("第三步测试完成！")
    print(f"{'=' * 60}")


def main():
    test_on_synthetic()


if __name__ == '__main__':
    main()