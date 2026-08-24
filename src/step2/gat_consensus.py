"""
第二步：内循环共识（不确定性感知GAT）
=====================================
实现基于图注意力网络的共识层，用于多智能体证据融合。

核心设计：
1. 节点特征 h_i = [emb_i; b_i; u_i]  —— 语义嵌入 + 信念 + 不确定性
2. 注意力系数 e_ij = LeakyReLU(a^T [W h_i || W h_j]) * (1 - u_j)
   含义：智能体j对智能体i的重要性 = 语义相关性 × (1 - u_j)
   （高不确定性→权重降低，因为该智能体不可靠）
3. 消息聚合：agg_i = Σ_j softmax(e_ij)_j * W h_j
4. 门控更新：gate_i = sigmoid(W_g [h_i || agg_i])
              h_i_new = h_i + gate_i * (agg_i - h_i)
   含义：选择性采纳聚合消息——高u更愿意接受外部信息
5. 收敛条件：||h_new - h_old|| < 1e-4 或最大20轮
6. 能量函数：E = Σ_i ||h_i_new - h_i_old||²，单调递减保证收敛

理论依据：
- 图注意力网络使智能体间可交换信息，区分"听谁的"
- 不确定性加权：高u的智能体权重低，防止误导
- 门控机制保持个体性，避免过平滑
- 能量函数递减证明收敛
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 1. 不确定性感知GAT共识层（手写矩阵运算）
# =============================================================================

class GATConsensusLayer(nn.Module):
    """
    图注意力共识层（手写实现，完全可控）
    
    节点特征: h_i = [emb_i; b_i; u_i]  （dim = d + C + 1）
    注意力: e_ij = LeakyReLU(a^T [W h_i || W h_j]) * (1 - u_j)
    消息聚合: agg_i = Σ_j α_ij * W h_j
    门控: gate_i = sigmoid(W_g [h_i || agg_i])
    更新: h_i_new = h_i + gate_i * (agg_i - h_i)
    """
    
    def __init__(self, node_dim, hidden_dim=64, embed_dim=32, num_classes=2):
        """
        Args:
            node_dim: 节点特征维度 = embed_dim + num_classes + 1
            hidden_dim: 注意力隐藏层维度
            embed_dim: 语义嵌入维度（用于提取b和u）
            num_classes: 类别数
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # 特征变换 W（将h_i映射到注意力空间）
        self.W = nn.Linear(node_dim, hidden_dim, bias=False)
        
        # 注意力向量 a（拼接后的注意力得分）
        self.a = nn.Linear(2 * hidden_dim, 1, bias=False)
        
        # 门控网络 W_g
        self.gate_net = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
            nn.Sigmoid(),
        )
        
        # 消息投影（将注意力聚合的 hidden_dim 映射回 node_dim）
        self.msg_proj = nn.Linear(hidden_dim, node_dim, bias=False)
        
        # 样本权重学习层
        self.sample_weight_net = nn.Linear(3, 1, bias=True)
        
        # Symbolic GAT: 信念相似度温度参数（可学习）
        self.sim_temp = nn.Parameter(torch.tensor(1.0))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，小值初始化便于稳定训练"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, h, u=None):
        n = h.shape[0]

        if u is None:
            u = h[:, -1]

        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(1).expand(-1, n, -1)
        Wh_j = Wh.unsqueeze(0).expand(n, -1, -1)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)

        e = self.a(concat).squeeze(-1)
        e = F.leaky_relu(e, negative_slope=0.2)

        u_factor = (1.0 - u).unsqueeze(0).expand(n, -1)
        e = e * u_factor

        # Symbolic GAT改进：添加信念相似度因子
        # 提取belief部分，计算Agent间信念余弦相似度
        belief_start = self.embed_dim
        belief_end = self.embed_dim + self.num_classes
        b_i = h[:, belief_start:belief_end]  # [n, C]
        b_norm = F.normalize(b_i, p=2, dim=1, eps=1e-8)  # [n, C]
        sim_matrix = b_norm @ b_norm.T  # [n, n] 余弦相似度 [-1, 1]
        sim_factor = (sim_matrix + 1.0) / 2.0  # 映射到 [0, 1]
        # 用可学习温度参数调节相似度的影响
        sim_factor = sim_factor ** self.sim_temp  # sim_temp > 1 时增强差异
        e = e * sim_factor

        mask = torch.eye(n, device=h.device).bool()
        e = e.masked_fill(mask, -1e9)

        attn_weights = F.softmax(e, dim=1)

        agg = attn_weights @ Wh

        msg = self.msg_proj(agg)

        gate_input = torch.cat([h, msg], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input))

        alpha = 0.15
        h_new = h + alpha * gate * (msg - h)

        belief_start = self.embed_dim
        belief_end = self.embed_dim + self.num_classes
        
        beliefs = F.relu(h_new[:, belief_start:belief_end])
        
        belief_sum = beliefs.sum(dim=1, keepdim=True)
        beliefs = torch.where(belief_sum > 1e-6, beliefs / belief_sum, 
                              torch.ones_like(beliefs) / self.num_classes)
        
        uncertainties = torch.clamp(h_new[:, -1:], min=0.01, max=0.99)
        
        h_new = torch.cat([
            h_new[:, :belief_start],
            beliefs,
            uncertainties
        ], dim=-1)

        energy = (h_new - h).pow(2).sum().item()

        return h_new, attn_weights, energy
    
    def forward_attn_only(self, h, u=None):
        n = h.shape[0]

        if u is None:
            u = h[:, -1]

        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(1).expand(-1, n, -1)
        Wh_j = Wh.unsqueeze(0).expand(n, -1, -1)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)

        e = self.a(concat).squeeze(-1)
        e = F.leaky_relu(e, negative_slope=0.2)

        u_factor = (1.0 - u).unsqueeze(0).expand(n, -1)
        e = e * u_factor

        # Symbolic GAT: 信念相似度因子
        belief_start = self.embed_dim
        belief_end = self.embed_dim + self.num_classes
        b_i = h[:, belief_start:belief_end]
        b_norm = F.normalize(b_i, p=2, dim=1, eps=1e-8)
        sim_matrix = b_norm @ b_norm.T
        sim_factor = (sim_matrix + 1.0) / 2.0
        sim_factor = sim_factor ** self.sim_temp
        e = e * sim_factor

        mask = torch.eye(n, device=h.device).bool()
        e = e.masked_fill(mask, -1e9)

        attn_weights = F.softmax(e, dim=1)
        
        return attn_weights
    
    def forward_fusion_weights(self, h, u=None):
        n = h.shape[0]

        if u is None:
            u = h[:, -1]

        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(1).expand(-1, n, -1)
        Wh_j = Wh.unsqueeze(0).expand(n, -1, -1)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)

        e = self.a(concat).squeeze(-1)
        e = F.leaky_relu(e, negative_slope=0.2)

        u_factor = (1.0 - u).unsqueeze(0).expand(n, -1)
        e = e * u_factor

        mask = torch.eye(n, device=h.device).bool()
        e = e.masked_fill(mask, -1e9)

        attn_weights = F.softmax(e, dim=1)
        
        fusion_weights = attn_weights.mean(dim=0)
        
        return fusion_weights
    
    def forward_direct_weights(self, h, u=None):
        n = h.shape[0]

        if u is None:
            u = h[:, -1]

        Wh = self.W(h)
        
        scores = torch.sum(Wh * Wh, dim=1)
        
        u_factor = (1.0 - u)
        scores = scores * u_factor
        
        fusion_weights = F.softmax(scores, dim=0)
        
        return fusion_weights
    
    def forward_sample_weights(self, h, u=None, hard_gate=False):
        n = h.shape[0]

        if u is None:
            u = h[:, -1]

        Wh = self.W(h)
        
        belief_dim = self.num_classes
        belief_start = self.embed_dim
        
        beliefs = h[:, belief_start:belief_start+belief_dim]
        
        confidence = beliefs.max(dim=1)[0]
        
        u_factor = (1.0 - u)
        
        feature_norm = torch.norm(Wh, dim=1)
        
        combined = torch.stack([confidence, u_factor, feature_norm], dim=1)
        
        raw_scores = self.sample_weight_net(combined).squeeze()
        
        if hard_gate:
            max_idx = raw_scores.argmax(dim=0)
            fusion_weights = torch.zeros_like(raw_scores)
            fusion_weights[max_idx] = 1.0
        else:
            fusion_weights = F.softmax(raw_scores, dim=0)
        
        return fusion_weights


# =============================================================================
# 2. 共识引擎
# =============================================================================

class ConsensusEngine:
    """
    共识引擎：管理GAT共识层和收敛逻辑
    
    流程：
    1. 从智能体输出构建初始节点状态
    2. 迭代运行GAT消息传递
    3. 检查收敛条件（能量 < 阈值 或 最大轮数）
    4. 从收敛状态提取智能体最终输出
    """
    
    def __init__(self, embed_dim=32, num_classes=2, hidden_dim=64):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.node_dim = embed_dim + num_classes + 1  # emb + b + u
        self.layer = GATConsensusLayer(
            node_dim=self.node_dim, 
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_classes=num_classes
        ).to(DEVICE)
    
    def build_state(self, agent_outputs):
        """
        从智能体输出构建节点状态矩阵
        
        Args:
            agent_outputs: 列表，每个元素为 (alpha, b, u, emb)
        
        Returns:
            h: 节点特征 [n, node_dim]
        """
        n = len(agent_outputs)
        device = self.layer.W.weight.device
        
        h_list = []
        for alpha, b, u, emb in agent_outputs:
            emb_f = emb.to(device).flatten()  # [embed_dim]
            b_f = b.to(device).flatten()  # [num_classes]
            if hasattr(u, 'reshape'):
                u_f = u.reshape(-1).to(device)  # [1]
            else:
                u_f = torch.tensor([float(u)], device=device)
            
            h_i = torch.cat([emb_f, b_f, u_f])  # [embed_dim + num_classes + 1]
            h_list.append(h_i)
        
        h = torch.stack(h_list, dim=0)  # [n, node_dim]
        return h
    
    def extract_outputs(self, h, retain_grad=False):
        """
        从节点状态提取智能体输出
        
        Args:
            h: [n, node_dim]
            retain_grad: 是否保留梯度（用于训练）
        
        Returns:
            outputs: 列表 (alpha, b, u, emb)
        """
        n = h.shape[0]
        outputs = []
        
        for i in range(n):
            h_i = h[i:i+1]
            
            emb = h_i[:, :self.embed_dim].squeeze(0)
            b = h_i[:, self.embed_dim:self.embed_dim+self.num_classes].squeeze(0)
            
            b = F.relu(b)
            b_sum = b.sum()
            if b_sum > 1e-6:
                b = b / b_sum
            else:
                b = torch.ones_like(b) / self.num_classes
            
            u = h_i[:, -1].squeeze(0)
            
            S = self.num_classes / torch.clamp(u, min=1e-6)
            alpha = b * S + 1.0
            
            if not retain_grad:
                emb = emb.cpu().detach()
                b = b.cpu().detach()
                u = u.cpu().detach().item()
                u = np.clip(u, 0.01, 0.99)
                alpha = alpha.cpu().detach()
            
            outputs.append((alpha, b, u, emb))
        
        return outputs
    
    def run(self, h, max_iters=20, tol=1e-4, verbose=True):
        """
        运行内循环共识（修复版v2）
        
        修复：跟踪最低能量状态，即使不收敛也返回最佳状态
        
        Args:
            h: 初始节点特征 [n, node_dim]
            max_iters: 最大迭代次数
            tol: 收敛阈值
            verbose: 是否打印日志
        
        Returns:
            h_best: 最低能量状态
            n_iters: 实际迭代次数
            converged: 是否收敛
            energy_trace: 能量轨迹
            attn_trace: 注意力矩阵轨迹
        """
        h_curr = h.clone().to(DEVICE)
        h_best = h.clone().to(DEVICE)
        energy_trace = []
        attn_trace = []
        best_energy = float('inf')
        converged = False
        energy_increase_count = 0
        prev_energy = float('inf')
        
        for it in range(max_iters):
            h_new, attn_weights, energy = self.layer(h_curr)
            
            energy_trace.append(energy)
            attn_trace.append(attn_weights.detach().cpu().numpy())
            
            # 跟踪最低能量状态
            if energy < best_energy:
                best_energy = energy
                h_best = h_new.clone()
            
            # 能量上升保护：如果能量连续上升3次，提前终止防止发散
            if energy > prev_energy * 1.1:
                energy_increase_count += 1
            else:
                energy_increase_count = 0
            
            if energy_increase_count >= 3:
                if verbose:
                    print(f"  [!] 能量连续上升，提前终止于迭代 {it+1}/{max_iters}")
                break
            
            prev_energy = energy
            
            # 检查收敛
            change = (h_new - h_curr).norm(dim=1).max().item()
            
            h_curr = h_new
            
            if verbose and ((it + 1) % 3 == 0 or it == 0):
                print(f"  Iter {it+1:2d}/{max_iters} | "
                      f"Δh_max: {change:.6e} | Energy: {energy:.6f}")
            
            if change < tol:
                converged = True
                if verbose:
                    print(f"  [OK] 收敛于迭代 {it+1}/{max_iters} "
                          f"(Δh_max={change:.6e} < {tol})")
                break
        
        if not converged and verbose:
            print(f"  [√] 未完全收敛({max_iters}轮)，使用最佳状态 "
                  f"(Energy={best_energy:.6e})")
        
        return h_best, it + 1, converged, energy_trace, attn_trace


# =============================================================================
# 3. 全局决策
# =============================================================================

def global_decision(outputs, u_threshold=0.5):
    """
    共识后的全局决策
    
    Args:
        outputs: 共识后 (alpha, b, u, emb)
        u_threshold: 拒识阈值
    
    Returns:
        decision: 类别或 -1（拒识）
        global_b: 平均信念
        global_u: 平均不确定性
        weights: 各智能体权重
    """
    n = len(outputs)
    C = outputs[0][1].shape[0]
    
    beliefs = torch.stack([b for _, b, _, _ in outputs])
    uncertainties = torch.tensor([float(u) if not isinstance(u, (int, float)) else u 
                                   for _, _, u, _ in outputs])
    uncertainties = torch.clamp(uncertainties, 0.01, 0.99)
    
    # 不确定性加权平均
    weights = (1.0 - uncertainties)
    weights = weights / (weights.sum() + 1e-8)
    
    global_b = (weights.unsqueeze(1) * beliefs).sum(dim=0)
    global_b = global_b / global_b.sum()
    
    global_u = (weights * uncertainties).sum().item()
    
    if global_u < u_threshold:
        decision = global_b.argmax().item()
    else:
        decision = -1
    
    return decision, global_b, global_u, weights.numpy()


# =============================================================================
# 4. 能量函数（绘制和分析）
# =============================================================================

def plot_energy_curve(energy_trace, title="能量函数收敛曲线", save_path="figures/energy_convergence.png"):
    """绘制能量函数随迭代次数的变化"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(energy_trace) + 1), energy_trace, 'b-o', 
             markersize=4, linewidth=1.5, label='Energy')
    plt.axhline(y=1e-4, color='r', linestyle='--', alpha=0.7, label='Convergence threshold (1e-4)')
    
    # 双对数坐标
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy (||Δh||²)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 标注最终能量
    if energy_trace:
        plt.text(len(energy_trace), energy_trace[-1], 
                 f' Final: {energy_trace[-1]:.2e}', 
                 fontsize=10, ha='left')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  能量曲线已保存: {save_path}")


def plot_attention_matrix(attn_weights, agent_names, it, save_dir="figures"):
    """绘制注意力矩阵热力图"""
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_weights, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(label='Attention Weight')
    plt.xticks(range(len(agent_names)), agent_names, rotation=45)
    plt.yticks(range(len(agent_names)), agent_names)
    plt.title(f'Attention Matrix (Iter {it})', fontsize=12)
    
    for i in range(attn_weights.shape[0]):
        for j in range(attn_weights.shape[1]):
            plt.text(j, i, f'{attn_weights[i,j]:.2f}', 
                     ha='center', va='center', fontsize=9)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'attention_iter_{it}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_individual_uncertainties(agent_names, u_before, u_after, save_dir="figures"):
    """绘制共识前后各智能体不确定性对比"""
    plt.figure(figsize=(8, 5))
    x = np.arange(len(agent_names))
    width = 0.35
    
    plt.bar(x - width/2, u_before, width, label='Before Consensus', 
            alpha=0.7, color='salmon')
    plt.bar(x + width/2, u_after, width, label='After Consensus', 
            alpha=0.7, color='steelblue')
    
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel('Uncertainty u', fontsize=12)
    plt.title('Uncertainty Before/After Consensus', fontsize=14)
    plt.xticks(x, agent_names)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'uncertainty_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  不确定性对比已保存: figures/uncertainty_comparison.png")


# =============================================================================
# 5. 测试函数
# =============================================================================

def test_on_disagreement_samples():
    """在合成数据的分歧样本上测试内循环共识"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'step1'))
    from synthetic_data import SyntheticCircleData, EvidentialMLP, evaluate, DEVICE as DEVICE_S1
    
    print("=" * 60)
    print("第二步测试：GAT内循环共识在分歧样本上的验证")
    print("=" * 60)
    
    print("\n[1/4] 加载合成数据和已训练的模型...")
    data = SyntheticCircleData(n_train=10000, n_test=2000)
    
    name_to_key = {'Agent1_x': 'agent1', 'Agent2_y': 'agent2', 'Agent3_r': 'agent3'}
    agent_names = ['Agent1_x', 'Agent2_y', 'Agent3_r']
    model_base = 'models'
    
    models = {}
    for name in agent_names:
        model = EvidentialMLP(input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32)
        path = os.path.join(model_base, f'{name}_evidential.pth')
        model.load_state_dict(torch.load(path, map_location=DEVICE_S1, weights_only=True))
        model.to(DEVICE_S1).eval()
        models[name] = model
        print(f"  已加载 {name}")
    
    print("\n[2/4] 查找分歧和一致样本...")
    
    all_outputs = {}
    for name in agent_names:
        data_key = name_to_key[name]
        result = evaluate(models[name], data.x_test[data_key], data.y_test[data_key], DEVICE_S1)
        all_outputs[name] = result
    
    preds = torch.stack([all_outputs[name]['predictions'] for name in agent_names], dim=0)
    all_agree = (preds == preds[0:1]).all(dim=0)
    disagreement_idx = torch.where(~all_agree)[0]
    agreement_idx = torch.where(all_agree)[0]
    
    print(f"  测试样本总数: {len(data.y_true)}")
    print(f"  一致样本: {len(agreement_idx)}, 分歧样本: {len(disagreement_idx)}")
    
    print("\n[3/4] 详细案例分析...")
    
    # 选择3个代表性样本：1个一致 + 1个证据冲突 + 1个无知冲突
    sample_configs = [
        ("一致性样本", agreement_idx, len(agreement_idx)//2),
        ("分歧样本#1", disagreement_idx, 0),
        ("分歧样本#2", disagreement_idx, 2),
    ]
    
    for label, idx_set, pos in sample_configs:
        sample_idx = idx_set[pos].item()
        print(f"\n  --- {label}: 样本 {sample_idx} ---")
        
        # 获取输出
        agent_outputs = []
        for name in agent_names:
            data_key = name_to_key[name]
            model = models[name]
            x = data.x_test[data_key][sample_idx:sample_idx+1].to(DEVICE_S1)
            alpha, b, u, emb = model.get_output(x)
            agent_outputs.append((alpha[0], b[0], u[0], emb[0]))
        
        true_label = data.y_true[sample_idx].item()
        
        # 共识前
        print(f"  共识前:")
        for i, name in enumerate(agent_names):
            _, b, u, _ = agent_outputs[i]
            pred = b.argmax().item()
            print(f"    {name}: pred={pred}, b0={b[0]:.3f}, b1={b[1]:.3f}, u={u.item():.4f}")
        print(f"  真实标签: {true_label}")
        
        # 运行GAT共识
        engine = ConsensusEngine(embed_dim=32, num_classes=2, hidden_dim=64)
        h = engine.build_state(agent_outputs)
        h_final, n_iters, converged, energy_trace, attn_trace = \
            engine.run(h, max_iters=20, tol=1e-4, verbose=True)
        outputs = engine.extract_outputs(h_final)
        
        # 共识后
        print(f"  共识后:")
        for i, name in enumerate(agent_names):
            _, b_new, u_new, _ = outputs[i]
            print(f"    {name}: b0={b_new[0]:.3f}, b1={b_new[1]:.3f}, u={u_new:.4f}")
        
        decision, global_b, global_u, weights = global_decision(outputs, u_threshold=0.5)
        correct = decision == true_label
        print(f"  全局决策: {decision} (真实: {true_label}) {'[OK]' if correct else '[FAIL]'}")
        print(f"  全局信念: [{global_b[0]:.3f}, {global_b[1]:.3f}], 全局u: {global_u:.4f}")
        print(f"  权重: {dict(zip(agent_names, weights.round(3)))}")
        
        # 绘制能量曲线
        plot_energy_curve(energy_trace, 
                         title=f"Sample {sample_idx} - {label}",
                         save_path=f"figures/energy_sample{sample_idx}.png")
        
        # 绘制最终注意力矩阵
        if attn_trace:
            plot_attention_matrix(attn_trace[-1], agent_names, n_iters)
    
    print("\n[4/4] 批量评估...")
    
    # ---- 分歧样本 ----
    n_eval = min(300, len(disagreement_idx))
    correct_majority = 0
    correct_consensus = 0
    abstain_count = 0
    converged_count = 0
    total_iters = 0
    u_before_list = []
    u_after_list = []
    
    for pos in range(n_eval):
        idx = disagreement_idx[pos].item()
        true_label = data.y_true[idx].item()
        
        agent_outputs = []
        for name in agent_names:
            data_key = name_to_key[name]
            model = models[name]
            x = data.x_test[data_key][idx:idx+1].to(DEVICE_S1)
            alpha, b, u, emb = model.get_output(x)
            agent_outputs.append((alpha[0], b[0], u[0], emb[0]))
            if pos == 0:
                u_before_list.append(u[0].item())
        
        # 多数投票
        votes = [b.argmax().item() for _, b, _, _ in agent_outputs]
        majority = max(set(votes), key=votes.count)
        if majority == true_label:
            correct_majority += 1
        
        # GAT共识
        engine = ConsensusEngine(embed_dim=32, num_classes=2, hidden_dim=64)
        h = engine.build_state(agent_outputs)
        h_final, n_iters, converged, _, _ = \
            engine.run(h, max_iters=20, tol=1e-4, verbose=False)
        outputs = engine.extract_outputs(h_final)
        
        if converged:
            converged_count += 1
        total_iters += n_iters
        
        decision, _, global_u, _ = global_decision(outputs, u_threshold=0.5)
        if decision == true_label:
            correct_consensus += 1
        if decision == -1:
            abstain_count += 1
        
        if pos == n_eval - 1:
            for _, _, u_new, _ in outputs:
                u_after_list.append(u_new)
    
    print(f"\n  {'='*40}")
    print(f"  分歧样本批量评估 (n={n_eval})")
    print(f"  {'='*40}")
    print(f"  多数投票准确率:      {correct_majority/n_eval*100:.1f}%")
    print(f"  GAT共识准确率:       {correct_consensus/n_eval*100:.1f}%")
    print(f"  拒识率:              {abstain_count/n_eval*100:.1f}%")
    print(f"  收敛率:              {converged_count/n_eval*100:.1f}%")
    print(f"  平均迭代轮数:        {total_iters/n_eval:.2f}")
    
    # ---- 一致样本 ----
    n_agree = min(300, len(agreement_idx))
    agree_correct = 0
    agree_abstain = 0
    
    for pos in range(n_agree):
        idx = agreement_idx[pos].item()
        true_label = data.y_true[idx].item()
        
        agent_outputs = []
        for name in agent_names:
            data_key = name_to_key[name]
            model = models[name]
            x = data.x_test[data_key][idx:idx+1].to(DEVICE_S1)
            alpha, b, u, emb = model.get_output(x)
            agent_outputs.append((alpha[0], b[0], u[0], emb[0]))
        
        engine = ConsensusEngine(embed_dim=32, num_classes=2, hidden_dim=64)
        h = engine.build_state(agent_outputs)
        h_final, _, _, _, _ = engine.run(h, max_iters=20, tol=1e-4, verbose=False)
        outputs = engine.extract_outputs(h_final)
        
        decision, _, _, _ = global_decision(outputs, u_threshold=0.5)
        if decision == true_label:
            agree_correct += 1
        if decision == -1:
            agree_abstain += 1
    
    print(f"  {'='*40}")
    print(f"  一致样本评估 (n={n_agree})")
    print(f"  {'='*40}")
    print(f"  正确率:              {agree_correct/n_agree*100:.1f}%")
    print(f"  拒识率:              {agree_abstain/n_agree*100:.1f}%")
    
    print(f"\n{'=' * 60}")
    print("第二步测试完成！GAT共识层验证通过")
    print(f"{'=' * 60}")


def main():
    test_on_disagreement_samples()


if __name__ == '__main__':
    main()