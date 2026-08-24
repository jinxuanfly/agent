"""
第五步：因果反事实反思（Causal Counterfactual Reflection）
========================================================
当证据交换失败后，触发因果归因与反事实修正。

核心组件：
1. FeatureAttributor: 基于Captum Integrated Gradients的特征归因
2. CounterfactualEngine: 反事实修改（掩码/扰动重要特征）
3. CompensationPrompter: 生成补偿提示（掩码/特征忽略列表）
4. CausalReflectionLoop: 外循环反思机制

理论依据：
- Integrated Gradients满足敏感性和实现不变性公理
- 反事实推理：如果改变某特征，输出如何变化？
- 补偿提示是"注意力的重新定向"：告诉模型忽略误导特征
- 最多反思3次，仍不收敛则最终拒识

操作流程（对合成数据）：
1. 归因：计算每个输入特征对输出的贡献
2. 掩码：找到最重要的冲突特征并遮罩
3. 重新输入：用遮罩后的特征重新计算智能体状态
4. 检查：如果共识达成则停止，否则继续下一次反思
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import List, Tuple, Dict, Optional, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.synthetic_data import SEED, DEVICE

# =============================================================================
# 1. FeatureAttributor - 基于Integrated Gradients的特征归因
# =============================================================================

try:
    from captum.attr import IntegratedGradients, GradientShap, Occlusion
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("警告: captum未安装，将使用简化归因方法。 pip install captum")


class SimpleIntegratedGradients:
    """
    简化的Integrated Gradients实现（不依赖captum）
    
    原理：
    IG_i(x) = (x_i - x'_i) * ∫_{α=0}^{1} ∂F(x' + α*(x-x'))/∂x_i dα
    
    通过Riemann求和近似积分：
    IG_i(x) ≈ (x_i - x'_i) * (1/m) * Σ_{k=1}^{m} ∂F(x' + k/m*(x-x'))/∂x_i
    """
    
    def __init__(self, model: nn.Module, target_fn: Callable, n_steps: int = 50):
        self.model = model
        self.target_fn = target_fn
        self.n_steps = n_steps
        
    def attribute(self, x: torch.Tensor, baseline: Optional[torch.Tensor] = None,
                  target_class: Optional[int] = None) -> torch.Tensor:
        """
        计算输入特征的归因值
        """
        x_in = x.clone().detach().requires_grad_(True)
        if baseline is None:
            baseline = torch.zeros_like(x_in)
        if target_class is None:
            with torch.no_grad():
                alpha, _, _, _ = self.model(x_in)
                target_class = alpha.argmax(dim=1).item()
        
        # 逐点梯度（简化：仅在输入点计算一次梯度）
        output = self.target_fn(self.model, x_in)
        if output.dim() > 1:
            output = output[:, target_class]
        output.backward()
        grad = x_in.grad.clone().detach() if x_in.grad is not None else torch.zeros_like(x_in)
        
        # 简化的IG：用当前梯度近似积分
        attributions = (x_in - baseline) * grad
        return attributions


class FeatureAttributor:
    """
    特征归因器：定位每个智能体的重要特征
    """
    
    def __init__(self, model: nn.Module, feature_dim: int, is_image_model: bool = False):
        self.model = model
        self.feature_dim = feature_dim
        self.is_image_model = is_image_model
        
        def target_fn(model, x):
            alpha, _, _, _ = model(x)
            return alpha
        
        self.attributor = SimpleIntegratedGradients(model, target_fn)
    
    def get_attributions(self, x: torch.Tensor, 
                         target_class: Optional[int] = None) -> np.ndarray:
        """
        获取特征归因
        """
        x = x.clone().detach().to(DEVICE)
        x.requires_grad_(True)
        
        if self.is_image_model:
            return self._image_attribution(x, target_class)
        else:
            return self._numerical_attribution(x, target_class)
    
    def _numerical_attribution(self, x: torch.Tensor, 
                                target_class: Optional[int] = None) -> np.ndarray:
        """数值特征归因"""
        attr = self.attributor.attribute(x, target_class=target_class)
        return attr.detach().cpu().numpy()
    
    def _image_attribution(self, x: torch.Tensor,
                            target_class: Optional[int] = None) -> np.ndarray:
        """图像特征归因（基于patch的简化方法）"""
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        patch_size = 8
        attr_map = np.zeros((H, W))
        
        with torch.no_grad():
            base_alpha, _, _, _ = self.model(x)
            if target_class is None:
                target_class = base_alpha.argmax(dim=1).item()
            base_logit = base_alpha[0, target_class].item()
        
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                x_masked = x.clone()
                x_masked[:, :, i:min(i+patch_size, H), j:min(j+patch_size, W)] = 0.0
                
                with torch.no_grad():
                    masked_alpha, _, _, _ = self.model(x_masked)
                    masked_logit = masked_alpha[0, target_class].item()
                
                importance = base_logit - masked_logit
                attr_map[i:min(i+patch_size, H), j:min(j+patch_size, W)] = importance
        
        return attr_map[np.newaxis, np.newaxis, :, :]


# =============================================================================
# 2. CounterfactualEngine - 反事实修改
# =============================================================================

class CounterfactualEngine:
    """
    反事实引擎：对重要特征进行掩码/扰动并观察输出变化
    """
    
    def __init__(self, model: nn.Module, feature_dim: int, 
                 is_image_model: bool = False, threshold: float = 0.3):
        self.model = model
        self.feature_dim = feature_dim
        self.is_image_model = is_image_model
        self.threshold = threshold
    
    def analyze_and_mask(self, x: torch.Tensor, attributions: np.ndarray,
                         mask_ratio: float = 0.3) -> Tuple[torch.Tensor, Dict]:
        """
        分析归因并生成掩码
        """
        info = {}
        
        if self.is_image_model:
            x_masked, mask_info = self._mask_image(x, attributions, mask_ratio)
            info['mask'] = mask_info['mask']
            info['masked_regions'] = mask_info['regions']
        else:
            x_masked, mask_info = self._mask_numerical(x, attributions, mask_ratio)
            info['masked_indices'] = mask_info['indices']
            info['mask'] = mask_info['mask']
        
        info['original_output'] = self._get_output(x)
        info['masked_output'] = self._get_output(x_masked)
        info['change_significant'] = self._check_significant_change(
            info['original_output'], info['masked_output']
        )
        
        return x_masked, info
    
    def _mask_numerical(self, x: torch.Tensor, attributions: np.ndarray,
                         mask_ratio: float) -> Tuple[torch.Tensor, Dict]:
        """遮罩数值特征：将top-k特征设为0"""
        attr_flat = np.abs(attributions.flatten())
        n_mask = max(1, int(len(attr_flat) * mask_ratio))
        top_indices = np.argsort(attr_flat)[::-1][:n_mask]
        
        x_masked = x.clone()
        mask_indices = []
        
        for idx in top_indices:
            if idx < x.shape[1]:
                x_masked[0, idx] = 0.0
                mask_indices.append(int(idx))
        
        info = {'indices': mask_indices, 'mask': None}
        return x_masked, info
    
    def _mask_image(self, x: torch.Tensor, attributions: np.ndarray,
                     mask_ratio: float) -> Tuple[torch.Tensor, Dict]:
        """遮罩图像特征"""
        H, W = x.shape[2], x.shape[3]
        patch_size = 8
        
        if len(attributions.shape) == 4:
            attr_map = np.abs(attributions[0, 0])
        else:
            attr_map = np.abs(attributions).squeeze()
        
        n_patches_h = H // patch_size
        n_patches_w = W // patch_size
        patch_attributions = np.zeros((n_patches_h, n_patches_w))
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch_attr = attr_map[
                    i*patch_size:(i+1)*patch_size, 
                    j*patch_size:(j+1)*patch_size
                ]
                patch_attributions[i, j] = patch_attr.mean()
        
        patch_flat = patch_attributions.flatten()
        n_mask = max(1, int(len(patch_flat) * mask_ratio))
        top_patches = np.argsort(patch_flat)[::-1][:n_mask]
        
        x_masked = x.clone()
        mask = torch.zeros((1, 1, H, W))
        masked_regions = []
        
        for patch_idx in top_patches:
            i = patch_idx // n_patches_w
            j = patch_idx % n_patches_w
            h_start, h_end = i * patch_size, (i + 1) * patch_size
            w_start, w_end = j * patch_size, (j + 1) * patch_size
            x_masked[:, :, h_start:h_end, w_start:w_end] = 0.0
            mask[:, :, h_start:h_end, w_start:w_end] = 1.0
            masked_regions.append((i, j))
        
        info = {'mask': mask, 'regions': masked_regions}
        return x_masked, info
    
    def _get_output(self, x: torch.Tensor) -> Dict:
        """获取模型输出"""
        with torch.no_grad():
            alpha, belief, uncertainty, _ = self.model(x)
            return {
                'alpha': alpha.cpu().numpy(),
                'belief': belief.cpu().numpy(),
                'uncertainty': uncertainty.cpu().numpy(),
                'prediction': belief.argmax(dim=1).item(),
            }
    
    def _check_significant_change(self, output1: Dict, output2: Dict) -> bool:
        """检查输出是否显著变化"""
        b1 = output1['belief'].flatten()
        b2 = output2['belief'].flatten()
        max_diff = np.max(np.abs(b1 - b2))
        pred_change = output1['prediction'] != output2['prediction']
        return max_diff > self.threshold or pred_change


# =============================================================================
# 3. CompensationPrompter - 补偿提示生成
# =============================================================================

class CompensationPrompter:
    """
    补偿提示生成器：根据反事实分析结果生成补偿提示
    """
    
    def __init__(self):
        self.counter = 0
    
    def generate_prompt(self, agent_name: str, counterfactual_info: Dict,
                         attributions: np.ndarray) -> Dict:
        """生成补偿提示"""
        self.counter += 1
        
        prompt = {
            'agent': agent_name,
            'type': 'feature_mask',
            'mask': counterfactual_info.get('masked_indices', counterfactual_info.get('mask', None)),
            'attributions': attributions,
            'round': self.counter,
            'instruction': self._create_instruction(agent_name, counterfactual_info),
        }
        return prompt
    
    def _create_instruction(self, agent_name: str, info: Dict) -> str:
        """生成人可读的指令"""
        if 'masked_indices' in info:
            return f"Ignore features at indices {info['masked_indices']} for {agent_name}"
        elif 'masked_regions' in info:
            return f"Ignore {len(info['masked_regions'])} patches for {agent_name}"
        else:
            return f"Apply feature mask for {agent_name}"
    
    def apply_prompt(self, model, x: torch.Tensor, prompt: Dict) -> torch.Tensor:
        """应用补偿提示到输入"""
        x_modified = x.clone()
        mask = prompt['mask']
        
        if prompt['type'] == 'feature_mask':
            if isinstance(mask, list):
                for idx in mask:
                    if idx < x_modified.shape[1]:
                        x_modified[0, idx] = 0.0
            elif isinstance(mask, torch.Tensor):
                x_modified = x_modified * (1 - mask.to(x.device))
        
        return x_modified


# =============================================================================
# 4. CausalReflectionLoop - 外循环反思机制
# =============================================================================

class CausalReflectionLoop:
    """
    因果反事实反思外循环
    
    流程：
    1. 对每个分歧智能体进行特征归因
    2. 生成反事实修改（掩码最重要特征）
    3. 验证修改是否显著改变输出
    4. 如果改变显著，生成补偿提示并应用
    5. 重新进入内循环共识
    6. 如果仍不收敛，最多再反思2次
    7. 超过3次则最终拒识
    """
    
    def __init__(self, agents: Dict[str, nn.Module], 
                 inner_consensus_fn: Callable,
                 max_reflections: int = 3,
                 rejection_threshold: float = 0.5):
        """
        Args:
            agents: {name: model} 智能体字典
            inner_consensus_fn: 内循环共识函数，签名为 fn(states_tensor, max_iters) -> dict
            max_reflections: 最大反思次数
            rejection_threshold: 拒识阈值
        """
        self.agents = agents
        self.inner_consensus_fn = inner_consensus_fn
        self.max_reflections = max_reflections
        self.rejection_threshold = rejection_threshold
        
        self.attributors = {}
        self.counterfactual_engines = {}
        self.prompters = {}
        
        for name, model in agents.items():
            is_image = 'ResNet' in name or 'ViT' in name or 'Pixel' in name
            feature_dim = self._get_feature_dim(model)
            
            self.attributors[name] = FeatureAttributor(
                model, feature_dim, is_image_model=is_image
            )
            self.counterfactual_engines[name] = CounterfactualEngine(
                model, feature_dim, is_image_model=is_image
            )
            self.prompters[name] = CompensationPrompter()
        
        self.reflection_log = []
    
    def _get_feature_dim(self, model) -> int:
        """推断模型输入特征维度"""
        try:
            name = model.__class__.__name__
            if 'ResNet' in name or 'ViT' in name:
                return 3 * 32 * 32
            elif 'PixelMLP' in name:
                return 3 * 32 * 32
            else:
                if hasattr(model, 'layers'):
                    return model.layers[0].in_features
                return 2
        except:
            return 10
    
    def reflect(self, sample_inputs: Dict[str, torch.Tensor],
                initial_states: Optional[torch.Tensor] = None,
                initial_alphas: Optional[Dict[str, torch.Tensor]] = None,
                ground_truth: Optional[int] = None) -> Dict:
        """
        执行因果反思循环
        
        Args:
            sample_inputs: {agent_name: tensor_input} 
            initial_states: 共识初始状态 [N, D]
            initial_alphas: {agent_name: alpha} 原始Dirichlet参数
            ground_truth: 真实类别标签
        
        Returns:
            result: {
                'converged': bool,
                'final_prediction': int or None,
                'reflections': int,
                'rejected': bool,
                'prompts': list of prompts,
                'states': consensus states,
                'log': list of log entries,
            }
        """
        self.reflection_log = []
        
        result = {
            'converged': False,
            'final_prediction': None,
            'reflections': 0,
            'rejected': False,
            'prompts': [],
            'states': None,
            'log': [],
        }
        
        current_inputs = {k: v.clone() for k, v in sample_inputs.items()}
        
        for reflection_round in range(self.max_reflections):
            result['reflections'] = reflection_round + 1
            
            # 步骤1: 获取当前智能体状态并尝试共识
            states = self._get_build_states(current_inputs)
            
            # 步骤2: 尝试内循环共识
            if states is not None and states.shape[0] >= 2:
                consensus_result = self.inner_consensus_fn(states, max_iters=20)
                
                if consensus_result.get('converged', False):
                    final_states = consensus_result.get('final_states', states)
                    final_pred = self._extract_prediction(final_states)
                    result['converged'] = True
                    result['final_prediction'] = final_pred
                    result['states'] = final_states
                    break
            
            # 步骤3: 共识失败，对每个智能体进行归因和修正
            target_agents = list(self.agents.keys())
            round_prompts = []
            
            for agent_name in target_agents:
                if agent_name not in current_inputs:
                    continue
                
                x = current_inputs[agent_name]
                
                # 归因
                attributor = self.attributors[agent_name]
                attributions = attributor.get_attributions(x)
                
                # 反事实分析
                counterfactual = self.counterfactual_engines[agent_name]
                x_masked, cf_info = counterfactual.analyze_and_mask(x, attributions)
                
                # 生成补偿提示
                prompter = self.prompters[agent_name]
                prompt = prompter.generate_prompt(agent_name, cf_info, attributions)
                round_prompts.append(prompt)
                
                # 应用补偿
                x_modified = prompter.apply_prompt(self.agents[agent_name], x, prompt)
                current_inputs[agent_name] = x_modified
            
            result['prompts'].extend(round_prompts)
            
            log_entry = {
                'round': reflection_round + 1,
                'target_agents': target_agents,
                'prompts': round_prompts,
                'consensus_result': consensus_result,
            }
            self.reflection_log.append(log_entry)
            result['log'].append(log_entry)
        
        if not result['converged']:
            result['rejected'] = True
            result['final_prediction'] = None
        
        return result
    
    def _get_build_states(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """构建状态矩阵 [N, D] 用于共识"""
        states = []
        for name, model in self.agents.items():
            x = inputs.get(name)
            if x is None:
                continue
            with torch.no_grad():
                # EvidentialMLP: .get_output(x) 返回 (alpha, b, u, embedding)
                alpha, belief, uncertainty, embedding = model.get_output(x)
            state = torch.cat([
                embedding.squeeze(0).flatten(),
                belief.squeeze(0).flatten(),
                uncertainty.squeeze(0).reshape(-1),
            ])
            states.append(state)
        
        if len(states) < 2:
            return None
        return torch.stack(states)
    
    def _extract_prediction(self, states: torch.Tensor) -> int:
        """从共识状态提取预测"""
        if states is None or states.shape[0] == 0:
            return -1
        
        # 取平均状态的信念部分（最后几维中除最后一维外的部分）
        mean_state = states.mean(dim=0)
        if mean_state.dim() == 1:
            # 假设最后是 classes + 1 维（信念 + 不确定性）
            # 使用后部：-2, -3 等作为信念
            d = len(mean_state)
            # 尝试从后往前取2维作为信念（二分类）
            belief_part = mean_state[d-3:d-1] if d >= 3 else mean_state[max(0, d-2):d]
            return belief_part.argmax().item()
        return 0


# =============================================================================
# 5. 合成数据测试
# =============================================================================

def _make_consensus_wrapper(engine):
    """将ConsensusEngine.run包装为内循环共识函数"""
    def inner_fn(states, max_iters=20):
        h_final, n_iters, converged, energy_trace, attn_trace = \
            engine.run(states, max_iters=max_iters, tol=1e-4, verbose=False)
        return {
            'converged': converged,
            'final_states': h_final,
            'n_iters': n_iters,
            'energy_trace': energy_trace,
            'attn_trace': attn_trace,
        }
    return inner_fn


def test_causal_reflection():
    """在合成数据上测试因果反事实反思"""
    print("=" * 60)
    print("因果反事实反思测试（合成数据）")
    print("=" * 60)
    
    from step1.synthetic_data import (
        HardConflictCircleData, EvidentialMLP, DEVICE
    )
    from step2.gat_consensus import ConsensusEngine
    
    # ====== 加载预训练模型 ======
    print("\n[1] 加载预训练模型...")
    agent_names = ['Agent1_x', 'Agent2_y', 'Agent3_r']
    agents = {}
    for name in agent_names:
        model = EvidentialMLP(input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32)
        path = f'models/{name}_evidential_hard.pth'
        if not os.path.exists(path):
            path = f'models/{name}_evidential.pth'
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            print(f"  [OK] 已加载 {name} 从 {path}")
        else:
            print(f"  [WARN] {path} 不存在，使用未训练模型")
        model.to(DEVICE).eval()
        agents[name] = model
    
    # ====== 生成硬样本 ======
    print("\n[2] 生成硬样本...")
    data = HardConflictCircleData(n_train=1000, n_test=200, noise_level=0.15, flip_ratio=0.15, hard_ratio=0.25)
    
    # 构造统一的测试张量 (x, y, r)
    test_x = data.x_test  # dict: {agent1: [N,1], agent2: [N,1], agent3: [N,1]}
    test_y = data.y_true  # [N] 真实标签
    
    # 构建一个"统一特征张量"，每行=[x, y, r] 方便迭代
    x_coords = torch.stack([
        test_x['agent1'].squeeze(),
        test_x['agent2'].squeeze(),
    ], dim=1)  # [N, 2]
    r_vals = test_x['agent3']  # [N, 1]
    unified_x = torch.cat([x_coords, r_vals], dim=1)  # [N, 3]
    
    # ====== 创建共识引擎包装 ======
    engine = ConsensusEngine(embed_dim=32, num_classes=2, hidden_dim=64)
    consensus_fn = _make_consensus_wrapper(engine)
    
    # ====== 创建反思循环 ======
    reflection_loop = CausalReflectionLoop(
        agents=agents,
        inner_consensus_fn=consensus_fn,
        max_reflections=3,
    )
    
    success_count = 0
    total_reflections = 0
    n_test = min(20, len(unified_x))
    
    for idx in range(n_test):
        x_i = unified_x[idx:idx+1].to(DEVICE)  # [1, 3]
        y_i = test_y[idx].item()
        
        sample_inputs = {
            'Agent1_x': x_i[:, [0]],    # x坐标 [1,1]
            'Agent2_y': x_i[:, [1]],    # y坐标 [1,1]
            'Agent3_r': x_i[:, 2:],     # 半径 [1,1]
        }
        
        result = reflection_loop.reflect(
            sample_inputs=sample_inputs,
            initial_states=None,
            initial_alphas=None,
            ground_truth=y_i,
        )
        
        total_reflections += result['reflections']
        
        if result['converged']:
            success_count += 1
            status = "[OK]"
        else:
            status = "[FAIL]"
        
        if idx < 5 or not result['converged']:
            print(f"  样本{idx:3d} (真值={y_i}): {status} "
                  f"反思={result['reflections']}轮, "
                  f"预测={result['final_prediction']}, "
                  f"{'拒识' if result['rejected'] else '一致'}")
    
    print(f"\n[3] 统计结果 ({success_count}/{n_test}):")
    avg_r = total_reflections / max(1, n_test)
    print(f"  平均反思轮数: {avg_r:.1f}")
    print(f"  成功率: {success_count / max(1, n_test)*100:.1f}%")
    
    return reflection_loop, result


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    test_causal_reflection()