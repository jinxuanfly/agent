"""
LLM增强的Hateful Memes评估管线
==============================
用大模型API替代本地弱Agent，保留完整共识框架。

核心替换:
  本地编码器 + 证据头 → 3个LLM Agent（不同模型/策略）
  保留: GAT共识层 / DS融合 / 分歧解构 / EMNet

用法:
  1. 设置API key环境变量:
     set DEEPSEEK_API_KEY=sk-xxx
     set QWEN_API_KEY=sk-xxx
     set GLM_API_KEY=sk-xxx
  
  2. 运行评估:
     python src/step4_hateful_memes/evaluate_with_llm.py --max_val 200
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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from plot_utils import setup_chinese_font, setup_plot_style
setup_chinese_font()
setup_plot_style()

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from step2.gat_consensus import ConsensusEngine, GATConsensusLayer

# LLM Agent
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.llm_agent import LLMAgent, BatchLLMProcessor, create_llm_agents, create_single_agent, AGENT_PROMPTS
from src.llm_api import LLMClient, BatchClassifier, PROVIDER_CONFIGS

warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# 配置
# =============================================================================

NUM_CLASSES = 2
U_THRESHOLD = 0.3

DATA_DIR = 'data/Hateful_Memes/data'
CHECKPOINT_DIR = 'checkpoints/hateful_memes'
FIGURE_DIR = 'figures'
RESULT_DIR = 'results/hateful_memes'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 设备（LLM API不需要CUDA，但GAT可能需要CPU推理）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


# =============================================================================
# 样本预检查工具
# =============================================================================

def pre_check_dataset(max_train=200, max_val=200, provider1='deepseek', provider2='glm', provider3='gpt'):
    """
    运行前预检查：验证样本质量、标签分布、API配置
    
    检查项：
    1. 数据集完整性（JSON文件是否存在）
    2. 标签分布平衡（分层抽样后是否满足阈值）
    3. 图像可用性（图像文件是否存在、能否正常加载）
    4. 文本内容完整性（是否有空文本）
    5. API密钥配置（是否已设置）
    
    返回：
        bool: True=检查通过, False=检查失败
    """
    print("=" * 70)
    print("[PRE-CHECK] 实验前预检查")
    print("=" * 70)
    
    all_passed = True
    issues = []
    
    # ========== 检查1: 数据集文件 ==========
    print("\n[检查1/5] 数据集文件")
    required_files = ['train.jsonl', 'dev.jsonl']
    missing_files = []
    for fname in required_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            # 尝试json格式
            fpath_json = fpath.replace('.jsonl', '.json')
            if not os.path.exists(fpath_json):
                missing_files.append(fname)
    
    if missing_files:
        print(f"  [FAIL] 缺失数据集文件: {', '.join(missing_files)}")
        all_passed = False
        issues.append(f"缺失数据集文件: {', '.join(missing_files)}")
    else:
        print("  [OK] 数据集文件完整")
    
    # ========== 检查2: 标签分布 ==========
    print("\n[检查2/5] 标签分布平衡")
    try:
        train_dataset = HatefulMemesDataset(split='train', max_samples=max_train, load_images=False, stratified=True)
        val_dataset = HatefulMemesDataset(split='dev', max_samples=max_val, load_images=False, stratified=True)
        
        train_labels = train_dataset.labels
        val_labels = val_dataset.labels
        
        def _check_dist(labels, name, threshold=0.6):
            counts = {0: labels.count(0), 1: labels.count(1)}
            total = len(labels)
            if total == 0:
                return False, f"{name}为空"
            ratio0 = counts[0] / total
            ratio1 = counts[1] / total
            max_ratio = max(ratio0, ratio1)
            passed = max_ratio <= threshold
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} [{name}] 标签0={counts[0]} ({ratio0*100:.1f}%), 标签1={counts[1]} ({ratio1*100:.1f}%), 最大占比={max_ratio*100:.1f}%")
            if not passed:
                return False, f"{name}标签分布不平衡: 最大占比{max_ratio*100:.1f}% > {threshold*100:.0f}%"
            return True, None
        
        train_ok, train_issue = _check_dist(train_labels, "训练集")
        val_ok, val_issue = _check_dist(val_labels, "验证集")
        
        if not train_ok:
            all_passed = False
            issues.append(train_issue)
        if not val_ok:
            all_passed = False
            issues.append(val_issue)
            
    except Exception as e:
        print(f"  [FAIL] 加载数据集失败: {e}")
        all_passed = False
        issues.append(f"加载数据集失败: {e}")
    
    # ========== 检查3: 图像可用性 ==========
    print("\n[检查3/5] 图像可用性")
    try:
        train_dataset = HatefulMemesDataset(split='train', max_samples=min(max_train, 50), load_images=True, stratified=True)
        val_dataset = HatefulMemesDataset(split='dev', max_samples=min(max_val, 50), load_images=True, stratified=True)
        
        train_img_ok = sum(1 for img in train_dataset.images if img is not None)
        train_img_total = len(train_dataset.images)
        val_img_ok = sum(1 for img in val_dataset.images if img is not None)
        val_img_total = len(val_dataset.images)
        
        train_ratio = train_img_ok / train_img_total * 100
        val_ratio = val_img_ok / val_img_total * 100
        
        status_train = "[OK]" if train_ratio >= 90 else "[WARN]" if train_ratio >= 70 else "[FAIL]"
        status_val = "[OK]" if val_ratio >= 90 else "[WARN]" if val_ratio >= 70 else "[FAIL]"
        
        print(f"  {status_train} 训练集图像: {train_img_ok}/{train_img_total} ({train_ratio:.1f}%)")
        print(f"  {status_val} 验证集图像: {val_img_ok}/{val_img_total} ({val_ratio:.1f}%)")
        
        if train_ratio < 70 or val_ratio < 70:
            all_passed = False
            issues.append(f"图像加载率过低: 训练集{train_ratio:.1f}%, 验证集{val_ratio:.1f}%")
        elif train_ratio < 90 or val_ratio < 90:
            print(f"  [WARN] 部分图像加载失败，建议检查数据路径")
            
    except Exception as e:
        print(f"  [FAIL] 图像检查失败: {e}")
        all_passed = False
        issues.append(f"图像检查失败: {e}")
    
    # ========== 检查4: 文本内容完整性 ==========
    print("\n[检查4/5] 文本内容完整性")
    empty_train = sum(1 for t in train_dataset.texts if not t or t.strip() == "")
    empty_val = sum(1 for t in val_dataset.texts if not t or t.strip() == "")
    
    status_train = "[OK]" if empty_train == 0 else "[WARN]" if empty_train < 5 else "[FAIL]"
    status_val = "[OK]" if empty_val == 0 else "[WARN]" if empty_val < 5 else "[FAIL]"
    
    print(f"  {status_train} 训练集空文本: {empty_train}/{len(train_dataset.texts)}")
    print(f"  {status_val} 验证集空文本: {empty_val}/{len(val_dataset.texts)}")
    
    if empty_train >= 5 or empty_val >= 5:
        all_passed = False
        issues.append(f"空文本过多: 训练集{empty_train}个, 验证集{empty_val}个")
    
    # ========== 检查5: API密钥配置 ==========
    print("\n[检查5/5] API密钥配置")
    providers = [provider1, provider2, provider3]
    missing_keys = []
    
    for i, prov in enumerate(providers):
        if prov == 'mock':
            print(f"  [WARN] Agent{i+1}: {prov} (模拟模式)")
            continue
        
        key_env = PROVIDER_CONFIGS[prov].get('env_key')
        if key_env and not os.getenv(key_env):
            # 尝试从keys.env加载
            keys_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'keys.env')
            if os.path.exists(keys_env_path):
                with open(keys_env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip().startswith(key_env):
                            # 找到密钥
                            break
                    else:
                        missing_keys.append(f"Agent{i+1} ({prov})")
            else:
                missing_keys.append(f"Agent{i+1} ({prov})")
        elif not key_env:
            print(f"  [WARN] Agent{i+1}: {prov} (无API密钥配置)")
        else:
            print(f"  [OK] Agent{i+1}: {prov} (API密钥已配置)")
    
    if missing_keys:
        print(f"  [FAIL] 缺少API密钥: {', '.join(missing_keys)}")
        all_passed = False
        issues.append(f"缺少API密钥: {', '.join(missing_keys)}")
        print(f"  [INFO] 提示: 请在环境变量或keys.env文件中设置API密钥")
    
    # ========== 总结 ==========
    print("\n" + "=" * 70)
    if all_passed:
        print("[PASS] 预检查全部通过！可以运行实验")
        print("=" * 70)
        return True
    else:
        print("[FAIL] 预检查失败，发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("[INFO] 请修复上述问题后再运行实验")
        print("=" * 70)
        return False


# =============================================================================
# DS融合决策函数（与原始管线完全相同）
# =============================================================================

def ds_fusion_decision(all_beliefs, all_uncertainties, u_threshold=0.5, agent_weights=None):
    """Dempster-Shafer融合决策"""
    B = all_beliefs[0].shape[0]
    C = all_beliefs[0].shape[1]
    N = len(all_beliefs)
    device = all_beliefs[0].device

    if agent_weights is None:
        agent_weights = torch.ones(N, device=device) / N

    b0 = all_beliefs[0]
    u0 = all_uncertainties[0]
    w0 = agent_weights[0]
    
    combined_belief = b0 * (1.0 - u0.unsqueeze(-1)) * w0
    combined_u = u0 * w0 + (1 - w0) * 0.5

    for b_idx in range(1, N):
        b = all_beliefs[b_idx]
        u = all_uncertainties[b_idx]
        w = agent_weights[b_idx]

        m1_b = combined_belief
        m1_u = combined_u
        m2_b = b * (1.0 - u.unsqueeze(-1)) * w
        m2_u = u * w + (1 - w) * 0.5

        sum_m1_b = m1_b.sum(dim=-1)
        sum_m2_b = m2_b.sum(dim=-1)
        agree = (m1_b * m2_b).sum(dim=-1)
        K = sum_m1_b * sum_m2_b - agree

        denom = 1.0 - K + 1e-8
        new_belief = (m1_b * m2_b + m1_b * m2_u.unsqueeze(-1) + m1_u.unsqueeze(-1) * m2_b) / denom.unsqueeze(-1)
        new_u = m1_u * m2_u / denom
        combined_belief = new_belief
        combined_u = new_u

    global_belief = combined_belief / (1.0 - combined_u.unsqueeze(-1) + 1e-8)
    global_u = combined_u
    preds = global_belief.argmax(dim=-1)
    rejected = global_u > u_threshold
    return preds, rejected, global_u


def generate_emnet_data_supervised(train_alphas, train_labels, n_synthetic=3000, num_classes=2, device='cpu'):
    """生成EMNet训练数据（与原始管线相同）"""
    N = train_alphas.shape[0]
    evidence_list = []
    target_list = []

    for i in range(N):
        alphas = train_alphas[i]
        S = alphas.sum(dim=-1)
        u = num_classes / S
        best_idx = u.argmin().item()
        sender_ev = (alphas[best_idx] - 1.0).detach()
        true_label = train_labels[i]
        target_ev = sender_ev.clone()
        target_ev[true_label] += 3.0
        for c in range(num_classes):
            if c != true_label and target_ev[c] > 0.5:
                target_ev[c] = max(0.1, target_ev[c] - 1.0)
        evidence_list.append(sender_ev)
        target_list.append(target_ev)

    syn_x = torch.rand(n_synthetic, num_classes, device=device) * 5.0 + 0.1
    syn_y = syn_x.clone() + torch.randn(n_synthetic, num_classes, device=device) * 0.5 + 1.0
    syn_y = F.relu(syn_y) + 0.1

    x = torch.stack(evidence_list + [syn_x])
    y = torch.stack(target_list + [syn_y])
    return x.to(device), y.to(device)


# =============================================================================
# 分歧解构器（与原始管线相同）
# =============================================================================

class DisagreementDeconstructor:
    """
    分歧解构器
    用Dempster-Shafer冲突系数K区分证据冲突与无知冲突
    """
    def __init__(self, u_threshold=0.5, K_threshold=0.3):
        self.u_threshold = u_threshold
        self.K_threshold = K_threshold

    def compute_pairwise_K(self, b1, u1, b2, u2):
        m1_b = b1 * (1.0 - u1)
        m2_b = b2 * (1.0 - u2)
        sum_m1_b = m1_b.sum(dim=-1)
        sum_m2_b = m2_b.sum(dim=-1)
        agree = (m1_b * m2_b).sum(dim=-1)
        K = sum_m1_b * sum_m2_b - agree
        return K

    def deconstruct(self, beliefs, uncertainties):
        """
        解构单个样本的三Agent分歧

        Returns:
            conflict_type: str, 'evidence_conflict', 'ignorance_conflict', 或 'none'
            avg_K: float
        """
        avg_us = uncertainties.mean().item()
        K01 = self.compute_pairwise_K(beliefs[0:1], uncertainties[0:1], beliefs[1:2], uncertainties[1:2])
        K02 = self.compute_pairwise_K(beliefs[0:1], uncertainties[0:1], beliefs[2:3], uncertainties[2:3])
        K12 = self.compute_pairwise_K(beliefs[1:2], uncertainties[1:2], beliefs[2:3], uncertainties[2:3])
        avg_K = float((K01 + K02 + K12).mean().item() / 3.0)

        if avg_K > self.K_threshold and avg_us < self.u_threshold:
            return 'evidence_conflict', avg_K
        elif avg_K > self.K_threshold and avg_us >= self.u_threshold:
            return 'ignorance_conflict', avg_K
        else:
            return 'none', avg_K

    def deconstruct_batch(self, all_beliefs, all_uncertainties):
        N = all_beliefs.shape[0]
        conflict_types = []
        K_values = []
        for i in range(N):
            ctype, K_val = self.deconstruct(all_beliefs[i], all_uncertainties[i])
            conflict_types.append(ctype)
            K_values.append(K_val)
        return conflict_types, K_values


# =============================================================================
# 小型EMNet（证据交换网络）
# =============================================================================

class SmallEMNet(nn.Module):
    """小型证据交换网络"""
    def __init__(self, num_classes=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 数据加载
# =============================================================================

class HatefulMemesDataset(Dataset):
    """Hateful Memes 数据集（支持图像加载 + 分层抽样）"""
    def __init__(self, split='train', max_samples=None, load_images=True, stratified=True):
        self.split = split
        self.max_samples = max_samples
        self.load_images = load_images
        self.stratified = stratified
        json_path = os.path.join(DATA_DIR, f'{split}.jsonl')
        if not os.path.exists(json_path):
            json_path = os.path.join(DATA_DIR, f'{split}.json')
        self.data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        if max_samples and max_samples < len(self.data):
            if self.stratified:
                self.data = self._stratified_sample(self.data, max_samples)
            else:
                self.data = self.data[:max_samples]
        
        self.texts = [item['text'] for item in self.data]
        self.labels = [item['label'] for item in self.data]
        self.img_paths = [os.path.join(DATA_DIR, item['img']) for item in self.data]
        self.images = []
        if self.load_images:
            for img_path in self.img_paths:
                if os.path.exists(img_path):
                    try:
                        self.images.append(Image.open(img_path).convert('RGB'))
                    except Exception as e:
                        print(f"  [警告] 加载图像失败 {img_path}: {e}")
                        self.images.append(None)
                else:
                    print(f"  [警告] 图像不存在: {img_path}")
                    self.images.append(None)
        
        label_counts = {0: self.labels.count(0), 1: self.labels.count(1)}
        total = len(self.data)
        print(f"  [{split}] 加载 {total} 样本 (图像: {sum(1 for img in self.images if img is not None)}/{total})")
        print(f"  [{split}] 标签分布: 0={label_counts[0]} ({label_counts[0]/total*100:.1f}%), 1={label_counts[1]} ({label_counts[1]/total*100:.1f}%)")
    
    def _stratified_sample(self, data, max_samples):
        """分层抽样：按label等比例采样，确保标签分布均匀"""
        import random
        labeled_data = {}
        for item in data:
            label = item['label']
            if label not in labeled_data:
                labeled_data[label] = []
            labeled_data[label].append(item)
        
        num_classes = len(labeled_data)
        samples_per_class = max_samples // num_classes
        remainder = max_samples % num_classes
        
        sampled_data = []
        for label in sorted(labeled_data.keys()):
            items = labeled_data[label]
            num_to_sample = samples_per_class + (1 if remainder > 0 else 0)
            remainder -= 1
            if num_to_sample > len(items):
                num_to_sample = len(items)
            sampled = random.sample(items, num_to_sample)
            sampled_data.extend(sampled)
        
        random.shuffle(sampled_data)
        return sampled_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = {'text': self.texts[idx], 'label': self.labels[idx], 'img_path': self.img_paths[idx]}
        if self.load_images:
            result['image'] = self.images[idx]
        return result


# =============================================================================
# 图像编码器（生成图像描述）
# =============================================================================

class ImageEncoder:
    """
    图像编码器：将图像转换为文本描述（用于异构多模态）
    
    设计：
    - Agent2（图像专家）只接收图像描述
    - Agent3（跨模态专家）同时接收文本和图像描述
    """
    
    def __init__(self, use_clip=True, device='cpu'):
        self.use_clip = use_clip
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        
        if self.use_clip:
            try:
                import clip
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                self.clip_model.eval()
                print(f"  [CLIP] 已加载CLIP模型 (device={device})")
            except Exception as e:
                print(f"  [警告] 无法加载CLIP模型: {e}")
                self.use_clip = False
        
        # 模拟图像描述词库（无CLIP时使用）
        self.mock_desc_keywords = [
            "一个人", "一群人", "卡通人物", "动物", "风景", "建筑物", "标志",
            "文字", "符号", "手势", "表情", "肤色", "衣服", "背景",
            "暴力", "攻击性", "侮辱", "歧视", "幽默", "讽刺", "中性"
        ]
    
    def encode_image(self, image):
        """
        将图像编码为文本描述
        
        Args:
            image: PIL Image 或 None
            
        Returns:
            description: 图像描述字符串
        """
        if image is None:
            return "无法识别图像"
        
        if self.use_clip and self.clip_model is not None:
            return self._clip_description(image)
        else:
            return self._mock_description(image)
    
    def _clip_description(self, image):
        """使用CLIP生成图像描述"""
        import clip
        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
            
            candidate_descriptions = [
                "hateful content", "offensive image", "racist symbol",
                "normal content", "neutral image", "funny meme",
                "group of people", "single person", "cartoon character",
                "text only", "image with text", "political meme",
                "violent scene", "hate speech", "discrimination",
                "homophobic content", "sexist image", "antisemitic symbol",
                "protest image", "celebrity photo", "animal picture",
                "nature landscape", "sports image", "food picture",
                "meme format", "satire", "irony", "propaganda",
                "historical figure", "religious symbol", "national flag",
                "police image", "military", "firearm", "weapon",
                "graffiti", "artwork", "illustration", "photograph",
                "screenshot", "news clip", "advertisement", "poster"
            ]
            
            text_inputs = clip.tokenize(candidate_descriptions).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)
            
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            top_k = 3
            top_indices = similarities.argsort(dim=-1, descending=True)[0, :top_k]
            top_desc = [candidate_descriptions[i] for i in top_indices]
            
            return f"图像内容：{', '.join(top_desc)}"
        except Exception as e:
            print(f"  [警告] CLIP编码失败: {e}")
            return self._mock_description(image)
    
    def _mock_description(self, image):
        """模拟图像描述（无CLIP时使用）"""
        import random
        width, height = image.size
        desc_parts = []
        
        if width > height:
            desc_parts.append("横向图像")
        else:
            desc_parts.append("纵向图像")
        
        colors = image.getcolors(maxcolors=1000)
        if colors is None:
            num_colors = 1000
        else:
            num_colors = len(colors)
        
        if num_colors < 50:
            desc_parts.append("简单图形")
        elif num_colors < 500:
            desc_parts.append("中等复杂度")
        else:
            desc_parts.append("复杂图像")
        
        desc_parts.extend(random.sample(self.mock_desc_keywords, 3))
        
        return "图像内容：" + ", ".join(desc_parts)
    
    def encode_batch(self, images):
        """批量编码图像"""
        descriptions = []
        for img in images:
            descriptions.append(self.encode_image(img))
        return descriptions


# =============================================================================
# LLM增强评估管线
# =============================================================================

def run_llm_evaluation(
    max_train=200,
    max_val=200,
    provider1='deepseek',
    provider2='glm',
    provider3='gpt',
    train_gat=True,
    enable_emnet=True,
    batch_size=8,
    save_cache=True,
    force_rerun_agent=None,
):
    """
    LLM增强的Hateful Memes评估管线（异构多模态版本）
    
    异构多模态设计：
    - Agent1: 仅文本输入（文本专家）
    - Agent2: 仅图像输入（图像专家，通过图像描述）
    - Agent3: 文本+图像（跨模态融合专家）
    
    Args:
        max_train: 训练样本数（用于GAT训练，少样本即可）
        max_val: 验证样本数
        provider1: Agent1的模型提供者
        provider2: Agent2的模型提供者
        provider3: Agent3的模型提供者
        train_gat: 是否训练GAT共识层
        enable_emnet: 是否启用EMNet纠偏
        batch_size: LLM API并发批次大小
        save_cache: 是否保存LLM输出缓存
    """
    print("=" * 70)
    print("LLM增强的 Hateful Memes 评估管线（异构多模态）")
    print("=" * 70)
    print(f"配置:")
    print(f"  Agent1: {provider1} (仅文本)")
    print(f"  Agent2: {provider2} (仅图像)")
    print(f"  Agent3: {provider3} (文本+图像)")
    print(f"  Train/Val: {max_train}/{max_val}")
    print(f"  GAT: {'启用' if train_gat else '禁用'}")
    print(f"  EMNet: {'启用' if enable_emnet else '禁用'}")
    print()

    # ========== 1. 数据加载 ==========
    print("[1] 加载数据...")
    train_dataset = HatefulMemesDataset(split='train', max_samples=max_train, load_images=True)
    val_dataset = HatefulMemesDataset(split='dev', max_samples=max_val, load_images=True)

    train_texts = train_dataset.texts
    train_labels = train_dataset.labels
    train_images = train_dataset.images
    val_texts = val_dataset.texts
    val_labels = val_dataset.labels
    val_images = val_dataset.images

    train_labels_t = torch.tensor(train_labels)
    val_labels_t = torch.tensor(val_labels)

    B_train = len(train_texts)
    B_val = len(val_texts)

    # ========== 1.1 标签分布检查 ==========
    def check_label_distribution(labels, name, max_imbalance_ratio=0.6):
        counts = {0: labels.count(0), 1: labels.count(1)}
        total = len(labels)
        if total == 0:
            return True
        ratio0 = counts[0] / total
        ratio1 = counts[1] / total
        max_ratio = max(ratio0, ratio1)
        print(f"  [{name}] 标签分布: 0={counts[0]}, 1={counts[1]}, 最大占比={max_ratio*100:.1f}%")
        if max_ratio > max_imbalance_ratio:
            print(f"  [警告] {name}标签分布严重不平衡! 最大占比={max_ratio*100:.1f}% > {max_imbalance_ratio*100:.0f}%")
            print(f"  [警告] 评估结果可能不可信，请检查数据集或使用stratified sampling")
            raise ValueError(f"{name}标签分布严重不平衡: {counts[0]}个0, {counts[1]}个1 (最大占比={max_ratio*100:.1f}%)")
        return True
    
    print("\n[1.1] 检查标签分布...")
    check_label_distribution(train_labels, "训练集")
    check_label_distribution(val_labels, "验证集")

    # ========== 1.5 图像编码（生成图像描述） ==========
    print("\n[1.5] 初始化图像编码器...")
    image_encoder = ImageEncoder(use_clip=True, device=DEVICE)
    
    print("  生成训练集图像描述...")
    train_image_descriptions = image_encoder.encode_batch(train_images)
    
    print("  生成验证集图像描述...")
    val_image_descriptions = image_encoder.encode_batch(val_images)

    # ========== 2. 创建LLM Agent（异构多模态） ==========
    print("\n[2] 创建LLM Agent...")
    
    # 检测API key配置
    providers = [provider1, provider2, provider3]
    agent_names = ["Agent1(文本专家)", "Agent2(图像专家)", "Agent3(跨模态)"]
    agent_prompts = ["text_focused", "image_focused", "multimodal_fusion"]
    
    agents = []
    for i, (prov, name, prompt_key) in enumerate(zip(providers, agent_names, agent_prompts)):
        client = LLMClient(
            provider=prov,
            temperature=0.1,
            max_retries=5,
            timeout=120,
        )
        use_direct_image = (i == 1 and prov == 'glm')
        agent = LLMAgent(
            client=client,
            name=name,
            system_prompt=AGENT_PROMPTS.get(prompt_key),
            embed_dim=256,
            num_classes=NUM_CLASSES,
            use_image=(i >= 2),
            use_direct_image=use_direct_image,
            verbose=True,
        )
        agents.append(agent)
        mode_str = "[模拟]" if client.mock_mode else "[API]"
        image_mode = "[直接图像]" if use_direct_image else "[图像描述]" if i >= 1 else "[仅文本]"
        print(f"  {name:<25s}: {prov}/{client.model} {mode_str} {image_mode}")

    # ========== 3. 训练集：LLM推理（异构多模态） ==========
    print(f"\n[3] 训练集LLM推理 ({B_train}样本)...")
    print(f"  第一次Agent调用可能较慢（API预热）...")
    print(f"  ★ 如果提示未设置API key，将使用模拟模式")
    print(f"  ★ 异构多模态策略: Agent1=文本, Agent2=图像, Agent3=文本+图像")
    
    all_train_alphas = torch.zeros(B_train, 3, NUM_CLASSES)
    all_train_beliefs = torch.zeros(B_train, 3, NUM_CLASSES)
    all_train_uncertainties = torch.zeros(B_train, 3)
    all_train_embs = torch.zeros(B_train, 3, 256)
    
    # 向后兼容：如果旧的合并缓存存在，先加载并拆分为独立缓存
    old_cache_path = os.path.join(CHECKPOINT_DIR, 'llm_outputs_train.pt')
    if os.path.exists(old_cache_path) and save_cache:
        print(f"  检测到旧格式缓存，正在转换为独立Agent缓存...")
        cached_train = torch.load(old_cache_path, map_location='cpu')
        for i in range(3):
            agent_cache_path = os.path.join(CHECKPOINT_DIR, f'llm_train_agent{i}.pt')
            torch.save({
                'alphas': cached_train['alphas'][:, i],
                'beliefs': cached_train['beliefs'][:, i],
                'uncertainties': cached_train['uncertainties'][:, i],
                'embs': cached_train['embs'][:, i],
            }, agent_cache_path)
        os.remove(old_cache_path)
        print(f"  旧缓存已转换并删除")
    
    # 按Agent独立缓存处理
    train_start = time.time()
    agents_to_run = []
    
    for i, agent in enumerate(agents):
        agent_cache_path = os.path.join(CHECKPOINT_DIR, f'llm_train_agent{i}.pt')
        force_rerun = force_rerun_agent is not None and i == force_rerun_agent
        
        if os.path.exists(agent_cache_path) and save_cache and not force_rerun:
            print(f"\n  加载缓存的 {agent.name} 输出...")
            cached_agent = torch.load(agent_cache_path, map_location='cpu')
            if cached_agent['alphas'].shape[0] != B_train:
                print(f"    缓存维度不匹配 ({cached_agent['alphas'].shape[0]} vs {B_train})，重新运行")
                agents_to_run.append((i, agent))
            else:
                all_train_alphas[:, i] = cached_agent['alphas']
                all_train_beliefs[:, i] = cached_agent['beliefs']
                all_train_uncertainties[:, i] = cached_agent['uncertainties']
                all_train_embs[:, i] = cached_agent['embs']
                print(f"    已加载 {B_train} 样本")
        else:
            agents_to_run.append((i, agent))
    
    # 只运行需要重新推理的Agent
    for i, agent in agents_to_run:
        print(f"\n  运行 {agent.name}...")
        for idx in tqdm(range(B_train), desc=f"    Agent{i+1}"):
            if i == 0:
                alpha, belief, uncertainty, emb = agent.forward(train_texts[idx], image_description=None)
            elif i == 1:
                if agent.use_direct_image:
                    alpha, belief, uncertainty, emb = agent.forward(train_texts[idx], image=train_images[idx])
                else:
                    alpha, belief, uncertainty, emb = agent.forward("", image_description=train_image_descriptions[idx])
            else:
                alpha, belief, uncertainty, emb = agent.forward(train_texts[idx], image_description=train_image_descriptions[idx])
            all_train_alphas[idx, i] = alpha
            all_train_beliefs[idx, i] = belief
            all_train_uncertainties[idx, i] = uncertainty
            all_train_embs[idx, i] = emb
        
        # 每个Agent完成后立即保存独立缓存
        if save_cache:
            agent_cache_path = os.path.join(CHECKPOINT_DIR, f'llm_train_agent{i}.pt')
            torch.save({
                'alphas': all_train_alphas[:, i],
                'beliefs': all_train_beliefs[:, i],
                'uncertainties': all_train_uncertainties[:, i],
                'embs': all_train_embs[:, i],
            }, agent_cache_path)
            print(f"    {agent.name} 输出已缓存: {agent_cache_path}")
    
    train_time = time.time() - train_start
    print(f"\n  训练集推理完成 ({train_time:.1f}秒)")
    
    # 计算投票预测
    all_train_preds = torch.zeros(B_train, dtype=torch.long)
    for b in range(B_train):
        values, counts = torch.unique(all_train_beliefs[b].argmax(dim=-1), return_counts=True)
        all_train_preds[b] = values[counts.argmax()]
    
    # 打印Agent统计
    print(f"\n  Agent统计:")
    for agent in agents:
        stats = agent.get_stats()
        print(f"    {stats['name']:<25s}: {stats['calls']:4d} calls, "
              f"avg_conf={stats['avg_confidence']:.3f}, "
              f"{'[API]' if not stats['mock_mode'] else '[模拟]'}")
    
    # Agent基线准确率
    print(f"\n  Agent基线准确率:")
    for i, agent in enumerate(agents):
        acc = (all_train_beliefs[:, i].argmax(dim=-1) == train_labels_t).float().mean().item() * 100
        avg_u = all_train_uncertainties[:, i].mean().item()
        print(f"    {agent.name:<25s}: Acc={acc:.2f}%, Avg_u={avg_u:.4f}")

    # ========== 4. GAT共识训练 ==========
    print(f"\n[4] GAT共识训练...")
    
    train_all_beliefs = [all_train_beliefs[:, i].to(DEVICE) for i in range(3)]
    train_all_us = [all_train_uncertainties[:, i].to(DEVICE) for i in range(3)]
    train_y = train_labels_t.to(DEVICE)
    
    # 计算训练集上的agent准确率（用于DS融合权重，避免数据泄露）
    train_agent_accs = []
    for i in range(3):
        train_preds_i = train_all_beliefs[i].argmax(dim=1)
        train_acc_i = (train_preds_i == train_y).float().mean().item()
        train_agent_accs.append(train_acc_i)
    
    train_acc_weights = torch.tensor(train_agent_accs, device=DEVICE)
    train_acc_weights = F.softmax(train_acc_weights, dim=0)
    
    print(f"  训练集Agent准确率: Agent1={train_agent_accs[0]:.4f}, Agent2={train_agent_accs[1]:.4f}, Agent3={train_agent_accs[2]:.4f}")
    print(f"  训练集Agent权重: {train_acc_weights.cpu().numpy()}")
    
    # DS融合基线（使用训练集权重）
    train_ds_preds, _, _ = ds_fusion_decision(train_all_beliefs, train_all_us, u_threshold=U_THRESHOLD, agent_weights=train_acc_weights)
    train_correct = (train_ds_preds == train_y)
    
    # 有分歧样本
    train_has_disagreement = torch.zeros(len(train_y), dtype=torch.bool, device=DEVICE)
    for i in range(3):
        for j in range(i+1, 3):
            train_has_disagreement |= (train_all_beliefs[i].argmax(dim=1) != train_all_beliefs[j].argmax(dim=1))
    
    train_gat_mask = train_has_disagreement
    train_gat_indices_all = torch.arange(len(train_y), device=DEVICE)
    train_gat_indices = torch.where(train_gat_mask)[0]
    
    print(f"  DS正确样本: {train_correct.sum().item()}/{len(train_y)}")
    print(f"  有分歧样本: {train_has_disagreement.sum().item()}/{len(train_y)}")
    print(f"  GAT训练样本(分歧): {len(train_gat_indices)}")
    print(f"  GAT训练样本(全部): {len(train_gat_indices_all)}")
    
    gat_engine = None
    if len(train_gat_indices) >= 5 and train_gat:
        gat_node_dim = 256 + NUM_CLASSES + 1  # embed_dim + num_classes + u
        gat_layer = GATConsensusLayer(
            node_dim=gat_node_dim, hidden_dim=64, embed_dim=256, num_classes=NUM_CLASSES
        ).to(DEVICE)
        
        gat_model_path = os.path.join(CHECKPOINT_DIR, 'gat_consensus_llm.pt')
        if os.path.exists(gat_model_path):
            gat_layer.load_state_dict(torch.load(gat_model_path, map_location=DEVICE, weights_only=True))
            gat_layer.eval()
            print(f"  加载已训练的GAT模型: {gat_model_path}")
        else:
            # 构造训练数据
            gat_train_alphas = []
            for i in range(3):
                S = NUM_CLASSES / train_all_us[i].clamp(min=1e-6)
                alpha_i = train_all_beliefs[i] * S.unsqueeze(-1) + 1.0
                gat_train_alphas.append(alpha_i)
            
            agent_accs = torch.zeros(3, device=DEVICE)
            for i in range(3):
                agent_accs[i] = (train_all_beliefs[i].argmax(dim=1) == train_y).float().mean()
            print(f"  Agent训练准确率: Agent1={agent_accs[0]:.3f}, Agent2={agent_accs[1]:.3f}, Agent3={agent_accs[2]:.3f}")
            
            gat_optimizer = torch.optim.Adam(gat_layer.parameters(), lr=3e-4, weight_decay=1e-5)
            gat_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gat_optimizer, T_max=100)
            
            best_gat_loss = float('inf')
            num_train = len(train_gat_indices_all)
            
            init_norm = sum(p.norm().item() for p in gat_layer.parameters())
            print(f"  GAT初始参数范数: {init_norm:.4f}")
            print(f"  训练样本数: {num_train}")
            
            for epoch in range(100):
                gat_layer.train()
                total_loss = 0.0
                total_energy = 0.0
                perm_indices = train_gat_indices_all[torch.randperm(num_train)]
                
                for start_idx in range(0, num_train, 16):
                    batch_idx = perm_indices[start_idx:start_idx+16]
                    batch_loss = torch.tensor(0.0, requires_grad=True, device=DEVICE)
                    batch_energy = 0.0
                    
                    for b_idx_cpu in batch_idx.cpu().numpy():
                        agent_outputs = []
                        for i in range(3):
                            b_i = train_all_beliefs[i][b_idx_cpu:b_idx_cpu+1]
                            u_i = train_all_us[i][b_idx_cpu:b_idx_cpu+1]
                            emb_i = all_train_embs[b_idx_cpu:b_idx_cpu+1, i].to(DEVICE)
                            u_val = float(u_i.squeeze(-1).item())
                            S = NUM_CLASSES / max(u_val, 1e-6)
                            alpha_i = b_i[0] * S + 1.0
                            agent_outputs.append((alpha_i, b_i[0], u_val, emb_i[0]))
                        
                        engine_tmp = ConsensusEngine(
                            embed_dim=256, num_classes=NUM_CLASSES, hidden_dim=64
                        )
                        engine_tmp.layer = gat_layer
                        try:
                            h = engine_tmp.build_state(agent_outputs)
                            
                            fusion_weights = gat_layer.forward_sample_weights(h)
                            
                            true_label = train_y[b_idx_cpu].unsqueeze(0)
                            
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
                            
                            batch_loss = batch_loss + loss
                        except Exception as e:
                            print(f"    [GAT训练错误] batch_idx={b_idx_cpu}: {e}")
                    
                    if batch_loss > 0:
                        gat_optimizer.zero_grad()
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(gat_layer.parameters(), 0.5)
                        gat_optimizer.step()
                        total_loss += batch_loss.item()
                        total_energy += batch_energy
                
                gat_scheduler.step()
                avg_loss = total_loss / max(num_train, 1)
                avg_energy = total_energy / max(num_train, 1)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"    GAT Epoch {epoch+1:3d}/50: loss={avg_loss:.4f}, energy={avg_energy:.4f}")
                if avg_loss < best_gat_loss:
                    best_gat_loss = avg_loss
                    torch.save(gat_layer.state_dict(), gat_model_path)
            
            gat_layer.load_state_dict(torch.load(gat_model_path, map_location=DEVICE, weights_only=True))
            gat_layer.eval()
            final_norm = sum(p.norm().item() for p in gat_layer.parameters())
            print(f"  GAT最终参数范数: {final_norm:.4f}")
            print(f"  GAT参数变化范数: {abs(final_norm - init_norm):.4f}")
            print(f"  ★ GAT训练完成, 最佳loss={best_gat_loss:.4f}")
        
        gat_engine = ConsensusEngine(
            embed_dim=256, num_classes=NUM_CLASSES, hidden_dim=64
        )
        gat_engine.layer = gat_layer
    else:
        if not train_gat:
            print("  GAT训练已禁用")
        else:
            print("  GAT训练样本不足，跳过")

    # ========== 5. 验证集：LLM推理（异构多模态） ==========
    print(f"\n[5] 验证集LLM推理 ({B_val}样本)...")
    
    all_val_alphas = torch.zeros(B_val, 3, NUM_CLASSES)
    all_val_beliefs = torch.zeros(B_val, 3, NUM_CLASSES)
    all_val_uncertainties = torch.zeros(B_val, 3)
    all_val_embs = torch.zeros(B_val, 3, 256)
    
    # 向后兼容：如果旧的合并缓存存在，先加载并拆分为独立缓存
    old_cache_path_val = os.path.join(CHECKPOINT_DIR, 'llm_outputs_val.pt')
    if os.path.exists(old_cache_path_val) and save_cache:
        print(f"  检测到旧格式验证集缓存，正在转换为独立Agent缓存...")
        cached_val = torch.load(old_cache_path_val, map_location='cpu')
        for i in range(3):
            agent_cache_path = os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}.pt')
            torch.save({
                'alphas': cached_val['alphas'][:, i],
                'beliefs': cached_val['beliefs'][:, i],
                'uncertainties': cached_val['uncertainties'][:, i],
                'embs': cached_val['embs'][:, i],
            }, agent_cache_path)
        os.remove(old_cache_path_val)
        print(f"  旧验证集缓存已转换并删除")
    
    # 按Agent独立缓存处理
    val_start = time.time()
    val_agents_to_run = []
    
    for i, agent in enumerate(agents):
        agent_cache_path = os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}.pt')
        force_rerun = force_rerun_agent is not None and i == force_rerun_agent
        
        if os.path.exists(agent_cache_path) and save_cache and not force_rerun:
            print(f"\n  加载缓存的 {agent.name} 验证集输出...")
            cached_agent = torch.load(agent_cache_path, map_location='cpu')
            if cached_agent['alphas'].shape[0] != B_val:
                print(f"    缓存维度不匹配 ({cached_agent['alphas'].shape[0]} vs {B_val})，重新运行")
                val_agents_to_run.append((i, agent))
            else:
                all_val_alphas[:, i] = cached_agent['alphas']
                all_val_beliefs[:, i] = cached_agent['beliefs']
                all_val_uncertainties[:, i] = cached_agent['uncertainties']
                all_val_embs[:, i] = cached_agent['embs']
                print(f"    已加载 {B_val} 样本")
        else:
            val_agents_to_run.append((i, agent))
    
    # 只运行需要重新推理的Agent
    for i, agent in val_agents_to_run:
        print(f"\n  运行 {agent.name}...")
        for idx in tqdm(range(B_val), desc=f"    Agent{i+1}"):
            if i == 0:
                alpha, belief, uncertainty, emb = agent.forward(val_texts[idx], image_description=None)
            elif i == 1:
                if agent.use_direct_image:
                    alpha, belief, uncertainty, emb = agent.forward(val_texts[idx], image=val_images[idx])
                else:
                    alpha, belief, uncertainty, emb = agent.forward("", image_description=val_image_descriptions[idx])
            else:
                alpha, belief, uncertainty, emb = agent.forward(val_texts[idx], image_description=val_image_descriptions[idx])
            all_val_alphas[idx, i] = alpha
            all_val_beliefs[idx, i] = belief
            all_val_uncertainties[idx, i] = uncertainty
            all_val_embs[idx, i] = emb
        
        # 每个Agent完成后立即保存独立缓存
        if save_cache:
            agent_cache_path = os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}.pt')
            torch.save({
                'alphas': all_val_alphas[:, i],
                'beliefs': all_val_beliefs[:, i],
                'uncertainties': all_val_uncertainties[:, i],
                'embs': all_val_embs[:, i],
            }, agent_cache_path)
            print(f"    {agent.name} 验证集输出已缓存: {agent_cache_path}")
    
    val_time = time.time() - val_start
    print(f"\n  验证集推理完成 ({val_time:.1f}秒)")
    
    # 计算投票预测
    all_val_preds = torch.zeros(B_val, dtype=torch.long)
    for b in range(B_val):
        values, counts = torch.unique(all_val_beliefs[b].argmax(dim=-1), return_counts=True)
        all_val_preds[b] = values[counts.argmax()]

    # ========== 6. 验证集评估 ==========
    print(f"\n[6] 验证集评估...")
    
    b1 = all_val_beliefs[:, 0].to(DEVICE)
    b2 = all_val_beliefs[:, 1].to(DEVICE)
    b3 = all_val_beliefs[:, 2].to(DEVICE)
    u1 = all_val_uncertainties[:, 0].to(DEVICE)
    u2 = all_val_uncertainties[:, 1].to(DEVICE)
    u3 = all_val_uncertainties[:, 2].to(DEVICE)
    
    y_true = val_labels_t.numpy()
    
    all_beliefs = torch.stack([b1, b2, b3], dim=1)
    all_uncertainties = torch.stack([u1, u2, u3], dim=1)
    all_embs = all_val_embs.to(DEVICE)
    
    print(f"\n  各LLM Agent基线:")
    print(f"  {'Agent':<20s} {'Acc%':<10s} {'F1%':<10s} {'Avg u':<10s}")
    print(f"  {'-'*50}")
    for idx, name in enumerate([f'Agent1({provider1})', f'Agent2({provider2})', f'Agent3({provider3})']):
        preds = all_beliefs[:, idx].argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_true, preds) * 100
        f1 = f1_score(y_true, preds, average='binary') * 100
        avg_u = all_uncertainties[:, idx].mean().item()
        print(f"  {name:<20s} {acc:<10.2f} {f1:<10.2f} {avg_u:<10.4f}")

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
    
    ds_preds, ds_rej, ds_u = ds_fusion_decision(b_list, u_list, u_threshold=U_THRESHOLD, agent_weights=train_acc_weights)

    # === 方法4: GAT共识 + DS ===
    if gat_engine is not None:
        print(f"\n  运行GAT共识层...")
        final_belief_list = []
        final_u_list = []

        gat_engine.layer.eval()
        with torch.no_grad():
            for b_idx in range(B_val):
                agent_outputs = []
                for i in range(3):
                    b_i = all_beliefs[b_idx, i].unsqueeze(0)
                    u_i = all_uncertainties[b_idx, i].unsqueeze(0)
                    emb_i = all_embs[b_idx, i].unsqueeze(0)
                    u_val = u_i.squeeze(-1).item()
                    S_val = NUM_CLASSES / max(u_val, 1e-6)
                    alpha_i = b_i[0] * S_val + 1.0
                    agent_outputs.append((alpha_i, b_i[0], u_val, emb_i[0]))

                h = gat_engine.build_state(agent_outputs)
                
                fusion_weights = gat_engine.layer.forward_sample_weights(h, hard_gate=False)
                
                original_beliefs = h[:, gat_engine.embed_dim:gat_engine.embed_dim+NUM_CLASSES]
                
                fused_belief = fusion_weights @ original_beliefs
                fused_belief = fused_belief / fused_belief.sum().clamp(min=1e-6)
                
                fs = []
                for i in range(3):
                    adjusted_belief = 0.8 * original_beliefs[i] + 0.2 * fused_belief
                    adjusted_belief = adjusted_belief / adjusted_belief.sum().clamp(min=1e-6)
                    fs.append(adjusted_belief)
                
                us = [all_uncertainties[b_idx, i].item() * 0.9 + 0.1 for i in range(3)]
                
                final_belief_list.append(torch.stack(fs, dim=0))
                final_u_list.append(torch.tensor(us))
                
                if b_idx < 3:
                    print(f"    样本{b_idx}: pre=[{all_beliefs[b_idx,0].argmax().item()},{all_beliefs[b_idx,1].argmax().item()},{all_beliefs[b_idx,2].argmax().item()}], "
                          f"post=[{fs[0].argmax().item()},{fs[1].argmax().item()},{fs[2].argmax().item()}], "
                          f"label={val_labels[b_idx]}")

        final_belief = torch.stack(final_belief_list, dim=0)
        final_uncertainty = torch.stack(final_u_list, dim=0)
        print(f"  共识后平均u: {final_uncertainty.mean().item():.4f} (从前{all_uncertainties.mean().item():.4f})")
    else:
        print(f"\n  GAT不可用，使用原始信念")
        final_belief = all_beliefs
        final_uncertainty = all_uncertainties
        avg_energy = 0.0

    gat_beliefs = [final_belief[:, i] for i in range(3)]
    gat_uncertainties = [final_uncertainty[:, i] for i in range(3)]
    gat_ds_preds, gat_ds_rej, gat_ds_u = ds_fusion_decision(gat_beliefs, gat_uncertainties, u_threshold=U_THRESHOLD, agent_weights=train_acc_weights)
    
    if gat_engine is not None:
        gat_fused_preds = []
        for b_idx in range(B_val):
            h = gat_engine.build_state([(all_val_alphas[b_idx, i], all_beliefs[b_idx, i], float(all_uncertainties[b_idx, i].item()), all_embs[b_idx, i]) for i in range(3)])
            fusion_weights = gat_engine.layer.forward_sample_weights(h, hard_gate=False)
            original_beliefs = h[:, gat_engine.embed_dim:gat_engine.embed_dim+NUM_CLASSES]
            fused_belief = fusion_weights @ original_beliefs
            fused_belief = fused_belief / fused_belief.sum().clamp(min=1e-6)
            gat_fused_preds.append(fused_belief.argmax().item())
        gat_fused_preds = torch.tensor(gat_fused_preds)
    else:
        gat_fused_preds = gat_ds_preds

    # === 方法5: 分歧解构 + 证据交换 ===
    print(f"\n  运行分歧解构...")
    deconstructor = DisagreementDeconstructor(u_threshold=0.5, K_threshold=0.3)
    conflict_types, K_values = deconstructor.deconstruct_batch(all_beliefs, all_uncertainties)

    evidence_count = sum(1 for c in conflict_types if c == 'evidence_conflict')
    ignorance_count = sum(1 for c in conflict_types if c == 'ignorance_conflict')
    no_conflict_count = sum(1 for c in conflict_types if c == 'none')
    print(f"  分歧分布: 证据冲突={evidence_count}, 无知冲突={ignorance_count}, 无分歧={no_conflict_count}")

    # 证据交换：最佳agent → 最差agent
    all_alphas = all_val_alphas.to(DEVICE)
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

    corr_b_list = [corrected_beliefs[:, i].to(DEVICE) for i in range(3)]
    corr_u_list = [corrected_uncertainties[:, i].to(DEVICE) for i in range(3)]
    corr_ds_preds, corr_ds_rej, corr_ds_u = ds_fusion_decision(corr_b_list, corr_u_list, u_threshold=U_THRESHOLD)

    # === 方法6: 混合策略 - 一致样本直接使用，分歧样本使用GAT ===
    hybrid_preds = torch.zeros(B_val, dtype=torch.long)
    for b_idx in range(B_val):
        if conflict_types[b_idx] == 'none':
            hybrid_preds[b_idx] = mv_preds[b_idx].item()
        else:
            hybrid_preds[b_idx] = gat_fused_preds[b_idx].item()

    # ========== 汇总结果 ==========
    results = {
        f'Agent1({provider1})': {
            'preds': b1.argmax(dim=1), 'rejected': torch.zeros(B_val, dtype=torch.bool),
        },
        f'Agent2({provider2})': {
            'preds': b2.argmax(dim=1), 'rejected': torch.zeros(B_val, dtype=torch.bool),
        },
        f'Agent3({provider3})': {
            'preds': b3.argmax(dim=1), 'rejected': torch.zeros(B_val, dtype=torch.bool),
        },
        'MajorityVoting': {
            'preds': mv_preds, 'rejected': torch.zeros(B_val, dtype=torch.bool),
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
        'GAT_Fusion': {
            'preds': gat_fused_preds, 'rejected': torch.zeros(B_val, dtype=torch.bool),
        },
        'GAT_EvidenceSwap': {
            'preds': corr_ds_preds.cpu(), 'rejected': corr_ds_rej.cpu(), 'uncertainty': corr_ds_u.cpu(),
        },
        'Hybrid_GAT': {
            'preds': hybrid_preds, 'rejected': torch.zeros(B_val, dtype=torch.bool),
        },
        'BestAgent': {
            'preds': b3.argmax(dim=1), 'rejected': torch.zeros(B_val, dtype=torch.bool),
        },
    }

    # ========== 计算指标 ==========
    print(f"\n{'='*120}")
    print(f"{'方法':<24s} {'Acc%':<10s} {'F1%':<10s} {'ECE':<12s} {'Rej%':<10s} {'Acc_All%':<10s}")
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
        print(f"{method_name:<24s} {acc:<10.2f} {f1:<10.2f} {ece:<12.4f} "
              f"{rej_rate:<10.2f} {acc_all:<10.2f}")

    print(f"{'='*120}")

    # ========== 分歧样本单独分析 ==========
    print(f"\n[分歧样本分析]")
    
    disagreement_mask = torch.tensor([c != 'none' for c in conflict_types], dtype=torch.bool)
    evidence_conflict_mask = torch.tensor([c == 'evidence_conflict' for c in conflict_types], dtype=torch.bool)
    
    print(f"  总样本: {B_val}, 分歧样本: {disagreement_mask.sum().item()}, 证据冲突: {evidence_conflict_mask.sum().item()}")
    
    if disagreement_mask.sum() > 0:
        print(f"\n  {'方法':<24s} {'分歧Acc%':<12s} {'证据冲突Acc%':<14s}")
        print(f"  {'-'*50}")
        
        for method_name, res in results.items():
            preds_np = res['preds'].cpu().numpy()
            
            # 分歧样本准确率
            if disagreement_mask.sum() > 0:
                acc_disagree = accuracy_score(y_true[disagreement_mask], preds_np[disagreement_mask]) * 100
            else:
                acc_disagree = 0.0
            
            # 证据冲突样本准确率
            if evidence_conflict_mask.sum() > 0:
                acc_evidence = accuracy_score(y_true[evidence_conflict_mask], preds_np[evidence_conflict_mask]) * 100
            else:
                acc_evidence = 0.0
            
            print(f"  {method_name:<24s} {acc_disagree:<12.2f} {acc_evidence:<14.2f}")

    # ========== 保存结果 ==========
    result_name = f'llm_{provider1}_{provider2}_{provider3}'
    result_path = os.path.join(RESULT_DIR, f'evaluation_{result_name}.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {result_path}")

    detail_data = {
        'y_true': y_true.tolist(),
        'conflict_types': conflict_types,
        'K_values': [float(k) for k in K_values],
    }
    for method_name, res in results.items():
        detail_data[f'{method_name}_preds'] = res['preds'].cpu().tolist()
        detail_data[f'{method_name}_rejected'] = res['rejected'].cpu().tolist()
        if 'uncertainty' in res:
            detail_data[f'{method_name}_uncertainty'] = res['uncertainty'].cpu().tolist()

    detail_path = os.path.join(RESULT_DIR, f'details_{result_name}.json')
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(detail_data, f, indent=2)
    print(f"详细信息保存至: {detail_path}")

    # ========== 可视化 ==========
    print(f"\n[7] 生成可视化...")

    # 对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    all_m = ['MajorityVoting', 'WeightedAvg', 'DS_Fusion', 'GAT_DS_Fusion', 'GAT_Fusion', 'Hybrid_GAT', 'GAT_EvidenceSwap']
    labels = ['多数投票', '加权平均', 'DS融合', 'GAT+DS', 'GAT直接融合', '混合GAT', 'GAT+证据交换']
    colors = ['#4EC4C4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFD93D', '#2ECC71', '#DDA0DD']

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

    plt.suptitle(f'LLM增强共识框架对比 (LLM: {provider1}/{provider2}/{provider3})', fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, f'llm_{result_name}_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图: {fig_path}")

    # 分歧分析图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    types = ['none', 'evidence_conflict', 'ignorance_conflict']
    counts = [sum(1 for c in conflict_types if c == t) for t in types]
    colors2 = ['#4EC4C4', '#FF6B6B', '#FFD93D']
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

    plt.suptitle('LLM Agent 分歧解构分析', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'llm_{result_name}_conflict_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  分歧分析图: figures/llm_{result_name}_conflict_analysis.png")

    # Agent统计图
    fig, ax = plt.subplots(figsize=(10, 5))
    agent_names_plot = [f'Agent1({provider1})', f'Agent2({provider2})', f'Agent3({provider3})']
    accs = [metrics[m]['accuracy'] for m in [f'Agent1({provider1})', f'Agent2({provider2})', f'Agent3({provider3})']]
    bars = ax.bar(agent_names_plot, accs, color=['#4EC4C4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax.set_ylabel('准确率 (%)')
    ax.set_title('各LLM Agent独立准确率', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'llm_{result_name}_agent_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Agent准确率图: figures/llm_{result_name}_agent_accuracy.png")

    print(f"\n{'='*80}")
    print(f"LLM增强评估完成！")
    print(f"  Agent: {provider1}/{provider2}/{provider3}")
    print(f"  DS融合准确率: {metrics['DS_Fusion']['accuracy']:.2f}%")
    print(f"  GAT+DS准确率: {metrics['GAT_DS_Fusion']['accuracy']:.2f}%")
    print(f"  GAT+证据交换准确率: {metrics['GAT_EvidenceSwap']['accuracy']:.2f}%")
    print(f"{'='*80}")

    return metrics, agents


# =============================================================================
# 入口
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LLM增强的Hateful Memes评估管线')
    parser.add_argument('--max_train', type=int, default=200,
                        help='训练样本数（用于GAT训练，推荐50-200）')
    parser.add_argument('--max_val', type=int, default=200,
                        help='验证样本数（推荐100-500）')
    parser.add_argument('--provider1', type=str, default='deepseek',
                        choices=list(PROVIDER_CONFIGS.keys()) + ['mock'],
                        help='Agent1模型提供者')
    parser.add_argument('--provider2', type=str, default='glm',
                        choices=list(PROVIDER_CONFIGS.keys()) + ['mock'],
                        help='Agent2模型提供者')
    parser.add_argument('--provider3', type=str, default='gpt',
                        choices=list(PROVIDER_CONFIGS.keys()) + ['mock'],
                        help='Agent3模型提供者')
    parser.add_argument('--no_gat', action='store_true', help='禁用GAT训练')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='LLM请求批次大小')
    parser.add_argument('--no_cache', action='store_true', help='不保存LLM输出缓存')
    parser.add_argument('--check', action='store_true',
                        help='仅运行预检查，不执行实验')
    parser.add_argument('--no_precheck', action='store_true',
                        help='跳过预检查，直接运行实验（不推荐）')
    parser.add_argument('--force_rerun_agent', type=int, default=None,
                        choices=[0, 1, 2],
                        help='强制重新运行指定Agent（0=Agent1, 1=Agent2, 2=Agent3）')

    args = parser.parse_args()

    if args.check:
        pre_check_dataset(
            max_train=args.max_train,
            max_val=args.max_val,
            provider1=args.provider1,
            provider2=args.provider2,
            provider3=args.provider3,
        )
    else:
        if not args.no_precheck:
            print("[PRE-CHECK] 运行实验前预检查...")
            if not pre_check_dataset(
                max_train=args.max_train,
                max_val=args.max_val,
                provider1=args.provider1,
                provider2=args.provider2,
                provider3=args.provider3,
            ):
                print("[FAIL] 预检查失败，中止实验")
                exit(1)
            print()
        
        run_llm_evaluation(
            max_train=args.max_train,
            max_val=args.max_val,
            provider1=args.provider1,
            provider2=args.provider2,
            provider3=args.provider3,
            train_gat=not args.no_gat,
            batch_size=args.batch_size,
            save_cache=not args.no_cache,
            force_rerun_agent=args.force_rerun_agent,
        )