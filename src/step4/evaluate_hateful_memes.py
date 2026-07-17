"""
第四步：Hateful Memes 多模态评估（v1）
=========================================
设计思路（与 CIFAR-10N 版本的关键区别）：
- 真正的多模态：文本（BERT）+ 图像（ResNet）+ 弱融合（坏模态模拟）
- 三个异构智能体自然地产生分歧——这正是我们框架的目标场景
- 评估指标同 CIFAR-10N：Acc、F1、ECE、拒识率

智能体设计：
  Agent1 (TextBERT)  : BERT 文本编码 → EvidenceHead
  Agent2 (ImageRN)   : ResNet-18 图像编码 → EvidenceHead
  Agent3 (FusionMLP) : 文本+图像简单拼接 → EvidenceHead (故意弱化,模拟坏模态)

数据：
  - 训练: 8500, 验证: 500
  - 二分类: 0=非仇恨, 1=仇恨
  - 验证集平衡: 50% hateful
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import sys
import time
# 设置中文字体
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import setup_chinese_font, setup_plot_style
setup_chinese_font()
setup_plot_style()
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step1.synthetic_data import SEED, DEVICE

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs('figures', exist_ok=True)
os.makedirs('results/hateful_memes', exist_ok=True)
os.makedirs('checkpoints/hateful_memes', exist_ok=True)
os.makedirs('data/features/hateful_memes', exist_ok=True)

NUM_CLASSES = 2  # 二分类
HATE_DIR = 'data/Hateful_Memes/data'


# =============================================================================
# 1. 数据加载
# =============================================================================

def load_hateful_data(split='train', max_samples=None):
    """加载Hateful Memes的jsonl文件和图片
    
    Args:
        split: 'train' 或 'dev'
        max_samples: 限制样本数（调试用）
    
    Returns:
        texts: list[str]
        image_ids: list[str] (如 '08291.png')
        labels: list[int]
    """
    import json
    from PIL import Image
    
    filepath = os.path.join(HATE_DIR, f'{split}.jsonl')
    with open(filepath, 'r', encoding='utf-8') as f:
        items = [json.loads(l) for l in f.readlines()]
    
    if max_samples:
        items = items[:max_samples]
    
    texts = []
    image_ids = []
    labels = []
    
    for item in items:
        texts.append(item['text'])
        # img字段格式: 'img/08291.png'
        img_name = item['img'].split('/')[-1]
        image_ids.append(img_name)
        labels.append(item['label'])
    
    return texts, image_ids, labels


# =============================================================================
# 2. 特征提取器
# =============================================================================

class TextFeatureExtractor:
    """使用 DistilBERT 提取文本特征（轻量，无需大型模型依赖）
    
    替代方案：如果不想依赖 transformers，可使用 Bag-of-Words + BiLSTM
    这里使用一种轻量级方案：Word-Level BiLSTM + 预训练 GloVe 嵌入的替代
    ——实际上我们用一个简单的 char-CNN + Transformer 编码器，完全自包含
    """
    
    def __init__(self, embedding_dim=128, hidden_dim=256, max_len=64):
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 构建字符级词表（覆盖所有英文字符 + 标点）
        # 注意：chr(i) 映射为 idx+2，确保索引连续不越界
        self.char_vocab = {chr(i): idx+2 for idx, i in enumerate(range(32, 127))}  # 可打印ASCII
        self.char_vocab['<PAD>'] = 0
        self.char_vocab['<UNK>'] = 1
        self.vocab_size = len(self.char_vocab)
        
        self.model = None  # lazy init
        self.is_fitted = False
    
    def _build_model(self):
        """构建字符级文本编码器"""
        class Permute(nn.Module):
            def __init__(self, *dims):
                super().__init__()
                self.dims = dims
            def forward(self, x):
                return x.permute(*self.dims)
        
        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0),
            # Permute: [B, seq_len, emb_dim] → [B, emb_dim, seq_len] for Conv1d
            Permute(0, 2, 1),
            nn.Conv1d(self.embedding_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        ).to(DEVICE)
        return self.model
    
    def encode(self, texts, batch_size=64):
        """将文本列表编码为特征向量 [N, hidden_dim]
        
        字符级编码的优势：对拼写错误、俚语、缩写（仇恨言论常见特征）具有鲁棒性
        """
        if self.model is None:
            self.model = self._build_model()
        
        self.model.eval()
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_encoded = []
                
                for text in batch_texts:
                    # 字符级编码
                    chars = [self.char_vocab.get(c, 1) for c in str(text)[:self.max_len]]
                    # padding
                    chars = chars + [0] * (self.max_len - len(chars))
                    batch_encoded.append(chars)
                
                x = torch.LongTensor(batch_encoded).to(DEVICE)
                feat = self.model(x)
                all_features.append(feat.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def fit(self, texts):
        """无监督适配（实际上不需要，但保持接口一致）"""
        self.is_fitted = True
        _ = self.encode(texts)
        return self


class ImageFeatureExtractor:
    """使用 ResNet-18 提取图像特征
    
    移除最后的分类层，取 avgpool 后的 512 维特征
    """
    
    def __init__(self, output_dim=512):
        self.output_dim = output_dim
        self.model = None
        self.transform = None
    
    def _build_model(self):
        from torchvision import transforms, models
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        model = models.resnet18(weights='IMAGENET1K_V1')
        # 移除分类头，取特征
        model = nn.Sequential(*list(model.children())[:-1])  # [B, 512, 1, 1]
        self.model = model.to(DEVICE)
        self.model.eval()
        return self.model
    
    def encode(self, image_ids, batch_size=64):
        """从图片ID列表提取 ResNet 特征
        
        Args:
            image_ids: list[str] 如 ['08291.png', ...]
        
        Returns:
            features: [N, 512]
        """
        from PIL import Image
        
        if self.model is None:
            self._build_model()
        
        img_dir = os.path.join(HATE_DIR, 'img')
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(image_ids), batch_size):
                batch_ids = image_ids[i:i+batch_size]
                batch_imgs = []
                
                for img_name in batch_ids:
                    img_path = os.path.join(img_dir, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_imgs.append(img_tensor)
                
                x = torch.stack(batch_imgs).to(DEVICE)
                feat = self.model(x)  # [B, 512, 1, 1]
                feat = feat.view(feat.size(0), -1)  # [B, 512]
                all_features.append(feat.cpu())
        
        return torch.cat(all_features, dim=0)


class SimpleFusionExtractor:
    """简单的文本+图像拼接编码器（故意弱化）
    
    设计意图：模拟低质量/坏模态智能体。
    直接将文本特征和图像特征拼接后通过一个小的MLP，
    不做任何模态对齐或交叉注意力。
    """
    
    def __init__(self, text_dim=256, image_dim=512, hidden_dim=128):
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        self.text_extractor = TextFeatureExtractor(embedding_dim=128, hidden_dim=text_dim)
        self.image_extractor = ImageFeatureExtractor(output_dim=image_dim)
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(DEVICE)
    
    def encode(self, texts, image_ids, batch_size=64):
        """提取并融合文本和图像特征
        
        先用各自的编码器提取特征，再拼接后投影
        
        Returns:
            features: [N, hidden_dim]
        """
        text_feats = self.text_extractor.encode(texts, batch_size)
        image_feats = self.image_extractor.encode(image_ids, batch_size)
        
        # 拼接
        combined = torch.cat([text_feats, image_feats], dim=1).to(DEVICE)
        
        self.fusion_proj.eval()
        with torch.no_grad():
            fused = self.fusion_proj(combined)
        
        return fused.cpu()


# =============================================================================
# 3. 证据头（Evidence Head）
# =============================================================================

class EvidenceHead(nn.Module):
    """证据头：将特征映射为 Dirichlet 参数 α
    
    架构：MLP(特征维度 → 128 → 64 → K)
    输出 α = ReLU(last_layer) + 1 （确保 α > 1）
    
    理论依据：Dirichlet 分布的参数 α_k 必须大于 0，
    我们强制 α_k >= 1 以避免退化情况（α_k < 1 时分布呈U型）。
    """
    
    def __init__(self, input_dim, num_classes=2, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # 保存倒数第二层，用于提取语义嵌入
        self.embedding_layer = None
    
    def forward(self, x):
        """前向传播，返回 α
        
        公式: α = ReLU(Wx + b) + 1
        """
        # 通过前两层
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        
        # 保存嵌入
        self.embedding = h.detach()
        
        # 继续过剩余层
        h = self.net[3](h)
        h = self.net[4](h)
        h = self.net[5](h)
        h = self.net[6](h)
        
        alpha = F.relu(h) + 1.0
        return alpha
    
    def get_embedding(self, x):
        """获取语义嵌入（倒数第二层输出）"""
        _ = self.forward(x)
        return self.embedding
    
    def get_output(self, x):
        """获取完整输出：(α, b, u, emb)
        
        Args:
            x: [B, input_dim] 特征
        
        Returns:
            alpha: [B, K] Dirichlet 参数
            b: [B, K] 信念质量（证据理论）
            u: [B, 1] 认知不确定性
            emb: [B, hidden_dim//2] 语义嵌入
        """
        alpha = self.forward(x)
        S = alpha.sum(dim=1, keepdim=True)
        K = self.num_classes
        
        b = (alpha - 1) / S  # 信念质量
        u = K / S            # 认知不确定性
        
        return alpha, b, u, self.embedding


# =============================================================================
# 4. 训练证据头
# =============================================================================

def train_evidence_head(train_feats, train_labels, val_feats, val_labels,
                        input_dim, agent_name, num_epochs=30, lr=1e-3):
    """训练证据头
    
    损失函数：E-MSE + λ * KL 正则化
    E-MSE: ||y - p||² 其中 p = α / S（期望概率）
    KL: KL(Dir(α̃) || Dir(1)) 其中 α̃ 是非误分类伪计数
    
    Args:
        train_feats: [N, input_dim]
        train_labels: [N]
        val_feats: [N_val, input_dim]
        val_labels: [N_val]
        input_dim: 特征维度
        agent_name: 智能体名称（用于保存）
    
    Returns:
        head: 训练好的 EvidenceHead
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    head = EvidenceHead(input_dim, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    val_dataset = TensorDataset(val_feats, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=128)
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(num_epochs):
        head.train()
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            alpha = head(x)
            S = alpha.sum(dim=1, keepdim=True)
            
            # E-MSE 损失
            p = alpha / S
            y_onehot = F.one_hot(y, NUM_CLASSES).float()
            mse_loss = F.mse_loss(p, y_onehot)
            
            # KL 正则化: KL(Dir(α̃) || Dir(1))
            # 其中 α̃ = 对正确类保持 α，对错误类置 1（非误分类伪计数）
            # KL = ∑ α̃_k · [log(α̃_k) - ψ(α̃_k) - log(S̃) + ψ(S̃)]
            alpha_tilde = alpha * (1 - y_onehot) + y_onehot
            S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
            log_alpha_tilde = torch.log(alpha_tilde)
            log_S_tilde = torch.log(S_tilde)
            digamma_alpha = torch.digamma(alpha_tilde)
            digamma_S_tilde = torch.digamma(S_tilde)
            kl_loss = (alpha_tilde * (log_alpha_tilde - digamma_alpha - log_S_tilde + digamma_S_tilde)).sum(dim=1).mean()
            
            loss = mse_loss + 0.001 * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        head.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                alpha = head(x)
                preds = alpha.argmax(dim=1)
                correct += (preds == y).sum().item()
        
        val_acc = correct / len(val_labels)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = head.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")
    
    head.load_state_dict(best_state)
    torch.save(head.state_dict(), f'checkpoints/hateful_memes/{agent_name}_head.pt')
    print(f"  [OK] {agent_name} 训练完成, 最佳验证准确率: {best_val_acc:.4f}")
    
    return head


# =============================================================================
# 5. DS融合（与CIFAR-10N版本一致）
# =============================================================================

def ds_fusion_decision(all_beliefs, all_uncertainties, u_threshold=0.5):
    """Dempster-Shafer融合做全局决策"""
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
# 6. 智能体输出提取
# =============================================================================

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
    
    return alpha.cpu(), b.cpu(), u.cpu(), emb.cpu()


# =============================================================================
# 7. 主评估流程
# =============================================================================

def run_pipeline(max_train=2000, max_val=500):
    """运行完整Hateful Memes评估流程
    
    流程：
    1. 加载数据
    2. 提取各模态特征（缓存到磁盘）
    3. 训练三个智能体的证据头
    4. 评估各方法（多数投票、加权平均、DS融合、DS+共识）
    
    Args:
        max_train: 训练样本数（限制以加速调试，设为None用全部）
        max_val: 验证样本数
    """
    print("=" * 60)
    print("Hateful Memes 多模态评估")
    print("=" * 60)
    
    # ========== 1. 加载数据 ==========
    print("\n[1] 加载Hateful Memes数据...")
    train_texts, train_img_ids, train_labels = load_hateful_data('train', max_train)
    val_texts, val_img_ids, val_labels = load_hateful_data('dev', max_val)
    
    print(f"  训练集: {len(train_texts)} 样本 ({sum(train_labels)} hateful)")
    print(f"  验证集: {len(val_texts)} 样本 ({sum(val_labels)} hateful)")
    
    train_labels_t = torch.LongTensor(train_labels)
    val_labels_t = torch.LongTensor(val_labels)
    
    # ========== 2. 提取特征（带缓存） ==========
    cache_dir = 'data/features/hateful_memes'
    
    def load_or_extract(cache_name, extract_fn, *args):
        """带缓存的特征提取"""
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_path):
            print(f"  加载缓存: {cache_name}")
            return torch.load(cache_path, map_location='cpu', weights_only=False)
        print(f"  提取: {cache_name}")
        result = extract_fn(*args)
        torch.save(result, cache_path)
        return result
    
    print("\n[2] 提取特征...")
    
    # Agent1: 文本特征 (CharCNN -> 256维)
    text_extractor = TextFeatureExtractor(hidden_dim=256)
    
    print("  Agent1 (文本CharCNN):")
    train_text_feats = load_or_extract(
        'train_text.pt', text_extractor.encode, train_texts[:max_train] if max_train else train_texts
    )
    val_text_feats = load_or_extract(
        'val_text.pt', text_extractor.encode, val_texts[:max_val] if max_val else val_texts
    )
    print(f"    训练: {train_text_feats.shape}, 验证: {val_text_feats.shape}")
    
    # Agent2: 图像特征 (ResNet-18 -> 512维)
    image_extractor = ImageFeatureExtractor(output_dim=512)
    
    print("  Agent2 (图像ResNet-18):")
    train_img_feats = load_or_extract(
        'train_img.pt', image_extractor.encode, train_img_ids[:max_train] if max_train else train_img_ids
    )
    val_img_feats = load_or_extract(
        'val_img.pt', image_extractor.encode, val_img_ids[:max_val] if max_val else val_img_ids
    )
    print(f"    训练: {train_img_feats.shape}, 验证: {val_img_feats.shape}")
    
    # Agent3: 融合特征 (弱化MLP -> 128维)
    fusion_extractor = SimpleFusionExtractor(text_dim=256, image_dim=512, hidden_dim=128)
    
    print("  Agent3 (弱融合MLP):")
    max_t = min(max_train or len(train_texts), len(train_texts))
    max_v = min(max_val or len(val_texts), len(val_texts))
    train_fusion_feats = load_or_extract(
        'train_fusion.pt',
        fusion_extractor.encode,
        train_texts[:max_t], train_img_ids[:max_t]
    )
    val_fusion_feats = load_or_extract(
        'val_fusion.pt',
        fusion_extractor.encode,
        val_texts[:max_v], val_img_ids[:max_v]
    )
    print(f"    训练: {train_fusion_feats.shape}, 验证: {val_fusion_feats.shape}")
    
    # ========== 3. 训练证据头 ==========
    print("\n[3] 训练证据头...")
    
    head_check = 'checkpoints/hateful_memes/agent1_head.pt'
    if os.path.exists(head_check):
        print("  已存在训练好的证据头，跳过训练")
        heads = {
            'agent1': EvidenceHead(256, NUM_CLASSES).to(DEVICE),
            'agent2': EvidenceHead(512, NUM_CLASSES).to(DEVICE),
            'agent3': EvidenceHead(128, NUM_CLASSES).to(DEVICE),
        }
        for name in ['agent1', 'agent2', 'agent3']:
            heads[name].load_state_dict(torch.load(f'checkpoints/hateful_memes/{name}_head.pt',
                                                    map_location=DEVICE, weights_only=True))
            heads[name].eval()
    else:
        heads = {}
        print("  Agent1 (文本):")
        heads['agent1'] = train_evidence_head(
            train_text_feats, train_labels_t, val_text_feats, val_labels_t,
            input_dim=256, agent_name='agent1'
        )
        print("  Agent2 (图像):")
        heads['agent2'] = train_evidence_head(
            train_img_feats, train_labels_t, val_img_feats, val_labels_t,
            input_dim=512, agent_name='agent2'
        )
        print("  Agent3 (融合):")
        heads['agent3'] = train_evidence_head(
            train_fusion_feats, train_labels_t, val_fusion_feats, val_labels_t,
            input_dim=128, agent_name='agent3'
        )
    
    # ========== 4. 验证集评估 ==========
    print("\n[4] 验证集评估...")
    B = len(val_labels)
    
    # 提取验证集上各智能体输出
    all_alphas, all_beliefs, all_uncertainties, all_embs = [], [], [], []
    feat_map = {
        'agent1': val_text_feats[:B],
        'agent2': val_img_feats[:B],
        'agent3': val_fusion_feats[:B],
    }
    
    for name in ['agent1', 'agent2', 'agent3']:
        alpha, b, u, emb = get_agent_outputs(feat_map[name], heads[name])
        all_alphas.append(alpha)
        all_beliefs.append(b)
        all_uncertainties.append(u)
        all_embs.append(emb)
    
    y_true = val_labels_t.numpy()
    
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
    
    # ========== 方法3: 纯DS融合 ==========
    t0 = time.time()
    ds_preds, ds_rej, ds_u = ds_fusion_decision(all_beliefs, all_uncertainties, u_threshold=0.5)
    t_ds = time.time() - t0
    
    # ========== 方法4: DS+个体准确率 ==========
    # 衡量每个智能体单独表现的基线
    t0 = time.time()
    agent1_preds = all_beliefs[0].argmax(dim=1)
    agent2_preds = all_beliefs[1].argmax(dim=1)
    agent3_preds = all_beliefs[2].argmax(dim=1)
    t_ag = time.time() - t0
    
    # ========== 5. 汇总结果 ==========
    results = {
        'Agent1_Text': {
            'preds': agent1_preds, 'rejected': torch.zeros(B, dtype=torch.bool),
            'time': t_ag
        },
        'Agent2_Image': {
            'preds': agent2_preds, 'rejected': torch.zeros(B, dtype=torch.bool),
            'time': t_ag
        },
        'Agent3_Fusion': {
            'preds': agent3_preds, 'rejected': torch.zeros(B, dtype=torch.bool),
            'time': t_ag
        },
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
    }
    
    # ========== 6. 打印指标 ==========
    print(f"\n{'='*110}")
    print(f"{'方法':<20s} {'Acc%':<10s} {'F1%':<10s} {'ECE':<12s} {'Rej%':<10s} {'Acc_All%':<10s} {'Time(ms)':<10s}")
    print(f"{'-'*110}")
    
    metrics = {}
    for method_name, res in results.items():
        preds_np = res['preds'].numpy()
        rej_np = res['rejected'].numpy()
        
        rej_rate = rej_np.mean() * 100
        acc_all = accuracy_score(y_true, preds_np) * 100
        
        accepted = ~rej_np
        if accepted.sum() > 0:
            acc = accuracy_score(y_true[accepted], preds_np[accepted]) * 100
            f1 = f1_score(y_true[accepted], preds_np[accepted], average='binary') * 100
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
        
        t_ms = res.get('time', 0) * 1000
        metrics[method_name] = {
            'accuracy': acc, 'f1': f1, 'ece': float(ece),
            'rejection_rate': rej_rate, 'accuracy_all': acc_all, 'time_ms': t_ms
        }
        
        print(f"{method_name:<20s} {acc:<10.2f} {f1:<10.2f} {ece:<12.4f} "
              f"{rej_rate:<10.2f} {acc_all:<10.2f} {t_ms:<10.2f}")
    
    print(f"{'='*110}")
    
    # ========== 7. 保存结果 ==========
    with open('results/hateful_memes/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果保存至: results/hateful_memes/evaluation_results.json")
    
    # ========== 8. 绘图 ==========
    _plot_comparison_bar(metrics)
    _plot_confusion_matrices(results, y_true)
    _plot_uncertainty_analysis(all_uncertainties, all_beliefs, y_true)
    
    return metrics


def _plot_comparison_bar(metrics):
    """绘制各方法对比柱状图"""
    # 只绘制融合方法
    fusion_methods = ['MajorityVoting', 'WeightedAvg', 'DS_Fusion']
    labels = ['多数投票', '加权平均', 'DS融合']
    
    acc = [metrics[m]['accuracy'] for m in fusion_methods]
    f1 = [metrics[m]['f1'] for m in fusion_methods]
    ece = [metrics[m]['ece'] for m in fusion_methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
    
    for ax, vals, title, ylabel in zip(
        axes, [acc, f1, ece],
        ['准确率 (接受样本)', 'F1 (接受样本)', 'ECE (置信度校准)'],
        ['Accuracy (%)', 'F1 Score (%)', 'ECE']):
        
        bars = ax.bar(labels, vals, color=colors, alpha=0.8, width=0.5)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Hateful Memes 融合方法对比', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/hateful_memes_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图: figures/hateful_memes_comparison.png")


def _plot_confusion_matrices(results, y_true):
    """绘制混淆矩阵"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    methods_to_plot = ['Agent1_Text', 'Agent2_Image', 'Agent3_Fusion',
                       'MajorityVoting', 'WeightedAvg', 'DS_Fusion']
    titles = ['Agent1 (文本)', 'Agent2 (图像)', 'Agent3 (弱融合)',
              '多数投票', '加权平均', 'DS融合']
    
    for idx, method in enumerate(methods_to_plot):
        if method not in results:
            continue
        ax = axes[idx]
        cm = confusion_matrix(y_true, results[method]['preds'].numpy(), labels=[0, 1])
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(titles[idx], fontsize=12)
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
    
    for i in range(len(methods_to_plot), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Hateful Memes 混淆矩阵', fontsize=15)
    plt.tight_layout()
    plt.savefig('figures/hateful_memes_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵: figures/hateful_memes_confusion.png")


def _plot_uncertainty_analysis(all_uncertainties, all_beliefs, y_true):
    """不确定性分析：各智能体的不确定性分布
    
    对比正确分类与错误分类样本的不确定性分布，
    用于验证认知不确定性与错误预测之间的相关性。
    
    Args:
        all_uncertainties: list of [N,1] tensors, 各智能体的 u
        all_beliefs: list of [N,K] tensors, 各智能体的信念质量 b
        y_true: [N] ground truth labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    agent_names = ['Agent1 (文本)', 'Agent2 (图像)', 'Agent3 (弱融合)']
    
    for idx, (ax, name) in enumerate(zip(axes, agent_names)):
        u = all_uncertainties[idx].numpy().flatten()
        correct = (all_beliefs[idx].argmax(dim=1).numpy() == y_true)
        
        # 正确和错误样本的 u 分布
        # 理论上：正确分类样本的 u 应较低，错误样本的 u 应较高
        ax.hist(u[correct], bins=20, alpha=0.6, label='正确', color='green', density=True)
        ax.hist(u[~correct], bins=20, alpha=0.6, label='错误', color='red', density=True)
        ax.set_xlabel('不确定性 u')
        ax.set_ylabel('密度')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('各智能体不确定性分布 (正确vs错误)', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/hateful_memes_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"不确定性分析图: figures/hateful_memes_uncertainty.png")


# =============================================================================
# 8. 入口
# =============================================================================

if __name__ == '__main__':
    run_pipeline(max_train=2000, max_val=500)