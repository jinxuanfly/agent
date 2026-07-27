"""
CIFAR-10 特征预提取脚本
=====================
用于在CPU上高效训练CIFAR-10N智能体。
策略：先一次性用backbone提取特征保存到磁盘，
      然后只训练小型证据头（MLP），避免重复跑backbone。

用法：
    python src/step4/extract_features.py
    结束后运行: python src/step4/train_heads.py
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import pickle
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cpu')

os.makedirs('data/features', exist_ok=True)

# =============================================================================
# 1. 加载CIFAR-10
# =============================================================================

def load_cifar10():
    """加载CIFAR-10原始图片"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform)
    
    return trainset, testset


# =============================================================================
# 2. 定义轻量级backbone（去掉证据头）
# =============================================================================

class ResNetBackbone(nn.Module):
    """ResNet-18 backbone, 输出512维特征"""
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(weights='DEFAULT')
        # 去掉最后的全连接层
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.output_dim = 512
    
    def forward(self, x):
        x = self.features(x)  # [B, 512, 1, 1]
        return x.view(x.size(0), -1)  # [B, 512]


class ViTBackbone(nn.Module):
    """ViT-Tiny backbone, 输出192维特征"""
    def __init__(self):
        super().__init__()
        self.patch_size = 4
        self.image_size = 32
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, 192, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 192))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, 192))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192, nhead=4, dim_feedforward=384, dropout=0,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(192)
        self.output_dim = 192
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]  # [B, 192]


# =============================================================================
# 3. 批量提取特征
# =============================================================================

@torch.no_grad()
def extract_features(backbone, dataloader, desc=""):
    """用backbone提取所有样本的特征"""
    backbone.eval()
    all_features = []
    
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        features = backbone(images)
        all_features.append(features.cpu())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  {desc}: {batch_idx+1}/{len(dataloader)} batches")
    
    return torch.cat(all_features, dim=0)


def extract_all_features():
    """提取所有智能体的特征"""
    print("=" * 60)
    print("CIFAR-10 特征预提取")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] 加载CIFAR-10...")
    trainset, testset = load_cifar10()
    print(f"  训练集: {len(trainset)} 样本")
    print(f"  测试集: {len(testset)} 样本")
    
    train_loader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    
    # 2. 创建backbone模型
    print("\n[2] 创建backbone模型...")
    backbones = {
        'resnet18': ResNetBackbone(),
        'vit_tiny': ViTBackbone(),
    }
    
    # 3. 提取训练集特征
    print("\n[3] 提取训练集特征...")
    feat_dims = {}
    for name, backbone in backbones.items():
        print(f"\n  提取 {name} 训练集特征...")
        t0 = time.time()
        train_feats = extract_features(backbone, train_loader, desc=name)
        feat_dims[name] = train_feats.shape[1]
        print(f"  训练集特征尺寸: {train_feats.shape}, 用时: {time.time()-t0:.1f}s")
        torch.save(train_feats, f'data/features/train_{name}.pt')
    
    # 4. 提取测试集特征
    print("\n[4] 提取测试集特征...")
    for name, backbone in backbones.items():
        t0 = time.time()
        test_feats = extract_features(backbone, test_loader, desc=name)
        print(f"  测试集特征尺寸: {test_feats.shape}, 用时: {time.time()-t0:.1f}s")
        torch.save(test_feats, f'data/features/test_{name}.pt')
    
    # 5. 保存原始标签
    print("\n[5] 保存标签...")
    train_labels = torch.LongTensor(trainset.targets)
    test_labels = torch.LongTensor(testset.targets)
    
    torch.save({
        'train_labels': train_labels,
        'test_labels': test_labels,
    }, 'data/features/labels.pt')
    
    print(f"\n  标签已保存至 data/features/labels.pt")
    
    print(f"\n{'='*60}")
    print("特征预提取完成!")
    print(f"文件已保存至 data/features/")
    print(f"  训练集特征: train_resnet18.pt ({feat_dims['resnet18']}d), train_vit_tiny.pt ({feat_dims['vit_tiny']}d)")
    print(f"  测试集特征: test_resnet18.pt, test_vit_tiny.pt")
    print(f"  标签: labels.pt")
    print(f"{'='*60}")


if __name__ == '__main__':
    print("注意: 第一次提取ResNet-18特征可能需要5-10分钟 (CPU)...")
    ans = input("继续? (y/n): ")
    if ans.lower() == 'y':
        extract_all_features()
