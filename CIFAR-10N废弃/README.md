# CIFAR-10N 废弃代码归档

本目录存放已废弃的 CIFAR-10N 相关代码，于 2026-08-15 从主项目中移除。

## 废弃原因

1. **定位冲突**：CIFAR-10N 为单模态图像分类任务，与论文"异构多模态 Agent 协作"的核心定位直接冲突。
2. **共识层假调用**：`evaluate_cifar10n.py` 中 `simple_consensus(use_gat=True)` 实际从未调用真正的 GAT 引擎，所有"共识有效"的结论不可用。
3. **Agent 设计偏离论文**：3 个 Agent 均为图像分类器（ResNet-18 / ViT-Tiny / PixelMLP），属于同模态集成，并非异构多模态。

## 移动文件清单

| 原路径 | 文件 | 作用 |
|--------|------|------|
| `src/step4/evaluate_cifar10n.py` | CIFAR-10N 端到端评估 | DS 融合 + 共识层（假调用） |
| `src/step4/extract_features.py` | 特征预提取 | ResNet/ViT/Pixel 特征 |
| `src/step4/train_heads.py` | 证据头训练 | 3 个 Agent 的 MLP 证据头 |
| `src/step4/diagnose_consensus_root_cause.py` | 共识层根因诊断 | 定位 DS_Consensus == DS_Fusion |
| `src/train_gat_consensus.py` | GAT 共识层训练脚本 | 在 CIFAR-10N 特征上训练 GAT |
| `src/diagnose_conflict.py` | 冲突系数 K 诊断 | CIFAR-10N 上的冲突分布分析 |

## 保留的文件

- `src/step4/evaluate_hateful_memes.py`：Hateful Memes 基础版评估，与 CIFAR-10N 无关，保留原位。

## 论文中相关验证点的替代方案

| 原验证点 | 替代方案 |
|---------|---------|
| 拜占庭容错 | 在 Hateful Memes 或合成数据上注入恶意 Agent |
| 过度自信分析 | 在 Hateful Memes 上分析 Agent 的 u 值分布 |
| 噪声标签鲁棒性 | 在 Hateful Memes 训练集中人工翻转部分标签 |

## 注意事项

- 本目录中的代码**不再维护**，导入路径仍指向原 `step4.*` 位置，无法直接运行。
- 如需参考历史实现，请阅读源码注释。
- 主项目入口 `main.py` 已移除 `--step 4`、`--setup`、`--ablate`、`--detail`、`--diagnose` 等参数。
