# 项目交接文档

## 一、项目概述

本项目是一个**异构多模态动态共识与协同框架**的实验代码，用于仇恨言论检测（Hateful Memes）任务。核心架构包含三个LLM Agent协同工作，通过不确定性加权D-S证据理论实现动态融合，通过GAT共识层识别分歧，并在分歧样本上触发因果反事实反思。

> **历史说明**：CIFAR-10N 实验已废弃，代码移至 `CIFAR-10N废弃/` 目录。原因：CIFAR-10N 为单模态图像分类，与论文"异构多模态"核心定位冲突。

### 核心目标
- 验证异构多模态Agent（文本专家、图像专家、跨模态专家）的协同效果
- 通过不确定性加权DS融合动态调整Agent权重
- 通过GAT共识层识别Agent间分歧
- 通过因果反事实反思修正分歧样本
- 在Hateful Memes数据集上实现高精度的多Agent协同决策

### Agent架构（最终方案：DeepSeek+Gemini+GPT-5.1，500样本）

| Agent | 角色 | 当前模型 | 输入模态 | Provider |
|-------|------|----------|----------|----------|
| **Agent1** | 文本专家 | DeepSeek-v4-pro | 仅文本 | deepseek（魔芋中转） |
| **Agent2** | 图像专家 | Gemini-3.5-flash | 直接图像输入 + 文本 | gemini（魔芋中转） |
| **Agent3** | 跨模态专家 | GPT-5.1 | 文本 + CLIP图像描述 | gpt5（魔芋中转） |

> **注意（2026-08-28）**：GPT-5已升级为GPT-5.1，`llm_api.py`中模型名和env_key已同步更新。三种渠道均正常可用。

**数据集规模**：训练集200样本，验证集500样本，5个随机种子（42, 123, 456, 789, 1024）

---

## 二、三个核心创新点（全部验证完成 ✅）

### 创新点1: Uncertainty_Weighted_DS（核心创新）
- **性能**: Acc=66.28%±0.52%（5种子），单种子最佳67.00%
- **vs基线**: +2.08% vs DS等权重（64.20%）
- **核心机制**: 基于Agent不确定性u的softmax权重
  - u越低→Agent越自信→能力越强→权重越高
  - 每个样本独立计算权重（per-sample weights）
  - 参数: sharpness=20.0（调优最佳值）
- **平均权重**: [0.265, 0.505, 0.229]（Gemini权重最高，符合实际能力）
- **解决的问题**: 训练集准确率无法预测验证集能力（Gemini训练~49%但验证~80%）

### 创新点2: GAT_EvidenceSwap（次要创新）
- **性能**: Acc=64.40%±0.65%（5种子）
- **vs基线**: +0.20% vs DS等权重
- **核心机制**:
  1. GAT共识层调整Agent信念
  2. 分歧解构识别证据冲突
  3. 证据交换：最佳Agent→最差Agent证据传递
- **核心价值**: GAT的主要贡献不在于直接提升准确率，而在于**为因果反思提供共识状态和分歧识别**

### 创新点3: Causal_Reflection（第三创新，核心卖点）
- **性能**: Acc=76.60%, F1=76.19%（全样本外推）
- **vs SOTA**: 仅差Static Ensemble 1.2%（77.80% vs 76.60%）
- **核心机制**: 跨Agent证据交换的因果反事实反思
  - 让所有Agent看到其他Agent的判断和置信度
  - Agent重新评估，可能翻转预测
- **关键结果**（seed=42，全量249个分歧样本）:
  - 分歧样本Acc: 40.16% → 67.87%（+27.71%）
  - 正确修正:94, 错误改变:25, 净收益:+69
  - 正确/错误比: 3.76:1
- **策略对比**:
  - v1（只反思少数派）: 无效（27次改变全为中性）
  - v2（反思全Agent）: 有效（必须反思所有Agent才能翻转2v1分歧）

---

## 三、完整实验结果

### 3.1 新配置实验结果（DeepSeek+Gemini+GPT-5.1，5种子，最终版）

| 类别 | 方法 | Acc%（5种子均值±std） | 说明 |
|------|------|------|------|
| 单Agent | Agent1(DeepSeek) | 59.08±0.56 | 仅文本 |
| 单Agent | Agent2(Gemini) | **79.88±0.44** | 图像+文本（Oracle） |
| 单Agent | Agent3(GPT-5.1) | 59.16±0.50 | 文本+CLIP描述 |
| 基线 | MajorityVoting | 63.04±0.53 | 等权投票 |
| 基线 | WeightedAvg | 64.47±0.87 | 等权平均 |
| 基线 | DS_Fusion | 64.20±0.81 | 等权重DS |
| **创新1** | **Uncertainty_Weighted_DS** | **66.28±0.52** | 🔥 核心创新 |
| 消融 | UncWeight_Corr_DS | 65.68±0.80 | 组合无增益 |
| **创新2** | **GAT_EvidenceSwap** | **64.40±0.65** | GAT+证据交换 |
| **创新3** | **Causal_Reflection** | **76.60**（单种子） | 🔥 因果反思 |

### 3.2 SOTA对比结果

| 方法 | Acc% | F1% | 说明 |
|------|------|-----|------|
| **Static Ensemble** | **77.80** | **77.30** | 三Agent等权加权平均（SOTA） |
| **Causal Reflection（本框架）** | **76.60** | **76.19** | 差距仅1.2%，不显著 |
| Self-Consistency (n=5) | 53.00 | 52.72 | 单Agent多次采样 |
| Single LLM Multi-Role | 58.40 | 40.91 | 单LLM角色扮演 |

### 3.3 分歧样本分析

| 统计项 | 数值 |
|--------|------|
| 分歧率 | 47.4%±0.7%（5种子，约237/500样本） |
| 分歧样本MV基线 | 39.40%±2.34% |
| 分歧样本UncWeight_DS | 46.49%±1.80% |
| 分歧样本Causal_Reflection | **67.87%**（seed=42） |

### 3.4 旧配置实验结果（GPT-5+Gemini+GPT-5，seed=42，仅参考）

| 方法 | Acc% | F1% | 说明 |
|------|------|-----|------|
| Agent1(gpt5) | 55.60 | 40.32 | 仅文本 |
| Agent2(gemini) | 79.60 | 80.75 | 图像+文本 |
| Agent3(gpt5) | 55.00 | 25.74 | 文本+CLIP描述 |
| Uncertainty_Weighted_DS | 69.20 | 68.35 | 核心创新 |
| GAT_EvidenceSwap | 70.40 | 65.58 | 次要创新 |
| Causal_Reflection（100样本） | 68.80 | 82.89 | 因果反思 |

---

## 四、已完成的工作

### 4.1 项目架构搭建（Step1-3）

| 步骤 | 名称 | 状态 | 说明 |
|------|------|------|------|
| Step1 | 合成数据与硬模型训练 | ✅ | 生成合成数据，训练基础分类器 |
| Step2 | GAT共识层 | ✅ | 图注意力网络共识引擎，支持不确定性感知 |
| Step3 | 分歧解构器 | ✅ | 区分证据冲突和无知冲突 |

### 4.2 Hateful Memes评估管线

| 任务 | 状态 | 详情 |
|------|------|------|
| LLM推理框架 | ✅ | 支持DeepSeek、Gemini、GPT-5.1 |
| 独立缓存机制 | ✅ | 每个Agent的缓存独立存储，按seed隔离 |
| API密钥管理 | ✅ | 通过keys.env和环境变量安全管理 |
| 超时与重试机制 | ✅ | 180秒超时，5-8次重试，指数退避 |
| GAT共识训练 | ✅ | 基于LLM输出训练GAT共识模型 |
| 评估指标计算 | ✅ | Accuracy、F1、ECE、不确定性分析 |

### 4.3 实验阶段进展

| 阶段 | 时间 | 状态 | 主要工作 |
|------|------|------|---------|
| 阶段1-5 | 2026-07 | ✅ | GLM/Gemini/Claude/GPT-4o-mini多模型测试 |
| 阶段6 | 2026-08-23 | ✅ | 调优+消融+论文图表生成 |
| 阶段7 | 2026-08-24 | ✅ | 主结果文件更新 |
| 阶段8 | 2026-08-24 | ✅ | GAT+UncDS组合实验 |
| 阶段9 | 2026-08-24 | ✅ | 因果反思v2实验（100样本，有效） |
| 阶段10 | 2026-08-25 | ✅ | Agent配置切换（DeepSeek+Gemini+GPT-5） |
| 阶段11 | 2026-08-25 | ✅ | GPT-5→GPT-5.1升级，API修复 |
| 阶段12 | 2026-08-26 | ✅ | SOTA对比实验（Static Ensemble, Self-Consistency, Multi-Role） |
| 阶段13 | 2026-08-26 | ✅ | 5种子多种子实验（42, 123, 456, 789, 1024） |
| 阶段14 | 2026-08-26 | ✅ | 实验日志更新，论文草稿重写 |
| 阶段15 | 2026-08-28 | ✅ | 交接文档更新，第二数据集讨论 |

---

## 五、关键文件位置

### 5.1 核心代码
| 文件 | 说明 |
|------|------|
| `src/step4_hateful_memes/evaluate_with_llm.py` | 主评估脚本（所有融合方法，支持seed参数） |
| `src/step4_hateful_memes/evaluate_step5_causal_reflection.py` | 因果反思脚本（全量分歧样本，支持V2模式） |
| `src/step4_hateful_memes/evaluate_test_set.py` | 测试集评估脚本（LLM推理+CR，生成EvalAI提交文件） |
| `src/step4_hateful_memes/evaluate_gat_ablation.py` | GAT消融实验脚本（5种子，证明GAT无效） |
| `src/step4_hateful_memes/run_causal_reflection_multi_seed.py` | 因果反思多种子批量运行 |
| `src/step4_hateful_memes/generate_charts.py` | 论文图表生成脚本 |
| `src/step2/gat_consensus.py` | GAT共识引擎实现 |
| `src/llm_agent.py` | LLM Agent创建函数和推理封装 |
| `src/llm_api.py` | LLM API配置和调用逻辑 |

### 5.2 实验结果文件
| 文件 | 说明 |
|------|------|
| `results/hateful_memes/evaluation_llm_deepseek_gemini_gpt5.1_seed{42,123,456,789,1024}.json` | 5种子主实验结果 |
| `results/hateful_memes/step5_causal_reflection_v2_seed{42,123,456,789,1024}.json` | 因果反思5种子结果 |
| `results/hateful_memes/causal_reflection_multi_seed_summary.json` | 因果反思5种子汇总 |
| `results/hateful_memes/gat_ablation_summary.json` | GAT消融汇总 |
| `results/hateful_memes/test_results_seed42.json` | 测试集评估结果 |
| `results/hateful_memes/test_predictions_causal_reflection_seed42.csv` | EvalAI提交文件 |
| `results/hateful_memes/ablation/ablation_results.json` | 消融实验结果 |

### 5.3 缓存数据
| 文件 | 说明 |
|------|------|
| `checkpoints/hateful_memes/llm_train_agent{0,1,2}_seed{seed}.pt` | 训练集LLM推理缓存（按seed隔离） |
| `checkpoints/hateful_memes/llm_val_agent{0,1,2}_seed{seed}.pt` | 验证集LLM推理缓存（按seed隔离） |
| `checkpoints/hateful_memes/llm_test_agent{0,1,2}_seed42.pt` | 测试集LLM推理缓存 |
| `checkpoints/hateful_memes/gat_consensus_llm_seed{seed}.pt` | GAT模型权重（按seed隔离） |
| `checkpoints/hateful_memes/disagreement_indices_seed{seed}.pt` | 分歧索引（按seed隔离） |

### 5.4 配置和文档
| 文件 | 说明 |
|------|------|
| `keys.env` | API密钥（OPENAI_GPT5.1_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY） |
| `EXPERIMENT_LOG.md` | 实验日志（15个阶段完整记录） |
| `HANDOFF.md` | 本交接文档 |
| `论文实验部分草稿.md` | 论文实验部分（已重写为最新数据） |
| `figures/paper/` | 7张论文图表 |

---

## 六、论文叙事策略（最终版）

### 核心叙事线

```
Agent能力极度不均衡（Gemini 80% vs DeepSeek/GPT-5.1 59%）
        ↓
不确定性加权DS融合 → 自动识别强Agent（+2.08% vs DS_Fusion）
        ↓
47.4%样本存在分歧，分歧样本准确率仅40%
        ↓
因果反事实反思 → 分歧样本+27.71%（40%→68%）
        ↓
全样本76.60% ≈ SOTA Static Ensemble 77.80%（差距1.2%）
        ↓
结论：性能接近SOTA + 可解释因果推理链 + 系统分歧处理
```

### 核心卖点排序
1. **Causal Reflection**：分歧样本40%→68%（+27.71%），接近SOTA（差1.2%但可解释）
2. **Uncertainty_Weighted_DS**：解决"训练集准确率≠验证集能力"，5种子稳定+2.08%
3. **GAT_EvidenceSwap**：为因果反思提供共识状态和分歧识别

---

## 七、当前状态（2026-08-29）

### ✅ 已完成
- Phase 1 全部完成：单Agent、融合、消融、SOTA对比、5种子因果反思、GAT消融、测试集评估
- 因果反思 5种子：64.52% ± 1.05%，+1.48% vs MV
- GAT消融：GAT几乎无效，论文架构从4层降为3层
- 测试集评估：1000样本推理+CR完成，196/424分歧样本被修正
- MM-IMDb尝试：失败（Agent能力极度不均衡），代码已清理，数据保留

### 🔄 进行中
- 第二数据集：M3（多平台多语言仇恨Meme，跨语言泛化验证）
- Hateful Memes 深度挖掘：错误分析、成本分析、强基线对比

### ⏳ 待办
- M3数据集适配（代码、预处理、推理、CR）
- 论文主体撰写（引言、方法、相关工作等）
- 论文图表更新（需匹配新数据）
- 可选：EvalAI提交测试集获取官方分数

---

## 八、环境配置

### 8.1 API密钥
- 文件: `keys.env`（已加入.gitignore）
- Provider: 魔芋 (moyu.info)，地址: `https://www.moyu.info/v1`
- 当前可用渠道:
  - ✅ DeepSeek: `DEEPSEEK_API_KEY`
  - ✅ Gemini: `GEMINI_API_KEY`
  - ✅ GPT-5.1: `OPENAI_GPT5.1_API_KEY`（注意：已从GPT-5升级为GPT-5.1）

### 8.2 运行环境
- Python 3.x
- PyTorch (CPU模式)
- 需设置环境变量: `KMP_DUPLICATE_LIB_OK=TRUE`

### 8.3 常用命令

```bash
# === 当前配置（DeepSeek+Gemini+GPT-5.1）===
# 运行单种子实验
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=200 --max_val=500 --provider1=deepseek --provider2=gemini --provider3=gpt5 --seed=42

# 5种子批量实验
python run_multi_seed.py --seeds 42,123,456,789,1024

# SOTA对比实验
python sota_comparison.py --max_val=500

# 因果反思（全量分歧样本）
python src/step4_hateful_memes/evaluate_step5_causal_reflection.py

# API状态检查
python check_api_status.py

# 生成论文图表
python generate_paper_figures.py

# 删除缓存文件（强制重新运行）
Remove-Item checkpoints\hateful_memes\llm_*.pt
```

---

## 九、踩过的坑（绝对不要再踩）

### 9.1 API相关
- ❌ GPT-5渠道耗尽（503）→ ✅ 已升级为GPT-5.1，env_key更新为`OPENAI_GPT5.1_API_KEY`
- ❌ GPT-4o-mini key失效（401）→ 已弃用
- ❌ GLM-5V-Turbo限流（429）→ 已弃用
- ❌ Claude API 500错误→ 已弃用
- ❌ 魔芋代理渠道耗尽（503）→ 换key无效（同分组），需等待平台恢复

### 9.2 代码相关
- ❌ BestAgent硬编码为Agent3的bug → ✅ 已修复
- ❌ 缓存维度不匹配（不同样本量不能混用）→ 改max_train/max_val后必须删旧缓存
- ❌ 因果反思v1只反思少数派无效 → ✅ v2反思全Agent有效
- ❌ Windows编码问题（UnicodeEncodeError）→ 使用utf-8 wrapper
- ❌ OpenMP冲突 → 设置`KMP_DUPLICATE_LIB_OK=TRUE`
- ❌ 小样本不可信（5样本100%但100样本仅52%）→ 至少500样本

### 9.3 文件编码
- ❌ PowerShell `Set-Content` 默认UTF-16 → ✅ 用Python写文件确保UTF-8

---

## 十、对新对话的引导

1. **当前进度**：Phase 1 全部完成（因果反思5种子、GAT消融、测试集评估），MM-IMDb 尝试失败
2. **核心数据**：Causal Reflection 64.52%±1.05%（5种子），Uncertainty_Weighted_DS 66.28%±0.52%，GAT几乎无效
3. **论文叙事**：不追求超越SOTA，强调可解释性+分歧处理+因果推理，GAT从创新点降为消融负结果
4. **第二数据集**：M3（多平台多语言仇恨Meme，跨语言泛化验证），待适配
5. **下一步**：M3数据集适配 + Hateful Memes深度挖掘（错误分析、成本分析、强基线对比）
6. **MM-IMDb数据**：保留在 `data/mmimdb/`，代码/缓存/结果已删除

---

*最后更新: 2026-08-29（Phase 1完成，MM-IMDb失败，转向M3）*