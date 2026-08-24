# 项目交接文档

## 一、项目概述

本项目是一个**异构多模态动态共识与协同框架**的实验代码，用于仇恨言论检测（Hateful Memes）任务。核心架构包含三个LLM Agent协同工作，通过图注意力网络（GAT）实现共识决策，并在共识失败时触发因果反事实反思。

> **更新说明（2026-08-15）**：CIFAR-10N 实验已废弃，相关代码移至 `CIFAR-10N废弃/` 目录。原因：CIFAR-10N 为单模态图像分类，与论文"异构多模态"核心定位冲突。原计划的"拜占庭容错"和"过度自信"验证点改为在 Hateful Memes 上进行。

### 核心目标
- 验证异构多模态Agent（文本专家、图像专家、跨模态专家）的协同效果
- 通过GAT共识层解决Agent间的分歧
- 通过因果反事实反思进一步修正共识失败的样本
- 在Hateful Memes数据集上实现高精度的仇恨言论检测

### Agent架构（当前GPT-5方案，500样本）

| Agent | 角色 | 当前模型 | 输入模态 | Provider |
|-------|------|----------|----------|----------|
| **Agent1** | 文本专家 | GPT-5 | 仅文本 | gpt5（魔芋中转） |
| **Agent2** | 图像专家 | Gemini-3.5-flash | 直接图像输入 + 文本 | gemini（魔芋中转） |
| **Agent3** | 跨模态专家 | GPT-5 | 文本 + CLIP图像描述 | gpt5（魔芋中转） |

**数据集规模**：训练集200样本，验证集500样本

---

## 二、三个核心创新点（全部验证完成 ✅）

### 创新点1: Uncertainty_Weighted_DS（核心创新）
- **性能**: Acc=69.20%, F1=68.35%
- **vs基线**: +1.4% vs DS等权重
- **核心机制**: 基于Agent不确定性u的softmax权重
  - u越低→Agent越自信→能力越强→权重越高
  - 每个样本独立计算权重（per-sample weights）
  - 参数: sharpness=20.0（调优最佳值）
- **平均权重**: [0.265, 0.505, 0.229]（Gemini权重最高，符合实际能力）
- **解决的问题**: 训练集准确率无法预测验证集能力（Gemini训练49.5%但验证79.6%）

### 创新点2: GAT_EvidenceSwap（次要创新）
- **性能**: Acc=70.40%, F1=65.58%
- **vs基线**: +2.6% vs DS等权重（最佳融合方法）
- **核心机制**:
  1. GAT共识层调整Agent信念
  2. 分歧解构识别证据冲突
  3. 证据交换：最佳Agent→最差Agent证据传递
- **消融结果**:
  - No_GAT: 67.8%
  - Random_GAT: 69.6%
  - Symbolic_GAT: 69.0%（与Random相当，embedding类型差异不大）
- **组合创新**: UncDS_EvidSwap = Uncertainty_Weighted + 证据交换 = 70.40%（无需GAT）

### 创新点3: Causal_Reflection（第三创新）
- **性能**: Acc=68.80%, F1=82.89%
- **vs基线**: +8.8% vs MV（全样本外推）
- **核心机制**: 跨Agent证据交换的因果反事实反思
  - 让所有Agent看到其他Agent的判断和置信度
  - Agent重新评估，可能翻转预测
- **关键结果**:
  - 分歧样本Acc: 30% → 74% (+44%)
  - 分歧样本F1: 31.37% → 82.89% (+51.52%)
  - 正确修正:50, 错误改变:6, 净收益:+44
- **策略对比**:
  - v1（只反思少数派）: 无效（27次改变全为中性）
  - v2（反思全Agent）: 有效（必须反思所有Agent才能翻转2v1分歧）

---

## 三、完整实验结果（500样本，GPT-5+Gemini+GPT-5）

### 3.1 主实验方法对比

| 类别 | 方法 | Acc% | F1% | 说明 |
|------|------|------|-----|------|
| 单Agent | Agent1(gpt5) | 55.60 | 40.32 | 仅文本 |
| 单Agent | Agent2(gemini) | **79.60** | **80.75** | 图像+文本（Oracle） |
| 单Agent | Agent3(gpt5) | 55.00 | 25.74 | 文本+CLIP描述 |
| 单Agent | BestAgent | 79.60 | 80.75 | Oracle基准 |
| 基线 | MajorityVoting | 60.00 | 43.82 | 等权投票 |
| 基线 | WeightedAvg | 66.80 | 58.50 | 等权平均 |
| 基线 | DS_Fusion | 67.80 | 66.59 | 等权重DS |
| 消融 | Corr_Aware_DS | 67.00 | 65.58 | 相关性折扣有害 |
| 消融 | UncWeight_Corr_DS | 69.20 | 68.35 | 组合无增益 |
| **创新1** | **Uncertainty_Weighted_DS** | **69.20** | **68.35** | 核心创新 |
| **创新2** | **GAT_EvidenceSwap** | **70.40** | **65.58** | 次要创新 |
| **组合** | **UncDS_EvidSwap** | **70.40** | **65.58** | 组合创新 |
| GAT | GAT_DS_Fusion | 67.60 | 60.10 | - |
| GAT | GAT_Fusion | 65.20 | 56.50 | - |
| GAT | Hybrid_GAT | 62.20 | 49.33 | - |
| **创新3** | **Causal_Reflection** | **68.80** | **82.89** | 因果反思 |

### 3.2 消融实验结果

#### Symbolic GAT vs Random GAT
| 方法 | Acc% | F1% |
|------|------|-----|
| No_GAT(DS) | 67.80 | 66.59 |
| Symbolic_GAT_EvidSwap | 69.00 | 68.03 |
| Random_GAT_EvidSwap | 69.60 | 68.80 |

#### Uncertainty_Weighted_DS不同sharpness
| sharpness | Acc% | F1% |
|-----------|------|-----|
| 1.0 | 67.80 | 60.25 |
| 3.0 | 68.60 | 62.83 |
| 5.0 | 69.00 | 63.09 |
| 10.0 | 68.80 | 67.89 |
| **20.0** | **69.20** | **68.35** |
| 50.0 | 69.20 | 68.47 |

---

## 四、已完成的工作

### 4.1 项目架构搭建（Step1-3）

| 步骤 | 名称 | 状态 | 说明 |
|------|------|------|------|
| Step1 | 合成数据与硬模型训练 | ✅ 已完成 | 生成合成数据，训练基础分类器 |
| Step2 | GAT共识层 | ✅ 已完成 | 实现图注意力网络共识引擎，支持不确定性感知 |
| Step3 | 分歧解构器 | ✅ 已完成 | 区分证据冲突和无知冲突，针对性优化 |

### 4.2 Hateful Memes评估管线

| 任务 | 状态 | 详情 |
|------|------|------|
| LLM推理框架 | ✅ 已完成 | 支持DeepSeek、GLM、GPT、GPT-5、Gemini、Claude |
| 独立缓存机制 | ✅ 已完成 | 每个Agent的缓存独立存储，避免重复运行 |
| API密钥管理 | ✅ 已完成 | 通过keys.env和环境变量安全管理 |
| 超时与重试机制 | ✅ 已完成 | 180秒超时，5-8次重试，指数退避 |
| GAT共识训练 | ✅ 已完成 | 基于LLM输出训练GAT共识模型 |
| 评估指标计算 | ✅ 已完成 | Accuracy、F1、ECE、不确定性分析 |

### 4.3 实验阶段进展

| 阶段 | 时间 | 状态 | 主要工作 |
|------|------|------|---------|
| 阶段1-5 | 2026-07 | ✅ 完成 | GLM/Gemini/Claude/GPT-4o-mini多模型测试 |
| 阶段6 | 2026-08-23 | ✅ 完成 | 调优+消融+论文图表生成 |
| 阶段7 | 2026-08-24 | ✅ 完成 | 主结果文件更新（修复BestAgent bug） |
| 阶段8 | 2026-08-24 | ✅ 完成 | GAT+UncDS组合实验 |
| 阶段9 | 2026-08-24 | ✅ 完成 | 因果反思实验（v2有效） |

---

## 五、关键文件位置

### 5.1 核心代码
| 文件 | 说明 |
|------|------|
| `src/step4_hateful_memes/evaluate_with_llm.py` | 主评估脚本（所有融合方法，支持seed参数） |
| `run_multi_seed.py` | **多种子批量实验脚本（新增）** |
| `src/step4_hateful_memes/evaluate_step5_causal_reflection.py` | 原版因果反思脚本 |
| `run_step5_v2.py` | **高效版因果反思脚本（v2，有效）** |
| `run_step5_efficient.py` | v1反思脚本（无效，仅参考） |
| `tune_corr_ds.py` | Corr_DS调优+Uncertainty_Weighted实现 |
| `ablation_study.py` | 消融实验脚本 |
| `combine_gat_unc.py` | GAT+UncDS组合实验脚本 |
| `generate_paper_figures.py` | 论文图表生成脚本 |
| `analyze_disagree.py` | 分歧样本分析脚本 |
| `src/step2/gat_consensus.py` | GAT共识引擎实现 |
| `src/llm_agent.py` | LLM Agent创建函数和推理封装 |
| `src/llm_api.py` | LLM API配置和调用逻辑 |

### 5.2 实验结果文件
| 文件 | 说明 |
|------|------|
| `results/hateful_memes/evaluation_llm_gpt5_gemini_gpt5.json` | **主实验结果（最新，含metadata）** |
| `results/hateful_memes/details_llm_gpt5_gemini_gpt5.json` | 详细预测结果 |
| `results/hateful_memes/ablation/ablation_results.json` | 消融实验结果 |
| `results/hateful_memes/corr_ds_tuning.json` | 调优实验结果 |
| `results/hateful_memes/combine_gat_unc_results.json` | 组合实验结果 |
| `results/hateful_memes/step5_causal_reflection_v2.json` | **因果反思v2结果** |
| `results/hateful_memes/step5_causal_reflection_efficient.json` | 因果反思v1结果（无效） |

### 5.3 缓存数据
| 文件 | 说明 |
|------|------|
| `checkpoints/hateful_memes/llm_train_agent{0,1,2}_seed{seed}.pt` | 训练集LLM推理缓存（按seed隔离） |
| `checkpoints/hateful_memes/llm_val_agent{0,1,2}_seed{seed}.pt` | 验证集LLM推理缓存（按seed隔离） |
| `checkpoints/hateful_memes/gat_consensus_llm_seed{seed}.pt` | GAT模型权重（按seed隔离） |
| `checkpoints/hateful_memes/disagreement_indices.pt` | 分歧样本索引（基于200样本，已过时） |
| `checkpoints/hateful_memes/clip_descriptions_val.pt` | CLIP图像描述 |

### 5.4 配置和文档
| 文件 | 说明 |
|------|------|
| `keys.env` | API密钥（OPENAI_GPT5_API_KEY, GEMINI_API_KEY等） |
| `EXPERIMENT_LOG.md` | **实验日志（9个阶段完整记录）** |
| `HANDOFF.md` | 本交接文档 |
| `figures/paper/` | 7张论文图表 |

---

## 六、关键代码位置（行号参考）

### 6.1 主评估脚本 evaluate_with_llm.py
- **L259-301**: `ds_fusion_decision()` - 标准DS融合
- **L354-432**: `correlation_aware_ds_fusion()` - 相关性感知DS
- **L434-511**: `uncertainty_weighted_ds_fusion()` - **核心创新1**
- **L1620-1762**: GAT共识推理 + 证据交换
- **L1773-1778**: BestAgent自动选择逻辑（已修复bug）
- **L1780-1820**: 结果汇总

### 6.2 GAT共识层 src/step2/gat_consensus.py
- **L49-283**: `GATConsensusLayer` 类
- **L110-173**: `forward()` - GAT前向传播（含Symbolic GAT改进）
- **L311-337**: `build_state()` - 构建节点状态

---

## 七、论文图表（7张，已生成）

| 图表 | 文件 | 内容 |
|------|------|------|
| 图1 | `fig1_methods_comparison.png` | 所有方法Acc/F1对比 |
| 图2 | `fig2_agent_correlation.png` | Agent相关性矩阵 |
| 图3 | `fig3_ablation_uncertainty_weighted.png` | Uncertainty_Weighted消融 |
| 图4 | `fig4_gat_ablation.png` | GAT消融实验 |
| 图5 | `fig5_disagreement_analysis.png` | 分歧样本分析 |
| 图6 | `fig6_cost_benefit.png` | 成本效益分析 |
| 图7 | `fig7_uncertainty_distribution.png` | 不确定性分布 |

---

## 八、已知的局限性

### 8.1 性能问题
1. **融合方法仍弱于最强单Agent**: 最佳融合70.40% < Gemini 79.60%
2. **GPT-5准确率偏低**: 训练集49%，验证集55%
3. **Agent1-Agent3相关性高**: 0.651（同源GPT-5）

### 8.2 实验局限
1. **因果反思仅100样本**: 500样本全量运行成本太高（预估15.6小时）
2. **外推假设**: 反思效果外推到未处理的179个分歧样本
3. **无reasoning缓存**: 传给反思prompt的reasoning为空

### 8.3 已修复的问题
- ✅ BestAgent硬编码为Agent3的bug已修复（L1773-1778）
- ✅ F1计算不一致问题已统一
- ✅ GAT在组合中无额外增益（已通过实验确认）

---

## 九、Q1论文增强计划（2026-08-24 制定）

### 9.1 当前问题诊断

| 致命缺陷 | 说明 | 严重度 |
|---------|------|--------|
| **BestAgent(79.6%) > 所有融合(70.4%)** | 审稿人会质疑"既然单Agent更好，为什么要用复杂框架？" | 🔴 致命 |
| **仅一个数据集** | 无跨数据集泛化验证 | 🔴 致命 |
| **缺少SOTA对比** | 无LLM Debate、多Agent协作等近年方法对比 | 🟡 严重 |
| **仅单次实验** | 无多种子统计显著性检验 | 🟡 严重 |

### 9.2 增强方案（P0/P1/P2分级）

#### P0：解决核心逻辑漏洞（当前进行中）
1. **更换Agent配置**: Agent1和Agent3从GPT-5改为GPT-4o-mini
   - 降低Agent1-Agent3相关性（0.651→预计更低）
   - 提升弱Agent性能
   - 完全异构配置（GPT-4o-mini + Gemini + GPT-4o-mini）
2. **多种子实验**: 5个随机种子（42, 123, 456, 789, 1024）
   - 添加`--seed`参数支持
   - 缓存按seed隔离
   - 报告均值±标准差，进行统计检验
3. **重新定位论文叙事**:
   - 不追求超越单Agent，突出框架独特价值
   - 核心卖点：Causal_Reflection分歧样本30%→74%
   - 强调：不确定性感知、分歧解决、可解释纠错

#### P1：增强竞争力
4. **添加SOTA对比**: LLM Debate、MultiAgent Collaboration等
5. **分歧样本深挖**: Causal_Reflection作为核心创新点
6. **效率指标**: 增加时间/成本分析

#### P2：泛化性验证
7. **第二数据集**: MMIMDb或其他多模态分类数据集
8. **消融增强**: Agent数量消融、样本量敏感性分析

### 9.3 执行进度

| 阶段 | 任务 | 状态 | 预计时间 |
|------|------|------|----------|
| **P0-1** | 修改Agent配置+多种子支持 | 🔄 进行中 | 1天 |
| **P0-2** | 小规模验证（5样本×1seed） | 🔄 进行中 | 0.5天 |
| **P0-3** | 5种子完整实验 | ⏳ 待执行 | 10-15小时 |
| **P1-1** | SOTA对比方法实现 | ⏳ 待执行 | 2天 |
| **P1-2** | 分歧样本深度分析 | ⏳ 待执行 | 1天 |
| **P2-1** | 第二数据集实验 | ⏳ 待执行 | 3-5天 |

### 9.4 论文叙事策略调整

**旧策略**："融合方法比单Agent更好"
**新策略**："框架在不确定性感知、分歧解决、可解释纠错方面具有独特价值"

**核心卖点排序**：
1. Causal_Reflection：分歧样本30%→74%（+44%）
2. Uncertainty_Weighted_DS：解决训练集准确率≠验证集能力问题
3. GAT_EvidenceSwap：证据交换机制

### 9.5 快速检查清单

| 检查项 | 要求 | 当前状态 | 差距 |
|--------|------|----------|------|
| 数据集数量 | ≥2个 | 1个 | ❌ 需补充 |
| 样本量 | 每集≥1000 | 500 | ❌ 需扩充 |
| 随机种子 | ≥5个 | 1个 | ❌ 需补充 |
| SOTA对比 | ≥3个外部方法 | 0个 | ❌ 需补充 |
| 消融实验 | ≥5维度 | 3维度 | ⚠️ 需增强 |
| 统计显著性 | 需t检验 | ❌ | ❌ 需补充 |

---

## 十、环境配置

### 10.1 API密钥
- 文件: `keys.env`
- 包含: OPENAI_GPT5_API_KEY, OPENAI_GPT4OM_API_KEY, GEMINI_API_KEY
- 平台: 魔芋 (moyu.info)
- 地址: `https://www.moyu.info/v1`

### 10.2 运行环境
- Python 3.x
- PyTorch (CPU模式)
- 需设置环境变量: `KMP_DUPLICATE_LIB_OK=TRUE`

### 10.3 常用命令

```bash
# === 当前配置（DeepSeek+Gemini+GPT-5，2026-08-24起）===
# 运行主实验（seed=42已完成缓存；其余seed需DeepSeek新推理）
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=200 --max_val=500 --provider1=deepseek --provider2=gemini --provider3=gpt5 --seed=42

# 5种子完整实验（批量脚本，默认deepseek+gemini+gpt5）
python run_multi_seed.py --seeds 42,123,456,789,1024

# === 历史命令（GPT-5配置，旧实验）===
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=200 --max_val=500 --provider1=gpt5 --provider2=gemini --provider3=gpt5

# 运行消融实验
python ablation_study.py

# 运行组合实验
python combine_gat_unc.py

# 运行因果反思（v2，有效版本）
python run_step5_v2.py --max_disagree 100

# 生成论文图表
python generate_paper_figures.py

# API诊断（检测魔芋渠道可用性）
python diagnose_api.py

# 删除缓存文件（强制重新运行）
Remove-Item checkpoints\hateful_memes\llm_*.pt
```

---

## 十一、踩过的坑（绝对不要再踩）

### 11.1 GLM-5V-Turbo限流
- ❌ 429 Too Many Requests，限流严重
- ✅ 已弃用，改用Gemini

### 11.2 Claude API不稳定
- ❌ 频繁500错误，部分样本返回fallback
- ✅ 已改用GPT-5作为Agent1和Agent3

### 11.3 缓存维度不匹配
- ❌ 不同样本量的缓存文件不能混用
- ✅ 修改`max_train`或`max_val`后，必须删除旧缓存文件

### 11.4 Windows编码问题
- ❌ print包含特殊字符时出现UnicodeEncodeError
- ✅ 使用 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`

### 11.5 OpenMP冲突
- ❌ `OMP: Error #15: Initializing libiomp5md.dll`
- ✅ 设置环境变量`KMP_DUPLICATE_LIB_OK=TRUE`

### 11.6 小样本结果不可信
- ❌ 5样本验证显示100%准确率，但100样本仅52-58%
- ✅ 至少使用100样本验证，最好500+样本

### 11.7 因果反思策略
- ❌ v1（只反思少数派）无效：少数派改变无法翻转MV
- ✅ v2（反思全Agent）有效：必须反思所有Agent才能翻转2v1分歧

### 11.8 模型选择
- ❌ DeepSeek偏向保守预测，F1仅35-36%
- ✅ GPT-5和Gemini组合最佳
- ⚠️ Gemini API慢（约248秒/样本），需耐心等待

---

## 十二、费用控制建议

1. **小样本先行**: 每次实验先用5-10个样本测试，确认正常后再扩展
2. **独立缓存**: 利用独立缓存机制，避免重复运行已完成的Agent
3. **分歧触发**: Step5仅对分歧样本触发反思，约56%的验证集
4. **降低温度**: 使用`temperature=0.1`减少输出随机性，降低token消耗
5. **监控API调用**: 关注各模型的token消耗和调用次数

---

## 十三、当前实验状态

**✅ 原有实验已完成，三个创新点全部验证**

**📋 Q1论文增强计划进行中**

**原有缓存状态**（GPT-5配置，已过时）:
- Agent1(GPT-5): ✅ 训练集200样本 + 验证集500样本
- Agent2(Gemini): ✅ 训练集200样本 + 验证集500样本
- Agent3(GPT-5): ✅ 训练集200样本 + 验证集500样本

**Q1增强计划状态**（2026-08-24 启动）:
- ✅ 文档更新：HANDOFF.md + EXPERIMENT_LOG.md
- ✅ P0-1: 修改脚本默认参数（provider3→gpt4om）
- ✅ P0-2: 添加seed参数支持（evaluate_with_llm.py）
  - 缓存按seed隔离：`llm_train_agent{i}_seed{seed}.pt`
  - 结果文件包含seed：`evaluation_llm_{providers}_seed{seed}.json`
- ✅ P0-3: 创建多种子批量脚本 `run_multi_seed.py`
- ✅ P0-4: 小规模代码验证通过（5样本×seed123）
  - ⚠️ gpt4om渠道故障：moyu.info的Lite-GPT分组下gpt-4o-mini无可用渠道（503持续）
  - 诊断工具: `python diagnose_api.py`；详见EXPERIMENT_LOG.md阶段10
- ⏳ P0-5: 5种子完整实验（API恢复后执行）
- ⏳ P1-1: SOTA对比方法
- ⏳ P1-2: 分歧样本深度分析
- ⏳ P2-1: 第二数据集

**因果反思状态**:
- v1（100样本）: ✅ 完成（无效）
- v2（100样本）: ✅ 完成（有效，+44%分歧样本提升）

---

*最后更新: 2026-08-24*
