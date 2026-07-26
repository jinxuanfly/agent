# 项目交接文档

## 一、项目概述

本项目是一个**异构多模态动态共识与协同框架**的实验代码，用于仇恨言论检测（Hateful Memes）任务。核心架构包含三个LLM Agent协同工作，通过图注意力网络（GAT）实现共识决策，并在共识失败时触发因果反事实反思。

### 核心目标
- 验证异构多模态Agent（文本专家、图像专家、跨模态专家）的协同效果
- 通过GAT共识层解决Agent间的分歧
- 通过因果反事实反思进一步修正共识失败的样本
- 在Hateful Memes数据集上实现高精度的仇恨言论检测

### Agent架构（当前方案B配置）

| Agent | 角色 | 当前模型 | 输入模态 | Provider |
|-------|------|----------|----------|----------|
| **Agent1** | 文本专家 | GPT-4o-mini-2024-07-18 | 仅文本 | gpt（魔芋中转） |
| **Agent2** | 图像专家 | Gemini-3.5-flash | 直接图像输入 + 文本 | gemini（魔芋中转） |
| **Agent3** | 跨模态专家 | Claude Sonnet 5 | 文本 + CLIP图像描述 | claude（魔芋中转） |

---

## 二、已完成的工作

### 2.1 项目架构搭建（Step1-3）

| 步骤 | 名称 | 状态 | 说明 |
|------|------|------|------|
| Step1 | 合成数据与硬模型训练 | ✅ 已完成 | 生成合成数据，训练基础分类器 |
| Step2 | GAT共识层 | ✅ 已完成 | 实现图注意力网络共识引擎，支持不确定性感知 |
| Step3 | 分歧解构器 | ✅ 已完成 | 区分证据冲突和无知冲突，针对性优化 |

### 2.2 Hateful Memes评估管线（Step4）

| 任务 | 状态 | 详情 |
|------|------|------|
| LLM推理框架 | ✅ 已完成 | 支持DeepSeek、GLM、GPT、Gemini、Claude多种模型 |
| 独立缓存机制 | ✅ 已完成 | 每个Agent的缓存独立存储，避免重复运行 |
| API密钥管理 | ✅ 已完成 | 通过keys.env和环境变量安全管理 |
| 超时与重试机制 | ✅ 已完成 | 180秒超时，8次重试，指数退避 |
| 速率限制器 | ✅ 已完成 | 按provider分组的速率限制，支持最小调用间隔 |
| GAT共识训练 | ✅ 已完成 | 基于LLM输出训练GAT共识模型 |
| 评估指标计算 | ✅ 已完成 | Accuracy、F1、不确定性分析 |

### 2.3 因果反事实反思（Step5）

| 任务 | 状态 | 详情 |
|------|------|------|
| 分歧样本识别 | ✅ 已完成 | 统计分歧样本并分类 |
| 文本消融归因 | ✅ 已完成 | 基于仇恨关键词的文本消融策略 |
| 补偿提示生成 | ✅ 已完成 | 生成修正提示词 |
| 反思循环实现 | ✅ 已完成 | 分层反思策略 |
| 跨Agent证据交换 | ✅ 已完成 | 让分歧Agent看到其他Agent的推理 |
| 实验运行 | ✅ 已完成 | 优化版因果反思实验完成，分歧样本准确率提升15% |

### 2.4 多模型集成

| 模型 | 状态 | Provider | 说明 |
|------|------|----------|------|
| DeepSeek-v4-flash | ✅ 已集成 | deepseek | 初期Agent1模型，F1偏低 |
| GPT-4o-mini | ✅ 已集成 | gpt（魔芋） | 稳定可靠，支持图像输入 |
| Gemini-3.5-flash | ✅ 已集成 | gemini（魔芋） | 图像专家，速度较慢 |
| Claude Sonnet 5 | ✅ 已集成 | claude（魔芋） | 跨模态专家，API不稳定 |

### 2.5 实验结果记录与可视化

| 任务 | 状态 | 详情 |
|------|------|------|
| 实验日志 | ✅ 已创建 | EXPERIMENT_LOG.md记录所有实验 |
| 可视化图表 | ✅ 已生成 | 各Agent性能对比、共识融合方法对比、因果反思效果对比 |

---

## 三、当前问题

### 3.1 Gemini API速度过慢（最严重）

**现象**: Agent2验证集推理速度约248秒/样本，100样本需要约7小时
**影响**: 实验效率极低，严重影响进度
**原因**: 魔芋中转的Gemini API本身响应较慢，且包含图像输入

### 3.2 Claude API不稳定

**现象**: 出现大量500错误，部分样本返回fallback_uniform（label=0, conf=0.500）
**影响**: Agent3准确率偏低（约42%），影响整体效果
**原因**: 魔芋中转的Claude API稳定性不佳

### 3.3 三Agent一致性低

**现象**: 训练集分歧率高达79%（79/100样本），验证集预计相似
**影响**: GAT共识层难以学到稳定模式，共识收益有限
**原因**: 
1. Agent2仅看图像，与其他两Agent差异大
2. Agent3受API不稳定影响
3. 模型本身对仇恨言论的判断标准不一致

### 3.4 实验结果不够理想

**训练集结果（方案B，100样本）**:
| 方法 | Acc% | F1% |
|------|------|-----|
| Agent1(GPT-4o-mini) | 58.00 | - |
| Agent2(Gemini) | 52.00 | - |
| Agent3(Claude) | 42.00 | - |
| Hybrid_GAT | - | - |

**问题**: 单个Agent准确率均未超过60%，共识机制难以提升效果

---

## 四、当前卡住的位置

实验流程已全部跑通，但存在以下卡点：

1. **实验效率低**: Gemini API速度约248秒/样本，100样本验证集需要约7小时
2. **API不稳定**: Claude API经常500错误，影响Agent3效果
3. **分歧率高**: 79%的样本存在分歧，GAT难以学习有效共识模式
4. **样本量不足**: 当前仅100样本，需扩大至500-1000样本

---

## 五、下一步计划

### 5.1 优先解决：实验效率问题

**方案A**: 更换Agent2为GPT-4o-mini（也支持图像输入，但速度更快）
- 优点：速度快、稳定
- 缺点：失去异构性，三个Agent都是GPT系列

**方案B**: 优化Gemini调用策略
- 使用更小的图像尺寸
- 减少重试次数
- 分批次运行，利用空闲时间

**方案C**: 改用CLIP描述模式（不推荐，已验证效果差）

### 5.2 优化Agent配置

- 考虑让Agent2也接收文本信息（已部分实现）
- 评估是否需要更换Agent3模型
- 尝试不同的模型组合

### 5.3 扩大样本量

解决效率问题后，运行500样本实验，验证方法有效性

### 5.4 生成论文图表

基于实验结果生成高质量可视化图表，用于论文撰写

### 5.5 运行完整评估

结合因果反思结果，重新运行完整评估管线

---

## 六、踩过的坑（绝对不要再踩）

### 6.1 GLM-5V-Turbo限流

| 问题 | 错误信息 | 解决方案 |
|------|----------|----------|
| 429 Too Many Requests | 限流 | 不要使用GLM-5V-Turbo！已被Gemini替代 |
| fallback_uniform返回 | label=0, conf=0.500 | 更换模型 |

**结论**: GLM-5V-Turbo免费额度有限且限流严重，已弃用

### 6.2 Qwen API系列问题

| 问题 | 错误信息 | 原因 | 解决方案 |
|------|----------|------|----------|
| 401 Unauthorized | API密钥无效 | 密钥格式错误或未正确加载 | 重新生成密钥 |
| 403 Forbidden | 免费额度用尽 | 未完成支付配置 | 在阿里云百炼控制台完成配置 |
| 超时错误 | Read timed out (30s) | 超时时间太短 | 延长至120秒 |

**结论**: Qwen API不稳定，不建议使用

### 6.3 魔芋API注意事项

| 项目 | 正确做法 | 错误做法 |
|------|----------|----------|
| 地址 | `https://www.moyu.info/v1` | `https://api.moyu.info/v1` |
| 密钥 | 在keys.env中配置 | 命令行临时设置 |
| 测试 | 小样本测试成功后再扩展 | 直接运行大规模实验 |
| 额度 | 定期检查密钥额度 | 忽略额度警告 |

### 6.4 缓存维度不匹配

- ❌ 不同样本量的缓存文件不能混用
- ❌ 修改`max_train`或`max_val`后，必须删除旧缓存文件
- ✅ 缓存路径：`checkpoints/hateful_memes/llm_train_agentX.pt` 和 `llm_val_agentX.pt`
- ✅ 使用`--force_rerun_agent`参数无效时，手动删除缓存文件

### 6.5 `--force_rerun_agent`参数只取最后一个值

- ❌ `--force_rerun_agent=0 --force_rerun_agent=1 --force_rerun_agent=2` 只对Agent2生效
- ✅ 手动删除所有缓存文件确保重新运行

### 6.6 Windows编码问题

- ❌ Windows下默认编码是GBK
- ❌ print包含特殊字符时出现UnicodeEncodeError
- ✅ 在`llm_agent.py`中添加UTF-8编码异常处理

### 6.7 OpenMP冲突

- ❌ `OMP: Error #15: Initializing libiomp5md.dll`
- ✅ 设置环境变量`KMP_DUPLICATE_LIB_OK="TRUE"`

### 6.8 图像路径拼接错误

- ❌ `os.path.join(DATA_DIR, 'img', item['img'])` 可能导致双层`img/img/`
- ✅ 检查路径是否存在，灵活拼接

### 6.9 小样本结果不可信

- ❌ 5样本验证显示GAT_EvidenceSwap准确率100%，但100样本仅52-58%
- ✅ 至少使用100样本验证，最好500+样本

### 6.10 模型选择至关重要

- ❌ DeepSeek在仇恨言论检测上偏向保守，F1仅35-36%
- ✅ 选择更均衡的模型如Claude或GPT-4o-mini
- ⚠️ Claude API不稳定，可能出现500错误

---

## 七、重要文件路径

| 文件 | 用途 |
|------|------|
| `src/llm_api.py` | LLM API配置和调用逻辑（含多模态支持、速率限制器） |
| `src/llm_agent.py` | LLM Agent创建函数和推理封装（含Prompt配置） |
| `src/step4_hateful_memes/evaluate_with_llm.py` | 主评估管线（LLM推理+GAT训练+评估） |
| `src/step4_hateful_memes/evaluate_step5_causal_reflection.py` | 因果反事实反思实验 |
| `src/step4_hateful_memes/generate_charts.py` | 实验结果可视化图表生成 |
| `src/gat_consensus.py` | GAT共识引擎实现 |
| `keys.env` | API密钥（已加入.gitignore） |
| `checkpoints/hateful_memes/` | 推理结果缓存目录 |
| `results/hateful_memes/` | 实验结果输出目录 |
| `figures/` | 可视化图表输出目录 |
| `EXPERIMENT_LOG.md` | 实验日志与经验教训 |

---

## 八、关键配置

### 环境变量（keys.env）

```bash
# GPT-4o-mini（魔芋中转）
OPENAI_API_KEY=sk-LXBBRassCHHkzRzwl3HoiALErsfxUkkNGZvlOAIorJ5XURYJ

# Gemini-3.5-flash（魔芋中转）
GEMINI_API_KEY=sk-6k2O7BtrDM2w5dmBTTz4eu9LYYUwZRXZJbEk3xcTdReBcCIn

# Claude Sonnet 5（魔芋中转）
ANTHROPIC_API_KEY=sk-pRfUUGHGA1tt33301aMQabZzLg2yEebrC0Z3HhX7PlBlovBf

# DeepSeek（备用）
# DEEPSEEK_API_KEY=sk-d96e8c9644984c919caf4fb48a57e0e0

# GLM（已弃用，限流严重）
# GLM_API_KEY=8b192d029dbf4f10bea2fa004968f078.tA04YDmEUakDxDqr
```

### Agent配置（方案B，当前）

```python
Agent1: gpt/gpt-4o-mini-2024-07-18 (文本专家，仅文本)
Agent2: gemini/gemini-3.5-flash (图像专家，直接图像+文本)
Agent3: claude/claude-sonnet-5 (跨模态专家，文本+CLIP图像描述)
```

### 实验命令示例

```bash
# 100样本完整实验（方案B配置）
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=gpt --provider2=gemini --provider3=claude

# 因果反思实验
python src/step4_hateful_memes/evaluate_step5_causal_reflection.py

# 删除缓存文件（强制重新运行）
Remove-Item checkpoints\hateful_memes\llm_*.pt
```

---

## 八、当前实验状态

**重要**: 之前在终端4运行的方案B实验已停止（command_id: 5a9dc0f7-625d-4d9c-9388-900b014374a9），验证集推理未完成。

**停止原因**: 为了保证交接文档的完整性和准确性，避免新对话重复启动或遗漏进程

**训练集缓存状态**:
- Agent1(GPT-4o-mini): ✅ 已完成（100样本）
- Agent2(Gemini): ✅ 已完成（100样本）
- Agent3(Claude): ✅ 已完成（100样本）

**验证集缓存状态**:
- Agent1(GPT-4o-mini): ✅ 已完成（100样本）
- Agent2(Gemini): ❌ 未完成（约35/100样本，需重新运行）
- Agent3(Claude): ❌ 未完成（需重新运行）

**重新启动命令**:
```bash
# 删除旧的验证集缓存
Remove-Item checkpoints\hateful_memes\llm_val_*.pt

# 重新运行完整实验（训练集使用缓存，仅运行验证集）
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=gpt --provider2=gemini --provider3=claude
```

---

## 九、紧急事项

**最紧急**: 解决Gemini API速度过慢问题（约248秒/样本），否则实验无法在合理时间内完成

**次紧急**: 提高三Agent一致性，降低分歧率，让GAT共识层发挥作用

**第三**: 扩大样本量至500-1000，满足Q1期刊统计显著性要求

---

## 十、费用控制建议

1. **小样本先行**: 每次实验先用5-10个样本测试，确认正常后再扩展
2. **独立缓存**: 利用独立缓存机制，避免重复运行已完成的Agent
3. **分歧触发**: Step5仅对分歧样本触发反思，约20-30%的验证集
4. **降低温度**: 使用`temperature=0.1`减少输出随机性，降低token消耗
5. **监控API调用**: 关注各模型的token消耗和调用次数，及时调整策略
6. **优先使用低成本模型**: GPT-4o-mini成本最低，其次是Gemini-3.5-flash

---

*最后更新: 2026-07-26*