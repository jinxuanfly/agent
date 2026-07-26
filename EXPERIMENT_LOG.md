# 实验日志与经验教训

## 项目概述
本项目实现了一个**异构多模态动态共识与协同框架**，用于仇恨言论检测（Hateful Memes）任务。框架包含三个LLM Agent协同工作，通过图注意力网络（GAT）实现共识决策，并在共识失败时触发因果反事实反思。

---

## 实验时间线

### 阶段1：GLM-5V-Turbo作为Agent2（2026-07-17）

**配置详情**:
| Agent | 模型 | 输入模态 | Provider |
|-------|------|----------|----------|
| Agent1 | DeepSeek-v4-flash | 仅文本 | deepseek |
| Agent2 | GLM-5V-Turbo | 直接图像输入 | glm |
| Agent3 | GPT-4o-mini | 文本+CLIP图像描述 | gpt（魔芋） |

**运行命令**:
```bash
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=200 --max_val=200 --provider1=deepseek --provider2=glm --provider3=gpt
```

**运行过程**:
- 训练集推理：Agent1和Agent3正常完成，Agent2大量返回fallback
- GAT共识训练：基于LLM输出训练
- 验证集推理：Agent2持续限流

**中断原因**: GLM-5V-Turbo严重限流（429 Too Many Requests）

**运行时间**: 约2小时（仅完成部分推理）

**结果**:
| 方法 | Acc% | F1% |
|------|------|-----|
| Agent1(deepseek) | 60.00 | 42.03 |
| Agent2(glm) | 53.50 | 29.01（大量fallback） |
| Agent3(gpt) | 65.50 | 66.99 |
| Hybrid_GAT | 64.50 | 58.48 |

**分析**:
- GLM-5V-Turbo免费额度有限，连续调用很快触发429限流
- 限流后返回`fallback_uniform`（label=0, conf=0.500），导致Agent2结果基本无效
- Agent3(GPT-4o-mini)表现最好，但共识机制未能超越最佳单Agent

---

### 阶段2：Gemini-3.5-flash作为Agent2（2026-07-18）

**配置详情**:
| Agent | 模型 | 输入模态 | Provider |
|-------|------|----------|----------|
| Agent1 | DeepSeek-v4-flash | 仅文本 | deepseek |
| Agent2 | Gemini-3.5-flash | 直接图像输入 | gemini（魔芋） |
| Agent3 | GPT-4o-mini | 文本+CLIP图像描述 | gpt（魔芋） |

**运行命令**:
```bash
# 5样本验证
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=5 --max_val=5 --provider1=deepseek --provider2=gemini --provider3=gpt

# 100样本测试
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=deepseek --provider2=gemini --provider3=gpt
```

**运行过程**:
- 5样本验证：成功运行，GAT_EvidenceSwap准确率100%（过拟合！）
- 100样本测试：三个Agent全部完成推理

**运行时间**:
- 5样本验证：约15分钟
- 100样本测试：约4小时

**结果**:
| 方法 | Acc% | F1% |
|------|------|-----|
| Agent1(deepseek) | 53 | 35.62 |
| Agent2(gemini) | 54 | 58.93 |
| Agent3(gpt) | 51 | 66.67 |
| GAT_EvidenceSwap | 52 | 38.46 |

**分析**:
- 5样本验证显示100%准确率是过拟合，100样本才是真实效果
- Agent1(DeepSeek)F1仅35.62%，严重偏向预测非仇恨言论，漏判严重
- Agent2(Gemini)仅看图像，准确率54%，但F1优于Agent1
- 分歧率高达78-82%，三Agent一致性低

---

### 阶段3：Agent1 Prompt优化（2026-07-19）

**配置详情**: 同阶段2，仅修改Agent1的Prompt

**改进内容**:
- 重写text_focused prompt，添加详细的仇恨言论类型定义
- 添加判断原则和推理步骤要求
- 增加对仇恨样本的判断权重

**运行命令**:
```bash
# 20样本验证
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=20 --max_val=20 --provider1=deepseek --provider2=gemini --provider3=gpt

# 100样本测试
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=deepseek --provider2=gemini --provider3=gpt
```

**运行过程**:
- 20样本验证：Agent1 F1从35.62%提升到50%
- 100样本测试：完整运行

**运行时间**: 约4小时

**结果**:
| 方法 | Acc% | F1% |
|------|------|-----|
| Agent1(deepseek) | 55 | 36.62 |
| Agent2(gemini) | 53 | 59.83 |
| Agent3(gpt) | 50 | 65.75 |
| GAT_EvidenceSwap | 58 | 44.74 |

**分析**:
- Prompt优化效果有限，F1仅提升1%（从35.62%到36.62%）
- 根本问题是DeepSeek模型本身偏向保守预测
- 需要更换Agent1模型

---

### 阶段4：Claude Sonnet 5作为Agent1（2026-07-20）

**配置详情**:
| Agent | 模型 | 输入模态 | Provider |
|-------|------|----------|----------|
| Agent1 | Claude Sonnet 5 | 仅文本 | claude（魔芋） |
| Agent2 | Gemini-3.5-flash | 直接图像输入 | gemini（魔芋） |
| Agent3 | GPT-4o-mini | 文本+CLIP图像描述 | gpt（魔芋） |

**运行命令**:
```bash
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=claude --provider2=gemini --provider3=gpt
```

**运行过程**:
- 训练集推理：Claude出现多次500错误，部分样本返回fallback
- GAT共识训练：完成
- 验证集推理：完成

**运行时间**: 约5小时（Claude API较慢）

**结果**:
| 方法 | Acc% | F1% | Avg_u |
|------|------|-----|-------|
| Agent1(claude) | 58 | 48.78 | 0.1586 |
| Agent2(gemini) | 42 | 50.00 | 0.0871 |
| Agent3(gpt) | 51 | 66.67 | 0.1282 |
| MajorityVoting | 49 | 58.54 | - |
| WeightedAvg | 51 | 60.80 | - |
| DS_Fusion | 51 | 60.80 | - |
| GAT_Fusion | 52 | 62.50 | - |
| GAT_EvidenceSwap | 52 | 53.85 | - |

**分歧统计**:
- 分歧样本：63个（63%）
- 简单分歧(2v1)：-
- 复杂分歧(1v1v1)：-

**分析**:
- Agent1(Claude) F1从36.62%提升至48.78%（+33%），效果显著
- Agent2(Gemini)准确率下降至42%，原因是仅看图像模式效果有限
- 分歧率从78-82%下降到63%，但仍超过半数
- Claude API不稳定，出现大量500错误

---

### 阶段5：因果反思实验（优化版，2026-07-21）

**配置详情**: 同阶段4

**改进内容**:
- 分层反思策略：简单分歧(2v1)直接多数投票，复杂分歧(1v1v1)触发完整反思
- 跨Agent证据交换：让分歧Agent看到其他Agent的推理和预测结果
- 基于仇恨关键词的文本消融策略

**运行命令**:
```bash
python src/step4_hateful_memes/evaluate_step5_causal_reflection.py --max_val=100 --provider1=claude --provider2=gemini --provider3=gpt
```

**运行过程**:
- 识别60个分歧样本
- 分层处理：简单分歧15个(25%)，复杂分歧45个(75%)
- 复杂分歧触发完整反思循环

**运行时间**: 约2小时33分钟

**结果**:
| 指标 | 反思前 | 反思后 | 提升 |
|------|--------|--------|------|
| 分歧样本准确率 | 48.33% | 63.33% | +15% |
| F1分数 | - | 71.79% | - |
| 收敛率 | - | 100% | - |
| 平均反思轮数 | - | 1.5 | - |
| 额外API调用 | - | 300次 | - |

**分析**:
- 优化版因果反思效果显著，分歧样本准确率提升15%
- 分层策略有效，25%的简单分歧无需完整反思，节省成本
- 跨Agent证据交换机制提高了收敛率（100%）
- 但额外300次API调用成本较高

---

### 阶段6：方案A - GPT-4o-mini作为Agent2（2026-07-22）

**配置详情**:
| Agent | 模型 | 输入模态 | Provider |
|-------|------|----------|----------|
| Agent1 | GPT-4o-mini | 仅文本 | gpt（魔芋） |
| Agent2 | GPT-4o-mini | 直接图像输入 | gpt（魔芋） |
| Agent3 | GPT-4o-mini | 文本+CLIP图像描述 | gpt（魔芋） |

**运行命令**:
```bash
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=gpt --provider2=gpt --provider3=gpt
```

**运行过程**:
- 全部使用GPT-4o-mini，速度快，API稳定
- 但用户反馈偏离论文"异构多模态框架"设计初衷

**运行时间**: 约2小时

**结果**:
- 各Agent性能稳定，但缺乏异构性
- 用户要求恢复异构配置

**分析**:
- GPT-4o-mini稳定可靠，但三个Agent都用同一模型失去了框架的异构性优势
- 需要保持不同模型的异构配置

---

### 阶段7：方案B - 异构配置优化（2026-07-23至2026-07-26）

**配置详情**:
| Agent | 模型 | 输入模态 | Provider |
|-------|------|----------|----------|
| Agent1 | GPT-4o-mini | 仅文本 | gpt（魔芋） |
| Agent2 | Gemini-3.5-flash | 直接图像输入+文本 | gemini（魔芋） |
| Agent3 | Claude Sonnet 5 | 文本+CLIP图像描述 | claude（魔芋） |

**改进内容**:
- Agent2增加文本输入，不再仅看图像
- 修改Agent2的Prompt，使其结合文本信息进行综合判断

**运行命令**:
```bash
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=gpt --provider2=gemini --provider3=claude
```

**运行过程**:
- 训练集推理：使用缓存，快速完成
- GAT共识训练：完成
- 验证集推理：Agent1完成，Agent2完成约35/100样本
- **第一次中断**: Gemini API额度用尽（401 TokenStatusExhausted）
- 用户充值后重新运行，验证集推理再次进行中
- **第二次中断**: 实验进程被手动停止，等待交接完成后重新启动

**运行时间**: 
- 已耗时：约3小时（Agent2完成约35/100样本）
- **当前状态**: ⚠️ **验证集推理未完成**，需在新对话中重新启动

**重新启动步骤**:
1. 删除旧的验证集缓存：`Remove-Item checkpoints\hateful_memes\llm_val_*.pt`
2. 重新运行：`python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=gpt --provider2=gemini --provider3=claude`

**训练集结果（缓存）**:
| 方法 | Acc% | Avg_u |
|------|------|-------|
| Agent1(gpt) | 58.00 | 0.1391 |
| Agent2(gemini) | 52.00 | 0.0901 |
| Agent3(claude) | 42.00 | 0.2059 |

**分歧统计（训练集）**:
- DS正确样本：50/100
- 有分歧样本：79/100（79%）
- GAT训练样本(分歧)：79
- GAT训练样本(全部)：100

**分析**:
- Agent1(GPT-4o-mini)表现稳定，准确率58%
- Agent2(Gemini)增加文本输入后准确率从42%提升至52%，有改善
- Agent3(Claude)受API不稳定影响，准确率仅42%
- 分歧率高达79%，三Agent一致性仍需提升
- Gemini API速度过慢（约248秒/样本），严重影响实验效率
- Claude API不稳定，出现500错误

---

## 关键数据对比

### 不同Agent1配置对比

| 模型 | 准确率 | F1分数 | 稳定性 | 成本 |
|------|--------|--------|--------|------|
| DeepSeek | 53-55% | 35.62-36.62% | 稳定 | 低 |
| Claude Sonnet 5 | 58% | 48.78% | 不稳定(500错误) | 中 |
| GPT-4o-mini | 58% | - | 稳定 | 低 |

### 不同Agent2配置对比

| 模型 | 准确率 | 速度 | 稳定性 | 成本 |
|------|--------|------|--------|------|
| GLM-5V-Turbo | 53.5%(大量fallback) | 中等 | 差(限流) | 低 |
| Gemini-3.5-flash(仅图像) | 42% | 慢(248秒/样本) | 中等 | 中 |
| Gemini-3.5-flash(图像+文本) | 52% | 慢(248秒/样本) | 中等 | 中 |
| GPT-4o-mini(图像) | - | 快 | 稳定 | 中 |

### 各方法性能对比（100样本，Claude+Gemini+GPT）

| 方法 | 准确率 | F1分数 |
|------|--------|--------|
| Agent1(claude) | 58.00% | 48.78% |
| Agent2(gemini) | 42.00% | 50.00% |
| Agent3(gpt) | 51.00% | 66.67% |
| MajorityVoting | 49.00% | 58.54% |
| WeightedAvg | 51.00% | 60.80% |
| DS_Fusion | 51.00% | 60.80% |
| GAT_Fusion | 52.00% | 62.50% |
| GAT_EvidenceSwap | 52.00% | 53.85% |

### 因果反思效果对比

| 指标 | 反思前 | 反思后 | 提升 |
|------|--------|--------|------|
| 分歧样本准确率 | 48.33% | 63.33% | +15% |
| F1分数 | - | 71.79% | - |
| 收敛率 | - | 100% | - |

---

## 经验教训

### 1. 小样本结果不可信
- **教训**: 5样本验证显示GAT_EvidenceSwap准确率100%，但扩大到100样本后仅52-58%
- **原因**: 小样本下随机波动大，模型可能恰好记住了样本特征
- **建议**: 至少使用100样本验证，最好500+样本

### 2. Agent1模型选择至关重要
- **教训**: DeepSeek的F1持续偏低（35.62-36.62%），更换为Claude Sonnet 5后F1提升至48.78%（+33%）
- **原因**: DeepSeek模型本身在仇恨言论检测任务上偏向保守预测，大量漏判仇恨样本
- **建议**: 选择对仇恨言论检测任务更均衡的模型如GPT-4o-mini或Claude

### 3. Agent2仅看图像效果有限
- **教训**: Agent2仅接收图像描述，准确率仅42%，低于同时看文本的Agent1和Agent3
- **原因**: 仇恨言论的关键信息常常在文本中，仅看图像容易误判
- **建议**: 让Agent2也接收文本信息（已实现），或更换更擅长跨模态理解的模型

### 4. 分歧率下降但仍需改进
- **教训**: 使用Claude后分歧率从78-82%下降到63%，但仍有超过半数样本存在分歧
- **原因**: Agent2仅看图像的模式与其他两个Agent差异较大
- **建议**: 改进Agent配置，减少模态偏见

### 5. 分层因果反思效果好
- **教训**: 优化版因果反思将分歧样本准确率从48.33%提升到63.33%（+15%）
- **改进**: 分层反思策略（简单分歧2v1直接多数投票，复杂分歧1v1v1触发完整反思）
- **成本**: 300次额外API调用，耗时约2.5小时
- **建议**: 当前配置已经较好，可考虑扩大样本量验证

### 6. API调用成本控制
- **教训**: 每次100样本完整实验约需600次API调用
- **成本估算**:
  - Agent1: 200次（训练+验证）
  - Agent2: 200次（训练+验证）
  - Agent3: 200次（训练+验证）
  - 因果反思: 额外300次（60个分歧样本）
- **建议**: 
  - 使用缓存机制避免重复调用
  - 先在小样本上验证prompt效果，再扩大规模
  - 定期检查API密钥额度

### 7. 魔芋API额度管理
- **教训**: Gemini API密钥额度用尽导致实验中断（401 TokenStatusExhausted）
- **原因**: 密钥本身的额度与账户充值是分离的，充值后可能需要重新生成密钥
- **建议**: 定期检查密钥额度，充值后测试API连接

### 8. 异构性的重要性
- **教训**: 方案A（全部GPT-4o-mini）虽然稳定，但偏离论文"异构多模态框架"设计初衷
- **原因**: 三个Agent使用同一模型失去了框架的异构性优势
- **建议**: 保持不同模型的异构配置，即使牺牲部分稳定性

---

## 当前瓶颈与下一步方向

### 当前瓶颈
1. **Gemini API速度过慢**（约248秒/样本）：100样本验证集需要约7小时
2. **Claude API不稳定**：大量500错误，部分样本返回fallback，影响准确率
3. **分歧率仍较高**（79%）：三个Agent一致性仍需改进
4. **样本量不足**（100样本）：需扩大至500-1000样本以提高统计显著性
5. **API成本较高**：大规模实验需要控制费用

### 下一步方向
1. **优化Agent2配置**：考虑更换为GPT-4o-mini（速度更快）或优化Gemini调用策略
2. **优化Agent3配置**：考虑更换为更稳定的模型
3. **扩大训练样本**：运行500样本实验验证方法有效性
4. **生成论文图表**：创建美观的可视化图表用于论文撰写
5. **运行完整评估**：结合因果反思结果，重新运行完整评估管线

---

## 文件位置与配置说明

### 关键文件
- `src/step4_hateful_memes/evaluate_with_llm.py` - 主评估脚本
- `src/step4_hateful_memes/evaluate_step5_causal_reflection.py` - 因果反思脚本
- `src/llm_agent.py` - Agent定义和prompt配置
- `src/llm_api.py` - LLM API调用层（含速率限制器）
- `keys.env` - API密钥配置
- `HANDOFF.md` - 项目交接文档

### 运行命令示例
```bash
# 100样本完整实验（方案B配置）
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=100 --max_val=100 --provider1=gpt --provider2=gemini --provider3=claude

# 因果反思实验
python src/step4_hateful_memes/evaluate_step5_causal_reflection.py

# 删除缓存文件（强制重新运行）
Remove-Item checkpoints\hateful_memes\llm_*.pt
```

---

## 注意事项

1. **API密钥**: 确保`keys.env`文件中配置了正确的API密钥
2. **图像数据**: 确保`data/Hateful_Memes/data/img/`目录下有图像文件
3. **缓存机制**: 脚本会自动缓存LLM输出，修改配置后需手动删除缓存文件
4. **环境变量**: 需要设置`KMP_DUPLICATE_LIB_OK="TRUE"`解决OpenMP冲突
5. **输出文件**: 结果保存在`results/hateful_memes/`目录，可视化图表保存在`figures/`目录
6. **API额度**: 定期检查各API密钥额度，避免实验中断

---

*最后更新: 2026-07-26*