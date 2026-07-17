# 项目交接文档

## 一、项目概述

本项目是一个**异构多模态动态共识与协同框架**的实验代码，用于仇恨言论检测（Hateful Memes）任务。核心架构包含三个LLM Agent协同工作，通过图注意力网络（GAT）实现共识决策，并在共识失败时触发因果反事实反思。

### 核心目标
- 验证异构多模态Agent（文本专家、图像专家、跨模态专家）的协同效果
- 通过GAT共识层解决Agent间的分歧
- 通过因果反事实反思进一步修正共识失败的样本
- 在Hateful Memes数据集上实现高精度的仇恨言论检测

### Agent架构

| Agent | 角色 | 当前模型 | 输入模态 |
|-------|------|----------|----------|
| **Agent1** | 文本专家 | DeepSeek-v4-flash | 仅文本 |
| **Agent2** | 图像专家 | GLM-5V-Turbo | 直接图像输入 |
| **Agent3** | 跨模态专家 | GPT-4o-mini（魔芋中转） | 文本+CLIP图像描述 |

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
| LLM推理框架 | ✅ 已完成 | 支持DeepSeek、GLM、GPT-4o-mini多种模型 |
| 独立缓存机制 | ✅ 已完成 | 每个Agent的缓存独立存储，避免重复运行 |
| API密钥管理 | ✅ 已完成 | 通过keys.env和环境变量安全管理 |
| 超时与重试机制 | ✅ 已完成 | 120秒超时，5次重试 |
| GAT共识训练 | ✅ 已完成 | 基于LLM输出训练GAT共识模型 |
| 评估指标计算 | ✅ 已完成 | Accuracy、F1、不确定性分析 |

### 2.3 因果反事实反思（Step5）

| 任务 | 状态 | 详情 |
|------|------|------|
| 分歧样本识别 | ✅ 已完成 | 统计200验证集中118个分歧样本（59%） |
| 文本消融归因 | ✅ 已完成 | 分割文本片段，逐一移除观察预测变化 |
| 补偿提示生成 | ✅ 已完成 | 生成"忽略XX特征"的修正提示词 |
| 反思循环实现 | ✅ 已完成 | 最多3轮反思，收敛或拒识 |
| 实验运行 | ✅ 已完成 | 118个分歧样本全部处理完毕 |

### 2.4 Agent3模型更换

| 任务 | 状态 | 详情 |
|------|------|------|
| 魔芋API接入 | ✅ 已完成 | 通过https://www.moyu.info/v1调用GPT-4o-mini |
| 小样本测试 | ✅ 已完成 | 5样本测试成功，验证集准确率40% |
| 完整推理 | ✅ 已完成 | Agent3训练集和验证集推理完成 |

---

## 三、当前问题

### 3.1 Agent2 GLM-5V-Turbo严重限流（最严重）

**现象**: 几乎所有样本都返回`fallback_uniform`（label=0, conf=0.500）

**错误信息**:
```
[警告] GLM-5V-Turbo调用失败，返回fallback: 限流: 429 Client Error: Too Many Requests
```

**影响**: Agent2的结果基本无效，严重影响整体实验结果和共识效果

**原因**: GLM免费额度非常有限，连续调用很快触发429限流

### 3.2 Step5因果反思收敛率低

**现象**: 118个分歧样本中仅8个收敛（6.8%），110个拒识（93.2%）

**原因分析**:
1. 文本消融策略不够精准（简单按句子分割）
2. 补偿提示不够有效（"忽略XX片段"对LLM影响有限）
3. Agent2大量fallback结果难以修正
4. 缺乏跨Agent证据交换机制

### 3.3 实验结果不理想

| 方法 | Acc% | F1% |
|------|------|-----|
| Agent1(deepseek) | 60.00 | 42.03 |
| Agent2(glm) | 53.50 | 29.01 |
| Agent3(gpt) | 65.50 | 66.99 |
| Hybrid_GAT | 64.50 | 58.48 |
| BestAgent | 65.50 | 66.99 |

**问题**: Agent2表现差拉低整体效果，共识机制未能显著超越最佳单Agent

---

## 四、当前卡住的位置

实验流程已全部跑通，但存在以下卡点：

1. **Agent2无效**: GLM-5V-Turbo限流导致90%+样本返回fallback，Agent2形同虚设
2. **Step5效果差**: 因果反思收敛率仅6.8%，几乎没有起到修正作用
3. **共识收益有限**: 当前共识结果未超越最佳单Agent（Agent3），体现不出框架价值

---

## 五、下一步计划

### 5.1 优先解决：Agent2限流问题

**方案A（推荐）**: 更换Agent2模型
- 将Agent2从GLM-5V-Turbo更换为通过魔芋中转的GPT-4o（支持图像输入）
- 优点：稳定、不限流、图像理解能力强
- 缺点：费用稍高

**方案B**: 优化调用策略
- 大幅降低调用频率（每30-60秒调用一次）
- 增加更长的重试等待时间（10-30秒）
- 分批次运行（每次50样本，间隔1小时）

**方案C**: 改用CLIP描述模式
- 对Agent2也使用CLIP生成的图像描述（而非直接图像输入）
- 使用纯文本模型（如DeepSeek）替代GLM-5V-Turbo
- 优点：费用低、稳定
- 缺点：失去直接图像输入的优势

### 5.2 优化Step5因果反思

**改进策略**:
1. **跨Agent证据交换**: 让分歧Agent看到其他Agent的reasoning和预测结果
2. **更精细的归因**: 使用关键词/实体识别替代简单句子分割
3. **分层反思**: 简单分歧（两Agent一致）直接采用多数投票，仅复杂分歧触发完整反思
4. **混合策略**: 将反思与GAT共识结果结合

### 5.3 重新运行完整实验

解决Agent2问题后，重新运行：
```bash
$env:OPENAI_API_KEY="sk-LXBBRassCHHkzRzwl3HoiALErsfxUkkNGZvlOAIorJ5XURYJ"
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=200 --max_val=200 --provider1=deepseek --provider2=gpt --provider3=gpt --force_rerun_agent=1
```

### 5.4 最终验证

验证完整框架的效果：
1. 各Agent独立表现
2. MajorityVoting、DS_Fusion、GAT_DS_Fusion、Hybrid_GAT共识效果
3. 因果反思后的修正效果
4. 拒识样本分析

---

## 六、踩过的坑（绝对不要再踩）

### 6.1 Qwen API系列问题

| 问题 | 错误信息 | 原因 | 解决方案 |
|------|----------|------|----------|
| 401 Unauthorized | API密钥无效 | 密钥格式错误或未正确加载 | 重新生成密钥，确保完整复制 |
| 403 Forbidden | 免费额度用尽 | 未完成支付配置或禁用"仅使用免费层级" | 在阿里云百炼控制台完成支付配置 |
| 超时错误 | Read timed out (30s) | 超时时间太短 | 延长至120秒，增加重试次数 |

**结论**: Qwen API不稳定且免费额度有限，不建议继续使用

### 6.2 GLM-5V-Turbo限流

- ❌ 连续调用会很快触发429限流
- ❌ 限流后返回`fallback_uniform`（均匀分布），导致结果无效
- ❌ 不要期望GLM能一次性完成200样本的推理
- ✅ 如果必须使用，需大幅降低调用频率（每30-60秒一次）

### 6.3 魔芋API注意事项

- ✅ 正确地址：`https://www.moyu.info/v1`
- ❌ 错误地址：`https://api.moyu.info/v1`（无法解析）
- ✅ 运行时必须显式设置环境变量：`$env:OPENAI_API_KEY="xxx"`
- ✅ 小样本测试（5个）成功后再运行完整实验

### 6.4 缓存维度不匹配

- ❌ 不同样本量的缓存文件不能混用
- ❌ 修改`max_train`或`max_val`后，必须删除旧缓存文件
- ✅ 缓存路径：`checkpoints/hateful_memes/llm_train_agentX.pt` 和 `llm_val_agentX.pt`

### 6.5 PIL Image未导入

- ❌ `_image_to_base64`方法中使用`Image.LANCZOS`但未导入PIL
- ✅ 必须在`llm_api.py`顶部添加`from PIL import Image`

### 6.6 GAT空指针异常

- ❌ 训练样本不足时`gat_engine`为None
- ✅ 调用`gat_engine.build_state()`前必须检查是否为None

### 6.7 文件编码问题

- ❌ Windows下默认编码是GBK
- ✅ 读取包含中文的文件必须指定`encoding='utf-8'`

### 6.8 图像路径拼接错误

- ❌ `os.path.join(DATA_DIR, 'img', item['img'])` 可能导致双层`img/img/`
- ✅ 需要检查路径是否存在，灵活拼接

---

## 七、重要文件路径

| 文件 | 用途 |
|------|------|
| `src/llm_api.py` | LLM API配置和调用逻辑（含多模态支持） |
| `src/llm_agent.py` | LLM Agent创建函数和推理封装 |
| `src/step4_hateful_memes/evaluate_with_llm.py` | 主评估管线（LLM推理+GAT训练+评估） |
| `src/step4_hateful_memes/evaluate_step5_causal_reflection.py` | 因果反事实反思实验 |
| `src/step5/causal_reflection.py` | 原始因果反思框架（仅支持合成数据） |
| `src/gat_consensus.py` | GAT共识引擎实现 |
| `keys.env` | API密钥（已加入.gitignore） |
| `checkpoints/hateful_memes/` | 推理结果缓存目录 |
| `results/hateful_memes/` | 实验结果输出目录 |

---

## 八、关键配置

### 环境变量
```bash
$env:DEEPSEEK_API_KEY="xxx"      # Agent1和部分Agent3调用
$env:GLM_API_KEY="xxx"           # Agent2调用（当前限流严重）
$env:OPENAI_API_KEY="xxx"        # 魔芋中转GPT-4o-mini
```

### Agent配置（当前）
```python
Agent1: deepseek/deepseek-v4-flash (文本专家)
Agent2: glm/GLM-5V-Turbo (图像专家，限流严重)
Agent3: gpt/gpt-4o-mini-2024-07-18 (跨模态专家，魔芋中转)
```

### 魔芋API配置
```python
base_url: https://www.moyu.info/v1
model: gpt-4o-mini-2024-07-18
api_key: sk-LXBBRassCHHkzRzwl3HoiALErsfxUkkNGZvlOAIorJ5XURYJ
```

### 实验命令示例
```bash
# 运行完整实验（仅重新运行Agent2）
$env:OPENAI_API_KEY="sk-LXBBRassCHHkzRzwl3HoiALErsfxUkkNGZvlOAIorJ5XURYJ"
python src/step4_hateful_memes/evaluate_with_llm.py --max_train=200 --max_val=200 --provider1=deepseek --provider2=gpt --provider3=gpt --force_rerun_agent=1

# 运行因果反思实验
python src/step4_hateful_memes/evaluate_step5_causal_reflection.py
```

---

## 九、紧急事项

**最紧急**: 解决Agent2的有效性问题。当前GLM-5V-Turbo限流导致Agent2基本无效，建议优先考虑将Agent2更换为通过魔芋中转的GPT-4o（支持图像输入），或改用CLIP描述模式+纯文本模型。

**次紧急**: 优化Step5因果反思策略，当前6.8%的收敛率太低，需要引入跨Agent证据交换机制。

---

## 十、费用控制建议

1. **小样本先行**: 每次实验先用5-10个样本测试，确认正常后再扩展
2. **独立缓存**: 利用独立缓存机制，避免重复运行已完成的Agent
3. **分歧触发**: Step5仅对分歧样本触发反思，约20-30%的验证集
4. **降低温度**: 使用`temperature=0.1`减少输出随机性，降低token消耗
5. **监控API调用**: 关注各模型的token消耗和调用次数，及时调整策略

---

*最后更新: 2026-07-17*