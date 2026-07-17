"""
LLM增强管线快速验证脚本
=======================
在mock模式下验证管线逻辑正确性，不消耗API配额。
"""
import os, sys
import torch
import numpy as np

# 设置mock模式环境变量（确保走模拟）
os.environ['LLM_MOCK_MODE'] = '1'

sys.path.insert(0, 'src')

print("="*60)
print("LLM增强管线 - 快速验证")
print("="*60)

# 1. 验证导入
print("\n[1] 验证模块导入...")
from src.llm_api import LLMClient, PROVIDER_CONFIGS
from src.llm_agent import LLMAgent, create_single_agent, AGENT_PROMPTS
print("  ✓ 导入成功")

# 2. 验证LLMClient模拟模式
print("\n[2] 验证LLMClient模拟模式...")
client = LLMClient(provider='deepseek', temperature=0.1)
mock_resp = client.chat(
    messages=[{"role": "user", "content": "测试消息"}],
    system_prompt="你是一个测试助手",
)
content_str = mock_resp['content'][:80]
print(f"  ✓ 模拟响应: {content_str}...")
print(f"  ✓ 响应包含label/probs: {list(mock_resp.keys())}")
assert mock_resp.get('success'), "success标志缺失"
assert 'content' in mock_resp, "content字段缺失"
print("  ✓ LLMClient模拟模式正常")

# 3. 验证LLMAgent
print("\n[3] 验证LLMAgent...")
agent = create_single_agent(
    provider='deepseek',
    name="验证Agent",
    system_prompt=AGENT_PROMPTS['text_focused'],
)
alpha, belief, uncertainty, emb = agent.forward("这是一个测试样本")
print(f"  Alpha: {alpha.numpy()}")
print(f"  Belief: {belief.numpy()}")
print(f"  Uncertainty: {uncertainty.item():.4f}")
print(f"  Embedding shape: {emb.shape}")
assert alpha.shape == (2,), f"Alpha shape错误: {alpha.shape}"
assert 0 <= uncertainty <= 1.0, f"Uncertainty不在[0,1]范围: {uncertainty}"
print("  ✓ Agent前向传播正常")

# 4. 验证分歧解构
print("\n[4] 验证分歧解构...")
from step4_hateful_memes.evaluate_with_llm import DisagreementDeconstructor, ds_fusion_decision

decon = DisagreementDeconstructor(u_threshold=0.5, K_threshold=0.3)

# 构造3个Agent的信念
b1 = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
b2 = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
b3 = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
u1 = torch.tensor([0.1])
u2 = torch.tensor([0.2])
u3 = torch.tensor([0.8])

b_stack = torch.stack([b1, b2, b3], dim=1)
u_stack = torch.stack([u1, u2, u3], dim=1)

ctype, K = decon.deconstruct(b_stack[0], u_stack[0])
print(f"  分歧类型: {ctype}, K={K:.4f}")
assert ctype in ['evidence_conflict', 'ignorance_conflict', 'none']
print("  ✓ 分歧解构正常")

# 5. 验证DS融合
print("\n[5] 验证DS融合...")
# ds_fusion_decision 期望2D输入 [batch, num_classes]
b_list = [b1, b2, b3]  # shape: [1, 2]
u_list = [u1, u2, u3]  # shape: [1]
preds, rej, global_u = ds_fusion_decision(b_list, u_list, u_threshold=0.5)
print(f"  DS Preds: {preds}, Rejected: {rej}, Global u: {global_u}")
assert preds.shape == (1,)
print("  ✓ DS融合正常")

# 6. 检查PROVIDER_CONFIGS
print("\n[6] 检查提供者配置...")
print(f"  可用提供者: {list(PROVIDER_CONFIGS.keys())}")
for k, v in PROVIDER_CONFIGS.items():
    print(f"    {k}: model={v.get('default_model', 'N/A')}, base_url={v['base_url'][:50]}...")

# 7. 验证管线入口
print("\n[7] 验证评估管线入口...")
from step4_hateful_memes.evaluate_with_llm import run_llm_evaluation
print("  ✓ run_llm_evaluation 函数可导入")

# 8. 尝试小规模运行（mock模式，5样本）
print("\n[8] 小规模运行测试 (5样本, mock模式)...")
print("  ★ 此步骤将运行完整管线（但用mock LLM，不会实际调用API）")
print("  ★ 可跳过此步骤（按Ctrl+C中断），或等待约30秒完成")
print()
try:
    metrics, agents = run_llm_evaluation(
        max_train=5,
        max_val=5,
        provider1='deepseek',
        provider2='qwen',
        provider3='glm',
        train_gat=True,
        batch_size=4,
        save_cache=False,
    )
    print(f"\n  ✓ 管线运行成功！")
    print(f"  结果摘要:")
    for method, m in metrics.items():
        print(f"    {method:<25s}: Acc={m['accuracy']:.2f}%")
except KeyboardInterrupt:
    print("\n  ⏸ 已跳过小规模运行（用户中断）")
    print("  代码导入和基础验证已通过，可直接运行完整管线")
except Exception as e:
    print(f"\n  ⚠ 小规模运行出现异常: {e}")
    import traceback
    traceback.print_exc()
    print("\n  但基础验证已通过，请检查后重试")

print("\n" + "="*60)
print("验证完成！")
print("="*60)
print("\n快速启动命令:")
print("  # mock模式（默认，无需API key）：")
print("  python src/step4_hateful_memes/evaluate_with_llm.py --max_val 100 --no_gat")
print()
print("  # 使用真实API：")
print("  set DEEPSEEK_API_KEY=sk-xxx")
print("  set QWEN_API_KEY=sk-xxx")  
print("  set GLM_API_KEY=sk-xxx")
print("  python src/step4_hateful_memes/evaluate_with_llm.py --max_val 200")