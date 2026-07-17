"""统计Step4验证集中三个Agent的分歧情况"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

checkpoint_dir = 'checkpoints/hateful_memes'

agent_results = []
for i in range(3):
    path = os.path.join(checkpoint_dir, f'llm_val_agent{i}.pt')
    result = torch.load(path, map_location='cpu')
    agent_results.append(result)
    print(f"Agent{i}: keys={list(result.keys())}, belief shape={result['beliefs'].shape}")

beliefs = [r['beliefs'] for r in agent_results]
preds = [b.argmax(dim=1) for b in beliefs]

n_samples = preds[0].shape[0]
print(f"\n总样本数: {n_samples}")

# 统计分歧情况
all_same = 0
two_same = 0
all_different = 0
disagreement_indices = []

for idx in range(n_samples):
    p0, p1, p2 = preds[0][idx], preds[1][idx], preds[2][idx]
    if p0 == p1 == p2:
        all_same += 1
    elif p0 == p1 or p0 == p2 or p1 == p2:
        two_same += 1
        disagreement_indices.append(idx)
    else:
        all_different += 1
        disagreement_indices.append(idx)

print(f"\n一致样本（三Agent相同）: {all_same} ({all_same/n_samples*100:.1f}%)")
print(f"部分分歧（两Agent相同）: {two_same} ({two_same/n_samples*100:.1f}%)")
print(f"完全分歧（三Agent不同）: {all_different} ({all_different/n_samples*100:.1f}%)")
print(f"分歧样本总数: {len(disagreement_indices)} ({len(disagreement_indices)/n_samples*100:.1f}%)")

# 保存分歧样本索引
torch.save({'disagreement_indices': disagreement_indices}, 
           os.path.join(checkpoint_dir, 'disagreement_indices.pt'))
print(f"\n分歧样本索引已保存到 {checkpoint_dir}/disagreement_indices.pt")

# 分析分歧样本的详细信息
print("\n分歧样本详情（前10个）:")
for idx in disagreement_indices[:10]:
    p0, p1, p2 = preds[0][idx], preds[1][idx], preds[2][idx]
    b0, b1, b2 = beliefs[0][idx], beliefs[1][idx], beliefs[2][idx]
    print(f"  样本{idx}: Agent1={p0}(conf={b0[p0]:.2f}), Agent2={p1}(conf={b1[p1]:.2f}), Agent3={p2}(conf={b2[p2]:.2f})")