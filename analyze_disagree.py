# -*- coding: utf-8 -*-
"""分析分歧样本的基线表现"""
import os, sys, io, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from sklearn.metrics import accuracy_score, f1_score

CHECKPOINT_DIR = 'checkpoints/hateful_memes'
RESULT_DIR = 'results/hateful_memes'

# 加载缓存
b_list = []
for i in range(3):
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}.pt'), weights_only=False)
    b_list.append(ckpt['beliefs'].argmax(dim=1))

details = json.load(open(os.path.join(RESULT_DIR, 'details_llm_gpt5_gemini_gpt5.json'), encoding='utf-8'))
y = torch.tensor(details['y_true'])

# 分歧样本
disagree = []
for i in range(len(b_list[0])):
    if not (b_list[0][i] == b_list[1][i] == b_list[2][i]):
        disagree.append(i)

# 多数投票
mv = torch.mode(torch.stack(b_list, dim=1), dim=1)[0]

print(f"总样本: {len(b_list[0])}")
print(f"分歧样本: {len(disagree)}")
print(f"一致样本: {len(b_list[0]) - len(disagree)}")
print(f"\n全样本多数投票 Acc: {accuracy_score(y, mv)*100:.2f}%")
print(f"分歧样本 MV Acc: {accuracy_score(y[disagree], mv[disagree])*100:.2f}%")
print(f"分歧样本 MV F1: {f1_score(y[disagree], mv[disagree], average='binary')*100:.2f}%")

# 各Agent在分歧样本上的表现
for i in range(3):
    acc = accuracy_score(y[disagree], b_list[i][disagree]) * 100
    print(f"分歧样本 Agent{i+1} Acc: {acc:.2f}%")

# 分歧类型分析（二元分类只有2v1）
print(f"\n分歧类型: 全部为 2v1 (二元分类无 1v1v1)")
print(f"二元分类下 SIMPLE_DISAGREEMENT_THRESHOLD=3 意味着所有分歧都触发反思")

# 估算成本
n_complex = len(disagree)  # 全部为complex（threshold=3）
avg_rounds = 1.5
calls_per_round = 4  # 3 ablation + 1 compensation per agent, 但有3个agent
total_agents_per_round = 3
est_calls = n_complex * avg_rounds * calls_per_round * total_agents_per_round
# 但之前实际是~300/45=6.7 per sample
realistic_calls = n_complex * 6.7
est_time_h = realistic_calls / 300 * 2.5

print(f"\n=== 成本估算 ===")
print(f"复杂分歧样本: {n_complex}")
print(f"预估API调用: ~{int(realistic_calls)} 次")
print(f"预估时间: ~{est_time_h:.1f} 小时")
print(f"之前100样本实验: 60分歧, 300调用, 2.5小时")

# 分析：哪些分歧样本的多数投票是错的（反思有机会修正）
wrong_mv = [i for i in disagree if mv[i] != y[i]]
correct_mv = [i for i in disagree if mv[i] == y[i]]
print(f"\n分歧样本中 MV错误: {len(wrong_mv)} (反思有机会修正)")
print(f"分歧样本中 MV正确: {len(correct_mv)} (反思不应改变)")
