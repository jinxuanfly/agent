# -*- coding: utf-8 -*-
"""分析训练集缓存数据"""
import os, sys, io, torch, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

CHECKPOINT_DIR = 'checkpoints/hateful_memes'

print("=" * 70)
print("训练集缓存数据分析")
print("=" * 70)

# 加载训练集缓存
all_beliefs = []
all_uncertainties = []
all_embs = []
all_alphas = []

for i in range(3):
    path = os.path.join(CHECKPOINT_DIR, f'llm_train_agent{i}.pt')
    if not os.path.exists(path):
        print(f"[错误] 缓存文件不存在: {path}")
        sys.exit(1)
    data = torch.load(path, map_location='cpu', weights_only=True)
    all_beliefs.append(data['beliefs'])
    all_uncertainties.append(data['uncertainties'])
    all_embs.append(data['embs'])
    all_alphas.append(data['alphas'])
    print(f"\nAgent{i+1} 缓存:")
    print(f"  样本数: {data['beliefs'].shape[0]}")
    print(f"  beliefs shape: {data['beliefs'].shape}")
    print(f"  uncertainties shape: {data['uncertainties'].shape}")
    print(f"  embs shape: {data['embs'].shape}")

# 加载标签
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.step4_hateful_memes.evaluate_with_llm import HatefulMemesDataset
train_dataset = HatefulMemesDataset(split='train', max_samples=200, load_images=False)
train_labels = torch.tensor(train_dataset.labels)
print(f"\n训练集标签: shape={train_labels.shape}, 分布: 0={((train_labels==0).sum().item())}, 1={((train_labels==1).sum().item())}")

# Agent准确率
print("\n" + "=" * 70)
print("各Agent准确率")
print("=" * 70)
agent_names = ['Agent1(GPT-5文本)', 'Agent2(Gemini图像)', 'Agent3(GPT-5跨模态)']
for i in range(3):
    preds = all_beliefs[i].argmax(dim=1)
    acc = (preds == train_labels).float().mean().item() * 100
    avg_u = all_uncertainties[i].mean().item()
    avg_conf = 1 - avg_u
    # F1
    tp = ((preds == 1) & (train_labels == 1)).sum().item()
    fp = ((preds == 1) & (train_labels == 0)).sum().item()
    fn = ((preds == 0) & (train_labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8) * 100
    print(f"\n{agent_names[i]}:")
    print(f"  Acc: {acc:.2f}%")
    print(f"  F1:  {f1:.2f}%")
    print(f"  Avg_u: {avg_u:.4f}  Avg_conf: {avg_conf:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    print(f"  Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%")

# 分歧分析
print("\n" + "=" * 70)
print("分歧分析")
print("=" * 70)
preds_list = [all_beliefs[i].argmax(dim=1) for i in range(3)]

# 两两分歧
for i in range(3):
    for j in range(i+1, 3):
        disagree = (preds_list[i] != preds_list[j]).sum().item()
        print(f"  Agent{i+1} vs Agent{j+1}: 分歧={disagree}/{len(train_labels)} ({disagree/len(train_labels)*100:.1f}%)")

# 三方分歧
all_agree = (preds_list[0] == preds_list[1]) & (preds_list[1] == preds_list[2])
any_disagree = ~all_agree
print(f"\n  全部一致: {all_agree.sum().item()}/{len(train_labels)} ({all_agree.float().mean().item()*100:.1f}%)")
print(f"  存在分歧: {any_disagree.sum().item()}/{len(train_labels)} ({any_disagree.float().mean().item()*100:.1f}%)")

# 分歧样本的准确率
if any_disagree.sum() > 0:
    disagree_indices = torch.where(any_disagree)[0]
    print(f"\n  分歧样本分析:")
    for i in range(3):
        acc_disagree = (preds_list[i][disagree_indices] == train_labels[disagree_indices]).float().mean().item() * 100
        print(f"    Agent{i+1}在分歧样本上的Acc: {acc_disagree:.2f}%")

# 相关性矩阵
print("\n" + "=" * 70)
print("Agent相关性矩阵")
print("=" * 70)
N = 3
for i in range(N):
    row = []
    for j in range(N):
        if i == j:
            row.append("1.000")
        else:
            agreement = (preds_list[i] == preds_list[j]).float().mean().item()
            p_i = preds_list[i].float().mean().item()
            p_j = preds_list[j].float().mean().item()
            random_agree = p_i * p_j + (1 - p_i) * (1 - p_j)
            denom = 1 - random_agree
            if denom > 1e-6:
                corr = (agreement - random_agree) / denom
            else:
                corr = 0.0
            corr = max(0.0, min(1.0, corr))
            row.append(f"{corr:.3f}")
    print(f"  Agent{i+1}: [{', '.join(row)}]")

# DS融合基线
print("\n" + "=" * 70)
print("DS融合基线")
print("=" * 70)
import torch.nn.functional as F
agent_accs = torch.tensor([(preds_list[i] == train_labels).float().mean().item() for i in range(3)])
agent_weights = F.softmax(agent_accs, dim=0)
print(f"  Agent权重: {agent_weights.numpy()}")

# 简单DS融合
b0 = all_beliefs[0]
u0 = all_uncertainties[0]
w0 = agent_weights[0]
combined_b = b0 * (1.0 - u0.unsqueeze(-1)) * w0
combined_u = u0 * w0 + (1 - w0) * 0.5

for idx in range(1, 3):
    b = all_beliefs[idx]
    u = all_uncertainties[idx]
    w = agent_weights[idx]
    m1_b = combined_b
    m1_u = combined_u
    m2_b = b * (1.0 - u.unsqueeze(-1)) * w
    m2_u = u * w + (1 - w) * 0.5
    sum_m1_b = m1_b.sum(dim=-1)
    sum_m2_b = m2_b.sum(dim=-1)
    agree = (m1_b * m2_b).sum(dim=-1)
    K = sum_m1_b * sum_m2_b - agree
    denom = 1.0 - K + 1e-8
    new_b = (m1_b * m2_b + m1_b * m2_u.unsqueeze(-1) + m1_u.unsqueeze(-1) * m2_b) / denom.unsqueeze(-1)
    new_u = m1_u * m2_u / denom
    combined_b = new_b
    combined_u = new_u

global_b = combined_b / (1.0 - combined_u.unsqueeze(-1) + 1e-8)
ds_preds = global_b.argmax(dim=-1)
ds_acc = (ds_preds == train_labels).float().mean().item() * 100
print(f"  DS融合Acc: {ds_acc:.2f}%")

# Majority Voting
mv_preds = torch.stack(preds_list).mode(dim=0).values
mv_acc = (mv_preds == train_labels).float().mean().item() * 100
print(f"  Majority Voting Acc: {mv_acc:.2f}%")

# Best Agent
best_acc = max(agent_accs).item() * 100
print(f"  Best Agent Acc: {best_acc:.2f}%")

# Embedding分析
print("\n" + "=" * 70)
print("Symbolic Embedding分析")
print("=" * 70)
for i in range(3):
    emb = all_embs[i]
    nonzero = (emb.abs() > 1e-6).sum(dim=1).float().mean().item()
    emb_norm = emb.norm(dim=1).mean().item()
    print(f"  Agent{i+1}: 非零维度均值={nonzero:.1f}/256, L2范数均值={emb_norm:.4f}")

# 检查embedding是否相同（旧随机投影问题）
emb_diff_01 = (all_embs[0] - all_embs[1]).norm(dim=1).mean().item()
emb_diff_02 = (all_embs[0] - all_embs[2]).norm(dim=1).mean().item()
emb_diff_12 = (all_embs[1] - all_embs[2]).norm(dim=1).mean().item()
print(f"\n  Agent间embedding距离:")
print(f"    Agent1-Agent2: {emb_diff_01:.4f}")
print(f"    Agent1-Agent3: {emb_diff_02:.4f}")
print(f"    Agent2-Agent3: {emb_diff_12:.4f}")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
