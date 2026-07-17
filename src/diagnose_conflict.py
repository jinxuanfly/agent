"""
诊断CIFAR-10N上的冲突系数K分布和DS融合行为
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from step4.evaluate_cifar10n import load_features_and_heads, get_agent_outputs, ds_fusion_decision
from plot_utils import setup_chinese_font
setup_chinese_font()

features, test_labels = load_features_and_heads()
B = 500
all_alphas, all_beliefs, all_uncertainties, all_embs = [], [], [], []
for name in ['agent1', 'agent2', 'agent3']:
    feats, head = features[name]
    alpha, b, u, emb = get_agent_outputs(feats[:B], head)
    all_alphas.append(alpha)
    all_beliefs.append(b)
    all_uncertainties.append(u)
    all_embs.append(emb)

# === 1. 分析冲突系数K分布 ===
print("=" * 60)
print("冲突系数K分析")
print("=" * 60)

N = len(all_beliefs)
agreements = []
for b_idx in range(B):
    for i in range(N):
        for j in range(i+1, N):
            agree = (all_beliefs[i][b_idx] * all_beliefs[j][b_idx]).sum().item()
            agreements.append(agree)

agreements = np.array(agreements)
K_values = 1 - agreements

print(f"agreement: mean={agreements.mean():.4f}, std={agreements.std():.4f}")
print(f"agreement 分布:")
for p in [5, 25, 50, 75, 95]:
    print(f"  {p}%分位: {np.percentile(agreements, p):.4f}")
print(f"\nK=1-agreement: mean={K_values.mean():.4f}")
print(f"K > 0.15: {(K_values > 0.15).mean()*100:.1f}%")
print(f"K > 0.30: {(K_values > 0.30).mean()*100:.1f}%")
print(f"K > 0.50: {(K_values > 0.50).mean()*100:.1f}%")

# === 2. 逐样本展示冲突 & 信念分布 ===
print("\n" + "=" * 60)
print("逐样本分析（前15个）")
print("=" * 60)

for b_idx in range(15):
    print(f"\n样本{b_idx} (真实={test_labels[b_idx].item()}):")
    for i in range(N):
        b = all_beliefs[i][b_idx]
        top_k = b.topk(3)
        top_classes = top_k.indices.tolist()
        top_vals = top_k.values.tolist()
        print(f"  A{i+1}: pred={b.argmax().item()}, "
              f"b={[f'{v:.3f}' for v in b.numpy().round(3)]}, "
              f"u={all_uncertainties[i][b_idx].item():.4f}")
    
    # DS结果
    ds_pred, ds_rej, ds_u = ds_fusion_decision(
        [ab[b_idx:b_idx+1] for ab in all_beliefs],
        [au[b_idx:b_idx+1] for au in all_uncertainties]
    )
    print(f"  DS: pred={ds_pred.item()}, u={ds_u.item():.4f}, rej={ds_rej.item()}, "
          f"correct={'✓' if ds_pred.item()==test_labels[b_idx].item() else '✗'}")
    
    # 冲突分析
    for i in range(N):
        for j in range(i+1, N):
            agree = (all_beliefs[i][b_idx] * all_beliefs[j][b_idx]).sum().item()
            print(f"  A{i+1}↔A{j+1}: agreement={agree:.4f}, K={1-agree:.4f}")

# === 3. 找出DS融合"应该错"但可能救回的样本 ===
print("\n" + "=" * 60)
print("DS融合错误分析")
print("=" * 60)

ds_preds_all, ds_rej_all, ds_u_all = ds_fusion_decision(all_beliefs, all_uncertainties)
y_true = test_labels[:B]

# 错误但被接受(未拒识)的样本
errors_accepted = (~ds_rej_all) & (ds_preds_all != y_true)
n_ea = errors_accepted.sum().item()
print(f"DS融合错误（接受了但错误）: {n_ea}/{B} ({n_ea/B*100:.1f}%)")

if n_ea > 0:
    # 这些样本的冲突分布
    ea_indices = torch.where(errors_accepted)[0]
    print(f"\n错误接受样本的冲突分析:")
    for idx in ea_indices[:10]:
        idx_int = idx.item()
        agreements_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                agree = (all_beliefs[i][idx_int] * all_beliefs[j][idx_int]).sum().item()
                agreements_pairs.append(agree)
        
        mean_agree = np.mean(agreements_pairs)
        print(f"  样本{idx_int}: 真实={y_true[idx_int].item()}, "
              f"DS_pred={ds_preds_all[idx_int].item()}, "
              f"DS_u={ds_u_all[idx_int].item():.4f}, "
              f"mean_agreement={mean_agree:.4f}")

# === 4. 手动在错误样本上测试共识 ===
print("\n" + "=" * 60)
print("手动共识测试（在DS错误样本上）")
print("=" * 60)

from step4.evaluate_cifar10n import similarity_consensus_batch

n_tested = 0
n_fixed = 0
for b_idx in range(B):
    if not errors_accepted[b_idx]:
        continue
    if n_tested >= 20:
        break
    
    n_tested += 1
    single_beliefs = [ab[b_idx:b_idx+1] for ab in all_beliefs]
    single_us = [au[b_idx:b_idx+1] for au in all_uncertainties]
    
    # 运行共识
    new_b, new_u, conv, n_iter, new_a = similarity_consensus_batch(
        single_beliefs, single_us, max_iters=15, verbose=False
    )
    
    new_pred, new_rej, new_u_val = ds_fusion_decision(new_b, new_u)
    
    old_pred = ds_preds_all[b_idx]
    old_correct = old_pred == y_true[b_idx]
    new_correct = new_pred[0] == y_true[b_idx]
    
    fixed = (not old_correct) and new_correct
    
    if fixed:
        n_fixed += 1
        print(f"  样本{b_idx}: DS_pred={old_pred.item()}(错误)→共识_pred={new_pred[0].item()}(正确!)")
    else:
        print(f"  样本{b_idx}: DS_pred={old_pred.item()}, 共识_pred={new_pred[0].item()}, "
              f"DS_rej={ds_rej_all[b_idx].item()}, 共识_rej={new_rej[0].item()}")

if n_tested > 0:
    print(f"\n共识挽救率: {n_fixed}/{n_tested} = {n_fixed/n_tested*100:.1f}%")