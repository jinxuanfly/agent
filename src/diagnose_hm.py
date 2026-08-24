"""Diagnose Hateful Memes evaluation results"""
import json
import numpy as np

with open('results/hateful_memes/evaluation_details_enhanced.json') as f:
    data = json.load(f)
y_true = np.array(data['y_true'])

# 1. Check if DS, GAT_DS, and EMNet are identical
for name in ['DS_Fusion_preds', 'GAT_DS_Fusion_preds', 'GAT_EMNet_Fusion_preds']:
    p = np.array(data[name])
    print(f'{name}: first 20={p[:20].tolist()}')

# 2. Uncertainty distributions
for name, label in [('DS_Fusion_uncertainty', 'DS'), ('GAT_DS_Fusion_uncertainty', 'GAT'), ('GAT_EMNet_Fusion_uncertainty', 'EMNet')]:
    u = np.array(data[name])
    print(f'{label}_U: mean={u.mean():.4f} std={u.std():.4f} min={u.min():.4f} max={u.max():.4f}')
    print(f'  rejected (u>{0.3}): {(u > 0.3).sum()}/{len(u)}')

# 3. Energy history
print(f'\nEnergy history: {data["energy_history"]}')
print(f'GAT iterations: {data["n_iters"]}')

# 4. Model-by-model analysis
for model in ['Agent1_Text', 'Agent2_Image', 'Agent3_Fusion']:
    preds = np.array(data[f'{model}_preds'])
    tp = ((preds == 1) & (y_true == 1)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    print(f'\n{model}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}')

# 5. Check if GAT outputs differ from raw
ds_u = np.array(data['DS_Fusion_uncertainty'])
gat_u = np.array(data['GAT_DS_Fusion_uncertainty'])
print(f'\nGAT vs DS uncertainty diff: max={np.abs(gat_u - ds_u).max():.6f}')
ds_preds = np.array(data['DS_Fusion_preds'])
gat_preds = np.array(data['GAT_DS_Fusion_preds'])
print(f'GAT vs DS preds same: {(ds_preds == gat_preds).sum()}/{len(ds_preds)}')

# 6. Check conflict K values
K = np.array(data['K_values'])
print(f'\nK values: mean={K.mean():.4f}, median={np.median(K):.4f}, >0.5: {(K > 0.5).sum()}/{len(K)}')
types = data['conflict_types']
evidence = sum(1 for t in types if t == 'evidence_conflict')
ignorance = sum(1 for t in types if t == 'ignorance_conflict')
none_c = sum(1 for t in types if t == 'none')
print(f'Conflict types: evidence={evidence}, ignorance={ignorance}, none={none_c}')