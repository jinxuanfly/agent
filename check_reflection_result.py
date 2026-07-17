import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

r = torch.load('results/hateful_memes/causal_reflection_results.pt')

print("因果反思结果:")
print(f"  收敛样本: {sum(r['converged'])}/{len(r['converged'])} ({sum(r['converged'])/len(r['converged'])*100:.1f}%)")
print(f"  拒识样本: {sum(r['rejected'])}/{len(r['rejected'])} ({sum(r['rejected'])/len(r['rejected'])*100:.1f}%)")
print(f"  平均反思轮数: {sum(r['reflections_used'])/len(r['reflections_used']):.1f}")
print(f"  额外API调用次数: {r['api_calls']}")