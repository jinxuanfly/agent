# -*- coding: utf-8 -*-
"""检查旧缓存样本数 + 清理测试垃圾缓存"""
import torch
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
CKPT = 'checkpoints/hateful_memes'

print("=" * 60)
print("[1] 旧缓存（gpt5+gemini+gpt5, 500样本实验）检查:")
print("=" * 60)
for i, agent in enumerate(['Agent1(gpt5,弃用)', 'Agent2(gemini,复用)', 'Agent3(gpt5,复用)']):
    for split in ['train', 'val']:
        f = os.path.join(CKPT, f'llm_{split}_agent{i}.pt')
        if os.path.exists(f):
            d = torch.load(f, map_location='cpu', weights_only=False)
            n = d['alphas'].shape[0]
            # 检查是否是fallback数据（beliefs全是均匀分布）
            bel = d['beliefs']
            is_uniform = bool((torch.abs(bel - 1.0 / bel.shape[-1]) < 1e-6).all())
            print(f"  {agent} {split}: {n}样本, keys={list(d.keys())}, 全fallback={is_uniform}")
        else:
            print(f"  {agent} {split}: 文件不存在")

print()
print("=" * 60)
print("[2] 之前测试产生的垃圾seed缓存（5样本mock数据，需删除）:")
print("=" * 60)
junk = []
for f in os.listdir(CKPT):
    if 'seed42' in f or 'seed123' in f:
        path = os.path.join(CKPT, f)
        d = torch.load(path, map_location='cpu', weights_only=False)
        n = d['alphas'].shape[0]
        junk.append(path)
        print(f"  {f}: {n}样本")

if junk:
    print(f"\n共{len(junk)}个垃圾文件（5样本测试数据）")
    for p in junk:
        os.remove(p)
    print("已全部删除")
else:
    print("无垃圾文件")
