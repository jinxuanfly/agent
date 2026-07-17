"""
第一步扩展：使用 HardConflictCircleData 训练模型
=================================================
专门为硬冲突场景训练模型，生成在边界样本上高置信度对立的智能体。

用途：
- 生成的数据中，边界区域 (0.85<r<1.15) 的样本被强制构造为 Agent1↔Agent2 对立
- 训练出的模型在这些样本上产生高置信度冲突（双方均u<0.2但预测相反）
- 这些样本在 GAT 共识时可能不收敛或需要极多轮次才能稳定
- 用于测试分歧解构器和 EMNet 的纠偏效果
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.synthetic_data import (
    HardConflictCircleData, EvidentialMLP, EvidentialLoss,
    train_epoch, evaluate, compute_ece, plot_calibration_curve,
    visualize_predictions, print_divergence_analysis, DEVICE, SEED
)

np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    print("=" * 60)
    print("第一步扩展：HardConflict 模型训练")
    print("=" * 60)
    
    # [1] 生成硬冲突数据
    print("\n[1/5] 生成 HardConflictCircleData ...")
    data = HardConflictCircleData(
        n_train=10000, n_test=2000, 
        noise_level=0.15, flip_ratio=0.15, hard_ratio=0.25
    )
    
    # 打印硬样本信息
    print(f"\n  硬冲突测试样本数: {len(data.hard_test_indices)}")
    if len(data.hard_test_indices) > 0:
        print(f"  硬样本示例索引: {data.hard_test_indices[:10].tolist()}")
        # 检查硬样本的真实标签分布
        hard_labels = data.y_true[data.hard_test_indices]
        print(f"  硬样本真实标签分布: 圆内={(hard_labels==1).sum()}, 圆外={(hard_labels==0).sum()}")
    
    # [2] 创建模型
    print("\n[2/5] 创建智能体证据网络...")
    agent_names = ['Agent1_x', 'Agent2_y', 'Agent3_r']
    models = {}
    
    models['Agent1_x'] = EvidentialMLP(input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32).to(DEVICE)
    models['Agent2_y'] = EvidentialMLP(input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32).to(DEVICE)
    models['Agent3_r'] = EvidentialMLP(input_dim=1, hidden_dim=128, output_dim=2, embed_dim=32).to(DEVICE)
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {total_params} 参数")
    
    # [3] 训练
    print("\n[3/5] 训练智能体 (在硬冲突数据上)...")
    agent_results = {}
    batch_size = 256
    n_epochs = 250  # 更多轮次，让模型充分学习硬样本
    
    name_to_key = {'Agent1_x': 'agent1', 'Agent2_y': 'agent2', 'Agent3_r': 'agent3'}
    
    for name in agent_names:
        model = models[name]
        data_key = name_to_key[name]
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = EvidentialLoss(annealing_step=50, kl_max_weight=0.4, target_S=12.0, S_weight=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5
        )
        
        train_dataset = TensorDataset(data.x_train[data_key], data.y_train[data_key])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"\n{'='*50}")
        print(f"训练 {name} (HardConflict)")
        print(f"{'='*50}")
        
        for epoch in range(n_epochs):
            loss, mse, kl, S_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, DEVICE)
            
            if (epoch + 1) % 25 == 0 or epoch == 0:
                eval_result = evaluate(model, data.x_test[data_key], data.y_test[data_key], DEVICE)
                acc = eval_result['accuracy']
                scheduler.step(loss)
                
                S_mean = eval_result['alpha'].sum(dim=1).mean().item()
                u_mean = eval_result['u'].mean().item()
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {loss:.4f} | Acc: {acc:.4f} | "
                      f"S: {S_mean:.1f} | u: {u_mean:.4f}")
        
        # 最终评估
        eval_result = evaluate(model, data.x_test[data_key], data.y_test[data_key], DEVICE)
        agent_results[name] = eval_result
        
        test_y_cpu = data.y_test[data_key].cpu()
        ece = compute_ece(eval_result['b'], test_y_cpu)
        print(f"\n  {name} 训练完成! Acc={eval_result['accuracy']:.4f}, ECE={ece:.4f}")
        
        # 保存模型
        model_path = f'models/{name}_evidential_hard.pth'
        torch.save(model.state_dict(), model_path)
        print(f"  模型已保存: {model_path}")
        
        # 校准曲线
        plot_calibration_curve(eval_result['b'], test_y_cpu, f'{name}_hard')
    
    # [4] 分析硬样本上的表现
    print("\n[4/5] 硬样本分析...")
    
    hard_test_indices = data.hard_test_indices[:30]  # 分析前30个硬样本
    print(f"\n  {'='*60}")
    print(f"  硬冲突样本分析 (前{len(hard_test_indices)}个):")
    print(f"  {'='*60}")
    
    severe_conflict = 0  # 双方高置信度冲突
    
    for pos, idx in enumerate(hard_test_indices):
        true_label = data.y_true[idx].item()
        
        info_line = f"  样本{idx:4d}: "
        b_values = []
        u_values = []
        preds = []
        
        for name in agent_names:
            data_key = name_to_key[name]
            x = data.x_test[data_key][idx:idx+1].to(DEVICE)
            model = models[name]
            alpha, b, u, emb = model.get_output(x)
            b_values.append(b[0].cpu())
            u_values.append(u[0].item())
            preds.append(b[0].argmax().item())
            info_line += f"{name}: pred={preds[-1]} b={b[0][0].item():.2f}/{b[0][1].item():.2f} u={u[0].item():.3f} | "
        
        if preds[0] != preds[1] and u_values[0] < 0.3 and u_values[1] < 0.3:
            severe_conflict += 1
            info_line += " [WARN] 严重证据冲突"
        
        r_val = data.r_test[idx]
        print(f"  {info_line} (r={r_val:.3f}, 真实={true_label})")
    
    print(f"\n  → 严重证据冲突样本: {severe_conflict}/{len(hard_test_indices)} "
          f"({severe_conflict/len(hard_test_indices)*100:.1f}%)")
    
    # 通常分歧分析
    print("\n[5/5] 整体分歧分析...")
    disagreement_idx = print_divergence_analysis(data, agent_results, agent_names)
    
    # 可视化（两种数据对比）
    visualize_predictions(data, agent_results, agent_names)
    plt.savefig('figures/hard_agent_predictions.png', dpi=150, bbox_inches='tight')
    print(f"\n  可视化已保存: figures/hard_agent_predictions.png")
    
    # 保存硬样本索引供后续步骤使用
    np.save('models/hard_test_indices.npy', data.hard_test_indices)
    np.save('models/disagreement_samples_hard.npy', disagreement_idx)
    print(f"\n  硬样本索引已保存: models/hard_test_indices.npy")
    
    print(f"\n{'=' * 60}")
    print("HardConflict 模型训练完成！")
    print("=" * 60)
    
    return data, models, agent_results


if __name__ == '__main__':
    main()