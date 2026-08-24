# -*- coding: utf-8 -*-
"""
生成论文图表
基于500样本实验结果和消融实验结果
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'hateful_memes')
ABLATION_DIR = os.path.join(RESULTS_DIR, 'ablation')
FIGS_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'paper')
os.makedirs(FIGS_DIR, exist_ok=True)

# 配色
COLORS = {
    'agent': '#4C72B0',
    'baseline': '#A0A0A0',
    'ds': '#55A868',
    'innovation': '#C44E52',
    'gat': '#8172B2',
    'best': '#CCB974',
}


# =============================================================
# 图1: 融合方法对比（主结果）
# =============================================================
def fig1_main_comparison():
    """主结果：各融合方法准确率对比"""
    print("[图1] 融合方法对比（主结果）...")

    methods = [
        ('Agent1 (GPT-5)', 55.60, 40.32, 'agent'),
        ('Agent2 (Gemini)', 79.60, 80.75, 'best'),
        ('Agent3 (GPT-5)', 55.00, 25.74, 'agent'),
        ('Majority Voting', 60.00, 43.82, 'baseline'),
        ('Weighted Avg', 66.80, 58.50, 'baseline'),
        ('DS Fusion', 67.80, 60.25, 'ds'),
        ('Corr-Aware DS', 67.00, 59.46, 'ds'),
        ('Uncertainty-Weighted DS', 69.20, 63.16, 'innovation'),
        ('GAT + DS', 67.60, 60.10, 'gat'),
        ('GAT + EvidenceSwap', 70.40, 65.58, 'gat'),
    ]

    names = [m[0] for m in methods]
    accs = [m[1] for m in methods]
    f1s = [m[2] for m in methods]
    colors = [COLORS[m[3]] for m in methods]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score (%)', color=colors, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')

    # Gemini基线
    ax.axhline(y=79.60, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Best Single Agent (Gemini)')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Comparison of Fusion Methods on Hateful Memes (500 samples)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 95)

    # 标注数值
    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, 'fig1_main_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  保存: {out_path}")


# =============================================================
# 图2: 消融实验 - Symbolic GAT贡献
# =============================================================
def fig2_ablation_gat():
    print("\n[图2] 消融: Symbolic GAT贡献...")

    methods = ['No GAT\n(DS Baseline)', 'Random GAT\n+ EvidenceSwap', 'Symbolic GAT\n+ EvidenceSwap']
    accs = [67.80, 69.60, 69.00]
    f1s = [66.59, 68.80, 68.03]
    colors = [COLORS['baseline'], COLORS['gat'], COLORS['gat']]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color=colors, alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score (%)', color=colors, alpha=0.4, edgecolor='black', hatch='//')

    ax.set_xlabel('GAT Configuration', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Ablation: Symbolic GAT vs Random GAT vs No GAT', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(60, 75)

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)

    # 标注提升
    ax.annotate(f'+1.2%', xy=(0, 67.80), xytext=(0.5, 72),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=11, color='green', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, 'fig2_ablation_gat.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  保存: {out_path}")


# =============================================================
# 图3: 消融实验 - Uncertainty_Weighted_DS贡献（核心创新）
# =============================================================
def fig3_ablation_uw():
    print("\n[图3] 消融: Uncertainty_Weighted_DS贡献（核心创新）...")

    methods = [
        'DS\n(equal weight)',
        'DS\n(train acc weight)',
        'Corr-Aware DS\n(ds=0.5)',
        'Uncertainty-Weighted\nDS (s=10)',
        'Uncertainty-Weighted\nDS (s=20)',
    ]
    accs = [67.80, 66.00, 66.80, 68.80, 69.20]
    f1s = [66.59, 64.46, 65.58, 67.89, 68.35]
    colors = [COLORS['ds'], COLORS['baseline'], COLORS['ds'], COLORS['innovation'], COLORS['innovation']]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color=colors, alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score (%)', color=colors, alpha=0.4, edgecolor='black', hatch='//')

    ax.set_xlabel('Fusion Method', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Ablation: Uncertainty-Weighted DS vs Baselines (Core Innovation)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(60, 75)

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 标注核心提升
    ax.annotate(f'+1.40%', xy=(4, 69.20), xytext=(3.5, 73),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=12, color='red', fontweight='bold')
    ax.annotate(f'+3.20%', xy=(4, 69.20), xytext=(2.5, 72),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                fontsize=12, color='darkred', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, 'fig3_ablation_uncertainty_weighted.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  保存: {out_path}")


# =============================================================
# 图4: 分歧样本上的表现（难样本分析）
# =============================================================
def fig4_disagreement_analysis():
    print("\n[图4] 分歧样本分析...")

    methods = [
        'DS (equal)',
        'DS (train acc)',
        'Corr-Aware DS',
        'Unc-Weighted DS (s=10)',
        'Unc-Weighted DS (s=20)',
    ]
    all_acc = [67.80, 66.00, 66.80, 68.80, 69.20]
    disagree_acc = [54.84, 51.61, 53.05, 56.63, 57.35]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, all_acc, width, label='All Samples Acc (%)', color='#4C72B0', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, disagree_acc, width, label='Disagreement Samples Acc (%)', color='#C44E52', alpha=0.85, edgecolor='black', hatch='//')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Performance on Disagreement Samples (Hard Cases, n=279/500)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(45, 75)

    for bar, val in zip(bars1, all_acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, disagree_acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)

    # 标注分歧样本提升
    ax.annotate(f'+2.51%', xy=(4, 57.35), xytext=(3.5, 65),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=12, color='red', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, 'fig4_disagreement_analysis.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  保存: {out_path}")


# =============================================================
# 图5: Agent能力与相关性矩阵
# =============================================================
def fig5_agent_analysis():
    print("\n[图5] Agent能力与相关性分析...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：Agent准确率
    ax1 = axes[0]
    agents = ['Agent1\n(GPT-5, Text)', 'Agent2\n(Gemini, Image)', 'Agent3\n(GPT-5, Multimodal)']
    train_acc = [56.0, 47.5, 52.5]
    val_acc = [55.6, 79.6, 55.0]

    x = np.arange(len(agents))
    width = 0.35
    ax1.bar(x - width/2, train_acc, width, label='Train Acc (%)', color='#4C72B0', alpha=0.85, edgecolor='black')
    ax1.bar(x + width/2, val_acc, width, label='Val Acc (%)', color='#CCB974', alpha=0.85, edgecolor='black')
    ax1.set_xlabel('Agent', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Agent Capabilities', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    for i, (t, v) in enumerate(zip(train_acc, val_acc)):
        ax1.text(i - width/2, t + 1, f'{t:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 右：相关性矩阵热力图
    ax2 = axes[1]
    corr = np.array([
        [1.000, 0.224, 0.651],
        [0.224, 1.000, 0.246],
        [0.651, 0.246, 1.000],
    ])
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=['Agent1\n(GPT-5)', 'Agent2\n(Gemini)', 'Agent3\n(GPT-5)'],
                yticklabels=['Agent1', 'Agent2', 'Agent3'],
                ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title('Agent Correlation Matrix', fontsize=12, fontweight='bold')

    # 标注高相关
    ax2.annotate('High corr\n(same model family)', xy=(2, 0), xytext=(2.3, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, 'fig5_agent_analysis.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  保存: {out_path}")


# =============================================================
# 图6: Uncertainty权重分布（核心创新的直观解释）
# =============================================================
def fig6_uncertainty_weights():
    print("\n[图6] Uncertainty权重分布（核心创新解释）...")

    # 加载验证集不确定性数据
    import torch
    val_u = []
    for i in range(3):
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints', 'hateful_memes',
                                        f'llm_val_agent{i}.pt'),
                          map_location='cpu', weights_only=False)
        val_u.append(ckpt['uncertainties'].numpy())

    val_u = np.array(val_u)  # [3, 500]
    confidence = 1 - val_u  # [3, 500]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：各Agent不确定性分布
    ax1 = axes[0]
    labels = ['Agent1 (GPT-5)', 'Agent2 (Gemini)', 'Agent3 (GPT-5)']
    colors = ['#4C72B0', '#CCB974', '#55A868']
    for i in range(3):
        ax1.hist(val_u[i], bins=30, alpha=0.6, label=labels[i], color=colors[i], edgecolor='black')
    ax1.axvline(x=np.mean(val_u[0]), color=colors[0], linestyle='--', alpha=0.8)
    ax1.axvline(x=np.mean(val_u[1]), color=colors[1], linestyle='--', alpha=0.8)
    ax1.axvline(x=np.mean(val_u[2]), color=colors[2], linestyle='--', alpha=0.8)
    ax1.set_xlabel('Uncertainty (u)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Uncertainty Distribution per Agent', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # 右：sharpness对权重分配的影响
    ax2 = axes[1]
    sharpness = np.array([0.5, 1, 2, 3, 5, 10, 20, 50])
    # 模拟softmax权重（基于平均u）
    mean_conf = 1 - np.mean(val_u, axis=1)  # [3]
    weights_per_sharp = []
    for s in sharpness:
        scaled = mean_conf * s
        scaled = scaled - scaled.max()
        exp_v = np.exp(scaled)
        w = exp_v / exp_v.sum()
        weights_per_sharp.append(w)
    weights_per_sharp = np.array(weights_per_sharp)  # [n_sharp, 3]

    ax2.plot(sharpness, weights_per_sharp[:, 0], 'o-', label='Agent1 (GPT-5)', color=colors[0], linewidth=2)
    ax2.plot(sharpness, weights_per_sharp[:, 1], 's-', label='Agent2 (Gemini)', color=colors[1], linewidth=2)
    ax2.plot(sharpness, weights_per_sharp[:, 2], '^-', label='Agent3 (GPT-5)', color=colors[2], linewidth=2)
    ax2.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5, label='Equal weight (1/3)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Sharpness Parameter', fontsize=11)
    ax2.set_ylabel('Average Weight', fontsize=11)
    ax2.set_title('Weight Allocation vs Sharpness', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    # 标注最佳sharpness=20
    ax2.axvline(x=20, color='red', linestyle='--', alpha=0.5)
    ax2.annotate('Best (s=20)', xy=(20, 0.5), xytext=(30, 0.6),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, 'fig6_uncertainty_weights.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  保存: {out_path}")


# =============================================================
# 图7: 综合性能雷达图
# =============================================================
def fig7_radar_chart():
    print("\n[图7] 综合性能雷达图...")

    # 5个维度：全样本Acc, F1, 分歧Acc, ECE(反向), 拒绝率(反向)
    categories = ['Accuracy', 'F1 Score', 'Disagreement Acc', 'Calibration (1-ECE)', 'Coverage (1-Rej)']

    methods_data = {
        'DS Fusion': [67.80, 60.25, 51.05, 73.32, 100.0],
        'Uncertainty-Weighted DS': [69.20, 63.16, 53.97, 73.25, 100.0],
        'GAT + EvidenceSwap': [70.40, 65.58, 56.49, 75.54, 100.0],
        'Best Agent (Gemini)': [79.60, 80.75, 75.73, 100.0, 100.0],
    }

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    colors = ['#55A868', '#C44E52', '#8172B2', '#CCB974']
    for (name, values), color in zip(methods_data.items(), colors):
        values_norm = [v / 100 * 100 for v in values]  # 已是百分比
        values_closed = values_norm + values_norm[:1]
        ax.plot(angles, values_closed, 'o-', linewidth=2, label=name, color=color, markersize=6)
        ax.fill(angles, values_closed, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(20, 100)
    ax.set_yticks([30, 50, 70, 90])
    ax.set_yticklabels(['30', '50', '70', '90'], fontsize=9)
    ax.set_rlabel_position(45)
    ax.grid(True, alpha=0.3)
    ax.set_title('Multi-dimensional Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, 'fig7_radar_chart.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  保存: {out_path}")


# =============================================================
# 主函数
# =============================================================
def main():
    print("=" * 80)
    print("生成论文图表")
    print("=" * 80)

    fig1_main_comparison()
    fig2_ablation_gat()
    fig3_ablation_uw()
    fig4_disagreement_analysis()
    fig5_agent_analysis()
    fig6_uncertainty_weights()
    fig7_radar_chart()

    print("\n" + "=" * 80)
    print(f"所有图表已生成，保存至: {FIGS_DIR}")
    print("=" * 80)
    print("图表列表:")
    for f in sorted(os.listdir(FIGS_DIR)):
        if f.endswith('.png'):
            path = os.path.join(FIGS_DIR, f)
            size = os.path.getsize(path) / 1024
            print(f"  {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()
