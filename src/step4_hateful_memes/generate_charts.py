import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.plot_utils import setup_chinese_font, setup_plot_style

setup_chinese_font()
setup_plot_style()

FIGURE_DIR = 'figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

EXP_DATA = {
    'agents': [
        {'name': 'Agent1(Claude)', 'accuracy': 58.00, 'f1': 48.78, 'color': '#667EEA'},
        {'name': 'Agent2(Gemini)', 'accuracy': 42.00, 'f1': 50.00, 'color': '#F093FB'},
        {'name': 'Agent3(GPT-4o)', 'accuracy': 51.00, 'f1': 66.67, 'color': '#4ADE80'},
    ],
    'fusion_methods': [
        {'name': 'MajorityVoting', 'accuracy': 49.00, 'f1': 58.54, 'color': '#94A3B8'},
        {'name': 'WeightedAvg', 'accuracy': 51.00, 'f1': 60.80, 'color': '#64748B'},
        {'name': 'DS_Fusion', 'accuracy': 51.00, 'f1': 60.80, 'color': '#475569'},
        {'name': 'GAT_Fusion', 'accuracy': 52.00, 'f1': 62.50, 'color': '#0EA5E9'},
        {'name': 'GAT_EvidenceSwap', 'accuracy': 52.00, 'f1': 53.85, 'color': '#8B5CF6'},
    ],
    'causal_reflection': {
        'before': {'accuracy': 48.33, 'f1': None},
        'after': {'accuracy': 63.33, 'f1': 71.79},
    },
    'agent1_comparison': [
        {'name': 'DeepSeek', 'accuracy': 54.00, 'f1': 36.12, 'color': '#9CA3AF'},
        {'name': 'Claude Sonnet 5', 'accuracy': 58.00, 'f1': 48.78, 'color': '#667EEA'},
    ],
    'conflict_distribution': [
        {'name': '无分歧', 'count': 37, 'percentage': 37, 'color': '#4ADE80'},
        {'name': '简单分歧(2v1)', 'count': 15, 'percentage': 15, 'color': '#FACC15'},
        {'name': '复杂分歧(1v1v1)', 'count': 45, 'percentage': 45, 'color': '#F87171'},
    ],
}

def plot_agent_comparison():
    agents = EXP_DATA['agents']
    names = [a['name'] for a in agents]
    accs = [a['accuracy'] for a in agents]
    f1s = [a['f1'] for a in agents]
    colors = [a['color'] for a in agents]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    bars1 = ax1.bar(names, accs, color=colors, alpha=0.85, width=0.6)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('各LLM Agent独立准确率', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 75)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    bars2 = ax2.bar(names, f1s, color=colors, alpha=0.85, width=0.6)
    ax2.set_ylabel('F1分数 (%)', fontsize=12)
    ax2.set_title('各LLM Agent F1分数', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0, 80)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars2, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(FIGURE_DIR, 'agent_performance_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  生成: {FIGURE_DIR}/agent_performance_comparison.png")

def plot_fusion_comparison():
    methods = EXP_DATA['fusion_methods']
    names = [m['name'] for m in methods]
    accs = [m['accuracy'] for m in methods]
    f1s = [m['f1'] for m in methods]
    colors = [m['color'] for m in methods]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='准确率', color=colors, alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1分数', color=colors, alpha=0.6)
    
    ax.set_ylabel('分数 (%)', fontsize=12)
    ax.set_title('不同共识融合方法性能对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 75)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
                f'{accs[i]:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                f'{f1s[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(FIGURE_DIR, 'fusion_methods_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  生成: {FIGURE_DIR}/fusion_methods_comparison.png")

def plot_causal_reflection():
    reflection = EXP_DATA['causal_reflection']
    labels = ['反思前', '反思后']
    accs = [reflection['before']['accuracy'], reflection['after']['accuracy']]
    f1s = [45.0, reflection['after']['f1']]
    colors = ['#94A3B8', '#0EA5E9']
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='准确率', color=colors, alpha=0.9)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1分数', color=colors, alpha=0.7)
    
    ax.set_ylabel('分数 (%)', fontsize=12)
    ax.set_title('因果反事实反思效果对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    improvement = accs[1] - accs[0]
    ax.annotate(f'+{improvement:.1f}%', 
                xy=(0.5, max(accs) + 5), 
                xytext=(0.5, max(accs) + 12),
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#0EA5E9',
                arrowprops=dict(arrowstyle='->', color='#0EA5E9', lw=2))
    
    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(FIGURE_DIR, 'causal_reflection_effect.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  生成: {FIGURE_DIR}/causal_reflection_effect.png")

def plot_agent1_comparison():
    agents = EXP_DATA['agent1_comparison']
    names = [a['name'] for a in agents]
    accs = [a['accuracy'] for a in agents]
    f1s = [a['f1'] for a in agents]
    colors = [a['color'] for a in agents]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='准确率', color=colors, alpha=0.9)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1分数', color=colors, alpha=0.7)
    
    ax.set_ylabel('分数 (%)', fontsize=12)
    ax.set_title('不同文本专家模型性能对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 70)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(FIGURE_DIR, 'agent1_model_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  生成: {FIGURE_DIR}/agent1_model_comparison.png")

def plot_conflict_distribution():
    conflicts = EXP_DATA['conflict_distribution']
    names = [c['name'] for c in conflicts]
    counts = [c['count'] for c in conflicts]
    percentages = [c['percentage'] for c in conflicts]
    colors = [c['color'] for c in conflicts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars = ax1.bar(names, counts, color=colors, alpha=0.85, width=0.6)
    ax1.set_ylabel('样本数', fontsize=12)
    ax1.set_title('分歧类型分布', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 50)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    wedges, texts, autotexts = ax2.pie(counts, labels=names, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 11})
    ax2.set_title('分歧类型占比', fontsize=14, fontweight='bold', pad=15)
    plt.setp(autotexts, fontweight='bold')
    
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(FIGURE_DIR, 'conflict_distribution.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  生成: {FIGURE_DIR}/conflict_distribution.png")

def plot_overall_comparison():
    all_methods = [
        {'name': 'Agent1(Claude)', 'accuracy': 58.00, 'color': '#667EEA'},
        {'name': 'Agent2(Gemini)', 'accuracy': 42.00, 'color': '#F093FB'},
        {'name': 'Agent3(GPT-4o)', 'accuracy': 51.00, 'color': '#4ADE80'},
        {'name': 'MajorityVoting', 'accuracy': 49.00, 'color': '#94A3B8'},
        {'name': 'DS_Fusion', 'accuracy': 51.00, 'color': '#475569'},
        {'name': 'GAT_Fusion', 'accuracy': 52.00, 'color': '#0EA5E9'},
        {'name': '因果反思', 'accuracy': 63.33, 'color': '#F59E0B'},
    ]
    
    names = [m['name'] for m in all_methods]
    accs = [m['accuracy'] for m in all_methods]
    colors = [m['color'] for m in all_methods]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(names, accs, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('异构多模态框架各组件性能对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 75)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='随机基线(50%)')
    ax.legend(fontsize=11)
    
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(FIGURE_DIR, 'overall_performance_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  生成: {FIGURE_DIR}/overall_performance_comparison.png")

if __name__ == '__main__':
    print("=" * 60)
    print("生成实验结果可视化图表")
    print("=" * 60)
    
    plot_agent_comparison()
    plot_fusion_comparison()
    plot_causal_reflection()
    plot_agent1_comparison()
    plot_conflict_distribution()
    plot_overall_comparison()
    
    print("=" * 60)
    print("所有图表生成完成！")
    print(f"图表保存在: {FIGURE_DIR}/")
    print("=" * 60)