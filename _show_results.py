"""显示Hateful Memes评估结果 + 自动生成对比图"""
import json
import os
import sys
import numpy as np

# 导入统一绘图工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.plot_utils import setup_chinese_font, setup_plot_style, AGENT_COLORS, METHOD_COLORS, save_figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置中文字体
setup_chinese_font()
setup_plot_style()

# ========== 智能体结果分析 ==========
def analyze_agent_results(agent_preds_dict, y_true, agent_names=None):
    """分析各智能体的预测结果"""
    from sklearn.metrics import accuracy_score, f1_score
    
    if agent_names is None:
        agent_names = list(agent_preds_dict.keys())
    
    results = {}
    for name in agent_names:
        if name in agent_preds_dict:
            preds = np.array(agent_preds_dict[name])
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average='binary', zero_division=0)
            results[name] = {'acc': acc, 'f1': f1}
    return results

# ========== 旧版评估结果 ==========
old_path = 'results/hateful_memes/evaluation_results.json'
enhanced_path = 'results/hateful_memes/evaluation_results_enhanced.json'
details_path = 'results/hateful_memes/evaluation_details_enhanced.json'

if os.path.exists(old_path):
    with open(old_path) as f:
        old_results = json.load(f)
    
    print('=' * 70)
    print('Hateful Memes 评估结果 (旧版: 2000训练/500验证)')
    print('=' * 70)
    print(f'{"方法":<20} {"Acc%":<8} {"F1%":<8} {"ECE":<8} {"Rej%":<8}')
    print('-' * 60)
    for method, metrics in old_results.items():
        print(f'{method:<20} {metrics["accuracy"]:<8.2f} {metrics["f1"]:<8.2f} {metrics["ece"]:<8.4f} {metrics["rejection_rate"]:<8.2f}')
    print('-' * 60)
    print()

# ========== 增强版评估结果 ==========
if os.path.exists(enhanced_path):
    with open(enhanced_path) as f:
        enhanced_results = json.load(f)
    
    print('=' * 70)
    print('Hateful Memes 增强评估结果 (8500训练/500验证 + GAT共识层 + 分歧解构 + 证据交换)')
    print('=' * 70)
    print(f'{"方法":<20} {"Acc%":<8} {"F1%":<8} {"ECE":<8} {"Rej%":<8}')
    print('-' * 60)
    for method, metrics in enhanced_results.items():
        print(f'{method:<20} {metrics["accuracy"]:<8.2f} {metrics["f1"]:<8.2f} {metrics["ece"]:<8.4f} {metrics["rejection_rate"]:<8.2f}')
    print('-' * 60)
    print()
    
    # ========== 生成对比图 ==========
    os.makedirs('figures', exist_ok=True)
    
    # 1. 主对比图 (Acc + F1)
    methods = list(enhanced_results.keys())
    accs = [enhanced_results[m]['accuracy'] for m in methods]
    f1s = [enhanced_results[m]['f1'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(methods))
    bars1 = ax1.bar(x, accs, color=colors, alpha=0.85, width=0.6)
    ax1.set_title('Hateful Memes 各方法准确率对比', fontsize=13)
    ax1.set_ylabel('准确率 (%)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='随机猜测基线 (50%)')
    ax1.legend(fontsize=9)
    for bar, v in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    bars2 = ax2.bar(x, f1s, color=colors, alpha=0.85, width=0.6)
    ax2.set_title('Hateful Memes 各方法F1对比', fontsize=13)
    ax2.set_ylabel('F1 分数 (%)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
    for bar, v in zip(bars2, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, 'figures/hateful_memes_enhanced_comparison.png')
    
    # ========== 分歧分析 ==========
    if os.path.exists(details_path):
        with open(details_path) as f:
            details = json.load(f)
        
        conflict_types = details.get('conflict_types', [])
        if conflict_types:
            evidence_count = sum(1 for c in conflict_types if c == 'evidence_conflict')
            ignorance_count = sum(1 for c in conflict_types if c == 'ignorance_conflict')
            none_count = sum(1 for c in conflict_types if c == 'none')
            K_values = details.get('K_values', [])
            avg_K = sum(K_values) / len(K_values) if K_values else 0
            over_half = sum(1 for k in K_values if k > 0.5) if K_values else 0
            
            print('=' * 70)
            print('分歧解构分析 (增强版)')
            print('=' * 70)
            print(f'  无分歧样本:   {none_count}/{len(conflict_types)} ({none_count/len(conflict_types)*100:.1f}%)')
            print(f'  证据冲突样本: {evidence_count}/{len(conflict_types)} ({evidence_count/len(conflict_types)*100:.1f}%)')
            print(f'  无知冲突样本: {ignorance_count}/{len(conflict_types)} ({ignorance_count/len(conflict_types)*100:.1f}%)')
            print(f'  平均冲突系数K: {avg_K:.4f}')
            print(f'  K > 0.5样本:  {over_half}/{len(K_values)} ({over_half/len(K_values)*100:.1f}%)')
        
        # 共识迭代
        n_iters = details.get('n_iters', 'N/A')
        energy = details.get('energy_history', [])
        if energy:
            print(f'  共识迭代次数: {n_iters}')
            print(f'  初始能量:     {energy[0]:.6f}')
            print(f'  最终能量:     {energy[-1]:.6f}')
        print()
        
        # 2. 分歧类型分布饼图
        if conflict_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            labels = ['无分歧', '证据冲突', '无知冲突']
            sizes = [none_count, evidence_count, ignorance_count]
            colors_pie = ['#2ECC71', '#E74C3C', '#F39C12']
            explode = (0, 0.05, 0.05)
            
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', shadow=False, startangle=90)
            ax1.set_title('分歧类型分布', fontsize=13)
            
            # 3. K值分布直方图
            if K_values:
                ax2.hist(K_values, bins=20, color='#3498DB', alpha=0.7, edgecolor='white')
                ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='K=0.5阈值')
                ax2.set_title('冲突系数K分布', fontsize=13)
                ax2.set_xlabel('K值', fontsize=11)
                ax2.set_ylabel('样本数', fontsize=11)
                ax2.legend(fontsize=9)
            
            plt.tight_layout()
            save_figure(fig, 'figures/hateful_memes_conflict_analysis.png')
        
        # 4. 不确定性分布图
        unc_keys = {
            'DS_Fusion_uncertainty': 'DS融合',
            'GAT_DS_Fusion_uncertainty': 'GAT+DS',
            'GAT_EMNet_Fusion_uncertainty': 'GAT+证据交换'
        }
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, label in unc_keys.items():
            u_vals = details.get(key, [])
            if u_vals:
                u_arr = np.array(u_vals)
                # 过滤异常值
                u_arr = u_arr[u_arr < 10]
                if len(u_arr) > 0:
                    ax.hist(u_arr, bins=30, alpha=0.5, label=f'{label} (均值={np.mean(u_arr):.3f})')
        
        ax.set_title('不同融合方法的不确定性分布对比', fontsize=13)
        ax.set_xlabel('不确定性 u', fontsize=11)
        ax.set_ylabel('样本数', fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        save_figure(fig, 'figures/hateful_memes_uncertainty_dist.png')

# ========== 图表文件位置 ==========
print('=' * 70)
print('图表文件位置:')
print('=' * 70)
print('  旧版对比图:         figures/hateful_memes_comparison.png')
print('  增强对比图:         figures/hateful_memes_enhanced_comparison.png')
print('  分歧分析图:         figures/hateful_memes_conflict_analysis.png')
print('  不确定性分布图:     figures/hateful_memes_uncertainty_dist.png')
print()

# ========== 诊断信息 ==========
print('=' * 70)
print('综合分析与诊断 (增强版)')
print('=' * 70)
print()
print('1. 智能体基线准确率 (Acc/F1):')
print('   Agent1(CharCNN文本):  55.0%/43.6%')
print('   Agent2(ResNet18图像): 55.4%/41.5%')
print('   Agent3(跨模态融合):   56.6%/42.1%')
print('   -> 三个Agent都在55%左右 (Hateful Memes极难，随机为50%)')
print()
print('2. 增强融合方法对比:')
if os.path.exists(enhanced_path):
    base = enhanced_results.get('WeightedAvg', {})
    ds = enhanced_results.get('DS_Fusion', {})
    gat = enhanced_results.get('GAT_DS_Fusion', {})
    emnet = enhanced_results.get('GAT_EMNet_Fusion', {})
    print(f'   加权平均     Acc={base.get("accuracy",0):.1f}% F1={base.get("f1",0):.1f}%')
    print(f'   DS融合       Acc={ds.get("accuracy",0):.1f}% F1={ds.get("f1",0):.1f}%')
    print(f'   GAT+DS       Acc={gat.get("accuracy",0):.1f}% F1={gat.get("f1",0):.1f}%')
    print(f'   GAT+证据交换 Acc={emnet.get("accuracy",0):.1f}% F1={emnet.get("f1",0):.1f}%')
print()
print('3. 共识层效果:')
print('   共识迭代 ~7次收敛')
print('   共识后平均u从0.31升至0.46 (良性增加：分歧样本的不确定性上升)')
print()
print('4. 分歧解构 (500验证样本):')
if os.path.exists(details_path):
    with open(details_path) as f:
        dt = json.load(f)
    ct = dt.get('conflict_types', [])
    K = dt.get('K_values', [])
    ev = sum(1 for c in ct if c == 'evidence_conflict')
    ig = sum(1 for c in ct if c == 'ignorance_conflict')
    no = sum(1 for c in ct if c == 'none')
    print(f'   无分歧: {no}/{len(ct)} ({no/len(ct)*100:.1f}%)')
    print(f'   证据冲突: {ev}/{len(ct)} ({ev/len(ct)*100:.1f}%)')
    print(f'   无知冲突: {ig}/{len(ct)} ({ig/len(ct)*100:.1f}%)')
    avgK = sum(K)/len(K) if K else 0
    print(f'   平均K={avgK:.4f}')
print()
print('5. 核心困难:')
print('   - Hateful Memes数据集本身极难 (SOTA仅~75%)')
print('   - 各Agent仅~55%导致DS融合提升有限')
print('   - EMNet在弱Agent场景下无法学习有效映射，改用确定性证据交换')
print('   - 共识层提升Agent多样性但未能提升准确率')
print()
print('6. 后续改进方向:')
print('   - 使用预训练语言模型(BERT/RoBERTa)替代CharCNN提升文本Agent')
print('   - 使用更强大的图像Model(ViT-B/16)替代ResNet18')
print('   - 扩大训练数据到完整训练集(~12000样本)')
print('   - 融合Agent使用微调后的跨模态预训练模型(OFA/ALBEF)')