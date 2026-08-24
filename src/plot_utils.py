"""
统一绘图工具模块
=================
功能：
1. 自动配置matplotlib中文字体
2. 提供统一的配色方案和样式
3. 提供快速保存函数
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# ========== 中文字体配置 ==========
# 尝试多种中文字体，取第一个可用的
_CHINESE_FONTS = [
    'Microsoft YaHei', 'SimHei', 'SimSun', 'DengXian',
    'STSong', 'STKaiti', 'STHeiti', 'Source Han Sans CN',
    'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
]

def _find_chinese_font():
    """查找系统中可用的中文字体"""
    available = set(f.name for f in fm.fontManager.ttflist)
    for font_name in _CHINESE_FONTS:
        if font_name in available:
            return font_name
    # Fallback: 查找任何支持中文的字体
    for f in fm.fontManager.ttflist:
        try:
            # 尝试使用字体渲染中文
            fig, ax = plt.subplots(figsize=(1, 0.5))
            ax.text(0.5, 0.5, '中', fontproperties=fm.FontProperties(family=f.name), fontsize=10)
            plt.close(fig)
            return f.name
        except Exception:
            continue
    return None

def setup_chinese_font():
    """配置matplotlib全局中文字体"""
    font_name = _find_chinese_font()
    if font_name:
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"[plot_utils] 已设置中文字体: {font_name}")
    else:
        print("[plot_utils] 警告: 未找到中文字体，中文可能显示为方框")
    return font_name is not None

# ========== 配色方案 ==========
# 智能体颜色
AGENT_COLORS = {
    'Agent1': '#4C72B0',   # 蓝色
    'Agent2': '#DD8452',   # 橙色
    'Agent3': '#55A868',   # 绿色
    'Agent1_x': '#4C72B0',
    'Agent2_y': '#DD8452',
    'Agent3_r': '#55A868',
    'TextBERT': '#4C72B0',
    'ImageRN': '#DD8452',
    'FusionMLP': '#55A868',
    'Agent1_Text': '#4C72B0',
    'Agent2_Image': '#DD8452',
    'Agent3_Fusion': '#55A868',
}

# 方法颜色
METHOD_COLORS = {
    'Majority Voting': '#E74C3C',
    'Weighted Avg': '#3498DB',
    'DS Fusion': '#2ECC71',
    'GAT+DS': '#9B59B6',
    'GAT+EMNet': '#F39C12',
    'Ours (Full)': '#E74C3C',
    'DS_Fusion': '#2ECC71',
    'GAT_DS_Fusion': '#9B59B6',
    'GAT_EMNet_Fusion': '#F39C12',
}

# 方法显示名称映射
METHOD_DISPLAY_NAMES = {
    'Majority Voting': '多数投票',
    'Weighted Avg': '加权平均',
    'DS Fusion': 'DS融合',
    'GAT+DS': 'GAT+DS',
    'GAT+EMNet': 'GAT+EMNet',
    'DS_Fusion': 'DS融合',
    'GAT_DS_Fusion': 'GAT+DS',
    'GAT_EMNet_Fusion': 'GAT+EMNet',
}

def save_figure(fig, path, dpi=150, tight=True):
    """统一保存图片"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = {'dpi': dpi, 'bbox_inches': 'tight'} if tight else {'dpi': dpi}
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"  图片已保存: {path}")

def setup_plot_style():
    """设置统一样式"""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })