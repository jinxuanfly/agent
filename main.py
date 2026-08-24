"""
异构多模态动态共识与协同框架
=============================
主入口：通过命令行参数选择运行阶段

用法：
    python main.py --step 1          # 合成数据与单智能体证据网络
    python main.py --step 2          # 内循环共识（GAT）
    python main.py --step 3          # 分歧解构器 + 简单纠偏
    python main.py --step 5          # 因果反事实反思
    python main.py --step 6          # Hateful Memes 多模态评估
    python main.py --all             # 运行所有步骤

注：CIFAR-10N（原 --step 4）已废弃，相关代码移至 CIFAR-10N废弃/ 目录。
原因：CIFAR-10N 为单模态图像分类，与论文"异构多模态"核心定位冲突。
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 确保能导入src模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

SEED = 42


def step1():
    """合成数据生成 + 单智能体训练"""
    from step1.train_hard_models import main as train_main
    from step1.synthetic_data import main as data_main
    
    print("=" * 70)
    print("步骤1：合成数据与单智能体证据网络")
    print("=" * 70)
    
    print("\n[1/2] 生成合成数据...")
    data_main()
    print("\n[2/2] 训练证据网络...")
    train_main()
    
    print("\n步骤1完成！检查 models/ 目录下的 .pth 文件。\n")


def step2():
    """内循环共识（GAT）"""
    from step2.gat_consensus import test_on_disagreement_samples
    
    print("=" * 70)
    print("步骤2：内循环共识（不确定性感知GAT）")
    print("=" * 70)
    
    test_on_disagreement_samples()
    print("\n步骤2完成！检查 figures/ 目录下能量曲线和注意力矩阵。\n")


def step3():
    """分歧解构器 + 简单纠偏"""
    from step3.disagreement_resolver import test_on_synthetic
    
    print("=" * 70)
    print("步骤3：分歧解构器 + 简单纠偏")
    print("=" * 70)
    
    test_on_synthetic()
    print("\n步骤3完成！\n")


def step5():
    """因果反事实反思"""
    # 目前依赖step1的合成数据
    print("=" * 70)
    print("步骤5：因果反事实反思")
    print("=" * 70)
    
    # 检查模型是否存在（兼容 _hard 后缀）
    model_dir = 'models'
    required_base = ['Agent1_x_evidential', 'Agent2_y_evidential', 'Agent3_r_evidential']
    
    found = True
    for base in required_base:
        if not (os.path.exists(os.path.join(model_dir, base + '.pth')) or 
                os.path.exists(os.path.join(model_dir, base + '_hard.pth'))):
            found = False
            break
    
    if not found:
        print("合成数据模型未找到，请先运行步骤1")
        print("运行: python main.py --step 1")
        return
    
    from step5.causal_reflection import test_causal_reflection
    test_causal_reflection()
    print("\n步骤5完成！检查 figures/causal_*.png 文件。\n")


def step6(hateful_samples=2000, val_samples=500, enhanced=False):
    """Hateful Memes 多模态评估"""
    if enhanced:
        from step4_hateful_memes.evaluate_hateful_memes_enhanced import run_pipeline_enhanced
        
        print("=" * 70)
        print("步骤6-增强版：Hateful Memes 多模态评估（完整实验）")
        print("=" * 70)
        
        run_pipeline_enhanced(max_train=hateful_samples, max_val=val_samples)
        print("\n步骤6-增强版完成！检查 figures/ 和 results/hateful_memes/ 目录。\n")
    else:
        from step4.evaluate_hateful_memes import run_pipeline
        
        print("=" * 70)
        print("步骤6：Hateful Memes 多模态评估（基础版）")
        print("=" * 70)
        
        data_dir = 'data/Hateful_Memes/data'
        if not os.path.exists(data_dir):
            print(f"数据目录 {data_dir} 不存在！")
            print("请先下载 Hateful Memes 数据集并解压到 data/Hateful_Memes/")
            print("Hateful Memes 下载地址：https://www.kaggle.com/datasets/parthplc/facebook-hateful-memes")
            return
        
        run_pipeline(max_train=hateful_samples, max_val=val_samples)
        print("\n步骤6完成！检查 figures/ 和 results/hateful_memes/ 目录。\n")


def main():
    parser = argparse.ArgumentParser(
        description='异构多模态动态共识与协同框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --step 1         # 合成数据+训练
    python main.py --step 2         # GAT共识
    python main.py --step 3         # 分歧解构
    python main.py --step 5         # 因果反思
    python main.py --step 6         # Hateful Memes评估
    python main.py --all            # 全部运行
        """
    )
    
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 5, 6],
                       help='运行指定步骤')
    parser.add_argument('--all', action='store_true',
                       help='运行所有步骤')
    parser.add_argument('--samples', type=int, default=500,
                       help='评估样本数（步骤6）')
    parser.add_argument('--hateful_samples', type=int, default=2000,
                       help='Hateful Memes训练样本数（步骤6）')
    parser.add_argument('--enhanced', action='store_true',
                       help='使用增强版（步骤6 Hateful Memes）')
    
    args = parser.parse_args()
    
    # 固定随机种子
    import numpy as np
    import torch
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    if args.all:
        print("运行所有步骤...\n")
        step1()
        step2()
        step3()
        step5()
        print("所有步骤完成！\n")
    elif args.step == 1:
        step1()
    elif args.step == 2:
        step2()
    elif args.step == 3:
        step3()
    elif args.step == 5:
        step5()
    elif args.step == 6:
        step6(hateful_samples=args.hateful_samples, val_samples=args.samples,
              enhanced=args.enhanced)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()