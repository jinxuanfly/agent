"""
异构多模态动态共识与协同框架
=============================
主入口：通过命令行参数选择运行阶段

用法：
    python main.py --step 1          # 合成数据与单智能体证据网络
    python main.py --step 2          # 内循环共识（GAT）
    python main.py --step 3          # 分歧解构器 + 简单纠偏
    python main.py --step 4          # CIFAR-10N 完整评估
    python main.py --step 5          # 因果反事实反思
    python main.py --step 6          # Hateful Memes 多模态评估
    python main.py --step 4 --ablate # CIFAR-10N 消融实验
    python main.py --step 4 --detail # CIFAR-10N 详细分析
    python main.py --all             # 运行所有步骤
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


def step4(num_samples=500, ablate=False, detail=False):
    """CIFAR-10N 端到端评估"""
    from step4.train_heads import train_all_heads
    from step4.evaluate_cifar10n import evaluate, ablation_study, detailed_analysis
    
    print("=" * 70)
    print("步骤4：CIFAR-10N 真实多模态数据评估")
    print("=" * 70)
    
    # 检查证据头是否存在
    heads_path = 'checkpoints/cifar10n/evidence_heads.pt'
    if not os.path.exists(heads_path):
        print(f"\n证据头未找到，准备训练...")
        # 检查特征是否存在
        feat_dir = 'data/features'
        feat_files = ['train_resnet18.pt', 'test_resnet18.pt', 'train_vit_tiny.pt', 
                      'test_vit_tiny.pt', 'labels.pt']
        all_feats_exist = all(os.path.exists(os.path.join(feat_dir, f)) for f in feat_files)
        
        if not all_feats_exist:
            print("特征未找到，请先运行特征提取...")
            print("运行: python -m src.step4.extract_features")
            print("或使用: python main.py --step 4 --setup")
            return
        
        # 训练证据头
        print("\n训练证据头...")
        train_all_heads()
    
    print(f"\n端到端评估 (n={num_samples}):")
    evaluate(num_test_samples=num_samples)
    
    if ablate:
        print("\n消融实验:")
        ablation_study(num_test_samples=min(num_samples, 200))
    
    if detail:
        print("\n详细分析:")
        detailed_analysis(num_samples=min(num_samples, 50))
    
    print("\n步骤4完成！检查 figures/ 和 results/cifar10n/ 目录。\n")


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


def diagnose_consensus():
    """DS_Consensus == DS_Fusion 根因诊断"""
    from step4.diagnose_consensus_root_cause import main as diag_main
    
    print("=" * 70)
    print("诊断：DS_Consensus == DS_Fusion 根因分析")
    print("=" * 70)
    
    diag_main()
    print("\n诊断完成！检查 results/cifar10n/diagnose_consensus_root_cause.json\n")


def main():
    parser = argparse.ArgumentParser(
        description='异构多模态动态共识与协同框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --step 1         # 合成数据+训练
    python main.py --step 2         # GAT共识
    python main.py --step 3         # 分歧解构
    python main.py --step 4         # CIFAR-10N评估
    python main.py --step 4 --detail  # CIFAR-10N详细分析
    python main.py --step 4 --ablate   # CIFAR-10N消融实验
    python main.py --step 6         # Hateful Memes评估
    python main.py --all            # 全部运行
        """
    )
    
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5, 6], 
                       help='运行指定步骤')
    parser.add_argument('--all', action='store_true',
                       help='运行所有步骤')
    parser.add_argument('--ablate', action='store_true',
                       help='运行消融实验（步骤4）')
    parser.add_argument('--detail', action='store_true',
                       help='运行详细分析（步骤4）')
    parser.add_argument('--samples', type=int, default=500,
                       help='评估样本数（步骤4/6）')
    parser.add_argument('--hateful_samples', type=int, default=2000,
                       help='Hateful Memes训练样本数（步骤6）')
    parser.add_argument('--setup', action='store_true',
                       help='设置数据环境（步骤4）')
    parser.add_argument('--enhanced', action='store_true',
                       help='使用增强版（步骤6 Hateful Memes）')
    parser.add_argument('--diagnose', action='store_true',
                       help='DS_Consensus == DS_Fusion 根因诊断')
    
    args = parser.parse_args()
    
    # 固定随机种子
    import numpy as np
    import torch
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    if args.setup:
        print("设置CIFAR-10N数据环境...")
        from step4.extract_features import extract_and_save
        extract_and_save()
        from step4.train_heads import main as train_heads_main
        train_heads_main()
        print("数据环境设置完成！")
        return
    
    if args.all:
        print("运行所有步骤...\n")
        step1()
        step2()
        step3()
        step4(num_samples=args.samples, ablate=True, detail=True)
        step5()
        print("所有步骤完成！\n")
    elif args.step == 1:
        step1()
    elif args.step == 2:
        step2()
    elif args.step == 3:
        step3()
    elif args.step == 4:
        step4(num_samples=args.samples, ablate=args.ablate, detail=args.detail)
    elif args.step == 5:
        step5()
    elif args.step == 6:
        step6(hateful_samples=args.hateful_samples, val_samples=args.samples, 
              enhanced=args.enhanced)
    elif args.diagnose:
        diagnose_consensus()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()