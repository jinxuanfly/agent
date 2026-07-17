"""测试GAT共识引擎在CIFAR-10N规模数据上的表现"""
import sys, os
sys.path.insert(0, 'src/step2')
import torch
import numpy as np

from gat_consensus import ConsensusEngine, global_decision

def test_single_sample():
    """测试单个样本的GAT共识"""
    K = 10
    embed_dim = 128
    N = 3

    agent_outputs = []
    for i in range(N):
        alpha = torch.ones(K) + torch.rand(K) * 5
        S = alpha.sum()
        b = (alpha - 1) / S
        u = K / S
        emb = torch.randn(embed_dim) * 0.5
        agent_outputs.append((alpha, b, u, emb))

    print('--- 单样本测试 ---')
    engine = ConsensusEngine(embed_dim=embed_dim, num_classes=K, hidden_dim=128)
    h = engine.build_state(agent_outputs)
    print(f'h shape: {h.shape}')
    print(f'h stats: mean={h.mean():.4f}, std={h.std():.4f}, max={h.max():.4f}, min={h.min():.4f}')
    
    h_final, n_iters, converged, energy_trace, attn_trace = engine.run(
        h, max_iters=20, tol=1e-4, verbose=True
    )
    print(f'Converged: {converged}, iters: {n_iters}')
    print(f'Energy trace (first 5): {[f"{e:.6f}" for e in energy_trace[:5]]}')
    
    outputs = engine.extract_outputs(h_final)
    decision, global_b, global_u, weights = global_decision(outputs, u_threshold=0.5)
    print(f'Decision: {decision}, global_u: {global_u:.4f}')
    
    has_nan = any(torch.isnan(o[0]).any() for o in outputs)
    print(f'Has NaN: {has_nan}')
    
    return not has_nan and converged


def test_batch_samples(n_samples=100):
    """批量测试"""
    K = 10
    embed_dim = 128
    N = 3

    print(f'\n--- 批量测试 (n={n_samples}) ---')
    correct = 0
    abstain = 0
    nan_count = 0
    converged_count = 0
    
    true_labels = torch.randint(0, K, (n_samples,))
    
    engine = ConsensusEngine(embed_dim=embed_dim, num_classes=K, hidden_dim=128)
    
    for idx in range(n_samples):
        true_label = true_labels[idx].item()
        
        agent_outputs = []
        for i in range(N):
            # 让Agent有合理的信念（接近真实标签）
            alpha = torch.ones(K) * 0.5
            alpha[true_label] += 10 + torch.randn(1).item() * 2
            alpha = alpha.clamp(min=1.0)
            S = alpha.sum()
            b = (alpha - 1) / S
            u = K / S
            emb = torch.randn(embed_dim) * 0.1
            agent_outputs.append((alpha, b, u, emb))
        
        h = engine.build_state(agent_outputs)
        h_final, n_iters, converged, _, _ = engine.run(
            h, max_iters=20, tol=1e-4, verbose=False
        )
        outputs = engine.extract_outputs(h_final)
        
        has_nan = any(torch.isnan(o[0]).any() for o in outputs)
        if has_nan:
            nan_count += 1
            continue
        
        if converged:
            converged_count += 1
        
        decision, _, global_u, _ = global_decision(outputs, u_threshold=0.5)
        if decision == true_label:
            correct += 1
        if decision == -1:
            abstain += 1
    
    print(f'Correct: {correct}/{n_samples} ({correct/n_samples*100:.1f}%)')
    print(f'Abstain: {abstain}/{n_samples} ({abstain/n_samples*100:.1f}%)')
    print(f'NaN count: {nan_count}/{n_samples}')
    print(f'Converged: {converged_count}/{n_samples} ({converged_count/n_samples*100:.1f}%)')


def test_with_cifar10n_embeddings():
    """使用真实的CIFAR-10N嵌入测试"""
    print('\n--- 使用CIFAR-10N真实嵌入测试 ---')
    
    # 检查特征文件是否存在
    feat_dir = 'data/features'
    if not os.path.exists(os.path.join(feat_dir, 'test_resnet18.pt')):
        print('CIFAR-10N特征文件不存在，跳过')
        return
    
    test_rn = torch.load(os.path.join(feat_dir, 'test_resnet18.pt'))
    test_vit = torch.load(os.path.join(feat_dir, 'test_vit_tiny.pt'))
    labels = torch.load(os.path.join(feat_dir, 'labels.pt'))
    test_labels = labels['test_labels']
    
    print(f'ResNet-18 features: {test_rn.shape}')
    print(f'ViT-Tiny features: {test_vit.shape}')
    print(f'Labels: {test_labels.shape}')
    
    # 加载证据头
    heads_path = 'checkpoints/cifar10n/evidence_heads.pt'
    if not os.path.exists(heads_path):
        print('证据头文件不存在，跳过')
        return
    
    heads = torch.load(heads_path, map_location='cpu', weights_only=False)
    
    # 处理100个样本
    B = min(100, len(test_labels))
    
    # 为Agent3生成像素特征
    cifar_dir = 'data/cifar-10-batches-py'
    import pickle
    if os.path.exists(cifar_dir):
        def unpickle(file):
            with open(file, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')
        test_data = unpickle(os.path.join(cifar_dir, 'test_batch'))[b'data']
        test_pixels = torch.FloatTensor(test_data) / 255.0
        torch.manual_seed(42)
        proj = torch.randn(3072, 256) * 0.1
        test_pixel = test_pixels @ proj
        test_pixel = (test_pixel - test_pixel.mean(dim=0)) / (test_pixel.std(dim=0) + 1e-8)
    else:
        test_pixel = torch.randn(10000, 256) * 0.1
    
    # 获取Agent输出
    def get_output(feats, head):
        head.eval()
        with torch.no_grad():
            alpha = head(feats)
            S = alpha.sum(dim=1, keepdim=True)
            K = alpha.shape[1]
            b = (alpha - 1) / S
            u = K / S
            if hasattr(head, 'get_embedding'):
                emb = head.get_embedding(feats)
            else:
                emb = feats
        return alpha, b, u, emb
    
    correct = 0
    abstain = 0
    nan_count = 0
    converged_count = 0
    
    embed_dim = 128
    K = 10
    engine = ConsensusEngine(embed_dim=embed_dim, num_classes=K, hidden_dim=128)
    
    for idx in range(B):
        true_label = test_labels[idx].item()
        
        agent_outputs = []
        for name, feats, head in [
            ('agent1', test_rn[idx:idx+1], heads['agent1']),
            ('agent2', test_vit[idx:idx+1], heads['agent2']),
            ('agent3', test_pixel[idx:idx+1], heads['agent3']),
        ]:
            alpha, b, u, emb = get_output(feats, head)
            agent_outputs.append((alpha[0], b[0], u[0], emb[0]))
        
        h = engine.build_state(agent_outputs)
        
        # 添加数值稳定化
        h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
        
        h_final, n_iters, converged, _, _ = engine.run(
            h, max_iters=20, tol=1e-4, verbose=False
        )
        outputs = engine.extract_outputs(h_final)
        
        has_nan = any(torch.isnan(o[0]).any() for o in outputs)
        if has_nan:
            nan_count += 1
            print(f'  NaN at sample {idx}')
            if nan_count >= 3:
                print(f'  Too many NaNs, stopping...')
                return
        
        if converged:
            converged_count += 1
        
        decision, _, global_u, _ = global_decision(outputs, u_threshold=0.5)
        if decision == true_label:
            correct += 1
        if decision == -1:
            abstain += 1
    
    print(f'Results on {B} CIFAR-10N samples:')
    print(f'  Correct: {correct}/{B} ({correct/B*100:.1f}%)')
    print(f'  Abstain: {abstain}/{B} ({abstain/B*100:.1f}%)')
    print(f'  NaN: {nan_count}/{B}')
    print(f'  Converged: {converged_count}/{B}')


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_single_sample()
    test_batch_samples(n_samples=50)
    
    # 尝试CIFAR-10N真实数据
    sys.path.insert(0, 'src')
    test_with_cifar10n_embeddings()
    
    print('\nAll tests done!')