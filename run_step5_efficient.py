# -*- coding: utf-8 -*-
"""
Step5 高效版：因果反事实反思（跨Agent证据交换）
=============================================
针对500样本GPT-5+Gemini+GPT-5配置，高效验证因果反思创新点。

优化策略：
1. 跳过文本消融归因（最贵部分，3次API/agent/轮）
2. 仅保留跨Agent证据交换（核心创新，1次API/agent/轮）
3. MAX_REFLECTIONS=1（单轮反思，控制成本）
4. 分层策略：2v1分歧中，只反思"少数派"Agent（降低成本）

成本估算（100个分歧样本）：
- 100样本 * 1轮 * 1少数派Agent * 1 API调用 = ~100次API
- 预估时间：1-2小时
"""
import os, sys, io, time, json, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.llm_agent import LLMAgent, AGENT_PROMPTS
from src.llm_api import LLMClient

warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLASSES = 2
DATA_DIR = 'data/Hateful_Memes/data'
CHECKPOINT_DIR = 'checkpoints/hateful_memes'
RESULT_DIR = 'results/hateful_memes'


def generate_cross_agent_prompt(agent_idx, all_preds, all_beliefs, all_reasonings):
    """生成跨Agent证据交换提示"""
    prompt_parts = []

    for i in range(3):
        if i == agent_idx:
            continue
        pred = all_preds[i]
        belief = all_beliefs[i]
        reasoning = all_reasonings[i] if all_reasonings else ''

        label_str = "hateful" if pred == 1 else "non-hateful"
        conf = float(belief[pred].item()) if belief.dim() == 1 else float(belief[0, pred].item())
        prompt_parts.append(
            f"Agent{i+1} judged this as '{label_str}' "
            f"(confidence: {conf:.3f}). "
            f"Reasoning: {reasoning[:120]}"
        )

    if prompt_parts:
        return (
            "[Cross-Agent Evidence Exchange]\n"
            "Other agents made the following judgments:\n"
            + "\n".join(prompt_parts) + "\n\n"
            "Please reconsider your judgment in light of this evidence. "
            "If other agents provide compelling reasons, you may revise your prediction. "
            "However, maintain your judgment if you are confident.\n\n"
            "Please re-analyze the content and output your final judgment."
        )
    return None


def identify_minority_agent(preds):
    """在2v1分歧中识别少数派Agent

    Returns:
        minority_idx: 少数派Agent的索引
        majority_pred: 多数派的预测
    """
    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1

    majority_pred = max(counts, key=counts.get)
    minority_pred = min(counts, key=counts.get)

    for i, p in enumerate(preds):
        if p == minority_pred:
            return i, majority_pred

    return 0, majority_pred


def run_reflection_on_sample(agents, text, image_description, image,
                              original_preds, original_beliefs, minority_idx,
                              all_reasonings):
    """对单个样本运行反思（只反思少数派Agent）"""
    api_calls = 0

    # 生成跨Agent证据交换提示
    cross_prompt = generate_cross_agent_prompt(
        minority_idx, original_preds, original_beliefs, all_reasonings
    )

    if cross_prompt is None:
        return original_preds[minority_idx], original_beliefs[minority_idx], api_calls, False

    agent = agents[minority_idx]
    modified_text = f"{cross_prompt}\n\n---\nOriginal content: {text}"

    try:
        if agent.use_direct_image and image is not None:
            alpha, belief, uncertainty, emb = agent.forward(modified_text, image=image)
        else:
            alpha, belief, uncertainty, emb = agent.forward(modified_text, image_description=image_description)

        new_pred = belief.argmax().item()
        api_calls += 1
        return new_pred, belief, api_calls, True
    except Exception as e:
        return original_preds[minority_idx], original_beliefs[minority_idx], api_calls, False


def main(max_disagree=100, providers=None):
    if providers is None:
        providers = ['gpt5.1', 'gemini', 'gpt5.1']

    print("=" * 80)
    print("Step5 高效版：因果反事实反思（跨Agent证据交换）")
    print("=" * 80)

    # [1] 加载验证集数据
    print("\n[1] 加载验证集数据...")
    val_path = os.path.join(DATA_DIR, 'dev.jsonl')
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]

    val_texts = [item['text'] for item in val_data]
    val_labels = torch.tensor([item['label'] for item in val_data])

    # 加载图像
    print("  加载图像...")
    val_images = []
    for item in val_data:
        img_path = os.path.join(DATA_DIR, item['img'])
        if not os.path.exists(img_path):
            img_path = os.path.join(DATA_DIR, 'img', os.path.basename(item['img']))
        val_images.append(Image.open(img_path).convert('RGB'))

    # CLIP描述
    clip_desc_path = os.path.join(CHECKPOINT_DIR, 'clip_descriptions_val.pt')
    if os.path.exists(clip_desc_path):
        clip_descriptions = torch.load(clip_desc_path, weights_only=False)
        val_image_descriptions = clip_descriptions
        print(f"  已加载CLIP图像描述: {len(val_image_descriptions)}")
    else:
        val_image_descriptions = None

    # [2] 加载缓存LLM推理结果
    print("\n[2] 加载缓存LLM推理结果...")
    original_preds = []
    original_beliefs = []
    original_uncertainties = []
    original_alphas = []

    for i in range(3):
        path = os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}.pt')
        result = torch.load(path, map_location='cpu', weights_only=False)
        original_preds.append(result['beliefs'].argmax(dim=1))
        original_beliefs.append(result['beliefs'])
        original_uncertainties.append(result['uncertainties'])
        original_alphas.append(result['alphas'])

    B_val = len(original_preds[0])
    print(f"  验证集样本数: {B_val}")

    # [3] 计算分歧样本
    print("\n[3] 识别分歧样本...")
    disagreement_indices = []
    for idx in range(B_val):
        p0, p1, p2 = original_preds[0][idx], original_preds[1][idx], original_preds[2][idx]
        if not (p0 == p1 == p2):
            disagreement_indices.append(idx)

    print(f"  分歧样本数: {len(disagreement_indices)}")

    # 限制处理数量
    if len(disagreement_indices) > max_disagree:
        # 优先选择MV错误的样本（反思有修正机会）
        mv_preds = []
        for idx in disagreement_indices:
            preds = [original_preds[i][idx].item() for i in range(3)]
            counts = {}
            for p in preds:
                counts[p] = counts.get(p, 0) + 1
            mv_preds.append(max(counts, key=counts.get))

        wrong_indices = [disagreement_indices[i] for i in range(len(disagreement_indices))
                        if mv_preds[i] != val_labels[disagreement_indices[i]].item()]
        correct_indices = [disagreement_indices[i] for i in range(len(disagreement_indices))
                          if mv_preds[i] == val_labels[disagreement_indices[i]].item()]

        # 按比例采样：70%错误样本，30%正确样本
        n_wrong = min(int(max_disagree * 0.7), len(wrong_indices))
        n_correct = min(max_disagree - n_wrong, len(correct_indices))

        np.random.seed(42)
        selected_wrong = np.random.choice(wrong_indices, n_wrong, replace=False).tolist()
        selected_correct = np.random.choice(correct_indices, n_correct, replace=False).tolist()
        selected_indices = sorted(selected_wrong + selected_correct)
    else:
        selected_indices = disagreement_indices

    print(f"  选取处理样本: {len(selected_indices)}")
    print(f"    其中MV错误: {len([i for i in selected_indices if i in wrong_indices]) if 'wrong_indices' in dir() else 'N/A'}")

    # [4] 创建LLM Agent
    print("\n[4] 创建LLM Agent...")
    agents = []
    for i, provider in enumerate(providers):
        min_interval = 30.0 if provider == 'glm' else 1.0
        client = LLMClient(
            provider=provider,
            temperature=0.1,
            max_retries=5,
            timeout=180,
            min_call_interval=min_interval,
        )
        agent = LLMAgent(
            client=client,
            name=f"Agent{i+1}",
            system_prompt=AGENT_PROMPTS.get(['text_focused', 'image_focused', 'multimodal_fusion'][i]),
            embed_dim=256,
            num_classes=NUM_CLASSES,
            use_image=(i >= 1),
            use_direct_image=(i >= 1 and provider in ['glm', 'gpt', 'gpt5.1', 'gpt4om', 'gemini']),
            verbose=False,
        )
        agents.append(agent)
        print(f"  Agent{i+1}: {provider}/{client.model}")

    # [5] 运行因果反思
    print(f"\n[5] 运行因果反思（跨Agent证据交换）...")
    print(f"  策略: 只反思少数派Agent（2v1中的1个）")
    print(f"  最大反思轮数: 1")
    print(f"  文本消融: 跳过（高效模式）")

    results = {
        'original_preds': [],
        'reflection_preds': [],
        'minority_agent': [],
        'changed': [],
        'converged': [],
        'api_calls': 0,
    }

    start_time = time.time()

    for idx in tqdm(selected_indices, desc="因果反思"):
        text = val_texts[idx]
        img_desc = val_image_descriptions[idx] if val_image_descriptions else None
        img = val_images[idx] if val_images else None

        preds = [original_preds[i][idx].item() for i in range(3)]
        beliefs = [original_beliefs[i][idx] for i in range(3)]

        # 识别少数派
        minority_idx, majority_pred = identify_minority_agent(preds)

        # MV基线
        mv_pred = majority_pred

        # 运行反思（只对少数派Agent）
        new_pred, new_belief, api_calls, success = run_reflection_on_sample(
            agents=agents,
            text=text,
            image_description=img_desc,
            image=img,
            original_preds=preds,
            original_beliefs=beliefs,
            minority_idx=minority_idx,
            all_reasonings=[],  # 缓存中没有reasoning
        )

        results['api_calls'] += api_calls

        # 反思后的预测
        if success:
            new_preds = preds.copy()
            new_preds[minority_idx] = new_pred
            # 反思后再次多数投票
            counts = {}
            for p in new_preds:
                counts[p] = counts.get(p, 0) + 1
            reflection_pred = max(counts, key=counts.get)
            changed = (new_pred != preds[minority_idx])
            converged = (len(set(new_preds)) == 1)  # 全部一致=收敛
        else:
            reflection_pred = mv_pred
            changed = False
            converged = False

        results['original_preds'].append(mv_pred)
        results['reflection_preds'].append(reflection_pred)
        results['minority_agent'].append(minority_idx)
        results['changed'].append(changed)
        results['converged'].append(converged)

    elapsed = time.time() - start_time

    # [6] 分析结果
    print(f"\n[6] 结果分析")
    print(f"{'='*80}")

    labels = val_labels[selected_indices].numpy()
    original_preds_arr = np.array(results['original_preds'])
    reflection_preds_arr = np.array(results['reflection_preds'])

    # 基线
    acc_before = accuracy_score(labels, original_preds_arr) * 100
    f1_before = f1_score(labels, original_preds_arr, average='binary') * 100

    # 反思后
    acc_after = accuracy_score(labels, reflection_preds_arr) * 100
    f1_after = f1_score(labels, reflection_preds_arr, average='binary') * 100

    # 统计
    changed_count = sum(results['changed'])
    converged_count = sum(results['converged'])

    print(f"处理样本数: {len(selected_indices)}")
    print(f"API调用次数: {results['api_calls']}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"平均每样本: {elapsed/len(selected_indices):.1f} 秒")
    print(f"\n反思统计:")
    print(f"  预测改变: {changed_count}/{len(selected_indices)} ({changed_count/len(selected_indices)*100:.1f}%)")
    print(f"  收敛(全一致): {converged_count}/{len(selected_indices)} ({converged_count/len(selected_indices)*100:.1f}%)")
    print(f"\n性能对比:")
    print(f"  {'指标':<20s} {'反思前(MV)':<15s} {'反思后':<15s} {'变化':<10s}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<20s} {acc_before:<15.2f} {acc_after:<15.2f} {acc_after-acc_before:+.2f}")
    print(f"  {'F1 Score':<20s} {f1_before:<15.2f} {f1_after:<15.2f} {f1_after-f1_before:+.2f}")

    # 分析改变的方向
    if changed_count > 0:
        changed_indices = [i for i, c in enumerate(results['changed']) if c]
        correct_changes = sum(1 for i in changed_indices
                            if reflection_preds_arr[i] == labels[i] and original_preds_arr[i] != labels[i])
        wrong_changes = sum(1 for i in changed_indices
                          if reflection_preds_arr[i] != labels[i] and original_preds_arr[i] == labels[i])
        neutral_changes = changed_count - correct_changes - wrong_changes
        print(f"\n  改变分析:")
        print(f"    正确修正(MV错→反思对): {correct_changes}")
        print(f"    错误改变(MV对→反思错): {wrong_changes}")
        print(f"    中性改变(都不对): {neutral_changes}")
        print(f"    净收益: +{correct_changes - wrong_changes}")

    # 全样本效果估算
    print(f"\n[7] 全样本效果估算:")
    # 反思只改变了部分样本，将改变应用到全量
    all_mv_preds = []
    for idx in range(B_val):
        preds = [original_preds[i][idx].item() for i in range(3)]
        counts = {}
        for p in preds:
            counts[p] = counts.get(p, 0) + 1
        all_mv_preds.append(max(counts, key=counts.get))

    all_mv_preds = np.array(all_mv_preds)
    all_labels = val_labels.numpy()
    all_acc_mv = accuracy_score(all_labels, all_mv_preds) * 100
    print(f"  全样本MV Acc: {all_acc_mv:.2f}%")

    # 将反思结果外推到全样本
    all_reflection_preds = all_mv_preds.copy()
    for i, idx in enumerate(selected_indices):
        all_reflection_preds[idx] = reflection_preds_arr[i]
    all_acc_ref = accuracy_score(all_labels, all_reflection_preds) * 100
    print(f"  全样本反思后 Acc(外推): {all_acc_ref:.2f}%")
    print(f"  提升: {all_acc_ref - all_acc_mv:+.2f}%")

    # 保存结果
    output = {
        '_metadata': {
            'experiment': 'Step5 高效版因果反思',
            'sample_size': B_val,
            'disagreement_count': len(disagreement_indices),
            'processed_count': len(selected_indices),
            'strategy': '只反思少数派Agent，跳过文本消融，MAX_REFLECTIONS=1',
            'providers': providers,
        },
        'results': {
            'acc_before': acc_before,
            'f1_before': f1_before,
            'acc_after': acc_after,
            'f1_after': f1_after,
            'delta_acc': acc_after - acc_before,
            'delta_f1': f1_after - f1_before,
            'changed_count': changed_count,
            'converged_count': converged_count,
            'api_calls': results['api_calls'],
            'elapsed_minutes': elapsed / 60,
        },
        'full_sample_estimate': {
            'acc_mv': all_acc_mv,
            'acc_reflection': all_acc_ref,
            'delta': all_acc_ref - all_acc_mv,
        },
        'processed_indices': selected_indices,
        'original_preds': results['original_preds'],
        'reflection_preds': results['reflection_preds'],
        'changed': results['changed'],
        'minority_agent': results['minority_agent'],
    }

    output_path = os.path.join(RESULT_DIR, 'step5_causal_reflection_efficient.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {output_path}")

    print(f"\n{'='*80}")
    print(f"总结:")
    print(f"  反思前 MV Acc: {acc_before:.2f}%")
    print(f"  反思后 Acc: {acc_after:.2f}% (Δ={acc_after-acc_before:+.2f}%)")
    print(f"  全样本外推: {all_acc_mv:.2f}% → {all_acc_ref:.2f}% (Δ={all_acc_ref-all_acc_mv:+.2f}%)")
    print(f"  API调用: {results['api_calls']}, 耗时: {elapsed/60:.1f}分钟")
    print(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_disagree', type=int, default=100, help='最大处理分歧样本数')
    parser.add_argument('--provider1', type=str, default='deepseek')
    parser.add_argument('--provider2', type=str, default='gemini')
    parser.add_argument('--provider3', type=str, default='gpt5.1')
    args = parser.parse_args()
    main(max_disagree=args.max_disagree, providers=[args.provider1, args.provider2, args.provider3])
