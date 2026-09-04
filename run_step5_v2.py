# -*- coding: utf-8 -*-
"""
Step5 v2：因果反事实反思（全Agent证据交换）
=============================================
v1只反思少数派，导致2v1中预测无法翻转。
v2反思所有Agent，使多数派也可能改变观点。

优化策略：
1. 反思所有3个Agent（不仅少数派）
2. 跳过文本消融（保持高效）
3. MAX_REFLECTIONS=1
4. 每样本3次API调用（3 agents * 1轮）

成本估算（100个分歧样本）：
- 100样本 * 3 agents * 1 API = ~300次API
- 预估时间：~60-90分钟
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


def generate_cross_agent_prompt(agent_idx, all_preds, all_beliefs):
    """生成跨Agent证据交换提示"""
    prompt_parts = []
    for i in range(3):
        if i == agent_idx:
            continue
        pred = all_preds[i]
        belief = all_beliefs[i]
        label_str = "hateful" if pred == 1 else "non-hateful"
        conf = float(belief[pred].item()) if belief.dim() == 1 else float(belief[0, pred].item())
        prompt_parts.append(
            f"Agent{i+1} judged this as '{label_str}' (confidence: {conf:.3f})."
        )

    if prompt_parts:
        return (
            "[Cross-Agent Evidence Exchange]\n"
            "Other agents made the following judgments:\n"
            + "\n".join(prompt_parts) + "\n\n"
            "Please reconsider your judgment in light of this evidence. "
            "If other agents provide compelling evidence, you may revise your prediction. "
            "However, maintain your judgment if you are confident in your analysis.\n\n"
            "Please re-analyze the content and output your final judgment."
        )
    return None


def reflect_all_agents(agents, text, image_description, image,
                       original_preds, original_beliefs):
    """对所有3个Agent运行反思"""
    new_preds = list(original_preds)
    new_beliefs = list(original_beliefs)
    api_calls = 0
    n_changed = 0

    for i, agent in enumerate(agents):
        cross_prompt = generate_cross_agent_prompt(i, original_preds, original_beliefs)
        if cross_prompt is None:
            continue

        modified_text = f"{cross_prompt}\n\n---\nOriginal content: {text}"

        try:
            if agent.use_direct_image and image is not None:
                alpha, belief, uncertainty, emb = agent.forward(modified_text, image=image)
            else:
                alpha, belief, uncertainty, emb = agent.forward(modified_text, image_description=image_description)

            new_pred = belief.argmax().item()
            api_calls += 1

            if new_pred != original_preds[i]:
                new_preds[i] = new_pred
                new_beliefs[i] = belief
                n_changed += 1
        except Exception:
            pass

    # 反思后多数投票
    counts = {}
    for p in new_preds:
        counts[p] = counts.get(p, 0) + 1
    final_pred = max(counts, key=counts.get)

    return final_pred, new_preds, api_calls, n_changed


def main(max_disagree=100, providers=None):
    if providers is None:
        providers = ['gpt5.1', 'gemini', 'gpt5.1']

    print("=" * 80)
    print("Step5 v2：因果反事实反思（全Agent证据交换）")
    print("=" * 80)

    # [1] 加载数据
    print("\n[1] 加载验证集数据...")
    val_path = os.path.join(DATA_DIR, 'dev.jsonl')
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]

    val_texts = [item['text'] for item in val_data]
    val_labels = torch.tensor([item['label'] for item in val_data])

    print("  加载图像...")
    val_images = []
    for item in val_data:
        img_path = os.path.join(DATA_DIR, item['img'])
        if not os.path.exists(img_path):
            img_path = os.path.join(DATA_DIR, 'img', os.path.basename(item['img']))
        val_images.append(Image.open(img_path).convert('RGB'))

    clip_desc_path = os.path.join(CHECKPOINT_DIR, 'clip_descriptions_val.pt')
    if os.path.exists(clip_desc_path):
        clip_descriptions = torch.load(clip_desc_path, weights_only=False)
        val_image_descriptions = clip_descriptions
        print(f"  已加载CLIP图像描述: {len(val_image_descriptions)}")
    else:
        val_image_descriptions = None

    # [2] 加载缓存
    print("\n[2] 加载缓存LLM推理结果...")
    original_preds = []
    original_beliefs = []
    for i in range(3):
        path = os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}.pt')
        result = torch.load(path, map_location='cpu', weights_only=False)
        original_preds.append(result['beliefs'].argmax(dim=1))
        original_beliefs.append(result['beliefs'])

    B_val = len(original_preds[0])

    # [3] 识别分歧样本
    print("\n[3] 识别分歧样本...")
    disagreement_indices = []
    for idx in range(B_val):
        p0, p1, p2 = original_preds[0][idx], original_preds[1][idx], original_preds[2][idx]
        if not (p0 == p1 == p2):
            disagreement_indices.append(idx)

    print(f"  分歧样本数: {len(disagreement_indices)}")

    # 选取样本
    if len(disagreement_indices) > max_disagree:
        mv_preds = []
        for idx in disagreement_indices:
            preds = [original_preds[i][idx].item() for i in range(3)]
            counts = {}
            for p in preds:
                counts[p] = counts.get(p, 0) + 1
            mv_preds.append(max(counts, key=counts.get))

        wrong = [disagreement_indices[i] for i in range(len(disagreement_indices))
                if mv_preds[i] != val_labels[disagreement_indices[i]].item()]
        correct = [disagreement_indices[i] for i in range(len(disagreement_indices))
                  if mv_preds[i] == val_labels[disagreement_indices[i]].item()]

        n_wrong = min(int(max_disagree * 0.7), len(wrong))
        n_correct = min(max_disagree - n_wrong, len(correct))

        np.random.seed(42)
        sel_wrong = np.random.choice(wrong, n_wrong, replace=False).tolist()
        sel_correct = np.random.choice(correct, n_correct, replace=False).tolist()
        selected = sorted(sel_wrong + sel_correct)
    else:
        selected = disagreement_indices

    print(f"  选取处理: {len(selected)} (MV错误: {n_wrong if len(disagreement_indices) > max_disagree else 'N/A'})")

    # [4] 创建Agent
    print("\n[4] 创建LLM Agent...")
    agents = []
    for i, provider in enumerate(providers):
        min_interval = 30.0 if provider == 'glm' else 1.0
        client = LLMClient(
            provider=provider, temperature=0.1, max_retries=5, timeout=180,
            min_call_interval=min_interval,
        )
        agent = LLMAgent(
            client=client, name=f"Agent{i+1}",
            system_prompt=AGENT_PROMPTS.get(['text_focused', 'image_focused', 'multimodal_fusion'][i]),
            embed_dim=256, num_classes=NUM_CLASSES,
            use_image=(i >= 1),
            use_direct_image=(i >= 1 and provider in ['glm', 'gpt', 'gpt5.1', 'gpt4om', 'gemini']),
            verbose=False,
        )
        agents.append(agent)
        print(f"  Agent{i+1}: {provider}/{client.model}")

    # [5] 运行反思
    print(f"\n[5] 运行因果反思（全Agent证据交换）...")
    print(f"  策略: 反思所有3个Agent, MAX_REFLECTIONS=1, 跳过文本消融")

    results = {
        'original_preds': [], 'reflection_preds': [],
        'agent_changes': [], 'api_calls': 0,
    }

    start_time = time.time()

    for idx in tqdm(selected, desc="因果反思"):
        text = val_texts[idx]
        img_desc = val_image_descriptions[idx] if val_image_descriptions else None
        img = val_images[idx] if val_images else None

        preds = [original_preds[i][idx].item() for i in range(3)]
        beliefs = [original_beliefs[i][idx] for i in range(3)]

        # MV基线
        counts = {}
        for p in preds:
            counts[p] = counts.get(p, 0) + 1
        mv_pred = max(counts, key=counts.get)

        # 全Agent反思
        final_pred, new_preds, api_calls, n_changed = reflect_all_agents(
            agents, text, img_desc, img, preds, beliefs
        )

        results['api_calls'] += api_calls
        results['original_preds'].append(mv_pred)
        results['reflection_preds'].append(final_pred)
        results['agent_changes'].append(n_changed)

    elapsed = time.time() - start_time

    # [6] 分析
    print(f"\n[6] 结果分析")
    print(f"{'='*80}")

    labels = val_labels[selected].numpy()
    orig_arr = np.array(results['original_preds'])
    refl_arr = np.array(results['reflection_preds'])

    acc_before = accuracy_score(labels, orig_arr) * 100
    f1_before = f1_score(labels, orig_arr, average='binary') * 100
    acc_after = accuracy_score(labels, refl_arr) * 100
    f1_after = f1_score(labels, refl_arr, average='binary') * 100

    total_changes = sum(1 for i in range(len(orig_arr)) if orig_arr[i] != refl_arr[i])
    correct_fixes = sum(1 for i in range(len(orig_arr))
                       if orig_arr[i] != labels[i] and refl_arr[i] == labels[i])
    wrong_changes = sum(1 for i in range(len(orig_arr))
                       if orig_arr[i] == labels[i] and refl_arr[i] != labels[i])

    print(f"处理样本: {len(selected)}")
    print(f"API调用: {results['api_calls']}")
    print(f"耗时: {elapsed/60:.1f}分钟 ({elapsed/len(selected):.1f}s/样本)")
    print(f"\n反思统计:")
    print(f"  预测翻转(MV改变): {total_changes}/{len(selected)} ({total_changes/len(selected)*100:.1f}%)")
    print(f"  正确修正(MV错→对): {correct_fixes}")
    print(f"  错误改变(MV对→错): {wrong_changes}")
    print(f"  净收益: +{correct_fixes - wrong_changes}")
    print(f"\n性能对比:")
    print(f"  {'指标':<20s} {'反思前(MV)':<15s} {'反思后':<15s} {'变化':<10s}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<20s} {acc_before:<15.2f} {acc_after:<15.2f} {acc_after-acc_before:+.2f}")
    print(f"  {'F1 Score':<20s} {f1_before:<15.2f} {f1_after:<15.2f} {f1_after-f1_before:+.2f}")

    # 全样本外推
    all_mv = []
    for idx in range(B_val):
        preds = [original_preds[i][idx].item() for i in range(3)]
        counts = {}
        for p in preds:
            counts[p] = counts.get(p, 0) + 1
        all_mv.append(max(counts, key=counts.get))
    all_mv = np.array(all_mv)
    all_labels = val_labels.numpy()
    all_acc_mv = accuracy_score(all_labels, all_mv) * 100

    all_refl = all_mv.copy()
    for i, idx in enumerate(selected):
        all_refl[idx] = refl_arr[i]
    all_acc_refl = accuracy_score(all_labels, all_refl) * 100

    print(f"\n[7] 全样本效果:")
    print(f"  全样本MV Acc: {all_acc_mv:.2f}%")
    print(f"  全样本反思后 Acc(外推): {all_acc_refl:.2f}%")
    print(f"  提升: {all_acc_refl - all_acc_mv:+.2f}%")

    # 保存
    output = {
        '_metadata': {
            'experiment': 'Step5 v2 因果反思（全Agent证据交换）',
            'sample_size': B_val,
            'disagreement_count': len(disagreement_indices),
            'processed_count': len(selected),
            'strategy': '反思所有Agent, 跳过文本消融, MAX_REFLECTIONS=1',
            'providers': providers,
            'date': '2026-08-24',
        },
        'results': {
            'acc_before': acc_before, 'f1_before': f1_before,
            'acc_after': acc_after, 'f1_after': f1_after,
            'delta_acc': acc_after - acc_before,
            'delta_f1': f1_after - f1_before,
            'total_changes': total_changes,
            'correct_fixes': correct_fixes,
            'wrong_changes': wrong_changes,
            'net_gain': correct_fixes - wrong_changes,
            'api_calls': results['api_calls'],
            'elapsed_minutes': elapsed / 60,
        },
        'full_sample': {
            'acc_mv': all_acc_mv, 'acc_reflection': all_acc_refl,
            'delta': all_acc_refl - all_acc_mv,
        },
        'processed_indices': selected,
        'original_preds': results['original_preds'],
        'reflection_preds': results['reflection_preds'],
    }

    output_path = os.path.join(RESULT_DIR, 'step5_causal_reflection_v2.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {output_path}")

    print(f"\n{'='*80}")
    print(f"总结:")
    print(f"  分歧样本 MV Acc: {acc_before:.2f}% → 反思后: {acc_after:.2f}% (Δ={acc_after-acc_before:+.2f}%)")
    print(f"  正确修正: {correct_fixes}, 错误改变: {wrong_changes}, 净收益: +{correct_fixes-wrong_changes}")
    print(f"  全样本外推: {all_acc_mv:.2f}% → {all_acc_refl:.2f}% (Δ={all_acc_refl-all_acc_mv:+.2f}%)")
    print(f"  API: {results['api_calls']}, 耗时: {elapsed/60:.1f}分钟")
    print(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_disagree', type=int, default=100)
    parser.add_argument('--provider1', type=str, default='deepseek')
    parser.add_argument('--provider2', type=str, default='gemini')
    parser.add_argument('--provider3', type=str, default='gpt5.1')
    args = parser.parse_args()
    main(max_disagree=args.max_disagree, providers=[args.provider1, args.provider2, args.provider3])
