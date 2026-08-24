"""
Step5: LLM因果反事实反思（优化版）
=================================
对分歧样本触发因果反思，通过跨Agent证据交换和分层反思修正预测。

核心策略:
1. 分层反思：简单分歧（2v1）直接采用多数投票，复杂分歧触发完整反思
2. 跨Agent证据交换：让分歧Agent看到其他Agent的reasoning和预测结果
3. 改进文本消融：使用关键词/实体识别替代简单句子分割
4. 补偿提示增强：结合其他Agent观点生成修正提示词
5. 最多2轮反思，控制费用

费用控制:
- 简单分歧（2v1）不触发反思，直接多数投票
- 复杂分歧仅触发1-2轮反思
- 每轮反思只重调分歧Agent
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys, time, json, pickle, warnings, re
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm_agent import LLMAgent, create_single_agent, AGENT_PROMPTS
from src.llm_api import LLMClient

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLASSES = 2
MAX_REFLECTIONS = 2
ABLATION_FRAGMENTS = 3
CONFIDENCE_THRESHOLD = 0.6
SIMPLE_DISAGREEMENT_THRESHOLD = 3

DATA_DIR = 'data/Hateful_Memes/data'
CHECKPOINT_DIR = 'checkpoints/hateful_memes'
RESULT_DIR = 'results/hateful_memes'

os.makedirs(RESULT_DIR, exist_ok=True)

HATE_KEYWORDS = [
    'parasite', 'terrorist', 'stupid', 'idiot', 'dumb', 'nigger', 'faggot',
    'bitch', 'whore', 'slut', 'retard', 'spic', 'chink', 'kike', 'dyke',
    'fag', 'cock', 'pussy', 'rape', 'kill', 'murder', 'die', 'destroy',
    'hate', 'hateful', 'discrimination', 'racist', 'sexist', 'homophobic',
    'antisemitic', 'bigot', 'prejudice', 'biased', 'offensive', 'insult',
    'derogatory', 'violent', 'aggression', 'attack', 'assault', 'oppression',
    'supremacist', 'white power', 'nazism', 'fascism', 'genocide', 'ethnic',
    'religious', 'minority', 'group', 'community', 'identity', 'targeted'
]

def extract_keywords(text):
    """从文本中提取仇恨相关关键词"""
    text_lower = text.lower()
    found_keywords = []
    for kw in HATE_KEYWORDS:
        if kw.lower() in text_lower:
            found_keywords.append(kw)
    return found_keywords

def split_text_by_keywords(text, n_fragments=3):
    """基于关键词分割文本，优先保留包含关键词的片段"""
    sentences = text.replace('?', '.').replace('!', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= n_fragments:
        return sentences
    
    kw_sentences = []
    non_kw_sentences = []
    
    for sent in sentences:
        if any(kw.lower() in sent.lower() for kw in HATE_KEYWORDS):
            kw_sentences.append(sent)
        else:
            non_kw_sentences.append(sent)
    
    fragments = kw_sentences[:n_fragments-1]
    
    remaining = non_kw_sentences
    if remaining:
        avg_len = len(remaining) // max(1, n_fragments - len(fragments))
        for i in range(n_fragments - len(fragments)):
            start = i * avg_len
            end = (i + 1) * avg_len if i < (n_fragments - len(fragments) - 1) else len(remaining)
            fragment = '. '.join(remaining[start:end])
            if fragment:
                fragments.append(fragment)
    
    if len(fragments) < n_fragments:
        fragments.extend([''] * (n_fragments - len(fragments)))
    
    return [f for f in fragments if f]

def text_ablation_attribution(agent, text, original_pred, image_description=None, image=None):
    """改进的文本消融归因：基于关键词分割"""
    fragments = split_text_by_keywords(text, n_fragments=ABLATION_FRAGMENTS)
    
    if len(fragments) < 2:
        return [], []
    
    attribution_scores = []
    
    for i, fragment in enumerate(fragments):
        remaining_text = '. '.join([f for j, f in enumerate(fragments) if j != i])
        
        if not remaining_text.strip():
            attribution_scores.append((i, fragment, 0.0, original_pred))
            continue
        
        try:
            if agent.use_direct_image and image is not None:
                alpha, belief, uncertainty, emb = agent.forward(remaining_text, image=image)
            else:
                alpha, belief, uncertainty, emb = agent.forward(remaining_text, image_description=image_description)
            
            new_pred = belief.argmax().item()
            confidence = float(belief[0, new_pred].item()) if belief.dim() > 1 else float(belief[new_pred].item())
            
            if new_pred != original_pred:
                score = 1.0 + (1.0 - confidence)
            else:
                score = abs(confidence - 0.5)
            
            attribution_scores.append((i, fragment, score, new_pred))
            
        except Exception as e:
            attribution_scores.append((i, fragment, 0.0, original_pred))
    
    attribution_scores.sort(key=lambda x: x[2], reverse=True)
    
    important_fragments = [(idx, frag, score) for idx, frag, score, _ in attribution_scores[:2]]
    
    return attribution_scores, important_fragments

def generate_cross_agent_prompt(agent_name, agent_idx, other_preds, other_beliefs, other_reasonings):
    """生成跨Agent证据交换提示：让Agent看到其他Agent的观点"""
    prompt_parts = []
    
    for i, (pred, belief, reasoning) in enumerate(zip(other_preds, other_beliefs, other_reasonings)):
        if i == agent_idx:
            continue
        
        label_str = "仇恨言论" if pred == 1 else "非仇恨言论"
        conf = float(belief[pred].item()) if belief.dim() > 1 else float(belief[pred].item())
        prompt_parts.append(f"Agent{i+1}判断为{label_str}(置信度{conf:.3f})，理由：{reasoning[:80]}")
    
    if prompt_parts:
        return "\n".join(prompt_parts)
    return None

def generate_compensation_prompt(agent_name, important_fragments, original_text, cross_agent_info=None):
    """生成补偿提示：结合文本消融和跨Agent证据"""
    if not important_fragments and not cross_agent_info:
        return None
    
    parts = ["[反思修正]"]
    
    if cross_agent_info:
        parts.append("其他Agent的判断如下：")
        parts.append(cross_agent_info)
        parts.append("请参考其他Agent的观点，重新评估。")
    
    if important_fragments:
        fragment_texts = [f[1] for f in important_fragments]
        fragments_str = "; ".join([f"片段{i+1}: '{f}'" for i, f in enumerate(fragment_texts)])
        parts.append(f"分析发现以下文本片段可能对您的判断产生重要影响：{fragments_str}")
        parts.append("请重新评估，但注意：这些片段可能包含误导性信息。")
    
    parts.append(f"原始文本：{original_text}")
    
    return "\n\n".join(parts)

def analyze_disagreement_type(preds):
    """分析分歧类型：简单分歧（2v1）或复杂分歧（1v1v1）"""
    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1
    
    if max(counts.values()) >= SIMPLE_DISAGREEMENT_THRESHOLD:
        return 'simple', max(counts, key=counts.get)
    else:
        return 'complex', None

def llm_reflection_loop(agents, texts, image_descriptions, images, 
                        original_preds, original_beliefs, disagreement_indices,
                        max_reflections=2):
    """优化的LLM因果反思循环"""
    results = {
        'converged': [],
        'reflections_used': [],
        'final_predictions': [],
        'rejected': [],
        'compensation_prompts': [],
        'api_calls': 0,
        'disagreement_types': [],
    }
    
    for idx in tqdm(disagreement_indices, desc="因果反思"):
        text = texts[idx]
        img_desc = image_descriptions[idx] if image_descriptions else None
        img = images[idx] if images else None
        
        current_preds = [original_preds[i][idx].item() for i in range(3)]
        current_beliefs = [original_beliefs[i][idx] for i in range(3)]
        
        converged = False
        reflections_used = 0
        final_pred = None
        rejected = False
        prompts_used = []
        disagreement_type = None
        
        disagreement_type, majority_pred = analyze_disagreement_type(current_preds)
        
        if disagreement_type == 'simple':
            converged = True
            final_pred = majority_pred
            reflections_used = 0
            rejected = False
            results['disagreement_types'].append('simple')
            results['converged'].append(converged)
            results['reflections_used'].append(reflections_used)
            results['final_predictions'].append(final_pred)
            results['rejected'].append(rejected)
            results['compensation_prompts'].append(prompts_used)
            continue
        
        results['disagreement_types'].append('complex')
        
        other_reasonings = [''] * 3
        
        for reflection_round in range(max_reflections):
            reflections_used = reflection_round + 1
            
            unique_preds = set(current_preds)
            if len(unique_preds) == 1:
                converged = True
                final_pred = current_preds[0]
                break
            
            new_preds = current_preds.copy()
            new_beliefs = current_beliefs.copy()
            
            for i, agent in enumerate(agents):
                cross_agent_info = generate_cross_agent_prompt(
                    agent.name, i, current_preds, current_beliefs, other_reasonings
                )
                
                orig_pred = current_preds[i]
                
                attribution_scores, important_fragments = text_ablation_attribution(
                    agent, text, orig_pred, 
                    image_description=img_desc if i != 1 else None,
                    image=img if i == 1 else None
                )
                results['api_calls'] += len(attribution_scores)
                
                comp_prompt = generate_compensation_prompt(
                    agent.name, important_fragments, text, cross_agent_info
                )
                
                if comp_prompt:
                    prompts_used.append(comp_prompt)
                    
                    modified_text = f"{comp_prompt}\n\n请基于以上分析重新判断：{text}"
                    
                    try:
                        if agent.use_direct_image and image is not None and i == 1:
                            alpha, belief, uncertainty, emb = agent.forward(modified_text, image=img)
                        else:
                            alpha, belief, uncertainty, emb = agent.forward(modified_text, image_description=img_desc)
                        
                        new_pred = belief.argmax().item()
                        new_preds[i] = new_pred
                        new_beliefs[i] = belief
                        results['api_calls'] += 1
                        
                    except Exception as e:
                        pass
            
            current_preds = new_preds
            current_beliefs = new_beliefs
        
        if not converged:
            counts = {}
            for p in current_preds:
                counts[p] = counts.get(p, 0) + 1
            if counts:
                final_pred = max(counts, key=counts.get)
                if max(counts.values()) >= 2:
                    converged = True
            
            if not converged:
                rejected = True
                final_pred = None
        
        results['converged'].append(converged)
        results['reflections_used'].append(reflections_used)
        results['final_predictions'].append(final_pred)
        results['rejected'].append(rejected)
        results['compensation_prompts'].append(prompts_used)
    
    return results

def main(max_val=200, providers=None):
    if providers is None:
        providers = ['gpt5', 'gemini', 'gpt5']
    print("=" * 70)
    print("Step5: LLM因果反事实反思（优化版）")
    print("=" * 70)
    
    print("\n[1] 加载验证集数据...")
    
    val_path = os.path.join(DATA_DIR, 'dev.jsonl')
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    
    val_data = val_data[:max_val]
    val_texts = [item['text'] for item in val_data]
    val_labels = torch.tensor([item['label'] for item in val_data])
    
    val_images = []
    for item in val_data:
        img_path = os.path.join(DATA_DIR, item['img'])
        if not os.path.exists(img_path):
            img_path = os.path.join(DATA_DIR, 'img', os.path.basename(item['img']))
        val_images.append(Image.open(img_path).convert('RGB'))
    
    clip_desc_path = os.path.join(CHECKPOINT_DIR, 'clip_descriptions_val.pt')
    if os.path.exists(clip_desc_path):
        clip_descriptions = torch.load(clip_desc_path)
        val_image_descriptions = clip_descriptions[:max_val]
        print(f"  已加载CLIP图像描述: {len(val_image_descriptions)}")
    else:
        val_image_descriptions = None
        print(f"  CLIP描述不存在，跳过")
    
    print("\n[2] 加载原始LLM推理结果...")
    
    original_preds = []
    original_beliefs = []
    
    for i in range(3):
        path = os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}.pt')
        result = torch.load(path, map_location='cpu')
        original_preds.append(result['beliefs'].argmax(dim=1))
        original_beliefs.append(result['beliefs'])
    
    disagreement_path = os.path.join(CHECKPOINT_DIR, 'disagreement_indices.pt')
    if os.path.exists(disagreement_path):
        disagreement_data = torch.load(disagreement_path)
        disagreement_indices = disagreement_data['disagreement_indices']
        disagreement_indices = [idx for idx in disagreement_indices if idx < max_val]
        print(f"  加载分歧样本: {len(disagreement_indices)}")
    else:
        disagreement_indices = []
        for idx in range(max_val):
            p0, p1, p2 = original_preds[0][idx], original_preds[1][idx], original_preds[2][idx]
            if not (p0 == p1 == p2):
                disagreement_indices.append(idx)
        print(f"  重新计算分歧样本: {len(disagreement_indices)}")
    
    print("\n[3] 创建LLM Agent...")
    
    agents = []
    
    for i, provider in enumerate(providers):
        min_interval = 30.0 if provider == 'glm' else 1.0
        client = LLMClient(
            provider=provider,
            temperature=0.1,
            max_retries=8,
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
            use_direct_image=(i >= 1 and provider in ['glm', 'gpt', 'gpt5', 'gpt4om', 'gemini']),
            verbose=False,
        )
        agents.append(agent)
        print(f"  Agent{i+1}: {provider}/{client.model} [间隔={min_interval:.0f}s]")
    
    print("\n[4] 运行因果反思...")
    print(f"  目标样本数: {len(disagreement_indices)}")
    print(f"  最大反思轮数: {MAX_REFLECTIONS}")
    print(f"  每样本消融片段数: {ABLATION_FRAGMENTS}")
    print(f"  分层反思: 简单分歧(2v1)直接多数投票，复杂分歧触发反思")
    
    start_time = time.time()
    
    reflection_results = llm_reflection_loop(
        agents=agents,
        texts=val_texts,
        image_descriptions=val_image_descriptions,
        images=val_images,
        original_preds=original_preds,
        original_beliefs=original_beliefs,
        disagreement_indices=disagreement_indices,
        max_reflections=MAX_REFLECTIONS,
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n[5] 反思结果分析")
    
    converged_count = sum(reflection_results['converged'])
    rejected_count = sum(reflection_results['rejected'])
    avg_reflections = np.mean(reflection_results['reflections_used'])
    
    simple_count = reflection_results['disagreement_types'].count('simple')
    complex_count = reflection_results['disagreement_types'].count('complex')
    
    print(f"\n反思统计:")
    print(f"  收敛样本: {converged_count}/{len(disagreement_indices)} ({converged_count/len(disagreement_indices)*100:.1f}%)")
    print(f"  拒识样本: {rejected_count}/{len(disagreement_indices)} ({rejected_count/len(disagreement_indices)*100:.1f}%)")
    print(f"  平均反思轮数: {avg_reflections:.1f}")
    print(f"  额外API调用次数: {reflection_results['api_calls']}")
    print(f"  总耗时: {elapsed_time:.1f}秒")
    print(f"  简单分歧(2v1): {simple_count}个")
    print(f"  复杂分歧(1v1v1): {complex_count}个")
    
    corrected_labels = val_labels[disagreement_indices]
    corrected_preds = []
    
    for i, idx in enumerate(disagreement_indices):
        if reflection_results['final_predictions'][i] is not None:
            corrected_preds.append(reflection_results['final_predictions'][i])
        else:
            corrected_preds.append(original_preds[0][idx].item())
    
    if corrected_preds:
        acc_after = accuracy_score(corrected_labels.numpy(), corrected_preds)
        f1_after = f1_score(corrected_labels.numpy(), corrected_preds)
        print(f"\n反思后分歧样本表现:")
        print(f"  准确率: {acc_after*100:.2f}%")
        print(f"  F1分数: {f1_after*100:.2f}%")
    
    original_correct = 0
    for i, idx in enumerate(disagreement_indices):
        if original_preds[0][idx].item() == val_labels[idx].item():
            original_correct += 1
    original_acc = original_correct / len(disagreement_indices)
    print(f"\n原始分歧样本表现:")
    print(f"  准确率: {original_acc*100:.2f}%")
    
    result_path = os.path.join(RESULT_DIR, 'causal_reflection_results_optimized.pt')
    torch.save(reflection_results, result_path)
    print(f"\n结果已保存到: {result_path}")
    
    return reflection_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_val', type=int, default=200)
    parser.add_argument('--provider1', type=str, default='deepseek')
    parser.add_argument('--provider2', type=str, default='gemini')
    parser.add_argument('--provider3', type=str, default='gpt5')
    args = parser.parse_args()
    main(max_val=args.max_val, providers=[args.provider1, args.provider2, args.provider3])