"""
Step5: LLM因果反事实反思（方案A）
=================================
对分歧样本触发因果反思，通过文本消融归因和补偿提示修正预测。

核心策略:
1. 仅对分歧样本触发反思（控制费用）
2. 文本消融归因：分割文本为片段，逐一移除后重新调用API观察变化
3. 补偿提示：生成"忽略XX特征"的修正提示词
4. 最多3轮反思，每轮只重新调用分歧Agent

费用控制:
- 分歧样本约118个（200验证集中的59%）
- 每轮反思只重调1-2个分歧Agent
- 文本消融仅做3-5个关键片段
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys, time, json, pickle, warnings
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

# =============================================================================
# 配置
# =============================================================================
NUM_CLASSES = 2
MAX_REFLECTIONS = 3
ABLATION_FRAGMENTS = 4
CONFIDENCE_THRESHOLD = 0.6

DATA_DIR = 'data/Hateful_Memes/data'
CHECKPOINT_DIR = 'checkpoints/hateful_memes'
RESULT_DIR = 'results/hateful_memes'

os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# 1. 文本消融归因
# =============================================================================

def split_text_into_fragments(text, n_fragments=4):
    """将文本分割为N个片段"""
    sentences = text.replace('?', '.').replace('!', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= n_fragments:
        return sentences
    
    avg_len = len(sentences) // n_fragments
    fragments = []
    for i in range(n_fragments):
        start = i * avg_len
        end = (i + 1) * avg_len if i < n_fragments - 1 else len(sentences)
        fragment = '. '.join(sentences[start:end])
        fragments.append(fragment)
    
    return fragments

def text_ablation_attribution(agent, text, original_pred, image_description=None, image=None):
    """
    文本消融归因：逐一移除文本片段，观察预测变化
    
    Args:
        agent: LLMAgent实例
        text: 原始文本
        original_pred: 原始预测（0或1）
        image_description: 图像描述（仅文本Agent为None）
        image: PIL图像（仅图像Agent）
    
    Returns:
        attribution_scores: 每个片段的归因分数（越高越重要）
        important_fragments: 重要片段列表
    """
    fragments = split_text_into_fragments(text, n_fragments=ABLATION_FRAGMENTS)
    
    if len(fragments) < 2:
        return [], []
    
    attribution_scores = []
    
    for i, fragment in enumerate(fragments):
        remaining_text = '. '.join([f for j, f in enumerate(fragments) if j != i])
        
        try:
            if agent.use_direct_image and image is not None:
                alpha, belief, uncertainty, emb = agent.forward(remaining_text, image=image)
            else:
                alpha, belief, uncertainty, emb = agent.forward(remaining_text, image_description=image_description)
            
            new_pred = belief.argmax().item()
            confidence = belief[0, new_pred].item()
            
            if new_pred != original_pred:
                score = 1.0 + (1.0 - confidence)
            else:
                score = abs(confidence - agent._get_original_confidence())
            
            attribution_scores.append((i, fragment, score, new_pred))
            
        except Exception as e:
            attribution_scores.append((i, fragment, 0.0, original_pred))
    
    attribution_scores.sort(key=lambda x: x[2], reverse=True)
    
    important_fragments = [(idx, frag, score) for idx, frag, score, _ in attribution_scores[:2]]
    
    return attribution_scores, important_fragments

# =============================================================================
# 2. 生成补偿提示
# =============================================================================

def generate_compensation_prompt(agent_name, important_fragments, original_text):
    """
    生成补偿提示：告诉模型忽略重要但可能误导的特征
    
    Args:
        agent_name: Agent名称
        important_fragments: 重要片段列表 [(idx, fragment, score)]
        original_text: 原始文本
    
    Returns:
        compensation_prompt: 补偿提示词
    """
    if not important_fragments:
        return None
    
    fragment_texts = [f[1] for f in important_fragments]
    fragments_str = "; ".join([f"片段{i+1}: '{f}'" for i, f in enumerate(fragment_texts)])
    
    compensation = f"""
[反思修正]
分析发现以下文本片段可能对您的判断产生重要影响：
{fragments_str}

请重新评估，但注意：这些片段可能包含误导性信息。
请忽略上述片段的过度影响，基于整体上下文重新判断。

原始文本：{original_text}
"""
    
    return compensation.strip()

# =============================================================================
# 3. LLM反思循环
# =============================================================================

def llm_reflection_loop(agents, texts, image_descriptions, images, 
                        original_preds, original_beliefs, disagreement_indices,
                        max_reflections=3):
    """
    LLM因果反思循环
    
    Args:
        agents: Agent列表 [agent1, agent2, agent3]
        texts: 验证集文本 [N]
        image_descriptions: 验证集图像描述 [N]
        images: 验证集图像 [N]
        original_preds: 原始预测 [3, N]
        original_beliefs: 原始信念 [3, N, 2]
        disagreement_indices: 分歧样本索引列表
        max_reflections: 最大反思次数
    
    Returns:
        reflection_results: 反思结果字典
    """
    results = {
        'converged': [],
        'reflections_used': [],
        'final_predictions': [],
        'rejected': [],
        'compensation_prompts': [],
        'api_calls': 0,
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
        
        for reflection_round in range(max_reflections):
            reflections_used = reflection_round + 1
            
            # 找到有分歧的Agent
            unique_preds = set(current_preds)
            if len(unique_preds) == 1:
                converged = True
                final_pred = current_preds[0]
                break
            
            # 对每个分歧Agent进行归因和修正
            new_preds = current_preds.copy()
            new_beliefs = current_beliefs.copy()
            
            for i, agent in enumerate(agents):
                if current_preds[i] == max(set(current_preds), key=current_preds.count):
                    continue
                
                # 文本消融归因
                orig_pred = current_preds[i]
                attribution_scores, important_fragments = text_ablation_attribution(
                    agent, text, orig_pred, 
                    image_description=img_desc if i != 1 else None,
                    image=img if i == 1 else None
                )
                results['api_calls'] += len(attribution_scores)
                
                if important_fragments:
                    # 生成补偿提示
                    comp_prompt = generate_compensation_prompt(
                        agent.name, important_fragments, text
                    )
                    prompts_used.append(comp_prompt)
                    
                    # 重新调用API（带补偿提示）
                    modified_text = f"{comp_prompt}\n\n请基于以上修正重新判断：{text}"
                    
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
            rejected = True
            final_pred = None
        
        results['converged'].append(converged)
        results['reflections_used'].append(reflections_used)
        results['final_predictions'].append(final_pred)
        results['rejected'].append(rejected)
        results['compensation_prompts'].append(prompts_used)
    
    return results

# =============================================================================
# 主函数
# =============================================================================

def main(max_val=200):
    print("=" * 70)
    print("Step5: LLM因果反事实反思")
    print("=" * 70)
    
    # ====== 加载数据 ======
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
    
    # 加载图像描述（如果存在）
    clip_desc_path = os.path.join(CHECKPOINT_DIR, 'clip_descriptions_val.pt')
    if os.path.exists(clip_desc_path):
        clip_descriptions = torch.load(clip_desc_path)
        val_image_descriptions = clip_descriptions[:max_val]
        print(f"  已加载CLIP图像描述: {len(val_image_descriptions)}")
    else:
        val_image_descriptions = None
        print(f"  CLIP描述不存在，跳过")
    
    # ====== 加载原始LLM结果 ======
    print("\n[2] 加载原始LLM推理结果...")
    
    original_preds = []
    original_beliefs = []
    
    for i in range(3):
        path = os.path.join(CHECKPOINT_DIR, f'llm_val_agent{i}.pt')
        result = torch.load(path, map_location='cpu')
        original_preds.append(result['beliefs'].argmax(dim=1))
        original_beliefs.append(result['beliefs'])
    
    # ====== 加载分歧样本索引 ======
    disagreement_path = os.path.join(CHECKPOINT_DIR, 'disagreement_indices.pt')
    if os.path.exists(disagreement_path):
        disagreement_data = torch.load(disagreement_path)
        disagreement_indices = disagreement_data['disagreement_indices']
        print(f"  分歧样本数: {len(disagreement_indices)}")
    else:
        disagreement_indices = []
        for idx in range(max_val):
            p0, p1, p2 = original_preds[0][idx], original_preds[1][idx], original_preds[2][idx]
            if not (p0 == p1 == p2):
                disagreement_indices.append(idx)
        print(f"  重新计算分歧样本: {len(disagreement_indices)}")
    
    # ====== 创建Agent（仅用于反思时重新调用） ======
    print("\n[3] 创建LLM Agent...")
    
    agents = []
    providers = ['deepseek', 'glm', 'deepseek']
    
    for i, provider in enumerate(providers):
        agent = create_single_agent(
            provider=provider,
            agent_type=i,
            temperature=0.1,
            timeout=120,
        )
        agents.append(agent)
        print(f"  Agent{i+1}: {agent.name} ({provider})")
    
    # ====== 运行因果反思 ======
    print("\n[4] 运行因果反思...")
    print(f"  目标样本数: {len(disagreement_indices)}")
    print(f"  最大反思轮数: {MAX_REFLECTIONS}")
    print(f"  每样本消融片段数: {ABLATION_FRAGMENTS}")
    
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
    
    # ====== 结果分析 ======
    print("\n[5] 反思结果分析")
    
    converged_count = sum(reflection_results['converged'])
    rejected_count = sum(reflection_results['rejected'])
    avg_reflections = np.mean(reflection_results['reflections_used'])
    
    print(f"\n反思统计:")
    print(f"  收敛样本: {converged_count}/{len(disagreement_indices)} ({converged_count/len(disagreement_indices)*100:.1f}%)")
    print(f"  拒识样本: {rejected_count}/{len(disagreement_indices)} ({rejected_count/len(disagreement_indices)*100:.1f}%)")
    print(f"  平均反思轮数: {avg_reflections:.1f}")
    print(f"  额外API调用次数: {reflection_results['api_calls']}")
    print(f"  总耗时: {elapsed_time:.1f}秒")
    
    # 计算反思后的准确率
    corrected_labels = val_labels[disagreement_indices]
    corrected_preds = []
    
    for i, idx in enumerate(disagreement_indices):
        if reflection_results['converged'][i]:
            corrected_preds.append(reflection_results['final_predictions'][i])
        else:
            corrected_preds.append(original_preds[0][idx].item())
    
    if corrected_preds:
        acc_after = accuracy_score(corrected_labels.numpy(), corrected_preds)
        f1_after = f1_score(corrected_labels.numpy(), corrected_preds)
        print(f"\n反思后分歧样本表现:")
        print(f"  准确率: {acc_after*100:.2f}%")
        print(f"  F1分数: {f1_after*100:.2f}%")
    
    # 保存结果
    result_path = os.path.join(RESULT_DIR, 'causal_reflection_results.pt')
    torch.save(reflection_results, result_path)
    print(f"\n结果已保存到: {result_path}")
    
    return reflection_results

if __name__ == '__main__':
    main(max_val=200)