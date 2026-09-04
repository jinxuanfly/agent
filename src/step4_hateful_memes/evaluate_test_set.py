#!/usr/bin/env python3
"""
Hateful Memes测试集评估脚本
============================
对test.jsonl（1000样本，无标签）运行完整推理管线，生成EvalAI提交文件。

流程：
  1. 加载 test.jsonl（1000样本，无标签）
  2. 运行 3个LLM Agent 独立推理（缓存到checkpoints）
  3. 执行 Majority Voting 得到基础预测
  4. 识别分歧样本，运行因果反思
  5. 生成最终预测文件（EvalAI格式）

用法：
  python src/step4_hateful_memes/evaluate_test_set.py --seed=42
  python src/step4_hateful_memes/evaluate_test_set.py --seed=42 --skip_reflection
  
输出:
  checkpoints/hateful_memes/llm_test_agent{i}_seed{seed}.pt  # LLM推理缓存
  results/hateful_memes/test_predictions_*_seed{seed}.csv     # EvalAI提交文件
  results/hateful_memes/test_results_seed{seed}.json          # 详细结果
"""

import os, sys, time, json, warnings
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm_agent import LLMAgent, AGENT_PROMPTS
from src.llm_api import LLMClient

warnings.filterwarnings('ignore')

# =============================================================================
# 配置
# =============================================================================
DATA_DIR = 'data/Hateful_Memes/data'
CHECKPOINT_DIR = 'checkpoints/hateful_memes'
RESULT_DIR = 'results/hateful_memes'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

NUM_CLASSES = 2
DEFAULT_PROVIDERS = ['deepseek', 'gemini', 'gpt5.1']


# =============================================================================
# 数据加载
# =============================================================================
def load_test_data(max_samples=None):
    test_path = os.path.join(DATA_DIR, 'test.jsonl')
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    test_ids = [item['id'] for item in test_data]
    test_texts = [item['text'] for item in test_data]
    
    test_images = []
    for item in test_data:
        img_path = os.path.join(DATA_DIR, item['img'])
        if not os.path.exists(img_path):
            img_path = os.path.join(DATA_DIR, 'img', os.path.basename(item['img']))
        test_images.append(Image.open(img_path).convert('RGB'))
    
    print(f"  测试集样本数: {len(test_data)}")
    return test_ids, test_texts, test_images


# =============================================================================
# LLM推理
# =============================================================================
def run_llm_inference(ids, texts, images, seed, providers, force_rerun=False):
    seed_suffix = f'_seed{seed}'
    all_beliefs = []
    all_uncertainties = []
    
    for i, provider in enumerate(providers):
        cache_path = os.path.join(CHECKPOINT_DIR, f'llm_test_agent{i}{seed_suffix}.pt')
        
        if os.path.exists(cache_path) and not force_rerun:
            print(f"  Agent{i} ({provider}): 加载已有缓存")
            cache = torch.load(cache_path, map_location='cpu')
            all_beliefs.append(cache['beliefs'])
            all_uncertainties.append(cache['uncertainties'])
            continue
        
        print(f"  Agent{i} ({provider}): 开始推理 {len(ids)} 样本...")
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
            use_direct_image=(i >= 1 and provider in ['glm', 'gpt', 'gpt5.1', 'gpt4om', 'gemini']),
            verbose=False,
        )
        
        beliefs = torch.zeros(len(ids), NUM_CLASSES)
        uncertainties = torch.zeros(len(ids))
        
        for j in tqdm(range(len(ids)), desc=f"  Agent{i}"):
            try:
                alpha, belief, uncertainty, emb = agent.forward(texts[j], image=images[j])
                
                if belief.dim() > 1:
                    beliefs[j] = belief[0]
                else:
                    beliefs[j] = belief
                
                if uncertainty.dim() > 0:
                    uncertainties[j] = uncertainty.item() if uncertainty.numel() == 1 else uncertainty.mean().item()
                else:
                    uncertainties[j] = float(uncertainty)
            except Exception as e:
                print(f"  [警告] Agent{i} 样本{j} API调用失败: {e}")
                beliefs[j, 0] = 0.5
                beliefs[j, 1] = 0.5
                uncertainties[j] = 1.0
            
            time.sleep(min_interval)
        
        all_beliefs.append(beliefs)
        all_uncertainties.append(uncertainties)
        
        torch.save({'beliefs': beliefs, 'uncertainties': uncertainties}, cache_path)
        print(f"  缓存已保存: {cache_path}")
    
    return all_beliefs, all_uncertainties


# =============================================================================
# 因果反思
# =============================================================================
def run_causal_reflection_on_test(ids, texts, images, all_beliefs, all_uncertainties, seed, providers):
    seed_suffix = f'_seed{seed}'
    N = len(ids)
    n_agents = len(providers)
    
    mv_predictions = []
    disagreement_indices = []
    
    for j in range(N):
        preds = [all_beliefs[a][j].argmax().item() for a in range(n_agents)]
        counter = Counter(preds)
        mv_pred = counter.most_common(1)[0][0]
        mv_predictions.append(mv_pred)
        
        if len(counter) > 1:
            disagreement_indices.append(j)
    
    print(f"  MV分歧样本: {len(disagreement_indices)}/{N}")
    
    if len(disagreement_indices) == 0:
        print("  无分歧样本，跳过因果反思")
        return mv_predictions, {}
    
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
            use_direct_image=(i >= 1 and provider in ['glm', 'gpt', 'gpt5.1', 'gpt4om', 'gemini']),
            verbose=False,
        )
        agents.append(agent)
    
    final_predictions = mv_predictions.copy()
    reflection_results = {}
    n_corrected = 0
    total_api_calls = 0
    
    for idx in tqdm(disagreement_indices, desc="因果反思"):
        preds = [all_beliefs[a][idx].argmax().item() for a in range(n_agents)]
        
        new_preds = []
        for a in range(n_agents):
            other_agents = [b for b in range(n_agents) if b != a]
            other_preds = [all_beliefs[b][idx].argmax().item() for b in other_agents]
            other_confidences = [all_beliefs[b][idx, other_preds[bi]].item() for bi, b in enumerate(other_agents)]
            
            evidence_parts = []
            for bi, b in enumerate(other_agents):
                other_label = '仇恨' if other_preds[bi] == 1 else '非仇恨'
                evidence_parts.append(
                    f"Agent{b+1}（{providers[b]}）认为答案是{other_label}，置信度{other_confidences[bi]:.3f}"
                )
            evidence_text = "；".join(evidence_parts)
            
            my_label = '仇恨' if preds[a] == 1 else '非仇恨'
            reflection_prompt = (
                f"你之前对以下内容做出了判断。现在请参考其他Agent的判断重新审视：\n\n"
                f"文本：{texts[idx]}\n"
                f"你的原始判断：{my_label}\n\n"
                f"其他Agent的判断：\n{evidence_text}\n\n"
                f"请重新评估，只回答 0（非仇恨）或 1（仇恨）。"
            )
            
            try:
                system_prompt = agents[a].system_prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": reflection_prompt}
                ]
                response = agents[a].client.chat(messages)
                total_api_calls += 1
                
                response_text = response.get('content', '').strip()
                if '1' in response_text and '0' not in response_text.replace('1', ''):
                    new_pred = 1
                elif '0' in response_text:
                    new_pred = 0
                else:
                    new_pred = preds[a]
                
                new_preds.append(new_pred)
            except Exception as e:
                print(f"  [警告] 反思Agent{a} 样本{idx} API调用失败: {e}")
                new_preds.append(preds[a])
            
            time.sleep(agents[a].client.min_call_interval)
        
        new_counter = Counter(new_preds)
        new_mv = new_counter.most_common(1)[0][0]
        
        if new_mv != mv_predictions[idx]:
            final_predictions[idx] = new_mv
            n_corrected += 1
        
        reflection_results[idx] = {
            'original_preds': preds,
            'new_preds': new_preds,
            'original_mv': mv_predictions[idx],
            'new_mv': new_mv,
            'changed': new_mv != mv_predictions[idx]
        }
    
    print(f"  反思修正样本数: {n_corrected}/{len(disagreement_indices)}")
    print(f"  额外API调用: {total_api_calls}")
    
    return final_predictions, reflection_results


# =============================================================================
# 生成提交文件
# =============================================================================
def generate_submission(ids, predictions, seed, method='causal_reflection'):
    import csv
    output_path = os.path.join(RESULT_DIR, f'test_predictions_{method}_seed{seed}.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'proba', 'label'])
        for i, pred in enumerate(predictions):
            proba = 0.8 if pred == 1 else 0.2
            writer.writerow([ids[i], proba, pred])
    print(f"  提交文件已保存: {output_path}")
    return output_path


# =============================================================================
# 主函数
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hateful Memes 测试集评估')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_test', type=int, default=None, help='最大测试样本数')
    parser.add_argument('--force_rerun', action='store_true', help='强制重新运行LLM推理')
    parser.add_argument('--skip_reflection', action='store_true', help='跳过因果反思，仅输出MV预测')
    parser.add_argument('--providers', nargs=3, default=DEFAULT_PROVIDERS, help='3个LLM provider')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Hateful Memes 测试集评估")
    print(f"Seed: {args.seed}")
    print(f"Providers: {args.providers}")
    print("=" * 70)
    
    print("\n[1] 加载测试集数据...")
    ids, texts, images = load_test_data(max_samples=args.max_test)
    
    print("\n[2] LLM推理（3个Agent）...")
    all_beliefs, all_uncertainties = run_llm_inference(
        ids, texts, images, args.seed, args.providers, force_rerun=args.force_rerun
    )
    
    if not args.skip_reflection:
        print("\n[3] 因果反思...")
        final_preds, reflection_results = run_causal_reflection_on_test(
            ids, texts, images, all_beliefs, all_uncertainties, args.seed, args.providers
        )
    else:
        print("\n[3] 跳过因果反思，使用MV预测...")
        n_agents = len(args.providers)
        final_preds = []
        for j in range(len(ids)):
            preds = [all_beliefs[a][j].argmax().item() for a in range(n_agents)]
            final_preds.append(Counter(preds).most_common(1)[0][0])
        reflection_results = {}
    
    print("\n[4] 生成EvalAI提交文件...")
    method = 'mv' if args.skip_reflection else 'causal_reflection'
    generate_submission(ids, final_preds, args.seed, method)
    
    print("\n[5] 统计信息...")
    n_agents = len(args.providers)
    mv_preds = []
    for j in range(len(ids)):
        preds = [all_beliefs[a][j].argmax().item() for a in range(n_agents)]
        mv_preds.append(Counter(preds).most_common(1)[0][0])
    
    n_hateful_mv = sum(1 for p in mv_preds if p == 1)
    n_hateful_cr = sum(1 for p in final_preds if p == 1)
    n_changed = sum(1 for i in range(len(ids)) if mv_preds[i] != final_preds[i])
    
    print(f"  总样本数: {len(ids)}")
    print(f"  MV预测仇恨: {n_hateful_mv} ({n_hateful_mv/len(ids)*100:.1f}%)")
    print(f"  CR预测仇恨: {n_hateful_cr} ({n_hateful_cr/len(ids)*100:.1f}%)")
    print(f"  反思修正样本: {n_changed}")
    
    result_path = os.path.join(RESULT_DIR, f'test_results_seed{args.seed}.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': args.seed,
            'n_samples': len(ids),
            'n_hateful_mv': n_hateful_mv,
            'n_hateful_cr': n_hateful_cr,
            'n_changed': n_changed,
            'n_reflection_samples': len(reflection_results),
        }, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {result_path}")
    
    print("\n" + "=" * 70)
    print("测试集评估完成！")
    print(f"提交文件: results/hateful_memes/test_predictions_{method}_seed{args.seed}.csv")
    print("请将上述CSV文件提交到 EvalAI: https://eval.ai/web/challenges/challenge-page/705/")
    print("=" * 70)


if __name__ == '__main__':
    main()