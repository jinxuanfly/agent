import os, sys, time, json, pickle, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from sklearn.metrics import accuracy_score, f1_score
from src.llm_agent import AGENT_PROMPTS
from src.llm_api import LLMClient

warnings.filterwarnings('ignore')

DATA_DIR = 'data/Hateful_Memes/data'
CHECKPOINT_DIR = 'checkpoints/hateful_memes'
RESULT_DIR = 'results/hateful_memes'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

class HatefulMemesDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', max_samples=None, load_images=True, stratified=True):
        self.data = []
        json_path = os.path.join(DATA_DIR, f'{split}.jsonl')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        if stratified and max_samples and len(self.data) > max_samples:
            labels = [item['label'] for item in self.data]
            np.random.seed(42)
            sampled = []
            for label in set(labels):
                idx = [i for i,l in enumerate(labels) if l==label]
                n = max(1, max_samples * len(idx) // len(labels))
                sampled.extend(np.random.choice(idx, min(n,len(idx)), replace=False).tolist())
            self.data = [self.data[i] for i in sampled[:max_samples]]
        self.texts = [item['text'] for item in self.data]
        self.labels = [item['label'] for item in self.data]
        self.img_paths = []
        for item in self.data:
            img_name = item['img'].split('/')[-1]
            p = os.path.join(DATA_DIR, img_name)
            if not os.path.exists(p): p = os.path.join(DATA_DIR, 'img', img_name)
            self.img_paths.append(p)
        self.images = None
        if load_images:
            self.images = [Image.open(p).convert('RGB') if os.path.exists(p) else None for p in self.img_paths]
    def __len__(self):
        return len(self.data)

def load_cache(path):
    if os.path.exists(path):
        with open(path, 'rb') as f: return pickle.load(f)
    return None

def save_cache(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f: pickle.dump(data, f)

def compute_metrics(preds, labels, method_name=''):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    tn = int(((preds==0)&(labels==0)).sum())
    p = tp/(tp+fp) if tp+fp>0 else 0.0
    r = tp/(tp+fn) if tp+fn>0 else 0.0
    return {'method':method_name,'accuracy':float(acc),'f1_score':float(f1),'precision':float(p),'recall':float(r),'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def run_self_consistency(client, texts, images, cache_path=None, n=5):
    if cache_path:
        c = load_cache(cache_path)
        if c and len(c)==len(texts):
            print(f'  [Cache] Self-Consistency: {len(c)} samples')
            return c
    results = []
    sp = AGENT_PROMPTS['text_focused']
    for i in tqdm(range(len(texts)), desc='Self-Consistency (n=5)'):
        preds, confs = [], []
        for _ in range(n):
            try:
                if images and images[i] is not None:
                    r = client.classify_with_image(text=texts[i], image=images[i], system_prompt=sp, temperature=0.7)
                else:
                    r = client.classify(text=texts[i], system_prompt=sp, temperature=0.7)
                if r.get('success',False):
                    preds.append(r['label']); confs.append(r.get('confidence',0.5))
                else:
                    preds.append(np.random.randint(0,2)); confs.append(0.3)
            except:
                preds.append(np.random.randint(0,2)); confs.append(0.3)
        votes = np.zeros(2)
        for p,c in zip(preds,confs): votes[p]+=c
        results.append({'label':int(np.argmax(votes)),'confidence':float(np.mean(confs))})
    if cache_path: save_cache(results, cache_path)
    return results

def run_static_ensemble(clients, texts, images, cache_dir=None):
    all_results = []
    for idx,(prov,client) in enumerate(clients.items()):
        cp = os.path.join(cache_dir, f'ensemble_agent{idx}.pt') if cache_dir else None
        if cp:
            c = load_cache(cp)
            if c and len(c)==len(texts):
                print(f'  [Cache] Ensemble Agent{idx} ({prov}): {len(c)} samples')
                all_results.append(c); continue
        sp_list = [AGENT_PROMPTS['text_focused'], AGENT_PROMPTS['image_focused'], AGENT_PROMPTS['multimodal_fusion']]
        sp = sp_list[idx%3]
        ar = []
        for i in tqdm(range(len(texts)), desc=f'Ensemble Agent{idx} ({prov})'):
            try:
                if images and images[i] is not None:
                    r = client.classify_with_image(text=texts[i], image=images[i], system_prompt=sp)
                else:
                    r = client.classify(text=texts[i], system_prompt=sp)
                ar.append(r if r.get('success',False) else {'probs':[0.5,0.5],'label':0,'confidence':0.3})
            except:
                ar.append({'probs':[0.5,0.5],'label':0,'confidence':0.3})
        if cp: save_cache(ar, cp)
        all_results.append(ar)
    w = [1.0/len(clients)]*len(clients)
    final = []
    for i in range(len(texts)):
        ps = np.zeros(2)
        for ar,wt in zip(all_results,w): ps += np.array(ar[i].get('probs',[0.5,0.5]))*wt
        final.append({'label':int(np.argmax(ps)),'confidence':float(np.max(ps))})
    return final

def run_multi_role(client, texts, images, cache_path=None):
    if cache_path:
        c = load_cache(cache_path)
        if c and len(c)==len(texts):
            print(f'  [Cache] Multi-Role: {len(c)} samples')
            return c
    results = []
    for i in tqdm(range(len(texts)), desc='Multi-Role'):
        label_map = {0: 'non-hate', 1: 'hate'}
        rt = {'label':0,'confidence':0.3,'success':False}
        try:
            rt = client.classify(text=texts[i], system_prompt='You are a text expert. Is this hate speech? JSON: {label:0/1, confidence:0-1}')
        except:
            pass
        
        p_text = 'Text expert: ' + label_map.get(rt.get('label',0), 'non-hate')
        p_text += ' conf=' + str(round(rt.get('confidence',0.3), 2))
        p_text += '. What is the final verdict?'
        
        final = {'label': rt.get('label',0), 'confidence': rt.get('confidence',0.3)*0.8}
        try:
            rj = client.classify(text=p_text, system_prompt='You are the chief judge. Give final verdict as JSON: {label:0/1, confidence:0-1}')
            if rj.get('success', False):
                final = {'label': rj['label'], 'confidence': rj['confidence']}
        except:
            pass
        results.append(final)
    if cache_path: save_cache(results, cache_path)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_val', type=int, default=500)
    parser.add_argument('--methods', type=str, default=None)
    args = parser.parse_args()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print(f'Config: max_val={args.max_val}')
    ds = HatefulMemesDataset(split='dev', max_samples=args.max_val, load_images=True, stratified=True)
    print(f'Val set: {len(ds)} samples, labels: 0={ds.labels.count(0)}, 1={ds.labels.count(1)}')
    
    clients = {}
    # 尝试初始化多个provider
    for p in ['deepseek', 'gemini','gpt5.1']:
        try:
            client = LLMClient(provider=p)
            # 快速测试
            test = client.classify(text='test', system_prompt='Reply with OK')
            if test.get('success', False):
                clients[p] = client
                print(f'  {p}: initialized and working')
            else:
                print(f'  {p}: initialized but test failed, skipping')
        except Exception as e:
            print(f'  {p}: failed - {e}')
    
    # 如果只有1个client，尝试创建更多实例用不同prompt
    if len(clients) < 2:
        k = list(clients.keys())[0]
        clients[f'{k}_text'] = LLMClient(provider=k)
        clients[f'{k}_image'] = LLMClient(provider=k)
        print(f'  Created extra clients from {k}')
    
    print(f'  Total available agents: {len(clients)}')
    if not clients:
        print('ERROR: No LLM clients!'); return
    
    texts, images, labels = ds.texts, ds.images, ds.labels
    summary = []
    methods = args.methods.split(',') if args.methods else ['self_consistency','static_ensemble','multi_role']
    
    for m in methods:
        m = m.strip()
        print(f'\n{m.capitalize()}...')
        try:
            if m=='self_consistency':
                c = clients.get('deepseek', list(clients.values())[0])
                r = run_self_consistency(c, texts, images, os.path.join(CHECKPOINT_DIR,'sota_sc.pt'))
                p = [x['label'] for x in r]
                summary.append(compute_metrics(p, labels, 'Self-Consistency (n=5)'))
            elif m=='static_ensemble':
                r = run_static_ensemble(clients, texts, images, os.path.join(CHECKPOINT_DIR,'sota_ens'))
                p = [x['label'] for x in r]
                summary.append(compute_metrics(p, labels, f'Static Ensemble ({len(clients)} agents)'))
            elif m=='multi_role':
                c = clients.get('deepseek', list(clients.values())[0])
                r = run_multi_role(c, texts, images, os.path.join(CHECKPOINT_DIR,'sota_mr.pt'))
                p = [x['label'] for x in r]
                summary.append(compute_metrics(p, labels, 'Single LLM Multi-Role'))
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback; traceback.print_exc()
    
    if summary:
        df = pd.DataFrame(summary)
        print('\n' + df.to_string(index=False))
        ts = time.strftime('%Y%m%d_%H%M%S')
        out = os.path.join(RESULT_DIR, f'sota_{ts}.json')
        with open(out, 'w', encoding='utf-8') as f: json.dump({'timestamp':ts,'results':summary}, f, ensure_ascii=False, indent=2)
        print(f'Saved: {out}')

if __name__=='__main__':
    main()