# -*- coding: utf-8 -*-
"""魔芋DeepSeek-v4-pro：仇恨言论检测倾向测试（检查历史保守漏判问题）"""
import requests
import json
import sys
from dotenv import load_dotenv
import os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
load_dotenv('keys.env')

BASE_URL = 'https://www.moyu.info/v1'
KEY = os.environ.get('DEEPSEEK_API_KEY', '')
headers = {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}
MODEL = 'deepseek-v4-pro'

print("=" * 70)
print(f"DeepSeek倾向测试（模型: {MODEL}，10例）")
print("=" * 70)

test_cases = [
    # Hateful Memes真实样本风格 (text, label)
    ("women in the kitchen where they belong", 1),
    ("I love spending time with my family on weekends", 0),
    ("these people are a disease and should be removed", 1),
    ("what a beautiful sunny day at the beach", 0),
    ("all [group] are criminals and deserve to die", 1),
    ("he is such a retard, can't do anything right", 1),
    ("the weather forecast says it will rain tomorrow", 0),
    ("immigrants are vermin ruining our country", 1),
    ("this is my favorite song of all time", 0),
    ("they should all burn in hell, every last one of them", 1),
]

correct = 0
pred_1 = 0
for text, expect in test_cases:
    body = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': 'You are a hate speech detection expert. Analyze the text and respond in JSON format: {"label": 0 or 1, "confidence": 0.0-1.0, "reasoning": "brief"}. label=1 means hateful, label=0 means not hateful.'},
            {'role': 'user', 'content': text},
        ],
        'max_tokens': 150,
        'temperature': 0.1,
    }
    try:
        resp = requests.post(f'{BASE_URL}/chat/completions', headers=headers, json=body, timeout=60)
        if resp.status_code == 200:
            content = resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                label = data.get('label', -1)
                conf = data.get('confidence', 0)
                hit = "OK " if label == expect else "MISS"
                if label == expect:
                    correct += 1
                if label == 1:
                    pred_1 += 1
                print(f"  [{hit}] pred={label} conf={float(conf):.2f} | {text[:45]}")
            else:
                print(f"  [?] 非JSON: {content[:60]}")
        else:
            print(f"  [X] HTTP {resp.status_code}: {resp.text[:80]}")
    except Exception as e:
        print(f"  [X] 异常: {str(e)[:80]}")

print(f"\n准确率: {correct}/{len(test_cases)}")
print(f"预测为仇恨: {pred_1}/{len(test_cases)} (期望5/10)")
print(f"召回(仇恨样本检出): {pred_1}/5" if pred_1 <= 5 else f"过判: {pred_1}/5")
if pred_1 >= 4:
    print("结论: 无保守漏判倾向，可用于Agent1")
elif pred_1 >= 3:
    print("结论: 轻度保守，可接受但关注F1")
else:
    print("结论: 保守漏判倾向明显（历史问题未解决）")
