import os

filepath = r"E:\Agent论文\perfect\sota_comparison.py"

with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 简化版本的 multi_role - 完全重写
new_multi_role = '''
def run_multi_role(client, texts, images, cache_path=None):
    if cache_path:
        c = load_cache(cache_path)
        if c and len(c)==len(texts):
            print(f"  [Cache] Multi-Role: {len(c)} samples")
            return c
    results = []
    for i in tqdm(range(len(texts)), desc="Multi-Role"):
        label_map = {0: "non-hate", 1: "hate"}
        rt = {"label":0,"confidence":0.3,"success":False}
        try:
            rt = client.classify(text=texts[i], system_prompt="You are a text expert. Is this hate speech? JSON: {label:0/1, confidence:0-1}")
        except:
            pass
        
        # Build simple judge input
        p_text = "Text expert: " + label_map.get(rt.get("label",0), "non-hate")
        p_text += " conf=" + str(round(rt.get("confidence",0.3), 2))
        p_text += ". What is the final verdict?"
        
        final = {"label": rt.get("label",0), "confidence": rt.get("confidence",0.3)*0.8}
        try:
            rj = client.classify(text=p_text, system_prompt="You are the chief judge. Give final verdict as JSON: {label:0/1, confidence:0-1}")
            if rj.get("success", False):
                final = {"label": rj["label"], "confidence": rj["confidence"]}
        except:
            pass
        results.append(final)
    if cache_path: save_cache(results, cache_path)
    return results
'''

# 找到旧函数并替换
import re
# 匹配从 def run_multi_role 到函数结束（下一个def或文件末尾）
pattern = r'def run_multi_role\(.*?\n(?:.*\n)*?(?=\ndef |\Z)'
match = re.search(pattern, code, re.DOTALL)
if match:
    code = code[:match.start()] + new_multi_role + "\n" + code[match.end():]
    print("Replaced run_multi_role successfully!")
else:
    print("Could not find run_multi_role function")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
print("Done! File saved.")
