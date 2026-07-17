"""测试GLM-5V-Turbo JSON格式输出"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_api import LLMClient
from PIL import Image

client = LLMClient(provider='glm', temperature=0.1, timeout=120)
print(f"模型: {client.provider}/{client.model}")

data_dir = 'data/Hateful_Memes/data'
img_dir = os.path.join(data_dir, 'img')
img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

for i, fname in enumerate(img_files[:10]):
    img_path = os.path.join(img_dir, fname)
    img = Image.open(img_path)
    print(f"\n图像 {i}: {fname}, 尺寸={img.size}")
    
    try:
        result = client.classify_with_image(
            text="分析这个图像，判断是否包含仇恨言论。",
            image=img,
        )
        print(f"  成功: label={result['label']}, conf={result['confidence']}, reasoning={result['reasoning'][:50]}")
    except Exception as e:
        print(f"  失败: {e}")
        try:
            result = client.chat_with_image(
                text="分析这个图像，判断是否包含仇恨言论。",
                image=img,
            )
            print(f"  降级成功: {result['content'][:100]}")
        except Exception as e2:
            print(f"  降级也失败: {e2}")