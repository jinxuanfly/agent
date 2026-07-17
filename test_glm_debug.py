"""调试GLM-5V-Turbo 400错误"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_api import LLMClient
from PIL import Image
import base64
import io
import requests

client = LLMClient(provider='glm', temperature=0.1, timeout=120)
print(f"模型: {client.provider}/{client.model}")

data_dir = 'data/Hateful_Memes/data'
img_dir = os.path.join(data_dir, 'img')
img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

for i, fname in enumerate(img_files[:5]):
    img_path = os.path.join(img_dir, fname)
    img = Image.open(img_path)
    print(f"\n图像 {i}: {fname}, 尺寸={img.size}, 模式={img.mode}")
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print(f"  base64长度: {len(b64)}")
    
    try:
        result = client.chat_with_image(
            text="分析这个图像，判断是否包含仇恨言论。",
            image=img,
            system_prompt="你是一个专业的仇恨言论检测助手。",
        )
        print(f"  成功: {result['success']}, label={result['content'][:50]}")
    except Exception as e:
        print(f"  失败: {e}")