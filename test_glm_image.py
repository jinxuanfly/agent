"""测试GLM-5V-Turbo多模态API调用"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_api import LLMClient

client = LLMClient(provider='glm', temperature=0.1, timeout=120)
print(f"模型: {client.provider}/{client.model}")
print(f"API密钥: {client.api_key[:20]}...")

from PIL import Image

data_dir = 'data/Hateful_Memes/data'
if os.path.exists(data_dir):
    img_dir = os.path.join(data_dir, 'img')
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if img_files:
        img_path = os.path.join(img_dir, img_files[0])
        img = Image.open(img_path)
        print(f"使用本地图像: {img_path}")
        print(f"图像尺寸: {img.size}")
    else:
        img = Image.new('RGB', (224, 224), color=(255, 100, 100))
        print("使用生成的测试图像")
else:
    img = Image.new('RGB', (224, 224), color=(255, 100, 100))
    print("使用生成的测试图像")

result = client.chat_with_image(
    text="分析这个图像，判断是否包含仇恨言论。请严格输出JSON格式：{\"label\": 0或1, \"confidence\": 0.0-1.0, \"probs\": [0.0, 0.0], \"reasoning\": \"简要推理过程\"}",
    image=img,
    system_prompt="你是一个专业的仇恨言论检测助手，专注于图像内容分析。",
    response_format={"type": "json_object"},
)

print(f"\n响应内容:")
print(result['content'])
print(f"\nfinish_reason: {result['finish_reason']}")
print(f"success: {result['success']}")

result2 = client.classify_with_image(
    text="分析这个图像，判断是否包含仇恨言论。",
    image=img,
)
print(f"\n分类结果:")
print(f"label: {result2['label']}")
print(f"confidence: {result2['confidence']}")
print(f"probs: {result2['probs']}")
print(f"reasoning: {result2['reasoning']}")
print(f"success: {result2['success']}")