import requests
import os
from dotenv import load_dotenv

print("=== 调试密钥加载 ===")

with open('keys.env', 'r', encoding='utf-8') as f:
    content = f.read()
    print(f"文件内容长度: {len(content)}")
    
for line in content.split('\n'):
    if line.startswith('QWEN_API_KEY'):
        print(f"原始行: {line}")
        key = line.split('=', 1)[1]
        print(f"直接解析密钥长度: {len(key)}")
        print(f"直接解析密钥前20字符: {key[:20]}")
        print(f"直接解析密钥后20字符: {key[-20:]}")

load_dotenv('keys.env', override=True)
api_key = os.getenv('QWEN_API_KEY')

print(f"\ndotenv加载密钥长度: {len(api_key) if api_key else 0}")
print(f"dotenv加载密钥前20字符: {api_key[:20] if api_key else 'None'}")

url = 'https://ws-1gt9f32njcy69eh1.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}
body = {
    'model': 'qwen3.6-plus',
    'messages': [{'role': 'user', 'content': 'Hello'}],
    'max_tokens': 50,
    'temperature': 0.1,
}

try:
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    print(f"\nStatus Code: {resp.status_code}")
    print(f"Response: {resp.text[:800]}")
except Exception as e:
    print(f"\nError: {e}")
