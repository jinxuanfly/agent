"""
大模型API调用层
==============
统一接口，支持多种模型API（DeepSeek、Qwen、GLM、GPT等）
通过API调用替代本地弱Agent，为共识框架提供高质量信念输出。

用法:
    from src.llm_api import LLMClient
    client = LLMClient(provider='deepseek', model='deepseek-chat', api_key='sk-xxx')
    response = client.chat(messages=[...])
"""

import os
import json
import time
import re
import base64
import requests
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from threading import Lock

try:
    from dotenv import load_dotenv
    keys_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'keys.env')
    if os.path.exists(keys_path):
        load_dotenv(keys_path)
        print(f"  [LLM] 已加载密钥文件: {keys_path}")
    else:
        print(f"  [LLM] 未找到密钥文件: {keys_path}")
except ImportError:
    pass

# 全局速率限制器（按provider分组）
_global_rate_limiters = {}
_global_rate_lock = Lock()

class RateLimiter:
    """简单的速率限制器，控制API调用频率"""
    def __init__(self, min_interval=1.0):
        self.min_interval = min_interval
        self.last_call_time = 0
        self.lock = Lock()
    
    def wait(self):
        """等待直到可以进行下一次调用"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_call_time = time.time()

def get_rate_limiter(provider: str, min_interval: float = 1.0) -> RateLimiter:
    """获取指定provider的速率限制器"""
    global _global_rate_limiters
    with _global_rate_lock:
        if provider not in _global_rate_limiters:
            _global_rate_limiters[provider] = RateLimiter(min_interval=min_interval)
        return _global_rate_limiters[provider]


# =============================================================================
# 支持的模型提供者配置
# =============================================================================

PROVIDER_CONFIGS = {
    'deepseek': {
        'base_url': 'https://www.moyu.info/v1',
        'default_model': 'deepseek-v4-pro',
        'env_key': 'DEEPSEEK_API_KEY',
        'headers': {'Content-Type': 'application/json'},
    },
    'qwen': {
        'base_url': 'https://ws-1gt9f32njcy69eh1.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
        'default_model': 'qwen3.6-plus',
        'env_key': 'QWEN_API_KEY',
        'headers': {'Content-Type': 'application/json'},
        'is_dashscope': False,
    },
    'glm': {
        'base_url': 'https://open.bigmodel.cn/api/paas/v4',
        'default_model': 'GLM-5V-Turbo',
        'env_key': 'GLM_API_KEY',
        'headers': {'Content-Type': 'application/json'},
    },
    # 'gpt': {
    #     'base_url': 'https://www.moyu.info/v1',
    #     'default_model': 'gpt-4o-mini-2024-07-18',
    #     'env_key': 'OPENAI_API_KEY',
    #     'headers': {'Content-Type': 'application/json'},
    # },
    'gpt5.1': {
        'base_url': 'https://www.moyu.info/v1',
        'default_model': 'gpt-5.1',
        'env_key': 'OPENAI_GPT5.1_API_KEY',
        'headers': {'Content-Type': 'application/json'},
        'is_reasoning_model': True,  # GPT-5是推理模型，需要特殊处理
        'default_max_tokens': 2000,  # 推理模型需要更多token
        'reasoning_effort': 'minimal',  # 减少推理token消耗
    },
    'gpt4om': {
        'base_url': 'https://www.moyu.info/v1',
        'default_model': 'gpt-4o-mini-2024-07-18',
        'env_key': 'OPENAI_GPT4OM_API_KEY',
        'headers': {'Content-Type': 'application/json'},
    },
    'gemini': {
        'base_url': 'https://www.moyu.info/v1',
        'default_model': 'gemini-3.5-flash',
        'env_key': 'GEMINI_API_KEY',
        'headers': {'Content-Type': 'application/json'},
        'is_reasoning_model': True,  # Gemini-3.5-flash也是推理模型
        'default_max_tokens': 2000,
        'reasoning_effort': 'minimal',
    },
    'claude': {
        'base_url': 'https://www.moyu.info/v1',
        'default_model': 'claude-sonnet-5',
        'env_key': 'ANTHROPIC_API_KEY',
        'headers': {'Content-Type': 'application/json'},
    },
}

# 默认分类系统提示词
SYSTEM_PROMPT_CLASSIFICATION = """你是一个专业的仇恨言论检测助手。请仔细分析社交媒体内容，严格识别仇恨言论。

仇恨言论的定义：针对特定群体（种族、宗教、性别、性取向、国籍、残疾等）的攻击性、贬低性、歧视性或煽动性言论。

请特别关注以下类型：
1. 种族/族裔歧视：使用种族歧视词汇，或基于肤色、国籍的刻板印象攻击
2. 宗教仇恨：攻击特定宗教群体，关联恐怖主义刻板印象
3. 性别歧视：贬低女性、跨性别者，使用性别侮辱性词汇
4. 性取向歧视：攻击LGBTQ+群体
5. 残疾歧视：嘲笑或侮辱残疾人
6. 仇恨符号与隐喻：使用纳粹符号、KKK相关内容、大屠杀隐喻等
7. 煽动暴力：号召对特定群体实施暴力

判断原则：
- 如果文本使用了明确的仇恨词汇或针对群体的攻击性语言 → 判定为仇恨言论(label=1)
- 如果文本包含刻板印象或偏见性描述 → 判定为仇恨言论(label=1)
- 如果文本包含讽刺但核心仍是仇恨 → 判定为仇恨言论(label=1)
- 如果文本只是中性陈述事实、合理批评或个人攻击（不针对群体）→ 判定为非仇恨言论(label=0)
- 不确定时，倾向于判定为仇恨言论，宁可错判不可漏判

请严格输出JSON格式：
{
    "label": 0 或 1,
    "confidence": 0.0-1.0,
    "probs": [0.0, 0.0],
    "reasoning": "简要推理过程"
}

注意：
1. confidence是你对判断的校准置信度
2. probs = [P(非仇恨), P(仇恨)]，两项和为1
3. 如果内容模糊或信息不足，请降低confidence并让probs接近[0.5, 0.5]
4. 即使很有把握，confidence也不应超过0.95
"""


# =============================================================================
# LLM API 客户端
# =============================================================================

class LLMClient:
    """统一的大模型API客户端"""

    def __init__(
        self,
        provider: str = 'deepseek',
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 5,
        timeout: int = 120,
        min_call_interval: float = 1.0,
    ):
        """
        初始化LLM客户端

        Args:
            provider: 模型提供者，支持 'deepseek', 'qwen', 'glm', 'gpt'
            model: 模型名称，None则使用provider的默认模型
            api_key: API密钥，None则从环境变量读取
            temperature: 生成温度（0.0-1.0），越低越确定
            max_retries: 最大重试次数
            timeout: 超时秒数
            min_call_interval: 最小调用间隔（秒），防止限流
        """
        if provider not in PROVIDER_CONFIGS:
            if provider == 'mock':
                # Mock mode for testing
                self.provider = 'mock'
                self._config = {}
                self.base_url = ''
                self.model = 'mock'
                self.api_key = ''
                self.headers = {}
                self.temperature = temperature
                self.max_retries = 0
                self.timeout = 1
                
                self.min_call_interval = min_call_interval
                self._rate_limiter = None
                self._last_call_time = 0
                
                self.mock_mode = True
                self._allow_mock_fallback = True
                self.call_count = 0
                self.total_tokens = 0
                self.total_cost = 0.0
                self._session_id = None
                self._cache = {}
                self._last_error = None
                return
            else:
                raise ValueError(f"不支持的provider: {provider}，可选: {list(PROVIDER_CONFIGS.keys())}")

        config = PROVIDER_CONFIGS[provider]
        self.provider = provider
        self._config = config
        self.base_url = config['base_url']
        self.model = model or config['default_model']
        self.api_key = api_key or os.environ.get(config['env_key'], '')
        self.headers = config['headers'].copy()
        if config.get('is_dashscope', False):
            self.headers['Authorization'] = f"Bearer {self.api_key}"
            self.headers['X-DashScope-SSE'] = 'false'
        else:
            self.headers['Authorization'] = f"Bearer {self.api_key}"
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.min_call_interval = min_call_interval
        self._rate_limiter = get_rate_limiter(provider, min_interval=min_call_interval)
        self._last_call_time = 0

        if not self.api_key:
            raise ValueError(f"未设置{config['env_key']}环境变量，请在keys.env中配置")
        self.mock_mode = False
        self._allow_mock_fallback = False

        self.call_count = 0
        self.total_tokens = 0

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict] = None,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """调用LLM聊天接口"""
        if self.mock_mode:
            return self._mock_response(messages)

        temp = temperature if temperature is not None else self.temperature

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        body = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temp,
        }
        if response_format:
            body["response_format"] = response_format

        # GPT-5推理模型适配：自动添加reasoning_effort参数
        if self._config.get('is_reasoning_model', False):
            # 只有GPT-5支持reasoning_effort参数
            if self.provider == 'gpt5.1':
                body['reasoning_effort'] = self._config.get('reasoning_effort', 'minimal')
            # 如果max_tokens太小，自动增大到默认值
            if max_tokens < self._config.get('default_max_tokens', 2000):
                body['max_tokens'] = self._config['default_max_tokens']

        url = f"{self.base_url}/chat/completions"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter.wait()
                
                resp = requests.post(
                    url, headers=self.headers, json=body, timeout=self.timeout
                )
                resp.raise_for_status()
                result = resp.json()

                self.call_count += 1
                if 'usage' in result:
                    self.total_tokens += result['usage'].get('total_tokens', 0)

                choice = result['choices'][0]
                content = choice.get('message', {}).get('content', '')
                finish_reason = choice.get('finish_reason', '')

                return {
                    'content': content,
                    'finish_reason': finish_reason,
                    'usage': result.get('usage', {}),
                    'success': True,
                }

            except requests.exceptions.Timeout as e:
                last_error = f"超时: {e}"
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** (attempt + 1), 60)
                    print(f"  [重试] 超时，等待{wait_time}秒后重试 ({attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if hasattr(e, 'response') else 0
                if status == 429:
                    last_error = f"限流: {e}"
                    if self.provider == 'glm':
                        wait_time = min(30 * (2 ** attempt), 300)
                    else:
                        wait_time = min(10 * (2 ** attempt), 120)
                    print(f"  [重试] 限流(429)，等待{wait_time}秒后重试 ({attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
                elif status >= 500:
                    last_error = f"服务端错误({status}): {e}"
                    if attempt < 3:
                        wait_time = min(2 ** (attempt + 1), 10)
                        print(f"  [重试] 服务端错误({status})，等待{wait_time}秒后重试 ({attempt+1}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"  [跳过] 服务端错误({status})，已重试{attempt+1}次，返回fallback")
                        break
                else:
                    last_error = f"HTTP错误({status}): {e}"
                    if attempt < self.max_retries - 1:
                        wait_time = min(2 ** (attempt + 1), 30)
                        time.sleep(wait_time)
            except Exception as e:
                last_error = f"未知错误: {e}"
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** (attempt + 1), 30)
                    time.sleep(wait_time)

        print(f"  [警告] API调用失败({self.provider}/{self.model}): {last_error}，返回fallback")
        return {
            'content': '{"label": 0, "confidence": 0.5, "probs": [0.5, 0.5], "reasoning": "fallback_uniform"}',
            'finish_reason': 'error',
            'usage': {},
            'success': True,
        }

    def _image_to_base64(self, image, max_size=800, quality=80):
        """将PIL Image转换为base64编码（限制大小避免400错误）"""
        import io
        width, height = image.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return b64_str

    def chat_with_image(
        self,
        text: str,
        image,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict] = None,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """调用支持图像输入的LLM聊天接口（GLM-5V-Turbo、GPT、Gemini等）"""
        if self.mock_mode:
            return self._mock_response([{"role": "user", "content": text}])

        temp = temperature if temperature is not None else self.temperature
        last_error = None

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        url = f"{self.base_url}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                self._rate_limiter.wait()
                
                if image is not None:
                    quality = max(40, 80 - attempt * 15)
                    image_base64 = self._image_to_base64(image, quality=quality)
                    content = [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                else:
                    content = text

                full_messages_current = full_messages.copy()
                full_messages_current.append({"role": "user", "content": content})

                body = {
                    "model": self.model,
                    "messages": full_messages_current,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                }
                if response_format:
                    body["response_format"] = response_format

                # GPT-5推理模型适配
                if self._config.get('is_reasoning_model', False):
                    if self.provider == 'gpt5.1':
                        body['reasoning_effort'] = self._config.get('reasoning_effort', 'minimal')
                    if max_tokens < self._config.get('default_max_tokens', 2000):
                        body['max_tokens'] = self._config['default_max_tokens']

                resp = requests.post(
                    url, headers=self.headers, json=body, timeout=self.timeout
                )
                resp.raise_for_status()
                result = resp.json()

                self.call_count += 1
                if 'usage' in result:
                    self.total_tokens += result['usage'].get('total_tokens', 0)

                choice = result['choices'][0]
                content = choice.get('message', {}).get('content', '')
                finish_reason = choice.get('finish_reason', '')

                return {
                    'content': content,
                    'finish_reason': finish_reason,
                    'usage': result.get('usage', {}),
                    'success': True,
                }

            except requests.exceptions.Timeout as e:
                last_error = f"超时: {e}"
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** (attempt + 1), 60)
                    print(f"  [重试] 超时，等待{wait_time}秒后重试 ({attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if hasattr(e, 'response') else 0
                if status == 429:
                    last_error = f"限流: {e}"
                    if self.provider == 'glm':
                        wait_time = min(30 * (2 ** attempt), 300)
                    else:
                        wait_time = min(10 * (2 ** attempt), 120)
                    print(f"  [重试] 限流(429)，等待{wait_time}秒后重试 ({attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
                elif status >= 500:
                    last_error = f"服务端错误({status}): {e}"
                    if attempt < 3:
                        wait_time = min(2 ** (attempt + 1), 10)
                        print(f"  [重试] 服务端错误({status})，等待{wait_time}秒后重试 ({attempt+1}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"  [跳过] 服务端错误({status})，已重试{attempt+1}次，返回fallback")
                        break
                elif status == 400:
                    last_error = f"请求错误({status}): {e}"
                    if attempt < self.max_retries - 1:
                        wait_time = min(2 ** (attempt + 1), 30)
                        print(f"  [重试] 请求错误({status})，等待{wait_time}秒后重试 ({attempt+1}/{self.max_retries})")
                        time.sleep(wait_time)
                else:
                    last_error = f"HTTP错误({status}): {e}"
                    if attempt < self.max_retries - 1:
                        wait_time = min(2 ** (attempt + 1), 30)
                        time.sleep(wait_time)
            except Exception as e:
                last_error = f"未知错误: {e}"
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** (attempt + 1), 30)
                    time.sleep(wait_time)

        print(f"  [警告] LLM调用失败，返回fallback: {last_error}")
        return {
            'content': '{"label": 0, "confidence": 0.5, "probs": [0.5, 0.5], "reasoning": "fallback_uniform"}',
            'finish_reason': 'error',
            'usage': {},
            'success': True,
        }

    def classify(
        self,
        text: str,
        image_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        统一分类接口（仅文本或文本+图像描述）

        Returns:
            {'label': 0|1, 'confidence': float, 'probs': [p0, p1],
             'reasoning': str, 'success': bool}
        """
        prompt = text
        if image_description:
            prompt = f"【图像描述】{image_description}\n\n【文本内容】{text}"

        result = self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt or SYSTEM_PROMPT_CLASSIFICATION,
            response_format={"type": "json_object"},
        )
        return self._parse_classification(result)

    def classify_with_image(
        self,
        text: str,
        image,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        统一分类接口（支持直接图像输入，用于GLM-5V-Turbo等多模态模型）

        Args:
            text: 文本内容
            image: PIL Image对象或None
            system_prompt: 系统提示词

        Returns:
            {'label': 0|1, 'confidence': float, 'probs': [p0, p1],
             'reasoning': str, 'success': bool}
        """
        result = self.chat_with_image(
            text=text,
            image=image,
            system_prompt=system_prompt or SYSTEM_PROMPT_CLASSIFICATION,
            response_format={"type": "json_object"},
        )
        return self._parse_classification(result)

    def _parse_classification(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """解析分类响应"""
        content = result.get('content', '').strip()

        if not content:
            return {**self._default_fallback(), 'success': result.get('success', False),
                    'error': result.get('error', '空响应')}

        # 提取JSON
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                label = int(data.get('label', 0))
                confidence = float(data.get('confidence', 0.5))
                probs = data.get('probs', None)
                reasoning = data.get('reasoning', '')

                if probs is None or len(probs) != 2:
                    probs = [1 - confidence, confidence] if label == 1 else [confidence, 1 - confidence]
                else:
                    probs = [float(p) for p in probs]
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]

                return {
                    'label': label, 'confidence': min(max(confidence, 0.0), 1.0),
                    'probs': probs, 'reasoning': reasoning, 'success': True,
                }
            except (json.JSONDecodeError, ValueError):
                pass

        return self._default_fallback()

    def _default_fallback(self) -> Dict[str, Any]:
        """默认回退：均匀分布"""
        return {
            'label': 0, 'confidence': 0.5,
            'probs': [0.5, 0.5], 'reasoning': 'fallback_uniform', 'success': True,
        }

    def _mock_response(self, messages: List[Dict]) -> Dict[str, Any]:
        """无API key时模拟响应"""
        self.call_count += 1
        last_msg = messages[-1]['content'] if messages else ''
        hate_words = ['hate', 'kill', 'stupid', 'idiot', 'dumb', 'terrorist']
        is_hate = any(w in last_msg.lower() for w in hate_words)
        mock_conf = 0.75 if is_hate else 0.80

        return {
            'content': json.dumps({
                'label': 1 if is_hate else 0, 'confidence': mock_conf,
                'probs': [1-mock_conf, mock_conf] if is_hate else [mock_conf, 1-mock_conf],
                'reasoning': 'keyword-based simulation',
            }),
            'finish_reason': 'stop', 'usage': {'total_tokens': 50}, 'success': True,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'provider': self.provider, 'model': self.model,
            'call_count': self.call_count, 'total_tokens': self.total_tokens,
            'mock_mode': self.mock_mode,
        }


# =============================================================================
# 批量分类器
# =============================================================================

class BatchClassifier:
    """
    批量分类器，支持缓存
    将LLM输出映射为证据头格式 (alpha, belief, uncertainty)
    """

    def __init__(
        self,
        client: LLMClient,
        batch_size: int = 8,
        verbose: bool = True,
    ):
        self.client = client
        self.batch_size = batch_size
        self.verbose = verbose
        self.cache = {}

    def classify_batch(
        self,
        texts: List[str],
        image_descriptions: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """批量分类"""
        N = len(texts)
        results = [None] * N
        cached_count = 0

        for i in range(N):
            if use_cache and texts[i] in self.cache:
                results[i] = self.cache[texts[i]]
                cached_count += 1

        if cached_count > 0 and self.verbose:
            print(f"    缓存命中: {cached_count}/{N}")

        to_call = [i for i in range(N) if results[i] is None]
        if not to_call:
            return results

        if self.verbose:
            print(f"    调用API: {len(to_call)}/{N} 样本 (batch_size={self.batch_size})")

        for batch_start in range(0, len(to_call), self.batch_size):
            batch_indices = to_call[batch_start:batch_start + self.batch_size]
            for idx in batch_indices:
                text = texts[idx]
                img_desc = image_descriptions[idx] if image_descriptions else None
                result = self.client.classify(
                    text=text, image_description=img_desc, system_prompt=system_prompt,
                )
                results[idx] = result
                self.cache[text] = result
                if self.verbose and (idx + 1) % 5 == 0:
                    print(f"      [{idx+1}/{N}] processed...")

        return results

    def to_evidential(
        self, results: List[Dict[str, Any]], num_classes: int = 2,
    ):
        """
        将LLM分类结果转换为证据头格式

        Returns:
            alpha: [N, C], belief: [N, C], uncertainty: [N]
        """
        import torch
        N = len(results)
        alpha = torch.zeros(N, num_classes)

        for i, res in enumerate(results):
            if res.get('success', False):
                probs = res['probs']
                confidence = res['confidence']
                # u = 1 - confidence, 保证在 [0.05, 0.95]
                u = max(min(1.0 - confidence, 0.95), 0.05)
                S = num_classes / u
                evidence = S * torch.tensor(probs)
                alpha[i] = evidence + 1.0
            else:
                alpha[i] = torch.ones(num_classes) * 1.5  # 高不确定性

        S = alpha.sum(dim=-1, keepdim=True)
        belief = alpha / S
        uncertainty = num_classes / S.squeeze(-1)
        return alpha, belief, uncertainty


# =============================================================================
# 便捷函数
# =============================================================================

def create_client(provider: str = 'deepseek', **kwargs) -> LLMClient:
    return LLMClient(provider=provider, **kwargs)


def verify_api_key(provider: str) -> bool:
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        return False
    return len(os.environ.get(config['env_key'], '')) > 0