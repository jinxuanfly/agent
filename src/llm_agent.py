"""
LLM Agent 包装器
================
将大模型API调用包装为「证据头」格式输出 (alpha, belief, uncertainty, embedding)
与现有共识框架完全兼容。

每个Agent使用不同的LLM（或不同的prompt策略），模拟多模态多智能体。

用法:
    from src.llm_agent import LLMAgent, create_llm_agents
    
    agents = create_llm_agents()
    alpha, belief, uncertainty, emb = agents[0].forward("some text")
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.llm_api import LLMClient, BatchClassifier, PROVIDER_CONFIGS


# =============================================================================
# LLM Agent 定义
# =============================================================================

# 不同Agent使用不同prompt策略，模拟不同"模态/视角"
AGENT_PROMPTS = {
    "text_focused": """你是一个专业的仇恨言论检测助手，专注于**文本分析**。

分析以下社交媒体内容的文本部分，判断是否包含仇恨言论。

请关注：
- 词汇选择：是否使用贬低性、攻击性词汇
- 语气：是否充满敌意
- 目标：是否针对特定群体

请严格输出JSON格式：
{
    "label": 0 或 1,
    "confidence": 0.0-1.0,
    "probs": [0.0, 0.0],
    "reasoning": "简要推理过程"
}
""",

    "context_aware": """你是一个专业的仇恨言论检测助手，擅长**结合上下文理解**。

分析以下社交媒体内容，判断是否包含仇恨言论。

请关注：
- 语境：是否为引用、反讽或报道性语言
- 作者意图：是攻击还是陈述事实
- 整体语义：不要被个别敏感词误导

请严格输出JSON格式：
{
    "label": 0 或 1,
    "confidence": 0.0-1.0,
    "probs": [0.0, 0.0],
    "reasoning": "简要推理过程"
}
""",

    "strict_detector": """你是一个专业的仇恨言论检测助手，采用**严格标准**。

分析以下社交媒体内容，判断是否包含仇恨言论。

采用严格标准：
- 任何针对群体的攻击性、贬低性言论 → 仇恨言论
- 即使是以玩笑形式 → 如果内容具有攻击性，仍判为仇恨言论
- 不确定时倾向判为仇恨言论（降低false negative）

请严格输出JSON格式：
{
    "label": 0 或 1,
    "confidence": 0.0-1.0,
    "probs": [0.0, 0.0],
    "reasoning": "简要推理过程"
}
""",

    "image_focused": """你是一个专业的仇恨言论检测助手，专注于**图像内容分析**。

根据以下图像描述，分析社交媒体内容是否包含仇恨言论。

请关注图像特征：
- 视觉元素：是否包含种族歧视符号、攻击性手势、侮辱性图像
- 人物特征：是否针对特定种族、性别或群体
- 图像内容：是否具有攻击性、仇恨性、歧视性

请严格输出JSON格式：
{
    "label": 0 或 1,
    "confidence": 0.0-1.0,
    "probs": [0.0, 0.0],
    "reasoning": "简要推理过程"
}
""",

    "multimodal_fusion": """你是一个专业的仇恨言论检测助手，擅长**多模态融合分析**。

综合分析以下文本和图像描述，判断社交媒体内容是否包含仇恨言论。

请关注：
- 文本内容：是否使用攻击性、贬低性词汇
- 图像内容：是否包含仇恨符号、歧视性视觉元素
- 跨模态关联：文本和图像如何相互强化或矛盾
- 整体语境：不要孤立看待单一模态

请严格输出JSON格式：
{
    "label": 0 或 1,
    "confidence": 0.0-1.0,
    "probs": [0.0, 0.0],
    "reasoning": "简要推理过程"
}
""",
}


class LLMAgent:
    """
    LLM Agent包装器
    
    将LLM API调用包装为与EvidenceHead兼容的输出格式。
    输出 (alpha, belief, uncertainty, embedding) 用于共识框架。
    """
    
    def __init__(
        self,
        client: LLMClient,
        name: str = "LLMAgent",
        system_prompt: Optional[str] = None,
        embed_dim: int = 256,
        num_classes: int = 2,
        use_image: bool = False,
        use_direct_image: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            client: LLM API客户端
            name: Agent名称
            system_prompt: 自定义系统提示词（使用默认分类prompt）
            embed_dim: 模拟embedding维度（用于共识网络）
            num_classes: 类别数
            use_image: 是否使用图像描述信息（文本描述方式）
            use_direct_image: 是否直接发送图像给多模态模型（GLM-5V-Turbo等）
            verbose: 是否打印详细信息
        """
        self.client = client
        self.name = name
        self.system_prompt = system_prompt
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_image = use_image
        self.use_direct_image = use_direct_image
        self.verbose = verbose
        
        # 用于生成稳定embedding的投影层（将LLM输出+推理文本投影到固定维度）
        self._proj = torch.nn.Linear(num_classes + 1 + 32, embed_dim)
        
        # 推理文本编码器（简单的词袋模型）
        self._reasoning_vocab = {
            'offensive': 0, 'hateful': 1, 'discrimination': 2, 'racist': 3,
            'sexist': 4, 'stereotype': 5, 'violence': 6, 'attack': 7,
            'neutral': 8, 'positive': 9, 'humor': 10, 'satire': 11,
            'funny': 12, 'meme': 13, 'image': 14, 'text': 15,
            'targeted': 16, 'slur': 17, 'insult': 18, 'derogatory': 19,
            'celebrity': 20, 'religious': 21, 'political': 22, 'cultural': 23,
            'group': 24, 'minority': 25, 'community': 26, 'identity': 27,
            'biased': 28, 'prejudice': 29, 'bigotry': 30, 'harmful': 31,
        }
        self._reasoning_dim = 32
        
        # 统计
        self.call_count = 0
        self.total_confidence = 0.0
    
    def forward(
        self,
        text: str,
        image_description: Optional[str] = None,
        image=None,
    ):
        """
        推理接口，与EvidenceHead.forward()兼容
        
        Args:
            text: 输入文本
            image_description: 图像描述（可选，文本描述方式）
            image: PIL Image对象（可选，直接图像输入方式）
        
        Returns:
            alpha: [C] Dirichlet参数
            belief: [C] 信念分布
            uncertainty: scalar 认知不确定性
            embedding: [D] 模拟的embedding（用于共识网络）
        """
        if self.use_direct_image and image is not None:
            result = self.client.classify_with_image(
                text=text,
                image=image,
                system_prompt=self.system_prompt,
            )
        else:
            result = self.client.classify(
                text=text,
                image_description=image_description if self.use_image else None,
                system_prompt=self.system_prompt,
            )
        
        self.call_count += 1
        
        if not result.get('success', False):
            # 失败时返回均匀分布
            alpha = torch.ones(self.num_classes)
            belief = torch.ones(self.num_classes) / self.num_classes
            uncertainty = torch.tensor(1.0)
            embedding = torch.zeros(self.embed_dim)
            return alpha, belief, uncertainty, embedding
        
        # 将LLM输出映射为证据参数
        probs = result['probs']
        confidence = result['confidence']
        
        # u = 1 - confidence, clamped to [0.05, 0.95]
        u = max(min(1.0 - confidence, 0.95), 0.05)
        
        S = self.num_classes / u  # total evidence
        evidence = S * torch.tensor(probs)
        alpha = evidence + 1.0
        
        S_total = alpha.sum()
        belief = alpha / S_total
        uncertainty = torch.tensor(self.num_classes / S_total)
        
        # 编码推理文本（词袋模型）
        reasoning = result.get('reasoning', '').lower()
        reasoning_vec = torch.zeros(self._reasoning_dim)
        for word, idx in self._reasoning_vocab.items():
            if word in reasoning:
                reasoning_vec[idx] = 1.0
        
        # 从LLM输出构造embedding: 拼接belief + uncertainty + 推理文本特征，投影到embed_dim
        feat = torch.cat([belief, uncertainty.unsqueeze(0), reasoning_vec])
        with torch.no_grad():
            embedding = self._proj(feat)
        
        self.total_confidence += confidence
        
        if self.verbose:
            print(f"    [{self.name}] label={result['label']}, conf={confidence:.3f}, u={u:.3f}, "
                  f"reasoning: {result.get('reasoning', '')[:60]}")
        
        return alpha, belief, uncertainty, embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_conf = self.total_confidence / max(self.call_count, 1)
        return {
            'name': self.name,
            'model': self.client.model,
            'provider': self.client.provider,
            'calls': self.call_count,
            'avg_confidence': avg_conf,
            'mock_mode': self.client.mock_mode,
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_llm_agents(
    primary_provider: str = 'deepseek',
    primary_model: Optional[str] = None,
    secondary_provider: str = 'glm',
    secondary_model: Optional[str] = None,
    tertiary_provider: str = 'gpt',
    tertiary_model: Optional[str] = None,
    embed_dim: int = 256,
    num_classes: int = 2,
    verbose: bool = True,
    **kwargs,
) -> List[LLMAgent]:
    """
    创建三个LLM Agent，每个使用不同的模型/策略
    
    Args:
        primary_provider: Agent1的模型提供者
        primary_model: Agent1的模型名称
        secondary_provider: Agent2的模型提供者
        secondary_model: Agent2的模型名称
        tertiary_provider: Agent3的模型提供者
        tertiary_model: Agent3的模型名称
        embed_dim: 模拟embedding维度
        num_classes: 类别数
        verbose: 是否打印详细信息
    
    Returns:
        [agent1, agent2, agent3] 三个LLM Agent
    """
    agents = []
    
    providers_and_prompts = [
        (primary_provider, primary_model, "Agent1(文本聚焦)", "text_focused"),
        (secondary_provider, secondary_model, "Agent2(图像专家)", "image_focused"),
        (tertiary_provider, tertiary_model, "Agent3(跨模态)", "multimodal_fusion"),
    ]
    
    for i, (provider, model, name, prompt_key) in enumerate(providers_and_prompts):
        if provider not in PROVIDER_CONFIGS:
            print(f"  [警告] 不支持的provider '{provider}'，回退到模拟模式")
            client = LLMClient(provider='deepseek', api_key='mock')
        else:
            client = LLMClient(
                provider=provider,
                model=model,
                **{k: kwargs[k] for k in ['api_key', 'temperature', 'max_retries', 'timeout']
                   if k in kwargs},
            )
        
        use_direct_image = (i == 1 and provider == 'glm')
        
        agent = LLMAgent(
            client=client,
            name=name,
            system_prompt=AGENT_PROMPTS.get(prompt_key),
            embed_dim=embed_dim,
            num_classes=num_classes,
            use_image=(i >= 2),
            use_direct_image=use_direct_image,
            verbose=verbose,
        )
        agents.append(agent)
        
        if verbose:
            mode_str = "[模拟]" if client.mock_mode else "[API]"
            print(f"    创建 {name}: {provider}/{client.model} {mode_str}")
    
    return agents


def create_single_agent(
    provider: str = 'deepseek',
    model: Optional[str] = None,
    name: str = "LLMAgent",
    system_prompt_key: str = "text_focused",
    embed_dim: int = 256,
    num_classes: int = 2,
    verbose: bool = True,
    **kwargs,
) -> LLMAgent:
    """创建单个LLM Agent"""
    client = LLMClient(
        provider=provider,
        model=model,
        **{k: kwargs[k] for k in ['api_key', 'temperature', 'max_retries', 'timeout']
           if k in kwargs},
    )
    
    agent = LLMAgent(
        client=client,
        name=name,
        system_prompt=AGENT_PROMPTS.get(system_prompt_key),
        embed_dim=embed_dim,
        num_classes=num_classes,
        verbose=verbose,
    )
    
    if verbose:
        mode_str = "[模拟]" if client.mock_mode else "[API]"
        print(f"    创建 {name}: {provider}/{client.model} {mode_str}")
    
    return agent


# =============================================================================
# 批量LLM Agent处理
# =============================================================================

class BatchLLMProcessor:
    """
    批量LLM处理器
    对一批样本运行所有Agent，输出与现有管线兼容的格式
    """
    
    def __init__(
        self,
        agents: List[LLMAgent],
        num_classes: int = 2,
        batch_size: int = 8,
        verbose: bool = True,
    ):
        self.agents = agents
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.verbose = verbose
    
    def process_batch(
        self,
        texts: List[str],
        image_descriptions: Optional[List[str]] = None,
    ):
        """
        批量处理，返回与现有评估管线兼容的输出
        
        Returns:
            all_alphas: [B, N, C]
            all_beliefs: [B, N, C]
            all_uncertainties: [B, N]
            all_embs: [B, N, D]
            all_preds: [B] 每个样本的多数投票结果
        """
        B = len(texts)
        N = len(self.agents)
        D = self.agents[0].embed_dim
        
        all_alphas = torch.zeros(B, N, self.num_classes)
        all_beliefs = torch.zeros(B, N, self.num_classes)
        all_uncertainties = torch.zeros(B, N)
        all_embs = torch.zeros(B, N, D)
        all_preds = torch.zeros(B, dtype=torch.long)
        
        for i, agent in enumerate(self.agents):
            if self.verbose:
                print(f"\n  运行 {agent.name} ({agent.client.provider}/{agent.client.model})...")
            
            for b in range(0, B, self.batch_size):
                batch_end = min(b + self.batch_size, B)
                batch_indices = list(range(b, batch_end))
                
                for idx in batch_indices:
                    text = texts[idx]
                    img_desc = image_descriptions[idx] if image_descriptions else None
                    
                    alpha, belief, uncertainty, emb = agent.forward(
                        text=text, image_description=img_desc,
                    )
                    
                    all_alphas[idx, i] = alpha
                    all_beliefs[idx, i] = belief
                    all_uncertainties[idx, i] = uncertainty
                    all_embs[idx, i] = emb
                
                if self.verbose:
                    print(f"    [{b+1}/{B}] {agent.name} done")
        
        # 多数投票
        preds_by_agent = all_beliefs.argmax(dim=-1)  # [B, N]
        for b in range(B):
            values, counts = torch.unique(preds_by_agent[b], return_counts=True)
            all_preds[b] = values[counts.argmax()]
        
        return all_alphas, all_beliefs, all_uncertainties, all_embs, all_preds
    
    def print_stats(self):
        """打印所有Agent的统计信息"""
        print(f"\n{'='*60}")
        print(f"LLM Agent 统计")
        print(f"{'='*60}")
        for agent in self.agents:
            stats = agent.get_stats()
            print(f"  {stats['name']:<25s}: {stats['calls']:4d} calls, "
                  f"avg_conf={stats['avg_confidence']:.3f}, "
                  f"{'[API]' if not stats['mock_mode'] else '[模拟]'}")
        print(f"{'='*60}")


# =============================================================================
# 快速测试
# =============================================================================

def test_llm_agents():
    """测试LLM Agent功能"""
    print("=" * 60)
    print("LLM Agent 快速测试")
    print("=" * 60)
    
    agents = create_llm_agents(verbose=True)
    
    test_texts = [
        "I love this product, it's amazing!",
        "All [group] are stupid and should be eliminated.",
        "The weather is nice today.",
        "Why do [group] always cause trouble? They don't belong here.",
    ]
    
    for text in test_texts:
        print(f"\n输入: {text}")
        for agent in agents:
            alpha, belief, uncertainty, emb = agent.forward(text)
            label = belief.argmax().item()
            conf = 1.0 - uncertainty.item()
            print(f"  {agent.name:<25s}: label={label}, conf={conf:.3f}, "
                  f"u={uncertainty.item():.3f}, alpha={alpha.numpy().round(2)}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_llm_agents()