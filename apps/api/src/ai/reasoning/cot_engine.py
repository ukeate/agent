"""链式思考(CoT)推理引擎"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from models.schemas.reasoning import (
    ReasoningChain,
    ReasoningStrategy,
    ThoughtStep,
    ThoughtStepType,
    ReasoningBranch,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningStreamChunk
)
from src.ai.openai_client import get_openai_client
from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseCoTEngine(ABC):
    """基础CoT推理引擎抽象类"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = get_openai_client()
        self.max_retries = 3
        self.retry_delay = 1.0

    @abstractmethod
    async def generate_prompt(self, problem: str, context: Optional[str] = None, **kwargs) -> str:
        """生成推理提示词"""
        pass

    @abstractmethod
    async def parse_response(self, response: str) -> Tuple[ThoughtStepType, str, str, float]:
        """解析模型响应，返回(步骤类型, 内容, 推理, 置信度)"""
        pass

    async def execute_step(
        self,
        chain: ReasoningChain,
        step_number: int,
        problem: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ThoughtStep:
        """执行单个推理步骤"""
        start_time = time.time()
        
        # 构建历史上下文
        history = self._build_history(chain)
        
        # 生成提示词
        prompt = await self.generate_prompt(problem, context, history=history, **kwargs)
        
        # 调用模型
        response = await self._call_model(prompt)
        
        # 解析响应
        step_type, content, reasoning, confidence = await self.parse_response(response)
        
        # 创建步骤
        duration_ms = int((time.time() - start_time) * 1000)
        step = ThoughtStep(
            step_number=step_number,
            step_type=step_type,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            duration_ms=duration_ms
        )
        
        return step

    async def execute_chain(
        self,
        request: ReasoningRequest,
        stream_callback=None
    ) -> ReasoningChain:
        """执行完整推理链"""
        # 创建推理链
        chain = ReasoningChain(
            strategy=request.strategy,
            problem=request.problem,
            context=request.context
        )
        
        try:
            # 逐步执行推理
            for step_num in range(1, request.max_steps + 1):
                step = await self.execute_step(
                    chain,
                    step_num,
                    request.problem,
                    request.context,
                    examples=request.examples
                )
                
                chain.add_step(step)
                
                # 流式回调
                if stream_callback:
                    chunk = ReasoningStreamChunk(
                        chain_id=chain.id,
                        step_number=step_num,
                        step_type=step.step_type,
                        content=step.content,
                        reasoning=step.reasoning,
                        confidence=step.confidence,
                        is_final=(step.step_type == ThoughtStepType.CONCLUSION)
                    )
                    await stream_callback(chunk)
                
                # 检查是否到达结论
                if step.step_type == ThoughtStepType.CONCLUSION:
                    chain.complete(step.content)
                    break
                
                # 检查是否需要分支
                if request.enable_branching and await self._should_branch(step):
                    await self._create_branch(chain, step)
            
            # 如果没有明确结论，生成一个
            if not chain.conclusion:
                conclusion = await self._generate_conclusion(chain)
                chain.complete(conclusion)
            
            logger.info(f"推理链完成: {chain.id}, 步骤数: {len(chain.steps)}")
            
        except Exception as e:
            logger.error(f"推理链执行失败: {e}")
            raise
        
        return chain

    async def _call_model(self, prompt: str) -> str:
        """调用AI模型"""
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个擅长链式思考的推理助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise e

    def _build_history(self, chain: ReasoningChain) -> str:
        """构建历史上下文"""
        if not chain.steps:
            return ""
        
        history_lines = ["之前的推理步骤:"]
        for step in chain.steps[-5:]:  # 只取最近5个步骤
            history_lines.append(
                f"\n步骤{step.step_number} [{step.step_type}]: {step.content}\n"
                f"推理: {step.reasoning} (置信度: {step.confidence:.2f})"
            )
        
        return "\n".join(history_lines)

    async def _should_branch(self, step: ThoughtStep) -> bool:
        """判断是否需要创建分支"""
        # 简单策略: 低置信度时分支
        return step.confidence < 0.6 and step.step_type != ThoughtStepType.CONCLUSION

    async def _create_branch(self, chain: ReasoningChain, parent_step: ThoughtStep) -> None:
        """创建推理分支"""
        branch = chain.create_branch(
            parent_step_id=parent_step.id,
            reason=f"低置信度({parent_step.confidence:.2f})，探索替代路径"
        )
        logger.info(f"创建推理分支: {branch.id}")

    async def _generate_conclusion(self, chain: ReasoningChain) -> str:
        """生成最终结论"""
        if not chain.steps:
            return "无法得出结论"
        
        # 基于所有步骤生成结论
        summary = "\n".join([f"- {s.content}" for s in chain.steps])
        prompt = f"""基于以下推理步骤，总结最终结论:
{summary}

结论:"""
        
        response = await self._call_model(prompt)
        return response.strip()


class ReasoningChainBuilder:
    """推理链构建器"""

    def __init__(self):
        self.chain = None
        self.current_step = 0

    def create(self, strategy: ReasoningStrategy, problem: str, context: Optional[str] = None) -> 'ReasoningChainBuilder':
        """创建新推理链"""
        self.chain = ReasoningChain(
            strategy=strategy,
            problem=problem,
            context=context
        )
        self.current_step = 0
        return self

    def add_step(
        self,
        step_type: ThoughtStepType,
        content: str,
        reasoning: str,
        confidence: float,
        duration_ms: Optional[int] = None
    ) -> 'ReasoningChainBuilder':
        """添加推理步骤"""
        if not self.chain:
            raise ValueError("请先调用create()创建推理链")
        
        self.current_step += 1
        step = ThoughtStep(
            step_number=self.current_step,
            step_type=step_type,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            duration_ms=duration_ms
        )
        self.chain.add_step(step)
        return self

    def add_branch(self, parent_step_id: Optional[str], reason: str) -> 'ReasoningChainBuilder':
        """添加分支"""
        if not self.chain:
            raise ValueError("请先调用create()创建推理链")
        
        self.chain.create_branch(parent_step_id, reason)
        return self

    def complete(self, conclusion: str) -> ReasoningChain:
        """完成推理链"""
        if not self.chain:
            raise ValueError("请先调用create()创建推理链")
        
        self.chain.complete(conclusion)
        return self.chain

    def build(self) -> ReasoningChain:
        """构建推理链"""
        if not self.chain:
            raise ValueError("请先调用create()创建推理链")
        
        return self.chain


class StepExecutor:
    """推理步骤执行器"""

    def __init__(self, engine: BaseCoTEngine):
        self.engine = engine
        self.validators = []

    def add_validator(self, validator) -> None:
        """添加验证器"""
        self.validators.append(validator)

    async def execute(
        self,
        chain: ReasoningChain,
        step_number: int,
        problem: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ThoughtStep:
        """执行并验证步骤"""
        # 执行步骤
        step = await self.engine.execute_step(
            chain, step_number, problem, context, **kwargs
        )
        
        # 验证步骤
        for validator in self.validators:
            validation = await validator.validate(step, chain)
            if not validation.is_valid:
                logger.warning(
                    f"步骤{step_number}验证失败: {validation.issues}"
                )
                # 可以选择重试或调整
        
        return step


def generate_cache_key(problem: str, strategy: ReasoningStrategy, context: Optional[str] = None) -> str:
    """生成缓存键"""
    data = {
        "problem": problem,
        "strategy": strategy.value,
        "context": context or ""
    }
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode()).hexdigest()