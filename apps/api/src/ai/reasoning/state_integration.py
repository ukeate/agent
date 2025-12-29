"""推理引擎与LangGraph状态管理集成"""

import asyncio
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.models.schemas.reasoning import (
    ReasoningChain,
    ReasoningStrategy,
    ThoughtStep,
    ThoughtStepType,
    ReasoningRequest
)
from src.ai.reasoning.cot_engine import BaseCoTEngine
from src.ai.reasoning.strategies.zero_shot import ZeroShotCoTEngine
from src.ai.reasoning.strategies.few_shot import FewShotCoTEngine
from src.ai.reasoning.strategies.auto_cot import AutoCoTEngine

logger = get_logger(__name__)

class ReasoningState(TypedDict):
    """推理状态定义"""

    chain_id: str
    problem: str
    context: Optional[str]
    strategy: str
    current_step: int
    max_steps: int
    steps: List[Dict[str, Any]]
    branches: List[Dict[str, Any]]
    current_branch_id: Optional[str]
    conclusion: Optional[str]
    confidence_score: Optional[float]
    is_complete: bool
    error: Optional[str]

class ReasoningGraphBuilder:
    """推理图构建器"""

    def __init__(self):
        self.engines = {
            ReasoningStrategy.ZERO_SHOT: ZeroShotCoTEngine(),
            ReasoningStrategy.FEW_SHOT: FewShotCoTEngine(),
            ReasoningStrategy.AUTO_COT: AutoCoTEngine()
        }
        self.checkpointer = MemorySaver()

    def build_graph(self) -> StateGraph:
        """构建推理状态图"""
        graph = StateGraph(ReasoningState)
        
        # 添加节点
        graph.add_node("initialize", self.initialize_reasoning)
        graph.add_node("execute_step", self.execute_reasoning_step)
        graph.add_node("validate_step", self.validate_step)
        graph.add_node("check_branching", self.check_branching)
        graph.add_node("generate_conclusion", self.generate_conclusion)
        graph.add_node("handle_error", self.handle_error)
        
        # 添加边
        graph.add_edge("initialize", "execute_step")
        graph.add_edge("execute_step", "validate_step")
        
        # 条件边
        graph.add_conditional_edges(
            "validate_step",
            self.should_continue,
            {
                "continue": "check_branching",
                "conclude": "generate_conclusion",
                "error": "handle_error"
            }
        )
        
        graph.add_conditional_edges(
            "check_branching",
            self.should_branch,
            {
                "branch": "execute_step",
                "continue": "execute_step",
                "conclude": "generate_conclusion"
            }
        )
        
        graph.add_edge("generate_conclusion", END)
        graph.add_edge("handle_error", END)
        
        # 设置入口点
        graph.set_entry_point("initialize")
        
        return graph.compile(checkpointer=self.checkpointer)

    async def initialize_reasoning(self, state: ReasoningState) -> ReasoningState:
        """初始化推理状态"""
        logger.info(f"初始化推理: {state['problem']}")
        
        state["chain_id"] = str(uuid4())
        state["current_step"] = 0
        state["steps"] = []
        state["branches"] = []
        state["is_complete"] = False
        state["error"] = None
        
        return state

    async def execute_reasoning_step(self, state: ReasoningState) -> ReasoningState:
        """执行推理步骤"""
        state["current_step"] += 1
        
        # 获取对应的推理引擎
        strategy = ReasoningStrategy(state["strategy"])
        engine = self.engines.get(strategy)
        
        if not engine:
            state["error"] = f"不支持的推理策略: {strategy}"
            return state
        
        try:
            # 创建临时推理链对象
            chain = ReasoningChain(
                id=state["chain_id"],
                strategy=strategy,
                problem=state["problem"],
                context=state.get("context")
            )
            
            # 恢复之前的步骤
            for step_data in state["steps"]:
                step = ThoughtStep(**step_data)
                chain.steps.append(step)
            
            # 执行新步骤
            step = await engine.execute_step(
                chain,
                state["current_step"],
                state["problem"],
                state.get("context")
            )
            
            # 保存步骤到状态
            state["steps"].append({
                "id": str(step.id),
                "step_number": step.step_number,
                "step_type": step.step_type.value,
                "content": step.content,
                "reasoning": step.reasoning,
                "confidence": step.confidence,
                "duration_ms": step.duration_ms
            })
            
            logger.info(
                f"步骤 {state['current_step']}: "
                f"{step.step_type.value} - 置信度: {step.confidence}"
            )
            
        except Exception as e:
            logger.error(f"执行推理步骤失败: {e}")
            state["error"] = str(e)
        
        return state

    async def validate_step(self, state: ReasoningState) -> ReasoningState:
        """验证推理步骤"""
        if not state["steps"]:
            return state
        
        last_step = state["steps"][-1]
        
        # 简单验证逻辑
        if last_step["confidence"] < 0.3:
            logger.warning(f"步骤 {last_step['step_number']} 置信度过低")
        
        # 检查是否达到结论
        if last_step["step_type"] == ThoughtStepType.CONCLUSION.value:
            state["conclusion"] = last_step["content"]
            state["is_complete"] = True
        
        return state

    async def check_branching(self, state: ReasoningState) -> ReasoningState:
        """检查是否需要分支"""
        if not state["steps"]:
            return state
        
        last_step = state["steps"][-1]
        
        # 低置信度时创建分支
        if last_step["confidence"] < 0.6:
            branch = {
                "id": str(uuid4()),
                "parent_step_id": last_step["id"],
                "reason": f"低置信度({last_step['confidence']:.2f})",
                "is_active": True
            }
            state["branches"].append(branch)
            state["current_branch_id"] = branch["id"]
            logger.info(f"创建分支: {branch['id']}")
        
        return state

    async def generate_conclusion(self, state: ReasoningState) -> ReasoningState:
        """生成最终结论"""
        if state["conclusion"]:
            return state
        
        # 基于所有步骤生成结论
        if state["steps"]:
            conclusions = [
                step["content"] 
                for step in state["steps"] 
                if step["step_type"] in [
                    ThoughtStepType.CONCLUSION.value,
                    ThoughtStepType.VALIDATION.value
                ]
            ]
            
            if conclusions:
                state["conclusion"] = conclusions[-1]
            else:
                # 使用最后一个步骤作为结论
                state["conclusion"] = state["steps"][-1]["content"]
            
            # 计算总体置信度
            confidences = [step["confidence"] for step in state["steps"]]
            state["confidence_score"] = sum(confidences) / len(confidences)
        else:
            state["conclusion"] = "无法得出结论"
            state["confidence_score"] = 0.0
        
        state["is_complete"] = True
        logger.info(f"生成结论: {state['conclusion'][:100]}...")
        
        return state

    async def handle_error(self, state: ReasoningState) -> ReasoningState:
        """处理错误"""
        logger.error(f"推理错误: {state.get('error', '未知错误')}")
        state["is_complete"] = True
        state["conclusion"] = f"推理失败: {state.get('error', '未知错误')}"
        return state

    def should_continue(self, state: ReasoningState) -> str:
        """判断是否继续推理"""
        if state.get("error"):
            return "error"
        
        if state.get("is_complete"):
            return "conclude"
        
        if state["current_step"] >= state["max_steps"]:
            return "conclude"
        
        return "continue"

    def should_branch(self, state: ReasoningState) -> str:
        """判断是否需要分支"""
        if state.get("current_branch_id"):
            return "branch"
        
        if state["current_step"] >= state["max_steps"]:
            return "conclude"
        
        return "continue"

class ReasoningStateMachine:
    """推理状态机"""

    def __init__(self):
        self.builder = ReasoningGraphBuilder()
        self.graph = self.builder.build_graph()

    async def execute(self, request: ReasoningRequest) -> ReasoningChain:
        """执行推理请求"""
        # 初始化状态
        initial_state = ReasoningState(
            chain_id="",
            problem=request.problem,
            context=request.context,
            strategy=request.strategy.value,
            current_step=0,
            max_steps=request.max_steps,
            steps=[],
            branches=[],
            current_branch_id=None,
            conclusion=None,
            confidence_score=None,
            is_complete=False,
            error=None
        )
        
        # 配置
        config = {
            "configurable": {
                "thread_id": str(uuid4())
            }
        }
        
        # 执行图
        final_state = await self.graph.ainvoke(initial_state, config)
        
        # 转换为ReasoningChain
        chain = self._state_to_chain(final_state)
        
        return chain

    def _state_to_chain(self, state: ReasoningState) -> ReasoningChain:
        """将状态转换为推理链"""
        chain = ReasoningChain(
            id=state["chain_id"],
            strategy=ReasoningStrategy(state["strategy"]),
            problem=state["problem"],
            context=state.get("context"),
            conclusion=state.get("conclusion"),
            confidence_score=state.get("confidence_score")
        )
        
        # 恢复步骤
        for step_data in state["steps"]:
            step = ThoughtStep(
                id=step_data["id"],
                step_number=step_data["step_number"],
                step_type=ThoughtStepType(step_data["step_type"]),
                content=step_data["content"],
                reasoning=step_data["reasoning"],
                confidence=step_data["confidence"],
                duration_ms=step_data.get("duration_ms")
            )
            chain.steps.append(step)
        
        # 如果有结论，标记为完成
        if state.get("conclusion"):
            chain.complete(state["conclusion"])
        
        return chain
from src.core.logging import get_logger
