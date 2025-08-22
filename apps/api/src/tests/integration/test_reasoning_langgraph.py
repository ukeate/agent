"""推理引擎与LangGraph集成测试"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.ai.reasoning.state_integration import (
    ReasoningState,
    ReasoningGraphBuilder,
    ReasoningStateMachine
)
from src.ai.langgraph.state_graph import LangGraphWorkflowBuilder
from models.schemas.reasoning import (
    ReasoningStrategy,
    ReasoningRequest,
    ThoughtStepType
)


class TestReasoningGraphBuilder:
    """测试推理图构建器"""

    @pytest.fixture
    def builder(self):
        with patch('ai.reasoning.state_integration.get_openai_client'):
            return ReasoningGraphBuilder()

    @pytest.mark.asyncio
    async def test_initialize_reasoning(self, builder):
        """测试初始化推理状态"""
        state = ReasoningState(
            chain_id="",
            problem="测试问题",
            context=None,
            strategy="zero_shot",
            current_step=0,
            max_steps=5,
            steps=[],
            branches=[],
            current_branch_id=None,
            conclusion=None,
            confidence_score=None,
            is_complete=False,
            error=None
        )
        
        result = await builder.initialize_reasoning(state)
        
        assert result["chain_id"] != ""
        assert result["current_step"] == 0
        assert result["steps"] == []
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_execute_reasoning_step(self, builder):
        """测试执行推理步骤"""
        state = ReasoningState(
            chain_id=str(uuid4()),
            problem="计算 2+2",
            context=None,
            strategy="zero_shot",
            current_step=0,
            max_steps=5,
            steps=[],
            branches=[],
            current_branch_id=None,
            conclusion=None,
            confidence_score=None,
            is_complete=False,
            error=None
        )
        
        # Mock OpenAI响应
        with patch.object(builder.engines[ReasoningStrategy.ZERO_SHOT], 'execute_step') as mock_execute:
            mock_step = Mock()
            mock_step.id = uuid4()
            mock_step.step_number = 1
            mock_step.step_type = ThoughtStepType.ANALYSIS
            mock_step.content = "这是加法"
            mock_step.reasoning = "基本算术"
            mock_step.confidence = 0.9
            mock_step.duration_ms = 100
            
            mock_execute.return_value = mock_step
            
            result = await builder.execute_reasoning_step(state)
            
            assert result["current_step"] == 1
            assert len(result["steps"]) == 1
            assert result["steps"][0]["content"] == "这是加法"

    @pytest.mark.asyncio
    async def test_validate_step(self, builder):
        """测试验证推理步骤"""
        state = ReasoningState(
            chain_id=str(uuid4()),
            problem="问题",
            context=None,
            strategy="zero_shot",
            current_step=1,
            max_steps=5,
            steps=[{
                "id": str(uuid4()),
                "step_number": 1,
                "step_type": ThoughtStepType.CONCLUSION.value,
                "content": "答案是4",
                "reasoning": "2+2=4",
                "confidence": 1.0,
                "duration_ms": 100
            }],
            branches=[],
            current_branch_id=None,
            conclusion=None,
            confidence_score=None,
            is_complete=False,
            error=None
        )
        
        result = await builder.validate_step(state)
        
        assert result["conclusion"] == "答案是4"
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_check_branching(self, builder):
        """测试分支检查"""
        state = ReasoningState(
            chain_id=str(uuid4()),
            problem="复杂问题",
            context=None,
            strategy="zero_shot",
            current_step=1,
            max_steps=5,
            steps=[{
                "id": str(uuid4()),
                "step_number": 1,
                "step_type": ThoughtStepType.ANALYSIS.value,
                "content": "不确定的分析",
                "reasoning": "需要进一步探索",
                "confidence": 0.4,  # 低置信度
                "duration_ms": 100
            }],
            branches=[],
            current_branch_id=None,
            conclusion=None,
            confidence_score=None,
            is_complete=False,
            error=None
        )
        
        result = await builder.check_branching(state)
        
        assert len(result["branches"]) == 1
        assert result["current_branch_id"] is not None

    @pytest.mark.asyncio
    async def test_generate_conclusion(self, builder):
        """测试生成结论"""
        state = ReasoningState(
            chain_id=str(uuid4()),
            problem="问题",
            context=None,
            strategy="zero_shot",
            current_step=3,
            max_steps=5,
            steps=[
                {
                    "id": str(uuid4()),
                    "step_number": 1,
                    "step_type": ThoughtStepType.OBSERVATION.value,
                    "content": "观察1",
                    "reasoning": "原因1",
                    "confidence": 0.8,
                    "duration_ms": 100
                },
                {
                    "id": str(uuid4()),
                    "step_number": 2,
                    "step_type": ThoughtStepType.ANALYSIS.value,
                    "content": "分析",
                    "reasoning": "原因2",
                    "confidence": 0.9,
                    "duration_ms": 100
                },
                {
                    "id": str(uuid4()),
                    "step_number": 3,
                    "step_type": ThoughtStepType.VALIDATION.value,
                    "content": "验证结果",
                    "reasoning": "原因3",
                    "confidence": 0.95,
                    "duration_ms": 100
                }
            ],
            branches=[],
            current_branch_id=None,
            conclusion=None,
            confidence_score=None,
            is_complete=False,
            error=None
        )
        
        result = await builder.generate_conclusion(state)
        
        assert result["conclusion"] == "验证结果"
        assert result["confidence_score"] == pytest.approx(0.883, 0.01)
        assert result["is_complete"] is True

    def test_should_continue(self, builder):
        """测试继续判断逻辑"""
        # 有错误时
        state = {"error": "错误"}
        assert builder.should_continue(state) == "error"
        
        # 已完成时
        state = {"is_complete": True}
        assert builder.should_continue(state) == "conclude"
        
        # 达到最大步数时
        state = {"current_step": 10, "max_steps": 10}
        assert builder.should_continue(state) == "conclude"
        
        # 继续执行
        state = {"current_step": 3, "max_steps": 10}
        assert builder.should_continue(state) == "continue"


class TestReasoningStateMachine:
    """测试推理状态机"""

    @pytest.fixture
    def machine(self):
        with patch('ai.reasoning.state_integration.get_openai_client'):
            return ReasoningStateMachine()

    @pytest.mark.asyncio
    async def test_execute_reasoning_request(self, machine):
        """测试执行推理请求"""
        request = ReasoningRequest(
            problem="计算 2+2",
            strategy=ReasoningStrategy.ZERO_SHOT,
            max_steps=5
        )
        
        # Mock图执行
        mock_final_state = ReasoningState(
            chain_id=str(uuid4()),
            problem="计算 2+2",
            context=None,
            strategy="zero_shot",
            current_step=1,
            max_steps=5,
            steps=[{
                "id": str(uuid4()),
                "step_number": 1,
                "step_type": ThoughtStepType.CONCLUSION.value,
                "content": "4",
                "reasoning": "2+2=4",
                "confidence": 1.0,
                "duration_ms": 100
            }],
            branches=[],
            current_branch_id=None,
            conclusion="4",
            confidence_score=1.0,
            is_complete=True,
            error=None
        )
        
        with patch.object(machine.graph, 'ainvoke', return_value=mock_final_state):
            chain = await machine.execute(request)
            
            assert chain.problem == "计算 2+2"
            assert chain.conclusion == "4"
            assert len(chain.steps) == 1
            assert chain.confidence_score == 1.0


class TestLangGraphIntegration:
    """测试LangGraph集成"""

    @pytest.mark.asyncio
    async def test_add_reasoning_node(self):
        """测试添加推理节点到LangGraph"""
        builder = LangGraphWorkflowBuilder()
        
        async def reasoning_handler(state):
            state["metadata"]["reasoning_executed"] = True
            return state
        
        builder.add_reasoning_node("reasoning", reasoning_handler)
        
        assert "reasoning" in builder.nodes
        assert builder.nodes["reasoning"].node_type == "reasoning"

    @pytest.mark.asyncio
    async def test_reasoning_workflow_integration(self):
        """测试推理工作流集成"""
        from ai.langgraph.state import create_initial_state
        
        builder = LangGraphWorkflowBuilder()
        
        # 添加推理节点
        async def init_reasoning(state):
            state["metadata"]["reasoning_started"] = True
            return state
        
        async def execute_reasoning(state):
            state["metadata"]["reasoning_step"] = 1
            return state
        
        async def conclude_reasoning(state):
            state["metadata"]["reasoning_completed"] = True
            return state
        
        builder.add_reasoning_node("init", init_reasoning)
        builder.add_reasoning_node("execute", execute_reasoning)
        builder.add_reasoning_node("conclude", conclude_reasoning)
        
        # 构建图
        graph = builder.build()
        graph.add_edge("init", "execute")
        graph.add_edge("execute", "conclude")
        graph.set_entry_point("init")
        
        compiled = graph.compile()
        
        # 执行
        initial_state = create_initial_state(
            messages=[],
            workflow_id="test_reasoning"
        )
        
        result = await compiled.ainvoke(initial_state)
        
        assert result["metadata"]["reasoning_started"] is True
        assert result["metadata"]["reasoning_step"] == 1
        assert result["metadata"]["reasoning_completed"] is True