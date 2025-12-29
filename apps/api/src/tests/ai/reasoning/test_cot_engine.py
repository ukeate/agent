"""链式思考引擎单元测试"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from src.ai.reasoning.cot_engine import (
    BaseCoTEngine,
    ReasoningChainBuilder,
    StepExecutor,
    generate_cache_key
)
from src.ai.reasoning.strategies.zero_shot import ZeroShotCoTEngine
from src.ai.reasoning.strategies.few_shot import FewShotCoTEngine
from src.ai.reasoning.strategies.auto_cot import AutoCoTEngine
from models.schemas.reasoning import (
    ReasoningStrategy,
    ReasoningRequest,
    ReasoningChain,
    ThoughtStepType,
    ThoughtStep

)

class TestZeroShotCoTEngine:
    """测试Zero-shot CoT引擎"""

    @pytest.fixture
    def engine(self):
        with patch('ai.reasoning.cot_engine.get_openai_client'):
            return ZeroShotCoTEngine()

    @pytest.mark.asyncio
    async def test_generate_prompt(self, engine):
        """测试生成Zero-shot提示词"""
        prompt = await engine.generate_prompt(
            problem="计算 2+2",
            context="基本算术"
        )
        
        assert "让我们一步一步地思考" in prompt
        assert "计算 2+2" in prompt
        assert "基本算术" in prompt
        assert "步骤类型" in prompt

    @pytest.mark.asyncio
    async def test_parse_response(self, engine):
        """测试解析响应"""
        response = """
        步骤类型: OBSERVATION
        内容: 这是一个加法问题
        推理: 需要计算两个数字的和
        置信度: 0.9
        """
        
        step_type, content, reasoning, confidence = await engine.parse_response(response)
        
        assert step_type == ThoughtStepType.OBSERVATION
        assert "加法问题" in content
        assert "计算" in reasoning
        assert confidence == 0.9

    @pytest.mark.asyncio
    async def test_parse_response_with_invalid_format(self, engine):
        """测试解析格式错误的响应"""
        response = "这是一个没有格式的响应"
        
        step_type, content, reasoning, confidence = await engine.parse_response(response)
        
        # 应该返回默认值
        assert step_type == ThoughtStepType.ANALYSIS
        assert len(content) > 0
        assert len(reasoning) > 0
        assert 0 <= confidence <= 1

class TestFewShotCoTEngine:
    """测试Few-shot CoT引擎"""

    @pytest.fixture
    def engine(self):
        with patch('ai.reasoning.cot_engine.get_openai_client'):
            return FewShotCoTEngine()

    @pytest.mark.asyncio
    async def test_generate_prompt_with_examples(self, engine):
        """测试使用示例生成提示词"""
        examples = [
            {
                "problem": "1+1",
                "answer": "2",
                "steps": [
                    {"number": 1, "content": "识别加法", "reasoning": "基本运算"}
                ]
            }
        ]
        
        prompt = await engine.generate_prompt(
            problem="2+2",
            examples=examples
        )
        
        assert "示例" in prompt
        assert "1+1" in prompt
        assert "2+2" in prompt

    @pytest.mark.asyncio
    async def test_generate_prompt_without_examples(self, engine):
        """测试没有示例时使用默认示例"""
        prompt = await engine.generate_prompt(problem="解决问题")
        
        assert "示例" in prompt
        assert "概率" in prompt  # 默认示例中包含概率问题

class TestAutoCoTEngine:
    """测试Auto-CoT引擎"""

    @pytest.fixture
    def engine(self):
        with patch('ai.reasoning.cot_engine.get_openai_client'):
            return AutoCoTEngine()

    def test_identify_problem_type(self, engine):
        """测试问题类型识别"""
        assert engine._identify_problem_type("计算 2+2") == "数学"
        assert engine._identify_problem_type("分析逻辑关系") == "逻辑"
        assert engine._identify_problem_type("实现一个算法") == "编程"
        assert engine._identify_problem_type("实验观察") == "科学"
        assert engine._identify_problem_type("一般问题") == "通用"

    def test_generate_strategy(self, engine):
        """测试策略生成"""
        math_strategy = engine._generate_strategy("数学")
        assert len(math_strategy) > 0
        assert "计算" in ''.join(math_strategy)
        
        logic_strategy = engine._generate_strategy("逻辑")
        assert len(logic_strategy) > 0
        assert "推理" in ''.join(logic_strategy)

    @pytest.mark.asyncio
    async def test_generate_prompt_auto(self, engine):
        """测试Auto-CoT提示词生成"""
        prompt = await engine.generate_prompt(
            problem="计算两个数的和"
        )
        
        assert "数学类型" in prompt
        assert "策略" in prompt
        assert "计算两个数的和" in prompt

class TestReasoningChainBuilder:
    """测试推理链构建器"""

    def test_build_chain(self):
        """测试构建推理链"""
        builder = ReasoningChainBuilder()
        
        chain = builder.create(
            strategy=ReasoningStrategy.ZERO_SHOT,
            problem="测试问题"
        ).add_step(
            step_type=ThoughtStepType.OBSERVATION,
            content="观察内容",
            reasoning="观察原因",
            confidence=0.8
        ).add_step(
            step_type=ThoughtStepType.CONCLUSION,
            content="结论",
            reasoning="总结",
            confidence=0.9
        ).complete("最终答案").build()
        
        assert chain.problem == "测试问题"
        assert len(chain.steps) == 2
        assert chain.conclusion == "最终答案"
        assert chain.completed_at is not None

    def test_build_chain_with_branch(self):
        """测试构建带分支的推理链"""
        builder = ReasoningChainBuilder()
        
        chain = builder.create(
            strategy=ReasoningStrategy.ZERO_SHOT,
            problem="复杂问题"
        ).add_branch(
            parent_step_id=None,
            reason="探索替代方案"
        ).build()
        
        assert len(chain.branches) == 1
        assert chain.branches[0].branch_reason == "探索替代方案"

class TestStepExecutor:
    """测试步骤执行器"""

    @pytest.fixture
    def executor(self):
        mock_engine = Mock(spec=BaseCoTEngine)
        mock_engine.execute_step = AsyncMock(return_value=ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.ANALYSIS,
            content="分析",
            reasoning="推理",
            confidence=0.8
        ))
        return StepExecutor(mock_engine)

    @pytest.mark.asyncio
    async def test_execute_step(self, executor):
        """测试执行步骤"""
        chain = ReasoningChain(
            strategy=ReasoningStrategy.ZERO_SHOT,
            problem="问题"
        )
        
        step = await executor.execute(
            chain=chain,
            step_number=1,
            problem="问题"
        )
        
        assert step.step_number == 1
        assert step.step_type == ThoughtStepType.ANALYSIS
        assert step.confidence == 0.8

class TestUtilityFunctions:
    """测试工具函数"""

    def test_generate_cache_key(self):
        """测试缓存键生成"""
        key1 = generate_cache_key(
            problem="问题",
            strategy=ReasoningStrategy.ZERO_SHOT,
            context="上下文"
        )
        
        key2 = generate_cache_key(
            problem="问题",
            strategy=ReasoningStrategy.ZERO_SHOT,
            context="上下文"
        )
        
        key3 = generate_cache_key(
            problem="不同问题",
            strategy=ReasoningStrategy.ZERO_SHOT,
            context="上下文"
        )
        
        assert key1 == key2  # 相同输入应该生成相同的键
        assert key1 != key3  # 不同输入应该生成不同的键
        assert len(key1) == 64  # SHA256的十六进制长度

class TestBaseCoTEngineIntegration:
    """测试BaseCoTEngine集成"""

    @pytest.mark.asyncio
    async def test_execute_chain_complete_flow(self):
        """测试完整的推理链执行流程"""
        with patch('ai.reasoning.cot_engine.get_openai_client') as mock_client:
            # 模拟 OpenAI 客户端
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="""
                步骤类型: CONCLUSION
                内容: 答案是4
                推理: 2+2=4
                置信度: 1.0
            """))]
            
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            
            engine = ZeroShotCoTEngine()
            request = ReasoningRequest(
                problem="计算 2+2",
                strategy=ReasoningStrategy.ZERO_SHOT,
                max_steps=5
            )
            
            chain = await engine.execute_chain(request)
            
            assert chain.problem == "计算 2+2"
            assert len(chain.steps) > 0
            assert chain.conclusion is not None
