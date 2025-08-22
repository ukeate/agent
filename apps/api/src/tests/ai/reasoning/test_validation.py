"""推理验证模块单元测试"""

import pytest
from uuid import uuid4

from src.ai.reasoning.validation import (
    ConsistencyValidator,
    ConfidenceValidator,
    SelfCheckValidator,
    CompositeValidator,
    calculate_chain_quality_score
)
from src.ai.reasoning.recovery import (
    FailureDetector,
    BacktrackMechanism,
    AlternativePathGenerator,
    RecoveryManager,
    RecoveryStrategy
)
from models.schemas.reasoning import (
    ReasoningChain,
    ThoughtStep,
    ThoughtStepType,
    ReasoningStrategy as RS
)


class TestConsistencyValidator:
    """测试一致性验证器"""

    @pytest.fixture
    def validator(self):
        return ConsistencyValidator()

    @pytest.fixture
    def chain(self):
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试问题"
        )
        # 添加一些步骤
        chain.add_step(ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.OBSERVATION,
            content="这是一个数学问题",
            reasoning="观察问题类型",
            confidence=0.9
        ))
        chain.add_step(ThoughtStep(
            step_number=2,
            step_type=ThoughtStepType.ANALYSIS,
            content="需要计算",
            reasoning="分析解决方法",
            confidence=0.8
        ))
        return chain

    @pytest.mark.asyncio
    async def test_validate_consistent_step(self, validator, chain):
        """测试验证一致的步骤"""
        step = ThoughtStep(
            step_number=3,
            step_type=ThoughtStepType.VALIDATION,
            content="验证计算结果",
            reasoning="基于分析进行验证",
            confidence=0.85
        )
        
        validation = await validator.validate(step, chain)
        
        assert validation.is_valid
        assert validation.consistency_score > 0.5
        assert len(validation.issues) == 0

    @pytest.mark.asyncio
    async def test_validate_contradictory_step(self, validator, chain):
        """测试验证矛盾的步骤"""
        step = ThoughtStep(
            step_number=3,
            step_type=ThoughtStepType.ANALYSIS,
            content="这不是数学问题",  # 与第一步矛盾
            reasoning="重新分析",
            confidence=0.7
        )
        
        validation = await validator.validate(step, chain)
        
        # 可能检测到矛盾
        assert validation.consistency_score < 1.0

    @pytest.mark.asyncio
    async def test_validate_conclusion_after_conclusion(self, validator, chain):
        """测试结论后再出现观察"""
        # 添加结论
        chain.add_step(ThoughtStep(
            step_number=3,
            step_type=ThoughtStepType.CONCLUSION,
            content="答案是42",
            reasoning="最终结果",
            confidence=0.95
        ))
        
        # 结论后再观察
        step = ThoughtStep(
            step_number=4,
            step_type=ThoughtStepType.OBSERVATION,
            content="新的观察",
            reasoning="额外观察",
            confidence=0.8
        )
        
        validation = await validator.validate(step, chain)
        
        # 应该检测到不合理的步骤序列
        assert "步骤类型序列不合理" in validation.issues


class TestConfidenceValidator:
    """测试置信度验证器"""

    @pytest.fixture
    def validator(self):
        return ConfidenceValidator(min_confidence=0.3, warning_threshold=0.5)

    @pytest.fixture
    def chain(self):
        return ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试问题"
        )

    @pytest.mark.asyncio
    async def test_validate_high_confidence(self, validator, chain):
        """测试高置信度步骤"""
        step = ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.ANALYSIS,
            content="分析",
            reasoning="推理",
            confidence=0.9
        )
        
        validation = await validator.validate(step, chain)
        
        assert validation.is_valid
        assert len(validation.issues) == 0

    @pytest.mark.asyncio
    async def test_validate_low_confidence(self, validator, chain):
        """测试低置信度步骤"""
        step = ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.ANALYSIS,
            content="不确定的分析",
            reasoning="缺乏信息",
            confidence=0.2
        )
        
        validation = await validator.validate(step, chain)
        
        assert not validation.is_valid
        assert "置信度过低" in ' '.join(validation.issues)

    @pytest.mark.asyncio
    async def test_validate_declining_confidence_trend(self, validator, chain):
        """测试置信度下降趋势"""
        # 添加置信度递减的步骤
        for i in range(3):
            chain.add_step(ThoughtStep(
                step_number=i+1,
                step_type=ThoughtStepType.ANALYSIS,
                content=f"分析{i+1}",
                reasoning="推理",
                confidence=0.9 - i*0.2
            ))
        
        # 添加更低置信度的步骤
        step = ThoughtStep(
            step_number=4,
            step_type=ThoughtStepType.ANALYSIS,
            content="分析4",
            reasoning="推理",
            confidence=0.35
        )
        
        validation = await validator.validate(step, chain)
        
        assert "置信度持续下降" in ' '.join(validation.issues)


class TestSelfCheckValidator:
    """测试自我检查验证器"""

    @pytest.fixture
    def validator(self):
        return SelfCheckValidator()

    @pytest.fixture
    def chain(self):
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试问题"
        )
        chain.add_step(ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.OBSERVATION,
            content="观察到现象A",
            reasoning="基础观察",
            confidence=0.8
        ))
        return chain

    @pytest.mark.asyncio
    async def test_validate_matching_reasoning(self, validator, chain):
        """测试推理与内容匹配"""
        step = ThoughtStep(
            step_number=2,
            step_type=ThoughtStepType.ANALYSIS,
            content="分析现象A的原因",
            reasoning="现象A可能由因素X引起",
            confidence=0.75
        )
        
        validation = await validator.validate(step, chain)
        
        assert validation.is_valid
        assert len(validation.issues) == 0

    @pytest.mark.asyncio
    async def test_validate_circular_reasoning(self, validator, chain):
        """测试循环推理"""
        # 添加相似内容
        step = ThoughtStep(
            step_number=2,
            step_type=ThoughtStepType.ANALYSIS,
            content="观察到现象A",  # 重复第一步
            reasoning="重新观察",
            confidence=0.8
        )
        
        validation = await validator.validate(step, chain)
        
        assert "循环推理" in ' '.join(validation.issues)

    @pytest.mark.asyncio
    async def test_validate_baseless_hypothesis(self, validator):
        """测试无依据的假设"""
        # 空链，没有观察或分析
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试问题"
        )
        
        step = ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.HYPOTHESIS,
            content="假设结果是X",
            reasoning="直觉",
            confidence=0.6
        )
        
        validation = await validator.validate(step, chain)
        
        assert "假设缺乏依据" in ' '.join(validation.issues)


class TestCompositeValidator:
    """测试组合验证器"""

    @pytest.fixture
    def validator(self):
        return CompositeValidator()

    @pytest.fixture
    def chain(self):
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试问题"
        )
        chain.add_step(ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.OBSERVATION,
            content="观察",
            reasoning="原因",
            confidence=0.8
        ))
        return chain

    @pytest.mark.asyncio
    async def test_validate_with_all_validators(self, validator, chain):
        """测试所有验证器组合"""
        step = ThoughtStep(
            step_number=2,
            step_type=ThoughtStepType.ANALYSIS,
            content="分析内容",
            reasoning="分析原因",
            confidence=0.75
        )
        
        validation = await validator.validate(step, chain)
        
        # 组合验证器应该综合所有结果
        assert validation.is_valid
        assert validation.consistency_score > 0


class TestRecoveryMechanisms:
    """测试恢复机制"""

    @pytest.fixture
    def detector(self):
        return FailureDetector()

    @pytest.fixture
    def backtrack(self):
        return BacktrackMechanism()

    @pytest.fixture
    def manager(self):
        return RecoveryManager()

    @pytest.mark.asyncio
    async def test_detect_failure(self, detector):
        """测试失败检测"""
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试"
        )
        
        step = ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.ANALYSIS,
            content="分析",
            reasoning="推理",
            confidence=0.1  # 非常低的置信度
        )
        
        failure_info = await detector.detect_failure(step, chain)
        
        assert failure_info is not None
        assert failure_info["failure_type"] == "low_confidence"
        assert failure_info["severity"] == "critical"

    def test_create_and_find_checkpoint(self, backtrack):
        """测试创建和查找检查点"""
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试"
        )
        
        # 添加高质量步骤
        for i in range(3):
            chain.add_step(ThoughtStep(
                step_number=i+1,
                step_type=ThoughtStepType.ANALYSIS,
                content=f"分析{i+1}",
                reasoning="推理",
                confidence=0.9
            ))
            backtrack.create_checkpoint(chain, i+1)
        
        # 添加低质量步骤
        chain.add_step(ThoughtStep(
            step_number=4,
            step_type=ThoughtStepType.ANALYSIS,
            content="错误分析",
            reasoning="错误",
            confidence=0.2
        ))
        
        # 应该找到之前的高质量检查点
        backtrack_point = backtrack.find_backtrack_point(chain)
        assert backtrack_point is not None
        assert backtrack_point <= 3

    def test_backtrack_to_checkpoint(self, backtrack):
        """测试回溯到检查点"""
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试"
        )
        
        # 添加步骤并创建检查点
        for i in range(5):
            chain.add_step(ThoughtStep(
                step_number=i+1,
                step_type=ThoughtStepType.ANALYSIS,
                content=f"分析{i+1}",
                reasoning="推理",
                confidence=0.8
            ))
            if i == 2:  # 在第3步创建检查点
                backtrack.create_checkpoint(chain, i+1)
        
        # 回溯到检查点
        success = backtrack.backtrack_to(chain, 3)
        
        assert success
        assert len(chain.steps) == 3

    @pytest.mark.asyncio
    async def test_generate_alternative_path(self):
        """测试生成替代路径"""
        generator = AlternativePathGenerator()
        
        failed_step = ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.ANALYSIS,
            content="失败的分析",
            reasoning="错误推理",
            confidence=0.3
        )
        
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试"
        )
        
        failure_info = {
            "failure_type": "low_confidence",
            "suggestions": ["收集更多信息"]
        }
        
        alternative = await generator.generate_alternative(
            failed_step, chain, failure_info
        )
        
        assert alternative is not None
        assert "strategy" in alternative
        assert "prompt_modifier" in alternative

    @pytest.mark.asyncio
    async def test_recovery_manager_flow(self, manager):
        """测试完整的恢复管理流程"""
        chain = ReasoningChain(
            strategy=RS.ZERO_SHOT,
            problem="测试"
        )
        
        # 低质量步骤
        step = ThoughtStep(
            step_number=1,
            step_type=ThoughtStepType.ANALYSIS,
            content="不确定的分析",
            reasoning="缺乏信息",
            confidence=0.2
        )
        
        strategy = await manager.handle_failure(step, chain)
        
        assert strategy is not None
        assert isinstance(strategy, RecoveryStrategy)


def test_calculate_chain_quality_score():
    """测试计算推理链质量分数"""
    chain = ReasoningChain(
        strategy=RS.ZERO_SHOT,
        problem="测试"
    )
    
    # 空链
    assert calculate_chain_quality_score(chain) == 0.0
    
    # 添加完整的推理步骤
    chain.add_step(ThoughtStep(
        step_number=1,
        step_type=ThoughtStepType.OBSERVATION,
        content="观察",
        reasoning="原因",
        confidence=0.8
    ))
    chain.add_step(ThoughtStep(
        step_number=2,
        step_type=ThoughtStepType.ANALYSIS,
        content="分析",
        reasoning="原因",
        confidence=0.85
    ))
    chain.add_step(ThoughtStep(
        step_number=3,
        step_type=ThoughtStepType.CONCLUSION,
        content="结论",
        reasoning="原因",
        confidence=0.9
    ))
    
    score = calculate_chain_quality_score(chain)
    assert score > 0.8  # 应该有较高的分数