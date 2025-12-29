"""
Task 5社交智能决策引擎完整单元测试套件
测试SocialIntelligenceEngine的所有功能和决策逻辑
"""

from src.core.utils.timezone_utils import utc_now
import pytest
import asyncio
from datetime import timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
from ai.emotion_modeling.social_intelligence_engine import (
    SocialIntelligenceEngine,
    DecisionType,
    DecisionPriority, 
    DecisionContext,
    SocialDecision,
    DecisionOutcome
)
from ai.emotion_modeling.models import EmotionVector, SocialContext
from ai.emotion_modeling.social_context_adapter import SocialEnvironment
from ai.emotion_modeling.cultural_context_analyzer import CulturalProfile

@pytest.fixture
def engine():
    """创建社交智能引擎实例"""
    return SocialIntelligenceEngine()

@pytest.fixture
def mock_cultural_profile():
    """创建测试文化档案"""
    return Mock(spec=CulturalProfile)

@pytest.fixture
def sample_decision_context(mock_cultural_profile):
    """创建测试决策上下文"""
    mock_cultural_profile.culture_id = "western_individualistic"
    
    return DecisionContext(
        session_id="test_session_001",
        timestamp=utc_now(),
        participants=[
            {"user_id": "user1", "role": "participant"},
            {"user_id": "user2", "role": "facilitator"}
        ],
        current_emotions={
            "user1": EmotionVector(
                emotions={"happiness": 0.6, "confidence": 0.7},
                intensity=0.7,
                confidence=0.8,
                context={}
            ),
            "user2": EmotionVector(
                emotions={"neutral": 0.8, "focused": 0.6},
                intensity=0.5,
                confidence=0.9,
                context={}
            )
        },
        group_dynamics={
            "cohesion_score": 0.7,
            "energy_level": 0.6,
            "participation_balance": 0.8,
            "relationship_tensions": 0.2
        },
        social_environment=SocialEnvironment(
            environment_type="business_meeting",
            formality_level=0.7,
            participants_count=2,
            time_pressure=0.4,
            physical_setup="virtual",
            cultural_context="mixed"
        ),
        cultural_profiles=[mock_cultural_profile],
        historical_context=[
            {"event": "previous_meeting", "outcome": "successful"}
        ],
        constraints={"time_limit": 60, "budget_limit": 1000}
    )

@pytest.fixture
def high_conflict_context(mock_cultural_profile):
    """创建高冲突测试上下文"""
    mock_cultural_profile.culture_id = "east_asian_collectivistic"
    
    return DecisionContext(
        session_id="conflict_session_001",
        timestamp=utc_now(),
        participants=[
            {"user_id": "user1", "role": "manager"},
            {"user_id": "user2", "role": "employee"},
            {"user_id": "user3", "role": "employee"}
        ],
        current_emotions={
            "user1": EmotionVector(
                emotions={"anger": 0.8, "frustration": 0.7, "tension": 0.9},
                intensity=0.9,
                confidence=0.8,
                context={}
            ),
            "user2": EmotionVector(
                emotions={"anxiety": 0.8, "defensiveness": 0.7, "anger": 0.6},
                intensity=0.8,
                confidence=0.7,
                context={}
            ),
            "user3": EmotionVector(
                emotions={"worry": 0.7, "tension": 0.8, "discomfort": 0.6},
                intensity=0.7,
                confidence=0.8,
                context={}
            )
        },
        group_dynamics={
            "cohesion_score": 0.2,
            "energy_level": 0.9,
            "participation_balance": 0.3,
            "relationship_tensions": 0.9
        },
        social_environment=SocialEnvironment(
            environment_type="conflict_resolution",
            formality_level=0.8,
            participants_count=3,
            time_pressure=0.7,
            physical_setup="in_person",
            cultural_context="homogeneous"
        ),
        cultural_profiles=[mock_cultural_profile],
        historical_context=[
            {"event": "previous_conflict", "outcome": "unresolved"}
        ]
    )

class TestSocialIntelligenceEngine:
    """社交智能引擎基础功能测试"""
    
    def test_initialization(self, engine):
        """测试初始化"""
        assert engine is not None
        assert hasattr(engine, 'group_analyzer')
        assert hasattr(engine, 'relationship_analyzer') 
        assert hasattr(engine, 'context_adapter')
        assert hasattr(engine, 'cultural_analyzer')
        assert len(engine.decision_history) == 0
        assert len(engine.outcome_history) == 0
        assert engine.learning_enabled is True
        assert engine.max_history_size == 1000
    
    @pytest.mark.asyncio
    async def test_analyze_and_decide_basic(self, engine, sample_decision_context):
        """测试基础分析和决策"""
        decisions = await engine.analyze_and_decide(sample_decision_context)
        
        assert isinstance(decisions, list)
        assert len(decisions) > 0
        assert all(isinstance(decision, SocialDecision) for decision in decisions)
        
        # 验证决策被存储到历史中
        assert len(engine.decision_history) == len(decisions)
        
        # 验证决策按优先级排序
        if len(decisions) > 1:
            priorities = [engine._priority_score(d.priority) for d in decisions]
            assert priorities == sorted(priorities, reverse=True)
    
    @pytest.mark.asyncio
    async def test_analyze_and_decide_with_specific_types(self, engine, sample_decision_context):
        """测试指定决策类型"""
        specific_types = [DecisionType.COMMUNICATION_STYLE, DecisionType.GROUP_FACILITATION]
        
        decisions = await engine.analyze_and_decide(sample_decision_context, specific_types)
        
        assert len(decisions) <= len(specific_types)
        decision_types = [d.decision_type for d in decisions]
        assert all(dt in specific_types for dt in decision_types)
    
    @pytest.mark.asyncio
    async def test_analyze_and_decide_exception_handling(self, engine, sample_decision_context):
        """测试异常处理"""
        with patch.object(engine, '_identify_required_decisions', side_effect=Exception("测试异常")):
            decisions = await engine.analyze_and_decide(sample_decision_context)
            assert decisions == []

class TestDecisionTypeIdentification:
    """决策类型识别测试"""
    
    @pytest.mark.asyncio
    async def test_identify_required_decisions_normal(self, engine, sample_decision_context):
        """测试正常情况下的决策类型识别"""
        decisions = await engine._identify_required_decisions(sample_decision_context)
        
        assert isinstance(decisions, list)
        assert len(decisions) > 0
        assert all(isinstance(dt, DecisionType) for dt in decisions)
    
    @pytest.mark.asyncio
    async def test_identify_high_emotion_intensity(self, engine, sample_decision_context):
        """测试高情感强度识别"""
        # 修改情感强度
        for emotion_vec in sample_decision_context.current_emotions.values():
            emotion_vec.intensity = 0.9
            emotion_vec.emotions = {"excitement": 0.95, "energy": 0.9}
        
        decisions = await engine._identify_required_decisions(sample_decision_context)
        
        assert DecisionType.EMOTIONAL_REGULATION in decisions
    
    @pytest.mark.asyncio
    async def test_identify_conflict_emotions(self, engine, high_conflict_context):
        """测试冲突情感识别"""
        decisions = await engine._identify_required_decisions(high_conflict_context)
        
        assert DecisionType.CONFLICT_INTERVENTION in decisions
        assert DecisionType.EMOTIONAL_REGULATION in decisions
    
    @pytest.mark.asyncio
    async def test_identify_large_group(self, engine, sample_decision_context):
        """测试大型群体识别"""
        sample_decision_context.social_environment.participants_count = 10
        
        decisions = await engine._identify_required_decisions(sample_decision_context)
        
        assert DecisionType.GROUP_FACILITATION in decisions
    
    @pytest.mark.asyncio
    async def test_identify_high_formality(self, engine, sample_decision_context):
        """测试高正式度识别"""
        sample_decision_context.social_environment.formality_level = 0.9
        
        decisions = await engine._identify_required_decisions(sample_decision_context)
        
        assert DecisionType.COMMUNICATION_STYLE in decisions
    
    @pytest.mark.asyncio
    async def test_identify_cultural_diversity(self, engine, sample_decision_context):
        """测试文化多样性识别"""
        mock_profile2 = Mock()
        mock_profile2.culture_id = "east_asian_collectivistic"
        sample_decision_context.cultural_profiles.append(mock_profile2)
        
        decisions = await engine._identify_required_decisions(sample_decision_context)
        
        assert DecisionType.CULTURAL_ADAPTATION in decisions
    
    @pytest.mark.asyncio
    async def test_identify_relationship_tensions(self, engine, sample_decision_context):
        """测试关系紧张识别"""
        sample_decision_context.group_dynamics["relationship_tensions"] = 0.8
        
        decisions = await engine._identify_required_decisions(sample_decision_context)
        
        assert DecisionType.RELATIONSHIP_BUILDING in decisions

class TestSpecificDecisionGeneration:
    """具体决策生成测试"""
    
    @pytest.mark.asyncio
    async def test_decide_communication_style_formal(self, engine, sample_decision_context):
        """测试正式沟通风格决策"""
        sample_decision_context.social_environment.formality_level = 0.8
        
        decision = await engine._decide_communication_style("test_id", sample_decision_context)
        
        assert decision is not None
        assert decision.decision_type == DecisionType.COMMUNICATION_STYLE
        assert "adopt_formal_communication_style" in decision.recommended_actions
        assert "professional" in decision.reasoning.lower()
        assert decision.confidence_score > 0.5
    
    @pytest.mark.asyncio
    async def test_decide_communication_style_casual(self, engine, sample_decision_context):
        """测试非正式沟通风格决策"""
        sample_decision_context.social_environment.formality_level = 0.2
        
        decision = await engine._decide_communication_style("test_id", sample_decision_context)
        
        assert "use_casual_friendly_tone" in decision.recommended_actions
        assert "casual" in decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_decide_communication_style_large_group(self, engine, sample_decision_context):
        """测试大型群体沟通决策"""
        sample_decision_context.social_environment.participants_count = 15
        
        decision = await engine._decide_communication_style("test_id", sample_decision_context)
        
        structured_actions = [
            "use_clear_projection_techniques",
            "implement_turn_taking_system", 
            "provide_structured_participation"
        ]
        assert any(action in decision.recommended_actions for action in structured_actions)
    
    @pytest.mark.asyncio
    async def test_decide_communication_style_cultural_diversity(self, engine, sample_decision_context):
        """测试文化多样性沟通决策"""
        mock_profile2 = Mock()
        mock_profile2.culture_id = "latin_expressive"
        sample_decision_context.cultural_profiles.append(mock_profile2)
        
        decision = await engine._decide_communication_style("test_id", sample_decision_context)
        
        inclusive_actions = [
            "use_inclusive_language",
            "avoid_culture_specific_references",
            "check_for_understanding_regularly"
        ]
        assert any(action in decision.recommended_actions for action in inclusive_actions)
    
    @pytest.mark.asyncio
    async def test_decide_conflict_intervention(self, engine, high_conflict_context):
        """测试冲突干预决策"""
        decision = await engine._decide_conflict_intervention("test_id", high_conflict_context)
        
        assert decision is not None
        assert decision.decision_type == DecisionType.CONFLICT_INTERVENTION
        assert decision.priority in [DecisionPriority.CRITICAL, DecisionPriority.HIGH]
        assert len(decision.recommended_actions) > 0
        assert "conflict" in decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_decide_group_facilitation(self, engine, sample_decision_context):
        """测试群体引导决策"""
        decision = await engine._decide_group_facilitation("test_id", sample_decision_context)
        
        assert decision is not None
        assert decision.decision_type == DecisionType.GROUP_FACILITATION
        assert len(decision.recommended_actions) > 0
        assert decision.confidence_score > 0.0
    
    @pytest.mark.asyncio
    async def test_decide_emotional_regulation(self, engine, high_conflict_context):
        """测试情感调节决策"""
        decision = await engine._decide_emotional_regulation("test_id", high_conflict_context)
        
        assert decision is not None
        assert decision.decision_type == DecisionType.EMOTIONAL_REGULATION
        assert len(decision.recommended_actions) > 0
        regulation_keywords = ["calm", "regulate", "manage", "reduce", "balance"]
        assert any(keyword in " ".join(decision.recommended_actions).lower() for keyword in regulation_keywords)
    
    @pytest.mark.asyncio
    async def test_generate_decision_unknown_type(self, engine, sample_decision_context):
        """测试未知决策类型处理"""
        # 创建一个不存在的决策类型枚举值进行模拟
        with patch('ai.emotion_modeling.social_intelligence_engine.DecisionType') as mock_enum:
            mock_enum.UNKNOWN_TYPE = "unknown_type"
            decision = await engine._generate_decision(
                sample_decision_context, mock_enum.UNKNOWN_TYPE
            )
            assert decision is None

class TestDecisionPriority:
    """决策优先级测试"""
    
    def test_priority_score_calculation(self, engine):
        """测试优先级分数计算"""
        assert engine._priority_score(DecisionPriority.CRITICAL) == 4
        assert engine._priority_score(DecisionPriority.HIGH) == 3
        assert engine._priority_score(DecisionPriority.MEDIUM) == 2
        assert engine._priority_score(DecisionPriority.LOW) == 1
    
    @pytest.mark.asyncio
    async def test_decisions_sorted_by_priority(self, engine, high_conflict_context):
        """测试决策按优先级排序"""
        decisions = await engine.analyze_and_decide(high_conflict_context)
        
        if len(decisions) > 1:
            for i in range(len(decisions) - 1):
                current_priority = engine._priority_score(decisions[i].priority)
                next_priority = engine._priority_score(decisions[i + 1].priority)
                assert current_priority >= next_priority

class TestDecisionOutcomeTracking:
    """决策结果追踪测试"""
    
    @pytest.mark.asyncio
    async def test_record_decision_outcome_success(self, engine, sample_decision_context):
        """测试成功决策结果记录"""
        decisions = await engine.analyze_and_decide(sample_decision_context)
        assert len(decisions) > 0
        
        test_decision = decisions[0]
        outcome = DecisionOutcome(
            decision_id=test_decision.decision_id,
            execution_success=True,
            actual_outcomes={"improved_communication": True, "reduced_tension": True},
            participant_feedback={"user1": 0.8, "user2": 0.9},
            effectiveness_score=0.85,
            lessons_learned=["Clear communication worked well"],
            timestamp=utc_now()
        )
        
        await engine.record_decision_outcome(outcome)
        
        assert test_decision.decision_id in engine.outcome_history
        stored_outcome = engine.outcome_history[test_decision.decision_id]
        assert stored_outcome.effectiveness_score == 0.85
        assert stored_outcome.execution_success is True
    
    @pytest.mark.asyncio
    async def test_record_decision_outcome_failure(self, engine, sample_decision_context):
        """测试失败决策结果记录"""
        decisions = await engine.analyze_and_decide(sample_decision_context)
        test_decision = decisions[0]
        
        outcome = DecisionOutcome(
            decision_id=test_decision.decision_id,
            execution_success=False,
            actual_outcomes={"communication_breakdown": True},
            participant_feedback={"user1": 0.3, "user2": 0.2},
            effectiveness_score=0.25,
            lessons_learned=["Approach was too direct for this cultural context"],
            timestamp=utc_now()
        )
        
        await engine.record_decision_outcome(outcome)
        
        stored_outcome = engine.outcome_history[test_decision.decision_id]
        assert stored_outcome.execution_success is False
        assert stored_outcome.effectiveness_score == 0.25

class TestLearningSystem:
    """学习系统测试"""
    
    @pytest.mark.asyncio 
    async def test_learn_from_successful_outcome(self, engine, sample_decision_context):
        """测试从成功结果学习"""
        decisions = await engine.analyze_and_decide(sample_decision_context)
        test_decision = decisions[0]
        
        # 记录成功结果
        outcome = DecisionOutcome(
            decision_id=test_decision.decision_id,
            execution_success=True,
            actual_outcomes={"success": True},
            participant_feedback={"user1": 0.9},
            effectiveness_score=0.9,
            lessons_learned=["Strategy worked well"],
            timestamp=utc_now()
        )
        
        await engine.record_decision_outcome(outcome)
        
        # 验证学习发生
        if engine.learning_enabled:
            decision_type = test_decision.decision_type
            if decision_type in engine.success_metrics:
                assert engine.success_metrics[decision_type] > 0
    
    @pytest.mark.asyncio
    async def test_learning_disabled(self, engine, sample_decision_context):
        """测试学习功能禁用"""
        engine.learning_enabled = False
        
        decisions = await engine.analyze_and_decide(sample_decision_context)
        test_decision = decisions[0]
        
        outcome = DecisionOutcome(
            decision_id=test_decision.decision_id,
            execution_success=True,
            actual_outcomes={"success": True},
            participant_feedback={},
            effectiveness_score=0.8,
            lessons_learned=[],
            timestamp=utc_now()
        )
        
        original_patterns_count = len(engine.decision_patterns)
        await engine.record_decision_outcome(outcome)
        
        # 验证学习没有发生
        assert len(engine.decision_patterns) == original_patterns_count

class TestHistoryManagement:
    """历史记录管理测试"""
    
    @pytest.mark.asyncio
    async def test_cleanup_history_within_limit(self, engine, sample_decision_context):
        """测试历史记录在限制内清理"""
        engine.max_history_size = 5
        
        # 生成多个决策
        for i in range(3):
            context_copy = sample_decision_context
            context_copy.session_id = f"session_{i}"
            await engine.analyze_and_decide(context_copy)
        
        await engine._cleanup_history()
        
        # 验证历史记录没有超过限制
        assert len(engine.decision_history) <= engine.max_history_size
        assert len(engine.outcome_history) <= engine.max_history_size
    
    @pytest.mark.asyncio
    async def test_cleanup_history_exceeds_limit(self, engine, sample_decision_context):
        """测试历史记录超出限制时的清理"""
        engine.max_history_size = 2
        
        # 生成超过限制的决策
        for i in range(5):
            context_copy = sample_decision_context
            context_copy.session_id = f"session_{i}"
            context_copy.timestamp = utc_now() + timedelta(minutes=i)
            await engine.analyze_and_decide(context_copy)
        
        await engine._cleanup_history()
        
        # 验证清理后的数量
        assert len(engine.decision_history) <= engine.max_history_size
    
    def test_get_decision_history(self, engine):
        """测试获取决策历史"""
        history = engine.get_decision_history()
        assert isinstance(history, dict)
    
    def test_get_outcome_history(self, engine):
        """测试获取结果历史"""
        history = engine.get_outcome_history()
        assert isinstance(history, dict)

class TestDecisionQuality:
    """决策质量测试"""
    
    @pytest.mark.asyncio
    async def test_decision_completeness(self, engine, sample_decision_context):
        """测试决策完整性"""
        decisions = await engine.analyze_and_decide(sample_decision_context)
        
        for decision in decisions:
            assert decision.decision_id is not None
            assert decision.decision_type is not None
            assert decision.priority is not None
            assert decision.context is not None
            assert len(decision.recommended_actions) > 0
            assert decision.reasoning is not None
            assert 0.0 <= decision.confidence_score <= 1.0
            assert isinstance(decision.expected_outcomes, dict)
            assert isinstance(decision.alternative_options, list)
            assert decision.execution_timeline is not None
            assert isinstance(decision.monitoring_metrics, list)
            assert isinstance(decision.risk_assessment, dict)
    
    @pytest.mark.asyncio
    async def test_decision_confidence_scores(self, engine, sample_decision_context):
        """测试决策信心分数"""
        decisions = await engine.analyze_and_decide(sample_decision_context)
        
        for decision in decisions:
            assert 0.0 <= decision.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_expected_outcomes_probabilities(self, engine, sample_decision_context):
        """测试期望结果概率"""
        decisions = await engine.analyze_and_decide(sample_decision_context)
        
        for decision in decisions:
            for outcome, probability in decision.expected_outcomes.items():
                assert 0.0 <= probability <= 1.0
    
    @pytest.mark.asyncio
    async def test_risk_assessment_values(self, engine, sample_decision_context):
        """测试风险评估值"""
        decisions = await engine.analyze_and_decide(sample_decision_context)
        
        for decision in decisions:
            for risk, probability in decision.risk_assessment.items():
                assert 0.0 <= probability <= 1.0

class TestErrorHandlingAndEdgeCases:
    """错误处理和边界条件测试"""
    
    @pytest.mark.asyncio
    async def test_empty_current_emotions(self, engine, sample_decision_context):
        """测试空当前情感"""
        sample_decision_context.current_emotions = {}
        
        decisions = await engine.analyze_and_decide(sample_decision_context)
        
        # 应该仍然能生成基础决策
        assert isinstance(decisions, list)
    
    @pytest.mark.asyncio
    async def test_no_cultural_profiles(self, engine, sample_decision_context):
        """测试无文化档案"""
        sample_decision_context.cultural_profiles = []
        
        decisions = await engine.analyze_and_decide(sample_decision_context)
        
        # 不应该包含文化适配决策
        decision_types = [d.decision_type for d in decisions]
        assert DecisionType.CULTURAL_ADAPTATION not in decision_types
    
    @pytest.mark.asyncio
    async def test_invalid_emotion_values(self, engine, sample_decision_context):
        """测试无效情感值"""
        # 创建包含无效值的情感向量
        invalid_emotion = EmotionVector(
            emotions={"invalid": float('nan'), "negative": -0.5},
            intensity=-1.0,
            confidence=2.0,
            context={}
        )
        
        sample_decision_context.current_emotions["invalid_user"] = invalid_emotion
        
        # 应该能够处理无效值而不崩溃
        decisions = await engine.analyze_and_decide(sample_decision_context)
        assert isinstance(decisions, list)
    
    @pytest.mark.asyncio
    async def test_extreme_participant_count(self, engine, sample_decision_context):
        """测试极端参与者数量"""
        # 测试0个参与者
        sample_decision_context.social_environment.participants_count = 0
        decisions_zero = await engine.analyze_and_decide(sample_decision_context)
        assert isinstance(decisions_zero, list)
        
        # 测试大量参与者
        sample_decision_context.social_environment.participants_count = 1000
        decisions_large = await engine.analyze_and_decide(sample_decision_context)
        assert isinstance(decisions_large, list)
        
        # 大量参与者应该触发群体引导决策
        decision_types = [d.decision_type for d in decisions_large]
        assert DecisionType.GROUP_FACILITATION in decision_types

class TestIntegrationScenarios:
    """集成场景测试"""
    
    @pytest.mark.asyncio
    async def test_complete_decision_workflow(self, engine, sample_decision_context):
        """测试完整决策工作流"""
        # 1. 生成决策
        decisions = await engine.analyze_and_decide(sample_decision_context)
        assert len(decisions) > 0
        
        # 2. 选择一个决策执行
        primary_decision = decisions[0]
        
        # 3. 模拟执行结果
        outcome = DecisionOutcome(
            decision_id=primary_decision.decision_id,
            execution_success=True,
            actual_outcomes={"communication_improved": True},
            participant_feedback={"user1": 0.8, "user2": 0.9},
            effectiveness_score=0.85,
            lessons_learned=["Strategy was effective for this context"],
            timestamp=utc_now()
        )
        
        # 4. 记录结果
        await engine.record_decision_outcome(outcome)
        
        # 5. 验证学习发生
        assert primary_decision.decision_id in engine.outcome_history
        
        # 6. 生成后续决策应该受到学习影响
        follow_up_decisions = await engine.analyze_and_decide(sample_decision_context)
        assert len(follow_up_decisions) > 0
    
    @pytest.mark.asyncio
    async def test_multi_cultural_complex_scenario(self, engine):
        """测试多文化复杂场景"""
        # 创建复杂的多文化场景
        mock_profiles = []
        for culture_id in ["western_individualistic", "east_asian_collectivistic", "latin_expressive"]:
            profile = Mock()
            profile.culture_id = culture_id
            mock_profiles.append(profile)
        
        complex_context = DecisionContext(
            session_id="multi_cultural_session",
            timestamp=utc_now(),
            participants=[
                {"user_id": f"user_{i}", "role": "participant"} 
                for i in range(8)
            ],
            current_emotions={
                f"user_{i}": EmotionVector(
                    emotions={"mixed_emotions": 0.5, "cultural_tension": 0.3},
                    intensity=0.6,
                    confidence=0.7,
                    context={}
                ) for i in range(8)
            },
            group_dynamics={
                "cohesion_score": 0.4,
                "energy_level": 0.7,
                "participation_balance": 0.5,
                "relationship_tensions": 0.6
            },
            social_environment=SocialEnvironment(
                environment_type="international_conference",
                formality_level=0.8,
                participants_count=8,
                time_pressure=0.6,
                physical_setup="hybrid",
                cultural_context="highly_diverse"
            ),
            cultural_profiles=mock_profiles
        )
        
        decisions = await engine.analyze_and_decide(complex_context)
        
        # 验证生成了多种类型的决策
        decision_types = [d.decision_type for d in decisions]
        expected_types = [
            DecisionType.CULTURAL_ADAPTATION,
            DecisionType.COMMUNICATION_STYLE,
            DecisionType.GROUP_FACILITATION
        ]
        assert any(dt in decision_types for dt in expected_types)
        
        # 验证文化适配决策的存在
        assert DecisionType.CULTURAL_ADAPTATION in decision_types

class TestPerformanceAndOptimization:
    """性能和优化测试"""
    
    @pytest.mark.asyncio
    async def test_decision_generation_performance(self, engine):
        """测试决策生成性能"""
        import time
        
        # 创建大型场景
        large_context = DecisionContext(
            session_id="performance_test",
            timestamp=utc_now(),
            participants=[{"user_id": f"user_{i}"} for i in range(50)],
            current_emotions={
                f"user_{i}": EmotionVector(
                    emotions={"neutral": 0.8},
                    intensity=0.5,
                    confidence=0.8,
                    context={}
                ) for i in range(50)
            },
            group_dynamics={
                "cohesion_score": 0.6,
                "energy_level": 0.5,
                "participation_balance": 0.7,
                "relationship_tensions": 0.3
            },
            social_environment=SocialEnvironment(
                environment_type="large_conference",
                formality_level=0.7,
                participants_count=50,
                time_pressure=0.5,
                physical_setup="auditorium",
                cultural_context="mixed"
            ),
            cultural_profiles=[Mock()]
        )
        
        start_time = time.time()
        decisions = await engine.analyze_and_decide(large_context)
        end_time = time.time()
        
        # 验证性能
        processing_time = end_time - start_time
        assert processing_time < 10.0  # 10秒内完成
        assert len(decisions) > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_history(self, engine, sample_decision_context):
        """测试大量历史记录的内存使用"""
        import sys
        
        initial_size = sys.getsizeof(engine.decision_history)
        
        # 生成大量决策
        for i in range(50):
            context_copy = sample_decision_context
            context_copy.session_id = f"memory_test_{i}"
            await engine.analyze_and_decide(context_copy)
        
        final_size = sys.getsizeof(engine.decision_history)
        
        # 验证内存使用合理
        memory_growth = final_size - initial_size
        assert memory_growth > 0  # 应该有增长
        # 但不应该过度增长（具体阈值依据实际情况调整）

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
