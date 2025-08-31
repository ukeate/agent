import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

from ai.emotion_modeling.social_context_adapter import (
    SocialContextAdapter,
    SocialScenario,
    ContextRule,
    AdaptationResult,
    EmotionData,
    SocialDynamics
)
from ai.emotion_modeling.models import (
    EmotionVector,
    SocialContext,
    CulturalProfile
)

class TestSocialContextAdapter:
    """社交场景适配系统测试套件"""
    
    @pytest.fixture
    def adapter(self):
        """创建测试用的社交上下文适配器实例"""
        return SocialContextAdapter()
    
    @pytest.fixture
    def sample_emotion_data(self):
        """创建示例情感数据"""
        return EmotionData(
            user_id="user_123",
            emotion_vector=EmotionVector(
                joy=0.3,
                sadness=0.1,
                anger=0.05,
                fear=0.2,
                surprise=0.15,
                disgust=0.02,
                trust=0.18
            ),
            confidence=0.85,
            context_markers=["workplace", "meeting", "presentation"],
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_social_context(self):
        """创建示例社交上下文"""
        return SocialContext(
            scenario=SocialScenario.TEAM_MEETING,
            participants=["user_123", "user_456", "user_789"],
            formality_level=0.7,
            power_dynamics={"user_456": 0.8, "user_123": 0.5, "user_789": 0.6},
            cultural_context="western_individualistic",
            time_pressure=0.4,
            relationship_matrix={
                ("user_123", "user_456"): 0.7,
                ("user_123", "user_789"): 0.8,
                ("user_456", "user_789"): 0.6
            }
        )
    
    @pytest.fixture
    def sample_cultural_profile(self):
        """创建示例文化档案"""
        return CulturalProfile(
            cultural_dimension="western_individualistic",
            power_distance=0.3,
            individualism=0.8,
            uncertainty_avoidance=0.4,
            long_term_orientation=0.6,
            communication_style="direct",
            emotional_expression_norms={
                "joy": 0.8,
                "sadness": 0.3,
                "anger": 0.2,
                "fear": 0.4
            }
        )

    # 基本功能测试
    def test_adapter_initialization(self, adapter):
        """测试适配器初始化"""
        assert adapter is not None
        assert hasattr(adapter, 'adaptation_rules')
        assert hasattr(adapter, 'scenario_configs')
        assert len(adapter.adaptation_rules) > 0
        assert len(adapter.scenario_configs) > 0
    
    def test_scenario_detection(self, adapter, sample_social_context):
        """测试场景识别功能"""
        detected_scenario = adapter.detect_scenario(sample_social_context)
        assert isinstance(detected_scenario, SocialScenario)
        assert detected_scenario == SocialScenario.TEAM_MEETING
    
    @pytest.mark.asyncio
    async def test_basic_adaptation(self, adapter, sample_emotion_data, sample_social_context):
        """测试基本的情感适配功能"""
        result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=sample_social_context
        )
        
        assert isinstance(result, AdaptationResult)
        assert result.adapted_emotion_vector is not None
        assert result.confidence_score > 0
        assert result.adaptation_rationale is not None
        assert len(result.applied_rules) > 0
    
    # 场景特定适配测试
    @pytest.mark.asyncio
    async def test_workplace_meeting_adaptation(self, adapter, sample_emotion_data):
        """测试工作场所会议场景的适配"""
        meeting_context = SocialContext(
            scenario=SocialScenario.TEAM_MEETING,
            participants=["user_123", "manager_001"],
            formality_level=0.8,
            power_dynamics={"manager_001": 0.9, "user_123": 0.4},
            cultural_context="western_individualistic"
        )
        
        result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=meeting_context
        )
        
        # 在正式场合，负面情感应该被抑制
        original_anger = sample_emotion_data.emotion_vector.anger
        adapted_anger = result.adapted_emotion_vector.anger
        assert adapted_anger <= original_anger
        
        # 应该有相关的适配规则被应用
        rule_types = [rule.rule_type for rule in result.applied_rules]
        assert "suppress_negative_emotion" in rule_types or "formality_adjustment" in rule_types
    
    @pytest.mark.asyncio
    async def test_informal_social_adaptation(self, adapter, sample_emotion_data):
        """测试非正式社交场景的适配"""
        informal_context = SocialContext(
            scenario=SocialScenario.CASUAL_CONVERSATION,
            participants=["user_123", "friend_001"],
            formality_level=0.2,
            power_dynamics={"user_123": 0.5, "friend_001": 0.5},
            cultural_context="western_individualistic",
            relationship_matrix={("user_123", "friend_001"): 0.9}
        )
        
        result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=informal_context
        )
        
        # 在非正式场合，情感表达应该更加自由
        assert result.adapted_emotion_vector.joy >= sample_emotion_data.emotion_vector.joy * 0.9
        assert result.confidence_score > 0.7
    
    # 文化适应测试
    @pytest.mark.asyncio
    async def test_cultural_adaptation(self, adapter, sample_emotion_data, sample_cultural_profile):
        """测试文化适应功能"""
        # 东方集体主义文化上下文
        eastern_context = SocialContext(
            scenario=SocialScenario.GROUP_DISCUSSION,
            participants=["user_123", "user_456", "user_789"],
            formality_level=0.6,
            cultural_context="eastern_collectivistic"
        )
        
        result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=eastern_context,
            cultural_profile=sample_cultural_profile
        )
        
        # 在集体主义文化中，个人情感表达应该更加克制
        assert result.adapted_emotion_vector.anger <= sample_emotion_data.emotion_vector.anger
        assert "cultural_adjustment" in [rule.rule_type for rule in result.applied_rules]
    
    # 权力动态测试
    @pytest.mark.asyncio
    async def test_power_dynamics_adaptation(self, adapter, sample_emotion_data):
        """测试权力动态适配"""
        hierarchical_context = SocialContext(
            scenario=SocialScenario.FORMAL_PRESENTATION,
            participants=["user_123", "executive_001"],
            formality_level=0.9,
            power_dynamics={"executive_001": 0.95, "user_123": 0.3},
            cultural_context="western_individualistic"
        )
        
        # 测试向上级表达情感
        result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=hierarchical_context
        )
        
        # 向权威人物表达时应该更加谨慎
        assert result.adapted_emotion_vector.fear <= sample_emotion_data.emotion_vector.fear * 1.2
        assert result.confidence_score > 0.6
        assert any("hierarchy_respect" in rule.rule_id for rule in result.applied_rules)
    
    # 时间压力适配测试
    @pytest.mark.asyncio
    async def test_time_pressure_adaptation(self, adapter, sample_emotion_data):
        """测试时间压力下的情感适配"""
        urgent_context = SocialContext(
            scenario=SocialScenario.CRISIS_MANAGEMENT,
            participants=["user_123", "team_lead_001"],
            formality_level=0.8,
            time_pressure=0.9,  # 高时间压力
            cultural_context="western_individualistic"
        )
        
        result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=urgent_context
        )
        
        # 在紧急情况下，焦虑可能会增加，但愤怒应该被控制
        assert result.adapted_emotion_vector.anger <= sample_emotion_data.emotion_vector.anger
        assert "time_pressure_adjustment" in [rule.rule_type for rule in result.applied_rules]
    
    # 关系矩阵测试
    @pytest.mark.asyncio
    async def test_relationship_matrix_influence(self, adapter, sample_emotion_data):
        """测试关系矩阵对适配的影响"""
        close_relationship_context = SocialContext(
            scenario=SocialScenario.INTERPERSONAL_CONFLICT,
            participants=["user_123", "close_friend_001"],
            formality_level=0.3,
            relationship_matrix={("user_123", "close_friend_001"): 0.95},  # 非常亲密的关系
            cultural_context="western_individualistic"
        )
        
        distant_relationship_context = SocialContext(
            scenario=SocialScenario.INTERPERSONAL_CONFLICT,
            participants=["user_123", "stranger_001"],
            formality_level=0.3,
            relationship_matrix={("user_123", "stranger_001"): 0.1},  # 疏远的关系
            cultural_context="western_individualistic"
        )
        
        close_result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=close_relationship_context
        )
        
        distant_result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=distant_relationship_context
        )
        
        # 与亲密朋友的情感表达应该比与陌生人更真实
        assert close_result.adapted_emotion_vector.joy >= distant_result.adapted_emotion_vector.joy
    
    # 规则系统测试
    def test_rule_priority_system(self, adapter):
        """测试规则优先级系统"""
        rules = adapter.get_applicable_rules(SocialScenario.TEAM_MEETING)
        assert len(rules) > 0
        
        # 规则应该按照优先级排序
        priorities = [rule.priority for rule in rules]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_rule_conflict_resolution(self, adapter):
        """测试规则冲突解决机制"""
        # 创建冲突的规则场景
        conflicting_context = SocialContext(
            scenario=SocialScenario.TEAM_MEETING,
            participants=["user_123", "manager_001"],
            formality_level=0.9,  # 高正式度要求抑制情感
            power_dynamics={"manager_001": 0.9, "user_123": 0.3},
            relationship_matrix={("user_123", "manager_001"): 0.9},  # 但关系很亲密
            cultural_context="western_individualistic"
        )
        
        applicable_rules = adapter.get_applicable_rules(
            conflicting_context.scenario,
            formality_level=conflicting_context.formality_level,
            relationship_strength=0.9
        )
        
        # 应该有机制处理规则冲突
        assert len(applicable_rules) > 0
        # 高优先级规则应该生效
        assert max(rule.priority for rule in applicable_rules) >= 7
    
    # 边界条件测试
    @pytest.mark.asyncio
    async def test_empty_emotion_data(self, adapter, sample_social_context):
        """测试空情感数据的处理"""
        empty_emotion = EmotionData(
            user_id="user_123",
            emotion_vector=EmotionVector(
                joy=0, sadness=0, anger=0, fear=0, surprise=0, disgust=0, trust=0
            ),
            confidence=0.1,
            context_markers=[],
            timestamp=datetime.now()
        )
        
        result = await adapter.adapt_emotion_response(
            emotion_data=empty_emotion,
            social_context=sample_social_context
        )
        
        assert result is not None
        assert result.confidence_score >= 0
        assert result.adapted_emotion_vector is not None
    
    @pytest.mark.asyncio
    async def test_extreme_formality_levels(self, adapter, sample_emotion_data):
        """测试极端正式度水平"""
        # 极端正式情况
        extreme_formal_context = SocialContext(
            scenario=SocialScenario.FORMAL_PRESENTATION,
            participants=["user_123", "ceo_001"],
            formality_level=1.0,
            power_dynamics={"ceo_001": 1.0, "user_123": 0.1},
            cultural_context="western_individualistic"
        )
        
        # 极端非正式情况
        extreme_casual_context = SocialContext(
            scenario=SocialScenario.CASUAL_CONVERSATION,
            participants=["user_123", "best_friend"],
            formality_level=0.0,
            power_dynamics={"user_123": 0.5, "best_friend": 0.5},
            relationship_matrix={("user_123", "best_friend"): 1.0},
            cultural_context="western_individualistic"
        )
        
        formal_result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=extreme_formal_context
        )
        
        casual_result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=extreme_casual_context
        )
        
        # 极端正式情况下情感应该被显著抑制
        assert formal_result.adapted_emotion_vector.anger < sample_emotion_data.emotion_vector.anger
        # 极端非正式情况下情感表达应该更自由
        assert casual_result.confidence_score >= formal_result.confidence_score
    
    # 性能测试
    @pytest.mark.asyncio
    async def test_adaptation_performance(self, adapter, sample_emotion_data, sample_social_context):
        """测试适配性能"""
        import time
        
        start_time = time.time()
        
        tasks = []
        for i in range(10):
            task = adapter.adapt_emotion_response(
                emotion_data=sample_emotion_data,
                social_context=sample_social_context
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # 10个适配操作应该在合理时间内完成
        assert end_time - start_time < 5.0  # 5秒内完成
        assert len(results) == 10
        assert all(isinstance(result, AdaptationResult) for result in results)
    
    # 缓存测试
    @pytest.mark.asyncio
    async def test_adaptation_caching(self, adapter, sample_emotion_data, sample_social_context):
        """测试适配结果缓存"""
        # 第一次适配
        result1 = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=sample_social_context
        )
        
        # 相同输入的第二次适配
        result2 = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=sample_social_context
        )
        
        # 结果应该一致（如果有缓存机制）
        assert result1.adapted_emotion_vector.joy == result2.adapted_emotion_vector.joy
        assert result1.confidence_score == result2.confidence_score
    
    # 错误处理测试
    @pytest.mark.asyncio
    async def test_invalid_scenario_handling(self, adapter, sample_emotion_data):
        """测试无效场景的处理"""
        invalid_context = SocialContext(
            scenario=None,  # 无效场景
            participants=["user_123"],
            formality_level=0.5,
            cultural_context="unknown_culture"
        )
        
        # 应该优雅处理无效输入
        with pytest.raises((ValueError, TypeError)):
            await adapter.adapt_emotion_response(
                emotion_data=sample_emotion_data,
                social_context=invalid_context
            )
    
    @pytest.mark.asyncio
    async def test_malformed_emotion_data(self, adapter, sample_social_context):
        """测试格式错误的情感数据处理"""
        malformed_emotion = EmotionData(
            user_id="",  # 空用户ID
            emotion_vector=None,  # 空情感向量
            confidence=-0.5,  # 无效置信度
            context_markers=None,
            timestamp=None
        )
        
        with pytest.raises((ValueError, TypeError, AttributeError)):
            await adapter.adapt_emotion_response(
                emotion_data=malformed_emotion,
                social_context=sample_social_context
            )
    
    # 集成测试
    @pytest.mark.asyncio
    async def test_full_adaptation_pipeline(self, adapter):
        """测试完整的适配管道"""
        # 创建复杂的测试场景
        complex_emotion = EmotionData(
            user_id="complex_user",
            emotion_vector=EmotionVector(
                joy=0.6, sadness=0.2, anger=0.3, fear=0.1,
                surprise=0.4, disgust=0.05, trust=0.7
            ),
            confidence=0.9,
            context_markers=["workplace", "conflict", "deadline"],
            timestamp=datetime.now()
        )
        
        complex_context = SocialContext(
            scenario=SocialScenario.INTERPERSONAL_CONFLICT,
            participants=["complex_user", "colleague_001", "manager_002"],
            formality_level=0.7,
            power_dynamics={
                "manager_002": 0.9,
                "complex_user": 0.5,
                "colleague_001": 0.5
            },
            cultural_context="western_individualistic",
            time_pressure=0.8,
            relationship_matrix={
                ("complex_user", "colleague_001"): 0.3,  # 紧张关系
                ("complex_user", "manager_002"): 0.6,    # 一般关系
                ("colleague_001", "manager_002"): 0.8    # 良好关系
            }
        )
        
        result = await adapter.adapt_emotion_response(
            emotion_data=complex_emotion,
            social_context=complex_context
        )
        
        # 验证复杂场景下的适配结果
        assert result is not None
        assert result.confidence_score > 0.5
        assert len(result.applied_rules) > 2  # 应该应用多个规则
        assert result.adaptation_rationale is not None
        assert len(result.adaptation_rationale) > 50  # 应该有详细的解释
        
        # 在冲突场景中，愤怒应该被控制
        assert result.adapted_emotion_vector.anger <= complex_emotion.emotion_vector.anger
        
        # 应该考虑权力动态和关系质量
        applied_rule_types = [rule.rule_type for rule in result.applied_rules]
        assert any(rule_type in ["hierarchy_respect", "conflict_mediation", "professional_restraint"] 
                  for rule_type in applied_rule_types)
    
    # 监控和日志测试
    def test_adaptation_logging(self, adapter, sample_emotion_data, sample_social_context):
        """测试适配过程的日志记录"""
        with patch('logging.Logger.info') as mock_log:
            asyncio.run(adapter.adapt_emotion_response(
                emotion_data=sample_emotion_data,
                social_context=sample_social_context
            ))
            
            # 应该有日志记录
            assert mock_log.called
    
    def test_metrics_collection(self, adapter):
        """测试指标收集"""
        # 检查适配器是否有指标收集功能
        assert hasattr(adapter, 'get_adaptation_metrics') or hasattr(adapter, 'metrics')
        
        if hasattr(adapter, 'get_adaptation_metrics'):
            metrics = adapter.get_adaptation_metrics()
            assert isinstance(metrics, dict)
            assert 'total_adaptations' in metrics or 'adaptation_count' in metrics

    # 规则验证测试
    def test_rule_validation(self, adapter):
        """测试规则验证功能"""
        # 获取所有规则
        all_rules = adapter.adaptation_rules
        
        for rule in all_rules:
            # 验证规则结构
            assert hasattr(rule, 'rule_id')
            assert hasattr(rule, 'rule_type')
            assert hasattr(rule, 'priority')
            assert hasattr(rule, 'applicable_scenarios')
            
            # 验证优先级范围
            assert 0 <= rule.priority <= 10
            
            # 验证适用场景
            assert len(rule.applicable_scenarios) > 0
            for scenario in rule.applicable_scenarios:
                assert isinstance(scenario, SocialScenario)
    
    # 适配质量测试
    @pytest.mark.asyncio
    async def test_adaptation_quality_metrics(self, adapter, sample_emotion_data, sample_social_context):
        """测试适配质量指标"""
        result = await adapter.adapt_emotion_response(
            emotion_data=sample_emotion_data,
            social_context=sample_social_context
        )
        
        # 检查适配质量
        assert result.confidence_score > 0
        assert result.confidence_score <= 1.0
        
        # 适配后的情感向量应该是有效的
        adapted_vector = result.adapted_emotion_vector
        total_emotion = (adapted_vector.joy + adapted_vector.sadness + 
                        adapted_vector.anger + adapted_vector.fear + 
                        adapted_vector.surprise + adapted_vector.disgust + 
                        adapted_vector.trust)
        assert total_emotion > 0  # 至少应该有一些情感强度
        
        # 情感强度不应该超过合理范围
        for emotion_value in [adapted_vector.joy, adapted_vector.sadness, 
                             adapted_vector.anger, adapted_vector.fear,
                             adapted_vector.surprise, adapted_vector.disgust, 
                             adapted_vector.trust]:
            assert 0 <= emotion_value <= 1.0
