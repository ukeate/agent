"""
Task 4文化背景感知模块完整单元测试套件
测试CulturalContextAnalyzer的所有功能和边界条件
"""

import pytest
import asyncio
from typing import Dict, List, Any
from unittest.mock import Mock, patch
from ai.emotion_modeling.cultural_context_analyzer import (
    CulturalContextAnalyzer,
    CulturalDimension,
    CommunicationStyle,
    CulturalProfile,
    CulturalAdaptation
)
from ai.emotion_modeling.models import EmotionVector, SocialContext

@pytest.fixture
def analyzer():
    """创建文化背景分析器实例"""
    return CulturalContextAnalyzer()

@pytest.fixture
def sample_emotion_vector():
    """创建测试情感向量"""
    return EmotionVector(
        emotions={
            "happiness": 0.6,
            "excitement": 0.3,
            "confidence": 0.5,
            "assertiveness": 0.4
        },
        intensity=0.7,
        confidence=0.8,
        context={"interaction_type": "business"}
    )

@pytest.fixture
def sample_participants():
    """创建测试参与者"""
    return [
        {
            "user_id": "user_1",
            "cultural_indicators": {
                "language": "en",
                "region": "US",
                "communication_style": "direct",
                "dimensions": {
                    "power_distance": 0.3,
                    "individualism": 0.8
                }
            }
        },
        {
            "user_id": "user_2",
            "cultural_indicators": {
                "language": "zh",
                "region": "CN",
                "communication_style": "high_context",
                "dimensions": {
                    "power_distance": 0.7,
                    "individualism": 0.2
                }
            }
        }
    ]

class TestCulturalContextAnalyzer:
    """文化背景分析器基础功能测试"""
    
    def test_initialization(self, analyzer):
        """测试初始化"""
        assert analyzer is not None
        assert len(analyzer.cultural_profiles) >= 4
        assert len(analyzer.adaptation_rules) > 0
        assert analyzer.learning_enabled is True
        
        # 验证预设文化档案
        expected_cultures = [
            "western_individualistic",
            "east_asian_collectivistic", 
            "latin_expressive",
            "northern_european_reserved"
        ]
        for culture_id in expected_cultures:
            assert culture_id in analyzer.cultural_profiles
    
    def test_cultural_profiles_structure(self, analyzer):
        """测试文化档案结构完整性"""
        for culture_id, profile in analyzer.cultural_profiles.items():
            assert isinstance(profile, CulturalProfile)
            assert profile.culture_id == culture_id
            assert profile.name is not None
            assert len(profile.dimensions) == 6  # 6个Hofstede维度
            assert profile.communication_style in CommunicationStyle
            assert len(profile.emotional_expression_norms) > 0
            assert isinstance(profile.social_hierarchies, list)
            assert isinstance(profile.taboo_topics, list)
            assert isinstance(profile.greeting_customs, list)
    
    @pytest.mark.asyncio
    async def test_analyze_cultural_context_basic(self, analyzer, sample_participants):
        """测试基础文化背景分析"""
        cultures, confidence = await analyzer.analyze_cultural_context(
            sample_participants, {"context": "business_meeting"}
        )
        
        assert len(cultures) == 2
        assert 0.0 <= confidence <= 1.0
        assert all(isinstance(culture, CulturalProfile) for culture in cultures)
    
    @pytest.mark.asyncio 
    async def test_analyze_cultural_context_empty_participants(self, analyzer):
        """测试空参与者列表"""
        cultures, confidence = await analyzer.analyze_cultural_context(
            [], {"context": "test"}
        )
        
        assert len(cultures) == 0
        assert confidence == 0.5
    
    def test_calculate_cultural_match_score(self, analyzer):
        """测试文化匹配分数计算"""
        indicators = {
            "communication_style": "direct",
            "dimensions": {
                "power_distance": 0.3,
                "individualism": 0.8
            },
            "language": "en"
        }
        
        profile = analyzer.cultural_profiles["western_individualistic"]
        score = analyzer._calculate_cultural_match_score(indicators, profile)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # 应该是较好的匹配

class TestCulturalAdaptation:
    """文化适配功能测试"""
    
    @pytest.mark.asyncio
    async def test_adapt_for_cultural_context_basic(
        self, analyzer, sample_emotion_vector, sample_participants
    ):
        """测试基础文化适配"""
        cultures, _ = await analyzer.analyze_cultural_context(
            sample_participants, {"context": "business"}
        )
        
        adaptation = await analyzer.adapt_for_cultural_context(
            sample_emotion_vector,
            cultures,
            {"interaction_type": "formal_meeting"}
        )
        
        assert isinstance(adaptation, CulturalAdaptation)
        assert adaptation.original_emotion == sample_emotion_vector
        assert adaptation.adapted_emotion is not None
        assert adaptation.cultural_context in cultures
        assert len(adaptation.adaptation_strategies) > 0
        assert 0.0 <= adaptation.cultural_sensitivity_score <= 1.0
        assert isinstance(adaptation.potential_misunderstandings, list)
        assert adaptation.recommended_approach is not None
    
    @pytest.mark.asyncio
    async def test_adapt_for_empty_cultures(self, analyzer, sample_emotion_vector):
        """测试空文化列表适配"""
        adaptation = await analyzer.adapt_for_cultural_context(
            sample_emotion_vector, [], {"context": "test"}
        )
        
        assert adaptation.original_emotion == sample_emotion_vector
        assert adaptation.adapted_emotion == sample_emotion_vector
        assert adaptation.cultural_context.culture_id == "western_individualistic"
        assert "use_default_approach" in adaptation.adaptation_strategies
        assert adaptation.cultural_sensitivity_score == 0.5
        assert adaptation.recommended_approach == "neutral_professional"
    
    def test_apply_cultural_adaptation_emotional_restraint(
        self, analyzer, sample_emotion_vector
    ):
        """测试情感克制文化适配"""
        # 创建高情感克制文化档案
        restrained_culture = CulturalProfile(
            culture_id="test_restrained",
            name="测试克制文化",
            dimensions={
                CulturalDimension.POWER_DISTANCE: 0.8,
                CulturalDimension.INDIVIDUALISM: 0.2,
                CulturalDimension.MASCULINITY: 0.5,
                CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.6,
                CulturalDimension.LONG_TERM_ORIENTATION: 0.7,
                CulturalDimension.INDULGENCE: 0.3
            },
            communication_style=CommunicationStyle.HIGH_CONTEXT,
            emotional_expression_norms={"emotional_restraint": 0.9},
            social_hierarchies=["age-based"],
            taboo_topics=["emotions"],
            greeting_customs=["formal_bow"],
            conflict_resolution_style="avoidance",
            time_orientation="flexible"
        )
        
        # 创建包含强烈情感的向量
        strong_emotion = EmotionVector(
            emotions={"anger": 0.8, "frustration": 0.7, "excitement": 0.6},
            intensity=0.9,
            confidence=0.8,
            context={}
        )
        
        adapted = analyzer._apply_cultural_adaptation(strong_emotion, restrained_culture)
        
        # 验证强烈情感被降低
        assert adapted.emotions["anger"] < strong_emotion.emotions["anger"]
        assert adapted.emotions["frustration"] < strong_emotion.emotions["frustration"]
        assert adapted.emotions["excitement"] < strong_emotion.emotions["excitement"]
    
    def test_apply_cultural_adaptation_emotional_warmth(
        self, analyzer, sample_emotion_vector
    ):
        """测试情感温暖文化适配"""
        warm_culture = analyzer.cultural_profiles["latin_expressive"]
        
        adapted = analyzer._apply_cultural_adaptation(sample_emotion_vector, warm_culture)
        
        # 验证积极情感可能被增强
        assert adapted.emotions["happiness"] >= sample_emotion_vector.emotions["happiness"]
    
    def test_generate_adaptation_strategies_high_context(self, analyzer):
        """测试高语境文化适配策略"""
        profile = analyzer.cultural_profiles["east_asian_collectivistic"]
        emotion_vector = EmotionVector(
            emotions={"neutral": 1.0},
            intensity=0.5,
            confidence=0.8,
            context={}
        )
        
        strategies = analyzer._generate_adaptation_strategies(
            profile, emotion_vector, {"context": "business"}
        )
        
        assert len(strategies) > 0
        high_context_strategies = [
            "pay_attention_to_nonverbal_cues",
            "read_between_the_lines",
            "build_relationship_first",
            "use_indirect_communication"
        ]
        assert any(strategy in strategies for strategy in high_context_strategies)
    
    def test_generate_adaptation_strategies_low_context(self, analyzer):
        """测试低语境文化适配策略"""
        profile = analyzer.cultural_profiles["western_individualistic"]
        emotion_vector = EmotionVector(
            emotions={"confidence": 1.0},
            intensity=0.7,
            confidence=0.8,
            context={}
        )
        
        strategies = analyzer._generate_adaptation_strategies(
            profile, emotion_vector, {"context": "business"}
        )
        
        low_context_strategies = [
            "be_explicit_and_direct",
            "focus_on_facts_and_details",
            "avoid_ambiguity",
            "get_to_the_point"
        ]
        assert any(strategy in strategies for strategy in low_context_strategies)

class TestCulturalMisunderstandingDetection:
    """文化误解检测测试"""
    
    def test_identify_potential_misunderstandings_communication_style(self, analyzer):
        """测试沟通风格冲突检测"""
        primary = analyzer.cultural_profiles["western_individualistic"]  # 直接
        others = [analyzer.cultural_profiles["east_asian_collectivistic"]]  # 间接
        
        emotion_vector = EmotionVector(
            emotions={"assertiveness": 0.8},
            intensity=0.7,
            confidence=0.8,
            context={}
        )
        
        misunderstandings = analyzer._identify_potential_misunderstandings(
            primary, emotion_vector, others
        )
        
        assert len(misunderstandings) > 0
        assert "directness_may_seem_rude" in misunderstandings
    
    def test_identify_potential_misunderstandings_power_distance(self, analyzer):
        """测试权力距离冲突检测"""
        high_pd = analyzer.cultural_profiles["east_asian_collectivistic"]
        low_pd = analyzer.cultural_profiles["northern_european_reserved"]
        
        emotion_vector = EmotionVector(
            emotions={"neutral": 1.0},
            intensity=0.5,
            confidence=0.8,
            context={}
        )
        
        misunderstandings = analyzer._identify_potential_misunderstandings(
            high_pd, emotion_vector, [low_pd]
        )
        
        assert "hierarchy_expectations_differ" in misunderstandings
    
    def test_identify_emotional_expression_misunderstanding(self, analyzer):
        """测试情感表达冲突检测"""
        restrained_culture = analyzer.cultural_profiles["northern_european_reserved"]
        
        # 创建高强度情感向量
        intense_emotion = EmotionVector(
            emotions={"anger": 0.9, "excitement": 0.8},
            intensity=0.9,
            confidence=0.8,
            context={}
        )
        
        # 模拟高情感克制规范
        original_norms = restrained_culture.emotional_expression_norms
        restrained_culture.emotional_expression_norms = {"emotional_restraint": 0.9}
        
        misunderstandings = analyzer._identify_potential_misunderstandings(
            restrained_culture, intense_emotion, []
        )
        
        # 恢复原始规范
        restrained_culture.emotional_expression_norms = original_norms
        
        assert "emotional_expression_may_be_inappropriate" in misunderstandings

class TestRecommendationSystem:
    """推荐系统测试"""
    
    def test_recommend_approach_high_context_high_power(self, analyzer):
        """测试高语境高权力距离推荐"""
        profile = analyzer.cultural_profiles["east_asian_collectivistic"]
        emotion_vector = EmotionVector(
            emotions={"respectful": 0.8},
            intensity=0.6,
            confidence=0.8,
            context={}
        )
        
        approach = analyzer._recommend_approach(profile, emotion_vector, {})
        assert approach == "formal_respectful_indirect"
    
    def test_recommend_approach_direct_individualistic(self, analyzer):
        """测试直接个人主义推荐"""
        profile = analyzer.cultural_profiles["western_individualistic"]
        emotion_vector = EmotionVector(
            emotions={"confidence": 0.8},
            intensity=0.7,
            confidence=0.8,
            context={}
        )
        
        approach = analyzer._recommend_approach(profile, emotion_vector, {})
        assert approach == "direct_task_focused"
    
    def test_calculate_cultural_sensitivity_score(self, analyzer, sample_emotion_vector):
        """测试文化敏感度分数计算"""
        profile = analyzer.cultural_profiles["east_asian_collectivistic"]
        
        # 创建适配后的情感向量
        adapted = analyzer._apply_cultural_adaptation(sample_emotion_vector, profile)
        
        score = analyzer._calculate_cultural_sensitivity(
            sample_emotion_vector, adapted, profile
        )
        
        assert 0.0 <= score <= 1.0
        
        # 如果有适配，分数应该高于基础分数
        if sample_emotion_vector != adapted:
            assert score > 0.5

class TestLearningSystem:
    """学习系统测试"""
    
    @pytest.mark.asyncio
    async def test_learn_cultural_patterns_basic(self, analyzer):
        """测试基础文化模式学习"""
        interaction_data = {
            "culture_combination": "western_east_asian",
            "strategies_used": ["build_relationship_first", "be_respectful"]
        }
        
        outcome_feedback = {"success_score": 0.8}
        
        await analyzer.learn_cultural_patterns(interaction_data, outcome_feedback)
        
        assert "western_east_asian" in analyzer.cross_cultural_patterns
        pattern = analyzer.cross_cultural_patterns["western_east_asian"]
        assert len(pattern["success_rates"]) == 1
        assert pattern["success_rates"][0] == 0.8
        assert len(pattern["effective_strategies"]) > 0
    
    @pytest.mark.asyncio
    async def test_learn_cultural_patterns_multiple_interactions(self, analyzer):
        """测试多次交互学习"""
        culture_combo = "test_combination"
        
        # 多次学习
        for i in range(3):
            await analyzer.learn_cultural_patterns(
                {
                    "culture_combination": culture_combo,
                    "strategies_used": ["strategy_1", "strategy_2"]
                },
                {"success_score": 0.7 + i * 0.1}
            )
        
        pattern = analyzer.cross_cultural_patterns[culture_combo]
        assert len(pattern["success_rates"]) == 3
        assert pattern["effective_strategies"]["strategy_1"] == 3
        assert pattern["effective_strategies"]["strategy_2"] == 3
    
    @pytest.mark.asyncio
    async def test_learn_cultural_patterns_disabled(self, analyzer):
        """测试禁用学习功能"""
        analyzer.learning_enabled = False
        
        await analyzer.learn_cultural_patterns(
            {"culture_combination": "test"},
            {"success_score": 0.8}
        )
        
        assert len(analyzer.cross_cultural_patterns) == 0
    
    @pytest.mark.asyncio
    async def test_get_cross_cultural_insights(self, analyzer):
        """测试跨文化洞察获取"""
        # 添加一些学习数据
        test_combinations = [
            ("combo_1", [0.8, 0.9, 0.7]),
            ("combo_2", [0.6, 0.5, 0.7]),
            ("combo_3", [0.9, 0.8, 0.9])
        ]
        
        for combo, scores in test_combinations:
            for score in scores:
                await analyzer.learn_cultural_patterns(
                    {"culture_combination": combo, "strategies_used": []},
                    {"success_score": score}
                )
        
        insights = await analyzer.get_cross_cultural_insights()
        
        assert insights["total_patterns"] == 3
        assert len(insights["most_successful_combinations"]) == 3
        
        # 验证排序（combo_3应该排在最前面）
        top_combo = insights["most_successful_combinations"][0]
        assert top_combo["combination"] == "combo_3"
        assert top_combo["success_rate"] > 0.8

class TestUtilityMethods:
    """工具方法测试"""
    
    def test_calculate_emotion_similarity_identical(self, analyzer):
        """测试相同情感向量的相似度"""
        emotion = EmotionVector(
            emotions={"happiness": 0.6, "confidence": 0.4},
            intensity=0.7,
            confidence=0.8,
            context={}
        )
        
        similarity = analyzer._calculate_emotion_similarity(emotion, emotion)
        assert similarity == 1.0
    
    def test_calculate_emotion_similarity_different(self, analyzer):
        """测试不同情感向量的相似度"""
        emotion1 = EmotionVector(
            emotions={"happiness": 0.8, "confidence": 0.2},
            intensity=0.7,
            confidence=0.8,
            context={}
        )
        
        emotion2 = EmotionVector(
            emotions={"happiness": 0.2, "confidence": 0.8},
            intensity=0.7,
            confidence=0.8,
            context={}
        )
        
        similarity = analyzer._calculate_emotion_similarity(emotion1, emotion2)
        assert 0.0 <= similarity < 1.0
    
    def test_calculate_emotion_similarity_no_common_emotions(self, analyzer):
        """测试无共同情感的相似度"""
        emotion1 = EmotionVector(
            emotions={"happiness": 0.8},
            intensity=0.7,
            confidence=0.8,
            context={}
        )
        
        emotion2 = EmotionVector(
            emotions={"sadness": 0.6},
            intensity=0.5,
            confidence=0.7,
            context={}
        )
        
        similarity = analyzer._calculate_emotion_similarity(emotion1, emotion2)
        assert similarity == 0.0
    
    def test_get_cultural_profiles(self, analyzer):
        """测试获取文化档案"""
        profiles = analyzer.get_cultural_profiles()
        
        assert len(profiles) >= 4
        assert isinstance(profiles, dict)
        
        # 验证返回的是副本
        original_count = len(analyzer.cultural_profiles)
        profiles.clear()
        assert len(analyzer.cultural_profiles) == original_count
    
    def test_add_cultural_profile(self, analyzer):
        """测试添加新文化档案"""
        new_profile = CulturalProfile(
            culture_id="test_culture",
            name="测试文化",
            dimensions={
                CulturalDimension.POWER_DISTANCE: 0.5,
                CulturalDimension.INDIVIDUALISM: 0.5,
                CulturalDimension.MASCULINITY: 0.5,
                CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.5,
                CulturalDimension.LONG_TERM_ORIENTATION: 0.5,
                CulturalDimension.INDULGENCE: 0.5
            },
            communication_style=CommunicationStyle.DIRECT,
            emotional_expression_norms={"balanced": 0.5},
            social_hierarchies=["merit-based"],
            taboo_topics=["test_topic"],
            greeting_customs=["handshake"],
            conflict_resolution_style="discussion",
            time_orientation="punctual"
        )
        
        original_count = len(analyzer.cultural_profiles)
        analyzer.add_cultural_profile(new_profile)
        
        assert len(analyzer.cultural_profiles) == original_count + 1
        assert "test_culture" in analyzer.cultural_profiles
        assert analyzer.cultural_profiles["test_culture"] == new_profile

class TestErrorHandling:
    """错误处理测试"""
    
    @pytest.mark.asyncio
    async def test_adapt_for_cultural_context_exception(self, analyzer, sample_emotion_vector):
        """测试文化适配异常处理"""
        with patch.object(analyzer, '_apply_cultural_adaptation', side_effect=Exception("测试异常")):
            adaptation = await analyzer.adapt_for_cultural_context(
                sample_emotion_vector,
                [analyzer.cultural_profiles["western_individualistic"]],
                {}
            )
            
            assert adaptation.original_emotion == sample_emotion_vector
            assert adaptation.adapted_emotion == sample_emotion_vector
            assert adaptation.cultural_sensitivity_score == 0.3
            assert "adaptation_failed: 测试异常" in adaptation.adaptation_strategies
            assert "technical_error" in adaptation.potential_misunderstandings
            assert adaptation.recommended_approach == "cautious_neutral"
    
    def test_calculate_cultural_match_score_no_indicators(self, analyzer):
        """测试无指标的匹配分数计算"""
        profile = analyzer.cultural_profiles["western_individualistic"]
        score = analyzer._calculate_cultural_match_score({}, profile)
        
        assert score == 0.5  # 默认分数
    
    def test_calculate_cultural_match_score_unknown_language(self, analyzer):
        """测试未知语言的匹配分数"""
        indicators = {
            "language": "unknown_language",
            "communication_style": "direct"
        }
        
        profile = analyzer.cultural_profiles["western_individualistic"]
        score = analyzer._calculate_cultural_match_score(indicators, profile)
        
        assert 0.0 <= score <= 1.0

class TestPerformanceAndOptimization:
    """性能和优化测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_many_participants_performance(self, analyzer):
        """测试大量参与者的性能"""
        import time
        
        # 创建100个参与者
        participants = []
        for i in range(100):
            participants.append({
                "user_id": f"user_{i}",
                "cultural_indicators": {
                    "language": "en" if i % 2 == 0 else "zh",
                    "communication_style": "direct" if i % 3 == 0 else "indirect",
                    "dimensions": {"power_distance": i / 100.0}
                }
            })
        
        start_time = time.time()
        cultures, confidence = await analyzer.analyze_cultural_context(
            participants, {"context": "large_meeting"}
        )
        end_time = time.time()
        
        # 验证结果
        assert len(cultures) == 100
        assert 0.0 <= confidence <= 1.0
        
        # 验证性能（应该在合理时间内完成）
        processing_time = end_time - start_time
        assert processing_time < 5.0  # 5秒内完成
    
    def test_cultural_profiles_memory_efficiency(self, analyzer):
        """测试文化档案内存效率"""
        import sys
        
        # 测试添加大量文化档案不会导致内存泄露
        original_size = sys.getsizeof(analyzer.cultural_profiles)
        
        # 添加100个测试档案
        for i in range(100):
            test_profile = CulturalProfile(
                culture_id=f"test_{i}",
                name=f"测试文化{i}",
                dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.5,
                    CulturalDimension.INDIVIDUALISM: 0.5,
                    CulturalDimension.MASCULINITY: 0.5,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.5,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.5,
                    CulturalDimension.INDULGENCE: 0.5
                },
                communication_style=CommunicationStyle.DIRECT,
                emotional_expression_norms={"test": 0.5},
                social_hierarchies=["test"],
                taboo_topics=["test"],
                greeting_customs=["test"],
                conflict_resolution_style="test",
                time_orientation="test"
            )
            analyzer.add_cultural_profile(test_profile)
        
        new_size = sys.getsizeof(analyzer.cultural_profiles)
        
        # 验证内存使用合理增长
        assert new_size > original_size
        assert len(analyzer.cultural_profiles) >= 104  # 原有4个+新增100个

class TestIntegrationScenarios:
    """集成场景测试"""
    
    @pytest.mark.asyncio
    async def test_complete_cultural_adaptation_workflow(
        self, analyzer, sample_participants
    ):
        """测试完整的文化适配工作流"""
        # 1. 分析文化背景
        cultures, confidence = await analyzer.analyze_cultural_context(
            sample_participants, {"context": "business_negotiation"}
        )
        
        assert len(cultures) > 0
        assert confidence > 0.0
        
        # 2. 创建需要适配的情感
        emotion = EmotionVector(
            emotions={"assertiveness": 0.8, "confidence": 0.7, "impatience": 0.3},
            intensity=0.8,
            confidence=0.9,
            context={"urgency": "high"}
        )
        
        # 3. 执行文化适配
        adaptation = await analyzer.adapt_for_cultural_context(
            emotion, cultures, {"meeting_type": "formal"}
        )
        
        # 4. 验证适配结果
        assert adaptation.original_emotion == emotion
        assert adaptation.adapted_emotion is not None
        assert len(adaptation.adaptation_strategies) > 0
        assert adaptation.cultural_sensitivity_score > 0.0
        
        # 5. 学习交互结果
        await analyzer.learn_cultural_patterns(
            {
                "culture_combination": f"{cultures[0].culture_id}_{cultures[1].culture_id if len(cultures) > 1 else 'single'}",
                "strategies_used": adaptation.adaptation_strategies[:3]
            },
            {"success_score": 0.8}
        )
        
        # 6. 获取学习洞察
        insights = await analyzer.get_cross_cultural_insights()
        assert insights["total_patterns"] > 0
    
    @pytest.mark.asyncio
    async def test_cross_cultural_conflict_scenario(self, analyzer):
        """测试跨文化冲突场景"""
        # 创建文化冲突参与者
        conflict_participants = [
            {
                "user_id": "direct_user",
                "cultural_indicators": {
                    "communication_style": "direct",
                    "language": "en",
                    "dimensions": {"power_distance": 0.2, "individualism": 0.9}
                }
            },
            {
                "user_id": "indirect_user", 
                "cultural_indicators": {
                    "communication_style": "high_context",
                    "language": "zh",
                    "dimensions": {"power_distance": 0.8, "individualism": 0.1}
                }
            }
        ]
        
        # 分析文化背景
        cultures, _ = await analyzer.analyze_cultural_context(
            conflict_participants, {"context": "conflict_resolution"}
        )
        
        # 创建冲突情感
        conflict_emotion = EmotionVector(
            emotions={"frustration": 0.7, "assertiveness": 0.8, "tension": 0.6},
            intensity=0.8,
            confidence=0.7,
            context={"conflict": True}
        )
        
        # 执行适配
        adaptation = await analyzer.adapt_for_cultural_context(
            conflict_emotion, cultures, {"situation": "disagreement"}
        )
        
        # 验证冲突识别和适配
        assert len(adaptation.potential_misunderstandings) > 0
        conflict_strategies = [
            "pay_attention_to_nonverbal_cues",
            "build_relationship_first", 
            "indirect_communication",
            "maintain_face_for_all"
        ]
        assert any(strategy in adaptation.adaptation_strategies for strategy in conflict_strategies)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
