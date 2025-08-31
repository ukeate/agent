"""
情感智能决策引擎测试
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from ....ai.emotional_intelligence.decision_engine import EmotionalDecisionEngine, DecisionStrategy
from ....ai.emotional_intelligence.models import (
    DecisionContext, EmotionalDecision, RiskAssessment, RiskLevel, DecisionType
)


class MockDecisionStrategy(DecisionStrategy):
    """测试用决策策略"""
    
    def __init__(self, name: str, evaluation_score: float = 0.8):
        self._name = name
        self._evaluation_score = evaluation_score
    
    async def evaluate(self, context: DecisionContext) -> float:
        return self._evaluation_score
    
    async def execute(self, context: DecisionContext) -> dict:
        return {'result': f'执行策略: {self._name}'}
    
    @property
    def strategy_name(self) -> str:
        return self._name


class TestEmotionalDecisionEngine:
    """情感智能决策引擎测试类"""
    
    @pytest.fixture
    def decision_engine(self):
        """创建决策引擎实例"""
        engine = EmotionalDecisionEngine()
        
        # 注册测试策略
        engine.register_strategy(MockDecisionStrategy('supportive_strategy', 0.9))
        engine.register_strategy(MockDecisionStrategy('intervention_strategy', 0.7))
        
        return engine
    
    @pytest.fixture
    def sample_context(self):
        """创建示例决策上下文"""
        return DecisionContext(
            user_id='test_user_123',
            session_id='session_456',
            current_emotion_state={
                'emotion': 'sadness',
                'intensity': 0.7,
                'valence': -0.6,
                'arousal': 0.3,
                'dominance': 0.4
            },
            emotion_history=[
                {
                    'emotion': 'anxiety',
                    'intensity': 0.8,
                    'valence': -0.7,
                    'timestamp': (datetime.now() - timedelta(hours=1)).isoformat()
                },
                {
                    'emotion': 'neutral',
                    'intensity': 0.5,
                    'valence': 0.0,
                    'timestamp': (datetime.now() - timedelta(hours=2)).isoformat()
                }
            ],
            user_input='我感觉很沮丧，不知道该怎么办',
            environmental_factors={'stress_level': 0.6}
        )
    
    @pytest.mark.asyncio
    async def test_make_decision_success(self, decision_engine, sample_context):
        """测试成功生成决策"""
        # 执行决策
        decision = await decision_engine.make_decision(sample_context)
        
        # 验证结果
        assert isinstance(decision, EmotionalDecision)
        assert decision.user_id == sample_context.user_id
        assert decision.session_id == sample_context.session_id
        assert decision.chosen_strategy in ['supportive_strategy', 'intervention_strategy']
        assert 0 <= decision.confidence_score <= 1
        assert len(decision.reasoning) > 0
        assert decision.decision_type in [dt.value for dt in DecisionType]
    
    @pytest.mark.asyncio
    async def test_assess_emotional_risk_low_risk(self, decision_engine):
        """测试低风险情感评估"""
        # 创建低风险上下文
        low_risk_context = DecisionContext(
            user_id='test_user_123',
            current_emotion_state={
                'emotion': 'happiness',
                'intensity': 0.8,
                'valence': 0.7,
                'arousal': 0.5,
                'dominance': 0.6
            },
            environmental_factors={'stress_level': 0.2}
        )
        
        # 执行风险评估
        risk_assessment = await decision_engine.assess_emotional_risk(low_risk_context)
        
        # 验证结果
        assert isinstance(risk_assessment, RiskAssessment)
        assert risk_assessment.user_id == 'test_user_123'
        assert risk_assessment.risk_level == RiskLevel.LOW.value
        assert risk_assessment.risk_score < 0.5
        assert not risk_assessment.alert_triggered
    
    @pytest.mark.asyncio
    async def test_assess_emotional_risk_high_risk(self, decision_engine):
        """测试高风险情感评估"""
        # 创建高风险上下文
        high_risk_context = DecisionContext(
            user_id='test_user_456',
            current_emotion_state={
                'emotion': 'depression',
                'intensity': 0.9,
                'valence': -0.9,
                'arousal': 0.1,
                'dominance': 0.2
            },
            emotion_history=[
                {
                    'emotion': 'despair',
                    'intensity': 0.9,
                    'valence': -0.8,
                    'timestamp': (datetime.now() - timedelta(hours=1)).isoformat()
                }
            ] * 10,  # 重复负面情感
            environmental_factors={'stress_level': 0.9, 'social_isolation_score': 0.8}
        )
        
        # 执行风险评估
        risk_assessment = await decision_engine.assess_emotional_risk(high_risk_context)
        
        # 验证结果
        assert risk_assessment.risk_level in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]
        assert risk_assessment.risk_score > 0.6
        assert risk_assessment.alert_triggered
        assert len(risk_assessment.risk_factors) > 0
        assert len(risk_assessment.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_detect_crisis_no_crisis(self, decision_engine):
        """测试无危机情况的检测"""
        # 创建正常上下文
        normal_context = DecisionContext(
            user_id='test_user_123',
            current_emotion_state={
                'emotion': 'neutral',
                'intensity': 0.5,
                'valence': 0.0,
                'arousal': 0.4
            },
            user_input='今天心情还可以，工作有点忙'
        )
        
        # 执行危机检测
        crisis_assessment = await decision_engine.detect_crisis(normal_context)
        
        # 验证结果
        assert not crisis_assessment.crisis_detected
        assert crisis_assessment.severity_level == 'mild'
        assert crisis_assessment.risk_score < 0.6
        assert not crisis_assessment.professional_required
    
    @pytest.mark.asyncio
    async def test_detect_crisis_with_crisis_keywords(self, decision_engine):
        """测试包含危机关键词的检测"""
        # 创建包含危机关键词的上下文
        crisis_context = DecisionContext(
            user_id='test_user_789',
            current_emotion_state={
                'emotion': 'despair',
                'intensity': 0.9,
                'valence': -0.9,
                'arousal': 0.2
            },
            user_input='我真的不想活了，感觉生活毫无意义'
        )
        
        # 执行危机检测
        crisis_assessment = await decision_engine.detect_crisis(crisis_context)
        
        # 验证结果
        print(f"Crisis detected: {crisis_assessment.crisis_detected}")
        print(f"Indicators: {crisis_assessment.indicators}")
        assert crisis_assessment.crisis_detected
        assert crisis_assessment.severity_level in ['severe', 'critical']
        assert crisis_assessment.risk_score > 0.7
        assert crisis_assessment.professional_required
        assert len(crisis_assessment.immediate_actions) > 0
        assert any('language_indicator' in str(ind) for ind in crisis_assessment.indicators)
    
    @pytest.mark.asyncio
    async def test_create_intervention_plan_high_risk(self, decision_engine):
        """测试为高风险用户创建干预计划"""
        # 创建高风险评估
        high_risk_assessment = RiskAssessment(
            user_id='test_user_123',
            risk_level=RiskLevel.HIGH.value,
            risk_score=0.8,
            recommended_actions=['intensive_support', 'professional_referral']
        )
        
        # 创建干预计划
        intervention_plan = await decision_engine.create_intervention_plan(high_risk_assessment)
        
        # 验证结果
        assert intervention_plan.user_id == 'test_user_123'
        assert intervention_plan.intervention_type == 'corrective'
        assert intervention_plan.urgency_level in ['high', 'critical']
        assert len(intervention_plan.strategies) > 0
        assert intervention_plan.primary_strategy is not None
        assert intervention_plan.monitoring_frequency <= timedelta(hours=2)
    
    def test_register_strategy(self, decision_engine):
        """测试策略注册"""
        initial_count = len(decision_engine.strategies)
        
        # 注册新策略
        new_strategy = MockDecisionStrategy('test_strategy', 0.6)
        decision_engine.register_strategy(new_strategy)
        
        # 验证策略已注册
        assert len(decision_engine.strategies) == initial_count + 1
        assert 'test_strategy' in decision_engine.strategies
        assert decision_engine.strategies['test_strategy'] == new_strategy
    
    @pytest.mark.asyncio
    async def test_decision_with_no_strategies(self):
        """测试无策略情况下的决策"""
        # 创建无策略的决策引擎
        engine = EmotionalDecisionEngine()
        
        sample_context = DecisionContext(
            user_id='test_user',
            current_emotion_state={'emotion': 'neutral', 'intensity': 0.5},
            user_input='测试输入'
        )
        
        # 执行决策
        decision = await engine.make_decision(sample_context)
        
        # 验证使用默认策略
        assert decision.chosen_strategy == 'default_supportive'
        assert decision.confidence_score == 0.5
    
    @pytest.mark.asyncio
    async def test_risk_assessment_with_empty_history(self, decision_engine):
        """测试空情感历史的风险评估"""
        empty_context = DecisionContext(
            user_id='test_user',
            current_emotion_state={
                'emotion': 'neutral',
                'intensity': 0.5,
                'valence': 0.0
            },
            emotion_history=[]  # 空历史
        )
        
        # 执行风险评估
        risk_assessment = await decision_engine.assess_emotional_risk(empty_context)
        
        # 验证结果
        assert risk_assessment.risk_level == RiskLevel.LOW.value
        assert len(risk_assessment.risk_factors) == 0 or all(
            factor.score <= 0.3 for factor in risk_assessment.risk_factors
        )
    
    @pytest.mark.asyncio
    async def test_decision_reasoning_generation(self, decision_engine, sample_context):
        """测试决策推理生成"""
        decision = await decision_engine.make_decision(sample_context)
        
        # 验证推理内容
        assert len(decision.reasoning) > 0
        reasoning_text = ' '.join(decision.reasoning)
        assert '风险评估' in reasoning_text or 'risk' in reasoning_text.lower()
        assert '情感状态' in reasoning_text or 'emotion' in reasoning_text.lower()
        assert decision.chosen_strategy in reasoning_text
    
    @pytest.mark.asyncio
    async def test_decision_confidence_calculation(self, decision_engine):
        """测试决策置信度计算"""
        # 创建高置信度上下文（清晰的情感状态和历史）
        high_confidence_context = DecisionContext(
            user_id='test_user',
            current_emotion_state={
                'emotion': 'anxiety',
                'intensity': 0.8,
                'valence': -0.7,
                'confidence': 0.9
            },
            emotion_history=[{
                'emotion': 'anxiety',
                'intensity': 0.8,
                'timestamp': datetime.now().isoformat()
            }] * 5,
            personality_profile={'extraversion': 0.3, 'neuroticism': 0.8}
        )
        
        decision = await decision_engine.make_decision(high_confidence_context)
        
        # 高质量数据应该产生较高置信度
        assert decision.confidence_score >= 0.6
    
    @pytest.mark.asyncio
    async def test_error_handling_in_decision_making(self, decision_engine):
        """测试决策制定中的错误处理"""
        # 创建可能导致错误的上下文
        invalid_context = DecisionContext(
            user_id='test_user',
            current_emotion_state=None,  # 无效的情感状态
            user_input=''  # 空输入
        )
        
        # 执行决策，应该不抛出异常
        try:
            decision = await decision_engine.make_decision(invalid_context)
            assert decision is not None  # 应该有某种形式的响应
        except Exception as e:
            # 如果抛出异常，应该是可预期的类型
            assert isinstance(e, (ValueError, TypeError))


@pytest.mark.asyncio
async def test_decision_engine_integration():
    """决策引擎集成测试"""
    engine = EmotionalDecisionEngine()
    
    # 注册多个策略
    strategies = [
        MockDecisionStrategy('strategy_1', 0.9),
        MockDecisionStrategy('strategy_2', 0.7),
        MockDecisionStrategy('strategy_3', 0.5)
    ]
    
    for strategy in strategies:
        engine.register_strategy(strategy)
    
    # 创建复杂的决策上下文
    complex_context = DecisionContext(
        user_id='integration_test_user',
        session_id='integration_session',
        current_emotion_state={
            'emotion': 'mixed_emotions',
            'intensity': 0.6,
            'valence': -0.3,
            'arousal': 0.7
        },
        emotion_history=[
            {
                'emotion': 'happiness',
                'intensity': 0.8,
                'valence': 0.7,
                'timestamp': (datetime.now() - timedelta(hours=3)).isoformat()
            },
            {
                'emotion': 'sadness',
                'intensity': 0.6,
                'valence': -0.5,
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ],
        user_input='我的心情很复杂，有开心也有难过',
        environmental_factors={
            'stress_level': 0.5,
            'social_support': 0.7,
            'life_events': ['work_promotion', 'family_conflict']
        }
    )
    
    # 执行完整的决策流程
    decision = await engine.make_decision(complex_context)
    risk_assessment = await engine.assess_emotional_risk(complex_context)
    crisis_assessment = await engine.detect_crisis(complex_context)
    
    # 验证集成结果
    assert decision.user_id == complex_context.user_id
    assert decision.chosen_strategy == 'strategy_1'  # 应该选择最高分策略
    
    assert risk_assessment.user_id == complex_context.user_id
    assert risk_assessment.risk_level in [level.value for level in RiskLevel]
    
    assert crisis_assessment.user_id == complex_context.user_id
    assert not crisis_assessment.crisis_detected  # 非危机情况
    
    # 验证决策历史记录
    assert len(engine.decision_history) == 1
    assert engine.decision_history[0].decision_id == decision.decision_id