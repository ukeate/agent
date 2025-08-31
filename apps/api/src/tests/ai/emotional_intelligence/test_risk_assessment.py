"""
情感健康风险评估系统测试
"""
import pytest
from datetime import datetime, timedelta
import numpy as np

from ....ai.emotional_intelligence.risk_assessment import RiskAssessmentEngine
from ....ai.emotional_intelligence.models import RiskAssessment, RiskLevel, RiskFactor
from ....ai.emotion_modeling.models import EmotionState, PersonalityProfile


class TestRiskAssessmentEngine:
    """风险评估引擎测试类"""
    
    @pytest.fixture
    def risk_engine(self):
        """创建风险评估引擎实例"""
        return RiskAssessmentEngine()
    
    @pytest.fixture
    def sample_emotions_stable(self):
        """创建稳定的情感历史数据"""
        emotions = []
        base_time = datetime.now()
        
        for i in range(20):
            emotion = EmotionState(
                user_id='test_user',
                emotion='neutral',
                intensity=0.5 + np.random.normal(0, 0.1),
                valence=0.1 + np.random.normal(0, 0.2),
                arousal=0.4 + np.random.normal(0, 0.1),
                timestamp=base_time - timedelta(hours=i)
            )
            emotions.append(emotion)
        
        return emotions
    
    @pytest.fixture
    def sample_emotions_depressive(self):
        """创建抑郁倾向的情感历史数据"""
        emotions = []
        base_time = datetime.now()
        
        for i in range(20):
            # 模拟抑郁症状：低效价、低唤醒度、高强度负面情感
            emotion = EmotionState(
                user_id='test_user',
                emotion='sadness' if i % 3 == 0 else 'depression',
                intensity=0.7 + np.random.normal(0, 0.1),
                valence=-0.6 + np.random.normal(0, 0.2),
                arousal=0.2 + np.random.normal(0, 0.1),
                timestamp=base_time - timedelta(hours=i),
                triggers=['work_stress', 'loneliness'] if i % 2 == 0 else ['fatigue']
            )
            emotions.append(emotion)
        
        return emotions
    
    @pytest.fixture
    def sample_emotions_anxious(self):
        """创建焦虑倾向的情感历史数据"""
        emotions = []
        base_time = datetime.now()
        
        for i in range(15):
            # 模拟焦虑症状：负面效价、高唤醒度
            emotion = EmotionState(
                user_id='test_user',
                emotion='anxiety' if i % 2 == 0 else 'worry',
                intensity=0.8 + np.random.normal(0, 0.05),
                valence=-0.4 + np.random.normal(0, 0.3),
                arousal=0.8 + np.random.normal(0, 0.1),
                timestamp=base_time - timedelta(hours=i)
            )
            emotions.append(emotion)
        
        return emotions
    
    @pytest.fixture
    def sample_personality_stable(self):
        """创建稳定的个性画像"""
        return PersonalityProfile(
            user_id='test_user',
            emotional_traits={
                'extraversion': 0.6,
                'neuroticism': 0.3,
                'agreeableness': 0.7,
                'conscientiousness': 0.8,
                'openness': 0.5
            },
            emotion_volatility=0.3,
            recovery_rate=0.7
        )
    
    @pytest.fixture
    def sample_personality_unstable(self):
        """创建不稳定的个性画像"""
        return PersonalityProfile(
            user_id='test_user',
            emotional_traits={
                'extraversion': 0.2,
                'neuroticism': 0.8,
                'agreeableness': 0.4,
                'conscientiousness': 0.3,
                'openness': 0.4
            },
            emotion_volatility=0.8,
            recovery_rate=0.2
        )
    
    @pytest.mark.asyncio
    async def test_assess_stable_emotions_low_risk(
        self, 
        risk_engine, 
        sample_emotions_stable, 
        sample_personality_stable
    ):
        """测试稳定情感的低风险评估"""
        assessment = await risk_engine.assess_comprehensive_risk(
            user_id='test_user',
            emotion_history=sample_emotions_stable,
            personality_profile=sample_personality_stable
        )
        
        assert isinstance(assessment, RiskAssessment)
        assert assessment.user_id == 'test_user'
        assert assessment.risk_level in [RiskLevel.LOW.value, RiskLevel.MEDIUM.value]
        assert assessment.risk_score < 0.6
        assert not assessment.alert_triggered
        assert assessment.prediction_confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_assess_depressive_emotions_high_risk(
        self,
        risk_engine,
        sample_emotions_depressive,
        sample_personality_unstable
    ):
        """测试抑郁情感的高风险评估"""
        assessment = await risk_engine.assess_comprehensive_risk(
            user_id='test_user',
            emotion_history=sample_emotions_depressive,
            personality_profile=sample_personality_unstable
        )
        
        assert assessment.risk_level in [RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]
        assert assessment.risk_score > 0.4
        
        # 检查是否识别出抑郁风险因子
        depression_factors = [
            factor for factor in assessment.risk_factors
            if factor.factor_type == 'depression_indicators'
        ]
        assert len(depression_factors) > 0
        assert depression_factors[0].score > 0.3
    
    @pytest.mark.asyncio
    async def test_assess_anxious_emotions(
        self,
        risk_engine,
        sample_emotions_anxious
    ):
        """测试焦虑情感评估"""
        assessment = await risk_engine.assess_comprehensive_risk(
            user_id='test_user',
            emotion_history=sample_emotions_anxious
        )
        
        # 检查是否识别出焦虑风险因子
        anxiety_factors = [
            factor for factor in assessment.risk_factors
            if factor.factor_type == 'anxiety_indicators'
        ]
        
        if anxiety_factors:  # 如果检测到焦虑因子
            assert anxiety_factors[0].score > 0.2
        
        assert assessment.risk_score >= 0.0
        assert assessment.risk_level in [level.value for level in RiskLevel]
    
    @pytest.mark.asyncio
    async def test_predict_crisis_probability_stable(
        self,
        risk_engine,
        sample_emotions_stable
    ):
        """测试稳定情感的危机概率预测"""
        probability, details = await risk_engine.predict_crisis_probability(
            emotion_history=sample_emotions_stable,
            time_horizon=timedelta(hours=24)
        )
        
        assert 0.0 <= probability <= 1.0
        assert probability < 0.5  # 稳定情感应该有较低危机概率
        assert isinstance(details, dict)
        assert 'prediction_method' in details
        assert 'time_horizon_hours' in details
        assert details['time_horizon_hours'] == 24
    
    @pytest.mark.asyncio
    async def test_predict_crisis_probability_deteriorating(
        self,
        risk_engine,
        sample_emotions_depressive
    ):
        """测试恶化情感的危机概率预测"""
        # 创建恶化趋势的情感数据
        deteriorating_emotions = []
        base_time = datetime.now()
        
        for i in range(10):
            # 情感状态逐渐恶化
            deterioration_factor = i / 10.0
            emotion = EmotionState(
                user_id='test_user',
                emotion='despair' if i > 7 else 'sadness',
                intensity=0.5 + deterioration_factor * 0.4,
                valence=-0.3 - deterioration_factor * 0.6,
                arousal=0.4 - deterioration_factor * 0.3,
                timestamp=base_time - timedelta(hours=10-i)
            )
            deteriorating_emotions.append(emotion)
        
        probability, details = await risk_engine.predict_crisis_probability(
            emotion_history=deteriorating_emotions,
            time_horizon=timedelta(hours=24)
        )
        
        assert probability > 0.3  # 恶化趋势应该有较高危机概率
        assert 'negative_trend' in details
        assert 'deterioration_rate' in details
    
    @pytest.mark.asyncio
    async def test_analyze_risk_trends_insufficient_data(self, risk_engine):
        """测试数据不足的风险趋势分析"""
        # 只有1个评估记录
        assessments = [
            RiskAssessment(
                user_id='test_user',
                risk_level=RiskLevel.LOW.value,
                risk_score=0.3,
                timestamp=datetime.now()
            )
        ]
        
        trends = await risk_engine.analyze_risk_trends(
            user_id='test_user',
            assessments=assessments
        )
        
        assert trends['trend'] == 'insufficient_data'
        assert trends['confidence'] == 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_risk_trends_improving(self, risk_engine):
        """测试改善中的风险趋势分析"""
        # 创建改善趋势的评估记录
        assessments = []
        base_time = datetime.now()
        
        # 从高风险到低风险的趋势
        risk_scores = [0.8, 0.7, 0.5, 0.4, 0.2]
        
        for i, score in enumerate(risk_scores):
            assessment = RiskAssessment(
                user_id='test_user',
                risk_level=RiskLevel.HIGH.value if score > 0.6 else 
                          RiskLevel.MEDIUM.value if score > 0.4 else RiskLevel.LOW.value,
                risk_score=score,
                timestamp=base_time - timedelta(days=len(risk_scores)-i-1)
            )
            assessments.append(assessment)
        
        trends = await risk_engine.analyze_risk_trends(
            user_id='test_user',
            assessments=assessments
        )
        
        assert trends['trend'] == 'improving'
        assert trends['confidence'] > 0.0
        assert trends['current_risk'] == 0.2
        assert 'risk_range' in trends
    
    @pytest.mark.asyncio
    async def test_analyze_risk_trends_deteriorating(self, risk_engine):
        """测试恶化中的风险趋势分析"""
        # 创建恶化趋势的评估记录
        assessments = []
        base_time = datetime.now()
        
        # 从低风险到高风险的趋势
        risk_scores = [0.2, 0.3, 0.5, 0.7, 0.8]
        
        for i, score in enumerate(risk_scores):
            assessment = RiskAssessment(
                user_id='test_user',
                risk_level=RiskLevel.LOW.value if score < 0.4 else 
                          RiskLevel.MEDIUM.value if score < 0.7 else RiskLevel.HIGH.value,
                risk_score=score,
                timestamp=base_time - timedelta(days=len(risk_scores)-i-1)
            )
            assessments.append(assessment)
        
        trends = await risk_engine.analyze_risk_trends(
            user_id='test_user',
            assessments=assessments
        )
        
        assert trends['trend'] == 'deteriorating'
        assert trends['volatility'] >= 0.0
        assert 'level_changes' in trends
        assert trends['level_changes']['deteriorations'] > 0
    
    def test_risk_factor_creation(self, risk_engine):
        """测试风险因子创建和权重"""
        # 验证风险权重配置
        assert 'depression_indicators' in risk_engine.risk_weights
        assert 'anxiety_indicators' in risk_engine.risk_weights
        assert sum(risk_engine.risk_weights.values()) == 1.0
        
        # 验证风险阈值配置
        assert risk_engine.risk_thresholds['low'] < risk_engine.risk_thresholds['medium']
        assert risk_engine.risk_thresholds['medium'] < risk_engine.risk_thresholds['high']
        assert risk_engine.risk_thresholds['high'] < risk_engine.risk_thresholds['critical']
    
    @pytest.mark.asyncio
    async def test_depression_indicators_analysis(self, risk_engine):
        """测试抑郁指标分析"""
        # 创建明显的抑郁模式情感数据
        depressive_emotions = []
        base_time = datetime.now()
        
        for i in range(15):
            emotion = EmotionState(
                user_id='test_user',
                emotion='depression',
                intensity=0.8,
                valence=-0.8,  # 强负面效价
                arousal=0.1,   # 低唤醒度
                timestamp=base_time - timedelta(hours=i)
            )
            depressive_emotions.append(emotion)
        
        # 分析抑郁指标
        depression_score = await risk_engine._analyze_depression_indicators(
            depressive_emotions, None
        )
        
        assert depression_score > 0.5  # 明显抑郁模式应该有高分数
        assert depression_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_anxiety_indicators_analysis(self, risk_engine):
        """测试焦虑指标分析"""
        # 创建明显的焦虑模式情感数据
        anxious_emotions = []
        base_time = datetime.now()
        
        for i in range(15):
            emotion = EmotionState(
                user_id='test_user',
                emotion='anxiety',
                intensity=0.8,
                valence=-0.5,
                arousal=0.9,   # 高唤醒度
                timestamp=base_time - timedelta(hours=i)
            )
            anxious_emotions.append(emotion)
        
        # 分析焦虑指标
        anxiety_score = await risk_engine._analyze_anxiety_indicators(
            anxious_emotions, None
        )
        
        assert anxiety_score > 0.4  # 明显焦虑模式应该有较高分数
        assert anxiety_score <= 1.0
    
    def test_emotional_volatility_calculation(self, risk_engine):
        """测试情感波动性计算"""
        # 创建高波动性的情感数据
        volatile_emotions = []
        base_time = datetime.now()
        
        # 交替的极端情感状态
        for i in range(10):
            valence = 0.8 if i % 2 == 0 else -0.8  # 极端交替
            emotion = EmotionState(
                user_id='test_user',
                emotion='happiness' if valence > 0 else 'sadness',
                intensity=0.8,
                valence=valence,
                arousal=0.5,
                timestamp=base_time - timedelta(hours=i)
            )
            volatile_emotions.append(emotion)
        
        volatility = risk_engine._calculate_emotional_volatility(volatile_emotions)
        
        assert volatility > 0.5  # 高波动性
        assert volatility <= 1.0
        
        # 测试稳定情感的低波动性
        stable_emotions = []
        for i in range(10):
            emotion = EmotionState(
                user_id='test_user',
                emotion='neutral',
                intensity=0.5,
                valence=0.1,  # 稳定的小正值
                arousal=0.4,
                timestamp=base_time - timedelta(hours=i)
            )
            stable_emotions.append(emotion)
        
        stable_volatility = risk_engine._calculate_emotional_volatility(stable_emotions)
        assert stable_volatility < volatility  # 稳定情感应该有更低的波动性
    
    @pytest.mark.asyncio
    async def test_context_based_risk_factors(self, risk_engine, sample_emotions_stable):
        """测试基于上下文的风险因子"""
        # 包含高风险上下文信息
        high_risk_context = {
            'sleep_disruption': 0.8,
            'appetite_changes': 0.7,
            'social_withdrawal': 0.9,
            'user_inputs': ['我感觉很绝望', '没有人理解我', '生活毫无意义']
        }
        
        assessment = await risk_engine.assess_comprehensive_risk(
            user_id='test_user',
            emotion_history=sample_emotions_stable,
            context=high_risk_context
        )
        
        # 应该检测到行为变化风险因子
        behavioral_factors = [
            factor for factor in assessment.risk_factors
            if factor.factor_type == 'behavioral_changes'
        ]
        
        if behavioral_factors:
            assert behavioral_factors[0].score > 0.5
        
        # 整体风险应该被上下文因素提升
        assert assessment.risk_score > 0.3
    
    @pytest.mark.asyncio 
    async def test_empty_emotion_history_handling(self, risk_engine):
        """测试空情感历史的处理"""
        assessment = await risk_engine.assess_comprehensive_risk(
            user_id='test_user',
            emotion_history=[],  # 空历史
            personality_profile=None,
            context=None
        )
        
        # 应该返回低风险评估
        assert assessment.risk_level == RiskLevel.LOW.value
        assert assessment.risk_score < 0.5
        assert len(assessment.risk_factors) == 0
        assert assessment.prediction_confidence < 0.5
    
    def test_risk_level_determination(self, risk_engine):
        """测试风险等级确定逻辑"""
        # 测试各个风险等级的阈值
        assert risk_engine._determine_risk_level(0.1) == RiskLevel.LOW.value
        assert risk_engine._determine_risk_level(0.5) == RiskLevel.MEDIUM.value
        assert risk_engine._determine_risk_level(0.7) == RiskLevel.HIGH.value
        assert risk_engine._determine_risk_level(0.96) == RiskLevel.CRITICAL.value
        
        # 边界情况测试
        assert risk_engine._determine_risk_level(0.0) == RiskLevel.LOW.value
        assert risk_engine._determine_risk_level(1.0) == RiskLevel.CRITICAL.value
    
    @pytest.mark.asyncio
    async def test_prediction_confidence_calculation(self, risk_engine, sample_emotions_stable):
        """测试预测置信度计算"""
        # 高质量数据应该产生高置信度
        high_quality_profile = PersonalityProfile(
            user_id='test_user',
            sample_count=100,  # 大量样本
            confidence_score=0.9
        )
        
        assessment = await risk_engine.assess_comprehensive_risk(
            user_id='test_user',
            emotion_history=sample_emotions_stable,
            personality_profile=high_quality_profile
        )
        
        # 置信度应该合理
        assert 0.0 <= assessment.prediction_confidence <= 1.0
        
        # 更多风险因子应该提高置信度
        if len(assessment.risk_factors) > 2:
            assert assessment.prediction_confidence > 0.5