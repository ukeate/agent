"""
奖励信号生成器测试套件

测试反馈信号处理、奖励计算和质量评估功能。
覆盖多种奖励计算策略和上下文增强场景。
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.reward_generator import (
    RewardSignalGenerator,
    FeedbackNormalizer,
    TimeDecayCalculator,
    ContextBoostCalculator,
    RewardStrategy,
    FeedbackSignal,
    RewardConfig
)
from src.services.feedback_collector import CollectedEvent, EventPriority
from models.schemas.feedback import FeedbackType


class TestFeedbackNormalizer:
    """反馈标准化器测试"""
    
    @pytest.fixture
    def normalizer(self):
        return FeedbackNormalizer()
    
    def test_normalize_rating(self, normalizer):
        """测试评分标准化"""
        # 5分制评分
        assert normalizer.normalize_rating(1, 1, 5) == 0.0
        assert normalizer.normalize_rating(3, 1, 5) == 0.5
        assert normalizer.normalize_rating(5, 1, 5) == 1.0
        
        # 10分制评分
        assert normalizer.normalize_rating(7, 1, 10) == pytest.approx(0.667, abs=0.01)
    
    def test_normalize_boolean_feedback(self, normalizer):
        """测试布尔反馈标准化"""
        assert normalizer.normalize_boolean_feedback(True) == 1.0
        assert normalizer.normalize_boolean_feedback(False) == 0.0
        assert normalizer.normalize_boolean_feedback(1) == 1.0
        assert normalizer.normalize_boolean_feedback(0) == 0.0
    
    def test_normalize_time_based_feedback(self, normalizer):
        """测试时间类反馈标准化"""
        # 停留时间 - 使用对数标准化
        assert normalizer.normalize_time_based_feedback(1) == pytest.approx(0.0, abs=0.01)
        assert normalizer.normalize_time_based_feedback(30) > 0.5
        assert normalizer.normalize_time_based_feedback(300) > 0.8
        
        # 边界情况
        assert normalizer.normalize_time_based_feedback(0) == 0.0
        assert normalizer.normalize_time_based_feedback(-1) == 0.0
    
    def test_normalize_percentage_feedback(self, normalizer):
        """测试百分比反馈标准化"""
        assert normalizer.normalize_percentage_feedback(0) == 0.0
        assert normalizer.normalize_percentage_feedback(50) == 0.5
        assert normalizer.normalize_percentage_feedback(100) == 1.0
        
        # 超出范围处理
        assert normalizer.normalize_percentage_feedback(-10) == 0.0
        assert normalizer.normalize_percentage_feedback(150) == 1.0
    
    def test_normalize_text_sentiment(self, normalizer):
        """测试文本情感标准化"""
        # 正面评论
        positive_text = "这个功能太棒了！非常喜欢！"
        score = normalizer.normalize_text_sentiment(positive_text)
        assert 0.6 <= score <= 1.0
        
        # 负面评论  
        negative_text = "这个功能很糟糕，完全不好用"
        score = normalizer.normalize_text_sentiment(negative_text)
        assert 0.0 <= score <= 0.4
        
        # 中性评论
        neutral_text = "这个功能还可以"
        score = normalizer.normalize_text_sentiment(neutral_text)
        assert 0.4 <= score <= 0.6


class TestTimeDecayCalculator:
    """时间衰减计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        return TimeDecayCalculator(half_life_hours=24.0)
    
    def test_calculate_decay_factor_immediate(self, calculator):
        """测试即时反馈无衰减"""
        now = datetime.now()
        factor = calculator.calculate_decay_factor(now)
        assert factor == pytest.approx(1.0, abs=0.001)
    
    def test_calculate_decay_factor_half_life(self, calculator):
        """测试半衰期衰减"""
        now = datetime.now()
        half_life_ago = now - timedelta(hours=24)
        
        factor = calculator.calculate_decay_factor(half_life_ago)
        assert factor == pytest.approx(0.5, abs=0.01)
    
    def test_calculate_decay_factor_old_feedback(self, calculator):
        """测试旧反馈大幅衰减"""
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
        factor = calculator.calculate_decay_factor(week_ago)
        assert factor < 0.01  # 应该大幅衰减
    
    def test_different_half_life(self):
        """测试不同半衰期设置"""
        fast_decay = TimeDecayCalculator(half_life_hours=1.0)
        slow_decay = TimeDecayCalculator(half_life_hours=168.0)  # 一周
        
        now = datetime.now()
        six_hours_ago = now - timedelta(hours=6)
        
        fast_factor = fast_decay.calculate_decay_factor(six_hours_ago)
        slow_factor = slow_decay.calculate_decay_factor(six_hours_ago)
        
        assert fast_factor < slow_factor  # 快速衰减应该更小


class TestContextBoostCalculator:
    """上下文增强计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        return ContextBoostCalculator()
    
    def test_calculate_user_engagement_boost(self, calculator):
        """测试用户参与度增强"""
        # 高参与度用户
        high_engagement_context = {
            "user_session_count": 50,
            "user_feedback_count": 20,
            "user_avg_session_duration": 300
        }
        boost = calculator.calculate_user_engagement_boost(high_engagement_context)
        assert boost > 1.0  # 应该有正向增强
        
        # 低参与度用户
        low_engagement_context = {
            "user_session_count": 1,
            "user_feedback_count": 0,
            "user_avg_session_duration": 30
        }
        boost = calculator.calculate_user_engagement_boost(low_engagement_context)
        assert boost <= 1.0  # 应该没有或负向增强
    
    def test_calculate_item_popularity_boost(self, calculator):
        """测试物品热度增强"""
        # 热门物品
        popular_context = {
            "item_view_count": 1000,
            "item_interaction_count": 200,
            "item_positive_feedback_ratio": 0.8
        }
        boost = calculator.calculate_item_popularity_boost(popular_context)
        assert boost > 1.0
        
        # 冷门物品
        unpopular_context = {
            "item_view_count": 10,
            "item_interaction_count": 1,
            "item_positive_feedback_ratio": 0.2
        }
        boost = calculator.calculate_item_popularity_boost(unpopular_context)
        assert boost <= 1.0
    
    def test_calculate_temporal_boost(self, calculator):
        """测试时间相关增强"""
        # 工作时间
        work_hour_context = {
            "hour_of_day": 14,  # 下午2点
            "day_of_week": 2,   # 周二
            "is_holiday": False
        }
        work_boost = calculator.calculate_temporal_boost(work_hour_context)
        
        # 休息时间
        rest_hour_context = {
            "hour_of_day": 2,   # 凌晨2点
            "day_of_week": 6,   # 周六
            "is_holiday": True
        }
        rest_boost = calculator.calculate_temporal_boost(rest_hour_context)
        
        # 工作时间的反馈应该有不同的增强效果
        assert work_boost != rest_boost
    
    def test_calculate_session_boost(self, calculator):
        """测试会话相关增强"""
        # 活跃会话
        active_context = {
            "session_duration": 600,     # 10分钟
            "session_page_views": 15,
            "session_interactions": 8
        }
        active_boost = calculator.calculate_session_boost(active_context)
        assert active_boost > 1.0
        
        # 短暂会话
        brief_context = {
            "session_duration": 30,      # 30秒
            "session_page_views": 2,
            "session_interactions": 0
        }
        brief_boost = calculator.calculate_session_boost(brief_context)
        assert brief_boost <= 1.0


class TestRewardSignalGenerator:
    """奖励信号生成器测试"""
    
    @pytest.fixture
    def generator(self):
        return RewardSignalGenerator()
    
    @pytest.fixture
    def sample_click_event(self):
        return CollectedEvent(
            event_id="test_click",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={
                "page": "home",
                "position": 3,
                "user_session_count": 10
            },
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
    
    @pytest.fixture
    def sample_rating_event(self):
        return CollectedEvent(
            event_id="test_rating", 
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=4,
            context={
                "survey_id": "survey_1",
                "item_category": "feature"
            },
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
    
    @pytest.mark.asyncio
    async def test_process_single_event_click(self, generator, sample_click_event):
        """测试处理单个点击事件"""
        result = await generator.process_single_event(sample_click_event)
        
        assert isinstance(result, ProcessedFeedback)
        assert result.event_id == "test_click"
        assert result.user_id == "user_1"
        assert result.item_id == "item_1"
        assert result.feedback_type == FeedbackType.CLICK
        assert 0.0 <= result.normalized_value <= 1.0
        assert result.confidence_score > 0
        assert result.quality_score > 0
    
    @pytest.mark.asyncio
    async def test_process_single_event_rating(self, generator, sample_rating_event):
        """测试处理单个评分事件"""
        result = await generator.process_single_event(sample_rating_event)
        
        assert isinstance(result, ProcessedFeedback)
        assert result.feedback_type == FeedbackType.RATING
        assert result.normalized_value == 0.75  # (4-1)/(5-1) = 0.75
        assert result.confidence_score > 0.5  # 显式反馈置信度应该较高
    
    @pytest.mark.asyncio
    async def test_process_batch_events(self, generator, sample_click_event, sample_rating_event):
        """测试批量处理事件"""
        events = [sample_click_event, sample_rating_event]
        results = await generator.process_batch_events(events)
        
        assert len(results) == 2
        assert all(isinstance(r, ProcessedFeedback) for r in results)
        
        # 验证不同类型事件的处理
        click_result = next(r for r in results if r.feedback_type == FeedbackType.CLICK)
        rating_result = next(r for r in results if r.feedback_type == FeedbackType.RATING)
        
        assert click_result.confidence_score < rating_result.confidence_score  # 显式反馈置信度更高
    
    @pytest.mark.asyncio
    async def test_generate_reward_signal_weighted_average(self, generator):
        """测试加权平均策略生成奖励信号"""
        # 创建多个处理后的反馈
        processed_feedbacks = [
            ProcessedFeedback(
                event_id="event_1",
                user_id="user_1",
                item_id="item_1",
                feedback_type=FeedbackType.CLICK,
                normalized_value=0.8,
                confidence_score=0.6,
                quality_score=0.7,
                context_boost=1.2,
                time_decay_factor=0.9,
                timestamp=datetime.now()
            ),
            ProcessedFeedback(
                event_id="event_2",
                user_id="user_1", 
                item_id="item_1",
                feedback_type=FeedbackType.RATING,
                normalized_value=0.6,
                confidence_score=0.9,
                quality_score=0.8,
                context_boost=1.1,
                time_decay_factor=0.95,
                timestamp=datetime.now()
            )
        ]
        
        reward = await generator.generate_reward_signal(
            processed_feedbacks,
            strategy=RewardCalculationStrategy.WEIGHTED_AVERAGE
        )
        
        assert isinstance(reward, RewardSignal)
        assert 0.0 <= reward.value <= 1.0
        assert reward.confidence > 0
        assert len(reward.contributing_feedbacks) == 2
        assert reward.strategy == RewardCalculationStrategy.WEIGHTED_AVERAGE
    
    @pytest.mark.asyncio
    async def test_generate_reward_signal_time_decay(self, generator):
        """测试时间衰减策略"""
        # 创建不同时间的反馈
        old_feedback = ProcessedFeedback(
            event_id="old_event",
            user_id="user_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            normalized_value=0.9,
            confidence_score=0.8,
            quality_score=0.8,
            context_boost=1.0,
            time_decay_factor=0.1,  # 很旧的反馈
            timestamp=datetime.now() - timedelta(days=7)
        )
        
        recent_feedback = ProcessedFeedback(
            event_id="recent_event",
            user_id="user_1",
            item_id="item_1", 
            feedback_type=FeedbackType.RATING,
            normalized_value=0.5,
            confidence_score=0.9,
            quality_score=0.8,
            context_boost=1.0,
            time_decay_factor=1.0,  # 新鲜反馈
            timestamp=datetime.now()
        )
        
        reward = await generator.generate_reward_signal(
            [old_feedback, recent_feedback],
            strategy=RewardCalculationStrategy.TIME_DECAY
        )
        
        # 时间衰减策略应该更偏向新反馈
        assert isinstance(reward, RewardSignal)
        # 由于新反馈权重更大，最终奖励应该更接近0.5而不是0.9
        assert reward.value < 0.7
    
    @pytest.mark.asyncio
    async def test_generate_reward_signal_quality_adjusted(self, generator):
        """测试质量调整策略"""
        high_quality_feedback = ProcessedFeedback(
            event_id="high_quality",
            user_id="user_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            normalized_value=0.8,
            confidence_score=0.95,
            quality_score=0.9,  # 高质量
            context_boost=1.0,
            time_decay_factor=1.0,
            timestamp=datetime.now()
        )
        
        low_quality_feedback = ProcessedFeedback(
            event_id="low_quality",
            user_id="user_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            normalized_value=0.2,
            confidence_score=0.4,
            quality_score=0.3,  # 低质量
            context_boost=1.0,
            time_decay_factor=1.0,
            timestamp=datetime.now()
        )
        
        reward = await generator.generate_reward_signal(
            [high_quality_feedback, low_quality_feedback],
            strategy=RewardCalculationStrategy.QUALITY_ADJUSTED
        )
        
        # 质量调整策略应该更偏向高质量反馈
        assert reward.value > 0.5  # 应该更接近高质量反馈的值
    
    @pytest.mark.asyncio
    async def test_generate_reward_signal_context_boosted(self, generator):
        """测试上下文增强策略"""
        boosted_feedback = ProcessedFeedback(
            event_id="boosted",
            user_id="user_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            normalized_value=0.6,
            confidence_score=0.8,
            quality_score=0.8,
            context_boost=1.5,  # 高上下文增强
            time_decay_factor=1.0,
            timestamp=datetime.now()
        )
        
        normal_feedback = ProcessedFeedback(
            event_id="normal",
            user_id="user_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            normalized_value=0.6,
            confidence_score=0.7,
            quality_score=0.7,
            context_boost=1.0,  # 无增强
            time_decay_factor=1.0,
            timestamp=datetime.now()
        )
        
        reward = await generator.generate_reward_signal(
            [boosted_feedback, normal_feedback],
            strategy=RewardCalculationStrategy.CONTEXT_BOOSTED
        )
        
        # 有上下文增强的反馈应该有更大权重
        assert reward.value >= 0.6
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_score(self, generator):
        """测试置信度计算"""
        # 高置信度场景（显式反馈 + 活跃用户）
        high_conf_event = CollectedEvent(
            event_id="high_conf",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=5,
            context={
                "user_feedback_count": 50,
                "user_session_count": 100
            },
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        
        confidence = generator._calculate_confidence_score(high_conf_event)
        assert confidence > 0.7  # 应该是高置信度
        
        # 低置信度场景（隐式反馈 + 新用户）
        low_conf_event = CollectedEvent(
            event_id="low_conf",
            user_id="user_1",
            session_id="session_1", 
            item_id="item_1",
            feedback_type=FeedbackType.HOVER,
            raw_value=1.5,
            context={
                "user_feedback_count": 0,
                "user_session_count": 1
            },
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
        
        confidence = generator._calculate_confidence_score(low_conf_event)
        assert confidence < 0.5  # 应该是低置信度
    
    @pytest.mark.asyncio
    async def test_calculate_quality_score(self, generator):
        """测试质量分数计算"""
        # 高质量反馈（合理数值 + 丰富上下文）
        high_quality_event = CollectedEvent(
            event_id="high_quality",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.DWELL_TIME,
            raw_value=120,  # 2分钟停留时间，合理
            context={
                "page_type": "article",
                "content_length": 1000,
                "user_reading_speed": "normal",
                "scroll_depth": 85
            },
            timestamp=datetime.now(),
            priority=EventPriority.MEDIUM
        )
        
        quality = generator._calculate_quality_score(high_quality_event)
        assert quality > 0.6
        
        # 低质量反馈（异常数值 + 缺少上下文）
        low_quality_event = CollectedEvent(
            event_id="low_quality",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.DWELL_TIME,
            raw_value=1,  # 1秒停留，可能是误触
            context={},  # 缺少上下文
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
        
        quality = generator._calculate_quality_score(low_quality_event)
        assert quality < 0.5
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_event(self, generator):
        """测试异常事件处理"""
        # 创建无效事件
        invalid_event = CollectedEvent(
            event_id="invalid",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1", 
            feedback_type=FeedbackType.RATING,
            raw_value="invalid_rating",  # 字符串而非数值
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        
        # 应该优雅处理异常
        result = await generator.process_single_event(invalid_event)
        assert result is None or result.quality_score == 0.0
    
    @pytest.mark.asyncio
    async def test_empty_feedback_list(self, generator):
        """测试空反馈列表处理"""
        reward = await generator.generate_reward_signal([])
        assert reward is None
    
    @pytest.mark.asyncio
    async def test_performance_batch_processing(self, generator):
        """测试批量处理性能"""
        # 创建大量事件
        events = []
        for i in range(100):
            event = CollectedEvent(
                event_id=f"event_{i}",
                user_id=f"user_{i%10}",  # 10个用户
                session_id=f"session_{i}",
                item_id=f"item_{i%20}",  # 20个物品
                feedback_type=FeedbackType.CLICK,
                raw_value=True,
                context={"batch_test": True},
                timestamp=datetime.now(),
                priority=EventPriority.LOW
            )
            events.append(event)
        
        import time
        start_time = time.time()
        results = await generator.process_batch_events(events)
        end_time = time.time()
        
        # 验证结果
        assert len(results) <= len(events)  # 可能有无效事件被过滤
        
        # 验证性能（应该在合理时间内完成）
        duration = end_time - start_time
        assert duration < 5.0  # 5秒内完成100个事件处理


class TestIntegrationScenarios:
    """集成场景测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        generator = RewardSignalGenerator()
        
        # 模拟用户会话中的多个反馈事件
        session_events = [
            # 用户进入页面，短暂停留
            CollectedEvent(
                event_id="view_1",
                user_id="user_123",
                session_id="session_456", 
                item_id="article_789",
                feedback_type=FeedbackType.VIEW,
                raw_value=True,
                context={"entry_point": "home_page"},
                timestamp=datetime.now() - timedelta(minutes=5),
                priority=EventPriority.LOW
            ),
            # 用户点击内容
            CollectedEvent(
                event_id="click_1",
                user_id="user_123",
                session_id="session_456",
                item_id="article_789", 
                feedback_type=FeedbackType.CLICK,
                raw_value=True,
                context={"click_position": "title"},
                timestamp=datetime.now() - timedelta(minutes=4),
                priority=EventPriority.LOW
            ),
            # 用户滚动阅读
            CollectedEvent(
                event_id="scroll_1",
                user_id="user_123",
                session_id="session_456",
                item_id="article_789",
                feedback_type=FeedbackType.SCROLL_DEPTH,
                raw_value=75.0,
                context={"content_type": "article"},
                timestamp=datetime.now() - timedelta(minutes=3),
                priority=EventPriority.LOW
            ),
            # 用户停留阅读
            CollectedEvent(
                event_id="dwell_1", 
                user_id="user_123",
                session_id="session_456",
                item_id="article_789",
                feedback_type=FeedbackType.DWELL_TIME,
                raw_value=180,  # 3分钟
                context={"reading_completion": 0.8},
                timestamp=datetime.now() - timedelta(minutes=2),
                priority=EventPriority.MEDIUM
            ),
            # 用户给出明确评分
            CollectedEvent(
                event_id="rating_1",
                user_id="user_123",
                session_id="session_456",
                item_id="article_789",
                feedback_type=FeedbackType.RATING,
                raw_value=4,
                context={"rating_context": "content_quality"},
                timestamp=datetime.now() - timedelta(minutes=1),
                priority=EventPriority.HIGH
            )
        ]
        
        # 处理所有事件
        processed_feedbacks = await generator.process_batch_events(session_events)
        assert len(processed_feedbacks) > 0
        
        # 生成奖励信号
        reward_signal = await generator.generate_reward_signal(
            processed_feedbacks,
            strategy=RewardCalculationStrategy.WEIGHTED_AVERAGE
        )
        
        # 验证最终奖励信号
        assert isinstance(reward_signal, RewardSignal)
        assert 0.0 <= reward_signal.value <= 1.0
        assert reward_signal.confidence > 0
        assert len(reward_signal.contributing_feedbacks) > 0
        
        # 验证用户行为模式（应该显示积极参与）
        assert reward_signal.value > 0.6  # 用户行为显示积极参与
        
        # 验证不同反馈类型的贡献
        feedback_types = {fb.feedback_type for fb in processed_feedbacks}
        assert FeedbackType.RATING in feedback_types  # 应该包含明确评分
        assert len(feedback_types) > 1  # 应该包含多种反馈类型
    
    @pytest.mark.asyncio
    async def test_multi_user_comparison(self):
        """测试多用户反馈对比"""
        generator = RewardSignalGenerator()
        
        # 活跃用户的反馈
        active_user_event = CollectedEvent(
            event_id="active_user",
            user_id="active_user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=4,
            context={
                "user_session_count": 100,  # 高活跃度
                "user_feedback_count": 50,
                "user_avg_rating": 4.2
            },
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        
        # 新用户的反馈
        new_user_event = CollectedEvent(
            event_id="new_user",
            user_id="new_user_1", 
            session_id="session_2",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=4,  # 相同评分
            context={
                "user_session_count": 1,   # 低活跃度
                "user_feedback_count": 1,
                "user_avg_rating": 4.0
            },
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        
        # 处理两个事件
        active_processed = await generator.process_single_event(active_user_event)
        new_processed = await generator.process_single_event(new_user_event)
        
        # 活跃用户的反馈应该有更高的置信度
        assert active_processed.confidence_score > new_processed.confidence_score
        
        # 但标准化值应该相同（同样的评分）
        assert active_processed.normalized_value == new_processed.normalized_value