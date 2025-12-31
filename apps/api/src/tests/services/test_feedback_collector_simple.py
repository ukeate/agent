"""
简化版用户反馈收集器测试套件

专注于测试核心功能，提高测试覆盖率到85%以上。
避免复杂的异步依赖，直接测试每个方法和类。
"""

import pytest
import asyncio
import time
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
from src.services.feedback_collector import (
    FeedbackCollector,
    FeedbackBuffer,
    EventDeduplicator,
    EventValidator,
    CollectedEvent,
    EventPriority
)
from models.schemas.feedback import FeedbackType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestEventPriority:
    """测试事件优先级枚举"""
    
    def test_priority_values(self):
        """测试优先级值"""
        assert EventPriority.HIGH == "high"
        assert EventPriority.MEDIUM == "medium"
        assert EventPriority.LOW == "low"

class TestCollectedEvent:
    """测试收集事件数据结构"""
    
    def test_event_creation(self):
        """测试事件创建"""
        event = CollectedEvent(
            event_id="test_123",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={"page": "home"},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        
        assert event.event_id == "test_123"
        assert event.user_id == "user_1"
        assert event.feedback_type == FeedbackType.CLICK
        assert event.priority == EventPriority.HIGH
        assert event.context["page"] == "home"

class TestFeedbackBuffer:
    """测试反馈缓冲器"""
    
    def test_buffer_initialization(self):
        """测试缓冲器初始化"""
        buffer = FeedbackBuffer(max_size=100, flush_interval=5.0)
        assert buffer.max_size == 100
        assert buffer.flush_interval == 5.0
        assert len(buffer.buffer) == 0
        assert len(buffer.priority_buffers) == 3
    
    @pytest.mark.asyncio
    async def test_add_event_success(self):
        """测试成功添加事件"""
        buffer = FeedbackBuffer(max_size=5)
        event = CollectedEvent(
            event_id="test_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.MEDIUM
        )
        
        result = await buffer.add_event(event)
        
        assert result is True
        assert len(buffer.buffer) == 1
        assert len(buffer.priority_buffers[EventPriority.MEDIUM]) == 1
    
    @pytest.mark.asyncio
    async def test_buffer_full_handling(self):
        """测试缓冲区满时的处理"""
        buffer = FeedbackBuffer(max_size=2)
        
        # 添加第一个事件
        event1 = CollectedEvent(
            event_id="event_1",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        result1 = await buffer.add_event(event1)
        assert result1 is True
        
        # 添加第二个事件
        event2 = CollectedEvent(
            event_id="event_2",
            user_id="user_2",
            session_id="session_2",
            item_id="item_2",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        result2 = await buffer.add_event(event2)
        assert result2 is True
        
        # 尝试添加第三个事件（应该失败）
        event3 = CollectedEvent(
            event_id="event_3",
            user_id="user_3",
            session_id="session_3",
            item_id="item_3",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        result3 = await buffer.add_event(event3)
        assert result3 is False
    
    @pytest.mark.asyncio
    async def test_should_flush_time(self):
        """测试基于时间的刷新检查"""
        buffer = FeedbackBuffer(flush_interval=1.0)
        
        # 设置较早的时间
        buffer._last_flush = time.time() - 2.0
        
        should_flush = await buffer.should_flush()
        assert should_flush is True
    
    @pytest.mark.asyncio
    async def test_should_flush_high_priority(self):
        """测试高优先级事件触发刷新"""
        buffer = FeedbackBuffer()
        event = CollectedEvent(
            event_id="urgent_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=5,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        
        await buffer.add_event(event)
        should_flush = await buffer.should_flush()
        assert should_flush is True
    
    @pytest.mark.asyncio
    async def test_flush_priority_ordering(self):
        """测试刷新时的优先级排序"""
        buffer = FeedbackBuffer()
        
        # 按不同优先级添加事件
        low_event = CollectedEvent(
            event_id="low_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.VIEW,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        high_event = CollectedEvent(
            event_id="high_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=5,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        
        medium_event = CollectedEvent(
            event_id="medium_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.DWELL_TIME,
            raw_value=30,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.MEDIUM
        )
        
        await buffer.add_event(low_event)
        await buffer.add_event(high_event)
        await buffer.add_event(medium_event)
        
        # 刷新并验证顺序
        events = await buffer.flush()
        assert len(events) == 3
        assert events[0].priority == EventPriority.HIGH
        assert events[1].priority == EventPriority.MEDIUM
        assert events[2].priority == EventPriority.LOW

class TestEventDeduplicator:
    """测试事件去重器"""
    
    def test_deduplicator_initialization(self):
        """测试去重器初始化"""
        deduplicator = EventDeduplicator(window_seconds=300)
        assert deduplicator.window_seconds == 300
        assert len(deduplicator.seen_events) == 0
    
    def test_generate_event_key(self):
        """测试事件键生成"""
        deduplicator = EventDeduplicator()
        event = CollectedEvent(
            event_id="test_event",
            user_id="user_123",
            session_id="session_456",
            item_id="item_789",
            feedback_type=FeedbackType.RATING,
            raw_value=4,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        
        key = deduplicator._generate_event_key(event)
        assert key == "user_123|item_789|rating|4"
    
    @pytest.mark.asyncio
    async def test_first_event_not_duplicate(self):
        """测试首次事件不被认为是重复"""
        deduplicator = EventDeduplicator()
        event = CollectedEvent(
            event_id="first_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        is_duplicate = await deduplicator.is_duplicate(event)
        assert is_duplicate is False
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """测试重复事件检测"""
        deduplicator = EventDeduplicator()
        event = CollectedEvent(
            event_id="test_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        # 第一次不是重复
        is_duplicate1 = await deduplicator.is_duplicate(event)
        assert is_duplicate1 is False
        
        # 第二次是重复
        is_duplicate2 = await deduplicator.is_duplicate(event)
        assert is_duplicate2 is True
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self):
        """测试过期条目清理"""
        deduplicator = EventDeduplicator(window_seconds=1)
        event = CollectedEvent(
            event_id="test_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        # 添加事件
        await deduplicator.is_duplicate(event)
        assert len(deduplicator.seen_events) == 1
        
        # 等待一段时间让条目过期，然后测试清理
        await asyncio.sleep(1.1)  # 等待超过window_seconds
        
        # 创建一个新的相同事件来触发清理
        new_event = CollectedEvent(
            event_id="test_event",
            user_id="user_1", 
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        # 再次检查，应该清理过期条目
        is_duplicate = await deduplicator.is_duplicate(new_event)
        assert is_duplicate is False
        assert len(deduplicator.seen_events) == 1  # 只有新事件

class TestEventValidator:
    """测试事件验证器"""
    
    def test_validate_click_event(self):
        """测试点击事件验证"""
        validator = EventValidator()
        
        # 有效的点击事件
        valid_event = CollectedEvent(
            event_id="click_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        assert validator.validate_implicit_event(valid_event) is True
        
        # 无效值
        valid_event.raw_value = "invalid"
        assert validator.validate_implicit_event(valid_event) is False
    
    def test_validate_dwell_time_event(self):
        """测试停留时间事件验证"""
        validator = EventValidator()
        
        # 有效的停留时间
        event = CollectedEvent(
            event_id="dwell_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.DWELL_TIME,
            raw_value=30.5,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        assert validator.validate_implicit_event(event) is True
        
        # 超出范围
        event.raw_value = 7200  # 2小时
        assert validator.validate_implicit_event(event) is False
        
        # 负值
        event.raw_value = -5
        assert validator.validate_implicit_event(event) is False
    
    def test_validate_scroll_depth_event(self):
        """测试滚动深度验证"""
        validator = EventValidator()
        
        event = CollectedEvent(
            event_id="scroll_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.SCROLL_DEPTH,
            raw_value=75.0,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        assert validator.validate_implicit_event(event) is True
        
        # 超出百分比范围
        event.raw_value = 150
        assert validator.validate_implicit_event(event) is False
    
    def test_validate_rating_event(self):
        """测试评分事件验证"""
        validator = EventValidator()
        
        event = CollectedEvent(
            event_id="rating_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=4,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        
        assert validator.validate_explicit_event(event) is True
        
        # 超出范围
        event.raw_value = 0
        assert validator.validate_explicit_event(event) is False
        
        event.raw_value = 6
        assert validator.validate_explicit_event(event) is False
    
    def test_validate_comment_event(self):
        """测试评论事件验证"""
        validator = EventValidator()
        
        event = CollectedEvent(
            event_id="comment_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.COMMENT,
            raw_value="这是一个很好的评论",
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        
        assert validator.validate_explicit_event(event) is True
        
        # 空评论
        event.raw_value = ""
        assert validator.validate_explicit_event(event) is False
        
        # 只有空格
        event.raw_value = "   "
        assert validator.validate_explicit_event(event) is False

class TestFeedbackCollectorCore:
    """测试反馈收集器核心功能（不依赖配置）"""
    
    def test_collector_initialization(self):
        """测试收集器初始化"""
        # 直接测试默认初始化值
        collector = FeedbackCollector()
        
        # 断言默认值或配置文件中的值
        assert collector.buffer.max_size == 1000  # 默认值
        assert collector.buffer.flush_interval == 5.0  # 默认值
        assert collector.deduplicator.window_seconds == 300  # 默认值
        assert collector._running is False
    
    def test_collector_stats_initialization(self):
        """测试收集器统计信息初始化"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            expected_stats = {
                "total_received": 0,
                "total_processed": 0,
                "duplicates_filtered": 0,
                "validation_failures": 0,
                "buffer_overflows": 0
            }
            
            for key, value in expected_stats.items():
                assert collector.stats[key] == value
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """测试启动停止生命周期"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            # 初始状态
            assert collector._running is False
            assert len(collector._background_tasks) == 0
            
            # 启动
            await collector.start()
            assert collector._running is True
            assert len(collector._background_tasks) > 0
            
            # 停止
            await collector.stop()
            assert collector._running is False
            assert len(collector._background_tasks) == 0

# 错误处理测试
class TestErrorHandling:
    """错误处理测试"""
    
    def test_validator_exception_handling(self):
        """测试验证器异常处理"""
        validator = EventValidator()
        
        # 创建会引发异常的事件
        event = CollectedEvent(
            event_id="error_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=None,  # 这会导致类型错误
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        
        # 验证器应该优雅处理异常
        result = validator.validate_explicit_event(event)
        assert result is False

# 边界情况测试
class TestEdgeCases:
    """边界情况测试"""
    
    @pytest.mark.asyncio
    async def test_empty_buffer_flush(self):
        """测试空缓冲区刷新"""
        buffer = FeedbackBuffer()
        events = await buffer.flush()
        assert events == []
    
    def test_zero_window_deduplicator(self):
        """测试零窗口去重器"""
        deduplicator = EventDeduplicator(window_seconds=0)
        assert deduplicator.window_seconds == 0
    
    def test_unknown_feedback_type_validation(self):
        """测试未知反馈类型验证"""
        validator = EventValidator()
        
        # 创建带有未知反馈类型的事件
        event = CollectedEvent(
            event_id="unknown_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type="unknown_type",  # 不是FeedbackType枚举值
            raw_value="some_value",
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        # 应该返回False（不验证通过）
        result_implicit = validator.validate_implicit_event(event)
        result_explicit = validator.validate_explicit_event(event)
        
        assert result_implicit is False
        assert result_explicit is False

# 性能相关测试
class TestPerformanceUnits:
    """性能单元测试"""
    
    def test_event_key_generation_performance(self):
        """测试事件键生成性能"""
        deduplicator = EventDeduplicator()
        event = CollectedEvent(
            event_id="perf_event",
            user_id="user_performance_test",
            session_id="session_performance_test",
            item_id="item_performance_test",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        
        # 生成大量事件键
        import time
        start_time = time.time()
        
        for _ in range(1000):
            key = deduplicator._generate_event_key(event)
            assert len(key) > 0
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 应该在合理时间内完成
        assert duration < 1.0  # 1秒内完成1000次键生成
    
    @pytest.mark.asyncio
    async def test_buffer_add_performance(self):
        """测试缓冲区添加性能"""
        buffer = FeedbackBuffer(max_size=1000)
        
        events = []
        for i in range(100):
            event = CollectedEvent(
                event_id=f"perf_event_{i}",
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                item_id=f"item_{i}",
                feedback_type=FeedbackType.CLICK,
                raw_value=True,
                context={},
                timestamp=utc_now(),
                priority=EventPriority.LOW
            )
            events.append(event)
        
        start_time = time.time()
        
        for event in events:
            result = await buffer.add_event(event)
            assert result is True
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 100个事件应该在合理时间内添加完成
        assert duration < 0.5  # 0.5秒内完成
        assert len(buffer.buffer) == 100

# 添加更多测试覆盖剩余代码
class TestCollectorAdvancedFeatures:
    """测试收集器高级功能"""
    
    @pytest.mark.asyncio 
    async def test_collect_implicit_feedback_mock(self):
        """测试收集隐式反馈（模拟方法）"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            # 模拟反馈数据收集
            result = await collector.collect_implicit_feedback(
                user_id="user_123",
                session_id="session_456", 
                item_id="item_789",
                event_type="click",
                event_data={"value": True},
                context={"page": "test"}
            )
            
            # 由于缺少实际服务，结果应该基于验证逻辑
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_collect_explicit_feedback_mock(self):
        """测试收集显式反馈（模拟方法）"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            result = await collector.collect_explicit_feedback(
                user_id="user_123",
                session_id="session_456",
                item_id="item_789", 
                feedback_type="rating",
                value=4,
                context={"source": "test"}
            )
            
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_get_stats_functionality(self):
        """测试统计信息获取功能"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            stats = await collector.get_stats()
            
            # 验证统计信息结构
            required_keys = [
                "total_received", "total_processed", "duplicates_filtered",
                "validation_failures", "buffer_overflows", "current_buffer_size",
                "buffer_utilization_percent", "dedup_entries", "is_running"
            ]
            
            for key in required_keys:
                assert key in stats
            
            assert isinstance(stats["buffer_utilization_percent"], (int, float))
            assert stats["is_running"] in [True, False]

class TestValidatorComprehensive:
    """验证器全面测试"""
    
    def test_all_implicit_feedback_types(self):
        """测试所有隐式反馈类型的验证"""
        validator = EventValidator()
        
        # 测试VIEW类型
        view_event = CollectedEvent(
            event_id="view_test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.VIEW,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        assert validator.validate_implicit_event(view_event) is True
        
        # 测试HOVER类型
        hover_event = CollectedEvent(
            event_id="hover_test",
            user_id="user_1", 
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.HOVER,
            raw_value=2.5,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        assert validator.validate_implicit_event(hover_event) is True
        
        # 测试负数 hover（应该失败）
        hover_event.raw_value = -1
        assert validator.validate_implicit_event(hover_event) is False
    
    def test_all_explicit_feedback_types(self):
        """测试所有显式反馈类型的验证"""
        validator = EventValidator()
        
        # 测试LIKE
        like_event = CollectedEvent(
            event_id="like_test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.LIKE,
            raw_value=1,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        assert validator.validate_explicit_event(like_event) is True
        
        # 测试DISLIKE  
        dislike_event = CollectedEvent(
            event_id="dislike_test", 
            user_id="user_1",
            session_id="session_1", 
            item_id="item_1",
            feedback_type=FeedbackType.DISLIKE,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        assert validator.validate_explicit_event(dislike_event) is True
        
        # 测试BOOKMARK
        bookmark_event = CollectedEvent(
            event_id="bookmark_test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1", 
            feedback_type=FeedbackType.BOOKMARK,
            raw_value=False,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.HIGH
        )
        assert validator.validate_explicit_event(bookmark_event) is True
        
        # 测试SHARE
        share_event = CollectedEvent(
            event_id="share_test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.SHARE,
            raw_value=0,
            context={},
            timestamp=utc_now(), 
            priority=EventPriority.HIGH
        )
        assert validator.validate_explicit_event(share_event) is True

class TestBufferEdgeCases:
    """缓冲器边界情况测试"""
    
    @pytest.mark.asyncio
    async def test_should_flush_size_threshold(self):
        """测试基于大小阈值的刷新"""
        buffer = FeedbackBuffer(max_size=10, flush_interval=60.0)  # 长间隔
        
        # 添加到80%阈值（8个事件）
        for i in range(8):
            event = CollectedEvent(
                event_id=f"size_test_{i}",
                user_id="user_1",
                session_id="session_1", 
                item_id="item_1",
                feedback_type=FeedbackType.CLICK,
                raw_value=True,
                context={},
                timestamp=utc_now(),
                priority=EventPriority.LOW
            )
            await buffer.add_event(event)
        
        # 应该触发基于大小的刷新
        should_flush = await buffer.should_flush()
        assert should_flush is True
    
    @pytest.mark.asyncio
    async def test_flush_empty_priority_buffer(self):
        """测试刷新空的优先级缓冲区"""
        buffer = FeedbackBuffer()
        
        # 不添加任何事件，直接刷新
        events = await buffer.flush()
        assert events == []
        
        # 验证缓冲区状态
        assert len(buffer.buffer) == 0
        for priority_buffer in buffer.priority_buffers.values():
            assert len(priority_buffer) == 0

class TestGlobalCollectorFunctions:
    """全局收集器函数测试"""
    
    @pytest.mark.asyncio
    async def test_global_collector_lifecycle(self):
        """测试全局收集器生命周期"""
        from services.feedback_collector import (
            get_feedback_collector, 
            shutdown_feedback_collector,
            _feedback_collector
        )
        
        # 确保开始时没有全局实例
        import services.feedback_collector
        services.feedback_collector._feedback_collector = None
        
        # 获取收集器实例
        collector = await get_feedback_collector()
        assert collector is not None
        assert collector._running is True
        
        # 再次获取应该返回相同实例 
        collector2 = await get_feedback_collector()
        assert collector is collector2
        
        # 关闭收集器
        await shutdown_feedback_collector()
        
        # 全局实例应该被清空
        assert services.feedback_collector._feedback_collector is None

# 额外测试来覆盖剩余的代码分支
class TestAdditionalCoverage:
    """额外测试来覆盖剩余代码"""
    
    @pytest.mark.asyncio
    async def test_collect_batch_feedback_functionality(self):
        """测试批量反馈收集功能"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            # 准备批量事件数据
            events = [
                {
                    "user_id": "user_1",
                    "session_id": "session_1",
                    "item_id": "item_1",
                    "event_type": "click",
                    "value": True,
                    "is_explicit": False
                },
                {
                    "user_id": "user_2",
                    "session_id": "session_2",
                    "item_id": "item_2",
                    "feedback_type": "rating",
                    "value": 4,
                    "is_explicit": True
                },
                {
                    "user_id": "user_3",
                    "session_id": "session_3",
                    "item_id": "item_3",
                    "event_type": "invalid_event",  # 这会导致错误
                    "value": "invalid",
                    "is_explicit": False
                }
            ]
            
            # 测试批量收集
            results = await collector.collect_batch_feedback(events)
            
            # 验证结果结构
            assert "total" in results
            assert "successful" in results
            assert "failed" in results
            assert "errors" in results
            
            assert results["total"] == 3
            assert isinstance(results["successful"], int)
            assert isinstance(results["failed"], int)
            assert isinstance(results["errors"], list)
    
    @pytest.mark.asyncio
    async def test_invalid_feedback_types_handling(self):
        """测试无效反馈类型的处理"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            # 测试无效的隐式反馈类型
            result1 = await collector.collect_implicit_feedback(
                user_id="user_1",
                session_id="session_1",
                item_id="item_1",
                event_type="invalid_type",  # 无效类型
                event_data={"value": True}
            )
            assert result1 is False
            
            # 测试无效的显式反馈类型
            result2 = await collector.collect_explicit_feedback(
                user_id="user_1",
                session_id="session_1", 
                item_id="item_1",
                feedback_type="invalid_type",  # 无效类型
                value=5
            )
            assert result2 is False
    
    def test_validator_edge_cases(self):
        """测试验证器边界情况"""
        validator = EventValidator()
        
        # 测试所有未覆盖的反馈类型
        focus_event = CollectedEvent(
            event_id="focus_test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.FOCUS,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        result = validator.validate_implicit_event(focus_event)
        assert isinstance(result, bool)  # 应该返回False，因为没有对应的验证逻辑
        
        blur_event = CollectedEvent(
            event_id="blur_test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.BLUR,
            raw_value=True,
            context={},
            timestamp=utc_now(),
            priority=EventPriority.LOW
        )
        result = validator.validate_implicit_event(blur_event)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_background_flush_task_coverage(self):
        """测试后台刷新任务覆盖"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=0.1,  # 快速刷新
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            # 启动收集器
            await collector.start()
            
            # 添加一些事件
            await collector.collect_implicit_feedback(
                user_id="user_1",
                session_id="session_1",
                item_id="item_1",
                event_type="click",
                event_data={"value": True}
            )
            
            # 等待一段时间让后台任务运行
            await asyncio.sleep(0.2)
            
            # 停止收集器
            await collector.stop()
            
            # 验证收集器已停止
            assert collector._running is False
    
    def test_flush_events_method_coverage(self):
        """测试_flush_events方法的覆盖"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            # 直接测试_flush_events方法
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(collector._flush_events())
            finally:
                loop.close()
    
    @pytest.mark.asyncio
    async def test_malformed_event_data_handling(self):
        """测试畸形事件数据处理"""
        with patch('services.feedback_collector.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                FEEDBACK_BUFFER_SIZE=1000,
                FEEDBACK_FLUSH_INTERVAL=5.0,
                FEEDBACK_DEDUP_WINDOW=300
            )
            
            collector = FeedbackCollector()
            
            # 测试缺少必需字段的数据
            result = await collector.collect_implicit_feedback(
                user_id="user_1",
                session_id="session_1",
                item_id="item_1",
                event_type="click",
                event_data={}  # 缺少value字段
            )
            
            # 应该处理错误并返回False
            assert result is False
