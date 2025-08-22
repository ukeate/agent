"""
用户反馈收集器测试套件

测试反馈事件的收集、缓冲、去重和验证功能。
覆盖隐式和显式反馈收集的各种场景。
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.feedback_collector import (
    FeedbackCollector,
    FeedbackBuffer,
    EventDeduplicator,
    EventValidator,
    CollectedEvent,
    EventPriority,
    get_feedback_collector,
    shutdown_feedback_collector
)
from models.schemas.feedback import FeedbackType


class TestCollectedEvent:
    """CollectedEvent数据结构测试"""
    
    def test_event_creation(self):
        """测试事件对象创建"""
        event = CollectedEvent(
            event_id="test_123",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={"page": "home"},
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        
        assert event.event_id == "test_123"
        assert event.user_id == "user_1"
        assert event.feedback_type == FeedbackType.CLICK
        assert event.priority == EventPriority.HIGH
        assert event.context["page"] == "home"


class TestFeedbackBuffer:
    """反馈缓冲器测试"""
    
    @pytest.fixture
    def buffer(self):
        return FeedbackBuffer(max_size=5, flush_interval=1.0)
    
    @pytest.fixture
    def sample_event(self):
        return CollectedEvent(
            event_id="test_event",
            user_id="user_1",
            session_id="session_1", 
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.MEDIUM
        )
    
    @pytest.mark.asyncio
    async def test_add_event_success(self, buffer, sample_event):
        """测试成功添加事件"""
        result = await buffer.add_event(sample_event)
        
        assert result is True
        assert len(buffer.buffer) == 1
        assert len(buffer.priority_buffers[EventPriority.MEDIUM]) == 1
    
    @pytest.mark.asyncio
    async def test_add_event_buffer_full(self, buffer):
        """测试缓冲区满时的处理"""
        # 填满缓冲区
        for i in range(6):  # 超过max_size=5
            event = CollectedEvent(
                event_id=f"test_{i}",
                user_id="user_1",
                session_id="session_1",
                item_id="item_1", 
                feedback_type=FeedbackType.CLICK,
                raw_value=True,
                context={},
                timestamp=datetime.now(),
                priority=EventPriority.LOW
            )
            result = await buffer.add_event(event)
            if i < 5:
                assert result is True
            else:
                assert result is False  # 第6个应该失败
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, buffer):
        """测试优先级排序"""
        # 添加不同优先级的事件
        events = []
        for priority in [EventPriority.LOW, EventPriority.HIGH, EventPriority.MEDIUM]:
            event = CollectedEvent(
                event_id=f"test_{priority}",
                user_id="user_1",
                session_id="session_1",
                item_id="item_1",
                feedback_type=FeedbackType.CLICK,
                raw_value=True,
                context={},
                timestamp=datetime.now(),
                priority=priority
            )
            await buffer.add_event(event)
            events.append(event)
        
        # 刷新缓冲区
        flushed_events = await buffer.flush()
        
        # 验证优先级排序：HIGH -> MEDIUM -> LOW
        assert len(flushed_events) == 3
        assert flushed_events[0].priority == EventPriority.HIGH
        assert flushed_events[1].priority == EventPriority.MEDIUM
        assert flushed_events[2].priority == EventPriority.LOW
    
    @pytest.mark.asyncio
    async def test_should_flush_time_threshold(self, buffer):
        """测试时间阈值触发刷新"""
        # 添加事件
        event = CollectedEvent(
            event_id="test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
        await buffer.add_event(event)
        
        # 模拟时间过去
        buffer._last_flush = time.time() - 2.0  # 2秒前
        
        should_flush = await buffer.should_flush()
        assert should_flush is True
    
    @pytest.mark.asyncio
    async def test_should_flush_high_priority(self, buffer):
        """测试高优先级事件立即触发刷新"""
        event = CollectedEvent(
            event_id="test",
            user_id="user_1", 
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=5,
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        await buffer.add_event(event)
        
        should_flush = await buffer.should_flush()
        assert should_flush is True


class TestEventDeduplicator:
    """事件去重器测试"""
    
    @pytest.fixture
    def deduplicator(self):
        return EventDeduplicator(window_seconds=60)
    
    @pytest.fixture
    def sample_event(self):
        return CollectedEvent(
            event_id="test_event",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1", 
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
    
    @pytest.mark.asyncio
    async def test_first_event_not_duplicate(self, deduplicator, sample_event):
        """测试首次事件不被认为是重复"""
        is_dup = await deduplicator.is_duplicate(sample_event)
        assert is_dup is False
    
    @pytest.mark.asyncio
    async def test_identical_event_is_duplicate(self, deduplicator, sample_event):
        """测试相同事件被识别为重复"""
        # 第一次添加
        await deduplicator.is_duplicate(sample_event)
        
        # 第二次添加相同事件
        is_dup = await deduplicator.is_duplicate(sample_event)
        assert is_dup is True
    
    @pytest.mark.asyncio
    async def test_different_user_not_duplicate(self, deduplicator, sample_event):
        """测试不同用户的相同事件不算重复"""
        # 第一次添加
        await deduplicator.is_duplicate(sample_event)
        
        # 创建不同用户的事件
        different_user_event = CollectedEvent(
            event_id="test_event_2",
            user_id="user_2",  # 不同用户
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
        
        is_dup = await deduplicator.is_duplicate(different_user_event)
        assert is_dup is False
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self, deduplicator, sample_event):
        """测试过期条目清理"""
        # 添加事件
        await deduplicator.is_duplicate(sample_event)
        
        # 模拟时间过去，超过去重窗口
        with patch('services.feedback_collector.datetime') as mock_datetime:
            future_time = datetime.now() + timedelta(seconds=120)  # 超过60秒窗口
            mock_datetime.now.return_value = future_time
            
            # 再次检查，应该不是重复（因为已过期）
            is_dup = await deduplicator.is_duplicate(sample_event)
            assert is_dup is False


class TestEventValidator:
    """事件验证器测试"""
    
    def test_validate_implicit_click(self):
        """测试点击事件验证"""
        event = CollectedEvent(
            event_id="test",
            user_id="user_1",
            session_id="session_1", 
            item_id="item_1",
            feedback_type=FeedbackType.CLICK,
            raw_value=True,
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
        
        assert EventValidator.validate_implicit_event(event) is True
        
        # 测试无效值
        event.raw_value = "invalid"
        assert EventValidator.validate_implicit_event(event) is False
    
    def test_validate_implicit_dwell_time(self):
        """测试停留时间事件验证"""
        event = CollectedEvent(
            event_id="test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1", 
            feedback_type=FeedbackType.DWELL_TIME,
            raw_value=30.5,  # 30.5秒
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
        
        assert EventValidator.validate_implicit_event(event) is True
        
        # 测试无效值（超出范围）
        event.raw_value = 7200  # 2小时，超过3600秒限制
        assert EventValidator.validate_implicit_event(event) is False
        
        # 测试负值
        event.raw_value = -1
        assert EventValidator.validate_implicit_event(event) is False
    
    def test_validate_implicit_scroll_depth(self):
        """测试滚动深度事件验证"""
        event = CollectedEvent(
            event_id="test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.SCROLL_DEPTH,
            raw_value=75.5,  # 75.5%
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.LOW
        )
        
        assert EventValidator.validate_implicit_event(event) is True
        
        # 测试超出百分比范围
        event.raw_value = 150
        assert EventValidator.validate_implicit_event(event) is False
    
    def test_validate_explicit_rating(self):
        """测试评分事件验证"""
        event = CollectedEvent(
            event_id="test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.RATING,
            raw_value=4,
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        
        assert EventValidator.validate_explicit_event(event) is True
        
        # 测试超出范围
        event.raw_value = 0  # 小于1
        assert EventValidator.validate_explicit_event(event) is False
        
        event.raw_value = 6  # 大于5
        assert EventValidator.validate_explicit_event(event) is False
    
    def test_validate_explicit_comment(self):
        """测试评论事件验证"""
        event = CollectedEvent(
            event_id="test",
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            feedback_type=FeedbackType.COMMENT,
            raw_value="这是一个很好的功能!",
            context={},
            timestamp=datetime.now(),
            priority=EventPriority.HIGH
        )
        
        assert EventValidator.validate_explicit_event(event) is True
        
        # 测试空评论
        event.raw_value = ""
        assert EventValidator.validate_explicit_event(event) is False
        
        event.raw_value = "   "  # 只有空格
        assert EventValidator.validate_explicit_event(event) is False


class TestFeedbackCollector:
    """反馈收集器主类测试"""
    
    @pytest.fixture
    async def collector(self):
        collector = FeedbackCollector()
        await collector.start()
        yield collector
        await collector.stop()
    
    @pytest.mark.asyncio
    async def test_collector_lifecycle(self):
        """测试收集器生命周期"""
        collector = FeedbackCollector()
        
        # 初始状态
        assert collector._running is False
        
        # 启动
        await collector.start()
        assert collector._running is True
        assert len(collector._background_tasks) > 0
        
        # 停止
        await collector.stop()
        assert collector._running is False
        assert len(collector._background_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_collect_implicit_feedback_success(self, collector):
        """测试成功收集隐式反馈"""
        result = await collector.collect_implicit_feedback(
            user_id="user_1",
            session_id="session_1",
            item_id="item_1", 
            event_type="click",
            event_data={"value": True, "page": "home"},
            context={"source": "test"}
        )
        
        assert result is True
        assert collector.stats["total_received"] == 1
        assert collector.stats["total_processed"] == 1
        assert collector.stats["validation_failures"] == 0
    
    @pytest.mark.asyncio
    async def test_collect_implicit_feedback_invalid(self, collector):
        """测试收集无效隐式反馈"""
        result = await collector.collect_implicit_feedback(
            user_id="user_1",
            session_id="session_1",
            item_id="item_1",
            event_type="click",
            event_data={"value": "invalid_click_value"},  # 无效值
        )
        
        assert result is False
        assert collector.stats["total_received"] == 1
        assert collector.stats["validation_failures"] == 1
        assert collector.stats["total_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_collect_explicit_feedback_success(self, collector):
        """测试成功收集显式反馈"""
        result = await collector.collect_explicit_feedback(
            user_id="user_1",
            session_id="session_1", 
            item_id="item_1",
            feedback_type="rating",
            value=5,
            context={"survey_id": "survey_1"}
        )
        
        assert result is True
        assert collector.stats["total_received"] == 1
        assert collector.stats["total_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_collect_duplicate_feedback(self, collector):
        """测试重复反馈过滤"""
        feedback_data = {
            "user_id": "user_1",
            "session_id": "session_1",
            "item_id": "item_1",
            "feedback_type": "rating",
            "value": 4
        }
        
        # 第一次收集
        result1 = await collector.collect_explicit_feedback(**feedback_data)
        assert result1 is True
        
        # 第二次收集相同反馈
        result2 = await collector.collect_explicit_feedback(**feedback_data)
        assert result2 is False
        assert collector.stats["duplicates_filtered"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_feedback_collection(self, collector):
        """测试批量反馈收集"""
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
                "value": 5,
                "is_explicit": True
            },
            {
                "user_id": "user_3",
                "session_id": "session_3", 
                "item_id": "item_3",
                "event_type": "click",
                "value": "invalid",  # 无效数据
                "is_explicit": False
            }
        ]
        
        results = await collector.collect_batch_feedback(events)
        
        assert results["total"] == 3
        assert results["successful"] == 2
        assert results["failed"] == 1
        assert len(results["errors"]) >= 0
    
    @pytest.mark.asyncio
    async def test_stats_collection(self, collector):
        """测试统计信息收集"""
        # 收集一些反馈
        await collector.collect_explicit_feedback(
            user_id="user_1",
            session_id="session_1",
            item_id="item_1", 
            feedback_type="rating",
            value=4
        )
        
        stats = await collector.get_stats()
        
        assert "total_received" in stats
        assert "total_processed" in stats
        assert "current_buffer_size" in stats
        assert "buffer_utilization_percent" in stats
        assert "is_running" in stats
        assert stats["is_running"] is True


class TestGlobalCollectorInstance:
    """全局收集器实例测试"""
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """测试单例模式"""
        collector1 = await get_feedback_collector()
        collector2 = await get_feedback_collector()
        
        # 应该返回同一个实例
        assert collector1 is collector2
        assert collector1._running is True
        
        # 清理
        await shutdown_feedback_collector()
    
    @pytest.mark.asyncio 
    async def test_shutdown_collector(self):
        """测试关闭收集器"""
        collector = await get_feedback_collector()
        assert collector._running is True
        
        await shutdown_feedback_collector()
        assert collector._running is False
        
        # 再次获取应该创建新实例
        new_collector = await get_feedback_collector()
        assert new_collector is not collector
        
        # 清理
        await shutdown_feedback_collector()


# 异常处理测试
class TestErrorHandling:
    """异常处理测试"""
    
    @pytest.mark.asyncio
    async def test_invalid_feedback_type(self):
        """测试无效反馈类型处理"""
        collector = FeedbackCollector()
        await collector.start()
        
        try:
            result = await collector.collect_implicit_feedback(
                user_id="user_1",
                session_id="session_1", 
                item_id="item_1",
                event_type="invalid_type",  # 无效类型
                event_data={"value": True}
            )
            
            assert result is False
        finally:
            await collector.stop()
    
    @pytest.mark.asyncio
    async def test_malformed_event_data(self):
        """测试畸形事件数据处理"""
        collector = FeedbackCollector()
        await collector.start()
        
        try:
            result = await collector.collect_implicit_feedback(
                user_id="user_1",
                session_id="session_1",
                item_id="item_1", 
                event_type="click",
                event_data={}  # 缺少value字段
            )
            
            assert result is False
        finally:
            await collector.stop()


# 性能测试
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_high_volume_collection(self):
        """测试高并发收集性能"""
        collector = FeedbackCollector()
        await collector.start()
        
        try:
            # 并发收集大量事件
            tasks = []
            for i in range(100):
                task = collector.collect_implicit_feedback(
                    user_id=f"user_{i}",
                    session_id=f"session_{i}",
                    item_id=f"item_{i}",
                    event_type="click", 
                    event_data={"value": True}
                )
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # 验证结果
            successful_count = sum(1 for r in results if r)
            assert successful_count > 90  # 至少90%成功
            
            # 验证性能（应该在合理时间内完成）
            duration = end_time - start_time
            assert duration < 2.0  # 2秒内完成
            
        finally:
            await collector.stop()