"""
事件收集器测试
"""

import pytest
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai.analytics.models import BehaviorEvent
from src.ai.analytics.behavior.event_collector import EventCollector


class TestEventCollector:
    """事件收集器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.collector = EventCollector()
        self.sample_event = BehaviorEvent(
            event_id="test-event-1",
            user_id="user-123",
            session_id="session-456",
            event_type="page_view",
            timestamp=utc_now(),
            properties={"page": "/dashboard", "referrer": "/login"},
            context={"user_agent": "Mozilla/5.0", "ip": "127.0.0.1"}
        )
    
    def teardown_method(self):
        """测试后清理"""
        asyncio.create_task(self.collector.stop())
    
    @pytest.mark.asyncio
    async def test_collect_single_event(self):
        """测试收集单个事件"""
        # 启动收集器
        await self.collector.start()
        
        # 收集事件
        await self.collector.collect_event(self.sample_event)
        
        # 验证事件在缓冲区
        assert len(self.collector.buffer) == 1
        assert self.collector.buffer[0].event_id == "test-event-1"
    
    @pytest.mark.asyncio
    async def test_batch_collection(self):
        """测试批量收集事件"""
        events = [
            BehaviorEvent(
                event_id=f"test-event-{i}",
                user_id=f"user-{i}",
                event_type="click",
                timestamp=utc_now()
            ) for i in range(5)
        ]
        
        await self.collector.start()
        
        # 批量收集
        for event in events:
            await self.collector.collect_event(event)
        
        assert len(self.collector.buffer) == 5
    
    @pytest.mark.asyncio
    async def test_buffer_flush_on_size_limit(self):
        """测试缓冲区达到大小限制时自动刷新"""
        # 设置较小的缓冲区大小
        self.collector.buffer_size = 3
        
        with patch.object(self.collector, '_flush_to_storage', new_callable=AsyncMock) as mock_flush:
            await self.collector.start()
            
            # 添加事件直到触发刷新
            for i in range(4):
                event = BehaviorEvent(
                    event_id=f"test-event-{i}",
                    user_id="user-123",
                    event_type="click",
                    timestamp=utc_now()
                )
                await self.collector.collect_event(event)
            
            # 验证刷新被调用
            mock_flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_buffer_flush_on_time_interval(self):
        """测试定时刷新缓冲区"""
        # 设置短的刷新间隔
        self.collector.flush_interval = 0.1
        
        with patch.object(self.collector, '_flush_to_storage', new_callable=AsyncMock) as mock_flush:
            await self.collector.start()
            
            # 添加一个事件
            await self.collector.collect_event(self.sample_event)
            
            # 等待定时刷新
            await asyncio.sleep(0.2)
            
            # 验证定时刷新被调用
            mock_flush.assert_called()
    
    @pytest.mark.asyncio
    async def test_quality_monitoring(self):
        """测试数据质量监控"""
        # 创建质量有问题的事件
        bad_event = BehaviorEvent(
            event_id="",  # 空ID
            user_id="user-123",
            event_type="",  # 空类型
            timestamp=utc_now()
        )
        
        await self.collector.start()
        
        # 收集有问题的事件
        await self.collector.collect_event(bad_event)
        
        # 验证质量监控统计
        stats = self.collector.get_quality_stats()
        assert stats["total_events"] == 1
        assert stats["quality_issues"] > 0
    
    @pytest.mark.asyncio
    async def test_event_compression(self):
        """测试事件数据压缩"""
        await self.collector.start()
        
        # 收集大量事件
        events = [
            BehaviorEvent(
                event_id=f"test-event-{i}",
                user_id="user-123",
                event_type="page_view",
                timestamp=utc_now(),
                properties={"large_data": "x" * 1000}  # 大数据
            ) for i in range(10)
        ]
        
        for event in events:
            await self.collector.collect_event(event)
        
        # 触发压缩
        compressed_data = await self.collector.compress_buffer()
        
        # 验证压缩效果
        assert len(compressed_data) < len(str(events))
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        await self.collector.start()
        
        # 模拟存储失败
        with patch.object(self.collector, '_flush_to_storage', side_effect=Exception("Storage error")):
            # 收集事件
            await self.collector.collect_event(self.sample_event)
            
            # 手动触发刷新
            await self.collector.flush_buffer()
            
            # 验证错误被处理，事件保留在缓冲区
            assert len(self.collector.buffer) > 0
    
    @pytest.mark.asyncio
    async def test_statistics_collection(self):
        """测试统计信息收集"""
        await self.collector.start()
        
        # 收集多个事件
        for i in range(5):
            await self.collector.collect_event(self.sample_event)
        
        # 获取统计信息
        stats = self.collector.get_statistics()
        
        assert stats["total_events_collected"] == 5
        assert stats["buffer_size"] == 5
        assert "collection_rate" in stats
    
    @pytest.mark.asyncio
    async def test_concurrent_collection(self):
        """测试并发事件收集"""
        await self.collector.start()
        
        # 创建并发任务
        tasks = []
        for i in range(10):
            event = BehaviorEvent(
                event_id=f"concurrent-event-{i}",
                user_id="user-123",
                event_type="click",
                timestamp=utc_now()
            )
            task = asyncio.create_task(self.collector.collect_event(event))
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        # 验证所有事件都被收集
        assert len(self.collector.buffer) == 10
    
    @pytest.mark.asyncio
    async def test_event_validation(self):
        """测试事件验证"""
        await self.collector.start()
        
        # 测试有效事件
        valid_event = BehaviorEvent(
            event_id="valid-event",
            user_id="user-123",
            event_type="click",
            timestamp=utc_now()
        )
        
        await self.collector.collect_event(valid_event)
        assert len(self.collector.buffer) == 1
        
        # 测试无效事件（None）
        await self.collector.collect_event(None)
        assert len(self.collector.buffer) == 1  # 没有增加
    
    @pytest.mark.asyncio
    async def test_memory_management(self):
        """测试内存管理"""
        await self.collector.start()
        
        # 填充大量数据
        for i in range(1000):
            event = BehaviorEvent(
                event_id=f"memory-test-{i}",
                user_id="user-123",
                event_type="click",
                timestamp=utc_now()
            )
            await self.collector.collect_event(event)
        
        # 检查内存使用情况
        memory_stats = self.collector.get_memory_stats()
        assert memory_stats["buffer_memory_usage"] > 0
        assert memory_stats["total_memory_usage"] > 0