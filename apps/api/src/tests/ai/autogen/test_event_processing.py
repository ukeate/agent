"""
事件处理框架单元测试
"""

import asyncio
import pytest
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import Mock, AsyncMock, patch
import uuid
import sys
import os
from src.ai.autogen.events import Event, EventType, EventPriority, EventBus
from src.ai.autogen.event_processors import (
    EventContext,
    ProcessingResult,
    EventProcessor,
    AsyncEventProcessingEngine,
    AgentMessageProcessor,
    TaskProcessor
)
from src.ai.autogen.event_store import EventStore, EventReplayService
from src.ai.autogen.event_router import (
    EventFilter,
    EventRouter,
    FilterCondition,
    FilterOperator,
    EventAggregator
)
from src.ai.autogen.error_recovery import (
    RetryPolicy,
    RetryStrategy,
    CircuitBreaker,
    RetryableEventProcessor,
    DeadLetterQueue,
    CompensationManager,
    CompensationAction

)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class TestEventProcessor(EventProcessor):
    """测试用事件处理器"""
    
    def __init__(self, name="TestProcessor", can_handle_types=None):
        super().__init__(name=name)
        self.can_handle_types = can_handle_types or [EventType.MESSAGE_SENT]
        self.processed_events = []
        self.should_fail = False
        self.fail_count = 0
        self.max_fails = 0
    
    async def process(self, event: Event, context: EventContext) -> ProcessingResult:
        """处理事件"""
        if self.should_fail and self.fail_count < self.max_fails:
            self.fail_count += 1
            raise Exception("Test failure")
        
        self.processed_events.append(event)
        return ProcessingResult(success=True, result="Test result")
    
    def can_handle(self, event: Event) -> bool:
        """判断是否能处理该事件"""
        return event.type in self.can_handle_types

@pytest.fixture
def event_bus():
    """创建事件总线"""
    return EventBus()

@pytest.fixture
def processing_engine():
    """创建事件处理引擎"""
    return AsyncEventProcessingEngine(max_workers=2, batch_size=10)

@pytest.fixture
def test_event():
    """创建测试事件"""
    return Event(
        id=str(uuid.uuid4()),
        type=EventType.MESSAGE_SENT,
        source="test_agent",
        target="target_agent",
        data={"message": "test"},
        priority=EventPriority.NORMAL
    )

@pytest.fixture
def event_context():
    """创建事件上下文"""
    return EventContext(
        correlation_id=str(uuid.uuid4()),
        user_id="test_user",
        session_id="test_session"
    )

class TestEventProcessingEngine:
    """测试事件处理引擎"""
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, processing_engine):
        """测试引擎启动和停止"""
        await processing_engine.start()
        assert processing_engine.running is True
        
        await processing_engine.stop()
        assert processing_engine.running is False
    
    @pytest.mark.asyncio
    async def test_register_processor(self, processing_engine):
        """测试注册处理器"""
        processor = TestEventProcessor()
        processing_engine.register_processor(processor)
        
        assert processor in processing_engine.processors
    
    @pytest.mark.asyncio
    async def test_submit_event(self, processing_engine, test_event):
        """测试提交事件"""
        processor = TestEventProcessor()
        processing_engine.register_processor(processor)
        
        await processing_engine.start()
        await processing_engine.submit_event(test_event)
        
        # 等待处理
        await asyncio.sleep(0.1)
        
        assert len(processor.processed_events) == 1
        assert processor.processed_events[0] == test_event
        
        await processing_engine.stop()
    
    @pytest.mark.asyncio
    async def test_priority_queues(self, processing_engine):
        """测试优先级队列"""
        await processing_engine.start()
        
        # 提交不同优先级的事件
        critical_event = Event(type=EventType.ERROR_OCCURRED, priority=EventPriority.CRITICAL)
        normal_event = Event(type=EventType.MESSAGE_SENT, priority=EventPriority.NORMAL)
        low_event = Event(type=EventType.MESSAGE_SENT, priority=EventPriority.LOW)
        
        await processing_engine.submit_event(low_event, EventPriority.LOW)
        await processing_engine.submit_event(normal_event, EventPriority.NORMAL)
        await processing_engine.submit_event(critical_event, EventPriority.CRITICAL)
        
        # 验证队列大小
        assert processing_engine.priority_queues[EventPriority.CRITICAL].qsize() == 1
        assert processing_engine.priority_queues[EventPriority.NORMAL].qsize() == 1
        assert processing_engine.priority_queues[EventPriority.LOW].qsize() == 1
        
        await processing_engine.stop()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processing_engine):
        """测试批处理"""
        await processing_engine.start()
        
        # 批量提交事件
        events = [
            Event(type=EventType.MESSAGE_SENT, data={"id": i})
            for i in range(5)
        ]
        
        for event in events:
            await processing_engine.submit_event(event, batch=True)
        
        # 验证缓冲区
        assert len(processing_engine.batch_buffers[EventPriority.NORMAL]) == 5
        
        # 等待批处理刷新
        await asyncio.sleep(1.5)
        
        # 验证事件已被处理
        assert len(processing_engine.batch_buffers[EventPriority.NORMAL]) == 0
        
        await processing_engine.stop()

class TestEventRouter:
    """测试事件路由器"""
    
    @pytest.mark.asyncio
    async def test_add_route(self):
        """测试添加路由"""
        router = EventRouter()
        processor = TestEventProcessor()
        
        filter = EventFilter(
            event_type_pattern=None,
            conditions=[
                FilterCondition(
                    field="type",
                    operator=FilterOperator.EQUALS,
                    value=EventType.MESSAGE_SENT
                )
            ]
        )
        
        route = router.add_route(filter, [processor], name="test_route")
        
        assert route in router.routes
        assert route.name == "test_route"
    
    @pytest.mark.asyncio
    async def test_route_event(self, test_event):
        """测试路由事件"""
        router = EventRouter()
        processor1 = TestEventProcessor(name="Processor1")
        processor2 = TestEventProcessor(name="Processor2")
        
        # 添加路由规则
        filter1 = EventFilter(
            conditions=[
                FilterCondition(
                    field="source",
                    operator=FilterOperator.EQUALS,
                    value="test_agent"
                )
            ]
        )
        
        filter2 = EventFilter(
            conditions=[
                FilterCondition(
                    field="type",
                    operator=FilterOperator.EQUALS,
                    value=EventType.MESSAGE_SENT
                )
            ]
        )
        
        router.add_route(filter1, [processor1])
        router.add_route(filter2, [processor2])
        
        # 路由事件
        pairs = await router.route_event(test_event)
        
        # 两个路由都应该匹配
        assert len(pairs) == 2
        assert any(p[0] == processor1 for p in pairs)
        assert any(p[0] == processor2 for p in pairs)
    
    def test_filter_conditions(self, test_event):
        """测试过滤条件"""
        # 等于条件
        condition = FilterCondition(
            field="source",
            operator=FilterOperator.EQUALS,
            value="test_agent"
        )
        assert condition.evaluate(test_event) is True
        
        # 包含条件
        condition = FilterCondition(
            field="data.message",
            operator=FilterOperator.CONTAINS,
            value="test"
        )
        assert condition.evaluate(test_event) is True
        
        # 存在条件
        condition = FilterCondition(
            field="source",
            operator=FilterOperator.EXISTS,
            value=None
        )
        assert condition.evaluate(test_event) is True

class TestErrorRecovery:
    """测试错误恢复机制"""
    
    def test_retry_policy(self):
        """测试重试策略"""
        policy = RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay_ms=100,
            multiplier=2.0,
            jitter=False
        )
        
        # 测试延迟计算
        assert policy.calculate_delay(0) == 100
        assert policy.calculate_delay(1) == 200
        assert policy.calculate_delay(2) == 400
        
        # 测试重试判断
        assert policy.should_retry(Exception(), 0) is True
        assert policy.should_retry(Exception(), 3) is False
    
    def test_circuit_breaker(self):
        """测试断路器"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1000,
            success_threshold=2
        )
        
        # 初始状态应该是关闭
        assert breaker.allow_request() is True
        
        # 记录失败直到打开
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.is_open() is True
        assert breaker.allow_request() is False
        
        # 半开状态
        breaker.half_open()
        assert breaker.state == "half_open"
        
        # 记录成功直到关闭
        for _ in range(2):
            breaker.record_success()
        
        assert breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_retryable_processor(self, test_event, event_context):
        """测试可重试处理器"""
        base_processor = TestEventProcessor()
        base_processor.should_fail = True
        base_processor.max_fails = 2
        
        retry_policy = RetryPolicy(
            max_retries=3,
            initial_delay_ms=10,
            strategy=RetryStrategy.FIXED_DELAY
        )
        
        processor = RetryableEventProcessor(base_processor, retry_policy)
        
        # 处理事件（前两次失败，第三次成功）
        result = await processor.process(test_event, event_context)
        
        assert result.success is True
        assert len(base_processor.processed_events) == 1
        assert processor.retry_stats["total_retries"] == 2
    
    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, test_event):
        """测试死信队列"""
        dlq = DeadLetterQueue()
        
        # 添加死信
        await dlq.add_dead_letter(
            test_event,
            "Test error",
            retry_count=3,
            processor_name="TestProcessor"
        )
        
        # 获取死信
        dead_letters = await dlq.get_dead_letters()
        
        assert len(dead_letters) == 1
        assert dead_letters[0]["error"] == "Test error"
        assert dead_letters[0]["retry_count"] == 3
        
        # 测试统计
        stats = dlq.get_stats()
        assert stats["total_dead_letters"] == 1
        assert stats["current_dead_letters"] == 1
    
    @pytest.mark.asyncio
    async def test_compensation_manager(self):
        """测试补偿管理器"""
        manager = CompensationManager()
        
        # 注册补偿处理器
        compensation_called = False
        
        async def compensate_create(data):
            nonlocal compensation_called
            compensation_called = True
        
        manager.register_compensation("create", compensate_create)
        
        # 开始Saga事务
        saga_id = "test_saga"
        await manager.start_saga(saga_id)
        
        # 记录操作
        await manager.record_operation(
            saga_id,
            "create",
            {"entity_id": "123"}
        )
        
        # 执行补偿
        result = await manager.compensate_saga(saga_id, "Test failure")
        
        assert result["success"] is True
        assert compensation_called is True
        assert "create" in result["compensated"]

class TestEventStore:
    """测试事件存储"""
    
    @pytest.mark.asyncio
    async def test_append_and_get_event(self):
        """测试存储和获取事件"""
        # 使用mock的Redis和PostgreSQL
        redis_mock = AsyncMock()
        postgres_mock = AsyncMock()
        
        # 配置postgres mock的acquire方法
        conn_mock = AsyncMock()
        conn_mock.execute = AsyncMock()
        
        # 创建一个模拟的异步上下文管理器
        acquire_mock = AsyncMock()
        acquire_mock.__aenter__ = AsyncMock(return_value=conn_mock)
        acquire_mock.__aexit__ = AsyncMock(return_value=None)
        postgres_mock.acquire = Mock(return_value=acquire_mock)
        
        store = EventStore(redis_mock, postgres_mock)
        
        event = Event(
            id=str(uuid.uuid4()),
            type=EventType.MESSAGE_SENT,
            source="test",
            data={"message": "test"}
        )
        
        # 存储事件
        event_id = await store.append_event(event)
        
        assert event_id is not None
        assert redis_mock.xadd.called
        assert postgres_mock.acquire.called
    
    @pytest.mark.asyncio
    async def test_replay_events(self):
        """测试事件重播"""
        postgres_mock = AsyncMock()
        
        # 模拟数据库返回
        mock_rows = [
            {
                "id": str(uuid.uuid4()),
                "type": EventType.MESSAGE_SENT.value,
                "source": "test",
                "target": None,
                "data": '{"message": "test"}',
                "timestamp": utc_now(),
                "correlation_id": None,
                "conversation_id": None,
                "session_id": None,
                "priority": "normal"
            }
        ]
        
        # 配置postgres mock的acquire方法
        conn_mock = AsyncMock()
        conn_mock.fetch = AsyncMock(return_value=mock_rows)
        
        # 创建一个模拟的异步上下文管理器
        acquire_mock = AsyncMock()
        acquire_mock.__aenter__ = AsyncMock(return_value=conn_mock)
        acquire_mock.__aexit__ = AsyncMock(return_value=None)
        postgres_mock.acquire = Mock(return_value=acquire_mock)
        
        store = EventStore(None, postgres_mock)
        
        # 重播事件
        events = await store.replay_events(
            start_time=utc_now() - timedelta(hours=1),
            end_time=utc_now()
        )
        
        assert len(events) == 1
        assert events[0].type == EventType.MESSAGE_SENT

class TestEventAggregator:
    """测试事件聚合器"""
    
    @pytest.mark.asyncio
    async def test_event_aggregation(self):
        """测试事件聚合"""
        aggregator = EventAggregator(
            window_size=3,
            time_window_seconds=60
        )
        
        # 添加聚合函数
        def count_events(events):
            return len(events)
        
        aggregator.add_aggregation_function("count", count_events)
        
        # 添加事件
        for i in range(3):
            event = Event(
                type=EventType.MESSAGE_SENT,
                data={"id": i}
            )
            await aggregator.add_event(event)
        
        # 窗口应该已满并刷新
        stats = aggregator.get_window_stats()
        assert stats["event_count"] == 0  # 窗口已刷新
        
        # 添加更多事件
        for i in range(2):
            event = Event(
                type=EventType.MESSAGE_SENT,
                data={"id": i + 3}
            )
            await aggregator.add_event(event)
        
        stats = aggregator.get_window_stats()
        assert stats["event_count"] == 2

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_event_processing(self):
        """端到端事件处理测试"""
        # 创建组件
        event_bus = EventBus()
        processing_engine = AsyncEventProcessingEngine(max_workers=2)
        router = EventRouter()
        dlq = DeadLetterQueue()
        
        # 创建处理器
        processor1 = TestEventProcessor(name="Processor1")
        processor2 = TestEventProcessor(name="Processor2")
        processor2.should_fail = True
        processor2.max_fails = 1
        
        # 配置路由
        filter1 = EventFilter(
            conditions=[
                FilterCondition(
                    field="priority",
                    operator=FilterOperator.EQUALS,
                    value=EventPriority.HIGH
                )
            ]
        )
        
        router.add_route(filter1, [processor1])
        router.add_default_processor(processor2)
        
        # 注册处理器
        processing_engine.register_processor(processor1)
        processing_engine.register_processor(processor2)
        
        # 启动引擎
        await event_bus.start()
        await processing_engine.start()
        
        # 发送事件
        high_priority_event = Event(
            type=EventType.MESSAGE_SENT,
            priority=EventPriority.HIGH,
            data={"message": "high"}
        )
        
        normal_priority_event = Event(
            type=EventType.MESSAGE_SENT,
            priority=EventPriority.NORMAL,
            data={"message": "normal"}
        )
        
        await event_bus.publish(high_priority_event)
        await event_bus.publish(normal_priority_event)
        
        # 通过路由器处理
        pairs1 = await router.route_event(high_priority_event)
        pairs2 = await router.route_event(normal_priority_event)
        
        for processor, event in pairs1:
            await processing_engine.submit_event(event)
        
        for processor, event in pairs2:
            await processing_engine.submit_event(event)
        
        # 等待处理
        await asyncio.sleep(0.5)
        
        # 验证结果
        assert len(processor1.processed_events) >= 1
        assert len(processor2.processed_events) >= 1  # 第二次尝试应该成功
        
        # 清理
        await event_bus.stop()
        await processing_engine.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
