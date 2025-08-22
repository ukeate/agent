"""事件处理系统集成测试"""

import pytest
import asyncio
from typing import List, Dict, Any
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import json


class MockEventBus:
    """模拟事件总线"""
    def __init__(self):
        self.subscribers = {}
        self.published_events = []
        self.is_running = False
    
    async def start(self):
        self.is_running = True
    
    async def stop(self):
        self.is_running = False
    
    def subscribe(self, event_type: str, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event: Dict[str, Any]):
        self.published_events.append(event)
        event_type = event.get("type")
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                # 确保handler是异步的
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)


class MockDistributedEventProcessor:
    """模拟分布式事件处理器"""
    def __init__(self):
        self.processed_events = []
        self.processing_nodes = []
    
    async def process(self, event: Dict[str, Any]):
        self.processed_events.append(event)
        return {"status": "processed", "event_id": event.get("id")}
    
    async def distribute(self, events: List[Dict[str, Any]]):
        results = []
        for event in events:
            result = await self.process(event)
            results.append(result)
        return results


class TestEventSystemIntegration:
    """事件系统集成测试"""
    
    @pytest.fixture
    def event_bus(self):
        """创建事件总线"""
        bus = MockEventBus()
        asyncio.run(bus.start())
        return bus
    
    @pytest.fixture
    def distributed_processor(self):
        """创建分布式处理器"""
        return MockDistributedEventProcessor()
    
    @pytest.mark.asyncio
    async def test_event_publishing_and_subscription(self, event_bus):
        """测试事件发布和订阅"""
        received_events = []
        
        # 订阅事件
        async def event_handler(event):
            received_events.append(event)
        
        event_bus.subscribe("test.event", event_handler)
        event_bus.subscribe("another.event", event_handler)
        
        # 发布事件
        test_events = [
            {"type": "test.event", "data": "test1"},
            {"type": "test.event", "data": "test2"},
            {"type": "another.event", "data": "test3"}
        ]
        
        for event in test_events:
            await event_bus.publish(event)
        
        # 等待异步处理
        await asyncio.sleep(0.1)
        
        # 验证事件接收
        assert len(received_events) == 3
        assert received_events[0]["data"] == "test1"
        assert received_events[2]["type"] == "another.event"
    
    @pytest.mark.asyncio
    async def test_event_filtering_and_routing(self, event_bus):
        """测试事件过滤和路由"""
        agent_events = []
        system_events = []
        
        # 创建不同的处理器
        async def agent_handler(event):
            if event.get("category") == "agent":
                agent_events.append(event)
        
        async def system_handler(event):
            if event.get("category") == "system":
                system_events.append(event)
        
        # 订阅事件
        event_bus.subscribe("agent.message", agent_handler)
        event_bus.subscribe("system.alert", system_handler)
        
        # 发布混合事件
        await event_bus.publish({
            "type": "agent.message",
            "category": "agent",
            "data": "Agent message"
        })
        
        await event_bus.publish({
            "type": "system.alert",
            "category": "system",
            "data": "System alert"
        })
        
        await event_bus.publish({
            "type": "agent.message",
            "category": "system",  # 错误的类别
            "data": "Misrouted message"
        })
        
        await asyncio.sleep(0.1)
        
        # 验证路由
        assert len(agent_events) == 1
        assert len(system_events) == 1
        assert agent_events[0]["data"] == "Agent message"
        assert system_events[0]["data"] == "System alert"
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self, event_bus):
        """测试并发事件处理"""
        processing_times = []
        
        async def slow_handler(event):
            start_time = time.time()
            await asyncio.sleep(0.1)  # 模拟慢处理
            processing_times.append({
                "event_id": event.get("id"),
                "duration": time.time() - start_time
            })
        
        # 订阅处理器
        event_bus.subscribe("concurrent.event", slow_handler)
        
        # 并发发布事件
        tasks = []
        for i in range(10):
            event = {
                "type": "concurrent.event",
                "id": f"event_{i}",
                "data": f"Data {i}"
            }
            task = asyncio.create_task(event_bus.publish(event))
            tasks.append(task)
        
        # 等待所有发布完成
        await asyncio.gather(*tasks)
        
        # 等待处理完成
        await asyncio.sleep(0.2)
        
        # 验证并发处理
        assert len(processing_times) == 10
        # 并发处理应该比串行快
        total_time = max(pt["duration"] for pt in processing_times)
        assert total_time < 1.0  # 如果是串行会需要1秒
    
    @pytest.mark.asyncio
    async def test_event_priority_handling(self, event_bus):
        """测试事件优先级处理"""
        processed_order = []
        
        async def priority_handler(event):
            processed_order.append(event.get("priority", 0))
        
        event_bus.subscribe("priority.event", priority_handler)
        
        # 发布不同优先级的事件
        events = [
            {"type": "priority.event", "priority": 3},
            {"type": "priority.event", "priority": 1},
            {"type": "priority.event", "priority": 2}
        ]
        
        # 模拟优先级队列处理
        sorted_events = sorted(events, key=lambda x: x["priority"])
        for event in sorted_events:
            await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # 验证处理顺序
        assert processed_order == [1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_event_error_handling(self, event_bus):
        """测试事件错误处理"""
        successful_events = []
        error_events = []
        
        async def faulty_handler(event):
            if event.get("should_fail"):
                raise Exception("Intentional error")
            successful_events.append(event)
        
        async def error_handler(event):
            try:
                await faulty_handler(event)
            except Exception as e:
                error_events.append({
                    "event": event,
                    "error": str(e)
                })
        
        event_bus.subscribe("error.test", error_handler)
        
        # 发布混合事件
        await event_bus.publish({"type": "error.test", "should_fail": False})
        await event_bus.publish({"type": "error.test", "should_fail": True})
        await event_bus.publish({"type": "error.test", "should_fail": False})
        
        await asyncio.sleep(0.1)
        
        # 验证错误处理
        assert len(successful_events) == 2
        assert len(error_events) == 1
        assert "Intentional error" in error_events[0]["error"]
    
    @pytest.mark.asyncio
    async def test_distributed_event_processing(self, distributed_processor):
        """测试分布式事件处理"""
        # 创建大批量事件
        events = []
        for i in range(100):
            events.append({
                "id": f"event_{i}",
                "type": "distributed.event",
                "data": f"Data {i}"
            })
        
        # 分布式处理
        start_time = time.time()
        results = await distributed_processor.distribute(events)
        processing_time = time.time() - start_time
        
        # 验证处理结果
        assert len(results) == 100
        assert all(r["status"] == "processed" for r in results)
        
        # 验证处理效率
        assert processing_time < 1.0  # 应该快速处理
    
    @pytest.mark.asyncio
    async def test_event_replay_mechanism(self, event_bus):
        """测试事件重放机制"""
        event_history = []
        replayed_events = []
        
        async def history_handler(event):
            event_history.append(event)
        
        async def replay_handler(event):
            if event.get("is_replay"):
                replayed_events.append(event)
        
        event_bus.subscribe("history.event", history_handler)
        event_bus.subscribe("replay.event", replay_handler)
        
        # 记录原始事件
        original_events = [
            {"type": "history.event", "id": i, "data": f"Event {i}"}
            for i in range(5)
        ]
        
        for event in original_events:
            await event_bus.publish(event)
        
        # 重放事件
        for event in event_history:
            replay_event = {**event, "type": "replay.event", "is_replay": True}
            await event_bus.publish(replay_event)
        
        await asyncio.sleep(0.1)
        
        # 验证重放
        assert len(event_history) == 5
        assert len(replayed_events) == 5
        assert all(e["is_replay"] for e in replayed_events)
    
    @pytest.mark.asyncio
    async def test_event_batching_and_aggregation(self, event_bus):
        """测试事件批处理和聚合"""
        batched_events = []
        
        class BatchProcessor:
            def __init__(self, batch_size=5):
                self.batch = []
                self.batch_size = batch_size
            
            async def handle(self, event):
                self.batch.append(event)
                if len(self.batch) >= self.batch_size:
                    batched_events.append(list(self.batch))
                    self.batch = []
        
        processor = BatchProcessor()
        event_bus.subscribe("batch.event", processor.handle)
        
        # 发送多个事件
        for i in range(12):
            await event_bus.publish({
                "type": "batch.event",
                "id": i
            })
        
        await asyncio.sleep(0.1)
        
        # 验证批处理
        assert len(batched_events) == 2  # 12个事件，批大小5，应该有2个完整批
        assert len(batched_events[0]) == 5
        assert len(batched_events[1]) == 5
        assert len(processor.batch) == 2  # 剩余2个未满批
    
    @pytest.mark.asyncio
    async def test_event_circuit_breaker(self, event_bus):
        """测试事件熔断机制"""
        class CircuitBreaker:
            def __init__(self, threshold=3):
                self.failure_count = 0
                self.threshold = threshold
                self.is_open = False
                self.processed = []
            
            async def handle(self, event):
                if self.is_open:
                    return {"status": "circuit_open"}
                
                try:
                    if event.get("will_fail"):
                        self.failure_count += 1
                        if self.failure_count >= self.threshold:
                            self.is_open = True
                        raise Exception("Processing failed")
                    
                    self.processed.append(event)
                    self.failure_count = 0  # 重置失败计数
                    return {"status": "success"}
                except Exception:
                    return {"status": "failed"}
        
        breaker = CircuitBreaker()
        event_bus.subscribe("breaker.event", breaker.handle)
        
        # 发送事件触发熔断
        events = [
            {"type": "breaker.event", "will_fail": False},
            {"type": "breaker.event", "will_fail": True},
            {"type": "breaker.event", "will_fail": True},
            {"type": "breaker.event", "will_fail": True},  # 触发熔断
            {"type": "breaker.event", "will_fail": False},  # 被熔断阻止
        ]
        
        for event in events:
            await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # 验证熔断
        assert breaker.is_open
        assert len(breaker.processed) == 1  # 只有第一个成功
        assert breaker.failure_count >= 3