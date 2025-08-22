"""
重构后分布式事件处理测试
验证重构后的组件化设计和方法拆分的正确性
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from src.ai.autogen.distributed_events_refactored import (
    DistributedEvent, EventSerializer, EventPublisher, EventConsumer,
    LoadBalancingStrategy, LoadBalancer, EventLoopProcessor,
    RefactoredDistributedEventBus
)


@pytest.fixture
def mock_redis():
    """创建模拟Redis客户端"""
    redis_client = AsyncMock()
    redis_client.xadd = AsyncMock(return_value=b'123456-0')
    redis_client.lpush = AsyncMock(return_value=1)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.brpop = AsyncMock(return_value=None)
    redis_client.xread = AsyncMock(return_value=[])
    return redis_client


@pytest.fixture
def sample_event():
    """创建样例分布式事件"""
    return DistributedEvent(
        event_id="test_event_001",
        event_type="task_completed",
        source_node="node_001",
        target_nodes=["node_002", "node_003"],
        payload={"task_id": "task_123", "result": "success"},
        timestamp=datetime.now(timezone.utc),
        priority=1,
        retry_count=0,
        max_retries=3
    )


@pytest.fixture
def sample_nodes():
    """创建样例节点信息"""
    return {
        "node_001": {"load": 0.9, "status": "active"},
        "node_002": {"load": 0.3, "status": "active"},
        "node_003": {"load": 0.7, "status": "active"},
        "node_004": {"load": 0.2, "status": "active"},
    }


class TestEventSerializer:
    """事件序列化器测试"""
    
    def test_serialize_event(self, sample_event):
        """测试事件序列化"""
        serialized = EventSerializer.serialize(sample_event)
        
        assert serialized["event_id"] == sample_event.event_id
        assert serialized["event_type"] == sample_event.event_type
        assert serialized["source_node"] == sample_event.source_node
        assert serialized["target_nodes"] == sample_event.target_nodes
        assert serialized["payload"] == sample_event.payload
        assert serialized["priority"] == sample_event.priority
        assert serialized["retry_count"] == sample_event.retry_count
        assert serialized["max_retries"] == sample_event.max_retries
        assert "timestamp" in serialized
    
    def test_deserialize_event(self, sample_event):
        """测试事件反序列化"""
        # 先序列化再反序列化
        serialized = EventSerializer.serialize(sample_event)
        deserialized = EventSerializer.deserialize(serialized)
        
        assert deserialized.event_id == sample_event.event_id
        assert deserialized.event_type == sample_event.event_type
        assert deserialized.source_node == sample_event.source_node
        assert deserialized.target_nodes == sample_event.target_nodes
        assert deserialized.payload == sample_event.payload
        assert deserialized.priority == sample_event.priority
        assert deserialized.retry_count == sample_event.retry_count
        assert deserialized.max_retries == sample_event.max_retries
        assert deserialized.timestamp == sample_event.timestamp


class TestEventPublisher:
    """事件发布器测试"""
    
    @pytest.fixture
    def event_publisher(self, mock_redis):
        """创建事件发布器"""
        return EventPublisher(mock_redis)
    
    @pytest.mark.asyncio
    async def test_publish_to_stream_success(self, event_publisher, sample_event, mock_redis):
        """测试成功发布到流"""
        result = await event_publisher.publish_to_stream(sample_event)
        
        assert result is True
        mock_redis.xadd.assert_called_once()
        
        # 验证调用参数
        call_args = mock_redis.xadd.call_args
        stream_key = call_args[0][0]
        event_data = call_args[0][1]
        
        assert stream_key == f"events:{sample_event.event_type}"
        assert event_data["event_id"] == sample_event.event_id
    
    @pytest.mark.asyncio
    async def test_publish_to_stream_failure(self, event_publisher, sample_event, mock_redis):
        """测试发布到流失败"""
        mock_redis.xadd.side_effect = Exception("Redis error")
        
        result = await event_publisher.publish_to_stream(sample_event)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_notify_target_nodes_success(self, event_publisher, sample_event, mock_redis):
        """测试成功通知目标节点"""
        stream_key = "events:task_completed"
        
        result = await event_publisher.notify_target_nodes(sample_event, stream_key)
        
        assert result is True
        # 应该为每个目标节点调用lpush和expire
        assert mock_redis.lpush.call_count == len(sample_event.target_nodes)
        assert mock_redis.expire.call_count == len(sample_event.target_nodes)
    
    @pytest.mark.asyncio
    async def test_send_single_notification_success(self, event_publisher, mock_redis):
        """测试发送单个通知成功"""
        target_node = "test_node"
        notification_data = {"test": "data"}
        
        result = await event_publisher._send_single_notification(target_node, notification_data)
        
        assert result is True
        mock_redis.lpush.assert_called_once()
        mock_redis.expire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_single_notification_failure(self, event_publisher, mock_redis):
        """测试发送单个通知失败"""
        mock_redis.lpush.side_effect = Exception("Connection error")
        
        result = await event_publisher._send_single_notification("test_node", {})
        
        assert result is False


class TestEventConsumer:
    """事件消费器测试"""
    
    @pytest.fixture
    def event_consumer(self, mock_redis):
        """创建事件消费器"""
        return EventConsumer(mock_redis, "test_node")
    
    @pytest.mark.asyncio
    async def test_wait_for_notification_success(self, event_consumer, mock_redis):
        """测试等待通知成功"""
        test_data = b'{"test": "notification"}'
        mock_redis.brpop.return_value = ("key", test_data)
        
        result = await event_consumer._wait_for_notification("test_key")
        
        assert result == test_data
        mock_redis.brpop.assert_called_once_with("test_key", timeout=1)
    
    @pytest.mark.asyncio
    async def test_wait_for_notification_timeout(self, event_consumer, mock_redis):
        """测试等待通知超时"""
        mock_redis.brpop.return_value = None
        
        result = await event_consumer._wait_for_notification("test_key")
        
        assert result is None
    
    def test_parse_notification_success(self, event_consumer):
        """测试解析通知成功"""
        notification_data = b'{"event_id": "123", "event_type": "test"}'
        
        result = event_consumer._parse_notification(notification_data)
        
        assert result == {"event_id": "123", "event_type": "test"}
    
    def test_parse_notification_failure(self, event_consumer):
        """测试解析通知失败"""
        invalid_data = b'invalid json'
        
        result = event_consumer._parse_notification(invalid_data)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fetch_event_from_stream_success(self, event_consumer, mock_redis, sample_event):
        """测试从流获取事件成功"""
        # 模拟Redis流返回数据
        event_data = EventSerializer.serialize(sample_event)
        mock_redis.xread.return_value = [
            ("stream_key", [(b"123456-0", event_data)])
        ]
        
        notification = {
            "event_id": sample_event.event_id,
            "stream_key": "test_stream"
        }
        
        result = await event_consumer._fetch_event_from_stream(notification)
        
        assert result is not None
        assert result.event_id == sample_event.event_id
        assert result.event_type == sample_event.event_type
    
    @pytest.mark.asyncio
    async def test_fetch_event_from_stream_not_found(self, event_consumer, mock_redis):
        """测试从流获取事件失败 - 未找到"""
        mock_redis.xread.return_value = []
        
        notification = {
            "event_id": "non_existent",
            "stream_key": "test_stream"
        }
        
        result = await event_consumer._fetch_event_from_stream(notification)
        
        assert result is None


class TestLoadBalancingStrategy:
    """负载均衡策略测试"""
    
    def test_calculate_average_load(self, sample_nodes):
        """测试计算平均负载"""
        avg_load = LoadBalancingStrategy.calculate_average_load(sample_nodes)
        
        expected_avg = (0.9 + 0.3 + 0.7 + 0.2) / 4
        assert avg_load == expected_avg
    
    def test_calculate_average_load_empty_nodes(self):
        """测试空节点列表的平均负载"""
        avg_load = LoadBalancingStrategy.calculate_average_load({})
        assert avg_load == 0.0
    
    def test_identify_high_load_nodes(self, sample_nodes):
        """测试识别高负载节点"""
        high_load_nodes = LoadBalancingStrategy.identify_high_load_nodes(sample_nodes)
        
        # 平均负载为0.525，阈值为0.525 * 1.2 = 0.63
        # node_001 (0.9) 和 node_003 (0.7) 应该被识别为高负载
        assert "node_001" in high_load_nodes
        assert "node_003" in high_load_nodes
        assert "node_002" not in high_load_nodes
        assert "node_004" not in high_load_nodes
    
    def test_identify_low_load_nodes(self, sample_nodes):
        """测试识别低负载节点"""
        low_load_nodes = LoadBalancingStrategy.identify_low_load_nodes(sample_nodes)
        
        # 平均负载为0.525，阈值为0.525 * 0.8 = 0.42
        # node_002 (0.3) 和 node_004 (0.2) 应该被识别为低负载
        assert "node_002" in low_load_nodes
        assert "node_004" in low_load_nodes
        assert "node_001" not in low_load_nodes
        assert "node_003" not in low_load_nodes


class TestLoadBalancer:
    """负载均衡器测试"""
    
    @pytest.fixture
    def load_balancer(self):
        """创建负载均衡器"""
        return LoadBalancer("test_node")
    
    def test_analyze_load_distribution(self, load_balancer, sample_nodes):
        """测试分析负载分布"""
        analysis = load_balancer._analyze_load_distribution(sample_nodes)
        
        assert "average_load" in analysis
        assert "high_load_nodes" in analysis
        assert "low_load_nodes" in analysis
        assert "total_nodes" in analysis
        assert "needs_rebalancing" in analysis
        
        assert analysis["total_nodes"] == len(sample_nodes)
        assert analysis["needs_rebalancing"] is True  # 有高低负载节点
    
    def test_create_rebalancing_plan_needed(self, load_balancer):
        """测试创建重平衡计划 - 需要重平衡"""
        load_analysis = {
            "needs_rebalancing": True,
            "high_load_nodes": ["node_001", "node_003"],
            "low_load_nodes": ["node_002", "node_004"]
        }
        
        plan = load_balancer._create_rebalancing_plan(load_analysis)
        
        assert "actions" in plan
        assert len(plan["actions"]) == 2  # 两个高负载节点
        assert plan["actions"][0]["type"] == "migrate_tasks"
        assert plan["actions"][0]["from_node"] == "node_001"
        assert plan["actions"][0]["to_node"] == "node_002"
    
    def test_create_rebalancing_plan_not_needed(self, load_balancer):
        """测试创建重平衡计划 - 不需要重平衡"""
        load_analysis = {
            "needs_rebalancing": False,
            "high_load_nodes": [],
            "low_load_nodes": []
        }
        
        plan = load_balancer._create_rebalancing_plan(load_analysis)
        
        assert plan["actions"] == []
        assert "reason" in plan
    
    @pytest.mark.asyncio
    async def test_rebalance_load_not_leader(self, load_balancer, sample_nodes):
        """测试非领导者节点重平衡负载"""
        result = await load_balancer.rebalance_load(sample_nodes, "follower")
        
        assert "error" in result
        assert "领导者" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_single_action(self, load_balancer):
        """测试执行单个重平衡动作"""
        action = {
            "type": "migrate_tasks",
            "from_node": "node_001",
            "to_node": "node_002",
            "task_count": "auto"
        }
        
        result = await load_balancer._execute_single_action(action)
        
        assert result["success"] is True
        assert "transferred_tasks" in result
        assert "execution_time" in result


class TestEventLoopProcessor:
    """事件循环处理器测试"""
    
    @pytest.fixture
    def event_processor(self, mock_redis):
        """创建事件循环处理器"""
        return EventLoopProcessor(mock_redis, "test_node")
    
    @pytest.mark.asyncio
    async def test_fetch_event_from_queue_success(self, event_processor, mock_redis):
        """测试从队列获取事件成功"""
        test_event_data = {"id": "test_event", "type": "test"}
        mock_redis.brpop.return_value = ("queue_key", json.dumps(test_event_data).encode())
        
        result = await event_processor._fetch_event_from_queue("test_queue")
        
        assert result == test_event_data
        mock_redis.brpop.assert_called_once_with("test_queue", timeout=1)
    
    @pytest.mark.asyncio
    async def test_fetch_event_from_queue_timeout(self, event_processor, mock_redis):
        """测试从队列获取事件超时"""
        mock_redis.brpop.return_value = None
        
        result = await event_processor._fetch_event_from_queue("test_queue")
        
        assert result is None
    
    def test_reconstruct_event_success(self, event_processor):
        """测试重建事件对象成功"""
        event_data = {
            "id": "test_event",
            "type": "test_type",
            "source": "test_source",
            "data": {"key": "value"}
        }
        
        result = event_processor._reconstruct_event(event_data)
        
        assert result["id"] == "test_event"
        assert result["type"] == "test_type"
        assert result["source"] == "test_source"
        assert result["data"] == {"key": "value"}
    
    def test_reconstruct_event_minimal_data(self, event_processor):
        """测试重建事件对象 - 最少数据"""
        event_data = {}
        
        result = event_processor._reconstruct_event(event_data)
        
        assert result["id"] is None
        assert result["type"] == "unknown"
        assert result["source"] == ""
        assert result["data"] == {}
    
    @pytest.mark.asyncio
    async def test_process_single_event_success(self, event_processor):
        """测试处理单个事件成功"""
        event = {"id": "test_event", "type": "test"}
        initial_received = event_processor.stats["events_received"]
        initial_processed = event_processor.stats["events_processed"]
        
        # Mock business logic handler
        event_processor._handle_event_business_logic = AsyncMock()
        
        await event_processor._process_single_event(event)
        
        assert event_processor.stats["events_received"] == initial_received + 1
        assert event_processor.stats["events_processed"] == initial_processed + 1
        event_processor._handle_event_business_logic.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_process_single_event_failure(self, event_processor):
        """测试处理单个事件失败"""
        event = {"id": "test_event", "type": "test"}
        initial_failed = event_processor.stats["events_failed"]
        
        # Mock business logic handler to raise exception
        event_processor._handle_event_business_logic = AsyncMock(side_effect=Exception("Test error"))
        
        await event_processor._process_single_event(event)
        
        assert event_processor.stats["events_failed"] == initial_failed + 1
    
    def test_get_processing_stats(self, event_processor):
        """测试获取处理统计"""
        stats = event_processor.get_processing_stats()
        
        assert "events_received" in stats
        assert "events_processed" in stats
        assert "events_failed" in stats
        assert isinstance(stats, dict)


class TestRefactoredDistributedEventBus:
    """重构后的分布式事件总线测试"""
    
    @pytest.fixture
    def event_bus(self, mock_redis):
        """创建重构后的事件总线"""
        return RefactoredDistributedEventBus(mock_redis, "test_node")
    
    @pytest.mark.asyncio
    async def test_startup_shutdown(self, event_bus):
        """测试启动和关闭"""
        # Mock the component start/stop methods
        event_bus.event_consumer.start_listening = AsyncMock()
        event_bus.event_consumer.stop_listening = AsyncMock()
        event_bus.event_processor.start_processing = AsyncMock()
        event_bus.event_processor.stop_processing = AsyncMock()
        
        await event_bus.start()
        assert event_bus.running is True
        
        await event_bus.stop()
        assert event_bus.running is False
        
        # Verify components were started and stopped
        event_bus.event_consumer.start_listening.assert_called_once()
        event_bus.event_consumer.stop_listening.assert_called_once()
        event_bus.event_processor.start_processing.assert_called_once()
        event_bus.event_processor.stop_processing.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_success(self, event_bus, sample_event):
        """测试发布事件成功"""
        # Mock the publisher methods
        event_bus.event_publisher.publish_to_stream = AsyncMock(return_value=True)
        event_bus.event_publisher.notify_target_nodes = AsyncMock(return_value=True)
        
        result = await event_bus.publish(sample_event)
        
        assert result is True
        event_bus.event_publisher.publish_to_stream.assert_called_once_with(sample_event)
        event_bus.event_publisher.notify_target_nodes.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_stream_failure(self, event_bus, sample_event):
        """测试发布事件失败 - 流发布失败"""
        event_bus.event_publisher.publish_to_stream = AsyncMock(return_value=False)
        
        result = await event_bus.publish(sample_event)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_subscribe(self, event_bus):
        """测试订阅事件"""
        handler = AsyncMock()
        
        await event_bus.subscribe("test_event", handler)
        
        assert "test_event" in event_bus.subscribers
        assert handler in event_bus.subscribers["test_event"]
    
    @pytest.mark.asyncio
    async def test_rebalance_cluster_load(self, event_bus, sample_nodes):
        """测试重平衡集群负载"""
        # Mock the load balancer
        expected_result = {"status": "completed"}
        event_bus.load_balancer.rebalance_load = AsyncMock(return_value=expected_result)
        
        result = await event_bus.rebalance_cluster_load(sample_nodes, "leader")
        
        assert result == expected_result
        event_bus.load_balancer.rebalance_load.assert_called_once_with(sample_nodes, "leader")
    
    def test_get_stats(self, event_bus):
        """测试获取统计信息"""
        # Mock the processor stats
        event_bus.event_processor.get_processing_stats = Mock(return_value={"processed": 10})
        
        stats = event_bus.get_stats()
        
        assert "node_id" in stats
        assert "running" in stats
        assert "processing_stats" in stats
        assert "subscribers" in stats
        assert stats["node_id"] == "test_node"


@pytest.mark.integration
class TestRefactoredIntegration:
    """重构后的集成测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_event_flow(self, mock_redis):
        """测试端到端事件流"""
        # 创建事件总线
        event_bus = RefactoredDistributedEventBus(mock_redis, "integration_node")
        
        # Mock组件方法
        event_bus.event_consumer.start_listening = AsyncMock()
        event_bus.event_consumer.stop_listening = AsyncMock()
        event_bus.event_processor.start_processing = AsyncMock()
        event_bus.event_processor.stop_processing = AsyncMock()
        
        # 启动总线
        await event_bus.start()
        
        try:
            # 创建测试事件
            test_event = DistributedEvent(
                event_id="integration_test",
                event_type="integration_event",
                source_node="integration_node",
                target_nodes=["target_node"],
                payload={"test": "data"},
                timestamp=datetime.now(timezone.utc)
            )
            
            # Mock发布组件
            event_bus.event_publisher.publish_to_stream = AsyncMock(return_value=True)
            event_bus.event_publisher.notify_target_nodes = AsyncMock(return_value=True)
            
            # 发布事件
            result = await event_bus.publish(test_event)
            assert result is True
            
            # 验证组件交互
            event_bus.event_publisher.publish_to_stream.assert_called_once()
            event_bus.event_publisher.notify_target_nodes.assert_called_once()
            
        finally:
            await event_bus.stop()


if __name__ == "__main__":
    pytest.main([__file__])