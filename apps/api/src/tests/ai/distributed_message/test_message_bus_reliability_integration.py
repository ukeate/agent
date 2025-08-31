"""
消息总线与可靠性管理器集成测试
测试MessageBus与ReliabilityManager的完整集成
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch

from src.ai.distributed_message.message_bus import DistributedMessageBus
from src.ai.distributed_message.models import MessageType, MessagePriority
from src.ai.distributed_message.reliability import RetryConfig, RetryPolicy


class TestMessageBusReliabilityIntegration:
    """消息总线可靠性集成测试"""
    
    @pytest.fixture  
    def message_bus(self):
        """创建消息总线实例"""
        with patch('src.ai.distributed_message.message_bus.NATSClient') as mock_client_class:
            # 模拟NATSClient
            mock_client = Mock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.disconnect = AsyncMock(return_value=True)
            # 创建带有 unsubscribe 异步方法的 mock subscription
            mock_subscription = Mock()
            mock_subscription.unsubscribe = AsyncMock()
            mock_client.subscribe = AsyncMock(return_value=mock_subscription)
            mock_client.publish = AsyncMock(return_value=True)
            mock_client.js_publish = AsyncMock(return_value=Mock(sequence=1))
            mock_client.js = Mock()
            mock_client.is_connected = Mock(return_value=True)
            mock_client.metrics = Mock(
                messages_sent=0,
                messages_received=0,
                bytes_sent=0,
                bytes_received=0,
                messages_failed=0
            )
            mock_client_class.return_value = mock_client
            
            bus = DistributedMessageBus(
                nats_servers=["nats://localhost:4222"],
                agent_id="test-agent",
                cluster_name="test-cluster"
            )
            
            # 设置回调以避免实际NATS操作
            bus.client = mock_client
            
            yield bus
    
    @pytest.mark.asyncio
    async def test_send_reliable_message_with_ack(self, message_bus):
        """测试发送需要确认的可靠消息"""
        await message_bus.connect()
        
        try:
            # 发送可靠消息
            message_id = await message_bus.send_reliable_message(
                receiver_id="worker-agent",
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "process_data", "data": "test"},
                require_ack=True
            )
            
            assert message_id is not None
            
            # 验证消息发送
            message_bus.client.js_publish.assert_called_once()
            
            # 检查可靠性管理器状态
            stats = message_bus.get_reliability_statistics()
            assert stats["pending_messages"] == 1
            assert stats["awaiting_ack"] == 1
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_send_reliable_message_fire_and_forget(self, message_bus):
        """测试发送不需要确认的可靠消息"""
        await message_bus.connect()
        
        try:
            # 发送不需要确认的消息
            message_id = await message_bus.send_reliable_message(
                receiver_id="worker-agent",
                message_type=MessageType.PING,
                payload={"timestamp": "2025-08-26T12:00:00"},
                require_ack=False
            )
            
            assert message_id is not None
            
            # 验证消息发送
            message_bus.client.publish.assert_called_once()
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_reliable_message_with_retry(self, message_bus):
        """测试带重试的可靠消息"""
        # 模拟发送失败
        message_bus.client.js_publish = AsyncMock(return_value=None)
        
        await message_bus.connect()
        
        try:
            # 发送带重试配置的消息
            retry_config = RetryConfig(
                policy=RetryPolicy.FIXED_INTERVAL,
                initial_delay=0.1,
                max_retries=2
            )
            
            message_id = await message_bus.send_reliable_message(
                receiver_id="worker-agent",
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "process_data"},
                require_ack=True,
                retry_config=retry_config
            )
            
            assert message_id is not None
            
            # 手动触发重试
            await asyncio.sleep(0.2)
            await message_bus.reliability_manager._process_retries()
            await asyncio.sleep(0.2)
            await message_bus.reliability_manager._process_retries()
            await asyncio.sleep(0.2)
            await message_bus.reliability_manager._process_retries()
            
            # 检查死信队列
            stats = message_bus.get_reliability_statistics()
            assert stats["dead_letter_queue_size"] == 1
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_message_acknowledgment(self, message_bus):
        """测试消息确认"""
        await message_bus.connect()
        
        try:
            # 发送需要确认的消息
            message_id = await message_bus.send_reliable_message(
                receiver_id="worker-agent",
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "process_data"},
                require_ack=True
            )
            
            assert message_id is not None
            
            # 确认消息
            ack_result = await message_bus.acknowledge_message(message_id)
            assert ack_result is True
            
            # 验证统计信息
            stats = message_bus.get_reliability_statistics()
            assert stats["messages_acknowledged"] == 1
            assert stats["awaiting_ack"] == 0
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_message_rejection(self, message_bus):
        """测试消息拒绝（NACK）"""
        await message_bus.connect()
        
        try:
            # 发送需要确认的消息
            message_id = await message_bus.send_reliable_message(
                receiver_id="worker-agent", 
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "process_data"},
                require_ack=True,
                retry_config=RetryConfig(
                    policy=RetryPolicy.FIXED_INTERVAL,
                    initial_delay=0.1,
                    max_retries=3
                )
            )
            
            assert message_id is not None
            
            # 拒绝消息
            message_bus.reject_message(message_id, "处理失败")
            
            # 验证消息回到重试队列
            stats = message_bus.get_reliability_statistics()
            assert stats["pending_messages"] == 1
            assert stats["awaiting_ack"] == 0
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_dead_letter_queue_integration(self, message_bus):
        """测试死信队列集成"""
        # 模拟发送总是失败 (both js_publish and publish)
        message_bus.client.js_publish = AsyncMock(return_value=None)
        message_bus.client.publish = AsyncMock(return_value=False)
        
        await message_bus.connect()
        
        try:
            # 发送消息
            message_id = await message_bus.send_reliable_message(
                receiver_id="worker-agent",
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "process_data"},
                require_ack=False,
                retry_config=RetryConfig(
                    policy=RetryPolicy.FIXED_INTERVAL,
                    initial_delay=0.1,
                    max_retries=2
                )
            )
            
            assert message_id is not None
            
            # 等待重试完成并进入死信队列
            await asyncio.sleep(0.5)
            for _ in range(4):  # 多次触发重试处理
                await message_bus.reliability_manager._process_retries()
                await asyncio.sleep(0.2)
            
            # 验证死信队列
            dlq_messages = message_bus.get_dead_letter_messages()
            assert len(dlq_messages) == 1
            assert dlq_messages[0]["message_id"] == message_id
            
            stats = message_bus.get_reliability_statistics()
            assert stats["messages_dead_lettered"] == 1
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_reliability_statistics(self, message_bus):
        """测试可靠性统计信息"""
        await message_bus.connect()
        
        try:
            # 发送多个消息
            message_ids = []
            for i in range(3):
                message_id = await message_bus.send_reliable_message(
                    receiver_id=f"worker-agent-{i}",
                    message_type=MessageType.PING,
                    payload={"index": i},
                    require_ack=True
                )
                message_ids.append(message_id)
            
            # 确认部分消息
            await message_bus.acknowledge_message(message_ids[0])
            await message_bus.acknowledge_message(message_ids[1])
            
            # 拒绝一个消息
            message_bus.reject_message(message_ids[2], "测试拒绝")
            
            # 验证统计信息
            stats = message_bus.get_reliability_statistics()
            assert stats["messages_acknowledged"] == 2
            assert stats["pending_messages"] == 1  # 被拒绝的消息回到pending
            assert stats["awaiting_ack"] == 0
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio 
    async def test_reliability_manager_lifecycle(self, message_bus):
        """测试可靠性管理器的生命周期"""
        # 验证初始状态
        assert not message_bus.reliability_manager.is_running
        
        # 连接应启动可靠性管理器
        await message_bus.connect()
        assert message_bus.reliability_manager.is_running
        
        # 断开连接应停止可靠性管理器
        await message_bus.disconnect()
        assert not message_bus.reliability_manager.is_running