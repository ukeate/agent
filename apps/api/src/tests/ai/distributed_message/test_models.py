"""
分布式消息通信模型测试
"""

import pytest
import json
from datetime import datetime

from src.ai.distributed_message.models import (
    Message,
    MessageHeader,
    MessageType,
    MessagePriority,
    DeliveryMode,
    MessageHandler,
    StreamConfig,
    ConnectionMetrics,
    TopicConfig
)


class TestMessageHeader:
    """消息头测试"""
    
    def test_message_header_creation(self):
        """测试消息头创建"""
        header = MessageHeader(
            message_id="test-123",
            priority=MessagePriority.HIGH
        )
        
        assert header.message_id == "test-123"
        assert header.priority == MessagePriority.HIGH
        assert header.delivery_mode == DeliveryMode.AT_LEAST_ONCE
        assert header.retry_count == 0
        assert header.max_retries == 3
        assert isinstance(header.timestamp, datetime)


class TestMessage:
    """消息类测试"""
    
    def test_message_creation(self):
        """测试消息创建"""
        header = MessageHeader(
            message_id="test-123",
            priority=MessagePriority.HIGH
        )
        
        message = Message(
            header=header,
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.PING,
            payload={"test": "data"}
        )
        
        assert message.sender_id == "agent-1"
        assert message.receiver_id == "agent-2"
        assert message.message_type == MessageType.PING
        assert message.payload["test"] == "data"
        assert message.schema_version == "1.0"
        assert message.encoding == "json"
    
    def test_message_serialization(self):
        """测试消息序列化"""
        header = MessageHeader(
            message_id="test-123",
            priority=MessagePriority.HIGH
        )
        
        message = Message(
            header=header,
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.PING,
            payload={"test": "data"}
        )
        
        # 序列化
        serialized = message.to_bytes()
        assert isinstance(serialized, bytes)
        
        # 验证可以解析为JSON
        json_data = json.loads(serialized.decode('utf-8'))
        assert json_data["sender_id"] == "agent-1"
        assert json_data["message_type"] == "ping"
    
    def test_message_deserialization(self):
        """测试消息反序列化"""
        header = MessageHeader(
            message_id="test-123",
            priority=MessagePriority.HIGH
        )
        
        original_message = Message(
            header=header,
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.PING,
            payload={"test": "data"}
        )
        
        # 序列化后反序列化
        serialized = original_message.to_bytes()
        deserialized = Message.from_bytes(serialized)
        
        # 验证数据一致性
        assert deserialized.header.message_id == "test-123"
        assert deserialized.sender_id == "agent-1"
        assert deserialized.receiver_id == "agent-2"
        assert deserialized.message_type == MessageType.PING
        assert deserialized.payload["test"] == "data"
        assert deserialized.header.priority == MessagePriority.HIGH
    
    def test_broadcast_message(self):
        """测试广播消息"""
        header = MessageHeader(message_id="broadcast-123")
        
        message = Message(
            header=header,
            sender_id="agent-1",
            receiver_id=None,  # 广播消息
            message_type=MessageType.HEARTBEAT,
            payload={"status": "active"}
        )
        
        assert message.receiver_id is None
        
        # 序列化和反序列化测试
        serialized = message.to_bytes()
        deserialized = Message.from_bytes(serialized)
        assert deserialized.receiver_id is None


class TestMessageHandler:
    """消息处理器测试"""
    
    @pytest.mark.asyncio
    async def test_async_message_handler(self):
        """测试异步消息处理器"""
        handled_messages = []
        
        async def test_handler(message: Message):
            handled_messages.append(message)
            return "handled"
        
        handler = MessageHandler(
            message_type=MessageType.PING,
            handler=test_handler
        )
        
        # 创建测试消息
        header = MessageHeader(message_id="test-123")
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={"test": "data"}
        )
        
        # 处理消息
        result = await handler.handle_message(message)
        
        assert result == "handled"
        assert len(handled_messages) == 1
        assert handled_messages[0].header.message_id == "test-123"
        assert handler.stats["handled"] == 1
        assert handler.stats["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_sync_message_handler(self):
        """测试同步消息处理器"""
        handled_messages = []
        
        def sync_handler(message: Message):
            handled_messages.append(message)
            return "sync_handled"
        
        handler = MessageHandler(
            message_type=MessageType.PING,
            handler=sync_handler,
            is_async=False
        )
        
        # 创建测试消息
        header = MessageHeader(message_id="sync-test-123")
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={"test": "sync"}
        )
        
        # 处理消息
        result = await handler.handle_message(message)
        
        assert result == "sync_handled"
        assert len(handled_messages) == 1
        assert handler.stats["handled"] == 1
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self):
        """测试处理器错误处理"""
        async def error_handler(message: Message):
            raise ValueError("测试错误")
        
        handler = MessageHandler(
            message_type=MessageType.PING,
            handler=error_handler
        )
        
        header = MessageHeader(message_id="error-test")
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={}
        )
        
        # 处理应该抛出异常
        with pytest.raises(ValueError, match="测试错误"):
            await handler.handle_message(message)
        
        assert handler.stats["errors"] == 1
        assert handler.stats["handled"] == 0


class TestStreamConfig:
    """流配置测试"""
    
    def test_stream_config_creation(self):
        """测试流配置创建"""
        config = StreamConfig(
            name="TEST_STREAM",
            subjects=["test.>"],
            max_messages=1000,
            replicas=1
        )
        
        assert config.name == "TEST_STREAM"
        assert config.subjects == ["test.>"]
        assert config.max_messages == 1000
        assert config.replicas == 1
        assert config.storage == "file"
    
    def test_stream_config_to_nats(self):
        """测试转换为NATS配置"""
        config = StreamConfig(
            name="TEST_STREAM",
            subjects=["test.>"],
            max_messages=1000,
            replicas=1
        )
        
        nats_config = config.to_nats_config()
        
        assert nats_config["name"] == "TEST_STREAM"
        assert nats_config["subjects"] == ["test.>"]
        assert nats_config["max_msgs"] == 1000
        assert nats_config["num_replicas"] == 1


class TestConnectionMetrics:
    """连接指标测试"""
    
    def test_metrics_initialization(self):
        """测试指标初始化"""
        metrics = ConnectionMetrics()
        
        assert metrics.messages_sent == 0
        assert metrics.messages_received == 0
        assert metrics.messages_failed == 0
        assert metrics.bytes_sent == 0
        assert metrics.bytes_received == 0
        assert metrics.active_subscriptions == 0
        assert metrics.pending_requests == 0
    
    def test_metrics_update(self):
        """测试指标更新"""
        metrics = ConnectionMetrics()
        
        metrics.messages_sent = 10
        metrics.bytes_sent = 1024
        metrics.active_subscriptions = 5
        
        assert metrics.messages_sent == 10
        assert metrics.bytes_sent == 1024
        assert metrics.active_subscriptions == 5
    
    def test_metrics_reset(self):
        """测试指标重置"""
        metrics = ConnectionMetrics()
        
        # 设置一些值
        metrics.messages_sent = 10
        metrics.bytes_sent = 1024
        
        # 重置
        metrics.reset()
        
        assert metrics.messages_sent == 0
        assert metrics.bytes_sent == 0
    
    def test_metrics_to_dict(self):
        """测试指标转换为字典"""
        metrics = ConnectionMetrics()
        metrics.messages_sent = 5
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data["messages_sent"] == 5
        assert "messages_received" in data


class TestTopicConfig:
    """主题配置测试"""
    
    def test_topic_config_creation(self):
        """测试主题配置创建"""
        topics = TopicConfig.create_for_agent("agent-123", "test-cluster")
        
        assert topics.direct_messages == "agents.direct.agent-123"
        assert topics.broadcast == "agents.broadcast.test-cluster"
        assert topics.group_messages == "agents.group.test-cluster"
        assert topics.task_coordination == "agents.tasks.test-cluster"
        assert topics.resource_management == "agents.resources.test-cluster"
        assert topics.system_events == "agents.events.test-cluster"
        assert topics.data_streams == "agents.streams.test-cluster"