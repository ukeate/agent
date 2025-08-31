"""
消息可靠性保证测试
"""

import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timedelta

from src.ai.distributed_message.reliability import (
    ReliabilityManager, ReliableMessage, RetryConfig, RetryPolicy,
    MessageStatus, DeadLetterQueueConfig
)
from src.ai.distributed_message.models import Message, MessageHeader, MessageType, MessagePriority


class TestRetryConfig:
    """重试配置测试"""
    
    def test_default_retry_config(self):
        """测试默认重试配置"""
        config = RetryConfig()
        
        assert config.policy == RetryPolicy.EXPONENTIAL_BACKOFF
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
    
    def test_custom_retry_config(self):
        """测试自定义重试配置"""
        config = RetryConfig(
            policy=RetryPolicy.FIXED_INTERVAL,
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            backoff_factor=1.5,
            jitter=False
        )
        
        assert config.policy == RetryPolicy.FIXED_INTERVAL
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False


class TestReliableMessage:
    """可靠消息测试"""
    
    def test_create_reliable_message(self):
        """测试创建可靠消息"""
        header = MessageHeader(message_id=str(uuid.uuid4()))
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={"test": "data"},
            topic="test.topic"
        )
        
        retry_config = RetryConfig(max_retries=5)
        reliable_msg = ReliableMessage(
            message=message,
            message_id="reliable-123",
            retry_config=retry_config
        )
        
        assert reliable_msg.message == message
        assert reliable_msg.message_id == "reliable-123"
        assert reliable_msg.status == MessageStatus.PENDING
        assert reliable_msg.retry_count == 0
        assert reliable_msg.retry_config.max_retries == 5
        assert reliable_msg.can_retry()
    
    def test_retry_logic(self):
        """测试重试逻辑"""
        header = MessageHeader(message_id=str(uuid.uuid4()))
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver", 
            message_type=MessageType.PING,
            payload={},
            topic="test"
        )
        
        retry_config = RetryConfig(max_retries=3)
        reliable_msg = ReliableMessage(
            message=message,
            message_id="test-retry",
            retry_config=retry_config
        )
        
        # 初始状态
        assert reliable_msg.can_retry()
        assert reliable_msg.should_retry_now()
        
        # 记录失败尝试（第一次失败不增加retry_count）
        reliable_msg.record_attempt(False, "Network error")
        assert reliable_msg.status == MessageStatus.RETRYING
        assert reliable_msg.retry_count == 0  # 第一次失败不计入retry_count
        assert len(reliable_msg.error_history) == 1
        assert "Network error" in reliable_msg.error_history
        
        # 再次失败（现在增加retry_count）
        reliable_msg.record_attempt(False, "Timeout")
        assert reliable_msg.retry_count == 1
        assert reliable_msg.can_retry()
        
        # 第三次失败
        reliable_msg.record_attempt(False, "Another error")
        assert reliable_msg.retry_count == 2
        assert reliable_msg.can_retry()
        
        # 最后一次失败
        reliable_msg.record_attempt(False, "Final error")
        assert reliable_msg.retry_count == 3
        assert not reliable_msg.can_retry()
        assert reliable_msg.status == MessageStatus.DEAD_LETTER
    
    def test_retry_time_calculation(self):
        """测试重试时间计算"""
        header = MessageHeader(message_id=str(uuid.uuid4()))
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={},
            topic="test"
        )
        
        # 固定间隔重试
        config = RetryConfig(
            policy=RetryPolicy.FIXED_INTERVAL,
            initial_delay=2.0,
            jitter=False
        )
        
        reliable_msg = ReliableMessage(
            message=message,
            message_id="test-fixed",
            retry_config=config
        )
        
        # 第一次重试
        reliable_msg.record_attempt(False)
        next_retry = reliable_msg.calculate_next_retry_time()
        expected_delay = timedelta(seconds=2.0)
        actual_delay = next_retry - datetime.now()
        
        # 允许小的时间差
        assert abs(actual_delay.total_seconds() - expected_delay.total_seconds()) < 0.1
        
        # 指数退避重试
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            initial_delay=1.0,
            backoff_factor=2.0,
            jitter=False
        )
        
        reliable_msg.retry_config = config
        reliable_msg.retry_count = 2  # 第三次重试
        
        next_retry = reliable_msg.calculate_next_retry_time()
        expected_delay = timedelta(seconds=4.0)  # 1.0 * 2^2
        actual_delay = next_retry - datetime.now()
        
        assert abs(actual_delay.total_seconds() - expected_delay.total_seconds()) < 0.1
    
    def test_message_expiration(self):
        """测试消息过期"""
        header = MessageHeader(message_id=str(uuid.uuid4()), ttl=60)
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={},
            topic="test"
        )
        
        # 创建已过期的消息（创建时间设为过去）
        reliable_msg = ReliableMessage(
            message=message,
            message_id="expired-msg",
            created_at=datetime.now() - timedelta(seconds=120)  # 2分钟前
        )
        
        assert reliable_msg.is_expired()
        assert not reliable_msg.is_expired(180)  # 3分钟TTL时未过期


class TestReliabilityManager:
    """可靠性管理器测试"""
    
    @pytest.fixture
    def reliability_manager(self):
        """创建可靠性管理器"""
        config = RetryConfig(max_retries=3, initial_delay=0.1, jitter=False)
        dlq_config = DeadLetterQueueConfig(max_size=100, retention_hours=1)
        
        manager = ReliabilityManager(
            default_retry_config=config,
            dlq_config=dlq_config,
            ack_timeout=5.0
        )
        return manager
    
    @pytest.mark.asyncio
    async def test_send_reliable_message_success(self, reliability_manager):
        """测试成功发送可靠消息"""
        # 设置发送回调（成功）
        async def mock_send_callback(message):
            return True
        
        reliability_manager.set_send_callback(mock_send_callback)
        await reliability_manager.start()
        
        try:
            # 创建测试消息
            header = MessageHeader(message_id=str(uuid.uuid4()))
            message = Message(
                header=header,
                sender_id="test-sender",
                receiver_id="test-receiver",
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "test"},
                topic="test.topic"
            )
            
            # 发送消息
            message_id = await reliability_manager.send_reliable_message(
                message=message,
                require_ack=False  # 不需要确认
            )
            
            assert message_id is not None
            assert len(message_id) > 0
            
            # 验证统计信息
            stats = reliability_manager.get_statistics()
            assert stats["messages_sent"] == 1
            assert stats["acknowledged_messages"] == 1  # 不需要确认时直接标记为成功
        
        finally:
            await reliability_manager.stop()
    
    @pytest.mark.asyncio
    async def test_send_reliable_message_with_ack(self, reliability_manager):
        """测试需要确认的可靠消息"""
        # 设置发送回调（成功）
        async def mock_send_callback(message):
            return True
        
        reliability_manager.set_send_callback(mock_send_callback)
        await reliability_manager.start()
        
        try:
            # 创建测试消息
            header = MessageHeader(message_id=str(uuid.uuid4()))
            message = Message(
                header=header,
                sender_id="test-sender",
                receiver_id="test-receiver",
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "test"},
                topic="test.topic"
            )
            
            # 发送消息（需要确认）
            message_id = await reliability_manager.send_reliable_message(
                message=message,
                require_ack=True
            )
            
            # 验证消息在等待确认队列中
            assert message_id in reliability_manager.awaiting_ack
            
            # 确认消息
            result = reliability_manager.acknowledge_message(message_id)
            assert result is True
            
            # 验证消息移到已确认队列
            assert message_id not in reliability_manager.awaiting_ack
            assert message_id in reliability_manager.acknowledged_messages
            
            # 验证统计信息
            stats = reliability_manager.get_statistics()
            assert stats["messages_acknowledged"] == 1
        
        finally:
            await reliability_manager.stop()
    
    @pytest.mark.asyncio
    async def test_message_retry_mechanism(self, reliability_manager):
        """测试消息重试机制"""
        # 设置发送回调（前两次失败，第三次成功）
        call_count = 0
        
        async def mock_send_callback(message):
            nonlocal call_count
            call_count += 1
            return call_count >= 3  # 第三次调用时返回True
        
        reliability_manager.set_send_callback(mock_send_callback)
        await reliability_manager.start()
        
        try:
            # 创建测试消息
            header = MessageHeader(message_id=str(uuid.uuid4()))
            message = Message(
                header=header,
                sender_id="test-sender",
                receiver_id="test-receiver",
                message_type=MessageType.PING,
                payload={},
                topic="test.topic"
            )
            
            # 发送消息
            message_id = await reliability_manager.send_reliable_message(
                message=message,
                require_ack=False
            )
            
            # 等待重试处理
            await asyncio.sleep(1.5)  # 等待重试
            
            # 手动处理重试队列以确保重试执行
            await reliability_manager._process_retries()
            await asyncio.sleep(0.1)  # 等待第二次重试的延迟时间
            await reliability_manager._process_retries()
            
            # 验证最终成功
            stats = reliability_manager.get_statistics()
            assert call_count >= 3  # 应该至少调用了3次
            # 验证消息最终成功（在已确认队列中）
            assert len(reliability_manager.acknowledged_messages) >= 1
            
        finally:
            await reliability_manager.stop()
    
    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, reliability_manager):
        """测试死信队列"""
        # 设置发送回调（总是失败）
        async def mock_send_callback(message):
            return False
        
        reliability_manager.set_send_callback(mock_send_callback)
        await reliability_manager.start()
        
        try:
            # 创建测试消息
            header = MessageHeader(message_id=str(uuid.uuid4()))
            message = Message(
                header=header,
                sender_id="test-sender",
                receiver_id="test-receiver",
                message_type=MessageType.PING,
                payload={"test": "data"},
                topic="test.topic"
            )
            
            # 发送消息，使用很短的重试间隔
            message_id = await reliability_manager.send_reliable_message(
                message=message,
                require_ack=False,
                retry_config=RetryConfig(
                    policy=RetryPolicy.FIXED_INTERVAL,
                    initial_delay=0.1,  # 100ms 重试间隔
                    max_retries=3
                )
            )
            
            # 等待所有重试完成，手动触发重试处理
            await asyncio.sleep(0.5)
            await reliability_manager._process_retries()  # 第一次重试
            await asyncio.sleep(0.2)
            await reliability_manager._process_retries()  # 第二次重试
            await asyncio.sleep(0.2)
            await reliability_manager._process_retries()  # 第三次重试
            
            # 验证消息进入死信队列
            stats = reliability_manager.get_statistics()
            assert stats["dead_letter_queue_size"] == 1
            assert stats["messages_dead_lettered"] == 1
            
            # 获取死信队列消息
            dlq_messages = reliability_manager.get_dead_letter_messages()
            assert len(dlq_messages) == 1
            assert dlq_messages[0]["message_id"] == message_id
            assert dlq_messages[0]["retry_count"] == 3  # 重试配置的max_retries是3
            
        finally:
            await reliability_manager.stop()
    
    @pytest.mark.asyncio 
    async def test_nack_message(self, reliability_manager):
        """测试消息否认"""
        # 设置发送回调（成功）
        async def mock_send_callback(message):
            return True
        
        reliability_manager.set_send_callback(mock_send_callback)
        await reliability_manager.start()
        
        try:
            # 创建测试消息
            header = MessageHeader(message_id=str(uuid.uuid4()))
            message = Message(
                header=header,
                sender_id="test-sender",
                receiver_id="test-receiver",
                message_type=MessageType.TASK_REQUEST,
                payload={},
                topic="test.topic"
            )
            
            # 发送消息（需要确认）
            message_id = await reliability_manager.send_reliable_message(
                message=message,
                require_ack=True
            )
            
            # 否认消息
            reliability_manager.nack_message(message_id, "处理失败")
            
            # 验证消息回到pending队列进行重试
            assert message_id in reliability_manager.pending_messages
            reliable_msg = reliability_manager.pending_messages[message_id]
            assert reliable_msg.status == MessageStatus.RETRYING
            assert "处理失败" in reliable_msg.error_history
        
        finally:
            await reliability_manager.stop()
    
    @pytest.mark.asyncio
    async def test_ack_timeout(self, reliability_manager):
        """测试确认超时"""
        # 设置很短的确认超时时间
        reliability_manager.ack_timeout = 0.1
        
        # 设置发送回调（成功）
        async def mock_send_callback(message):
            return True
        
        reliability_manager.set_send_callback(mock_send_callback)
        await reliability_manager.start()
        
        try:
            # 创建测试消息
            header = MessageHeader(message_id=str(uuid.uuid4()))
            message = Message(
                header=header,
                sender_id="test-sender",
                receiver_id="test-receiver",
                message_type=MessageType.PING,
                payload={},
                topic="test.topic"
            )
            
            # 发送消息（需要确认）
            message_id = await reliability_manager.send_reliable_message(
                message=message,
                require_ack=True
            )
            
            # 等待确认超时
            await asyncio.sleep(0.2)
            
            # 手动触发超时检查
            await reliability_manager._check_ack_timeouts()
            
            # 验证消息因超时被移回pending队列
            assert message_id not in reliability_manager.awaiting_ack
            
        finally:
            await reliability_manager.stop()
    
    def test_replay_dead_letter_message(self, reliability_manager):
        """测试重放死信消息"""
        # 手动添加死信消息
        header = MessageHeader(message_id=str(uuid.uuid4()))
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={},
            topic="test"
        )
        
        reliable_msg = ReliableMessage(
            message=message,
            message_id="dead-msg-123"
        )
        reliable_msg.status = MessageStatus.DEAD_LETTER
        reliable_msg.retry_count = 3
        reliable_msg.error_history.append("Test error")
        
        reliability_manager.dead_letter_queue["dead-msg-123"] = reliable_msg
        
        # 重放消息
        result = reliability_manager.replay_dead_letter_message("dead-msg-123")
        assert result is True
        
        # 验证消息被移回pending队列并重置状态
        assert "dead-msg-123" not in reliability_manager.dead_letter_queue
        assert "dead-msg-123" in reliability_manager.pending_messages
        
        replayed_msg = reliability_manager.pending_messages["dead-msg-123"]
        assert replayed_msg.status == MessageStatus.PENDING
        assert replayed_msg.retry_count == 0
        assert len(replayed_msg.error_history) == 0
    
    def test_get_message_status(self, reliability_manager):
        """测试获取消息状态"""
        # 添加测试消息到不同的队列
        header = MessageHeader(message_id=str(uuid.uuid4()))
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={},
            topic="test"
        )
        
        # Pending消息
        pending_msg = ReliableMessage(message=message, message_id="pending-123")
        reliability_manager.pending_messages["pending-123"] = pending_msg
        
        # 等待确认消息
        ack_msg = ReliableMessage(message=message, message_id="ack-456")
        ack_msg.status = MessageStatus.SENT
        reliability_manager.awaiting_ack["ack-456"] = ack_msg
        
        # 已确认消息
        confirmed_msg = ReliableMessage(message=message, message_id="confirmed-789")
        confirmed_msg.status = MessageStatus.ACKNOWLEDGED
        reliability_manager.acknowledged_messages["confirmed-789"] = confirmed_msg
        
        # 获取消息状态
        pending_status = reliability_manager.get_message_status("pending-123")
        assert pending_status is not None
        assert pending_status["status"] == "PENDING"
        assert pending_status["location"] == "pending"
        assert pending_status["retry_count"] == 0
        
        ack_status = reliability_manager.get_message_status("ack-456")
        assert ack_status is not None
        assert ack_status["status"] == "SENT"
        assert ack_status["location"] == "awaiting_ack"
        
        confirmed_status = reliability_manager.get_message_status("confirmed-789")
        assert confirmed_status is not None
        assert confirmed_status["status"] == "ACKNOWLEDGED"
        assert confirmed_status["location"] == "acknowledged"
        
        # 不存在的消息
        missing_status = reliability_manager.get_message_status("missing-999")
        assert missing_status is None
    
    def test_statistics(self, reliability_manager):
        """测试统计信息"""
        # 添加一些测试数据
        reliability_manager.stats["messages_sent"] = 10
        reliability_manager.stats["messages_acknowledged"] = 8
        reliability_manager.stats["messages_failed"] = 1
        reliability_manager.stats["messages_dead_lettered"] = 1
        
        # 添加一些消息到各个队列
        for i in range(3):
            msg = ReliableMessage(
                message=Mock(),
                message_id=f"pending-{i}"
            )
            reliability_manager.pending_messages[f"pending-{i}"] = msg
        
        for i in range(2):
            msg = ReliableMessage(
                message=Mock(),
                message_id=f"ack-{i}"
            )
            reliability_manager.awaiting_ack[f"ack-{i}"] = msg
        
        stats = reliability_manager.get_statistics()
        
        assert stats["pending_messages"] == 3
        assert stats["awaiting_ack"] == 2
        assert stats["messages_sent"] == 10
        assert stats["messages_acknowledged"] == 8
        assert stats["messages_failed"] == 1
        assert stats["messages_dead_lettered"] == 1
        assert stats["is_running"] is False
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_messages(self, reliability_manager):
        """测试清理过期消息"""
        # 设置短的保留时间
        reliability_manager.dlq_config.retention_hours = 0.001  # ~3.6秒
        
        # 添加过期的死信消息
        old_time = datetime.now() - timedelta(hours=1)
        header = MessageHeader(message_id=str(uuid.uuid4()))
        message = Message(
            header=header,
            sender_id="sender",
            receiver_id="receiver",
            message_type=MessageType.PING,
            payload={},
            topic="test"
        )
        
        expired_msg = ReliableMessage(
            message=message,
            message_id="expired-123",
            created_at=old_time
        )
        expired_msg.status = MessageStatus.DEAD_LETTER
        reliability_manager.dead_letter_queue["expired-123"] = expired_msg
        
        # 添加未过期的消息
        fresh_msg = ReliableMessage(
            message=message,
            message_id="fresh-456"
        )
        fresh_msg.status = MessageStatus.DEAD_LETTER
        reliability_manager.dead_letter_queue["fresh-456"] = fresh_msg
        
        # 执行清理
        await reliability_manager._cleanup_expired_messages()
        
        # 验证过期消息被清理
        assert "expired-123" not in reliability_manager.dead_letter_queue
        assert "fresh-456" in reliability_manager.dead_letter_queue