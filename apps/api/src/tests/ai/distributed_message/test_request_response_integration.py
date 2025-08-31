"""
请求-响应机制集成测试
测试MessageBus与RequestResponseManager的集成
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.ai.distributed_message.message_bus import DistributedMessageBus
from src.ai.distributed_message.models import Message, MessageHeader, MessageType, MessagePriority
from src.ai.distributed_message.protocol import MessageProtocol


class TestRequestResponseIntegration:
    """请求-响应机制集成测试"""
    
    @pytest.fixture
    def message_bus(self):
        """创建消息总线实例"""
        with patch('src.ai.distributed_message.message_bus.NATSClient') as mock_client_class:
            # 模拟NATSClient
            mock_client = Mock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.disconnect = AsyncMock(return_value=True)
            mock_client.subscribe = AsyncMock(return_value=Mock())
            mock_client.publish = AsyncMock()
            mock_client.js_publish = AsyncMock()
            mock_client.js = Mock()
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
    async def test_register_request_handler(self, message_bus):
        """测试注册请求处理器"""
        handler_called = False
        
        async def test_handler(message: Message):
            nonlocal handler_called
            handler_called = True
            return {"processed": True, "task_id": message.payload.get("task_id")}
        
        # 注册请求处理器
        message_bus.register_request_handler(
            MessageType.TASK_REQUEST,
            test_handler
        )
        
        # 验证处理器已注册
        assert MessageType.TASK_REQUEST in message_bus.request_response_manager.request_handlers
        handler_info = message_bus.request_response_manager.request_handlers[MessageType.TASK_REQUEST]
        assert handler_info['handler'] == test_handler
        assert handler_info['is_async'] is True
    
    @pytest.mark.asyncio
    async def test_register_response_callback(self, message_bus):
        """测试注册响应回调"""
        callback_called = False
        
        def test_callback(message: Message):
            nonlocal callback_called
            callback_called = True
        
        # 注册响应回调
        message_bus.register_response_callback(
            MessageType.ACK,
            test_callback
        )
        
        # 验证回调已注册
        assert MessageType.ACK in message_bus.request_response_manager.response_callbacks
        assert message_bus.request_response_manager.response_callbacks[MessageType.ACK] == test_callback
    
    @pytest.mark.asyncio
    async def test_send_request_with_manager(self, message_bus):
        """测试通过RequestResponseManager发送请求"""
        # 模拟响应处理
        async def mock_response_handler():
            await asyncio.sleep(0.1)  # 模拟网络延迟
            
            # 创建响应消息（需要匹配请求的correlation_id）
            # 实际情况下这会通过NATS接收到
            correlation_id = None
            
            # 获取发送的消息中的correlation_id
            if message_bus.client.js_publish.called:
                call_args = message_bus.client.js_publish.call_args
                message_data = call_args[1]['data']
                message_obj = Message.from_bytes(message_data)
                correlation_id = message_obj.header.correlation_id
            
            if correlation_id:
                response_header = MessageHeader(
                    message_id="response-123",
                    correlation_id=correlation_id
                )
                
                response_message = Message(
                    header=response_header,
                    sender_id="responder-agent",
                    receiver_id="test-agent",
                    message_type=MessageType.ACK,
                    payload={"result": "success", "processed": True},
                    topic="reply_topic"
                )
                
                # 模拟响应到达
                message_bus.request_response_manager.handle_response(response_message)
        
        # 启动响应处理任务
        response_task = asyncio.create_task(mock_response_handler())
        
        # 发送请求
        task_payload = MessageProtocol.create_task_request(
            task_id="task-123",
            task_type="data_processing",
            task_data={"input": "test data"},
            priority=8
        )
        
        response = await message_bus.send_request(
            receiver_id="worker-agent",
            message_type=MessageType.TASK_REQUEST,
            payload=task_payload,
            timeout=5.0
        )
        
        # 等待响应处理完成
        await response_task
        
        # 验证请求发送
        message_bus.client.js_publish.assert_called_once()
        
        # 验证响应
        assert response is not None
        assert response.message_type == MessageType.ACK
        assert response.payload["result"] == "success"
        assert response.payload["processed"] is True
    
    @pytest.mark.asyncio
    async def test_handle_incoming_request(self, message_bus):
        """测试处理传入的请求"""
        # 注册请求处理器
        processed_messages = []
        
        async def task_handler(message: Message):
            processed_messages.append(message)
            return {
                "result": "task_completed",
                "task_id": message.payload.get("task_id"),
                "processed_at": "2025-08-26T12:00:00"
            }
        
        message_bus.register_request_handler(MessageType.TASK_REQUEST, task_handler)
        
        # 创建传入的请求消息
        request_header = MessageHeader(
            message_id="request-456",
            correlation_id="corr-789",
            reply_to="agents.direct.requester-agent.reply.corr-789"
        )
        
        task_payload = MessageProtocol.create_task_request(
            task_id="incoming-task-001",
            task_type="image_analysis", 
            task_data={"image_url": "http://example.com/image.jpg"},
            priority=7
        )
        
        request_message = Message(
            header=request_header,
            sender_id="requester-agent",
            receiver_id="test-agent",
            message_type=MessageType.TASK_REQUEST,
            payload=task_payload,
            topic="agents.direct.test-agent"
        )
        
        # 模拟传入消息处理（绕过NATS）
        mock_msg = Mock()
        mock_msg.data = request_message.to_bytes()
        
        # 处理消息
        await message_bus._handle_direct_message(mock_msg)
        
        # 验证请求被处理
        assert len(processed_messages) == 1
        assert processed_messages[0].payload["task_id"] == "incoming-task-001"
        assert processed_messages[0].payload["task_type"] == "image_analysis"
        
        # 验证回复被发送
        message_bus.client.publish.assert_called_once()
        
        # 验证回复内容
        reply_call = message_bus.client.publish.call_args
        reply_subject = reply_call[1]['subject']
        reply_data = reply_call[1]['data']
        
        assert reply_subject == "agents.direct.requester-agent.reply.corr-789"
        
        reply_message = Message.from_bytes(reply_data)
        assert reply_message.message_type == MessageType.ACK
        assert reply_message.payload["result"] == "task_completed"
        assert reply_message.payload["task_id"] == "incoming-task-001"
    
    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, message_bus):
        """测试请求超时处理"""
        # 发送请求但不提供响应（超时场景）
        ping_payload = MessageProtocol.create_ping_message("test-agent")
        
        response = await message_bus.send_request(
            receiver_id="unreachable-agent",
            message_type=MessageType.PING,
            payload=ping_payload,
            timeout=0.1  # 很短的超时
        )
        
        # 验证请求超时
        assert response is None
        
        # 验证统计信息
        stats = message_bus.request_response_manager.get_statistics()
        assert stats["requests_sent"] == 1
        assert stats["requests_timed_out"] == 1
    
    @pytest.mark.asyncio  
    async def test_concurrent_requests_limit(self, message_bus):
        """测试并发请求限制"""
        # 设置较低的并发限制
        message_bus.request_response_manager.max_concurrent_requests = 2
        
        # 创建不会响应的任务
        async def send_hanging_request():
            return await message_bus.send_request(
                receiver_id="busy-agent",
                message_type=MessageType.PING,
                payload={},
                timeout=10.0  # 长超时
            )
        
        # 启动两个占用并发限制的请求
        task1 = asyncio.create_task(send_hanging_request())
        task2 = asyncio.create_task(send_hanging_request())
        
        # 等待一下确保请求开始
        await asyncio.sleep(0.01)
        
        # 第三个请求应该被拒绝
        response = await message_bus.send_request(
            receiver_id="another-agent",
            message_type=MessageType.PING, 
            payload={},
            timeout=1.0
        )
        
        assert response is None  # 应该被拒绝
        
        # 清理挂起的任务
        task1.cancel()
        task2.cancel()
        
        try:
            await task1
        except asyncio.CancelledError:
            pass
        
        try:
            await task2
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_request_handler_error_response(self, message_bus):
        """测试请求处理器错误响应"""
        # 注册会抛出异常的处理器
        async def error_handler(message: Message):
            raise RuntimeError("Simulated processing error")
        
        message_bus.register_request_handler(MessageType.TASK_REQUEST, error_handler)
        
        # 创建请求消息
        request_header = MessageHeader(
            message_id="error-request-123",
            correlation_id="error-corr-456",
            reply_to="agents.direct.requester.reply.error-corr-456"
        )
        
        error_request = Message(
            header=request_header,
            sender_id="requester-agent",
            receiver_id="test-agent", 
            message_type=MessageType.TASK_REQUEST,
            payload={"task_id": "error-task"},
            topic="agents.direct.test-agent"
        )
        
        # 模拟传入消息处理
        mock_msg = Mock()
        mock_msg.data = error_request.to_bytes()
        
        # 处理消息
        await message_bus._handle_direct_message(mock_msg)
        
        # 验证错误回复被发送
        message_bus.client.publish.assert_called_once()
        
        reply_call = message_bus.client.publish.call_args
        reply_data = reply_call[1]['data']
        
        reply_message = Message.from_bytes(reply_data)
        assert reply_message.message_type == MessageType.NACK
        assert reply_message.payload["error"] == "PROCESSING_ERROR"
        assert "Simulated processing error" in reply_message.payload["message"]
    
    @pytest.mark.asyncio
    async def test_response_callback_handling(self, message_bus):
        """测试响应回调处理"""
        callback_messages = []
        
        async def ack_callback(message: Message):
            callback_messages.append(message)
        
        # 注册响应回调
        message_bus.register_response_callback(MessageType.ACK, ack_callback)
        
        # 创建响应消息
        ack_header = MessageHeader(message_id="ack-123")
        ack_message = Message(
            header=ack_header,
            sender_id="callback-sender",
            receiver_id="test-agent",
            message_type=MessageType.ACK,
            payload={"status": "completed"},
            topic="test"
        )
        
        # 处理回调响应
        result = await message_bus.request_response_manager.handle_callback_response(ack_message)
        
        assert result is True
        assert len(callback_messages) == 1
        assert callback_messages[0].payload["status"] == "completed"
    
    def test_get_integration_statistics(self, message_bus):
        """测试获取集成统计信息"""
        # 注册一些处理器和回调
        message_bus.register_request_handler(MessageType.TASK_REQUEST, lambda x: x)
        message_bus.register_request_handler(MessageType.COLLABORATION_INVITE, lambda x: x)
        message_bus.register_response_callback(MessageType.ACK, lambda x: x)
        message_bus.register_response_callback(MessageType.NACK, lambda x: x)
        
        # 模拟一些统计数据
        message_bus.request_response_manager.stats["requests_sent"] = 15
        message_bus.request_response_manager.stats["responses_received"] = 12
        message_bus.request_response_manager.stats["requests_timed_out"] = 2
        message_bus.request_response_manager.stats["requests_failed"] = 1
        message_bus.request_response_manager.stats["requests_handled"] = 8
        
        # 获取统计信息
        stats = message_bus.request_response_manager.get_statistics()
        
        # 验证统计信息
        assert stats["request_handlers"] == 2
        assert stats["response_callbacks"] == 2
        assert stats["requests_sent"] == 15
        assert stats["responses_received"] == 12
        assert stats["requests_timed_out"] == 2
        assert stats["requests_failed"] == 1
        assert stats["requests_handled"] == 8