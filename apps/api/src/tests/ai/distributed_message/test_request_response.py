"""
请求-响应机制测试
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

from src.ai.distributed_message.request_response import RequestResponseManager, PendingRequest
from src.ai.distributed_message.models import Message, MessageHeader, MessageType, MessagePriority


class TestPendingRequest:
    """待处理请求测试"""
    
    def test_create_pending_request(self):
        """测试创建待处理请求"""
        correlation_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        request = PendingRequest(
            correlation_id=correlation_id,
            sender_id="agent-123",
            message_type=MessageType.TASK_REQUEST,
            created_at=datetime.now(),
            timeout=30.0,
            future=future,
            max_retries=3
        )
        
        assert request.correlation_id == correlation_id
        assert request.sender_id == "agent-123"
        assert request.message_type == MessageType.TASK_REQUEST
        assert request.timeout == 30.0
        assert request.future == future
        assert request.retry_count == 0
        assert request.max_retries == 3
        assert not request.is_expired()
        assert request.should_retry()
    
    def test_is_expired(self):
        """测试超时检查"""
        # 创建已过期的请求
        past_time = datetime.now() - timedelta(seconds=31)
        future = asyncio.Future()
        
        request = PendingRequest(
            correlation_id=str(uuid.uuid4()),
            sender_id="agent-123", 
            message_type=MessageType.PING,
            created_at=past_time,
            timeout=30.0,
            future=future
        )
        
        assert request.is_expired()
    
    def test_should_retry(self):
        """测试重试检查"""
        future = asyncio.Future()
        
        request = PendingRequest(
            correlation_id=str(uuid.uuid4()),
            sender_id="agent-123",
            message_type=MessageType.PING,
            created_at=datetime.now(),
            timeout=30.0,
            future=future,
            retry_count=2,
            max_retries=3
        )
        
        # 未达到最大重试次数且future未完成
        assert request.should_retry()
        
        # 达到最大重试次数
        request.retry_count = 3
        assert not request.should_retry()
        
        # future已完成
        request.retry_count = 1
        future.set_result("test")
        assert not request.should_retry()


class TestRequestResponseManager:
    """请求-响应管理器测试"""
    
    def test_create_manager(self):
        """测试创建管理器"""
        manager = RequestResponseManager(
            default_timeout=45.0,
            max_concurrent_requests=500
        )
        
        assert manager.default_timeout == 45.0
        assert manager.max_concurrent_requests == 500
        assert len(manager.pending_requests) == 0
        assert len(manager.request_handlers) == 0
        assert len(manager.response_callbacks) == 0
        assert manager.cleanup_task is None
    
    @pytest.mark.asyncio
    async def test_send_request_success(self):
        """测试成功发送请求"""
        manager = RequestResponseManager()
        
        # 模拟发送函数
        async def mock_sender(receiver_id, message_type, payload, correlation_id, priority):
            # 模拟响应
            response_header = MessageHeader(
                message_id=str(uuid.uuid4()),
                correlation_id=correlation_id
            )
            
            response_message = Message(
                header=response_header,
                sender_id=receiver_id,
                receiver_id="sender",
                message_type=MessageType.ACK,
                payload={"status": "success"},
                topic="test"
            )
            
            # 延迟一下然后设置响应
            def set_response():
                manager.handle_response(response_message)
            
            asyncio.get_event_loop().call_later(0.1, set_response)
            return True
        
        # 发送请求
        response = await manager.send_request(
            sender_function=mock_sender,
            receiver_id="agent-456",
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "test"},
            timeout=5.0
        )
        
        assert response is not None
        assert response.message_type == MessageType.ACK
        assert response.payload["status"] == "success"
        assert manager.stats["requests_sent"] == 1
        assert manager.stats["responses_received"] == 1
    
    @pytest.mark.asyncio
    async def test_send_request_timeout(self):
        """测试请求超时"""
        manager = RequestResponseManager()
        
        # 模拟发送函数（不响应）
        async def mock_sender(receiver_id, message_type, payload, correlation_id, priority):
            return True
        
        # 发送请求
        response = await manager.send_request(
            sender_function=mock_sender,
            receiver_id="agent-456",
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "test"},
            timeout=0.1  # 很短的超时
        )
        
        assert response is None
        assert manager.stats["requests_sent"] == 1
        assert manager.stats["requests_timed_out"] == 1
    
    @pytest.mark.asyncio
    async def test_send_request_max_concurrent(self):
        """测试并发请求限制"""
        manager = RequestResponseManager(max_concurrent_requests=2)
        
        # 模拟永不响应的发送函数
        async def mock_sender(receiver_id, message_type, payload, correlation_id, priority):
            return True
        
        # 发送两个请求占满并发限制
        task1 = asyncio.create_task(manager.send_request(
            sender_function=mock_sender,
            receiver_id="agent-1",
            message_type=MessageType.PING,
            payload={},
            timeout=10.0
        ))
        
        task2 = asyncio.create_task(manager.send_request(
            sender_function=mock_sender,
            receiver_id="agent-2", 
            message_type=MessageType.PING,
            payload={},
            timeout=10.0
        ))
        
        # 稍等一下让任务开始
        await asyncio.sleep(0.01)
        
        # 第三个请求应该被拒绝
        response = await manager.send_request(
            sender_function=mock_sender,
            receiver_id="agent-3",
            message_type=MessageType.PING,
            payload={},
            timeout=1.0
        )
        
        assert response is None  # 应该被拒绝
        
        # 取消前两个任务
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
    
    def test_handle_response_success(self):
        """测试成功处理响应"""
        manager = RequestResponseManager()
        correlation_id = str(uuid.uuid4())
        
        # 手动添加pending request
        future = asyncio.Future()
        pending_request = PendingRequest(
            correlation_id=correlation_id,
            sender_id="sender",
            message_type=MessageType.TASK_REQUEST,
            created_at=datetime.now(),
            timeout=30.0,
            future=future
        )
        
        manager.pending_requests[correlation_id] = pending_request
        
        # 创建响应消息
        response_header = MessageHeader(
            message_id=str(uuid.uuid4()),
            correlation_id=correlation_id
        )
        
        response_message = Message(
            header=response_header,
            sender_id="responder",
            receiver_id="sender",
            message_type=MessageType.ACK,
            payload={"result": "success"},
            topic="test"
        )
        
        # 处理响应
        result = manager.handle_response(response_message)
        
        assert result is True
        assert future.done()
        assert future.result() == response_message
    
    def test_handle_response_no_correlation_id(self):
        """测试处理没有correlation_id的响应"""
        manager = RequestResponseManager()
        
        response_header = MessageHeader(
            message_id=str(uuid.uuid4())
            # 没有correlation_id
        )
        
        response_message = Message(
            header=response_header,
            sender_id="responder",
            receiver_id="sender",
            message_type=MessageType.ACK,
            payload={"result": "success"},
            topic="test"
        )
        
        result = manager.handle_response(response_message)
        assert result is False
    
    def test_handle_response_no_pending_request(self):
        """测试处理没有对应pending request的响应"""
        manager = RequestResponseManager()
        
        response_header = MessageHeader(
            message_id=str(uuid.uuid4()),
            correlation_id="nonexistent-id"
        )
        
        response_message = Message(
            header=response_header,
            sender_id="responder",
            receiver_id="sender",
            message_type=MessageType.ACK,
            payload={"result": "success"},
            topic="test"
        )
        
        result = manager.handle_response(response_message)
        assert result is False
    
    def test_register_request_handler(self):
        """测试注册请求处理器"""
        manager = RequestResponseManager()
        
        async def test_handler(message: Message):
            return {"result": "processed"}
        
        manager.register_request_handler(
            MessageType.TASK_REQUEST,
            test_handler,
            is_async=True
        )
        
        assert MessageType.TASK_REQUEST in manager.request_handlers
        handler_info = manager.request_handlers[MessageType.TASK_REQUEST]
        assert handler_info['handler'] == test_handler
        assert handler_info['is_async'] is True
    
    @pytest.mark.asyncio
    async def test_handle_request_success(self):
        """测试成功处理请求"""
        manager = RequestResponseManager()
        
        # 注册请求处理器
        async def test_handler(message: Message):
            return {"result": "processed", "task_id": message.payload.get("task_id")}
        
        manager.register_request_handler(MessageType.TASK_REQUEST, test_handler)
        
        # 创建请求消息
        request_header = MessageHeader(
            message_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            reply_to="reply_subject"
        )
        
        request_message = Message(
            header=request_header,
            sender_id="requester",
            receiver_id="handler",
            message_type=MessageType.TASK_REQUEST,
            payload={"task_id": "task-123", "data": "test"},
            topic="test"
        )
        
        # 模拟回复函数
        reply_calls = []
        async def mock_reply(message, payload, message_type):
            reply_calls.append((message, payload, message_type))
        
        # 处理请求
        result = await manager.handle_request(request_message, mock_reply)
        
        assert result is True
        assert len(reply_calls) == 1
        
        replied_message, reply_payload, reply_type = reply_calls[0]
        assert replied_message == request_message
        assert reply_payload["result"] == "processed"
        assert reply_payload["task_id"] == "task-123"
        assert reply_type == MessageType.ACK
    
    @pytest.mark.asyncio
    async def test_handle_request_unsupported_type(self):
        """测试处理不支持的消息类型"""
        manager = RequestResponseManager()
        
        # 创建请求消息（没有注册处理器）
        request_header = MessageHeader(
            message_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            reply_to="reply_subject"
        )
        
        request_message = Message(
            header=request_header,
            sender_id="requester",
            receiver_id="handler",
            message_type=MessageType.COLLABORATION_INVITE,  # 未注册的类型
            payload={"data": "test"},
            topic="test"
        )
        
        # 模拟回复函数
        reply_calls = []
        async def mock_reply(message, payload, message_type):
            reply_calls.append((message, payload, message_type))
        
        # 处理请求
        result = await manager.handle_request(request_message, mock_reply)
        
        assert result is False
        assert len(reply_calls) == 1
        
        replied_message, reply_payload, reply_type = reply_calls[0]
        assert reply_payload["error"] == "UNSUPPORTED_MESSAGE_TYPE"
        assert reply_type == MessageType.NACK
    
    @pytest.mark.asyncio
    async def test_handle_request_handler_error(self):
        """测试处理器抛出异常"""
        manager = RequestResponseManager()
        
        # 注册会抛出异常的处理器
        async def error_handler(message: Message):
            raise ValueError("Test error")
        
        manager.register_request_handler(MessageType.TASK_REQUEST, error_handler)
        
        # 创建请求消息
        request_header = MessageHeader(
            message_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            reply_to="reply_subject"
        )
        
        request_message = Message(
            header=request_header,
            sender_id="requester",
            receiver_id="handler",
            message_type=MessageType.TASK_REQUEST,
            payload={"data": "test"},
            topic="test"
        )
        
        # 模拟回复函数
        reply_calls = []
        async def mock_reply(message, payload, message_type):
            reply_calls.append((message, payload, message_type))
        
        # 处理请求
        result = await manager.handle_request(request_message, mock_reply)
        
        assert result is False
        assert len(reply_calls) == 1
        
        replied_message, reply_payload, reply_type = reply_calls[0]
        assert reply_payload["error"] == "PROCESSING_ERROR"
        assert "Test error" in reply_payload["message"]
        assert reply_type == MessageType.NACK
    
    def test_register_response_callback(self):
        """测试注册响应回调"""
        manager = RequestResponseManager()
        
        def test_callback(message: Message):
            pass
        
        manager.register_response_callback(MessageType.ACK, test_callback)
        
        assert MessageType.ACK in manager.response_callbacks
        assert manager.response_callbacks[MessageType.ACK] == test_callback
    
    @pytest.mark.asyncio
    async def test_handle_callback_response(self):
        """测试处理回调响应"""
        manager = RequestResponseManager()
        
        # 注册回调
        callback_calls = []
        async def test_callback(message: Message):
            callback_calls.append(message)
        
        manager.register_response_callback(MessageType.ACK, test_callback)
        
        # 创建响应消息
        response_header = MessageHeader(message_id=str(uuid.uuid4()))
        response_message = Message(
            header=response_header,
            sender_id="responder",
            receiver_id="receiver",
            message_type=MessageType.ACK,
            payload={"status": "ok"},
            topic="test"
        )
        
        # 处理回调响应
        result = await manager.handle_callback_response(response_message)
        
        assert result is True
        assert len(callback_calls) == 1
        assert callback_calls[0] == response_message
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        manager = RequestResponseManager()
        
        # 添加一些pending requests
        for i in range(3):
            correlation_id = str(uuid.uuid4())
            future = asyncio.Future()
            pending_request = PendingRequest(
                correlation_id=correlation_id,
                sender_id=f"agent-{i}",
                message_type=MessageType.PING,
                created_at=datetime.now(),
                timeout=30.0,
                future=future
            )
            manager.pending_requests[correlation_id] = pending_request
        
        # 注册一些处理器和回调
        manager.register_request_handler(MessageType.TASK_REQUEST, lambda x: x)
        manager.register_response_callback(MessageType.ACK, lambda x: x)
        
        # 更新一些统计
        manager.stats["requests_sent"] = 10
        manager.stats["responses_received"] = 8
        manager.stats["requests_timed_out"] = 2
        
        stats = manager.get_statistics()
        
        assert stats["pending_requests"] == 3
        assert stats["request_handlers"] == 1
        assert stats["response_callbacks"] == 1
        assert stats["requests_sent"] == 10
        assert stats["responses_received"] == 8
        assert stats["requests_timed_out"] == 2
    
    def test_get_pending_requests_info(self):
        """测试获取待处理请求信息"""
        manager = RequestResponseManager()
        
        # 添加一些pending requests
        correlation_ids = []
        for i in range(2):
            correlation_id = str(uuid.uuid4())
            correlation_ids.append(correlation_id)
            future = asyncio.Future()
            pending_request = PendingRequest(
                correlation_id=correlation_id,
                sender_id=f"agent-{i}",
                message_type=MessageType.TASK_REQUEST if i == 0 else MessageType.PING,
                created_at=datetime.now(),
                timeout=30.0 + i * 10,
                future=future,
                retry_count=i,
                max_retries=3
            )
            manager.pending_requests[correlation_id] = pending_request
        
        requests_info = manager.get_pending_requests_info()
        
        assert len(requests_info) == 2
        
        info0 = requests_info[0]
        assert info0["correlation_id"] == correlation_ids[0]
        assert info0["sender_id"] == "agent-0"
        assert info0["message_type"] == MessageType.TASK_REQUEST.value
        assert info0["timeout"] == 30.0
        assert info0["retry_count"] == 0
        assert info0["max_retries"] == 3
        assert not info0["is_expired"]
        
        info1 = requests_info[1]
        assert info1["correlation_id"] == correlation_ids[1]
        assert info1["sender_id"] == "agent-1"
        assert info1["message_type"] == MessageType.PING.value
        assert info1["timeout"] == 40.0
        assert info1["retry_count"] == 1
        assert info1["max_retries"] == 3
    
    @pytest.mark.asyncio
    async def test_background_cleanup_task(self):
        """测试后台清理任务"""
        manager = RequestResponseManager(default_timeout=0.1)  # 很短的默认超时
        
        # 启动后台任务
        manager.start_background_tasks()
        
        # 添加一个会过期的请求
        correlation_id = str(uuid.uuid4())
        future = asyncio.Future()
        pending_request = PendingRequest(
            correlation_id=correlation_id,
            sender_id="agent-test",
            message_type=MessageType.PING,
            created_at=datetime.now() - timedelta(seconds=1),  # 已经过期
            timeout=0.1,
            future=future
        )
        manager.pending_requests[correlation_id] = pending_request
        
        # 等待清理任务运行
        manager.cleanup_interval = 0.1  # 加快清理间隔
        await asyncio.sleep(0.2)
        
        # 检查请求是否被清理
        assert correlation_id not in manager.pending_requests
        assert manager.stats["requests_timed_out"] >= 1
        
        # 停止后台任务
        await manager.stop_background_tasks()