"""
高级通信模式测试
测试多播通信、流式数据传输、事件流处理和智能路由
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock

from src.ai.distributed_message.advanced_patterns import (
    MulticastManager, StreamingManager, SmartRouter, AdvancedCommunicationManager,
    AgentCapability, MulticastGroup, DataStream, StreamChunk, StreamMode, RoutingStrategy
)
from src.ai.distributed_message.models import Message, MessageHeader, MessageType
from src.ai.distributed_message.client import NATSClient


class TestMulticastManager:
    """多播管理器测试"""
    
    @pytest.fixture
    def mock_client(self):
        client = Mock(spec=NATSClient)
        client.agent_id = "test-agent"
        client.js_publish = AsyncMock(return_value=Mock(sequence=1))
        return client
    
    @pytest.fixture
    def multicast_manager(self, mock_client):
        return MulticastManager(mock_client)
    
    def test_create_group(self, multicast_manager):
        """测试创建多播组"""
        group_id = multicast_manager.create_group(
            group_name="test-group",
            description="测试组",
            max_members=50
        )
        
        assert group_id in multicast_manager.groups
        group = multicast_manager.groups[group_id]
        assert group.group_name == "test-group"
        assert group.description == "测试组"
        assert group.max_members == 50
        assert len(group.members) == 0
    
    def test_join_leave_group(self, multicast_manager):
        """测试加入和离开组"""
        group_id = multicast_manager.create_group("test-group")
        
        # 加入组
        success = multicast_manager.join_group(group_id, "agent1")
        assert success is True
        assert "agent1" in multicast_manager.groups[group_id].members
        assert "agent1" in multicast_manager.agent_groups
        assert group_id in multicast_manager.agent_groups["agent1"]
        
        # 离开组
        success = multicast_manager.leave_group(group_id, "agent1")
        assert success is True
        assert "agent1" not in multicast_manager.groups[group_id].members
    
    def test_group_member_limit(self, multicast_manager):
        """测试组成员限制"""
        group_id = multicast_manager.create_group("test-group", max_members=2)
        
        # 添加2个成员应该成功
        assert multicast_manager.join_group(group_id, "agent1") is True
        assert multicast_manager.join_group(group_id, "agent2") is True
        
        # 添加第3个成员应该失败
        assert multicast_manager.join_group(group_id, "agent3") is False
    
    @pytest.mark.asyncio
    async def test_send_multicast(self, multicast_manager, mock_client):
        """测试发送多播消息"""
        group_id = multicast_manager.create_group("test-group")
        multicast_manager.join_group(group_id, "agent1")
        multicast_manager.join_group(group_id, "agent2")
        multicast_manager.join_group(group_id, "agent3")
        
        # 创建测试消息
        header = MessageHeader(message_id=str(uuid.uuid4()))
        message = Message(
            header=header,
            sender_id="sender-agent",
            receiver_id=None,
            message_type=MessageType.BROADCAST,
            payload={"content": "test broadcast"}
        )
        
        # 发送多播消息，排除agent2
        sent_count = await multicast_manager.send_multicast(
            group_id, message, exclude_agents={"agent2"}
        )
        
        assert sent_count == 2  # agent1 and agent3
        assert mock_client.js_publish.call_count == 2
    
    def test_get_group_info(self, multicast_manager):
        """测试获取组信息"""
        group_id = multicast_manager.create_group("test-group", description="测试组")
        multicast_manager.join_group(group_id, "agent1")
        
        info = multicast_manager.get_group_info(group_id)
        assert info is not None
        assert info["group_name"] == "test-group"
        assert info["description"] == "测试组"
        assert info["member_count"] == 1
        assert "agent1" in info["members"]


class TestStreamingManager:
    """流式数据传输管理器测试"""
    
    @pytest.fixture
    def mock_client(self):
        client = Mock(spec=NATSClient)
        client.agent_id = "test-agent"
        client.js_publish = AsyncMock(return_value=Mock(sequence=1))
        return client
    
    @pytest.fixture
    def streaming_manager(self, mock_client):
        return StreamingManager(mock_client)
    
    @pytest.mark.asyncio
    async def test_send_small_stream(self, streaming_manager, mock_client):
        """测试发送小数据流"""
        test_data = b"Hello, World! This is a test stream."
        
        stream_id = await streaming_manager.send_stream(
            receiver_id="receiver-agent",
            data=test_data,
            metadata={"type": "text", "encoding": "utf-8"}
        )
        
        assert stream_id is not None
        assert stream_id in streaming_manager.active_streams
        
        # 验证发送了正确数量的消息
        # START + CHUNK(s) + END
        expected_calls = 3  # START, 1 CHUNK, END
        assert mock_client.js_publish.call_count == expected_calls
    
    @pytest.mark.asyncio
    async def test_send_large_stream(self, streaming_manager, mock_client):
        """测试发送大数据流"""
        # 创建大于chunk_size的数据
        chunk_size = streaming_manager.chunk_size
        test_data = b"X" * (chunk_size * 2 + 1000)  # 2.5个chunk
        
        stream_id = await streaming_manager.send_stream(
            receiver_id="receiver-agent",
            data=test_data
        )
        
        assert stream_id is not None
        
        # 验证发送了正确数量的消息
        # START + 3 CHUNKs + END = 5 messages
        assert mock_client.js_publish.call_count == 5
    
    def test_split_data_into_chunks(self, streaming_manager):
        """测试数据分块"""
        test_data = b"0123456789" * 1000  # 10KB数据
        stream_id = str(uuid.uuid4())
        
        chunks = streaming_manager._split_data_into_chunks(test_data, stream_id)
        
        assert len(chunks) > 0
        assert chunks[0].sequence == 0
        assert chunks[-1].is_last is True
        
        # 验证重建数据
        rebuilt_data = b''.join(chunk.data for chunk in chunks)
        assert rebuilt_data == test_data
    
    @pytest.mark.asyncio
    async def test_handle_stream_messages(self, streaming_manager):
        """测试处理流消息"""
        stream_id = str(uuid.uuid4())
        
        # 创建流开始消息
        start_header = MessageHeader(message_id=str(uuid.uuid4()), stream_id=stream_id)
        start_message = Message(
            header=start_header,
            sender_id="sender-agent",
            receiver_id="receiver-agent",
            message_type=MessageType.DATA_STREAM_START,
            payload={
                "stream_id": stream_id,
                "total_chunks": 2,
                "chunk_size": 1024,
                "metadata": {"type": "test"}
            }
        )
        
        # 处理流开始消息
        success = await streaming_manager.handle_stream_start(start_message)
        assert success is True
        assert stream_id in streaming_manager.active_streams
        
        # 创建数据块消息
        test_data_1 = b"chunk1data"
        chunk_header_1 = MessageHeader(message_id=str(uuid.uuid4()), stream_id=stream_id)
        chunk_message_1 = Message(
            header=chunk_header_1,
            sender_id="sender-agent",
            receiver_id="receiver-agent",
            message_type=MessageType.DATA_STREAM_CHUNK,
            payload={
                "chunk_id": str(uuid.uuid4()),
                "sequence": 0,
                "data": test_data_1.hex(),
                "is_last": False
            }
        )
        
        test_data_2 = b"chunk2data"
        chunk_header_2 = MessageHeader(message_id=str(uuid.uuid4()), stream_id=stream_id)
        chunk_message_2 = Message(
            header=chunk_header_2,
            sender_id="sender-agent",
            receiver_id="receiver-agent",
            message_type=MessageType.DATA_STREAM_CHUNK,
            payload={
                "chunk_id": str(uuid.uuid4()),
                "sequence": 1,
                "data": test_data_2.hex(),
                "is_last": True
            }
        )
        
        # 处理数据块
        success = await streaming_manager.handle_stream_chunk(chunk_message_1)
        assert success is True
        success = await streaming_manager.handle_stream_chunk(chunk_message_2)
        assert success is True
        
        # 创建流结束消息
        end_header = MessageHeader(message_id=str(uuid.uuid4()), stream_id=stream_id)
        end_message = Message(
            header=end_header,
            sender_id="sender-agent",
            receiver_id="receiver-agent",
            message_type=MessageType.DATA_STREAM_END,
            payload={"stream_id": stream_id}
        )
        
        # 处理流结束消息
        complete_data = await streaming_manager.handle_stream_end(end_message)
        assert complete_data == test_data_1 + test_data_2
        assert stream_id not in streaming_manager.active_streams  # 应该被清理
    
    def test_get_stream_info(self, streaming_manager):
        """测试获取流信息"""
        stream_id = str(uuid.uuid4())
        stream = DataStream(
            stream_id=stream_id,
            sender_id="sender-agent",
            receiver_id="receiver-agent",
            stream_mode=StreamMode.PUSH,
            total_chunks=5,
            metadata={"type": "test"}
        )
        streaming_manager.active_streams[stream_id] = stream
        
        info = streaming_manager.get_stream_info(stream_id)
        assert info is not None
        assert info["stream_id"] == stream_id
        assert info["total_chunks"] == 5
        assert info["received_chunks"] == 0
        assert info["progress"] == 0.0
        assert info["is_complete"] is False


class TestSmartRouter:
    """智能路由器测试"""
    
    @pytest.fixture
    def mock_client(self):
        client = Mock(spec=NATSClient)
        client.agent_id = "router-agent"
        client.js_publish = AsyncMock(return_value=Mock(sequence=1))
        return client
    
    @pytest.fixture
    def smart_router(self, mock_client):
        return SmartRouter(mock_client)
    
    def test_register_agent_capability(self, smart_router):
        """测试注册智能体能力"""
        smart_router.register_agent_capability(
            agent_id="agent1",
            capabilities={"nlp", "vision"},
            load_factor=0.3,
            location="us-west",
            priority=5
        )
        
        assert "agent1" in smart_router.agent_capabilities
        capability = smart_router.agent_capabilities["agent1"]
        assert "nlp" in capability.capabilities
        assert "vision" in capability.capabilities
        assert capability.load_factor == 0.3
        assert capability.location == "us-west"
        assert capability.priority == 5
    
    def test_find_best_agent_least_loaded(self, smart_router):
        """测试最少负载路由策略"""
        # 注册三个智能体
        smart_router.register_agent_capability("agent1", {"nlp"}, load_factor=0.8)
        smart_router.register_agent_capability("agent2", {"nlp"}, load_factor=0.3)
        smart_router.register_agent_capability("agent3", {"nlp"}, load_factor=0.6)
        
        # 应该选择负载最低的agent2
        best = smart_router.find_best_agent("nlp", RoutingStrategy.LEAST_LOADED)
        assert best == "agent2"
    
    def test_find_best_agent_round_robin(self, smart_router):
        """测试轮询路由策略"""
        smart_router.register_agent_capability("agent1", {"nlp"})
        smart_router.register_agent_capability("agent2", {"nlp"})
        smart_router.register_agent_capability("agent3", {"nlp"})
        
        # 轮询选择
        agents = []
        for _ in range(6):  # 测试两轮
            agent = smart_router.find_best_agent("nlp", RoutingStrategy.ROUND_ROBIN)
            agents.append(agent)
        
        # 验证轮询模式
        assert len(set(agents)) == 3  # 应该包含所有三个智能体
        assert agents[0] != agents[1]  # 连续选择不应该相同
    
    def test_find_best_agent_priority_based(self, smart_router):
        """测试基于优先级的路由策略"""
        smart_router.register_agent_capability("agent1", {"nlp"}, priority=3)
        smart_router.register_agent_capability("agent2", {"nlp"}, priority=8)  # 最高优先级
        smart_router.register_agent_capability("agent3", {"nlp"}, priority=5)
        
        best = smart_router.find_best_agent("nlp", RoutingStrategy.PRIORITY_BASED)
        assert best == "agent2"
    
    def test_find_best_agent_with_exclusion(self, smart_router):
        """测试排除特定智能体的路由"""
        smart_router.register_agent_capability("agent1", {"nlp"}, load_factor=0.1)  # 最低负载
        smart_router.register_agent_capability("agent2", {"nlp"}, load_factor=0.5)
        
        # 排除负载最低的agent1
        best = smart_router.find_best_agent(
            "nlp",
            RoutingStrategy.LEAST_LOADED,
            exclude_agents={"agent1"}
        )
        assert best == "agent2"
    
    def test_find_best_agent_no_capability(self, smart_router):
        """测试找不到具有所需能力的智能体"""
        smart_router.register_agent_capability("agent1", {"vision"})
        smart_router.register_agent_capability("agent2", {"audio"})
        
        # 查找nlp能力的智能体
        best = smart_router.find_best_agent("nlp")
        assert best is None
    
    @pytest.mark.asyncio
    async def test_route_message(self, smart_router, mock_client):
        """测试消息路由"""
        smart_router.register_agent_capability("agent1", {"nlp"})
        
        # 创建测试消息
        header = MessageHeader(message_id=str(uuid.uuid4()))
        message = Message(
            header=header,
            sender_id="sender-agent",
            receiver_id=None,  # 将由路由器决定
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "analyze_text"}
        )
        
        # 路由消息
        success = await smart_router.route_message(message, "nlp")
        assert success is True
        mock_client.js_publish.assert_called_once()
        
        # 验证路由统计
        stats = smart_router.get_routing_statistics()
        assert stats["total_routes"] == 1
        assert len(stats["agent_statistics"]) == 1
        assert stats["agent_statistics"][0]["agent_id"] == "agent1"
    
    def test_update_agent_load(self, smart_router):
        """测试更新智能体负载"""
        smart_router.register_agent_capability("agent1", {"nlp"}, load_factor=0.0)
        
        # 更新负载
        smart_router.update_agent_load("agent1", 0.7)
        
        capability = smart_router.agent_capabilities["agent1"]
        assert capability.load_factor == 0.7
    
    def test_cleanup_stale_agents(self, smart_router):
        """测试清理过期智能体"""
        import datetime
        from unittest.mock import patch
        
        # 注册智能体
        smart_router.register_agent_capability("agent1", {"nlp"})
        
        # 模拟时间过去了45分钟
        with patch('src.ai.distributed_message.advanced_patterns.datetime') as mock_datetime:
            future_time = datetime.datetime.now() + datetime.timedelta(minutes=45)
            mock_datetime.now.return_value = future_time
            mock_datetime.timedelta = datetime.timedelta
            
            smart_router.cleanup_stale_agents(max_age_minutes=30)
        
        # agent1应该被清理
        assert "agent1" not in smart_router.agent_capabilities


class TestAdvancedCommunicationManager:
    """高级通信模式管理器测试"""
    
    @pytest.fixture
    def mock_client(self):
        client = Mock(spec=NATSClient)
        client.agent_id = "test-agent"
        
        # 创建带有 unsubscribe 异步方法的 mock subscription
        mock_subscription = Mock()
        mock_subscription.unsubscribe = AsyncMock()
        
        client.subscribe = AsyncMock(return_value=mock_subscription)
        return client
    
    @pytest.fixture
    def comm_manager(self, mock_client):
        return AdvancedCommunicationManager(mock_client)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, comm_manager):
        """测试启动和停止"""
        await comm_manager.start()
        assert len(comm_manager.event_subscriptions) > 0
        
        await comm_manager.stop()
        assert len(comm_manager.event_subscriptions) == 0
    
    def test_register_event_handler(self, comm_manager):
        """测试注册事件处理器"""
        async def test_handler(data):
            pass
        
        comm_manager.register_event_handler("test_event", test_handler)
        assert "test_event" in comm_manager.event_handlers
        
        comm_manager.unregister_event_handler("test_event")
        assert "test_event" not in comm_manager.event_handlers
    
    def test_get_advanced_statistics(self, comm_manager):
        """测试获取高级统计信息"""
        # 注册一些组件
        comm_manager.multicast_manager.create_group("test-group")
        comm_manager.smart_router.register_agent_capability("agent1", {"nlp"})
        comm_manager.register_event_handler("test_event", lambda x: x)
        
        stats = comm_manager.get_advanced_statistics()
        
        assert "multicast" in stats
        assert "streaming" in stats
        assert "routing" in stats
        assert "event_handlers" in stats
        assert stats["multicast"]["groups_count"] == 1
        assert "test_event" in stats["event_handlers"]