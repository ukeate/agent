"""
消息总线高级通信模式集成测试
测试MessageBus与AdvancedCommunicationManager的完整集成
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch

from src.ai.distributed_message.message_bus import DistributedMessageBus
from src.ai.distributed_message.models import MessageType, MessagePriority
from src.ai.distributed_message.advanced_patterns import RoutingStrategy


class TestMessageBusAdvancedIntegration:
    """消息总线高级通信模式集成测试"""
    
    @pytest.fixture
    def message_bus(self):
        """创建消息总线实例"""
        with patch('src.ai.distributed_message.message_bus.NATSClient') as mock_client_class:
            # 模拟NATSClient
            mock_client = Mock()
            mock_client.agent_id = "test-agent"  # 设置实际的字符串值而不是Mock
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
    async def test_multicast_group_operations(self, message_bus):
        """测试多播组操作"""
        await message_bus.connect()
        
        try:
            # 创建多播组
            group_id = message_bus.create_multicast_group(
                group_name="test-team",
                description="测试团队组",
                max_members=10
            )
            
            assert group_id is not None
            
            # 加入组
            success = message_bus.join_multicast_group(group_id)
            assert success is True
            
            # 另一个智能体也加入组
            success = message_bus.join_multicast_group(group_id, "other-agent")
            assert success is True
            
            # 发送多播消息
            sent_count = await message_bus.send_multicast_message(
                group_id=group_id,
                message_type=MessageType.BROADCAST,
                payload={"announcement": "团队会议开始"},
                priority=MessagePriority.HIGH
            )
            
            assert sent_count == 2  # 发送给两个成员
            
            # 获取组信息
            groups = message_bus.get_multicast_groups()
            assert len(groups) == 1
            assert groups[0]["group_name"] == "test-team"
            assert groups[0]["member_count"] == 2
            
            # 离开组
            success = message_bus.leave_multicast_group(group_id)
            assert success is True
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_data_streaming(self, message_bus):
        """测试数据流功能"""
        await message_bus.connect()
        
        try:
            # 发送数据流
            test_data = "这是一个测试数据流，包含一些中文内容和binary数据。".encode('utf-8') * 1000
            
            stream_id = await message_bus.send_data_stream(
                receiver_id="data-processor-agent",
                data=test_data,
                metadata={"type": "text", "encoding": "utf-8"}
            )
            
            assert stream_id is not None
            
            # 验证流已创建
            active_streams = message_bus.get_active_streams()
            assert len(active_streams) == 1
            assert active_streams[0]["stream_id"] == stream_id
            assert active_streams[0]["sender_id"] == "test-agent"
            assert active_streams[0]["receiver_id"] == "data-processor-agent"
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_smart_routing(self, message_bus):
        """测试智能路由功能"""
        await message_bus.connect()
        
        try:
            # 注册当前智能体的能力
            message_bus.register_agent_capability(
                capabilities={"nlp", "text_analysis"},
                load_factor=0.3,
                location="us-west",
                priority=8
            )
            
            # 注册其他智能体的能力（模拟）
            message_bus.advanced_comm_manager.smart_router.register_agent_capability(
                agent_id="nlp-specialist",
                capabilities={"nlp", "sentiment_analysis"},
                load_factor=0.1,  # 更低负载
                priority=9  # 更高优先级
            )
            
            # 创建需要路由的消息
            header = message_bus._create_message_header()
            message = message_bus._create_message(
                header=header,
                receiver_id=None,  # 将由路由器决定
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "analyze_sentiment", "text": "这是测试文本"}
            )
            
            # 测试最少负载路由策略
            success = await message_bus.route_message_by_capability(
                message, "nlp", RoutingStrategy.LEAST_LOADED
            )
            assert success is True
            
            # 验证路由统计
            routing_stats = message_bus.get_routing_statistics()
            assert routing_stats["total_routes"] == 1
            assert len(routing_stats["agent_statistics"]) > 0
            
            # 更新负载
            message_bus.update_agent_load(0.8)
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_event_handling(self, message_bus):
        """测试事件处理"""
        await message_bus.connect()
        
        try:
            # 注册事件处理器
            processed_events = []
            
            async def stream_complete_handler(stream_id, data):
                processed_events.append({"type": "stream_complete", "stream_id": stream_id, "data_size": len(data)})
            
            def group_message_handler(message):
                processed_events.append({"type": "group_message", "message_id": message.header.message_id})
            
            message_bus.register_event_handler("stream_complete", stream_complete_handler)
            message_bus.register_event_handler("group_message", group_message_handler)
            
            # 验证事件处理器已注册
            advanced_stats = message_bus.get_advanced_statistics()
            assert "stream_complete" in advanced_stats["event_handlers"]
            assert "group_message" in advanced_stats["event_handlers"]
            
            # 取消注册一个处理器
            message_bus.unregister_event_handler("group_message")
            
            # 验证处理器已移除
            advanced_stats = message_bus.get_advanced_statistics()
            assert "group_message" not in advanced_stats["event_handlers"]
            assert "stream_complete" in advanced_stats["event_handlers"]
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_comprehensive_advanced_statistics(self, message_bus):
        """测试综合的高级统计信息"""
        await message_bus.connect()
        
        try:
            # 创建多播组
            group_id = message_bus.create_multicast_group("stats-test-group")
            message_bus.join_multicast_group(group_id)
            
            # 发送数据流
            test_data = b"test data for statistics"
            await message_bus.send_data_stream("receiver", test_data)
            
            # 注册智能体能力
            message_bus.register_agent_capability({"stats_test"})
            
            # 注册事件处理器
            message_bus.register_event_handler("test_event", lambda x: x)
            
            # 获取统计信息
            stats = message_bus.get_advanced_statistics()
            
            # 验证统计信息
            assert stats["multicast"]["groups_count"] == 1
            assert stats["multicast"]["total_members"] == 1
            assert stats["streaming"]["active_streams"] == 1
            assert "test_event" in stats["event_handlers"]
            assert len(stats["event_subscriptions"]) > 0
            
            # 验证路由统计
            routing_stats = stats["routing"]
            assert "registered_agents" in routing_stats
            assert routing_stats["registered_agents"] == 1
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_advanced_manager_lifecycle(self, message_bus):
        """测试高级通信管理器的生命周期"""
        # 验证初始状态
        assert len(message_bus.advanced_comm_manager.event_subscriptions) == 0
        
        # 连接应启动高级通信管理器
        await message_bus.connect()
        assert len(message_bus.advanced_comm_manager.event_subscriptions) > 0
        
        # 断开连接应停止高级通信管理器
        await message_bus.disconnect()
        assert len(message_bus.advanced_comm_manager.event_subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_multicast_with_exclusion(self, message_bus):
        """测试带排除的多播消息"""
        await message_bus.connect()
        
        try:
            # 创建组并添加多个成员
            group_id = message_bus.create_multicast_group("exclusion-test-group")
            message_bus.join_multicast_group(group_id, "agent1")
            message_bus.join_multicast_group(group_id, "agent2") 
            message_bus.join_multicast_group(group_id, "agent3")
            
            # 发送多播消息，排除agent2
            sent_count = await message_bus.send_multicast_message(
                group_id=group_id,
                message_type=MessageType.BROADCAST,
                payload={"message": "只发给部分成员"},
                exclude_agents={"agent2"}
            )
            
            assert sent_count == 2  # 只发送给agent1和agent3
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_routing_strategies(self, message_bus):
        """测试不同的路由策略"""
        await message_bus.connect()
        
        try:
            # 注册多个智能体
            router = message_bus.advanced_comm_manager.smart_router
            
            router.register_agent_capability("agent_high_priority", {"task_processing"}, priority=10)
            router.register_agent_capability("agent_low_load", {"task_processing"}, load_factor=0.1)
            router.register_agent_capability("agent_high_load", {"task_processing"}, load_factor=0.9)
            
            # 测试优先级路由
            header = message_bus._create_message_header()
            message = message_bus._create_message(
                header=header,
                receiver_id=None,
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "process"}
            )
            
            # 优先级路由应该选择agent_high_priority
            success = await message_bus.route_message_by_capability(
                message, "task_processing", RoutingStrategy.PRIORITY_BASED
            )
            assert success is True
            
            # 最少负载路由应该选择agent_low_load
            success = await message_bus.route_message_by_capability(
                message, "task_processing", RoutingStrategy.LEAST_LOADED
            )
            assert success is True
            
            # 验证路由统计更新
            routing_stats = message_bus.get_routing_statistics()
            assert routing_stats["total_routes"] == 2
            
        finally:
            await message_bus.disconnect()