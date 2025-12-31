"""
异步智能体架构测试
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from unittest.mock import Mock, AsyncMock
from src.ai.autogen.events import EventBus, Event, EventType, MessageQueue, StateManager
from src.ai.autogen.async_manager import AsyncAgentManager
from src.ai.autogen.config import AgentConfig, AgentRole, AGENT_CONFIGS
from src.ai.autogen.enterprise import EnterpriseIntegrationService, SecurityContext, SecurityLevel
from src.ai.autogen.monitoring import PerformanceMonitor, ConversationTracker

class TestEventBus:
    """事件总线测试"""
    
    @pytest.mark.asyncio
    async def test_event_bus_basic_operations(self):
        """测试事件总线基本操作"""
        event_bus = EventBus(max_queue_size=100)
        
        # 启动事件总线
        await event_bus.start(worker_count=1)
        
        try:
            # 创建测试事件
            test_event = Event(
                type=EventType.AGENT_CREATED,
                source="test_agent",
                data={"test": "data"}
            )
            
            # 发布事件
            success = await event_bus.publish(test_event)
            assert success is True
            
            # 等待事件处理
            await asyncio.sleep(0.1)
            
            # 检查统计信息
            stats = event_bus.get_stats()
            assert stats["running"] is True
            assert stats["processed_events"] >= 1
            
        finally:
            await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_event_handler_subscription(self):
        """测试事件处理器订阅"""
        from src.ai.autogen.events import EventHandler
        
        class TestEventHandler(EventHandler):
            def __init__(self):
                self.received_events = []
            
            @property
            def supported_events(self):
                return [EventType.AGENT_CREATED]
            
            async def handle(self, event):
                self.received_events.append(event)
        
        event_bus = EventBus()
        handler = TestEventHandler()
        
        # 订阅事件
        event_bus.subscribe(EventType.AGENT_CREATED, handler)
        
        await event_bus.start(worker_count=1)
        
        try:
            # 发布事件
            test_event = Event(
                type=EventType.AGENT_CREATED,
                source="test",
                data={"test": "data"}
            )
            
            await event_bus.publish(test_event)
            await asyncio.sleep(0.1)
            
            # 验证事件被接收
            assert len(handler.received_events) == 1
            assert handler.received_events[0].type == EventType.AGENT_CREATED
            
        finally:
            await event_bus.stop()

class TestAsyncAgentManager:
    """异步智能体管理器测试"""
    
    @pytest_asyncio.fixture
    async def agent_manager(self):
        """创建测试用的智能体管理器"""
        event_bus = EventBus()
        message_queue = MessageQueue()
        state_manager = StateManager()
        
        await event_bus.start(worker_count=1)
        
        manager = AsyncAgentManager(
            event_bus=event_bus,
            message_queue=message_queue,
            state_manager=state_manager,
            max_concurrent_tasks=5
        )
        
        await manager.start()
        
        yield manager
        
        await manager.stop()
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_agent_creation_and_destruction(self, agent_manager):
        """测试智能体创建和销毁"""
        # Mock掉create_agent_from_config以避免API调用
        from unittest.mock import patch
        
        mock_agent = Mock()
        mock_agent.config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
        
        with patch('src.ai.autogen.async_manager.create_agent_from_config', return_value=mock_agent):
            # 创建智能体
            config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
            agent_id = await agent_manager.create_agent(config)
        
        assert agent_id is not None
        
        # 验证智能体信息
        agent_info = await agent_manager.get_agent_info(agent_id)
        assert agent_info is not None
        assert agent_info["name"] == config.name
        assert agent_info["role"] == config.role.value
        
        # 销毁智能体
        success = await agent_manager.destroy_agent(agent_id)
        assert success is True
        
        # 验证智能体已被销毁
        agent_info = await agent_manager.get_agent_info(agent_id)
        assert agent_info is None
    
    @pytest.mark.asyncio
    async def test_task_submission_and_execution(self, agent_manager):
        """测试任务提交和执行"""
        # Mock掉智能体创建和任务执行
        from unittest.mock import patch
        
        mock_agent = Mock()
        mock_agent.config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
        mock_agent.execute_task = AsyncMock(return_value={"success": True, "result": "测试结果"})
        
        with patch('src.ai.autogen.async_manager.create_agent_from_config', return_value=mock_agent):
            # 创建智能体
            config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
            agent_id = await agent_manager.create_agent(config)
        
        # 提交任务
        task_id = await agent_manager.submit_task(
            agent_id=agent_id,
            task_type="test_task",
            description="测试任务",
            input_data={"test": "data"},
            priority=1
        )
        
        assert task_id is not None
        
        # 等待任务处理
        await asyncio.sleep(0.5)
        
        # 检查任务状态
        task_info = await agent_manager.get_task_info(task_id)
        assert task_info is not None
        assert task_info["task_id"] == task_id
        assert task_info["agent_id"] == agent_id
    
    @pytest.mark.asyncio
    async def test_manager_statistics(self, agent_manager):
        """测试管理器统计信息"""
        stats = agent_manager.get_manager_stats()
        
        assert "running" in stats
        assert "agents" in stats
        assert "tasks" in stats
        assert "queue" in stats
        
        assert stats["running"] is True
        assert isinstance(stats["agents"]["total"], int)
        assert isinstance(stats["tasks"]["total"], int)

class TestEnterpriseIntegration:
    """企业级集成测试"""
    
    @pytest_asyncio.fixture
    async def enterprise_service(self):
        """创建测试用的企业级服务"""
        event_bus = EventBus()
        message_queue = MessageQueue()
        state_manager = StateManager()
        
        await event_bus.start(worker_count=1)
        
        agent_manager = AsyncAgentManager(
            event_bus=event_bus,
            message_queue=message_queue,
            state_manager=state_manager
        )
        
        await agent_manager.start()
        
        service = EnterpriseIntegrationService(
            event_bus=event_bus,
            agent_manager=agent_manager
        )
        
        yield service
        
        await agent_manager.stop()
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_security_context_validation(self, enterprise_service):
        """测试安全上下文验证"""
        # 创建安全上下文
        security_context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            permissions=["agent:test_agent:create"],
            security_level=SecurityLevel.INTERNAL
        )
        
        # 设置安全策略
        enterprise_service.security_manager.set_security_policy(
            "agent:test_agent",
            SecurityLevel.INTERNAL
        )
        
        # 检查权限
        has_permission = await enterprise_service.security_manager.check_permissions(
            security_context,
            "agent:test_agent",
            "create"
        )
        
        assert has_permission is True
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, enterprise_service):
        """测试审计日志"""
        # 记录操作日志
        await enterprise_service.audit_logger.log_action(
            user_id="test_user",
            session_id="test_session",
            action="agent.create",
            resource="test_agent",
            result="success",
            details={"test": "data"}
        )
        
        # 搜索日志
        logs = enterprise_service.audit_logger.search_logs(
            user_id="test_user",
            limit=10
        )
        
        assert len(logs) >= 1
        assert logs[0].user_id == "test_user"
        assert logs[0].action == "agent.create"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, enterprise_service):
        """测试错误处理"""
        # 模拟智能体错误
        test_error = ValueError("测试错误")
        
        success = await enterprise_service.error_handler.handle_agent_error(
            agent_id="test_agent",
            error=test_error,
            context={"test": "context"}
        )
        
        # 检查错误统计
        error_stats = enterprise_service.error_handler.get_error_stats()
        assert error_stats["total_errors"] >= 1
        assert "ValueError" in error_stats["errors_by_type"]

class TestMonitoringSystem:
    """监控系统测试"""
    
    @pytest_asyncio.fixture
    async def monitoring_components(self):
        """创建测试用的监控组件"""
        event_bus = EventBus()
        await event_bus.start(worker_count=1)
        
        performance_monitor = PerformanceMonitor(event_bus)
        conversation_tracker = ConversationTracker(event_bus)
        
        yield performance_monitor, conversation_tracker
        
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, monitoring_components):
        """测试性能监控"""
        performance_monitor, _ = monitoring_components
        
        # 记录性能指标
        performance_monitor.record_metric(
            name="test_metric",
            value=100.5,
            unit="ms",
            tags={"test": "tag"}
        )
        
        # 获取指标摘要
        summary = performance_monitor.get_metric_summary("test_metric", 60)
        
        assert summary["name"] == "test_metric"
        assert summary["count"] == 1
        assert summary["latest"] == 100.5
        assert summary["unit"] == "ms"
    
    @pytest.mark.asyncio
    async def test_conversation_tracking(self, monitoring_components):
        """测试对话追踪"""
        _, conversation_tracker = monitoring_components
        
        # 开始对话追踪
        conversation_id = "test_conversation"
        trace = conversation_tracker.start_conversation_trace(
            conversation_id=conversation_id,
            session_id="test_session",
            participants=["agent1", "agent2"]
        )
        
        assert trace.conversation_id == conversation_id
        assert len(trace.participants) == 2
        
        # 添加跨度
        span = conversation_tracker.add_span_to_conversation(
            conversation_id=conversation_id,
            operation_name="test_operation"
        )
        
        assert span is not None
        assert span.operation_name == "test_operation"
        
        # 结束追踪
        completed_trace = conversation_tracker.end_conversation_trace(conversation_id)
        assert completed_trace is not None
        assert completed_trace.end_time is not None
    
    @pytest.mark.asyncio
    async def test_measurement_timing(self, monitoring_components):
        """测试测量计时"""
        performance_monitor, _ = monitoring_components
        
        # 开始测量
        measurement_id = "test_measurement"
        performance_monitor.start_measurement(measurement_id)
        
        # 模拟一些处理时间
        await asyncio.sleep(0.1)
        
        # 结束测量
        duration = performance_monitor.end_measurement(
            measurement_id,
            "test_duration"
        )
        
        assert duration is not None
        assert duration >= 100  # 至少100毫秒
        
        # 验证指标被记录
        summary = performance_monitor.get_metric_summary("test_duration")
        assert summary["count"] == 1

class TestIntegrationScenarios:
    """集成场景测试"""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self):
        """测试完整的智能体生命周期"""
        # 初始化组件
        event_bus = EventBus()
        message_queue = MessageQueue()
        state_manager = StateManager()
        
        await event_bus.start(worker_count=1)
        
        agent_manager = AsyncAgentManager(
            event_bus=event_bus,
            message_queue=message_queue,
            state_manager=state_manager
        )
        
        await agent_manager.start()
        
        try:
            # Mock智能体创建和执行
            from unittest.mock import patch
            
            mock_agent = Mock()
            mock_agent.config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
            mock_agent.execute_task = AsyncMock(return_value={"success": True, "result": "测试结果"})
            
            with patch('src.ai.autogen.async_manager.create_agent_from_config', return_value=mock_agent):
                # 创建智能体
                config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
                agent_id = await agent_manager.create_agent(config)
            
            # 提交任务
            task_id = await agent_manager.submit_task(
                agent_id=agent_id,
                task_type="integration_test",
                description="集成测试任务",
                input_data={"test": "integration"}
            )
            
            # 等待任务处理
            await asyncio.sleep(0.5)
            
            # 检查结果
            task_info = await agent_manager.get_task_info(task_id)
            assert task_info is not None
            
            # 销毁智能体
            success = await agent_manager.destroy_agent(agent_id)
            assert success is True
            
        finally:
            await agent_manager.stop()
            await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """测试并发操作"""
        # 初始化组件
        event_bus = EventBus()
        message_queue = MessageQueue()
        state_manager = StateManager()
        
        await event_bus.start(worker_count=2)
        
        agent_manager = AsyncAgentManager(
            event_bus=event_bus,
            message_queue=message_queue,
            state_manager=state_manager,
            max_concurrent_tasks=5
        )
        
        await agent_manager.start()
        
        try:
            # Mock智能体创建
            from unittest.mock import patch
            
            mock_agent = Mock()
            mock_agent.config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
            mock_agent.execute_task = AsyncMock(return_value={"success": True, "result": "测试结果"})
            
            with patch('src.ai.autogen.async_manager.create_agent_from_config', return_value=mock_agent):
                # 并发创建多个智能体
                tasks = []
                for i in range(3):
                    config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
                    config.name = f"测试智能体_{i}"
                    task = agent_manager.create_agent(config)
                    tasks.append(task)
                
                agent_ids = await asyncio.gather(*tasks)
            assert len(agent_ids) == 3
            
            # 并发提交任务
            task_tasks = []
            for agent_id in agent_ids:
                task = agent_manager.submit_task(
                    agent_id=agent_id,
                    task_type="concurrent_test",
                    description=f"并发测试任务 for {agent_id}",
                    input_data={"agent_id": agent_id}
                )
                task_tasks.append(task)
            
            task_ids = await asyncio.gather(*task_tasks)
            assert len(task_ids) == 3
            
            # 等待任务处理
            await asyncio.sleep(1.0)
            
            # 检查所有任务
            for task_id in task_ids:
                task_info = await agent_manager.get_task_info(task_id)
                assert task_info is not None
            
        finally:
            await agent_manager.stop()
            await event_bus.stop()

if __name__ == "__main__":
    # 运行基本测试
    asyncio.run(pytest.main([__file__, "-v"]))
