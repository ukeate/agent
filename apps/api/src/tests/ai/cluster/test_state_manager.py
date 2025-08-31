"""
测试集群状态管理器
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import Mock, AsyncMock

from src.ai.cluster.state_manager import ClusterStateManager, StateChangeEvent
from src.ai.cluster.topology import AgentInfo, AgentStatus, AgentGroup, ResourceUsage, AgentHealthCheck


@pytest_asyncio.fixture
async def state_manager():
    """创建状态管理器实例"""
    manager = ClusterStateManager("test-cluster")
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def sample_agent():
    """创建示例智能体"""
    return AgentInfo(
        name="test-agent",
        host="localhost",
        port=8080
    )


@pytest.fixture
def sample_group():
    """创建示例分组"""
    return AgentGroup(
        name="test-group",
        description="Test group"
    )


class TestClusterStateManager:
    """测试集群状态管理器"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, state_manager):
        """测试初始化"""
        assert state_manager.cluster_id == "test-cluster"
        assert state_manager.topology.cluster_id == "test-cluster"
        assert state_manager._state_version == 0
        assert len(state_manager._change_listeners) == 0
    
    @pytest.mark.asyncio
    async def test_register_agent(self, state_manager, sample_agent):
        """测试注册智能体"""
        # 注册智能体
        success = await state_manager.register_agent(sample_agent)
        
        assert success is True
        assert sample_agent.agent_id in state_manager.topology.agents
        assert sample_agent.cluster_id == "test-cluster"
        assert state_manager._state_version > 0
        
        # 重复注册应该失败
        success = await state_manager.register_agent(sample_agent)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_unregister_agent(self, state_manager, sample_agent):
        """测试注销智能体"""
        # 先注册智能体
        await state_manager.register_agent(sample_agent)
        agent_id = sample_agent.agent_id
        
        # 注销智能体
        success = await state_manager.unregister_agent(agent_id)
        
        assert success is True
        assert agent_id not in state_manager.topology.agents
        
        # 注销不存在的智能体应该失败
        success = await state_manager.unregister_agent("non-existent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_update_agent_status(self, state_manager, sample_agent):
        """测试更新智能体状态"""
        # 先注册智能体
        await state_manager.register_agent(sample_agent)
        agent_id = sample_agent.agent_id
        
        # 更新状态
        success = await state_manager.update_agent_status(
            agent_id, AgentStatus.RUNNING, "Started successfully"
        )
        
        assert success is True
        agent = await state_manager.get_agent_info(agent_id)
        assert agent.status == AgentStatus.RUNNING
        assert agent.metadata["status_details"] == "Started successfully"
        
        # 更新不存在智能体的状态应该失败
        success = await state_manager.update_agent_status(
            "non-existent", AgentStatus.RUNNING
        )
        assert success is False
    
    @pytest.mark.asyncio
    async def test_update_agent_resource_usage(self, state_manager, sample_agent):
        """测试更新智能体资源使用"""
        # 先注册智能体
        await state_manager.register_agent(sample_agent)
        agent_id = sample_agent.agent_id
        
        # 更新资源使用
        usage = ResourceUsage(
            cpu_usage_percent=75.0,
            memory_usage_percent=60.0,
            active_tasks=5
        )
        
        success = await state_manager.update_agent_resource_usage(agent_id, usage)
        
        assert success is True
        agent = await state_manager.get_agent_info(agent_id)
        assert agent.resource_usage.cpu_usage_percent == 75.0
        assert agent.resource_usage.memory_usage_percent == 60.0
        assert agent.resource_usage.active_tasks == 5
        
        # 更新不存在智能体的资源使用应该失败
        success = await state_manager.update_agent_resource_usage("non-existent", usage)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_update_agent_health(self, state_manager, sample_agent):
        """测试更新智能体健康状态"""
        # 先注册智能体
        await state_manager.register_agent(sample_agent)
        agent_id = sample_agent.agent_id
        
        # 更新健康状态
        health = AgentHealthCheck(
            is_healthy=False,
            consecutive_failures=2
        )
        
        success = await state_manager.update_agent_health(agent_id, health)
        
        assert success is True
        agent = await state_manager.get_agent_info(agent_id)
        assert agent.health.is_healthy is False
        assert agent.health.consecutive_failures == 2
        
        # 更新不存在智能体的健康状态应该失败
        success = await state_manager.update_agent_health("non-existent", health)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_create_group(self, state_manager, sample_group):
        """测试创建分组"""
        success = await state_manager.create_group(sample_group)
        
        assert success is True
        assert sample_group.group_id in state_manager.topology.groups
    
    @pytest.mark.asyncio
    async def test_add_agent_to_group(self, state_manager, sample_agent, sample_group):
        """测试将智能体添加到分组"""
        # 先注册智能体和创建分组
        await state_manager.register_agent(sample_agent)
        await state_manager.create_group(sample_group)
        
        # 添加智能体到分组
        success = await state_manager.add_agent_to_group(sample_group.group_id, sample_agent.agent_id)
        
        assert success is True
        assert sample_agent.agent_id in sample_group.agent_ids
        
        agent = await state_manager.get_agent_info(sample_agent.agent_id)
        assert agent.group_id == sample_group.group_id
        
        # 添加到不存在的分组应该失败
        success = await state_manager.add_agent_to_group("non-existent", sample_agent.agent_id)
        assert success is False
        
        # 添加不存在的智能体应该失败
        success = await state_manager.add_agent_to_group(sample_group.group_id, "non-existent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_cluster_topology(self, state_manager, sample_agent):
        """测试获取集群拓扑"""
        # 先添加一些数据
        await state_manager.register_agent(sample_agent)
        
        topology = await state_manager.get_cluster_topology()
        
        assert topology.cluster_id == "test-cluster"
        assert sample_agent.agent_id in topology.agents
        # 应该返回拷贝，不是原对象
        assert topology is not state_manager.topology
    
    @pytest.mark.asyncio
    async def test_get_agent_info(self, state_manager, sample_agent):
        """测试获取智能体信息"""
        # 先注册智能体
        await state_manager.register_agent(sample_agent)
        
        agent = await state_manager.get_agent_info(sample_agent.agent_id)
        
        assert agent is not None
        assert agent.agent_id == sample_agent.agent_id
        
        # 获取不存在的智能体应该返回None
        agent = await state_manager.get_agent_info("non-existent")
        assert agent is None
    
    @pytest.mark.asyncio
    async def test_get_agents_by_status(self, state_manager):
        """测试按状态获取智能体"""
        # 创建不同状态的智能体
        agent1 = AgentInfo(name="running-agent")
        agent2 = AgentInfo(name="stopped-agent")
        
        await state_manager.register_agent(agent1)
        await state_manager.register_agent(agent2)
        
        # 更新状态
        await state_manager.update_agent_status(agent1.agent_id, AgentStatus.RUNNING)
        await state_manager.update_agent_status(agent2.agent_id, AgentStatus.STOPPED)
        
        running_agents = await state_manager.get_agents_by_status(AgentStatus.RUNNING)
        stopped_agents = await state_manager.get_agents_by_status(AgentStatus.STOPPED)
        
        assert len(running_agents) == 1
        assert running_agents[0].agent_id == agent1.agent_id
        assert len(stopped_agents) == 1
        assert stopped_agents[0].agent_id == agent2.agent_id
    
    @pytest.mark.asyncio
    async def test_get_healthy_agents(self, state_manager):
        """测试获取健康智能体"""
        # 创建健康和不健康的智能体
        agent1 = AgentInfo(name="healthy-agent")
        agent2 = AgentInfo(name="unhealthy-agent")
        
        await state_manager.register_agent(agent1)
        await state_manager.register_agent(agent2)
        
        # 设置健康状态
        await state_manager.update_agent_status(agent1.agent_id, AgentStatus.RUNNING)
        await state_manager.update_agent_status(agent2.agent_id, AgentStatus.RUNNING)
        
        healthy_check = AgentHealthCheck(is_healthy=True, last_heartbeat=time.time())
        unhealthy_check = AgentHealthCheck(is_healthy=False)
        
        await state_manager.update_agent_health(agent1.agent_id, healthy_check)
        await state_manager.update_agent_health(agent2.agent_id, unhealthy_check)
        
        healthy_agents = await state_manager.get_healthy_agents()
        
        assert len(healthy_agents) == 1
        assert healthy_agents[0].agent_id == agent1.agent_id
    
    @pytest.mark.asyncio
    async def test_get_cluster_stats(self, state_manager, sample_agent):
        """测试获取集群统计"""
        # 先添加一些数据
        await state_manager.register_agent(sample_agent)
        await state_manager.update_agent_status(sample_agent.agent_id, AgentStatus.RUNNING)
        
        stats = await state_manager.get_cluster_stats()
        
        assert stats["cluster_id"] == "test-cluster"
        assert stats["total_agents"] == 1
        assert stats["running_agents"] == 1
        assert "health_score" in stats
        assert "resource_usage" in stats
        assert "state_version" in stats
        assert "metrics" in stats
    
    @pytest.mark.asyncio
    async def test_state_change_events(self, state_manager, sample_agent):
        """测试状态变更事件"""
        events = []
        
        def event_listener(event: StateChangeEvent):
            events.append(event)
        
        # 添加监听器
        state_manager.add_change_listener(event_listener)
        
        # 执行会触发事件的操作
        await state_manager.register_agent(sample_agent)
        
        # 等待事件处理
        await asyncio.sleep(0.01)
        
        assert len(events) > 0
        event = events[0]
        assert event.event_type == "agent_registered"
        assert event.agent_id == sample_agent.agent_id
        
        # 移除监听器
        state_manager.remove_change_listener(event_listener)
    
    @pytest.mark.asyncio
    async def test_get_recent_events(self, state_manager, sample_agent):
        """测试获取最近事件"""
        # 执行一些操作产生事件
        await state_manager.register_agent(sample_agent)
        await state_manager.update_agent_status(sample_agent.agent_id, AgentStatus.RUNNING)
        
        events = await state_manager.get_recent_events(limit=10)
        
        assert len(events) >= 2
        assert all(isinstance(event, StateChangeEvent) for event in events)
    
    @pytest.mark.asyncio
    async def test_state_version_tracking(self, state_manager, sample_agent):
        """测试状态版本跟踪"""
        initial_version = state_manager._state_version
        
        # 执行状态变更操作
        await state_manager.register_agent(sample_agent)
        assert state_manager._state_version > initial_version
        
        old_version = state_manager._state_version
        await state_manager.update_agent_status(sample_agent.agent_id, AgentStatus.RUNNING)
        assert state_manager._state_version > old_version
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, state_manager):
        """测试并发操作"""
        # 创建多个智能体
        agents = [AgentInfo(name=f"agent-{i}") for i in range(10)]
        
        # 并发注册
        tasks = [state_manager.register_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks)
        
        # 所有操作都应该成功
        assert all(results)
        assert len(state_manager.topology.agents) == 10
        
        # 并发更新状态
        tasks = [
            state_manager.update_agent_status(agent.agent_id, AgentStatus.RUNNING)
            for agent in agents
        ]
        results = await asyncio.gather(*tasks)
        
        assert all(results)
        running_agents = await state_manager.get_agents_by_status(AgentStatus.RUNNING)
        assert len(running_agents) == 10
    
    @pytest.mark.asyncio
    async def test_health_check_loop(self, state_manager, sample_agent):
        """测试健康检查循环"""
        # 注册智能体并设置为运行状态
        await state_manager.register_agent(sample_agent)
        await state_manager.update_agent_status(sample_agent.agent_id, AgentStatus.RUNNING)
        
        # 设置健康检查为超时状态
        old_heartbeat = time.time() - 1000  # 很久之前的心跳
        health = AgentHealthCheck(
            is_healthy=True,
            last_heartbeat=old_heartbeat,
            health_check_interval=30.0
        )
        await state_manager.update_agent_health(sample_agent.agent_id, health)
        
        # 手动执行一次健康检查
        await state_manager._perform_health_checks()
        
        # 智能体应该被标记为失败
        agent = await state_manager.get_agent_info(sample_agent.agent_id)
        assert agent.status == AgentStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_storage_backend_integration(self):
        """测试存储后端集成"""
        # 创建模拟存储后端
        storage_backend = Mock()
        storage_backend.save_cluster_state = AsyncMock()
        storage_backend.load_cluster_state = AsyncMock(return_value=None)
        
        # 创建带存储后端的状态管理器
        manager = ClusterStateManager("test-cluster", storage_backend)
        await manager.start()
        
        try:
            # 验证启动时调用了加载方法
            storage_backend.load_cluster_state.assert_called_once()
            
            # 手动触发持久化
            await manager._persist_state()
            
            # 验证保存方法被调用
            storage_backend.save_cluster_state.assert_called_once()
            
        finally:
            await manager.stop()


class TestStateChangeEvent:
    """测试状态变更事件"""
    
    def test_event_creation(self):
        """测试事件创建"""
        event = StateChangeEvent(
            "test_event",
            "agent-123",
            {"old": "value"},
            {"new": "value"}
        )
        
        assert event.event_type == "test_event"
        assert event.agent_id == "agent-123"
        assert event.old_state == {"old": "value"}
        assert event.new_state == {"new": "value"}
        assert event.timestamp > 0
        assert event.event_id.startswith("test_event-agent-123-")
    
    def test_event_id_generation(self):
        """测试事件ID生成"""
        event1 = StateChangeEvent("test", "agent1", None, None)
        event2 = StateChangeEvent("test", "agent1", None, None)
        
        # 事件ID应该不同
        assert event1.event_id != event2.event_id