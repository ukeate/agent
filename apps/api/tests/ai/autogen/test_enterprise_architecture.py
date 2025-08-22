"""
企业级架构测试
测试企业级智能体管理器、负载均衡、池化管理等核心功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
import json
import redis.asyncio as redis

from src.ai.autogen.enterprise import (
    EnterpriseAgentManager, AgentPoolConfig, EnterpriseAgentInfo,
    LoadBalancingStrategy, PoolingStrategy, SecurityClearanceLevel,
    AgentPool, LoadBalancer, EnterpriseAuditLogger, RateLimiter
)
from src.ai.autogen.async_manager import AgentStatus, TaskStatus
from src.ai.autogen.events import EventBus, Event, EventType
from src.ai.autogen.enterprise_config import get_config_manager


@pytest.fixture
def event_bus():
    """创建事件总线实例"""
    return EventBus()


@pytest.fixture  
def redis_client():
    """创建模拟Redis客户端"""
    return AsyncMock(spec=redis.Redis)


@pytest.fixture
def agent_pool_config():
    """创建智能体池配置"""
    return AgentPoolConfig(
        min_size=2,
        max_size=8,
        initial_size=3,
        idle_timeout=300,
        scaling_threshold=0.8,
        scaling_factor=1.5,
        pooling_strategy=PoolingStrategy.DYNAMIC
    )


@pytest.fixture
def enterprise_manager(event_bus, redis_client):
    """创建企业级管理器实例"""
    return EnterpriseAgentManager(
        event_bus=event_bus,
        message_queue=AsyncMock(),
        state_manager=AsyncMock(),
        redis_client=redis_client,
        max_concurrent_tasks=50
    )


@pytest.fixture
def sample_agent_info():
    """创建样例智能体信息"""
    return EnterpriseAgentInfo(
        agent_id="test_agent_001",
        name="TestAgent",
        role="assistant",
        status=AgentStatus.IDLE,
        config={},
        pool_id="pool_001",
        resource_usage={"cpu": 0.3, "memory": 0.5},
        performance_metrics={"avg_response_time": 150, "success_rate": 0.95},
        health_score=0.9,
        security_clearance="standard"
    )


class TestAgentPoolConfig:
    """智能体池配置测试"""
    
    def test_from_config(self):
        """测试从配置管理器创建配置"""
        with patch('apps.api.src.ai.autogen.enterprise.get_config_manager') as mock_config:
            mock_manager = Mock()
            mock_manager.get_int.side_effect = lambda key, default: {
                'AGENT_POOL_MIN_SIZE': 2,
                'AGENT_POOL_MAX_SIZE': 15,
                'AGENT_POOL_INITIAL_SIZE': 5,
                'AGENT_POOL_IDLE_TIMEOUT': 600
            }.get(key, default)
            mock_manager.get_float.side_effect = lambda key, default: {
                'AGENT_POOL_SCALING_THRESHOLD': 0.75,
                'AGENT_POOL_SCALING_FACTOR': 2.0
            }.get(key, default)
            mock_manager.get.side_effect = lambda key, default: {
                'AGENT_POOL_STRATEGY': 'auto_scaling'
            }.get(key, default)
            mock_config.return_value = mock_manager
            
            config = AgentPoolConfig.from_config()
            
            assert config.min_size == 2
            assert config.max_size == 15
            assert config.initial_size == 5
            assert config.idle_timeout == 600
            assert config.scaling_threshold == 0.75
            assert config.scaling_factor == 2.0


class TestEnterpriseAgentInfo:
    """企业级智能体信息测试"""
    
    def test_calculate_load_score_idle(self, sample_agent_info):
        """测试空闲状态负载评分计算"""
        sample_agent_info.status = AgentStatus.IDLE
        score = sample_agent_info.calculate_load_score()
        assert score == 0.0
    
    def test_calculate_load_score_busy(self, sample_agent_info):
        """测试忙碌状态负载评分计算"""
        sample_agent_info.status = AgentStatus.BUSY
        score = sample_agent_info.calculate_load_score()
        assert score == 1.0
    
    def test_calculate_load_score_error(self, sample_agent_info):
        """测试错误状态负载评分计算"""
        sample_agent_info.status = AgentStatus.ERROR
        score = sample_agent_info.calculate_load_score()
        assert score >= 0.8  # 错误状态应该有高负载评分
    
    def test_update_health_score(self, sample_agent_info):
        """测试健康评分更新"""
        # 正常响应时间
        sample_agent_info.performance_metrics["avg_response_time"] = 100
        sample_agent_info.performance_metrics["success_rate"] = 0.98
        sample_agent_info.resource_usage["cpu"] = 0.4
        
        initial_score = sample_agent_info.health_score
        sample_agent_info.update_health_score()
        
        assert sample_agent_info.health_score >= initial_score
    
    def test_is_overloaded(self, sample_agent_info):
        """测试过载检测"""
        # 设置高负载
        sample_agent_info.resource_usage["cpu"] = 0.95
        sample_agent_info.resource_usage["memory"] = 0.9
        
        assert sample_agent_info.is_overloaded() is True
        
        # 设置正常负载
        sample_agent_info.resource_usage["cpu"] = 0.5
        sample_agent_info.resource_usage["memory"] = 0.6
        
        assert sample_agent_info.is_overloaded() is False


class TestAgentPool:
    """智能体池测试"""
    
    @pytest.fixture
    def agent_pool(self, agent_pool_config, event_bus):
        """创建智能体池实例"""
        return AgentPool(
            pool_id="test_pool",
            config=agent_pool_config,
            event_bus=event_bus
        )
    
    @pytest.mark.asyncio
    async def test_initialize_pool(self, agent_pool):
        """测试池初始化"""
        await agent_pool.initialize()
        
        assert len(agent_pool.agents) >= agent_pool.config.min_size
        assert len(agent_pool.agents) <= agent_pool.config.max_size
    
    @pytest.mark.asyncio
    async def test_get_available_agent(self, agent_pool, sample_agent_info):
        """测试获取可用智能体"""
        # 添加测试智能体
        sample_agent_info.status = AgentStatus.IDLE
        agent_pool.agents[sample_agent_info.agent_id] = sample_agent_info
        
        agent = await agent_pool.get_available_agent()
        
        assert agent is not None
        assert agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_get_available_agent_no_agents(self, agent_pool):
        """测试无可用智能体情况"""
        # 清空池
        agent_pool.agents.clear()
        
        agent = await agent_pool.get_available_agent()
        
        assert agent is None
    
    @pytest.mark.asyncio
    async def test_add_agent(self, agent_pool, sample_agent_info):
        """测试添加智能体"""
        result = await agent_pool.add_agent(sample_agent_info)
        
        assert result is True
        assert sample_agent_info.agent_id in agent_pool.agents
        assert agent_pool.agents[sample_agent_info.agent_id].pool_id == agent_pool.pool_id
    
    @pytest.mark.asyncio
    async def test_remove_agent(self, agent_pool, sample_agent_info):
        """测试移除智能体"""
        # 先添加智能体
        await agent_pool.add_agent(sample_agent_info)
        
        result = await agent_pool.remove_agent(sample_agent_info.agent_id)
        
        assert result is True
        assert sample_agent_info.agent_id not in agent_pool.agents
    
    @pytest.mark.asyncio
    async def test_scale_up(self, agent_pool):
        """测试扩容"""
        initial_count = len(agent_pool.agents)
        
        await agent_pool.scale_up(2)
        
        assert len(agent_pool.agents) >= initial_count + 2
        assert len(agent_pool.agents) <= agent_pool.config.max_size
    
    @pytest.mark.asyncio
    async def test_scale_down(self, agent_pool, sample_agent_info):
        """测试缩容"""
        # 添加多个智能体
        for i in range(5):
            agent = EnterpriseAgentInfo(
                agent_id=f"agent_{i}",
                name=f"Agent{i}",
                role="assistant",
                status=AgentStatus.IDLE,
                config={}
            )
            await agent_pool.add_agent(agent)
        
        initial_count = len(agent_pool.agents)
        
        await agent_pool.scale_down(2)
        
        assert len(agent_pool.agents) <= initial_count - 2
        assert len(agent_pool.agents) >= agent_pool.config.min_size
    
    def test_get_pool_metrics(self, agent_pool, sample_agent_info):
        """测试获取池指标"""
        # 添加不同状态的智能体
        agent_pool.agents["idle_agent"] = EnterpriseAgentInfo(
            agent_id="idle_agent", name="IdleAgent", role="assistant",
            status=AgentStatus.IDLE, config={}
        )
        agent_pool.agents["busy_agent"] = EnterpriseAgentInfo(
            agent_id="busy_agent", name="BusyAgent", role="assistant", 
            status=AgentStatus.BUSY, config={}
        )
        
        metrics = agent_pool.get_pool_metrics()
        
        assert "total_agents" in metrics
        assert "available_agents" in metrics
        assert "busy_agents" in metrics
        assert "average_load" in metrics
        assert metrics["total_agents"] == 2


class TestLoadBalancer:
    """负载均衡器测试"""
    
    @pytest.fixture
    def load_balancer(self):
        """创建负载均衡器实例"""
        return LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOADED)
    
    @pytest.fixture
    def agents_list(self):
        """创建测试智能体列表"""
        agents = []
        for i in range(3):
            agent = EnterpriseAgentInfo(
                agent_id=f"agent_{i}",
                name=f"Agent{i}",
                role="assistant",
                status=AgentStatus.IDLE,
                config={},
                resource_usage={"cpu": 0.2 + i * 0.2, "memory": 0.3 + i * 0.1}
            )
            agents.append(agent)
        return agents
    
    def test_select_agent_least_loaded(self, load_balancer, agents_list):
        """测试最少负载策略"""
        load_balancer.strategy = LoadBalancingStrategy.LEAST_LOADED
        
        selected_agent = load_balancer.select_agent(agents_list)
        
        # 应该选择负载最低的智能体（agent_0）
        assert selected_agent.agent_id == "agent_0"
    
    def test_select_agent_round_robin(self, load_balancer, agents_list):
        """测试轮询策略"""
        load_balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # 连续选择应该轮询
        selected_agents = []
        for _ in range(6):  # 两轮
            agent = load_balancer.select_agent(agents_list)
            selected_agents.append(agent.agent_id)
        
        # 应该有轮询模式
        assert selected_agents[0] != selected_agents[3] or len(set(selected_agents[:3])) == 3
    
    def test_select_agent_weighted(self, load_balancer, agents_list):
        """测试权重策略"""
        load_balancer.strategy = LoadBalancingStrategy.WEIGHTED
        
        # 设置权重
        for i, agent in enumerate(agents_list):
            agent.performance_metrics["weight"] = 1.0 + i * 0.5
        
        selected_agent = load_balancer.select_agent(agents_list)
        
        # 应该选择权重最高的智能体
        assert selected_agent.agent_id == "agent_2"
    
    def test_select_agent_no_available(self, load_balancer):
        """测试无可用智能体"""
        result = load_balancer.select_agent([])
        assert result is None
        
        # 测试所有智能体都忙碌
        busy_agents = [
            EnterpriseAgentInfo(
                agent_id="busy_agent",
                name="BusyAgent", 
                role="assistant",
                status=AgentStatus.BUSY,
                config={}
            )
        ]
        
        result = load_balancer.select_agent(busy_agents)
        assert result is None


class TestEnterpriseAgentManager:
    """企业级智能体管理器测试"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, enterprise_manager):
        """测试管理器初始化"""
        await enterprise_manager.start()
        
        assert enterprise_manager.running is True
        assert len(enterprise_manager.agent_pools) >= 0
        assert enterprise_manager.load_balancer is not None
        assert enterprise_manager.monitoring is not None
    
    @pytest.mark.asyncio
    async def test_create_agent_pool(self, enterprise_manager, agent_pool_config):
        """测试创建智能体池"""
        pool_id = "test_pool_create"
        
        pool = await enterprise_manager.create_agent_pool(pool_id, agent_pool_config)
        
        assert pool is not None
        assert pool.pool_id == pool_id
        assert pool_id in enterprise_manager.agent_pools
    
    @pytest.mark.asyncio
    async def test_create_agent(self, enterprise_manager):
        """测试创建智能体"""
        agent_config = {
            "name": "TestAgent",
            "role": "assistant",
            "system_prompt": "You are a helpful assistant."
        }
        
        agent = await enterprise_manager.create_agent(agent_config)
        
        assert agent is not None
        assert isinstance(agent, EnterpriseAgentInfo)
        assert agent.name == "TestAgent"
    
    @pytest.mark.asyncio  
    async def test_assign_task_with_load_balancing(self, enterprise_manager, sample_agent_info):
        """测试负载均衡任务分配"""
        # 创建测试池并添加智能体
        pool_config = AgentPoolConfig(min_size=1, max_size=5)
        pool = await enterprise_manager.create_agent_pool("test_pool", pool_config)
        await pool.add_agent(sample_agent_info)
        
        # 创建测试任务
        task = {
            "task_id": "test_task_001",
            "type": "conversation",
            "content": "Hello, how are you?",
            "priority": "normal"
        }
        
        result = await enterprise_manager.assign_task(task)
        
        assert result["success"] is True
        assert "assigned_agent" in result
        assert result["assigned_agent"] == sample_agent_info.agent_id
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, enterprise_manager):
        """测试获取系统指标"""
        metrics = await enterprise_manager.get_system_metrics()
        
        assert "total_pools" in metrics
        assert "total_agents" in metrics  
        assert "active_tasks" in metrics
        assert "system_load" in metrics
        assert "resource_utilization" in metrics
    
    @pytest.mark.asyncio
    async def test_health_check(self, enterprise_manager):
        """测试健康检查"""
        health_status = await enterprise_manager.health_check()
        
        assert "status" in health_status
        assert "checks" in health_status
        assert "timestamp" in health_status
        
        # 验证各个组件的健康状态
        checks = health_status["checks"]
        expected_checks = ["event_bus", "redis", "agent_pools", "load_balancer"]
        for check in expected_checks:
            assert check in checks
    
    @pytest.mark.asyncio
    async def test_automatic_scaling(self, enterprise_manager, agent_pool_config):
        """测试自动扩缩容"""
        # 创建池
        pool_id = "scaling_test_pool"
        pool = await enterprise_manager.create_agent_pool(pool_id, agent_pool_config)
        
        # 模拟高负载触发扩容
        with patch.object(pool, 'get_pool_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "total_agents": 3,
                "available_agents": 0,
                "busy_agents": 3,
                "average_load": 0.9  # 高负载
            }
            
            await enterprise_manager._check_and_scale_pools()
            
            # 验证扩容被触发
            assert len(pool.agents) >= agent_pool_config.min_size
    
    @pytest.mark.asyncio
    async def test_security_integration(self, enterprise_manager):
        """测试安全集成"""
        # 测试安全配置
        security_config = enterprise_manager.get_security_config()
        
        assert "clearance_levels" in security_config
        assert "access_policies" in security_config
        
        # 测试访问控制
        agent_id = "test_agent"
        resource = "sensitive_data"
        clearance_level = "standard"
        
        access_granted = await enterprise_manager.check_access_permission(
            agent_id, resource, clearance_level
        )
        
        assert isinstance(access_granted, bool)
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, enterprise_manager, sample_agent_info):
        """测试错误恢复"""
        # 模拟智能体错误
        sample_agent_info.status = AgentStatus.ERROR
        sample_agent_info.failover_count = 0
        
        result = await enterprise_manager.handle_agent_error(
            sample_agent_info.agent_id, "Connection timeout"
        )
        
        assert result["handled"] is True
        assert "recovery_action" in result
        
        # 验证错误计数增加
        assert sample_agent_info.failover_count > 0


class TestRateLimiter:
    """限流器测试"""
    
    @pytest.fixture
    def rate_limiter(self):
        """创建限流器实例"""
        return RateLimiter()
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limit(self, rate_limiter):
        """测试限制内的请求"""
        key = "test_user"
        
        # 在限制内的请求应该被允许
        for i in range(5):
            allowed = await rate_limiter.check_rate_limit(key)
            assert allowed is True
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter):
        """测试超出限制的请求"""
        key = "test_user_exceeded"
        
        # 超出限制的请求
        for i in range(12):  # 超过默认限制10
            allowed = await rate_limiter.check_rate_limit(key)
            if i < 10:
                assert allowed is True
            else:
                assert allowed is False
    
    @pytest.mark.asyncio
    async def test_reset_rate_limit(self, rate_limiter):
        """测试重置限流"""
        key = "test_user_reset"
        
        # 先达到限制
        for _ in range(12):
            await rate_limiter.check_rate_limit(key)
        
        # 重置
        await rate_limiter.reset_rate_limit(key)
        
        # 应该可以再次请求
        allowed = await rate_limiter.check_rate_limit(key)
        assert allowed is True


class TestEnterpriseAuditLogger:
    """企业审计日志测试"""
    
    @pytest.fixture
    def audit_logger(self, event_bus):
        """创建审计日志实例"""
        return EnterpriseAuditLogger(event_bus)
    
    @pytest.mark.asyncio
    async def test_log_action(self, audit_logger):
        """测试记录行为"""
        await audit_logger.log_action(
            user_id="test_user",
            action="create_agent",
            resource="agent_001",
            result="success",
            details={"agent_type": "assistant"}
        )
        
        assert len(audit_logger.audit_logs) == 1
        
        log_entry = audit_logger.audit_logs[0]
        assert log_entry.user_id == "test_user"
        assert log_entry.action == "create_agent"
        assert log_entry.resource == "agent_001"
        assert log_entry.result == "success"
    
    @pytest.mark.asyncio
    async def test_log_security_event(self, audit_logger):
        """测试记录安全事件"""
        await audit_logger.log_security_event(
            event_type="unauthorized_access",
            severity="high",
            source="agent_002",
            details={"attempted_resource": "classified_data"}
        )
        
        # 查找安全事件日志
        security_logs = [log for log in audit_logger.audit_logs 
                        if log.action == "security_event"]
        
        assert len(security_logs) == 1
        assert security_logs[0].details["event_type"] == "unauthorized_access"
    
    def test_log_rotation(self, audit_logger):
        """测试日志轮转"""
        # 填充日志到最大容量
        original_max = audit_logger.max_logs
        audit_logger.max_logs = 5  # 设置较小的限制用于测试
        
        for i in range(10):
            audit_logger.audit_logs.append(
                type('AuditLogEntry', (), {
                    'timestamp': datetime.now(),
                    'user_id': f'user_{i}',
                    'action': 'test',
                    'resource': 'test',
                    'result': 'success',
                    'details': {}
                })()
            )
        
        audit_logger._rotate_logs()
        
        # 应该保留最新的日志
        assert len(audit_logger.audit_logs) == 5
        
        # 恢复原始设置
        audit_logger.max_logs = original_max


@pytest.mark.integration
class TestEnterpriseIntegration:
    """企业级组件集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_enterprise_workflow(self, event_bus, redis_client):
        """测试完整企业级工作流"""
        # 1. 创建企业级管理器
        manager = EnterpriseAgentManager(
            event_bus=event_bus,
            message_queue=AsyncMock(),
            state_manager=AsyncMock(), 
            redis_client=redis_client,
            max_concurrent_tasks=10
        )
        
        await manager.start()
        
        # 2. 创建智能体池
        pool_config = AgentPoolConfig(min_size=2, max_size=5, initial_size=2)
        pool = await manager.create_agent_pool("integration_pool", pool_config)
        
        # 3. 创建智能体
        agent_configs = [
            {"name": "Agent1", "role": "assistant", "system_prompt": "Help with general queries"},
            {"name": "Agent2", "role": "specialist", "system_prompt": "Help with technical queries"}
        ]
        
        agents = []
        for config in agent_configs:
            agent = await manager.create_agent(config)
            agents.append(agent)
            await pool.add_agent(agent)
        
        # 4. 分配任务
        tasks = [
            {"task_id": "task_1", "type": "general", "content": "What's the weather?"},
            {"task_id": "task_2", "type": "technical", "content": "Explain machine learning"},
        ]
        
        results = []
        for task in tasks:
            result = await manager.assign_task(task)
            results.append(result)
        
        # 5. 验证结果
        assert all(result["success"] for result in results)
        assert len(set(result["assigned_agent"] for result in results)) >= 1
        
        # 6. 检查系统指标
        metrics = await manager.get_system_metrics()
        assert metrics["total_pools"] >= 1
        assert metrics["total_agents"] >= 2
        
        # 7. 健康检查
        health = await manager.health_check()
        assert health["status"] in ["healthy", "degraded"]
        
        # 8. 清理
        await manager.stop()


@pytest.mark.performance
class TestEnterprisePerformance:
    """企业级性能测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_assignment(self, enterprise_manager):
        """测试并发任务分配性能"""
        await enterprise_manager.start()
        
        # 创建多个智能体
        pool_config = AgentPoolConfig(min_size=5, max_size=10)
        pool = await enterprise_manager.create_agent_pool("perf_pool", pool_config)
        
        for i in range(5):
            agent = await enterprise_manager.create_agent({
                "name": f"PerfAgent{i}",
                "role": "assistant"
            })
            await pool.add_agent(agent)
        
        # 并发分配任务
        tasks = [
            {"task_id": f"perf_task_{i}", "type": "test", "content": f"Task {i}"}
            for i in range(20)
        ]
        
        start_time = datetime.now()
        
        # 并发执行任务分配
        results = await asyncio.gather(
            *[enterprise_manager.assign_task(task) for task in tasks],
            return_exceptions=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 验证性能
        successful_assignments = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        
        assert successful_assignments >= 15  # 至少75%成功率
        assert duration < 5.0  # 应该在5秒内完成
        
        # 验证负载均衡
        assigned_agents = [r["assigned_agent"] for r in results if isinstance(r, dict) and r.get("success")]
        unique_agents = set(assigned_agents)
        assert len(unique_agents) >= 3  # 任务应该分布到多个智能体
        
        await enterprise_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__])