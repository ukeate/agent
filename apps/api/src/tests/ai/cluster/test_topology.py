"""
测试集群拓扑模型
"""

import pytest
import time
from unittest.mock import Mock
from src.ai.cluster.topology import (
    AgentInfo, AgentStatus, AgentCapability, ResourceSpec, ResourceUsage,
    AgentHealthCheck, AgentGroup, ClusterTopology
)

class TestResourceSpec:
    """测试资源规格"""
    
    def test_resource_spec_creation(self):
        """测试资源规格创建"""
        spec = ResourceSpec(
            cpu_cores=2.0,
            memory_gb=8.0,
            storage_gb=100.0,
            gpu_count=1,
            network_bandwidth=1000.0
        )
        
        assert spec.cpu_cores == 2.0
        assert spec.memory_gb == 8.0
        assert spec.storage_gb == 100.0
        assert spec.gpu_count == 1
        assert spec.network_bandwidth == 1000.0
    
    def test_resource_spec_addition(self):
        """测试资源规格相加"""
        spec1 = ResourceSpec(cpu_cores=2.0, memory_gb=8.0, gpu_count=1)
        spec2 = ResourceSpec(cpu_cores=1.0, memory_gb=4.0, gpu_count=0)
        
        total = spec1 + spec2
        
        assert total.cpu_cores == 3.0
        assert total.memory_gb == 12.0
        assert total.gpu_count == 1
    
    def test_resource_spec_multiplication(self):
        """测试资源规格缩放"""
        spec = ResourceSpec(cpu_cores=2.0, memory_gb=8.0, gpu_count=1)
        scaled = spec * 2.0
        
        assert scaled.cpu_cores == 4.0
        assert scaled.memory_gb == 16.0
        assert scaled.gpu_count == 2

class TestResourceUsage:
    """测试资源使用情况"""
    
    def test_resource_usage_creation(self):
        """测试资源使用创建"""
        usage = ResourceUsage(
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            active_tasks=5,
            total_requests=100,
            failed_requests=5
        )
        
        assert usage.cpu_usage_percent == 50.0
        assert usage.memory_usage_percent == 60.0
        assert usage.active_tasks == 5
        assert usage.error_rate == 0.05
    
    def test_error_rate_calculation(self):
        """测试错误率计算"""
        usage = ResourceUsage(total_requests=100, failed_requests=10)
        assert usage.error_rate == 0.1
        
        # 测试零除法
        usage_no_requests = ResourceUsage(total_requests=0, failed_requests=0)
        assert usage_no_requests.error_rate == 0.0

class TestAgentHealthCheck:
    """测试智能体健康检查"""
    
    def test_health_check_creation(self):
        """测试健康检查创建"""
        health = AgentHealthCheck(
            is_healthy=True,
            health_check_interval=30.0,
            max_failures=3
        )
        
        assert health.is_healthy is True
        assert health.health_check_interval == 30.0
        assert health.max_failures == 3
        assert health.consecutive_failures == 0
    
    def test_is_responsive(self):
        """测试响应性检查"""
        current_time = time.time()
        health = AgentHealthCheck(last_heartbeat=current_time, health_check_interval=30.0)
        assert health.is_responsive is True
        
        # 测试超时情况
        old_heartbeat = current_time - 100
        health_timeout = AgentHealthCheck(last_heartbeat=old_heartbeat, health_check_interval=30.0)
        assert health_timeout.is_responsive is False
    
    def test_needs_restart(self):
        """测试重启需求检查"""
        health = AgentHealthCheck(consecutive_failures=2, max_failures=3)
        assert health.needs_restart is False
        
        health.consecutive_failures = 3
        assert health.needs_restart is True

class TestAgentInfo:
    """测试智能体信息"""
    
    def test_agent_info_creation(self):
        """测试智能体信息创建"""
        agent = AgentInfo(
            name="test-agent",
            host="localhost",
            port=8080,
            capabilities={AgentCapability.COMPUTE, AgentCapability.REASONING}
        )
        
        assert agent.name == "test-agent"
        assert agent.host == "localhost" 
        assert agent.port == 8080
        assert agent.endpoint == "http://localhost:8080"
        assert AgentCapability.COMPUTE in agent.capabilities
        assert AgentCapability.REASONING in agent.capabilities
        assert agent.status == AgentStatus.PENDING
    
    def test_post_init_endpoint_creation(self):
        """测试endpoint自动创建"""
        agent = AgentInfo(host="192.168.1.100", port=9090)
        assert agent.endpoint == "http://192.168.1.100:9090"
    
    def test_update_status(self):
        """测试状态更新"""
        agent = AgentInfo()
        original_time = agent.updated_at
        
        time.sleep(0.01)  # 确保时间差异
        agent.update_status(AgentStatus.RUNNING, "Started successfully")
        
        assert agent.status == AgentStatus.RUNNING
        assert agent.updated_at > original_time
        assert agent.started_at is not None
        assert agent.metadata["status_details"] == "Started successfully"
    
    def test_update_resource_usage(self):
        """测试资源使用更新"""
        agent = AgentInfo()
        original_time = agent.updated_at
        
        usage = ResourceUsage(cpu_usage_percent=75.0, memory_usage_percent=60.0)
        time.sleep(0.01)
        agent.update_resource_usage(usage)
        
        assert agent.resource_usage.cpu_usage_percent == 75.0
        assert agent.resource_usage.memory_usage_percent == 60.0
        assert agent.updated_at > original_time
    
    def test_add_label(self):
        """测试标签添加"""
        agent = AgentInfo()
        original_time = agent.updated_at
        
        time.sleep(0.01)
        agent.add_label("environment", "production")
        
        assert agent.labels["environment"] == "production"
        assert agent.updated_at > original_time
    
    def test_has_capability(self):
        """测试能力检查"""
        agent = AgentInfo(capabilities={AgentCapability.COMPUTE})
        
        assert agent.has_capability(AgentCapability.COMPUTE) is True
        assert agent.has_capability(AgentCapability.REASONING) is False
    
    def test_is_healthy(self):
        """测试健康状态"""
        agent = AgentInfo(status=AgentStatus.RUNNING)
        agent.health.is_healthy = True
        agent.health.last_heartbeat = time.time()
        
        assert agent.is_healthy is True
        
        # 测试不健康情况
        agent.health.is_healthy = False
        assert agent.is_healthy is False
    
    def test_uptime_seconds(self):
        """测试运行时间"""
        agent = AgentInfo()
        assert agent.uptime_seconds == 0.0
        
        # 设置启动时间
        current_time = time.time()
        agent.started_at = current_time - 3600  # 1小时前启动
        
        uptime = agent.uptime_seconds
        assert uptime >= 3599  # 允许1秒误差

class TestAgentGroup:
    """测试智能体分组"""
    
    def test_agent_group_creation(self):
        """测试分组创建"""
        group = AgentGroup(
            name="test-group",
            description="Test group",
            max_agents=5,
            min_agents=1
        )
        
        assert group.name == "test-group"
        assert group.description == "Test group"
        assert group.max_agents == 5
        assert group.min_agents == 1
        assert group.agent_count == 0
    
    def test_post_init_name_generation(self):
        """测试名称自动生成"""
        group = AgentGroup()
        assert group.name == group.group_id
    
    def test_add_agent(self):
        """测试添加智能体"""
        group = AgentGroup(max_agents=2)
        
        # 成功添加
        assert group.add_agent("agent1") is True
        assert "agent1" in group.agent_ids
        assert group.agent_count == 1
        
        # 再添加一个
        assert group.add_agent("agent2") is True
        assert group.agent_count == 2
        
        # 超过最大限制
        assert group.add_agent("agent3") is False
        assert group.agent_count == 2
    
    def test_remove_agent(self):
        """测试移除智能体"""
        group = AgentGroup(min_agents=1)
        group.add_agent("agent1")
        group.add_agent("agent2")
        
        # 成功移除
        assert group.remove_agent("agent2") is True
        assert "agent2" not in group.agent_ids
        assert group.agent_count == 1
        
        # 不能低于最小限制
        assert group.remove_agent("agent1") is False
        assert group.agent_count == 1
    
    def test_is_full(self):
        """测试满员检查"""
        group = AgentGroup(max_agents=2)
        
        assert group.is_full is False
        
        group.add_agent("agent1")
        assert group.is_full is False
        
        group.add_agent("agent2")
        assert group.is_full is True
    
    def test_can_scale_down(self):
        """测试缩容检查"""
        group = AgentGroup(min_agents=1)
        group.add_agent("agent1")
        
        assert group.can_scale_down is False
        
        group.add_agent("agent2")
        assert group.can_scale_down is True

class TestClusterTopology:
    """测试集群拓扑"""
    
    def test_cluster_topology_creation(self):
        """测试集群拓扑创建"""
        topology = ClusterTopology(
            cluster_id="test-cluster",
            name="Test Cluster",
            description="Test cluster description"
        )
        
        assert topology.cluster_id == "test-cluster"
        assert topology.name == "Test Cluster"
        assert topology.description == "Test cluster description"
        assert len(topology.agents) == 0
        assert len(topology.groups) == 0
    
    def test_post_init_name_generation(self):
        """测试名称自动生成"""
        topology = ClusterTopology(cluster_id="test-cluster")
        assert topology.name == "test-cluster"
    
    def test_add_agent(self):
        """测试添加智能体"""
        topology = ClusterTopology(cluster_id="test-cluster")
        agent = AgentInfo(name="test-agent")
        
        success = topology.add_agent(agent)
        
        assert success is True
        assert agent.agent_id in topology.agents
        assert agent.cluster_id == "test-cluster"
        assert topology.total_agents == 1
    
    def test_remove_agent(self):
        """测试移除智能体"""
        topology = ClusterTopology()
        agent = AgentInfo(name="test-agent")
        
        # 添加智能体
        topology.add_agent(agent)
        agent_id = agent.agent_id
        
        # 添加到分组
        group = AgentGroup()
        topology.add_group(group)
        group.add_agent(agent_id)
        
        # 添加依赖关系
        topology.add_dependency(agent_id, "other-agent")
        topology.set_communication_latency(agent_id, "other-agent", 10.0)
        
        # 移除智能体
        success = topology.remove_agent(agent_id)
        
        assert success is True
        assert agent_id not in topology.agents
        assert agent_id not in group.agent_ids
        assert agent_id not in topology.agent_dependencies
        assert agent_id not in topology.communication_paths
    
    def test_get_agent(self):
        """测试获取智能体"""
        topology = ClusterTopology()
        agent = AgentInfo(name="test-agent")
        topology.add_agent(agent)
        
        retrieved = topology.get_agent(agent.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == agent.agent_id
        
        # 测试不存在的智能体
        assert topology.get_agent("non-existent") is None
    
    def test_add_group(self):
        """测试添加分组"""
        topology = ClusterTopology()
        group = AgentGroup(name="test-group")
        
        success = topology.add_group(group)
        
        assert success is True
        assert group.group_id in topology.groups
    
    def test_get_agents_by_status(self):
        """测试按状态获取智能体"""
        topology = ClusterTopology()
        
        # 添加不同状态的智能体
        agent1 = AgentInfo(name="running-agent")
        agent1.status = AgentStatus.RUNNING
        agent2 = AgentInfo(name="stopped-agent") 
        agent2.status = AgentStatus.STOPPED
        agent3 = AgentInfo(name="running-agent2")
        agent3.status = AgentStatus.RUNNING
        
        topology.add_agent(agent1)
        topology.add_agent(agent2)
        topology.add_agent(agent3)
        
        running_agents = topology.get_agents_by_status(AgentStatus.RUNNING)
        stopped_agents = topology.get_agents_by_status(AgentStatus.STOPPED)
        
        assert len(running_agents) == 2
        assert len(stopped_agents) == 1
    
    def test_get_agents_by_capability(self):
        """测试按能力获取智能体"""
        topology = ClusterTopology()
        
        agent1 = AgentInfo(name="compute-agent", capabilities={AgentCapability.COMPUTE})
        agent2 = AgentInfo(name="reasoning-agent", capabilities={AgentCapability.REASONING})
        agent3 = AgentInfo(name="multimodal-agent", capabilities={AgentCapability.COMPUTE, AgentCapability.MULTIMODAL})
        
        topology.add_agent(agent1)
        topology.add_agent(agent2)
        topology.add_agent(agent3)
        
        compute_agents = topology.get_agents_by_capability(AgentCapability.COMPUTE)
        reasoning_agents = topology.get_agents_by_capability(AgentCapability.REASONING)
        multimodal_agents = topology.get_agents_by_capability(AgentCapability.MULTIMODAL)
        
        assert len(compute_agents) == 2
        assert len(reasoning_agents) == 1
        assert len(multimodal_agents) == 1
    
    def test_get_healthy_agents(self):
        """测试获取健康智能体"""
        topology = ClusterTopology()
        
        # 健康智能体
        agent1 = AgentInfo(name="healthy-agent")
        agent1.status = AgentStatus.RUNNING
        agent1.health.is_healthy = True
        agent1.health.last_heartbeat = time.time()
        
        # 不健康智能体
        agent2 = AgentInfo(name="unhealthy-agent")
        agent2.status = AgentStatus.RUNNING
        agent2.health.is_healthy = False
        
        topology.add_agent(agent1)
        topology.add_agent(agent2)
        
        healthy_agents = topology.get_healthy_agents()
        
        assert len(healthy_agents) == 1
        assert healthy_agents[0].agent_id == agent1.agent_id
    
    def test_dependency_management(self):
        """测试依赖关系管理"""
        topology = ClusterTopology()
        
        topology.add_dependency("agent1", "agent2")
        topology.add_dependency("agent1", "agent3")
        
        deps = topology.get_dependencies("agent1")
        
        assert "agent2" in deps
        assert "agent3" in deps
        assert len(deps) == 2
    
    def test_communication_latency(self):
        """测试通信延迟管理"""
        topology = ClusterTopology()
        
        topology.set_communication_latency("agent1", "agent2", 15.5)
        
        latency = topology.get_communication_latency("agent1", "agent2")
        assert latency == 15.5
        
        # 测试不存在的路径
        no_latency = topology.get_communication_latency("agent1", "agent3")
        assert no_latency is None
    
    def test_cluster_properties(self):
        """测试集群属性"""
        topology = ClusterTopology()
        
        # 添加不同状态的智能体
        agent1 = AgentInfo(name="running-agent")
        agent1.status = AgentStatus.RUNNING
        agent1.health.is_healthy = True
        agent1.health.last_heartbeat = time.time()
        
        agent2 = AgentInfo(name="stopped-agent")
        agent2.status = AgentStatus.STOPPED
        
        topology.add_agent(agent1)
        topology.add_agent(agent2)
        
        assert topology.total_agents == 2
        assert topology.running_agents == 1
        assert topology.healthy_agents == 1
    
    def test_cluster_resource_usage(self):
        """测试集群资源使用聚合"""
        topology = ClusterTopology()
        
        # 添加有资源使用的智能体
        agent1 = AgentInfo(name="agent1")
        agent1.status = AgentStatus.RUNNING
        agent1.health.is_healthy = True
        agent1.health.last_heartbeat = time.time()
        agent1.resource_usage = ResourceUsage(
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            active_tasks=5,
            total_requests=100,
            failed_requests=5,
            avg_response_time=200.0
        )
        
        agent2 = AgentInfo(name="agent2")
        agent2.status = AgentStatus.RUNNING
        agent2.health.is_healthy = True
        agent2.health.last_heartbeat = time.time()
        agent2.resource_usage = ResourceUsage(
            cpu_usage_percent=30.0,
            memory_usage_percent=40.0,
            active_tasks=3,
            total_requests=50,
            failed_requests=2,
            avg_response_time=150.0
        )
        
        topology.add_agent(agent1)
        topology.add_agent(agent2)
        
        cluster_usage = topology.cluster_resource_usage
        
        # 验证平均值计算
        assert cluster_usage.cpu_usage_percent == 40.0  # (50+30)/2
        assert cluster_usage.memory_usage_percent == 50.0  # (60+40)/2
        assert cluster_usage.active_tasks == 8  # 5+3
        assert cluster_usage.total_requests == 150  # 100+50
        assert cluster_usage.failed_requests == 7  # 5+2
        assert cluster_usage.error_rate == 7/150  # 总失败/总请求
        
        # 加权平均响应时间 (200*100 + 150*50) / (100+50) = 183.33
        expected_avg_response = (200.0 * 100 + 150.0 * 50) / 150
        assert abs(cluster_usage.avg_response_time - expected_avg_response) < 0.01
    
    def test_cluster_health_score(self):
        """测试集群健康评分"""
        topology = ClusterTopology()
        
        # 添加健康智能体
        agent1 = AgentInfo(name="healthy-agent")
        agent1.status = AgentStatus.RUNNING
        agent1.health.is_healthy = True
        agent1.health.last_heartbeat = time.time()
        agent1.resource_usage = ResourceUsage(
            cpu_usage_percent=50.0,  # 理想范围内
            total_requests=100,
            failed_requests=1  # 1% 错误率
        )
        
        # 添加不健康智能体
        agent2 = AgentInfo(name="unhealthy-agent")
        agent2.status = AgentStatus.FAILED
        agent2.health.is_healthy = False
        
        topology.add_agent(agent1)
        topology.add_agent(agent2)
        
        health_score = topology.cluster_health_score
        
        # 健康评分应该在0-1之间
        assert 0 <= health_score <= 1
        # 由于有一半智能体不健康，评分应该较低
        assert health_score < 0.8
    
    def test_to_dict(self):
        """测试转换为字典"""
        topology = ClusterTopology(cluster_id="test", name="Test Cluster")
        
        agent = AgentInfo(name="test-agent")
        agent.status = AgentStatus.RUNNING
        topology.add_agent(agent)
        
        group = AgentGroup(name="test-group")
        topology.add_group(group)
        
        data = topology.to_dict()
        
        assert data["cluster_id"] == "test"
        assert data["name"] == "Test Cluster"
        assert data["total_agents"] == 1
        assert data["running_agents"] == 1
        assert agent.agent_id in data["agents"]
        assert group.group_id in data["groups"]
        assert "cluster_resource_usage" in data
        assert "health_score" in data
