import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import aiohttp

from ....ai.fault_tolerance.fault_detector import (
    FaultDetector, 
    FaultType, 
    FaultSeverity, 
    FaultEvent, 
    HealthStatus
)

@pytest.fixture
def mock_cluster_manager():
    cluster_manager = Mock()
    cluster_manager.get_agent_info = AsyncMock()
    cluster_manager.get_cluster_topology = AsyncMock()
    return cluster_manager

@pytest.fixture
def mock_metrics_collector():
    metrics_collector = Mock()
    metrics_collector.get_recent_metrics = AsyncMock()
    return metrics_collector

@pytest.fixture
def fault_detector_config():
    return {
        "health_check_interval": 1,  # 1秒用于快速测试
        "response_timeout": 2.0,
        "error_rate_threshold": 0.1,
        "performance_threshold": 1.0
    }

@pytest.fixture
def fault_detector(mock_cluster_manager, mock_metrics_collector, fault_detector_config):
    return FaultDetector(
        cluster_manager=mock_cluster_manager,
        metrics_collector=mock_metrics_collector,
        config=fault_detector_config
    )

@pytest.mark.asyncio
class TestFaultDetector:
    
    async def test_fault_detector_initialization(self, fault_detector):
        """测试故障检测器初始化"""
        assert fault_detector.health_check_interval == 1
        assert fault_detector.response_timeout == 2.0
        assert fault_detector.error_rate_threshold == 0.1
        assert fault_detector.performance_threshold == 1.0
        assert fault_detector.running is False
        assert len(fault_detector.health_status) == 0
        assert len(fault_detector.fault_events) == 0
    
    async def test_register_fault_callback(self, fault_detector):
        """测试注册故障回调"""
        callback = AsyncMock()
        fault_detector.register_fault_callback(callback)
        
        assert len(fault_detector.fault_callbacks) == 1
        assert fault_detector.fault_callbacks[0] == callback
    
    async def test_check_component_health_success(self, fault_detector, mock_cluster_manager, mock_metrics_collector):
        """测试组件健康检查成功"""
        # 模拟组件信息
        component_info = Mock()
        component_info.endpoint = "http://agent-1:8000"
        mock_cluster_manager.get_agent_info.return_value = component_info
        
        # 模拟指标数据
        metric = Mock()
        metric.error_rate = 0.05
        metric.cpu_usage = 45.0
        metric.memory_usage = 60.0
        metric.disk_usage = 30.0
        mock_metrics_collector.get_recent_metrics.return_value = [metric]
        
        # 模拟HTTP健康检查
        with patch.object(fault_detector, '_make_http_health_request') as mock_http_request:
            mock_http_request.return_value = {"status": "healthy", "custom_metrics": {"test": 1}}
            
            health_status = await fault_detector.check_component_health("agent-1")
            
            assert health_status.component_id == "agent-1"
            assert health_status.status == "healthy"
            assert health_status.error_rate == 0.05
            assert health_status.resource_usage["cpu"] == 45.0
            assert health_status.resource_usage["memory"] == 60.0
            assert health_status.custom_metrics["test"] == 1
    
    async def test_check_component_health_degraded(self, fault_detector, mock_cluster_manager, mock_metrics_collector):
        """测试组件健康检查降级"""
        component_info = Mock()
        component_info.endpoint = "http://agent-1:8000"
        mock_cluster_manager.get_agent_info.return_value = component_info
        
        # 模拟高错误率
        metric = Mock()
        metric.error_rate = 0.15  # 超过阈值
        metric.cpu_usage = 95.0   # 高CPU使用率
        metric.memory_usage = 85.0
        metric.disk_usage = 30.0
        mock_metrics_collector.get_recent_metrics.return_value = [metric]
        
        with patch.object(fault_detector, '_make_http_health_request') as mock_http_request:
            mock_http_request.return_value = {"status": "healthy"}
            
            health_status = await fault_detector.check_component_health("agent-1")
            
            assert health_status.component_id == "agent-1"
            assert health_status.status == "degraded"
    
    async def test_check_component_health_unhealthy(self, fault_detector, mock_cluster_manager):
        """测试组件健康检查不健康"""
        component_info = Mock()
        component_info.endpoint = "http://agent-1:8000"
        mock_cluster_manager.get_agent_info.return_value = component_info
        
        # 模拟网络超时
        with patch.object(fault_detector, '_make_http_health_request') as mock_http_request:
            mock_http_request.side_effect = asyncio.TimeoutError()
            
            health_status = await fault_detector.check_component_health("agent-1")
            
            assert health_status.component_id == "agent-1"
            assert health_status.status == "unhealthy"
            assert health_status.response_time == float('inf')
            assert health_status.error_rate == 1.0
    
    async def test_check_component_health_unknown(self, fault_detector, mock_cluster_manager):
        """测试组件健康检查未知状态"""
        # 模拟组件不存在
        mock_cluster_manager.get_agent_info.return_value = None
        
        health_status = await fault_detector.check_component_health("unknown-agent")
        
        assert health_status.component_id == "unknown-agent"
        assert health_status.status == "unknown"
        assert health_status.response_time == 0.0
        assert health_status.error_rate == 1.0
    
    async def test_determine_health_status(self, fault_detector):
        """测试健康状态判断"""
        # 健康状态
        status = fault_detector._determine_health_status(
            {"status": "healthy"}, 0.5, 0.01, {"cpu": 30, "memory": 40}
        )
        assert status == "healthy"
        
        # 不健康状态
        status = fault_detector._determine_health_status(
            {"status": "unhealthy"}, 0.5, 0.01, {"cpu": 30, "memory": 40}
        )
        assert status == "unhealthy"
        
        # 性能降级
        status = fault_detector._determine_health_status(
            {"status": "healthy"}, 3.0, 0.01, {"cpu": 30, "memory": 40}
        )
        assert status == "degraded"
        
        # 高错误率
        status = fault_detector._determine_health_status(
            {"status": "healthy"}, 0.5, 0.15, {"cpu": 30, "memory": 40}
        )
        assert status == "degraded"
        
        # 资源耗尽
        status = fault_detector._determine_health_status(
            {"status": "healthy"}, 0.5, 0.01, {"cpu": 95, "memory": 40}
        )
        assert status == "degraded"
    
    async def test_detect_faults_from_health(self, fault_detector):
        """测试从健康状态检测故障"""
        callback = AsyncMock()
        fault_detector.register_fault_callback(callback)
        
        # 测试无响应故障
        health_status = HealthStatus(
            component_id="agent-1",
            status="unhealthy",
            last_check=datetime.now(),
            response_time=float('inf'),
            error_rate=1.0,
            resource_usage={}
        )
        
        await fault_detector._detect_faults_from_health(health_status)
        
        # 验证回调被调用
        callback.assert_called_once()
        fault_event = callback.call_args[0][0]
        assert fault_event.fault_type == FaultType.AGENT_UNRESPONSIVE
        assert fault_event.severity == FaultSeverity.HIGH
        assert "agent-1" in fault_event.affected_components
        
        # 测试性能降级故障
        callback.reset_mock()
        health_status.status = "degraded"
        health_status.response_time = 1.5
        
        await fault_detector._detect_faults_from_health(health_status)
        
        callback.assert_called_once()
        fault_event = callback.call_args[0][0]
        assert fault_event.fault_type == FaultType.PERFORMANCE_DEGRADATION
        assert fault_event.severity == FaultSeverity.LOW
        
        # 测试资源耗尽故障
        callback.reset_mock()
        health_status.resource_usage = {"cpu": 96, "memory": 50}
        
        await fault_detector._detect_faults_from_health(health_status)
        
        callback.assert_called_once()
        fault_event = callback.call_args[0][0]
        assert fault_event.fault_type == FaultType.RESOURCE_EXHAUSTION
        assert fault_event.severity == FaultSeverity.HIGH
    
    async def test_create_fault_event(self, fault_detector):
        """测试创建故障事件"""
        callback = AsyncMock()
        fault_detector.register_fault_callback(callback)
        
        await fault_detector._create_fault_event(
            FaultType.AGENT_ERROR,
            FaultSeverity.MEDIUM,
            ["agent-1"],
            "Test fault",
            {"test": "data"}
        )
        
        # 验证故障事件被创建
        assert len(fault_detector.fault_events) == 1
        fault_event = fault_detector.fault_events[0]
        assert fault_event.fault_type == FaultType.AGENT_ERROR
        assert fault_event.severity == FaultSeverity.MEDIUM
        assert fault_event.affected_components == ["agent-1"]
        assert fault_event.description == "Test fault"
        assert fault_event.context["test"] == "data"
        assert not fault_event.resolved
        
        # 验证回调被调用
        callback.assert_called_once_with(fault_event)
    
    async def test_get_fault_events(self, fault_detector):
        """测试获取故障事件"""
        # 创建测试故障事件
        await fault_detector._create_fault_event(
            FaultType.AGENT_ERROR, FaultSeverity.HIGH, ["agent-1"], "Error 1", {}
        )
        await fault_detector._create_fault_event(
            FaultType.PERFORMANCE_DEGRADATION, FaultSeverity.LOW, ["agent-2"], "Error 2", {}
        )
        
        # 标记第一个故障为已解决
        fault_detector.fault_events[0].resolved = True
        
        # 测试无过滤
        events = fault_detector.get_fault_events()
        assert len(events) == 2
        
        # 测试按故障类型过滤
        events = fault_detector.get_fault_events(fault_type=FaultType.AGENT_ERROR)
        assert len(events) == 1
        assert events[0].fault_type == FaultType.AGENT_ERROR
        
        # 测试按严重程度过滤
        events = fault_detector.get_fault_events(severity=FaultSeverity.HIGH)
        assert len(events) == 1
        assert events[0].severity == FaultSeverity.HIGH
        
        # 测试按解决状态过滤
        events = fault_detector.get_fault_events(resolved=False)
        assert len(events) == 1
        assert not events[0].resolved
        
        # 测试限制数量
        events = fault_detector.get_fault_events(limit=1)
        assert len(events) == 1
    
    async def test_get_system_health_summary(self, fault_detector):
        """测试获取系统健康状态摘要"""
        # 添加一些健康状态
        fault_detector.health_status["agent-1"] = HealthStatus(
            component_id="agent-1",
            status="healthy",
            last_check=datetime.now(),
            response_time=0.5,
            error_rate=0.01,
            resource_usage={}
        )
        fault_detector.health_status["agent-2"] = HealthStatus(
            component_id="agent-2",
            status="degraded",
            last_check=datetime.now(),
            response_time=1.5,
            error_rate=0.05,
            resource_usage={}
        )
        
        # 添加一些故障事件
        await fault_detector._create_fault_event(
            FaultType.AGENT_ERROR, FaultSeverity.HIGH, ["agent-3"], "Test error", {}
        )
        
        summary = fault_detector.get_system_health_summary()
        
        assert summary["total_components"] == 2
        assert summary["status_counts"]["healthy"] == 1
        assert summary["status_counts"]["degraded"] == 1
        assert summary["avg_response_time"] == 1.0  # (0.5 + 1.5) / 2
        assert abs(summary["avg_error_rate"] - 0.03) < 1e-10   # (0.01 + 0.05) / 2
        assert summary["health_ratio"] == 0.5      # 1 healthy / 2 total
        assert summary["active_faults"] == 1
        assert "last_update" in summary
    
    async def test_check_system_performance(self, fault_detector):
        """测试检查系统整体性能"""
        callback = AsyncMock()
        fault_detector.register_fault_callback(callback)
        
        # 添加大量不健康组件
        for i in range(10):
            fault_detector.health_status[f"agent-{i}"] = HealthStatus(
                component_id=f"agent-{i}",
                status="unhealthy" if i < 8 else "healthy",  # 80%不健康
                last_check=datetime.now(),
                response_time=2.0,
                error_rate=0.2,
                resource_usage={}
            )
        
        await fault_detector._check_system_performance()
        
        # 应该触发系统级故障
        callback.assert_called_once()
        fault_event = callback.call_args[0][0]
        assert fault_event.fault_type == FaultType.NODE_FAILURE
        assert fault_event.severity == FaultSeverity.CRITICAL
        assert "system" in fault_event.affected_components
    
    async def test_analyze_network_partitions(self, fault_detector):
        """测试分析网络分区"""
        # 测试完全连通网络
        connectivity_matrix = {
            "agent-1": {"agent-2": True, "agent-3": True},
            "agent-2": {"agent-1": True, "agent-3": True},
            "agent-3": {"agent-1": True, "agent-2": True}
        }
        
        partitions = fault_detector._analyze_network_partitions(connectivity_matrix)
        assert len(partitions) == 1
        assert set(partitions[0]) == {"agent-1", "agent-2", "agent-3"}
        
        # 测试分区网络
        connectivity_matrix = {
            "agent-1": {"agent-2": True, "agent-3": False},
            "agent-2": {"agent-1": True, "agent-3": False},
            "agent-3": {"agent-1": False, "agent-2": False}
        }
        
        partitions = fault_detector._analyze_network_partitions(connectivity_matrix)
        assert len(partitions) == 2
        
        # 验证分区内容
        partition_sets = [set(p) for p in partitions]
        assert {"agent-1", "agent-2"} in partition_sets
        assert {"agent-3"} in partition_sets
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, fault_detector):
        """测试启动和停止功能"""
        assert not fault_detector.running
        
        # 启动
        await fault_detector.start()
        assert fault_detector.running
        
        # 等待一小段时间让循环开始
        await asyncio.sleep(0.1)
        
        # 停止
        await fault_detector.stop()
        assert not fault_detector.running
    
    async def test_fault_event_limit(self, fault_detector):
        """测试故障事件数量限制"""
        # 创建大量故障事件
        for i in range(12000):  # 超过10000的限制
            await fault_detector._create_fault_event(
                FaultType.AGENT_ERROR,
                FaultSeverity.LOW,
                [f"agent-{i}"],
                f"Error {i}",
                {}
            )
        
        # 验证事件数量被限制
        assert len(fault_detector.fault_events) == 5000  # 应该被截断到5000