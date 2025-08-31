import pytest
import asyncio
import tempfile
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....ai.fault_tolerance import (
    FaultToleranceSystem,
    FaultDetector,
    RecoveryManager,
    BackupManager,
    ConsistencyManager,
    FaultType,
    FaultSeverity,
    FaultEvent,
    BackupType
)

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_cluster_manager():
    cluster_manager = Mock()
    cluster_manager.get_agent_info = AsyncMock()
    cluster_manager.get_cluster_topology = AsyncMock()
    cluster_manager.update_agent_status = AsyncMock()
    cluster_manager.apply_degradation = AsyncMock()
    return cluster_manager

@pytest.fixture
def mock_task_coordinator():
    task_coordinator = Mock()
    task_coordinator.reassign_task = AsyncMock()
    task_coordinator.get_agent_tasks = AsyncMock()
    return task_coordinator

@pytest.fixture
def mock_lifecycle_manager():
    lifecycle_manager = Mock()
    lifecycle_manager.stop_agent = AsyncMock()
    lifecycle_manager.start_agent = AsyncMock()
    return lifecycle_manager

@pytest.fixture
def mock_metrics_collector():
    metrics_collector = Mock()
    metrics_collector.get_recent_metrics = AsyncMock()
    return metrics_collector

@pytest.fixture
def fault_tolerance_config(temp_dir):
    return {
        "fault_detection": {
            "health_check_interval": 0.5,  # 0.5秒用于快速测试
            "response_timeout": 1.0,
            "error_rate_threshold": 0.1,
            "performance_threshold": 2.0
        },
        "recovery": {
            "max_retry_attempts": 3,
            "recovery_timeout": 30
        },
        "backup": {
            "backup_interval": 1,
            "retention_days": 1,
            "backup_location": temp_dir,
            "auto_backup_components": ["agent-1"]
        },
        "consistency": {
            "consistency_check_interval": 1,
            "critical_data_keys": ["cluster_state"],
            "auto_repair": True
        }
    }

@pytest.fixture
def fault_tolerance_system(
    mock_cluster_manager,
    mock_task_coordinator, 
    mock_lifecycle_manager,
    mock_metrics_collector,
    fault_tolerance_config
):
    return FaultToleranceSystem(
        cluster_manager=mock_cluster_manager,
        task_coordinator=mock_task_coordinator,
        lifecycle_manager=mock_lifecycle_manager,
        metrics_collector=mock_metrics_collector,
        config=fault_tolerance_config
    )

@pytest.mark.asyncio
class TestFaultToleranceIntegration:
    
    async def test_fault_tolerance_system_initialization(self, fault_tolerance_system):
        """测试容错系统初始化"""
        assert fault_tolerance_system.started is False
        assert isinstance(fault_tolerance_system.fault_detector, FaultDetector)
        assert isinstance(fault_tolerance_system.recovery_manager, RecoveryManager)
        assert isinstance(fault_tolerance_system.backup_manager, BackupManager)
        assert isinstance(fault_tolerance_system.consistency_manager, ConsistencyManager)
        
        # 验证故障检测和恢复管理器的连接
        assert fault_tolerance_system.recovery_manager.handle_fault_event in fault_tolerance_system.fault_detector.fault_callbacks
    
    async def test_start_and_stop_system(self, fault_tolerance_system):
        """测试启动和停止系统"""
        # 启动系统
        await fault_tolerance_system.start()
        assert fault_tolerance_system.started is True
        
        # 验证各组件都已启动
        assert fault_tolerance_system.fault_detector.running is True
        assert fault_tolerance_system.recovery_manager.running is True
        assert fault_tolerance_system.backup_manager.running is True
        assert fault_tolerance_system.consistency_manager.running is True
        
        # 停止系统
        await fault_tolerance_system.stop()
        assert fault_tolerance_system.started is False
        
        # 验证各组件都已停止
        assert fault_tolerance_system.fault_detector.running is False
        assert fault_tolerance_system.recovery_manager.running is False
        assert fault_tolerance_system.backup_manager.running is False
        assert fault_tolerance_system.consistency_manager.running is False
    
    async def test_fault_detection_to_recovery_flow(
        self, 
        fault_tolerance_system, 
        mock_cluster_manager,
        mock_lifecycle_manager
    ):
        """测试故障检测到恢复的完整流程"""
        # 启动系统
        await fault_tolerance_system.start()
        
        # 模拟组件信息
        component_info = Mock()
        component_info.endpoint = "http://agent-1:8000"
        mock_cluster_manager.get_agent_info.return_value = component_info
        
        # 模拟HTTP超时（触发故障）
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = asyncio.TimeoutError()
            
            # 模拟重启成功
            mock_lifecycle_manager.start_agent.return_value = True
            
            # 触发健康检查
            health_status = await fault_tolerance_system.fault_detector.check_component_health("agent-1")
            
            # 验证故障被检测
            assert health_status.status == "unhealthy"
            
            # 等待故障被处理
            await asyncio.sleep(0.2)
            
            # 验证恢复动作被执行
            assert len(fault_tolerance_system.recovery_manager.recovery_history) > 0
            
            # 验证重启被调用
            mock_lifecycle_manager.stop_agent.assert_called()
            mock_lifecycle_manager.start_agent.assert_called()
        
        await fault_tolerance_system.stop()
    
    async def test_manual_backup_and_restore(self, fault_tolerance_system):
        """测试手动备份和恢复"""
        await fault_tolerance_system.start()
        
        # 模拟组件数据收集
        with patch.object(fault_tolerance_system.backup_manager, '_collect_component_data') as mock_collect:
            mock_collect.return_value = {
                "component_id": "agent-1",
                "state": {"status": "active"},
                "timestamp": datetime.now().isoformat()
            }
            
            # 触发手动备份
            backup_results = await fault_tolerance_system.trigger_manual_backup(["agent-1"])
            
            assert backup_results["agent-1"] is True
            assert len(fault_tolerance_system.backup_manager.backup_records) == 1
            
            backup_record = fault_tolerance_system.backup_manager.backup_records[0]
            
            # 测试恢复备份
            with patch.object(fault_tolerance_system.backup_manager, '_restore_component_data', return_value=True):
                success = await fault_tolerance_system.restore_backup(backup_record.backup_id)
                assert success is True
        
        await fault_tolerance_system.stop()
    
    async def test_consistency_check_and_repair(self, fault_tolerance_system, mock_cluster_manager):
        """测试一致性检查和修复"""
        await fault_tolerance_system.start()
        
        # 模拟集群拓扑
        topology = Mock()
        topology.agents = {"agent-1": Mock(), "agent-2": Mock()}
        mock_cluster_manager.get_cluster_topology.return_value = topology
        
        # 模拟不一致的数据
        inconsistent_data = {
            "agent-1": "value_a",
            "agent-2": "value_b"  # 不同的值
        }
        
        with patch.object(fault_tolerance_system.consistency_manager, '_get_component_data') as mock_get_data:
            mock_get_data.side_effect = lambda component_id, data_key: inconsistent_data[component_id]
            
            # 触发一致性检查
            result = await fault_tolerance_system.trigger_manual_consistency_check(["cluster_state"])
            
            assert result.consistent is False
            assert len(result.inconsistencies) > 0
            assert len(result.repair_actions) > 0
            
            # 测试修复
            with patch.object(fault_tolerance_system.consistency_manager, '_execute_repair_action', return_value=True):
                repair_success = await fault_tolerance_system.repair_consistency_issues(result.check_id)
                assert repair_success is True
        
        await fault_tolerance_system.stop()
    
    async def test_fault_injection_and_recovery(self, fault_tolerance_system, mock_lifecycle_manager):
        """测试故障注入和恢复"""
        await fault_tolerance_system.start()
        
        # 模拟重启成功
        mock_lifecycle_manager.start_agent.return_value = True
        
        # 注入故障
        fault_id = await fault_tolerance_system.simulate_fault_injection(
            "agent-1", 
            FaultType.AGENT_ERROR,
            duration_seconds=1
        )
        
        assert fault_id is not None
        
        # 等待故障处理
        await asyncio.sleep(0.2)
        
        # 验证恢复历史
        assert len(fault_tolerance_system.recovery_manager.recovery_history) > 0
        
        # 等待故障自动解决
        await asyncio.sleep(1.1)
        
        # 验证故障被标记为已解决
        fault_events = fault_tolerance_system.fault_detector.get_fault_events()
        injected_fault = next((f for f in fault_events if f.fault_id == fault_id), None)
        assert injected_fault is not None
        assert injected_fault.resolved is True
        
        await fault_tolerance_system.stop()
    
    async def test_system_status_and_metrics(self, fault_tolerance_system):
        """测试系统状态和指标获取"""
        await fault_tolerance_system.start()
        
        # 添加一些测试数据
        fault_tolerance_system.fault_detector.health_status["agent-1"] = Mock()
        fault_tolerance_system.fault_detector.health_status["agent-1"].status = "healthy"
        fault_tolerance_system.fault_detector.health_status["agent-1"].response_time = 0.5
        fault_tolerance_system.fault_detector.health_status["agent-1"].error_rate = 0.01
        
        # 获取系统状态
        status = await fault_tolerance_system.get_system_status()
        
        assert status["system_started"] is True
        assert "health_summary" in status
        assert "recovery_statistics" in status
        assert "backup_statistics" in status
        assert "consistency_statistics" in status
        assert "active_faults" in status
        assert "last_updated" in status
        
        # 获取系统指标
        metrics = await fault_tolerance_system.get_system_metrics()
        
        assert "fault_detection_metrics" in metrics
        assert "recovery_metrics" in metrics
        assert "backup_metrics" in metrics
        assert "consistency_metrics" in metrics
        assert "system_availability" in metrics
        assert "last_updated" in metrics
        
        await fault_tolerance_system.stop()
    
    async def test_component_health_monitoring(self, fault_tolerance_system, mock_cluster_manager, mock_metrics_collector):
        """测试组件健康监控"""
        await fault_tolerance_system.start()
        
        # 模拟组件信息和指标
        component_info = Mock()
        component_info.endpoint = "http://agent-1:8000"
        mock_cluster_manager.get_agent_info.return_value = component_info
        
        metric = Mock()
        metric.error_rate = 0.02
        metric.cpu_usage = 30.0
        metric.memory_usage = 40.0
        metric.disk_usage = 20.0
        mock_metrics_collector.get_recent_metrics.return_value = [metric]
        
        # 模拟HTTP健康检查
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # 获取组件健康状态
            health = await fault_tolerance_system.get_component_health("agent-1")
            
            assert health["component_id"] == "agent-1"
            assert health["status"] == "healthy"
            assert health["error_rate"] == 0.02
            assert health["resource_usage"]["cpu"] == 30.0
        
        await fault_tolerance_system.stop()
    
    async def test_backup_validation(self, fault_tolerance_system):
        """测试备份验证"""
        await fault_tolerance_system.start()
        
        # 创建测试备份记录（模拟）
        with patch.object(fault_tolerance_system.backup_manager, '_collect_component_data') as mock_collect:
            mock_collect.return_value = {"test": "data"}
            
            backup_results = await fault_tolerance_system.trigger_manual_backup(["agent-1"])
            assert backup_results["agent-1"] is True
        
        # 验证所有备份
        validation_results = await fault_tolerance_system.validate_all_backups()
        
        assert len(validation_results) == 1
        backup_ids = list(validation_results.keys())
        assert validation_results[backup_ids[0]] is True  # 应该验证成功
        
        await fault_tolerance_system.stop()
    
    async def test_detailed_system_report(self, fault_tolerance_system):
        """测试详细系统报告"""
        await fault_tolerance_system.start()
        
        # 添加一些测试数据
        await fault_tolerance_system.fault_detector._create_fault_event(
            FaultType.PERFORMANCE_DEGRADATION,
            FaultSeverity.LOW,
            ["agent-1"],
            "Test degradation",
            {}
        )
        
        # 生成系统报告
        report = await fault_tolerance_system.get_detailed_system_report()
        
        assert "report_generated_at" in report
        assert "system_status" in report
        assert "system_metrics" in report
        assert "recent_faults" in report
        assert "recent_backups" in report
        assert "recommendations" in report
        
        # 验证建议生成
        assert len(report["recommendations"]) > 0
        
        await fault_tolerance_system.stop()
    
    async def test_system_availability_calculation(self, fault_tolerance_system):
        """测试系统可用性计算"""
        await fault_tolerance_system.start()
        
        # 添加健康组件
        fault_tolerance_system.fault_detector.health_status["agent-1"] = Mock()
        fault_tolerance_system.fault_detector.health_status["agent-1"].status = "healthy"
        fault_tolerance_system.fault_detector.health_status["agent-1"].response_time = 0.5
        fault_tolerance_system.fault_detector.health_status["agent-1"].error_rate = 0.01
        
        fault_tolerance_system.fault_detector.health_status["agent-2"] = Mock()
        fault_tolerance_system.fault_detector.health_status["agent-2"].status = "healthy"
        fault_tolerance_system.fault_detector.health_status["agent-2"].response_time = 0.8
        fault_tolerance_system.fault_detector.health_status["agent-2"].error_rate = 0.02
        
        availability = fault_tolerance_system._calculate_system_availability()
        
        # 两个健康组件，应该有100%可用性
        assert availability == 1.0
        
        # 添加活跃故障
        await fault_tolerance_system.fault_detector._create_fault_event(
            FaultType.AGENT_ERROR,
            FaultSeverity.HIGH,
            ["agent-3"],
            "Active fault",
            {}
        )
        
        # 重新计算可用性（应该降低）
        availability = fault_tolerance_system._calculate_system_availability()
        assert availability < 1.0
        
        await fault_tolerance_system.stop()
    
    async def test_force_consistency_repair(self, fault_tolerance_system, mock_cluster_manager):
        """测试强制一致性修复"""
        await fault_tolerance_system.start()
        
        # 模拟集群拓扑
        topology = Mock()
        topology.agents = {"agent-1": Mock(), "agent-2": Mock()}
        mock_cluster_manager.get_cluster_topology.return_value = topology
        
        with patch.object(fault_tolerance_system.consistency_manager, '_get_component_data') as mock_get_data, \
             patch.object(fault_tolerance_system.consistency_manager, '_force_set_component_data', return_value=True) as mock_set_data:
            
            # 模拟权威组件的数据
            mock_get_data.return_value = {"authoritative": "value"}
            
            success = await fault_tolerance_system.force_consistency_repair("test_key", "agent-1")
            
            assert success is True
            mock_get_data.assert_called_with("agent-1", "test_key")
            mock_set_data.assert_called()  # 应该为其他组件调用
        
        await fault_tolerance_system.stop()
    
    async def test_concurrent_operations(self, fault_tolerance_system, mock_lifecycle_manager):
        """测试并发操作"""
        await fault_tolerance_system.start()
        
        # 模拟成功重启
        mock_lifecycle_manager.start_agent.return_value = True
        
        # 并发执行多个操作
        tasks = []
        
        # 同时注入多个故障
        for i in range(3):
            task = fault_tolerance_system.simulate_fault_injection(
                f"agent-{i}",
                FaultType.AGENT_ERROR,
                duration_seconds=1
            )
            tasks.append(task)
        
        # 同时触发备份
        with patch.object(fault_tolerance_system.backup_manager, '_collect_component_data', return_value={"test": "data"}):
            backup_task = fault_tolerance_system.trigger_manual_backup(["agent-1", "agent-2"])
            tasks.append(backup_task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        # 验证故障注入成功
        fault_ids = results[:3]
        for fault_id in fault_ids:
            assert fault_id is not None
        
        # 验证备份成功
        backup_results = results[3]
        assert backup_results["agent-1"] is True
        assert backup_results["agent-2"] is True
        
        # 等待所有操作处理完成
        await asyncio.sleep(0.5)
        
        # 验证恢复历史
        assert len(fault_tolerance_system.recovery_manager.recovery_history) >= 3
        
        await fault_tolerance_system.stop()