from src.core.utils.timezone_utils import utc_now
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from ....ai.fault_tolerance.recovery_manager import (
    RecoveryManager,
    RecoveryStrategy
)
from ....ai.fault_tolerance.fault_detector import (
    FaultEvent,
    FaultType,
    FaultSeverity

)

@pytest.fixture
def mock_cluster_manager():
    cluster_manager = Mock()
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
def recovery_config():
    return {
        "max_retry_attempts": 3,
        "recovery_timeout": 120
    }

@pytest.fixture
def recovery_manager(mock_cluster_manager, mock_task_coordinator, mock_lifecycle_manager, recovery_config):
    return RecoveryManager(
        cluster_manager=mock_cluster_manager,
        task_coordinator=mock_task_coordinator,
        lifecycle_manager=mock_lifecycle_manager,
        config=recovery_config
    )

@pytest.fixture
def sample_fault_event():
    return FaultEvent(
        fault_id="test_fault_123",
        fault_type=FaultType.AGENT_UNRESPONSIVE,
        severity=FaultSeverity.HIGH,
        affected_components=["agent-1"],
        detected_at=utc_now(),
        description="Test agent unresponsive",
        context={"test": "data"}
    )

@pytest.mark.asyncio
class TestRecoveryManager:
    
    async def test_recovery_manager_initialization(self, recovery_manager):
        """测试恢复管理器初始化"""
        assert recovery_manager.running is False
        assert len(recovery_manager.recovery_history) == 0
        assert recovery_manager.recovery_queue.empty()
        
        # 验证策略配置
        strategies = recovery_manager.recovery_strategies
        assert FaultType.AGENT_UNRESPONSIVE in strategies
        assert RecoveryStrategy.GRACEFUL_RESTART in strategies[FaultType.AGENT_UNRESPONSIVE]
    
    async def test_start_and_stop(self, recovery_manager):
        """测试启动和停止功能"""
        assert not recovery_manager.running
        
        # 启动
        await recovery_manager.start()
        assert recovery_manager.running
        
        # 等待一小段时间让循环开始
        await asyncio.sleep(0.1)
        
        # 停止
        await recovery_manager.stop()
        assert not recovery_manager.running
    
    async def test_handle_fault_event(self, recovery_manager, sample_fault_event):
        """测试处理故障事件"""
        await recovery_manager.handle_fault_event(sample_fault_event)
        
        # 验证故障事件被放入队列
        assert not recovery_manager.recovery_queue.empty()
        queued_event = await recovery_manager.recovery_queue.get()
        assert queued_event.fault_id == sample_fault_event.fault_id
    
    async def test_immediate_restart_recovery(self, recovery_manager, mock_lifecycle_manager):
        """测试立即重启恢复策略"""
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.AGENT_ERROR,
            severity=FaultSeverity.MEDIUM,
            affected_components=["agent-1", "agent-2"],
            detected_at=utc_now(),
            description="Test error",
            context={}
        )
        
        # 模拟重启成功
        mock_lifecycle_manager.start_agent.return_value = True
        
        success = await recovery_manager._immediate_restart_recovery(fault_event)
        
        assert success is True
        
        # 验证调用次数
        assert mock_lifecycle_manager.stop_agent.call_count == 2
        assert mock_lifecycle_manager.start_agent.call_count == 2
        
        # 验证调用参数
        mock_lifecycle_manager.stop_agent.assert_any_call("agent-1", graceful=False)
        mock_lifecycle_manager.stop_agent.assert_any_call("agent-2", graceful=False)
    
    async def test_graceful_restart_recovery(self, recovery_manager, mock_lifecycle_manager):
        """测试优雅重启恢复策略"""
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.PERFORMANCE_DEGRADATION,
            severity=FaultSeverity.LOW,
            affected_components=["agent-1"],
            detected_at=utc_now(),
            description="Performance degraded",
            context={}
        )
        
        # 模拟重启成功
        mock_lifecycle_manager.start_agent.return_value = True
        
        success = await recovery_manager._graceful_restart_recovery(fault_event)
        
        assert success is True
        
        # 验证优雅停止被调用
        mock_lifecycle_manager.stop_agent.assert_called_once_with("agent-1", graceful=True)
        mock_lifecycle_manager.start_agent.assert_called_once_with("agent-1")
    
    async def test_task_migration_recovery(self, recovery_manager, mock_task_coordinator, mock_cluster_manager):
        """测试任务迁移恢复策略"""
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.RESOURCE_EXHAUSTION,
            severity=FaultSeverity.HIGH,
            affected_components=["agent-1"],
            detected_at=utc_now(),
            description="Resource exhausted",
            context={}
        )
        
        # 模拟活跃任务
        task1 = Mock()
        task1.id = "task-1"
        task1.status = "running"
        task2 = Mock()
        task2.id = "task-2"
        task2.status = "pending"
        
        mock_task_coordinator.get_agent_tasks.return_value = [task1, task2]
        mock_task_coordinator.reassign_task.return_value = True
        
        success = await recovery_manager._task_migration_recovery(fault_event)
        
        assert success is True
        
        # 验证任务迁移被调用
        assert mock_task_coordinator.reassign_task.call_count == 2
        mock_task_coordinator.reassign_task.assert_any_call("task-1")
        mock_task_coordinator.reassign_task.assert_any_call("task-2")
        
        # 验证智能体状态更新
        mock_cluster_manager.update_agent_status.assert_called_once_with("agent-1", "maintenance")
    
    async def test_service_degradation_recovery(self, recovery_manager, mock_cluster_manager):
        """测试服务降级恢复策略"""
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.NETWORK_PARTITION,
            severity=FaultSeverity.CRITICAL,
            affected_components=["network"],
            detected_at=utc_now(),
            description="Network partition",
            context={}
        )
        
        success = await recovery_manager._service_degradation_recovery(fault_event)
        
        assert success is True
        
        # 验证降级配置被应用（如果集群管理器支持）
        if hasattr(mock_cluster_manager, 'apply_degradation'):
            mock_cluster_manager.apply_degradation.assert_called_once()
    
    async def test_manual_intervention_recovery(self, recovery_manager):
        """测试手动干预恢复策略"""
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.DATA_CORRUPTION,
            severity=FaultSeverity.CRITICAL,
            affected_components=["agent-1"],
            detected_at=utc_now(),
            description="Data corruption detected",
            context={}
        )
        
        success = await recovery_manager._manual_intervention_recovery(fault_event)
        
        # 手动干预总是返回False，因为需要人工处理
        assert success is False
    
    async def test_apply_recovery_strategy(self, recovery_manager, mock_lifecycle_manager):
        """测试应用恢复策略"""
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.AGENT_ERROR,
            severity=FaultSeverity.MEDIUM,
            affected_components=["agent-1"],
            detected_at=utc_now(),
            description="Agent error",
            context={}
        )
        
        # 模拟重启成功
        mock_lifecycle_manager.start_agent.return_value = True
        
        # 测试立即重启策略
        success = await recovery_manager._apply_recovery_strategy(
            fault_event, 
            RecoveryStrategy.IMMEDIATE_RESTART
        )
        assert success is True
        
        # 测试未知策略
        success = await recovery_manager._apply_recovery_strategy(
            fault_event, 
            "unknown_strategy"
        )
        assert success is False
    
    async def test_execute_recovery_full_process(self, recovery_manager, mock_lifecycle_manager):
        """测试完整的恢复执行过程"""
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.AGENT_UNRESPONSIVE,
            severity=FaultSeverity.HIGH,
            affected_components=["agent-1"],
            detected_at=utc_now(),
            description="Agent unresponsive",
            context={}
        )
        
        # 模拟第一个策略失败，第二个成功
        call_count = 0
        def mock_start_agent(agent_id):
            nonlocal call_count
            call_count += 1
            return call_count > 1  # 第一次失败，第二次成功
        
        mock_lifecycle_manager.start_agent.side_effect = mock_start_agent
        
        await recovery_manager._execute_recovery(fault_event)
        
        # 验证恢复历史被记录
        assert len(recovery_manager.recovery_history) == 1
        
        recovery_record = recovery_manager.recovery_history[0]
        assert recovery_record["fault_id"] == "test_fault"
        assert recovery_record["fault_type"] == "agent_unresponsive"
        assert recovery_record["recovery_success"] is True
        assert len(recovery_record["recovery_actions"]) >= 1
        
        # 验证故障事件被标记为已解决
        assert fault_event.resolved is True
        assert fault_event.resolved_at is not None
        assert len(fault_event.recovery_actions) > 0
    
    async def test_execute_recovery_all_strategies_fail(self, recovery_manager, mock_lifecycle_manager):
        """测试所有恢复策略都失败的情况"""
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.AGENT_ERROR,
            severity=FaultSeverity.HIGH,
            affected_components=["agent-1"],
            detected_at=utc_now(),
            description="Agent error",
            context={}
        )
        
        # 模拟所有重启都失败
        mock_lifecycle_manager.start_agent.return_value = False
        
        await recovery_manager._execute_recovery(fault_event)
        
        # 验证恢复失败被记录
        recovery_record = recovery_manager.recovery_history[0]
        assert recovery_record["recovery_success"] is False
        
        # 验证故障事件未被标记为已解决
        assert fault_event.resolved is False
        assert fault_event.resolved_at is None
    
    async def test_get_recovery_statistics_empty(self, recovery_manager):
        """测试获取空的恢复统计信息"""
        stats = recovery_manager.get_recovery_statistics()
        
        assert stats["total_recoveries"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_recovery_time"] == 0.0
        assert stats["strategy_success_rates"] == {}
    
    async def test_get_recovery_statistics_with_data(self, recovery_manager):
        """测试获取有数据的恢复统计信息"""
        # 添加一些恢复历史记录
        recovery_manager.recovery_history = [
            {
                "fault_id": "fault-1",
                "fault_type": "agent_error",
                "recovery_success": True,
                "recovery_time": 30.0,
                "recovery_actions": [
                    {"strategy": "immediate_restart", "success": True}
                ]
            },
            {
                "fault_id": "fault-2",
                "fault_type": "performance_degradation",
                "recovery_success": False,
                "recovery_time": 60.0,
                "recovery_actions": [
                    {"strategy": "graceful_restart", "success": False},
                    {"strategy": "task_migration", "success": False}
                ]
            }
        ]
        
        stats = recovery_manager.get_recovery_statistics()
        
        assert stats["total_recoveries"] == 2
        assert stats["success_rate"] == 0.5  # 1 成功 / 2 总数
        assert stats["avg_recovery_time"] == 45.0  # (30 + 60) / 2
        
        # 验证策略成功率
        strategy_rates = stats["strategy_success_rates"]
        assert strategy_rates["immediate_restart"] == 1.0  # 1/1
        assert strategy_rates["graceful_restart"] == 0.0   # 0/1
        assert strategy_rates["task_migration"] == 0.0     # 0/1
        
        # 验证最近恢复记录
        assert len(stats["recent_recoveries"]) == 2
    
    async def test_recovery_processing_loop(self, recovery_manager, mock_lifecycle_manager):
        """测试恢复处理循环"""
        await recovery_manager.start()
        
        # 模拟成功重启
        mock_lifecycle_manager.start_agent.return_value = True
        
        fault_event = FaultEvent(
            fault_id="test_fault",
            fault_type=FaultType.AGENT_ERROR,
            severity=FaultSeverity.MEDIUM,
            affected_components=["agent-1"],
            detected_at=utc_now(),
            description="Test error",
            context={}
        )
        
        # 添加故障事件到队列
        await recovery_manager.handle_fault_event(fault_event)
        
        # 等待处理完成
        await asyncio.sleep(0.1)
        
        # 验证恢复历史被创建
        assert len(recovery_manager.recovery_history) == 1
        
        await recovery_manager.stop()
    
    async def test_recovery_history_size_limit(self, recovery_manager):
        """测试恢复历史大小限制"""
        # 添加大量恢复记录
        for i in range(1200):  # 超过1000的限制
            recovery_manager.recovery_history.append({
                "fault_id": f"fault-{i}",
                "recovery_success": True,
                "recovery_time": 10.0,
                "recovery_actions": []
            })
        
        # 创建一个新的故障来触发历史清理
        fault_event = FaultEvent(
            fault_id="trigger_cleanup",
            fault_type=FaultType.AGENT_ERROR,
            severity=FaultSeverity.LOW,
            affected_components=["agent-1"],
            detected_at=utc_now(),
            description="Trigger cleanup",
            context={}
        )
        
        await recovery_manager._execute_recovery(fault_event)
        
        # 验证历史记录被限制在500条
        assert len(recovery_manager.recovery_history) == 500
    
    async def test_get_agent_active_tasks(self, recovery_manager, mock_task_coordinator):
        """测试获取智能体活跃任务"""
        # 模拟任务列表
        running_task = Mock()
        running_task.id = "task-1"
        running_task.status = "running"
        
        pending_task = Mock()
        pending_task.id = "task-2"
        pending_task.status = "pending"
        
        completed_task = Mock()
        completed_task.id = "task-3"
        completed_task.status = "completed"
        
        mock_task_coordinator.get_agent_tasks.return_value = [running_task, pending_task, completed_task]
        
        active_tasks = await recovery_manager._get_agent_active_tasks("agent-1")
        
        # 应该只返回running和pending的任务
        assert len(active_tasks) == 2
        assert "task-1" in active_tasks
        assert "task-2" in active_tasks
        assert "task-3" not in active_tasks
    
    async def test_system_component_handling(self, recovery_manager, mock_lifecycle_manager):
        """测试系统级组件的处理"""
        fault_event = FaultEvent(
            fault_id="system_fault",
            fault_type=FaultType.NODE_FAILURE,
            severity=FaultSeverity.CRITICAL,
            affected_components=["system", "agent-1"],
            detected_at=utc_now(),
            description="System failure",
            context={}
        )
        
        mock_lifecycle_manager.start_agent.return_value = True
        
        success = await recovery_manager._immediate_restart_recovery(fault_event)
        
        # 验证只对真实智能体执行重启，跳过系统级组件
        assert mock_lifecycle_manager.stop_agent.call_count == 1
        assert mock_lifecycle_manager.start_agent.call_count == 1
        mock_lifecycle_manager.stop_agent.assert_called_with("agent-1", graceful=False)
