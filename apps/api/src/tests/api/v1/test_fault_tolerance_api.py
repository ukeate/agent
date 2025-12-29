from src.core.utils.timezone_utils import utc_now
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from ...main import app
from ...ai.fault_tolerance import FaultToleranceSystem

@pytest.fixture
def mock_fault_tolerance_system():
    system = Mock(spec=FaultToleranceSystem)
    
    # Mock async methods
    system.get_system_status = AsyncMock()
    system.get_component_health = AsyncMock()
    system.get_system_metrics = AsyncMock()
    system.get_detailed_system_report = AsyncMock()
    system.get_fault_events = AsyncMock()
    system.trigger_manual_backup = AsyncMock()
    system.restore_backup = AsyncMock()
    system.trigger_manual_consistency_check = AsyncMock()
    system.repair_consistency_issues = AsyncMock()
    system.simulate_fault_injection = AsyncMock()
    system.validate_all_backups = AsyncMock()
    system.force_consistency_repair = AsyncMock()
    system.start = AsyncMock()
    system.stop = AsyncMock()
    
    # Mock sync attributes
    system.fault_detector = Mock()
    system.recovery_manager = Mock()
    system.backup_manager = Mock()
    system.consistency_manager = Mock()
    
    return system

@pytest.fixture
def client():
    return TestClient(app)

class TestFaultToleranceAPI:
    
    def test_get_fault_tolerance_status(self, client, mock_fault_tolerance_system):
        """测试获取容错系统状态"""
        mock_status = {
            "system_started": True,
            "health_summary": {"total_components": 3, "health_ratio": 0.9},
            "recovery_statistics": {"success_rate": 0.95},
            "backup_statistics": {"total_backups": 10},
            "consistency_statistics": {"consistency_rate": 0.98},
            "active_faults": [],
            "last_updated": utc_now().isoformat()
        }
        
        mock_fault_tolerance_system.get_system_status.return_value = mock_status
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.get("/fault-tolerance/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["system_started"] is True
            assert data["health_summary"]["total_components"] == 3
    
    def test_get_health_summary(self, client, mock_fault_tolerance_system):
        """测试获取健康状态摘要"""
        mock_summary = {
            "total_components": 5,
            "status_counts": {"healthy": 4, "degraded": 1},
            "health_ratio": 0.8,
            "active_faults": 2
        }
        
        mock_fault_tolerance_system.fault_detector.get_system_health_summary.return_value = mock_summary
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.get("/fault-tolerance/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_components"] == 5
            assert data["health_ratio"] == 0.8
    
    def test_get_component_health(self, client, mock_fault_tolerance_system):
        """测试获取组件健康状态"""
        mock_health = {
            "component_id": "agent-1",
            "status": "healthy",
            "last_check": utc_now().isoformat(),
            "response_time": 0.5,
            "error_rate": 0.01,
            "resource_usage": {"cpu": 45.0, "memory": 60.0}
        }
        
        mock_fault_tolerance_system.get_component_health.return_value = mock_health
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.get("/fault-tolerance/health/agent-1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["component_id"] == "agent-1"
            assert data["status"] == "healthy"
    
    def test_get_fault_events(self, client, mock_fault_tolerance_system):
        """测试获取故障事件"""
        mock_events = [
            {
                "fault_id": "fault-1",
                "fault_type": "agent_error",
                "severity": "high",
                "affected_components": ["agent-1"],
                "detected_at": utc_now().isoformat(),
                "description": "Agent error",
                "resolved": False
            }
        ]
        
        mock_fault_tolerance_system.get_fault_events.return_value = mock_events
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            # 测试无过滤
            response = client.get("/fault-tolerance/faults")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["fault_id"] == "fault-1"
            
            # 测试带过滤
            response = client.get("/fault-tolerance/faults?fault_type=agent_error&limit=10")
            assert response.status_code == 200
    
    def test_get_fault_events_invalid_params(self, client, mock_fault_tolerance_system):
        """测试获取故障事件的无效参数"""
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            # 无效的故障类型
            response = client.get("/fault-tolerance/faults?fault_type=invalid_type")
            assert response.status_code == 400
            
            # 无效的严重程度
            response = client.get("/fault-tolerance/faults?severity=invalid_severity")
            assert response.status_code == 400
    
    def test_trigger_manual_backup(self, client, mock_fault_tolerance_system):
        """测试触发手动备份"""
        mock_results = {"agent-1": True, "agent-2": False}
        mock_fault_tolerance_system.trigger_manual_backup.return_value = mock_results
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/backup/manual", json={
                "component_ids": ["agent-1", "agent-2"],
                "backup_type": "full_backup"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["backup_results"]["agent-1"] is True
            assert data["backup_results"]["agent-2"] is False
            assert data["success_count"] == 1
            assert data["total_count"] == 2
    
    def test_trigger_manual_backup_invalid_type(self, client, mock_fault_tolerance_system):
        """测试触发手动备份的无效类型"""
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/backup/manual", json={
                "component_ids": ["agent-1"],
                "backup_type": "invalid_type"
            })
            
            assert response.status_code == 400
    
    def test_restore_backup(self, client, mock_fault_tolerance_system):
        """测试恢复备份"""
        mock_fault_tolerance_system.restore_backup.return_value = True
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/backup/restore", json={
                "backup_id": "backup-123",
                "target_component_id": "agent-2"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "restored"
            assert data["backup_id"] == "backup-123"
            assert data["target_component"] == "agent-2"
    
    def test_restore_backup_failure(self, client, mock_fault_tolerance_system):
        """测试恢复备份失败"""
        mock_fault_tolerance_system.restore_backup.return_value = False
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/backup/restore", json={
                "backup_id": "backup-123"
            })
            
            assert response.status_code == 500
    
    def test_validate_all_backups(self, client, mock_fault_tolerance_system):
        """测试验证所有备份"""
        mock_results = {
            "backup-1": True,
            "backup-2": False,
            "backup-3": True
        }
        mock_fault_tolerance_system.validate_all_backups.return_value = mock_results
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/backup/validate")
            
            assert response.status_code == 200
            data = response.json()
            assert data["valid_count"] == 2
            assert data["total_count"] == 3
            assert data["validation_rate"] == 2/3
    
    def test_trigger_consistency_check(self, client, mock_fault_tolerance_system):
        """测试触发一致性检查"""
        mock_result = Mock()
        mock_result.check_id = "check-123"
        mock_result.checked_at = utc_now()
        mock_result.components = ["agent-1", "agent-2"]
        mock_result.consistent = False
        mock_result.inconsistencies = [{"type": "value_mismatch", "data_key": "test"}]
        mock_result.repair_actions = ["repair_data:agent-1:test"]
        
        mock_fault_tolerance_system.trigger_manual_consistency_check.return_value = mock_result
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/consistency/check", json={
                "data_keys": ["cluster_state", "task_assignments"]
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["check_id"] == "check-123"
            assert data["consistent"] is False
            assert data["inconsistencies_count"] == 1
    
    def test_repair_consistency_issues(self, client, mock_fault_tolerance_system):
        """测试修复一致性问题"""
        mock_fault_tolerance_system.repair_consistency_issues.return_value = True
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/consistency/check-123/repair")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "repaired"
            assert data["check_id"] == "check-123"
    
    def test_force_consistency_repair(self, client, mock_fault_tolerance_system):
        """测试强制一致性修复"""
        mock_fault_tolerance_system.force_consistency_repair.return_value = True
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/consistency/force-repair", json={
                "data_key": "cluster_state",
                "authoritative_component_id": "agent-1"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "force_repaired"
            assert data["data_key"] == "cluster_state"
    
    def test_inject_fault_for_testing(self, client, mock_fault_tolerance_system):
        """测试注入故障进行测试"""
        mock_fault_tolerance_system.simulate_fault_injection.return_value = "fault-123"
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/testing/inject-fault", json={
                "component_id": "agent-1",
                "fault_type": "agent_error",
                "duration_seconds": 60
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "fault_injected"
            assert data["fault_id"] == "fault-123"
            assert data["component_id"] == "agent-1"
            assert data["fault_type"] == "agent_error"
    
    def test_inject_fault_invalid_params(self, client, mock_fault_tolerance_system):
        """测试注入故障的无效参数"""
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            # 无效的故障类型
            response = client.post("/fault-tolerance/testing/inject-fault", json={
                "component_id": "agent-1",
                "fault_type": "invalid_type",
                "duration_seconds": 60
            })
            assert response.status_code == 400
            
            # 无效的持续时间
            response = client.post("/fault-tolerance/testing/inject-fault", json={
                "component_id": "agent-1",
                "fault_type": "agent_error",
                "duration_seconds": 0
            })
            assert response.status_code == 400
            
            response = client.post("/fault-tolerance/testing/inject-fault", json={
                "component_id": "agent-1",
                "fault_type": "agent_error",
                "duration_seconds": 4000
            })
            assert response.status_code == 400
    
    def test_get_system_metrics(self, client, mock_fault_tolerance_system):
        """测试获取系统指标"""
        mock_metrics = {
            "fault_detection_metrics": {"total_components": 5, "healthy_components": 4},
            "recovery_metrics": {"success_rate": 0.95},
            "backup_metrics": {"total_backups": 10},
            "consistency_metrics": {"consistency_rate": 0.98},
            "system_availability": 0.96,
            "last_updated": utc_now().isoformat()
        }
        
        mock_fault_tolerance_system.get_system_metrics.return_value = mock_metrics
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.get("/fault-tolerance/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["system_availability"] == 0.96
            assert data["fault_detection_metrics"]["total_components"] == 5
    
    def test_get_detailed_system_report(self, client, mock_fault_tolerance_system):
        """测试获取详细系统报告"""
        mock_report = {
            "report_generated_at": utc_now().isoformat(),
            "system_status": {"system_started": True},
            "system_metrics": {"system_availability": 0.95},
            "recent_faults": [],
            "recent_backups": [],
            "recommendations": ["System is operating normally"]
        }
        
        mock_fault_tolerance_system.get_detailed_system_report.return_value = mock_report
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.get("/fault-tolerance/report")
            
            assert response.status_code == 200
            data = response.json()
            assert "report_generated_at" in data
            assert "recommendations" in data
            assert len(data["recommendations"]) > 0
    
    def test_start_fault_tolerance_system(self, client, mock_fault_tolerance_system):
        """测试启动容错系统"""
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/system/start")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            mock_fault_tolerance_system.start.assert_called_once()
    
    def test_stop_fault_tolerance_system(self, client, mock_fault_tolerance_system):
        """测试停止容错系统"""
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.post("/fault-tolerance/system/stop")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "stopped"
            mock_fault_tolerance_system.stop.assert_called_once()
    
    def test_get_enums(self, client):
        """测试获取枚举值"""
        # 测试故障类型枚举
        response = client.get("/fault-tolerance/enums/fault-types")
        assert response.status_code == 200
        data = response.json()
        assert "fault_types" in data
        assert len(data["fault_types"]) > 0
        
        # 测试严重程度枚举
        response = client.get("/fault-tolerance/enums/severities")
        assert response.status_code == 200
        data = response.json()
        assert "severities" in data
        
        # 测试备份类型枚举
        response = client.get("/fault-tolerance/enums/backup-types")
        assert response.status_code == 200
        data = response.json()
        assert "backup_types" in data
        
        # 测试恢复策略枚举
        response = client.get("/fault-tolerance/enums/recovery-strategies")
        assert response.status_code == 200
        data = response.json()
        assert "recovery_strategies" in data
    
    def test_error_handling(self, client, mock_fault_tolerance_system):
        """测试错误处理"""
        # 模拟系统未初始化
        with patch('....core.dependencies.get_fault_tolerance_system', side_effect=Exception("System not initialized")):
            response = client.get("/fault-tolerance/status")
            assert response.status_code == 500
        
        # 模拟系统方法异常
        mock_fault_tolerance_system.get_system_status.side_effect = Exception("Internal error")
        
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            response = client.get("/fault-tolerance/status")
            assert response.status_code == 500
    
    def test_request_validation(self, client, mock_fault_tolerance_system):
        """测试请求验证"""
        with patch('src.core.dependencies.get_fault_tolerance_system', return_value=mock_fault_tolerance_system):
            # 测试空的组件ID列表
            response = client.post("/fault-tolerance/backup/manual", json={
                "component_ids": []
            })
            assert response.status_code == 422  # Validation error
            
            # 测试空的数据键列表
            response = client.post("/fault-tolerance/consistency/check", json={
                "data_keys": []
            })
            assert response.status_code == 422
            
            # 测试缺少必需字段
            response = client.post("/fault-tolerance/backup/manual", json={})
            assert response.status_code == 422
