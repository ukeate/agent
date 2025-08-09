"""
Supervisor API测试
测试Supervisor相关的HTTP接口
"""
import pytest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.v1.supervisor import router
from models.schemas.supervisor import (
    TaskSubmissionRequest, TaskType, TaskPriority,
    SupervisorConfigUpdateRequest, RoutingStrategy
)
from ai.autogen.supervisor_agent import TaskAssignment

# 创建测试app
app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_supervisor_service():
    """模拟SupervisorService"""
    with patch('api.v1.supervisor.supervisor_service') as mock_service:
        yield mock_service


@pytest.fixture
def sample_task_assignment():
    """示例任务分配结果"""
    return TaskAssignment(
        task_id="task_12345",
        assigned_agent="code_expert",
        assignment_reason="最佳匹配智能体",
        confidence_level=0.85,
        estimated_completion_time=datetime.now(timezone.utc),
        alternative_agents=["architect", "doc_expert"],
        decision_metadata={
            "complexity": {"score": 0.6, "estimated_time": 300},
            "match_details": {"match_score": 0.85}
        }
    )


@pytest.fixture
def sample_supervisor_status():
    """示例Supervisor状态"""
    return {
        "supervisor_name": "main_supervisor",
        "status": "active",
        "available_agents": ["code_expert", "architect", "doc_expert"],
        "agent_loads": {
            "code_expert": 0.3,
            "architect": 0.1,
            "doc_expert": 0.0
        },
        "decision_history_count": 15,
        "task_queue_length": 3,
        "performance_metrics": {
            "average_confidence": 0.82,
            "total_decisions": 15
        },
        "current_config": {
            "routing_strategy": "hybrid",
            "load_threshold": 0.8
        }
    }


class TestTaskSubmission:
    """任务提交API测试"""
    
    def test_submit_task_success(self, mock_supervisor_service, sample_task_assignment):
        """测试成功提交任务"""
        mock_supervisor_service.submit_task.return_value = sample_task_assignment
        
        request_data = {
            "name": "实现用户登录",
            "description": "实现基于JWT的用户登录功能",
            "task_type": "code_generation",
            "priority": "high",
            "constraints": {"timeout": 30},
            "input_data": {"language": "Python"}
        }
        
        response = client.post(
            "/supervisor/tasks?supervisor_id=supervisor_123",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "任务提交成功"
        assert "data" in data
        assert data["data"]["task_id"] == "task_12345"
        assert data["data"]["assigned_agent"] == "code_expert"
        
        # 验证服务调用
        mock_supervisor_service.submit_task.assert_called_once()
        args = mock_supervisor_service.submit_task.call_args
        assert args[0] == "supervisor_123"
        assert args[1].name == "实现用户登录"
    
    def test_submit_task_invalid_supervisor(self, mock_supervisor_service):
        """测试无效Supervisor ID"""
        mock_supervisor_service.submit_task.side_effect = ValueError("Supervisor not found")
        
        request_data = {
            "name": "测试任务",
            "description": "测试描述",
            "task_type": "code_generation",
            "priority": "medium"
        }
        
        response = client.post(
            "/supervisor/tasks?supervisor_id=invalid_id",
            json=request_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
    
    def test_submit_task_missing_fields(self):
        """测试缺少必需字段"""
        request_data = {
            "name": "测试任务",
            # 缺少description和task_type
            "priority": "medium"
        }
        
        response = client.post(
            "/supervisor/tasks?supervisor_id=supervisor_123",
            json=request_data
        )
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_submit_task_invalid_task_type(self):
        """测试无效任务类型"""
        request_data = {
            "name": "测试任务",
            "description": "测试描述",
            "task_type": "invalid_type",
            "priority": "medium"
        }
        
        response = client.post(
            "/supervisor/tasks?supervisor_id=supervisor_123",
            json=request_data
        )
        
        assert response.status_code == 422


class TestSupervisorStatus:
    """Supervisor状态API测试"""
    
    def test_get_supervisor_status_success(self, mock_supervisor_service, sample_supervisor_status):
        """测试成功获取Supervisor状态"""
        mock_supervisor_service.get_supervisor_status.return_value = sample_supervisor_status
        
        response = client.get("/supervisor/status?supervisor_id=supervisor_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "状态查询成功"
        assert data["data"]["supervisor_name"] == "main_supervisor"
        assert data["data"]["status"] == "active"
        assert len(data["data"]["available_agents"]) == 3
        assert "agent_loads" in data["data"]
    
    def test_get_supervisor_status_not_found(self, mock_supervisor_service):
        """测试Supervisor不存在"""
        mock_supervisor_service.get_supervisor_status.side_effect = ValueError("Supervisor not found")
        
        response = client.get("/supervisor/status?supervisor_id=nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
    
    def test_get_supervisor_status_missing_id(self):
        """测试缺少Supervisor ID"""
        response = client.get("/supervisor/status")
        
        assert response.status_code == 422


class TestDecisionHistory:
    """决策历史API测试"""
    
    def test_get_decision_history_success(self, mock_supervisor_service):
        """测试成功获取决策历史"""
        sample_decisions = [
            {
                "id": "decision_1",
                "decision_id": "decision_12345",
                "task_id": "task_123",
                "task_description": "实现API接口",
                "assigned_agent": "code_expert",
                "confidence_level": 0.85,
                "timestamp": "2025-08-06T10:00:00Z"
            },
            {
                "id": "decision_2",
                "decision_id": "decision_12346",
                "task_id": "task_124",
                "task_description": "设计数据库",
                "assigned_agent": "architect",
                "confidence_level": 0.92,
                "timestamp": "2025-08-06T09:30:00Z"
            }
        ]
        
        mock_supervisor_service.get_decision_history.return_value = sample_decisions
        
        response = client.get(
            "/supervisor/decisions?supervisor_id=supervisor_123&limit=5&offset=0"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 2
        assert data["data"][0]["assigned_agent"] == "code_expert"
        assert "pagination" in data
    
    def test_get_decision_history_with_pagination(self, mock_supervisor_service):
        """测试分页获取决策历史"""
        mock_supervisor_service.get_decision_history.return_value = []
        
        response = client.get(
            "/supervisor/decisions?supervisor_id=supervisor_123&limit=20&offset=10"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["pagination"]["limit"] == 20
        assert data["pagination"]["offset"] == 10


class TestConfigUpdate:
    """配置更新API测试"""
    
    def test_update_supervisor_config_success(self, mock_supervisor_service):
        """测试成功更新配置"""
        mock_supervisor_service.update_supervisor_config.return_value = {
            "config_id": "config_123",
            "message": "配置更新成功",
            "updated_fields": ["routing_strategy", "load_threshold"]
        }
        
        request_data = {
            "routing_strategy": "hybrid",
            "load_threshold": 0.7,
            "capability_weight": 0.6,
            "enable_learning": True
        }
        
        response = client.put(
            "/supervisor/config?supervisor_id=supervisor_123",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["message"] == "配置更新成功"
    
    def test_update_supervisor_config_invalid_weights(self):
        """测试无效权重配置"""
        request_data = {
            "capability_weight": 0.6,
            "load_weight": 0.5,
            "availability_weight": 0.3  # 总和超过1.0
        }
        
        response = client.put(
            "/supervisor/config?supervisor_id=supervisor_123",
            json=request_data
        )
        
        assert response.status_code == 422


class TestAgentManagement:
    """智能体管理API测试"""
    
    def test_add_agent_success(self, mock_supervisor_service):
        """测试成功添加智能体"""
        response = client.post(
            "/supervisor/agents/new_agent?supervisor_id=supervisor_123"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "new_agent" in data["message"]
    
    def test_remove_agent_success(self, mock_supervisor_service):
        """测试成功移除智能体"""
        mock_supervisor_service.remove_agent_from_supervisor.return_value = None
        
        response = client.delete(
            "/supervisor/agents/old_agent?supervisor_id=supervisor_123"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "old_agent" in data["message"]


class TestTaskCompletion:
    """任务完成API测试"""
    
    def test_update_task_completion_success(self, mock_supervisor_service):
        """测试成功更新任务完成状态"""
        mock_supervisor_service.update_task_completion.return_value = None
        
        response = client.post(
            "/supervisor/tasks/task_123/complete?success=true&quality_score=0.9"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == "task_123"
        assert data["data"]["success"] is True
        assert data["data"]["quality_score"] == 0.9
        
        # 验证服务调用
        mock_supervisor_service.update_task_completion.assert_called_once_with(
            task_id="task_123",
            success=True,
            quality_score=0.9
        )
    
    def test_update_task_completion_failure(self, mock_supervisor_service):
        """测试更新任务完成状态（失败）"""
        mock_supervisor_service.update_task_completion.return_value = None
        
        response = client.post(
            "/supervisor/tasks/task_123/complete?success=false"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["success"] is False
        assert data["data"]["quality_score"] is None


class TestSupervisorInitialization:
    """Supervisor初始化API测试"""
    
    def test_initialize_supervisor_success(self, mock_supervisor_service):
        """测试成功初始化Supervisor"""
        mock_supervisor_service.initialize_supervisor.return_value = "supervisor_456"
        
        response = client.post(
            "/supervisor/initialize?supervisor_name=test_supervisor"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["supervisor_id"] == "supervisor_456"
        assert data["data"]["name"] == "test_supervisor"
    
    def test_initialize_supervisor_error(self, mock_supervisor_service):
        """测试初始化Supervisor失败"""
        mock_supervisor_service.initialize_supervisor.side_effect = Exception("Initialization failed")
        
        response = client.post(
            "/supervisor/initialize?supervisor_name=test_supervisor"
        )
        
        assert response.status_code == 500


class TestHealthCheck:
    """健康检查API测试"""
    
    def test_health_check_success(self):
        """测试健康检查成功"""
        response = client.get("/supervisor/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Supervisor服务运行正常"
        assert data["data"]["status"] == "healthy"


class TestLoadStatistics:
    """负载统计API测试"""
    
    def test_get_load_statistics_success(self, mock_supervisor_service):
        """测试成功获取负载统计"""
        sample_stats = {
            "total_tasks_assigned": 50,
            "average_load": 0.4,
            "agent_loads": {
                "code_expert": 0.6,
                "architect": 0.2
            },
            "load_distribution": {
                "low_load": 1,
                "medium_load": 1,
                "high_load": 0
            }
        }
        
        response = client.get("/supervisor/load-statistics?supervisor_id=supervisor_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """异步端点测试"""
    
    async def test_concurrent_task_submissions(self, mock_supervisor_service, sample_task_assignment):
        """测试并发任务提交"""
        mock_supervisor_service.submit_task.return_value = sample_task_assignment
        
        request_data = {
            "name": "并发任务",
            "description": "测试并发处理",
            "task_type": "code_generation",
            "priority": "medium"
        }
        
        # 模拟并发请求
        responses = []
        for i in range(5):
            response = client.post(
                "/supervisor/tasks?supervisor_id=supervisor_123",
                json={**request_data, "name": f"并发任务_{i}"}
            )
            responses.append(response)
        
        # 验证所有请求都成功
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["success"] for r in responses)


class TestErrorHandling:
    """错误处理测试"""
    
    def test_invalid_supervisor_id_format(self):
        """测试无效的Supervisor ID格式"""
        response = client.get("/supervisor/status?supervisor_id=")
        assert response.status_code == 422
    
    def test_malformed_json_request(self):
        """测试格式错误的JSON请求"""
        response = client.post(
            "/supervisor/tasks?supervisor_id=supervisor_123",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_service_unavailable(self, mock_supervisor_service):
        """测试服务不可用"""
        mock_supervisor_service.get_supervisor_status.side_effect = Exception("Service unavailable")
        
        response = client.get("/supervisor/status?supervisor_id=supervisor_123")
        
        assert response.status_code == 500
        data = response.json()
        assert "Service unavailable" in data["detail"] or "状态查询失败" in data["detail"]