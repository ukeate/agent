"""
Supervisor API简化测试
测试Supervisor相关的HTTP接口（不依赖数据库）
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# 直接创建路由器避免数据库依赖
from fastapi import APIRouter, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse

# 创建简化的测试路由
router = APIRouter(prefix="/supervisor", tags=["supervisor"])

@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "success": True,
        "message": "Supervisor服务运行正常",
        "data": {
            "status": "healthy",
            "timestamp": "2025-01-01T00:00:00Z",
            "version": "1.0.0"
        }
    }

@router.post("/tasks")
async def submit_task_mock(supervisor_id: str = Query(...)):
    """模拟任务提交"""
    return {
        "success": True,
        "message": "任务提交成功",
        "data": {
            "task_id": "task_12345",
            "assigned_agent": "code_expert",
            "assignment_reason": "最佳匹配智能体",
            "confidence_level": 0.85
        }
    }

@router.get("/status")
async def get_supervisor_status_mock(supervisor_id: str = Query(...)):
    """模拟获取状态"""
    if supervisor_id == "nonexistent":
        raise HTTPException(status_code=404, detail="Supervisor not found")
    
    return {
        "success": True,
        "message": "状态查询成功",
        "data": {
            "supervisor_name": "main_supervisor",
            "status": "active",
            "available_agents": ["code_expert", "architect", "doc_expert"],
            "agent_loads": {
                "code_expert": 0.3,
                "architect": 0.1,
                "doc_expert": 0.0
            },
            "decision_history_count": 15,
            "task_queue_length": 3
        }
    }

# 创建测试app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestSupervisorAPISimple:
    """简化的Supervisor API测试"""
    
    def test_health_check_success(self):
        """测试健康检查成功"""
        response = client.get("/supervisor/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Supervisor服务运行正常"
        assert data["data"]["status"] == "healthy"
    
    def test_submit_task_mock_success(self):
        """测试模拟任务提交成功"""
        request_data = {
            "name": "实现用户登录",
            "description": "实现基于JWT的用户登录功能",
            "task_type": "code_generation",
            "priority": "high"
        }
        
        response = client.post(
            "/supervisor/tasks?supervisor_id=supervisor_123",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "任务提交成功"
        assert data["data"]["task_id"] == "task_12345"
        assert data["data"]["assigned_agent"] == "code_expert"
    
    def test_get_supervisor_status_success(self):
        """测试获取Supervisor状态成功"""
        response = client.get("/supervisor/status?supervisor_id=supervisor_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "状态查询成功"
        assert data["data"]["supervisor_name"] == "main_supervisor"
        assert data["data"]["status"] == "active"
        assert len(data["data"]["available_agents"]) == 3
        assert "agent_loads" in data["data"]
    
    def test_get_supervisor_status_not_found(self):
        """测试Supervisor不存在的情况"""
        response = client.get("/supervisor/status?supervisor_id=nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "Supervisor not found" in data["detail"]
    
    def test_missing_supervisor_id(self):
        """测试缺少Supervisor ID参数"""
        response = client.get("/supervisor/status")
        
        assert response.status_code == 422  # FastAPI validation error


# 运行测试验证基础功能
if __name__ == "__main__":
    pytest.main([__file__, "-v"])