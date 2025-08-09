"""
Story 1.5 API接口单元测试
测试标准化的智能体API接口
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status
import json

from main import create_app
from models.schemas import (
    ChatRequest, TaskRequest, TaskType, TaskPriority,
    AgentHealth, TaskStatus
)
from services.agent_service import AgentService


class TestAgentInterfaceAPI:
    """智能体接口API测试类"""
    
    @pytest.fixture
    def app(self):
        """创建测试应用"""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_agent_service(self):
        """模拟智能体服务"""
        mock = AsyncMock(spec=AgentService)
        
        # 模拟创建会话响应
        mock.create_agent_session.return_value = {
            "conversation_id": "test-conv-123",
            "agent_id": "test-agent-123",
            "agent_type": "react"
        }
        
        # 模拟对话响应
        mock.chat_with_agent.return_value = {
            "response": "这是测试响应",
            "conversation_id": "test-conv-123",
            "steps": 3,
            "tool_calls": [
                {
                    "tool_name": "test_tool",
                    "arguments": {"param": "value"},
                    "result": "工具执行结果",
                    "execution_time": 0.5,
                    "status": "success"
                }
            ],
            "completed": True,
            "reasoning_steps": ["步骤1", "步骤2", "步骤3"],
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }
        
        # 模拟关闭会话
        mock.close_agent_session.return_value = None
        
        # 模拟智能体属性
        mock.agents = {"agent1": MagicMock()}
        mock.conversation_service = MagicMock()
        mock.conversation_service.active_sessions = {"session1": MagicMock()}
        
        return mock
    
    @pytest.fixture
    def mock_current_user(self):
        """模拟当前用户"""
        return "test-user-123"

    # ===== POST /api/v1/agent/chat 测试 =====
    
    def test_chat_success(self, client, mock_agent_service, mock_current_user):
        """测试聊天接口成功响应"""
        with patch('api.v1.agent_interface.get_agent_service', return_value=mock_agent_service), \
             patch('api.v1.agent_interface.get_current_user', return_value=mock_current_user):
            
            request_data = {
                "message": "你好，智能体",
                "stream": False,
                "context": {"test": "value"}
            }
            
            response = client.post("/api/v1/agent/chat", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            assert "data" in data
            assert data["data"]["message"] == "这是测试响应"
            assert data["data"]["conversation_id"] == "test-conv-123"
            assert len(data["data"]["tool_calls"]) == 1
            assert data["data"]["tool_calls"][0]["tool_name"] == "test_tool"
            assert "request_id" in data
    
    def test_chat_validation_error(self, client):
        """测试聊天接口输入验证错误"""
        request_data = {
            "message": "",  # 空消息应该失败
            "stream": False
        }
        
        response = client.post("/api/v1/agent/chat", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_chat_missing_message(self, client):
        """测试聊天接口缺少消息字段"""
        request_data = {
            "stream": False
        }
        
        response = client.post("/api/v1/agent/chat", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_chat_service_error(self, client, mock_agent_service, mock_current_user):
        """测试聊天接口服务错误"""
        mock_agent_service.create_agent_session.side_effect = Exception("服务错误")
        
        with patch('api.v1.agent_interface.get_agent_service', return_value=mock_agent_service), \
             patch('api.v1.agent_interface.get_current_user', return_value=mock_current_user):
            
            request_data = {
                "message": "测试消息",
                "stream": False
            }
            
            response = client.post("/api/v1/agent/chat", json=request_data)
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    # ===== POST /api/v1/agent/task 测试 =====
    
    def test_task_success(self, client, mock_agent_service, mock_current_user):
        """测试任务执行接口成功响应"""
        with patch('api.v1.agent_interface.get_agent_service', return_value=mock_agent_service), \
             patch('api.v1.agent_interface.get_current_user', return_value=mock_current_user):
            
            request_data = {
                "description": "执行测试任务",
                "task_type": TaskType.GENERAL,
                "priority": TaskPriority.HIGH,
                "requirements": ["要求1", "要求2"],
                "constraints": {"time_limit": 300},
                "expected_output": "任务结果",
                "timeout": 600
            }
            
            response = client.post("/api/v1/agent/task", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            assert "data" in data
            assert data["data"]["status"] == TaskStatus.COMPLETED
            assert data["data"]["progress"] == 100.0
            assert data["data"]["result"]["output"] == "这是测试响应"
            assert "task_id" in data["data"]
    
    def test_task_validation_error(self, client):
        """测试任务接口输入验证错误"""
        request_data = {
            "description": "",  # 空描述应该失败
            "task_type": TaskType.GENERAL
        }
        
        response = client.post("/api/v1/agent/task", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_task_with_minimal_data(self, client, mock_agent_service, mock_current_user):
        """测试任务接口最小数据"""
        with patch('api.v1.agent_interface.get_agent_service', return_value=mock_agent_service), \
             patch('api.v1.agent_interface.get_current_user', return_value=mock_current_user):
            
            request_data = {
                "description": "简单任务"
            }
            
            response = client.post("/api/v1/agent/task", json=request_data)
            assert response.status_code == status.HTTP_200_OK

    # ===== GET /api/v1/agent/status 测试 =====
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_status_success(self, mock_disk, mock_memory, mock_cpu, client, mock_agent_service):
        """测试状态查询接口成功响应"""
        # 模拟系统资源
        mock_cpu.return_value = 25.5
        mock_memory.return_value = MagicMock(percent=45.0)
        mock_disk.return_value = MagicMock(percent=60.0)
        
        with patch('api.v1.agent_interface.get_agent_service', return_value=mock_agent_service):
            
            response = client.get("/api/v1/agent/status")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            assert "data" in data
            assert data["data"]["health"] == AgentHealth.HEALTHY
            assert data["data"]["agent_info"]["agent_type"] == "react"
            assert data["data"]["system_resources"]["cpu_usage"] == 25.5
            assert data["data"]["system_resources"]["memory_usage"] == 45.0
            assert data["data"]["performance_metrics"]["success_rate"] == 99.5
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_status_degraded_health(self, mock_disk, mock_memory, mock_cpu, client, mock_agent_service):
        """测试状态查询接口降级健康状态"""
        # 模拟高CPU使用率
        mock_cpu.return_value = 85.0
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk.return_value = MagicMock(percent=60.0)
        
        with patch('api.v1.agent_interface.get_agent_service', return_value=mock_agent_service):
            
            response = client.get("/api/v1/agent/status")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["data"]["health"] == AgentHealth.DEGRADED

    # ===== GET /api/v1/agent/metrics 测试 =====
    
    def test_metrics_success(self, client):
        """测试性能指标接口成功响应"""
        with patch('api.middleware.middleware_manager.get_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "total_requests": 100,
                "average_response_time": 0.25,
                "error_rate": 2.0,
                "endpoint_metrics": {
                    "POST /api/v1/agent/chat": {
                        "count": 50,
                        "total_time": 12.5,
                        "errors": 1
                    }
                }
            }
            
            response = client.get("/api/v1/agent/metrics")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            assert data["data"]["total_requests"] == 100
            assert data["data"]["average_response_time"] == 0.25
            assert "timestamp" in data["data"]
            assert "api_version" in data["data"]

    # ===== 流式响应测试 =====
    
    def test_chat_stream_success(self, client, mock_agent_service, mock_current_user):
        """测试聊天流式响应"""
        # 模拟流式响应
        async def mock_stream():
            yield {"step": 1, "content": "第一步"}
            yield {"step": 2, "content": "第二步"}
            yield {"step": 3, "content": "完成"}
        
        mock_agent_service.chat_with_agent.return_value = mock_stream()
        
        with patch('api.v1.agent_interface.get_agent_service', return_value=mock_agent_service), \
             patch('api.v1.agent_interface.get_current_user', return_value=mock_current_user):
            
            request_data = {
                "message": "测试流式响应",
                "stream": True
            }
            
            response = client.post("/api/v1/agent/chat", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # ===== 认证测试 =====
    
    def test_chat_unauthorized(self, client):
        """测试未认证访问聊天接口"""
        with patch('api.v1.agent_interface.get_current_user') as mock_auth:
            mock_auth.side_effect = Exception("未认证")
            
            request_data = {
                "message": "测试消息",
                "stream": False
            }
            
            response = client.post("/api/v1/agent/chat", json=request_data)
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    # ===== 并发测试 =====
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, mock_agent_service, mock_current_user):
        """测试并发请求处理"""
        import concurrent.futures
        import threading
        
        def make_request():
            with patch('api.v1.agent_interface.get_agent_service', return_value=mock_agent_service), \
                 patch('api.v1.agent_interface.get_current_user', return_value=mock_current_user):
                
                request_data = {
                    "message": f"并发测试消息",
                    "stream": False
                }
                
                response = client.post("/api/v1/agent/chat", json=request_data)
                return response.status_code
        
        # 并发执行5个请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 所有请求都应该成功
        assert all(status_code == 200 for status_code in results)


class TestMiddlewareIntegration:
    """中间件集成测试"""
    
    @pytest.fixture
    def app(self):
        return create_app()
    
    @pytest.fixture  
    def client(self, app):
        return TestClient(app)
    
    def test_performance_headers(self, client):
        """测试性能监控头部"""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        assert "X-Request-ID" in response.headers
        # 注意：由于中间件设置问题，可能需要调整测试
    
    def test_rate_limiting(self, client):
        """测试频率限制"""
        # 快速连续发送多个请求
        responses = []
        for _ in range(65):  # 超过默认60请求/分钟限制
            response = client.get("/health")
            responses.append(response.status_code)
        
        # 应该有一些请求被限制
        assert any(code == 429 for code in responses[-10:])  # 最后几个请求应该被限制
    
    def test_error_handling_format(self, client):
        """测试错误响应格式标准化"""
        # 访问不存在的端点
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # 验证错误响应格式
        data = response.json()
        assert "success" in data
        assert data["success"] is False
        assert "error" in data
        assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])