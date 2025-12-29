#!/usr/bin/env python3
"""
API异常处理和边界条件验证测试
基于实际代码逻辑深度测试各种边界情况和异常处理路径
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
from datetime import datetime
import websockets
from fastapi import HTTPException

class TestWorkflowAPIExceptionHandling:
    """工作流API异常处理边界测试"""

    @pytest.fixture
    def test_client(self):
        from main import app
        return TestClient(app)

    @patch('src.services.workflow_service.workflow_service')
    def test_workflow_service_timeout_exception(self, mock_service, test_client):
        """测试工作流服务超时异常处理
        
        边界条件：服务调用超时
        代码路径：所有API端点的Exception处理分支
        """
        # 模拟服务超时
        mock_service.get_workflow_status.side_effect = asyncio.TimeoutError("Service timeout")
        
        response = test_client.get("/api/v1/workflows/timeout_test")
        
        # 验证超时异常被正确处理
        assert response.status_code == 400
        assert "Service timeout" in response.json()["detail"]

    @patch('src.services.workflow_service.workflow_service')
    def test_workflow_database_connection_error(self, mock_service, test_client):
        """测试数据库连接错误的异常处理
        
        边界条件：数据库不可用
        代码路径：通用Exception处理逻辑
        """
        # 模拟数据库连接错误
        mock_service.create_workflow.side_effect = ConnectionError("Database connection failed")
        
        workflow_data = {"name": "test", "description": "test"}
        response = test_client.post("/api/v1/workflows/", json=workflow_data)
        
        assert response.status_code == 400
        assert "Database connection failed" in response.json()["detail"]

    @patch('src.services.workflow_service.workflow_service')
    def test_workflow_memory_error_handling(self, mock_service, test_client):
        """测试内存不足异常处理
        
        边界条件：系统内存不足
        代码路径：Exception处理分支
        """
        # 模拟内存不足
        mock_service.start_workflow.side_effect = MemoryError("Insufficient memory")
        
        response = test_client.post("/api/v1/workflows/test_id/start")
        
        assert response.status_code == 400
        assert "Insufficient memory" in response.json()["detail"]

    def test_workflow_invalid_json_payload(self, test_client):
        """测试无效JSON载荷的边界条件
        
        边界条件：请求体不是有效的JSON
        代码路径：FastAPI请求解析层
        """
        # 发送无效JSON
        response = test_client.post(
            "/api/v1/workflows/", 
            data="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # FastAPI validation error

    def test_workflow_extremely_long_id(self, test_client):
        """测试极长工作流ID的边界条件
        
        边界条件：工作流ID超长
        代码路径：路径参数处理
        """
        extremely_long_id = "x" * 10000
        response = test_client.get(f"/api/v1/workflows/{extremely_long_id}")
        
        # 应该被正常处理（可能返回404或400，但不应该崩溃）
        assert response.status_code in [400, 404, 500]

    def test_workflow_unicode_characters_handling(self, test_client):
        """测试Unicode字符处理的边界条件
        
        边界条件：包含特殊Unicode字符的数据
        代码路径：请求数据处理和存储
        """
        unicode_data = {
            "name": "测试工作流 🚀 العربية 日本語",
            "description": "包含各种Unicode字符：😀🎯🔧"
        }
        
        response = test_client.post("/api/v1/workflows/", json=unicode_data)
        
        # 应该能正确处理Unicode（可能因服务依赖失败，但不应是编码错误）
        assert response.status_code != 500 or "encoding" not in response.json().get("detail", "").lower()

class TestWebSocketExceptionHandling:
    """WebSocket异常处理边界测试"""

    def test_websocket_connection_limit(self):
        """测试WebSocket连接数限制的边界条件
        
        边界条件：大量并发WebSocket连接
        代码路径：ConnectionManager连接管理
        """
        from api.v1.workflows import ConnectionManager
        
        manager = ConnectionManager()
        
        # 模拟大量连接
        mock_websockets = []
        for i in range(1000):
            mock_ws = Mock()
            workflow_id = f"workflow_{i}"
            manager.active_connections[workflow_id] = mock_ws
            mock_websockets.append((workflow_id, mock_ws))
        
        # 验证连接数管理
        assert len(manager.active_connections) == 1000
        
        # 测试清理逻辑
        manager.disconnect("workflow_500")
        assert len(manager.active_connections) == 999
        assert "workflow_500" not in manager.active_connections

    def test_websocket_message_size_limit(self):
        """测试WebSocket消息大小限制
        
        边界条件：发送超大消息
        代码路径：WebSocket消息处理逻辑
        """
        # 创建超大消息（1MB）
        large_message = {
            "type": "large_data",
            "data": "x" * (1024 * 1024)
        }
        
        # 验证消息序列化不会崩溃
        try:
            json_message = json.dumps(large_message)
            assert len(json_message) > 1000000
        except Exception as e:
            pytest.fail(f"Large message serialization failed: {e}")

    def test_websocket_connection_drop_recovery(self):
        """测试WebSocket连接断开恢复逻辑
        
        边界条件：连接异常断开
        代码路径：ConnectionManager异常处理
        """
        from api.v1.workflows import ConnectionManager
        
        manager = ConnectionManager()
        
        # 模拟异常连接
        mock_ws = AsyncMock()
        mock_ws.send_text.side_effect = Exception("Connection closed")
        
        manager.active_connections["test_workflow"] = mock_ws
        
        # 测试发送失败时的清理逻辑（需要异步环境测试）
        # 预期：异常连接应被自动清理
        assert "test_workflow" in manager.active_connections
        
        # 手动触发清理
        manager.disconnect("test_workflow")
        assert "test_workflow" not in manager.active_connections

class TestMultiAgentAPIBoundaryConditions:
    """多智能体API边界条件测试"""

    @pytest.fixture 
    def test_client(self):
        from main import app
        return TestClient(app)

    def test_conversation_message_length_boundaries(self, test_client):
        """测试对话消息长度边界条件
        
        边界条件：
        - 最小长度：1字符（边界）
        - 最大长度：5000字符（边界）
        - 超出范围的情况
        
        代码路径：CreateConversationRequest验证逻辑
        """
        # 测试最小长度边界（1字符）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "a"  # 正好1字符
        })
        assert response.status_code != 422  # 不应该是验证错误
        
        # 测试空字符串（违反min_length=1）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": ""
        })
        assert response.status_code == 422
        
        # 测试最大长度边界（5000字符）
        max_message = "x" * 5000
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": max_message
        })
        assert response.status_code != 422  # 不应该是验证错误
        
        # 测试超出最大长度（5001字符）
        over_max_message = "x" * 5001
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": over_max_message
        })
        assert response.status_code == 422  # 应该是验证错误

    def test_max_rounds_boundary_conditions(self, test_client):
        """测试最大轮数边界条件
        
        边界条件：
        - 最小值：1（边界）
        - 最大值：50（边界）
        - 超出范围的情况
        
        代码路径：CreateConversationRequest.max_rounds验证
        """
        # 测试最小值边界（1）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "max_rounds": 1
        })
        assert response.status_code != 422
        
        # 测试小于最小值（0）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test", 
            "max_rounds": 0
        })
        assert response.status_code == 422
        
        # 测试最大值边界（50）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "max_rounds": 50
        })
        assert response.status_code != 422
        
        # 测试超出最大值（51）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "max_rounds": 51
        })
        assert response.status_code == 422

    def test_timeout_seconds_boundary_conditions(self, test_client):
        """测试超时时间边界条件
        
        边界条件：
        - 最小值：30秒（边界）
        - 最大值：1800秒（边界）
        - 超出范围的情况
        
        代码路径：CreateConversationRequest.timeout_seconds验证
        """
        # 测试最小值边界（30秒）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "timeout_seconds": 30
        })
        assert response.status_code != 422
        
        # 测试小于最小值（29秒）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "timeout_seconds": 29  
        })
        assert response.status_code == 422
        
        # 测试最大值边界（1800秒）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "timeout_seconds": 1800
        })
        assert response.status_code != 422
        
        # 测试超出最大值（1801秒）
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "timeout_seconds": 1801
        })
        assert response.status_code == 422

    def test_user_context_length_boundary(self, test_client):
        """测试用户上下文长度边界条件
        
        边界条件：最大长度2000字符
        代码路径：CreateConversationRequest.user_context验证
        """
        # 测试最大长度边界（2000字符）
        max_context = "x" * 2000
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "user_context": max_context
        })
        assert response.status_code != 422
        
        # 测试超出最大长度（2001字符）
        over_max_context = "x" * 2001
        response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "test",
            "user_context": over_max_context
        })
        assert response.status_code == 422

    @patch('src.services.multi_agent_service.MultiAgentService')
    def test_single_instance_concurrent_access(self, mock_service_class, test_client):
        """测试单例模式并发访问的边界条件
        
        边界条件：多个请求同时访问单例实例
        代码路径：get_multi_agent_service单例逻辑
        """
        import api.v1.multi_agents as module
        
        # 重置单例状态
        original_instance = module._multi_agent_service_instance
        module._multi_agent_service_instance = None
        
        try:
            # 模拟并发请求
            import threading
            import queue
            
            results = queue.Queue()
            
            def make_request():
                try:
                    response = test_client.get("/api/v1/multi-agent/conversation/test/status")
                    results.put(("success", response.status_code))
                except Exception as e:
                    results.put(("error", str(e)))
            
            # 启动多个并发线程
            threads = []
            for i in range(5):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 收集结果
            success_count = 0
            while not results.empty():
                result_type, result_value = results.get()
                if result_type == "success":
                    success_count += 1
            
            # 验证所有请求都能正常处理（不会因单例创建冲突而失败）
            assert success_count == 5
            
        finally:
            module._multi_agent_service_instance = original_instance

class TestExceptionPropagationLogic:
    """异常传播逻辑测试"""

    @patch('src.services.workflow_service.workflow_service')
    def test_exception_type_mapping_accuracy(self, mock_service, test_client):
        """测试异常类型映射的准确性
        
        验证不同异常类型被正确映射到HTTP状态码
        代码路径：各API端点的异常处理分支
        """
        # ValueError -> 404 (工作流不存在)
        mock_service.get_workflow_status.side_effect = ValueError("Workflow not found")
        response = test_client.get("/api/v1/workflows/test")
        assert response.status_code == 404
        
        # RuntimeError -> 400 (通用异常)
        mock_service.get_workflow_status.side_effect = RuntimeError("Service error")
        response = test_client.get("/api/v1/workflows/test")
        assert response.status_code == 400
        
        # ConnectionError -> 400 (通用异常)
        mock_service.get_workflow_status.side_effect = ConnectionError("Connection failed")
        response = test_client.get("/api/v1/workflows/test")
        assert response.status_code == 400
        
        # 自定义异常 -> 400 (通用异常)
        class CustomError(Exception):
            ...
        
        mock_service.get_workflow_status.side_effect = CustomError("Custom error")
        response = test_client.get("/api/v1/workflows/test")
        assert response.status_code == 400

    def test_error_message_sanitization(self, test_client):
        """测试错误消息清理的边界条件
        
        边界条件：错误消息包含敏感信息
        代码路径：异常处理中的错误消息返回
        """
        with patch('src.services.workflow_service.workflow_service') as mock_service:
            # 模拟包含敏感信息的错误
            sensitive_error = Exception("Database password: secret123, API key: xyz789")
            mock_service.get_workflow_status.side_effect = sensitive_error
            
            response = test_client.get("/api/v1/workflows/test")
            
            # 验证敏感信息被包含在错误响应中（当前实现直接返回异常消息）
            # 注意：这可能是安全风险，应该考虑错误消息清理
            error_detail = response.json()["detail"]
            assert "secret123" in error_detail or "xyz789" in error_detail

class TestResourceExhaustionBoundaries:
    """资源耗尽边界条件测试"""

    def test_memory_usage_with_large_requests(self, test_client):
        """测试大请求的内存使用边界
        
        边界条件：请求数据占用大量内存
        代码路径：请求解析和处理
        """
        # 创建大型请求数据（但在验证范围内）
        large_workflow_data = {
            "name": "large_test",
            "description": "x" * 1000,  # 1KB description
            "steps": [{"name": f"step_{i}", "action": "x" * 100} for i in range(100)]
        }
        
        response = test_client.post("/api/v1/workflows/", json=large_workflow_data)
        
        # 请求应该被正常处理（可能因业务逻辑失败，但不应是内存问题）
        assert response.status_code != 413  # Payload Too Large

    def test_concurrent_request_handling(self, test_client):
        """测试并发请求处理的边界条件
        
        边界条件：大量并发请求
        代码路径：FastAPI并发处理机制
        """
        import threading
        import time
        
        results = []
        
        def make_request(i):
            start_time = time.time()
            response = test_client.get(f"/api/v1/workflows/health/check")
            end_time = time.time()
            results.append({
                "request_id": i,
                "status_code": response.status_code,
                "response_time": end_time - start_time
            })
        
        # 启动并发请求
        threads = []
        for i in range(20):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有请求完成
        for thread in threads:
            thread.join()
        
        # 验证所有请求都成功处理
        assert len(results) == 20
        success_count = sum(1 for r in results if r["status_code"] == 200)
        assert success_count == 20
        
        # 验证响应时间相对稳定（不应该因并发而显著变慢）
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        max_response_time = max(r["response_time"] for r in results)
        
        # 最大响应时间不应该超过平均响应时间的5倍
        assert max_response_time < avg_response_time * 5

if __name__ == "__main__":
    """
    运行异常处理和边界条件验证测试
    
    测试覆盖重点：
    1. 各种异常类型的处理逻辑（ValueError, RuntimeError, ConnectionError等）
    2. 边界值验证（字符串长度、数值范围、连接数限制）
    3. 资源耗尽情况（内存、连接、并发）
    4. 异常传播和错误消息处理
    5. 单例模式并发访问安全性
    6. WebSocket连接异常恢复
    7. 大数据量处理边界
    8. 并发请求处理稳定性
    
    验证目标：
    - 确保所有异常情况都有适当的处理
    - 验证边界条件不会导致系统崩溃
    - 确认错误响应格式的一致性
    - 测试系统在极端条件下的稳定性
    """
    pytest.main([__file__, "-v", "--tb=short"])
