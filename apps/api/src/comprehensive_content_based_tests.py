#!/usr/bin/env python3
"""
基于代码内容的综合测试用例
集成所有API模块的深度业务逻辑测试、异常处理验证、边界条件测试
不只是测试API端点，而是全面覆盖代码执行路径和业务逻辑实现
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import threading
import time
import uuid

class TestComprehensiveAPIContentCoverage:
    """综合API代码内容覆盖测试"""

    @pytest.fixture
    def test_client(self):
        """测试客户端fixture"""
        from main import app
        return TestClient(app)

    # ============================================================================
    # Workflows API 业务逻辑深度测试
    # ============================================================================

    @patch('src.services.workflow_service.workflow_service')
    def test_workflow_create_complete_business_flow(self, mock_service, test_client):
        """测试工作流创建的完整业务流程
        
        基于workflows.py:29-35代码逻辑：
        1. 接收WorkflowCreate数据
        2. 调用workflow_service.create_workflow
        3. 异常处理：Exception -> HTTPException(400)
        4. 返回WorkflowResponse格式
        """
        # 设置正常流程的mock
        mock_service.create_workflow.return_value = {
            "id": "wf_12345",
            "name": "integration_test_workflow",
            "status": "created",
            "created_at": "2025-01-01T00:00:00Z",
            "steps": []
        }
        
        # 测试数据
        workflow_data = {
            "name": "Integration Test Workflow",
            "description": "Complete business flow test",
            "steps": [
                {"id": 1, "name": "initial_step", "type": "process"},
                {"id": 2, "name": "validation_step", "type": "validate"}
            ]
        }
        
        # 执行测试
        response = test_client.post("/api/v1/workflows/", json=workflow_data)
        
        # 验证正常流程
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == "wf_12345"
        assert result["status"] == "created"
        
        # 验证服务调用
        mock_service.create_workflow.assert_called_once()
        call_args = mock_service.create_workflow.call_args[0][0]
        assert call_args.name == "Integration Test Workflow"
        
        # 测试异常流程
        mock_service.create_workflow.side_effect = RuntimeError("Database connection failed")
        response = test_client.post("/api/v1/workflows/", json=workflow_data)
        
        assert response.status_code == 400
        assert "Database connection failed" in response.json()["detail"]

    @patch('src.services.workflow_service.workflow_service')
    def test_workflow_control_action_branching_logic(self, mock_service, test_client):
        """测试工作流控制的动作分支逻辑
        
        基于workflows.py:92-126代码逻辑：
        - pause动作：调用pause_workflow，成功返回消息，失败抛出400
        - resume动作：调用resume_workflow，成功返回消息，失败抛出400  
        - cancel动作：调用cancel_workflow，成功返回消息，失败抛出400
        - 无效动作：抛出400错误
        """
        workflow_id = "test_workflow_123"
        
        # 测试pause动作成功分支
        mock_service.pause_workflow.return_value = True
        response = test_client.put(f"/api/v1/workflows/{workflow_id}/control", json={
            "action": "pause"
        })
        
        assert response.status_code == 200
        assert response.json()["message"] == "工作流已暂停"
        mock_service.pause_workflow.assert_called_with(workflow_id)
        
        # 测试pause动作失败分支
        mock_service.pause_workflow.return_value = False
        response = test_client.put(f"/api/v1/workflows/{workflow_id}/control", json={
            "action": "pause"
        })
        
        assert response.status_code == 400
        assert "暂停工作流失败" in response.json()["detail"]
        
        # 测试resume动作分支
        mock_service.resume_workflow.return_value = True
        response = test_client.put(f"/api/v1/workflows/{workflow_id}/control", json={
            "action": "resume"
        })
        
        assert response.status_code == 200
        assert response.json()["message"] == "工作流已恢复"
        
        # 测试cancel动作分支
        mock_service.cancel_workflow.return_value = True
        response = test_client.put(f"/api/v1/workflows/{workflow_id}/control", json={
            "action": "cancel"
        })
        
        assert response.status_code == 200
        assert response.json()["message"] == "工作流已取消"
        
        # 测试无效动作分支
        response = test_client.put(f"/api/v1/workflows/{workflow_id}/control", json={
            "action": "invalid_operation"
        })
        
        assert response.status_code == 400
        assert "不支持的操作: invalid_operation" in response.json()["detail"]

    # ============================================================================
    # Multi-Agent API 业务逻辑深度测试
    # ============================================================================

    @patch('src.services.multi_agent_service.MultiAgentService')
    def test_multi_agent_conversation_config_building_logic(self, mock_service_class, test_client):
        """测试多智能体对话配置构建逻辑
        
        基于multi_agents.py:123-128代码逻辑：
        - max_rounds: request.max_rounds or 10
        - timeout_seconds: request.timeout_seconds or 300
        - auto_reply: request.auto_reply if not None else True
        """
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        mock_service.create_multi_agent_conversation.return_value = {
            "conversation_id": "conv_123",
            "status": "created",
            "participants": [],
            "created_at": "2025-01-01T00:00:00Z",
            "config": {},
            "initial_status": {}
        }
        
        # 测试默认值处理逻辑
        request_without_config = {
            "message": "Start default conversation"
        }
        
        response = test_client.post("/api/v1/multi-agent/conversation", json=request_without_config)
        
        # 验证配置构建逻辑
        call_args = mock_service.create_multi_agent_conversation.call_args
        config = call_args.kwargs['conversation_config']
        
        assert config.max_rounds == 10  # default value
        assert config.timeout_seconds == 300  # default value
        assert config.auto_reply == True  # default value
        
        # 测试自定义值覆盖逻辑
        request_with_custom_config = {
            "message": "Custom config conversation",
            "max_rounds": 25,
            "timeout_seconds": 600,
            "auto_reply": False
        }
        
        response = test_client.post("/api/v1/multi-agent/conversation", json=request_with_custom_config)
        
        call_args = mock_service.create_multi_agent_conversation.call_args
        config = call_args.kwargs['conversation_config']
        
        assert config.max_rounds == 25
        assert config.timeout_seconds == 600
        assert config.auto_reply == False

    @patch('src.services.multi_agent_service.MultiAgentService')
    def test_multi_agent_exception_type_mapping(self, mock_service_class, test_client):
        """测试多智能体API异常类型映射逻辑
        
        基于multi_agents.py异常处理逻辑：
        - ValueError -> 404 NOT_FOUND
        - 其他Exception -> 500 INTERNAL_SERVER_ERROR
        """
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        
        # 测试ValueError映射到404
        mock_service.get_conversation_status.side_effect = ValueError("Conversation not found")
        response = test_client.get("/api/v1/multi-agent/conversation/nonexistent/status")
        
        assert response.status_code == 404
        assert "Conversation not found" in response.json()["detail"]
        
        # 测试其他异常映射到500
        mock_service.get_conversation_status.side_effect = RuntimeError("Service error")
        response = test_client.get("/api/v1/multi-agent/conversation/test/status")
        
        assert response.status_code == 500
        assert "获取状态失败: Service error" in response.json()["detail"]

    # ============================================================================
    # 单例模式业务逻辑测试
    # ============================================================================

    def test_multi_agent_service_singleton_pattern(self, test_client):
        """测试多智能体服务单例模式逻辑
        
        基于multi_agents.py:26-35代码逻辑：
        1. 全局变量_multi_agent_service_instance初始为None
        2. 第一次调用创建实例并记录日志
        3. 后续调用返回同一实例
        """
        import api.v1.multi_agents as module
        
        # 保存原始状态
        original_instance = module._multi_agent_service_instance
        
        try:
            # 重置为初始状态
            module._multi_agent_service_instance = None
            
            # 第一次调用应该创建实例
            with patch('api.v1.multi_agents.logger') as mock_logger:
                response1 = test_client.get("/api/v1/multi-agent/conversation/test1/status")
                
                # 验证创建日志被记录
                mock_logger.info.assert_any_call("MultiAgentService单例实例创建成功")
            
            # 记录第一个实例
            first_instance = module._multi_agent_service_instance
            assert first_instance is not None
            
            # 第二次调用应该返回相同实例
            with patch('api.v1.multi_agents.logger') as mock_logger:
                response2 = test_client.get("/api/v1/multi-agent/conversation/test2/status")
                
                # 验证没有再次记录创建日志
                create_calls = [call for call in mock_logger.info.call_args_list 
                              if "MultiAgentService单例实例创建成功" in str(call)]
                assert len(create_calls) == 0  # 没有新的创建日志
            
            # 验证是同一实例
            second_instance = module._multi_agent_service_instance
            assert first_instance is second_instance
            
        finally:
            # 恢复原始状态
            module._multi_agent_service_instance = original_instance

    # ============================================================================
    # 请求数据验证边界条件测试
    # ============================================================================

    def test_request_validation_comprehensive_boundaries(self, test_client):
        """综合测试请求数据验证的所有边界条件"""
        
        # 工作流相关边界测试
        self._test_workflow_validation_boundaries(test_client)
        
        # 多智能体相关边界测试
        self._test_multi_agent_validation_boundaries(test_client)
    
    def _test_workflow_validation_boundaries(self, test_client):
        """工作流请求验证边界条件"""
        
        # 测试空请求体
        response = test_client.post("/api/v1/workflows/", json={})
        assert response.status_code == 422  # 缺少必需字段
        
        # 测试部分字段
        response = test_client.post("/api/v1/workflows/", json={
            "name": "test_workflow"
            # 缺少其他可能的必需字段
        })
        # 应该不是验证错误（name可能是唯一必需字段）
        assert response.status_code != 422 or "name" not in response.json()["detail"]
    
    def _test_multi_agent_validation_boundaries(self, test_client):
        """多智能体请求验证边界条件"""
        
        # message字段边界测试
        test_cases = [
            ("", 422, "空消息"),
            ("a", 200, "最短有效消息"),
            ("x" * 5000, 200, "最长有效消息"),
            ("x" * 5001, 422, "超长消息")
        ]
        
        for message, expected_status, description in test_cases:
            response = test_client.post("/api/v1/multi-agent/conversation", json={
                "message": message
            })
            
            if expected_status == 422:
                assert response.status_code == 422, f"{description}应该返回422"
            else:
                assert response.status_code != 422, f"{description}不应该返回422"

    # ============================================================================
    # WebSocket连接管理业务逻辑测试
    # ============================================================================

    def test_websocket_connection_manager_business_logic(self):
        """测试WebSocket连接管理器业务逻辑"""
        from api.v1.workflows import ConnectionManager
        
        manager = ConnectionManager()
        
        # 测试连接存储逻辑
        mock_websockets = {}
        for i in range(100):
            workflow_id = f"workflow_{i}"
            mock_ws = Mock()
            manager.active_connections[workflow_id] = mock_ws
            mock_websockets[workflow_id] = mock_ws
        
        assert len(manager.active_connections) == 100
        
        # 测试断开连接逻辑
        manager.disconnect("workflow_50")
        assert len(manager.active_connections) == 99
        assert "workflow_50" not in manager.active_connections
        
        # 测试批量清理逻辑
        to_disconnect = [f"workflow_{i}" for i in range(0, 50)]
        for workflow_id in to_disconnect:
            if workflow_id in manager.active_connections:
                manager.disconnect(workflow_id)
        
        remaining_count = len(manager.active_connections)
        assert remaining_count < 100

    # ============================================================================
    # 并发和资源管理测试
    # ============================================================================

    def test_concurrent_request_handling_stability(self, test_client):
        """测试并发请求处理稳定性"""
        
        def make_health_check_request(results, request_id):
            """执行健康检查请求"""
            try:
                start_time = time.time()
                response = test_client.get("/api/v1/workflows/health/check")
                end_time = time.time()
                
                results.append({
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                })
            except Exception as e:
                results.append({
                    "request_id": request_id,
                    "error": str(e),
                    "success": False
                })
        
        # 并发测试
        results = []
        threads = []
        
        # 启动并发请求
        for i in range(50):
            thread = threading.Thread(
                target=make_health_check_request, 
                args=(results, i)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有请求完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == 50
        
        successful_requests = [r for r in results if r.get("success", False)]
        success_rate = len(successful_requests) / len(results)
        
        # 成功率应该很高（至少95%）
        assert success_rate >= 0.95, f"Success rate too low: {success_rate}"
        
        # 验证响应时间分布合理
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # 最大响应时间不应该过度超过平均值
            assert max_response_time < avg_response_time * 10

    # ============================================================================
    # 异常传播和错误处理测试
    # ============================================================================

    @patch('src.services.workflow_service.workflow_service')
    def test_exception_propagation_consistency(self, mock_service, test_client):
        """测试异常传播的一致性"""
        
        # 定义异常测试用例
        exception_test_cases = [
            (ValueError("Not found"), 404, "ValueError应映射到404"),
            (RuntimeError("Runtime error"), 400, "RuntimeError应映射到400"),
            (ConnectionError("Connection failed"), 400, "ConnectionError应映射到400"),
            (TimeoutError("Timeout"), 400, "TimeoutError应映射到400"),
            (MemoryError("Memory error"), 400, "MemoryError应映射到400"),
            (Exception("Generic error"), 400, "通用Exception应映射到400")
        ]
        
        # 测试不同端点的异常处理一致性
        endpoints = [
            ("GET", "/api/v1/workflows/test", "get_workflow_status"),
            ("POST", "/api/v1/workflows/test/start", "start_workflow"),
            ("PUT", "/api/v1/workflows/test/control", "pause_workflow")  # 简化测试
        ]
        
        for exception, expected_status, description in exception_test_cases:
            for method, endpoint, service_method in endpoints:
                # 设置服务方法抛出异常
                getattr(mock_service, service_method).side_effect = exception
                
                if method == "GET":
                    response = test_client.get(endpoint)
                elif method == "POST":
                    response = test_client.post(endpoint, json={})
                elif method == "PUT":
                    response = test_client.put(endpoint, json={"action": "pause"})
                
                # 验证状态码一致性
                if expected_status == 404 and isinstance(exception, ValueError):
                    assert response.status_code == 404, f"{description} in {endpoint}"
                else:
                    assert response.status_code == 400, f"{description} in {endpoint}"

    # ============================================================================
    # 数据完整性和状态一致性测试
    # ============================================================================

    @patch('src.services.workflow_service.workflow_service')
    def test_workflow_state_consistency_logic(self, mock_service, test_client):
        """测试工作流状态一致性逻辑"""
        
        # 模拟工作流生命周期
        workflow_states = [
            ("created", True, True, True),      # 可以启动、控制、删除
            ("running", False, True, True),     # 不能启动、可以控制、可以删除
            ("paused", False, True, True),      # 不能启动、可以控制、可以删除
            ("completed", False, False, True),  # 不能启动、不能控制、可以删除
            ("failed", False, False, True),     # 不能启动、不能控制、可以删除
        ]
        
        workflow_id = "state_test_workflow"
        
        for state, can_start, can_control, can_delete in workflow_states:
            # 设置工作流状态
            mock_service.get_workflow_status.return_value = {
                "id": workflow_id,
                "status": state,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:05:00Z"
            }
            
            # 测试启动操作
            if can_start:
                mock_service.start_workflow.return_value = {"id": workflow_id, "status": "starting"}
                response = test_client.post(f"/api/v1/workflows/{workflow_id}/start")
                assert response.status_code == 200, f"应该能启动{state}状态的工作流"
            else:
                mock_service.start_workflow.side_effect = ValueError("Cannot start workflow in current state")
                response = test_client.post(f"/api/v1/workflows/{workflow_id}/start")
                assert response.status_code == 404, f"不应该能启动{state}状态的工作流"
            
            # 重置side_effect
            mock_service.start_workflow.side_effect = None

    # ============================================================================
    # 综合集成测试
    # ============================================================================

    @patch('src.services.workflow_service.workflow_service')
    @patch('src.services.multi_agent_service.MultiAgentService')
    def test_comprehensive_api_integration(self, mock_multi_service_class, mock_workflow_service, test_client):
        """综合API集成测试"""
        
        # 设置workflow服务mock
        mock_workflow_service.create_workflow.return_value = {
            "id": "integration_wf_123",
            "name": "integration_test",
            "status": "created"
        }
        
        # 设置multi-agent服务mock
        mock_multi_service = AsyncMock()
        mock_multi_service_class.return_value = mock_multi_service
        mock_multi_service.create_multi_agent_conversation.return_value = {
            "conversation_id": "integration_conv_123",
            "status": "created",
            "participants": [],
            "created_at": "2025-01-01T00:00:00Z",
            "config": {},
            "initial_status": {}
        }
        
        # 集成测试场景：创建工作流 + 创建对话
        workflow_response = test_client.post("/api/v1/workflows/", json={
            "name": "Integration Workflow",
            "description": "Full integration test"
        })
        
        conversation_response = test_client.post("/api/v1/multi-agent/conversation", json={
            "message": "Start integration conversation",
            "max_rounds": 15
        })
        
        # 验证两个API都正常工作
        assert workflow_response.status_code == 200
        assert conversation_response.status_code == 200
        
        workflow_data = workflow_response.json()
        conversation_data = conversation_response.json()
        
        assert workflow_data["id"] == "integration_wf_123"
        assert conversation_data["conversation_id"] == "integration_conv_123"
        
        # 验证服务调用
        mock_workflow_service.create_workflow.assert_called_once()
        mock_multi_service.create_multi_agent_conversation.assert_called_once()

    # ============================================================================
    # 内存和性能边界测试
    # ============================================================================

    def test_memory_usage_boundaries(self, test_client):
        """测试内存使用边界条件"""
        
        # 创建大型但合法的请求数据
        large_workflow_request = {
            "name": "Large Workflow Test",
            "description": "x" * 2000,  # 2KB description
            "metadata": {
                "tags": [f"tag_{i}" for i in range(100)],
                "properties": {f"prop_{i}": f"value_{i}" * 10 for i in range(50)}
            }
        }
        
        # 应该能处理大型请求
        response = test_client.post("/api/v1/workflows/", json=large_workflow_request)
        
        # 不应该是内存相关错误
        assert response.status_code != 413  # Payload Too Large
        assert response.status_code != 507  # Insufficient Storage
        
        # 创建多个大型请求测试内存管理
        responses = []
        for i in range(10):
            large_request = {
                "name": f"Large Test {i}",
                "description": "x" * 1000,
                "data": [f"item_{j}" for j in range(100)]
            }
            response = test_client.post("/api/v1/workflows/", json=large_request)
            responses.append(response.status_code)
        
        # 所有请求都应该得到一致的处理
        unique_statuses = set(responses)
        assert len(unique_statuses) <= 2  # 最多两种状态（成功和业务失败）

if __name__ == "__main__":
    """
    运行基于代码内容的综合测试
    
    测试覆盖策略：
    1. 业务逻辑完整性：测试每个API函数的完整执行路径
    2. 异常处理准确性：验证异常类型到HTTP状态码的映射
    3. 边界条件稳定性：测试各种极端输入和系统状态  
    4. 数据一致性：验证状态转换和数据完整性
    5. 并发处理能力：测试多线程和高并发场景
    6. 资源管理：验证内存、连接等资源的合理使用
    7. 集成协作：测试多个API模块之间的协作
    8. 设计模式：验证单例、工厂等设计模式的正确实现
    
    测试价值：
    - 超越简单的端点测试，深入验证业务逻辑实现
    - 确保代码在各种异常情况下的稳定性
    - 验证系统的可扩展性和性能表现
    - 提供对代码质量的全面评估
    """
    pytest.main([__file__, "-v", "--tb=short", "--maxfail=5"])
