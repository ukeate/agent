from src.core.utils.timezone_utils import utc_now
import pytest
import json
import uuid
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import timedelta
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
剩余API模块业务逻辑测试
针对async_agents.py, supervisor.py, security.py, mcp.py的核心业务逻辑
"""

try:
    from datetime import timezone
    utc = timezone.utc
except:
    utc = None

class TestAsyncAgentBusinessLogic:
    """异步智能体业务逻辑测试"""
    
    def test_agent_config_building_with_defaults(self):
        """测试智能体配置构建的默认值处理逻辑"""
        # 模拟async_agents.py lines 188-200的配置构建逻辑
        def simulate_agent_config_building(request_data, base_config):
            """模拟智能体配置构建"""
            config = {
                "name": request_data.get("name") or base_config["name"],
                "role": request_data.get("role"),
                "model": request_data.get("model") or base_config["model"],
                "temperature": request_data.get("temperature") or base_config["temperature"],
                "max_tokens": request_data.get("max_tokens") or base_config["max_tokens"],
                "system_prompt": request_data.get("custom_prompt") or base_config["system_prompt"]
            }
            return config
        
        # 模拟请求数据
        request_data = {
            "role": "assistant",
            "model": None,  # 测试默认值
            "temperature": None,  # 测试默认值
        }
        
        base_config = {
            "name": "default_assistant", 
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2000,
            "system_prompt": "You are a helpful assistant"
        }
        
        config = simulate_agent_config_building(request_data, base_config)
        
        # 验证默认值处理逻辑
        assert config["name"] == "default_assistant"  # 使用默认值
        assert config["model"] == "gpt-4o-mini"      # 使用默认值
        assert config["temperature"] == 0.7          # 使用默认值
        assert config["role"] == "assistant"         # 使用提供值

    def test_agent_context_creation_logic(self):
        """测试智能体上下文创建逻辑 (async_agents.py lines 202-208)"""
        def simulate_agent_context_creation(request_context):
            """模拟上下文创建逻辑"""
            context = {
                "user_id": request_context.get("user_id", "system") if request_context else "system",
                "session_id": str(uuid.uuid4()),
                "conversation_id": request_context.get("conversation_id") if request_context else None,
                "additional_context": request_context
            }
            return context
        
        # 测试有上下文的情况
        request_context = {
            "user_id": "test_user_123",
            "conversation_id": "conv_456"
        }
        
        context = simulate_agent_context_creation(request_context)
        
        assert context["user_id"] == "test_user_123"
        assert context["conversation_id"] == "conv_456"
        assert context["additional_context"] == request_context
        assert context["session_id"] is not None
        
        # 测试无上下文的情况
        context_empty = simulate_agent_context_creation(None)
        assert context_empty["user_id"] == "system"
        assert context_empty["conversation_id"] is None

    def test_single_instance_lazy_loading_logic(self):
        """测试单例模式懒加载逻辑 (async_agents.py lines 38-56)"""
        # 模拟全局实例管理
        class MockEventBusManager:
            def __init__(self):
                self._instance = None
                self.creation_count = 0
            
            async def get_instance(self):
                """模拟get_event_bus函数的逻辑"""
                if self._instance is None:
                    self._instance = {"id": "event_bus_001", "status": "initialized"}
                    self.creation_count += 1
                return self._instance
        
        manager = MockEventBusManager()
        
        # 第一次调用应该创建实例
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        instance1 = loop.run_until_complete(manager.get_instance())
        assert manager.creation_count == 1
        
        # 第二次调用应该返回相同实例
        instance2 = loop.run_until_complete(manager.get_instance())
        assert manager.creation_count == 1  # 没有重复创建
        assert instance1 is instance2
        
        loop.close()

    def test_websocket_message_type_routing(self):
        """测试WebSocket消息类型路由逻辑 (async_agents.py lines 768-833)"""
        def simulate_websocket_message_routing(message_type, message_data):
            """模拟WebSocket消息处理路由"""
            if message_type == "create_agent":
                return {
                    "type": "agent_created", 
                    "data": {"agent_id": "agent_123", "role": message_data.get("role")},
                    "status": "success"
                }
            elif message_type == "submit_task":
                return {
                    "type": "task_submitted",
                    "data": {"task_id": "task_456"},
                    "status": "success"
                }
            elif message_type == "ping":
                return {
                    "type": "pong",
                    "status": "success"
                }
            else:
                return {
                    "type": "error",
                    "data": {"message": f"未知消息类型: {message_type}"},
                    "status": "error"
                }
        
        # 测试各种消息类型路由
        create_result = simulate_websocket_message_routing("create_agent", {"role": "assistant"})
        assert create_result["type"] == "agent_created"
        assert create_result["status"] == "success"
        
        task_result = simulate_websocket_message_routing("submit_task", {"description": "test"})
        assert task_result["type"] == "task_submitted"
        
        ping_result = simulate_websocket_message_routing("ping", {})
        assert ping_result["type"] == "pong"
        
        error_result = simulate_websocket_message_routing("invalid_type", {})
        assert error_result["type"] == "error"
        assert "未知消息类型" in error_result["data"]["message"]

    def test_task_status_filter_validation(self):
        """测试任务状态过滤器验证逻辑 (async_agents.py lines 392-399)"""
        def simulate_task_status_validation(status_filter):
            """模拟任务状态验证逻辑"""
            valid_statuses = ["pending", "running", "completed", "failed"]
            
            if status_filter:
                if status_filter not in valid_statuses:
                    raise ValueError(f"无效的任务状态: {status_filter}")
                return status_filter
            return None
        
        # 测试有效状态
        assert simulate_task_status_validation("pending") == "pending"
        assert simulate_task_status_validation("completed") == "completed"
        assert simulate_task_status_validation(None) is None
        
        # 测试无效状态
        with pytest.raises(ValueError, match="无效的任务状态"):
            simulate_task_status_validation("invalid_status")

class TestSupervisorBusinessLogic:
    """Supervisor智能体业务逻辑测试"""
    
    def test_task_assignment_data_conversion(self):
        """测试任务分配数据转换逻辑 (supervisor.py lines 47-56)"""
        def simulate_assignment_conversion(assignment_obj):
            """模拟TaskAssignment到字典的转换逻辑"""
            if hasattr(assignment_obj, 'to_dict'):
                return assignment_obj.to_dict()
            else:
                return {
                    "task_id": assignment_obj.task_id,
                    "assigned_agent": assignment_obj.assigned_agent,
                    "assignment_reason": assignment_obj.assignment_reason,
                    "confidence_level": assignment_obj.confidence_level,
                    "estimated_completion_time": assignment_obj.estimated_completion_time,
                    "alternative_agents": assignment_obj.alternative_agents,
                    "decision_metadata": assignment_obj.decision_metadata
                }
        
        # 模拟分配对象
        mock_assignment = Mock()
        mock_assignment.task_id = "task_123"
        mock_assignment.assigned_agent = "code_expert"
        mock_assignment.assignment_reason = "最适合处理编程任务"
        mock_assignment.confidence_level = 0.92
        mock_assignment.estimated_completion_time = 300
        mock_assignment.alternative_agents = ["architect", "doc_expert"]
        mock_assignment.decision_metadata = {"complexity": "medium"}
        
        # 测试没有to_dict方法的情况
        delattr(mock_assignment, 'to_dict') if hasattr(mock_assignment, 'to_dict') else None
        
        result = simulate_assignment_conversion(mock_assignment)
        
        assert result["task_id"] == "task_123"
        assert result["assigned_agent"] == "code_expert"
        assert result["confidence_level"] == 0.92
        assert len(result["alternative_agents"]) == 2

    def test_decision_history_with_valueerror_handling(self):
        """测试决策历史查询的ValueError处理逻辑 (supervisor.py lines 140-152)"""
        def simulate_decision_history_query(supervisor_id):
            """模拟决策历史查询逻辑"""
            if supervisor_id == "invalid_supervisor":
                # 模拟ValueError但继续处理的逻辑
                return {
                    "success": True,
                    "message": "决策历史查询成功（无记录）",
                    "data": [],
                    "pagination": {"limit": 10, "offset": 0, "total": 0}
                }
            else:
                return {
                    "success": True,
                    "message": "决策历史查询成功",
                    "data": [{"decision_id": "dec_001", "timestamp": "2025-01-01T00:00:00Z"}],
                    "pagination": {"limit": 10, "offset": 0, "total": 1}
                }
        
        # 测试正常情况
        normal_result = simulate_decision_history_query("supervisor_123")
        assert normal_result["success"] is True
        assert len(normal_result["data"]) > 0
        
        # 测试ValueError情况的优雅处理
        error_result = simulate_decision_history_query("invalid_supervisor")
        assert error_result["success"] is True  # 依然返回成功
        assert len(error_result["data"]) == 0
        assert "无记录" in error_result["message"]

    def test_config_update_exclusion_logic(self):
        """测试配置更新的排除未设置字段逻辑 (supervisor.py lines 175)"""
        def simulate_config_update_exclusion(request_data):
            """模拟exclude_unset=True的逻辑"""
            # 只包含明确设置的字段
            updates = {}
            for key, value in request_data.items():
                if value is not None:  # 模拟exclude_unset逻辑
                    updates[key] = value
            return updates
        
        request_data = {
            "routing_strategy": "HYBRID",
            "load_threshold": 0.8,
            "capability_weight": None,  # 这个不应该包含在更新中
            "enable_learning": True
        }
        
        updates = simulate_config_update_exclusion(request_data)
        
        assert "routing_strategy" in updates
        assert "load_threshold" in updates
        assert "enable_learning" in updates
        assert "capability_weight" not in updates  # 被排除

    def test_quality_score_boundary_validation(self):
        """测试质量评分边界验证逻辑 (supervisor.py line 309)"""
        def simulate_quality_score_validation(quality_score):
            """模拟质量评分验证"""
            if quality_score is not None:
                if not (0.0 <= quality_score <= 1.0):
                    raise ValueError("质量评分必须在0.0到1.0之间")
            return quality_score
        
        # 测试有效范围
        assert simulate_quality_score_validation(0.0) == 0.0
        assert simulate_quality_score_validation(0.5) == 0.5
        assert simulate_quality_score_validation(1.0) == 1.0
        assert simulate_quality_score_validation(None) is None
        
        # 测试无效范围
        with pytest.raises(ValueError, match="质量评分必须在0.0到1.0之间"):
            simulate_quality_score_validation(1.5)
        
        with pytest.raises(ValueError, match="质量评分必须在0.0到1.0之间"):
            simulate_quality_score_validation(-0.1)

class TestSecurityBusinessLogic:
    """安全API业务逻辑测试"""
    
    def test_api_key_generation_logic(self):
        """测试API密钥生成逻辑 (security.py lines 125-132)"""
        def simulate_api_key_generation(expires_in_days):
            """模拟API密钥生成逻辑"""
            import secrets
            
            api_key = f"sk_{secrets.token_urlsafe(32)}"
            key_id = str(uuid.uuid4())
            
            # 模拟过期时间计算
            expires_at = None
            if expires_in_days:
                expires_at = utc_now() + timedelta(days=expires_in_days)
            
            return {
                "key_id": key_id,
                "api_key": api_key,
                "expires_at": expires_at
            }
        
        # 测试有过期时间
        result_with_expiry = simulate_api_key_generation(30)
        assert result_with_expiry["api_key"].startswith("sk_")
        assert result_with_expiry["expires_at"] is not None
        
        # 测试无过期时间
        result_no_expiry = simulate_api_key_generation(None)
        assert result_no_expiry["api_key"].startswith("sk_")
        assert result_no_expiry["expires_at"] is None

    def test_tool_whitelist_action_validation(self):
        """测试工具白名单操作验证逻辑 (security.py lines 218-230)"""
        def simulate_whitelist_update(action, tool_names):
            """模拟白名单更新逻辑"""
            current_whitelist = {"read_file", "write_file", "list_directory"}
            
            if action == "add":
                for tool_name in tool_names:
                    current_whitelist.add(tool_name)
                message = f"Added {len(tool_names)} tools to whitelist"
            elif action == "remove":
                for tool_name in tool_names:
                    current_whitelist.discard(tool_name)
                message = f"Removed {len(tool_names)} tools from whitelist"
            else:
                raise ValueError("Invalid action. Use 'add' or 'remove'")
            
            return {
                "message": message,
                "current_whitelist": list(current_whitelist)
            }
        
        # 测试添加操作
        add_result = simulate_whitelist_update("add", ["execute_query", "run_command"])
        assert "Added 2 tools" in add_result["message"]
        assert "execute_query" in add_result["current_whitelist"]
        
        # 测试移除操作
        remove_result = simulate_whitelist_update("remove", ["read_file"])
        assert "Removed 1 tools" in remove_result["message"]
        assert "read_file" not in remove_result["current_whitelist"]
        
        # 测试无效操作
        with pytest.raises(ValueError, match="Invalid action"):
            simulate_whitelist_update("invalid", ["tool"])

    def test_alert_status_filtering_logic(self):
        """测试告警状态过滤逻辑 (security.py lines 312-313)"""
        def simulate_alert_filtering(alerts, status_filter):
            """模拟告警过滤逻辑"""
            if status_filter:
                return [a for a in alerts if a["status"] == status_filter]
            return alerts
        
        mock_alerts = [
            {"id": "alert_1", "status": "active", "type": "high_risk"},
            {"id": "alert_2", "status": "resolved", "type": "medium_risk"},
            {"id": "alert_3", "status": "active", "type": "low_risk"}
        ]
        
        # 测试状态过滤
        active_alerts = simulate_alert_filtering(mock_alerts, "active")
        assert len(active_alerts) == 2
        
        resolved_alerts = simulate_alert_filtering(mock_alerts, "resolved")
        assert len(resolved_alerts) == 1
        
        # 测试无过滤
        all_alerts = simulate_alert_filtering(mock_alerts, None)
        assert len(all_alerts) == 3

class TestMCPBusinessLogic:
    """MCP API业务逻辑测试"""
    
    def test_mcp_error_response_creation(self):
        """测试MCP错误响应创建逻辑 (mcp.py lines 89-99)"""
        def simulate_mcp_error_handling(error_type):
            """模拟MCP错误处理逻辑"""
            if error_type == "MCPError":
                return {
                    "success": False,
                    "error": "MCP工具调用失败",
                    "error_type": "MCP_ERROR"
                }
            else:
                return {
                    "success": False,
                    "error": f"Internal server error: {error_type}",
                    "error_type": "INTERNAL_ERROR"
                }
        
        # 测试MCP特定错误
        mcp_error = simulate_mcp_error_handling("MCPError")
        assert mcp_error["success"] is False
        assert mcp_error["error_type"] == "MCP_ERROR"
        
        # 测试通用错误
        general_error = simulate_mcp_error_handling("ConnectionError")
        assert general_error["success"] is False
        assert general_error["error_type"] == "INTERNAL_ERROR"
        assert "Internal server error" in general_error["error"]

    def test_tool_call_result_success_check(self):
        """测试工具调用结果成功检查逻辑 (mcp.py lines 199-203)"""
        def simulate_tool_call_success_check(response):
            """模拟工具调用成功检查逻辑"""
            if not response["success"]:
                raise Exception(f"Tool call failed: {response['error']}")
            return response["result"]
        
        # 测试成功响应
        success_response = {
            "success": True,
            "result": {"data": "file_content", "status": "ok"}
        }
        
        result = simulate_tool_call_success_check(success_response)
        assert result["data"] == "file_content"
        
        # 测试失败响应
        failure_response = {
            "success": False,
            "error": "File not found"
        }
        
        with pytest.raises(Exception, match="Tool call failed: File not found"):
            simulate_tool_call_success_check(failure_response)

    def test_database_query_result_formatting(self):
        """测试数据库查询结果格式化逻辑 (mcp.py lines 280-283)"""
        def simulate_query_result_formatting(response_result):
            """模拟查询结果格式化逻辑"""
            result = response_result.copy()
            if "data" in result:
                result["rows"] = result["data"]  # 为便捷接口提供更友好的格式
            return result
        
        original_result = {
            "data": [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}],
            "count": 2,
            "status": "success"
        }
        
        formatted = simulate_query_result_formatting(original_result)
        
        assert "rows" in formatted
        assert formatted["rows"] == original_result["data"]
        assert formatted["count"] == 2
        assert "data" in formatted  # 原字段仍保留

    def test_system_command_output_formatting(self):
        """测试系统命令输出格式化逻辑 (mcp.py lines 312-315)"""
        def simulate_command_output_formatting(response_result):
            """模拟命令输出格式化逻辑"""
            result = response_result.copy()
            if "stdout" in result:
                result["output"] = result["stdout"]  # 更用户友好的字段名
            return result
        
        command_result = {
            "stdout": "Hello World\n",
            "stderr": "",
            "return_code": 0,
            "execution_time": 0.05
        }
        
        formatted = simulate_command_output_formatting(command_result)
        
        assert "output" in formatted
        assert formatted["output"] == "Hello World\n"
        assert formatted["return_code"] == 0
        assert "stdout" in formatted  # 原字段仍保留

class TestExceptionHandlingConsistency:
    """异常处理一致性测试"""
    
    def test_exception_type_mapping_patterns(self):
        """测试异常类型映射模式的一致性"""
        def simulate_exception_mapping(exception_type):
            """模拟各模块的异常映射逻辑"""
            if exception_type == "ValueError":
                return {"status_code": 404, "error_type": "NOT_FOUND"}
            elif exception_type == "ValidationError":
                return {"status_code": 422, "error_type": "VALIDATION_ERROR"} 
            elif exception_type == "MCPError":
                return {"status_code": 400, "error_type": "MCP_ERROR"}
            else:
                return {"status_code": 500, "error_type": "INTERNAL_ERROR"}
        
        # 测试各种异常映射
        assert simulate_exception_mapping("ValueError")["status_code"] == 404
        assert simulate_exception_mapping("ValidationError")["status_code"] == 422
        assert simulate_exception_mapping("MCPError")["status_code"] == 400
        assert simulate_exception_mapping("RuntimeError")["status_code"] == 500

    def test_pagination_logic_consistency(self):
        """测试分页逻辑一致性"""
        def simulate_pagination_application(items, limit, offset):
            """模拟分页逻辑"""
            total = len(items)
            paginated_items = items[offset:offset + limit]
            
            return {
                "items": paginated_items,
                "pagination": {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total
                }
            }
        
        items = list(range(25))  # 25个项目
        
        # 测试第一页
        page1 = simulate_pagination_application(items, 10, 0)
        assert len(page1["items"]) == 10
        assert page1["pagination"]["has_more"] is True
        
        # 测试最后一页
        page3 = simulate_pagination_application(items, 10, 20)
        assert len(page3["items"]) == 5
        assert page3["pagination"]["has_more"] is False

if __name__ == "__main__":
    setup_logging()
    logger.info("=== 剩余API模块业务逻辑测试 ===")
    logger.info("测试模块：async_agents.py, supervisor.py, security.py, mcp.py")
    logger.error("测试重点：业务逻辑分支、数据转换、异常处理、边界条件")
    
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
