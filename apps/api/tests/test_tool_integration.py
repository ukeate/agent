"""
工具集成功能验证测试
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

# 简化的工具调用结果处理测试
class TestToolIntegration:
    """工具集成功能测试类"""
    
    def test_tool_result_standardization(self):
        """测试工具调用结果标准化"""
        # 模拟不同类型的工具返回结果
        test_cases = [
            # 文本结果
            {"input": "Hello World", "expected_type": str},
            # 字典结果  
            {"input": {"status": "success", "data": "test"}, "expected_type": dict},
            # 列表结果
            {"input": ["item1", "item2"], "expected_type": list},
            # 数值结果
            {"input": 42, "expected_type": int},
            # 布尔结果
            {"input": True, "expected_type": bool},
            # 错误结果
            {"input": {"error": "工具执行失败"}, "expected_type": dict}
        ]
        
        for case in test_cases:
            result = self._standardize_tool_result(case["input"])
            assert isinstance(result, case["expected_type"])
            
            # 错误情况特殊检查
            if isinstance(case["input"], dict) and "error" in case["input"]:
                assert "error" in result
                assert result["error"] == "工具执行失败"
    
    def test_tool_call_error_recovery(self):
        """测试工具调用错误恢复机制"""
        error_scenarios = [
            # 工具不存在
            {"error_type": "tool_not_found", "tool_name": "non_existent_tool"},
            # 参数错误
            {"error_type": "invalid_params", "tool_name": "read_file", "params": {"invalid": "param"}},
            # 网络超时
            {"error_type": "timeout", "tool_name": "web_search"},
            # 权限不足
            {"error_type": "permission_denied", "tool_name": "system_command"}
        ]
        
        for scenario in error_scenarios:
            recovery_result = self._handle_tool_error(scenario)
            
            # 所有错误都应该有统一的错误格式
            assert "error" in recovery_result
            assert "error_type" in recovery_result
            assert "suggestion" in recovery_result
            
            # 验证恢复建议的合理性
            assert len(recovery_result["suggestion"]) > 0
    
    def test_tool_call_history_tracking(self):
        """测试工具调用历史追踪"""
        history_tracker = ToolCallHistoryTracker()
        
        # 模拟一系列工具调用
        calls = [
            {"tool_name": "read_file", "args": {"file_path": "test.txt"}, "result": "file content"},
            {"tool_name": "write_file", "args": {"file_path": "output.txt", "content": "data"}, "result": {"status": "success"}},
            {"tool_name": "web_search", "args": {"query": "test"}, "result": {"error": "timeout"}}
        ]
        
        for call in calls:
            history_tracker.add_call(call["tool_name"], call["args"], call["result"])
        
        # 验证历史记录
        history = history_tracker.get_history()
        assert len(history) == 3
        
        # 验证成功调用统计
        successful_calls = history_tracker.get_successful_calls()
        assert len(successful_calls) == 2
        
        # 验证失败调用统计
        failed_calls = history_tracker.get_failed_calls()
        assert len(failed_calls) == 1
        assert failed_calls[0]["tool_name"] == "web_search"
    
    def test_mcp_tool_discovery(self):
        """测试MCP工具发现机制"""
        # 模拟MCP客户端返回的工具列表
        mock_tools = {
            "filesystem": [
                {"name": "read_file", "description": "读取文件内容"},
                {"name": "write_file", "description": "写入文件内容"},
                {"name": "list_directory", "description": "列出目录内容"}
            ],
            "database": [
                {"name": "query_db", "description": "执行数据库查询"},
                {"name": "update_db", "description": "更新数据库记录"}
            ],
            "system": [
                {"name": "run_command", "description": "执行系统命令"},
                {"name": "get_env", "description": "获取环境变量"}
            ]
        }
        
        tool_registry = ToolRegistry()
        tool_registry.register_tools(mock_tools)
        
        # 测试工具查找
        assert tool_registry.find_tool("read_file") == "filesystem"
        assert tool_registry.find_tool("query_db") == "database"
        assert tool_registry.find_tool("run_command") == "system"
        assert tool_registry.find_tool("non_existent") is None
        
        # 测试按服务器类型获取工具
        fs_tools = tool_registry.get_tools_by_server("filesystem")
        assert len(fs_tools) == 3
        assert any(tool["name"] == "read_file" for tool in fs_tools)
    
    def _standardize_tool_result(self, result):
        """模拟工具结果标准化逻辑"""
        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"], "success": False}
        elif isinstance(result, (str, int, float, bool, list, dict)):
            return result
        else:
            return str(result)
    
    def _handle_tool_error(self, scenario):
        """模拟工具错误处理逻辑"""
        error_suggestions = {
            "tool_not_found": "检查工具名称拼写或查看可用工具列表",
            "invalid_params": "检查工具参数格式和必需字段",
            "timeout": "稍后重试或检查网络连接",
            "permission_denied": "检查工具执行权限或联系管理员"
        }
        
        return {
            "error": f"工具调用失败: {scenario['tool_name']}",
            "error_type": scenario["error_type"],
            "suggestion": error_suggestions.get(scenario["error_type"], "请检查工具调用参数")
        }


class ToolCallHistoryTracker:
    """工具调用历史追踪器"""
    
    def __init__(self):
        self.history = []
    
    def add_call(self, tool_name, args, result):
        """添加工具调用记录"""
        call_record = {
            "tool_name": tool_name,
            "args": args,
            "result": result,
            "success": not (isinstance(result, dict) and "error" in result),
            "timestamp": __import__('time').time()
        }
        self.history.append(call_record)
    
    def get_history(self):
        """获取完整历史记录"""
        return self.history.copy()
    
    def get_successful_calls(self):
        """获取成功的调用记录"""
        return [call for call in self.history if call["success"]]
    
    def get_failed_calls(self):
        """获取失败的调用记录"""
        return [call for call in self.history if not call["success"]]


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools_by_server = {}
        self.tool_to_server = {}
    
    def register_tools(self, tools_dict):
        """注册工具"""
        self.tools_by_server = tools_dict.copy()
        
        # 构建工具名到服务器的映射
        for server_type, tools in tools_dict.items():
            for tool in tools:
                self.tool_to_server[tool["name"]] = server_type
    
    def find_tool(self, tool_name):
        """查找工具对应的服务器类型"""
        return self.tool_to_server.get(tool_name)
    
    def get_tools_by_server(self, server_type):
        """按服务器类型获取工具列表"""
        return self.tools_by_server.get(server_type, [])