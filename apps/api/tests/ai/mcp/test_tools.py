"""测试MCP工具实现"""

import pytest
import os
import tempfile
from pathlib import Path

from src.ai.mcp.tools.filesystem import call_filesystem_tool
from src.ai.mcp.tools.system import call_system_tool
from src.ai.mcp.exceptions import MCPSecurityError
from src.ai.mcp.client import get_mcp_client_manager


@pytest.mark.asyncio
class TestFilesystemTools:
    """测试文件系统工具"""
    
    async def test_read_file_success(self):
        """测试读取文件成功"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Hello, World!")
            temp_path = f.name
        
        try:
            result = await call_filesystem_tool("read_file", {"path": temp_path})
            assert result["success"] is True
            assert result["content"] == "Hello, World!"
            assert result["size"] > 0
        finally:
            os.unlink(temp_path)
    
    async def test_read_file_not_found(self):
        """测试读取不存在的文件"""
        result = await call_filesystem_tool("read_file", {"path": "/tmp/nonexistent.txt"})
        assert result["success"] is False
        assert result["error_type"] == "FileNotFound"
    
    async def test_write_file_success(self):
        """测试写入文件成功"""
        temp_path = "/tmp/test_write.txt"
        
        try:
            result = await call_filesystem_tool("write_file", {
                "path": temp_path,
                "content": "Test content"
            })
            assert result["success"] is True
            assert os.path.exists(temp_path)
            
            # 验证内容
            with open(temp_path, 'r') as f:
                content = f.read()
            assert content == "Test content"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def test_list_directory_success(self):
        """测试列出目录成功"""
        result = await call_filesystem_tool("list_directory", {"path": "/tmp"})
        assert result["success"] is True
        assert "entries" in result
        assert isinstance(result["entries"], list)
    
    async def test_file_info_success(self):
        """测试获取文件信息成功"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            result = await call_filesystem_tool("file_info", {"path": temp_path})
            assert result["success"] is True
            assert result["type"] == "file"
            assert "size" in result
            assert "modified_time" in result
        finally:
            os.unlink(temp_path)
    
    async def test_security_validation(self):
        """测试安全验证"""
        # 尝试访问受限路径
        result = await call_filesystem_tool("read_file", {"path": "/etc/passwd"})
        assert result["success"] is False
        assert result["error_type"] == "SecurityError"


@pytest.mark.asyncio
class TestSystemTools:
    """测试系统工具"""
    
    async def test_run_command_success(self):
        """测试执行命令成功"""
        result = await call_system_tool("run_command", {"command": "echo hello"})
        assert result["success"] is True
        assert "hello" in result["stdout"]
        assert result["return_code"] == 0
    
    async def test_run_command_security(self):
        """测试命令安全性检查"""
        result = await call_system_tool("run_command", {"command": "rm -rf /"})
        assert result["success"] is False
        assert result["error_type"] == "SecurityError"
    
    async def test_check_process_by_name(self):
        """测试按名称检查进程"""
        result = await call_system_tool("check_process", {"process_name": "python"})
        assert result["success"] is True
        assert "processes" in result
        assert isinstance(result["processes"], list)
    
    async def test_get_env_success(self):
        """测试获取环境变量成功"""
        result = await call_system_tool("get_env", {"var_name": "HOME"})
        assert result["success"] is True
        assert "value" in result
    
    async def test_get_env_security(self):
        """测试环境变量安全性检查"""
        result = await call_system_tool("get_env", {"var_name": "DANGEROUS_VAR"})
        assert result["success"] is False
        assert result["error_type"] == "SecurityError"
    
    async def test_get_system_info(self):
        """测试获取系统信息"""
        result = await call_system_tool("get_system_info", {})
        assert result["success"] is True
        assert "system" in result
        assert "resources" in result


@pytest.mark.asyncio
class TestMCPIntegration:
    """测试MCP集成"""
    
    async def test_filesystem_integration(self):
        """测试文件系统工具集成"""
        manager = await get_mcp_client_manager()
        
        # 创建临时文件用于测试
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Integration test")
            temp_path = f.name
        
        try:
            # 通过MCP客户端调用工具
            result = await manager.call_tool("filesystem", "read_file", {"path": temp_path})
            assert result["success"] is True
            assert result["content"] == "Integration test"
        finally:
            os.unlink(temp_path)
            await manager.close_all()
    
    async def test_system_integration(self):
        """测试系统工具集成"""
        manager = await get_mcp_client_manager()
        
        try:
            # 通过MCP客户端调用工具
            result = await manager.call_tool("system", "run_command", {"command": "pwd"})
            assert result["success"] is True
            assert "stdout" in result
        finally:
            await manager.close_all()
    
    async def test_tool_discovery(self):
        """测试工具发现"""
        manager = await get_mcp_client_manager()
        
        try:
            tools = await manager.list_available_tools()
            
            # 验证文件系统工具
            assert "filesystem" in tools
            fs_tools = tools["filesystem"]
            tool_names = [tool["name"] for tool in fs_tools]
            assert "read_file" in tool_names
            assert "write_file" in tool_names
            assert "list_directory" in tool_names
            assert "file_info" in tool_names
            
            # 验证系统工具
            assert "system" in tools
            sys_tools = tools["system"]
            sys_tool_names = [tool["name"] for tool in sys_tools]
            assert "run_command" in sys_tool_names
            assert "check_process" in sys_tool_names
            assert "get_env" in sys_tool_names
            assert "get_system_info" in sys_tool_names
            
        finally:
            await manager.close_all()


@pytest.mark.asyncio 
class TestErrorHandling:
    """测试错误处理"""
    
    async def test_invalid_tool_name(self):
        """测试无效工具名称"""
        result = await call_filesystem_tool("invalid_tool", {})
        assert result["success"] is False
        assert result["error_type"] == "UnknownTool"
    
    async def test_missing_parameters(self):
        """测试缺失参数"""
        result = await call_filesystem_tool("read_file", {})
        assert result["success"] is False
        # 应该有适当的错误处理
    
    async def test_invalid_parameters(self):
        """测试无效参数"""
        result = await call_system_tool("run_command", {"command": ""})
        assert result["success"] is False
        assert result["error_type"] == "SecurityError"