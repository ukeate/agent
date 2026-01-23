"""MCP集成测试"""

import tempfile
import os
import pytest
from fastapi.testclient import TestClient


def test_mcp_health_check(client):
    """测试MCP健康检查API"""
    response = client.get("/api/v1/mcp/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "initialized" in data
    assert "overall_healthy" in data
    assert "servers" in data
    assert data["initialized"] is True


def test_list_available_tools(client):
    """测试列出可用工具API"""
    response = client.get("/api/v1/mcp/tools")
        
    assert response.status_code == 200
    data = response.json()
    
    assert "tools" in data
    tools = data["tools"]
    
    # 检查默认服务器类型
    assert "filesystem" in tools
    assert "database" in tools
    assert "system" in tools
    
    # 检查工具是否存在
    assert len(tools["filesystem"]) > 0
    assert len(tools["database"]) > 0
    assert len(tools["system"]) > 0


def test_filesystem_tool_call_api(client):
    """测试文件系统工具调用API"""
    # 创建临时文件进行测试
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content for API")
        temp_path = f.name
    
    try:
        # 测试通用工具调用接口
        response = client.post(
            "/api/v1/mcp/tools/call",
            json={
                "server_type": "filesystem",
                "tool_name": "read_file",
                "arguments": {"path": temp_path}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["tool_name"] == "read_file"
        assert data["server_type"] == "filesystem"
        assert "result" in data
        
    finally:
        os.unlink(temp_path)


def test_filesystem_write_and_read(client):
    """测试文件系统写入和读取API"""
    # 使用临时文件路径
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_path = temp_file.name
    
    try:
        # 写入文件
        write_response = client.post(
            "/api/v1/mcp/tools/filesystem/write",
            json={"path": temp_path, "content": "Hello from API test"}
        )
        
        assert write_response.status_code == 200
        write_result = write_response.json()
        assert write_result["success"] is True
        
        # 读取文件
        read_response = client.post(
            "/api/v1/mcp/tools/filesystem/read",
            json={"path": temp_path}
        )
        
        assert read_response.status_code == 200
        read_result = read_response.json()
        assert read_result["content"] == "Hello from API test"
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_security_validation(client):
    """测试安全验证"""
    # 尝试访问受限路径
    response = client.post(
        "/api/v1/mcp/tools/call",
        json={
            "server_type": "filesystem",
            "tool_name": "read_file",
            "arguments": {"path": "/etc/passwd"}
        }
    )
    
    assert response.status_code == 200  # API调用成功
    data = response.json()
    
    # 检查结果中是否包含安全错误
    result = data.get("result", {})
    if isinstance(result, dict):
        # 工具执行应该失败
        assert result.get("success") is False
        assert "SecurityError" in result.get("error_type", "") or "Access denied" in result.get("error", "")
    else:
        # 或者外层调用失败
        assert data["success"] is False


def test_system_command_api(client):
    """测试系统命令API"""
    # 测试安全的命令
    response = client.post(
        "/api/v1/mcp/tools/system/command",
        json={"command": "echo 'Hello World'"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "Hello World" in data["output"]


def test_database_query_api(client):
    """测试数据库查询API"""
    # 测试简单查询
    response = client.post(
        "/api/v1/mcp/tools/database/query",
        json={"query": "SELECT 1 as test_value"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "rows" in data


def test_metrics_api(client):
    """测试指标API"""
    # 先执行一些操作生成指标
    client.get("/api/v1/mcp/health")
    client.get("/api/v1/mcp/tools")
    
    # 获取指标
    response = client.get("/api/v1/mcp/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "monitoring_stats" in data
    assert "retry_stats" in data


def test_error_handling(client):
    """测试错误处理"""
    # 测试不存在的服务器类型
    response = client.post(
        "/api/v1/mcp/tools/call",
        json={
            "server_type": "nonexistent",
            "tool_name": "some_tool",
            "arguments": {}
        }
    )
    
    assert response.status_code == 200  # API调用成功
    data = response.json()
    assert data["success"] is False
    assert "error" in data
