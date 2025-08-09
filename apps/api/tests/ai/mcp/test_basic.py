"""基础MCP功能测试"""

import pytest
from src.ai.mcp.client import MCPClientManager, get_mcp_client_manager
from src.ai.mcp.registry import MCPToolRegistry, get_mcp_tool_registry


@pytest.mark.asyncio
async def test_mcp_client_manager_initialization():
    """测试MCP客户端管理器初始化"""
    manager = MCPClientManager()
    await manager.initialize()
    
    assert manager._initialized
    assert "filesystem" in manager.connection_pools
    assert "database" in manager.connection_pools
    assert "system" in manager.connection_pools
    
    await manager.close_all()


@pytest.mark.asyncio 
async def test_mcp_tool_registry_initialization():
    """测试MCP工具注册表初始化"""
    registry = MCPToolRegistry()
    await registry.initialize()
    
    assert registry._initialized
    assert len(registry.tools) > 0
    
    # 检查默认工具是否注册
    assert "read_file" in registry.tools
    assert "write_file" in registry.tools
    assert "execute_query" in registry.tools
    assert "run_command" in registry.tools


@pytest.mark.asyncio
async def test_dependency_injection():
    """测试依赖注入功能"""
    manager = await get_mcp_client_manager()
    assert isinstance(manager, MCPClientManager)
    assert manager._initialized
    
    registry = await get_mcp_tool_registry()
    assert isinstance(registry, MCPToolRegistry)
    assert registry._initialized
    
    await manager.close_all()


@pytest.mark.asyncio
async def test_basic_tool_call():
    """测试基础工具调用"""
    import tempfile
    import os
    
    manager = await get_mcp_client_manager()
    
    # 创建临时文件进行测试
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content")
        temp_path = f.name
    
    try:
        result = await manager.call_tool(
            "filesystem",
            "read_file", 
            {"path": temp_path}
        )
        
        assert result["success"] is True
        assert result["content"] == "Test content"
    finally:
        os.unlink(temp_path)
        await manager.close_all()


@pytest.mark.asyncio
async def test_list_available_tools():
    """测试列出可用工具"""
    manager = await get_mcp_client_manager()
    
    tools = await manager.list_available_tools()
    
    assert "filesystem" in tools
    assert "database" in tools
    assert "system" in tools
    assert len(tools["filesystem"]) > 0
    
    await manager.close_all()


@pytest.mark.asyncio
async def test_tool_schema():
    """测试工具架构获取"""
    registry = await get_mcp_tool_registry()
    
    schema = registry.get_tool_schema("read_file")
    assert schema is not None
    assert schema["name"] == "read_file"
    assert schema["server_type"] == "filesystem"
    assert "parameters" in schema