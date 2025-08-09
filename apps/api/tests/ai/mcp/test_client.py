"""测试MCP客户端连接管理器"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, patch

from ai.mcp.client import MCPClientManager, MCPConnectionError, get_mcp_client_manager


@pytest.mark.asyncio
class TestMCPClientManager:
    """测试MCP客户端管理器"""
    
    @pytest_asyncio.fixture
    async def client_manager(self):
        """创建测试用的客户端管理器"""
        manager = MCPClientManager()
        await manager.initialize()
        yield manager
        await manager.close_all()
    
    async def test_initialization(self, client_manager):
        """测试客户端管理器初始化"""
        assert client_manager._initialized
        assert "filesystem" in client_manager.connection_pools
        assert "database" in client_manager.connection_pools  
        assert "system" in client_manager.connection_pools
    
    async def test_get_client_filesystem(self, client_manager):
        """测试获取文件系统客户端"""
        async with client_manager.get_client("filesystem") as client:
            assert client is not None
            tools = await client.list_tools()
            assert len(tools) > 0
            assert any(tool["name"] == "read_file" for tool in tools)
    
    async def test_get_client_database(self, client_manager):
        """测试获取数据库客户端"""
        async with client_manager.get_client("database") as client:
            assert client is not None
            tools = await client.list_tools()
            assert len(tools) > 0
            assert any(tool["name"] == "execute_query" for tool in tools)
    
    async def test_get_client_system(self, client_manager):
        """测试获取系统客户端"""
        async with client_manager.get_client("system") as client:
            assert client is not None
            tools = await client.list_tools()
            assert len(tools) > 0
            assert any(tool["name"] == "run_command" for tool in tools)
    
    async def test_get_client_unknown_type(self, client_manager):
        """测试获取未知类型客户端"""
        with pytest.raises(MCPConnectionError):
            async with client_manager.get_client("unknown"):
                pass
    
    async def test_list_available_tools(self, client_manager):
        """测试列出可用工具"""
        tools = await client_manager.list_available_tools()
        assert "filesystem" in tools
        assert "database" in tools
        assert "system" in tools
        
        # 测试指定服务器类型
        fs_tools = await client_manager.list_available_tools("filesystem")
        assert "filesystem" in fs_tools
        assert len(fs_tools["filesystem"]) > 0
    
    async def test_call_tool(self, client_manager):
        """测试工具调用"""
        result = await client_manager.call_tool(
            "filesystem", 
            "read_file", 
            {"path": "/tmp/test_file.txt"}
        )
        assert result["success"] is False  # 文件不存在，应该失败
        assert "error" in result
    
    async def test_call_tool_error_handling(self, client_manager):
        """测试工具调用错误处理"""
        # 测试不存在的服务器类型
        with pytest.raises(MCPConnectionError):
            await client_manager.call_tool("nonexistent", "some_tool", {"param": "value"})
    
    async def test_close_all(self, client_manager):
        """测试关闭所有连接"""
        # 确保有连接存在
        async with client_manager.get_client("filesystem"):
            pass
        
        # 注意：由于fixture会自动清理，我们不能测试close_all的效果
        # 但可以验证close_all方法可以被调用
        await client_manager.close_all()


@pytest.mark.asyncio
class TestMCPClientManagerDependency:
    """测试MCP客户端管理器依赖注入"""
    
    async def test_get_mcp_client_manager(self):
        """测试获取客户端管理器"""
        manager = await get_mcp_client_manager()
        assert isinstance(manager, MCPClientManager)
        assert manager._initialized
        
        # 再次调用应该返回同一个实例
        manager2 = await get_mcp_client_manager()
        assert manager is manager2
        
        # 清理
        await manager.close_all()


@pytest.mark.asyncio
class TestMCPConnectionPool:
    """测试MCP连接池功能"""
    
    async def test_connection_pool_concurrent_access(self):
        """测试连接池并发访问"""
        manager = MCPClientManager()
        await manager.initialize()
        
        try:
            # 并发获取多个客户端
            async def get_client_and_call():
                async with manager.get_client("filesystem") as client:
                    await asyncio.sleep(0.1)  # 模拟工作时间
                    return await client.list_tools()
            
            # 并发执行5个任务
            tasks = [get_client_and_call() for _ in range(5)]
            results = await asyncio.gather(*tasks)
            
            # 验证所有任务都成功完成
            assert len(results) == 5
            for result in results:
                assert len(result) > 0
                
        finally:
            await manager.close_all()
    
    async def test_connection_pool_timeout(self):
        """测试连接池超时"""
        manager = MCPClientManager()
        await manager.initialize()
        
        try:
            # 占用所有连接
            contexts = []
            for _ in range(6):  # 超过连接池大小
                ctx = manager.get_client("filesystem")
                contexts.append(ctx)
                await ctx.__aenter__()
            
            # 此时再获取连接应该能成功（创建新连接）
            async with manager.get_client("filesystem") as client:
                assert client is not None
                
        finally:
            # 清理上下文
            for ctx in contexts:
                try:
                    await ctx.__aexit__(None, None, None)
                except:
                    pass
            await manager.close_all()