"""MCP客户端连接管理器"""

import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.core.config import get_settings
from .exceptions import MCPConnectionError, MCPToolError, MCPTimeoutError, handle_mcp_exception
from .retry import RetryManager, RetryConfig, get_retry_manager
from .monitoring import MonitoringContextManager, get_mcp_monitor

logger = get_logger(__name__)

class MCPClientManager:
    """MCP客户端连接管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.clients: Dict[str, ClientSession] = {}
        self.connection_pools: Dict[str, asyncio.Queue[ClientSession]] = {}
        self._initialized = False
        self.retry_manager = RetryManager(RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            max_delay=5.0
        ))
        self.monitor = get_mcp_monitor()
    
    async def initialize(self):
        """初始化MCP客户端管理器"""
        if self._initialized:
            return
        
        async with MonitoringContextManager(operation_name="mcp_manager_init"):
            try:
                # 创建默认MCP服务器连接池
                await self._create_connection_pools()
                self._initialized = True
                logger.info("MCP客户端管理器初始化成功")
                
            except Exception as e:
                logger.error(f"MCP客户端管理器初始化失败: {str(e)}")
                raise MCPConnectionError(f"Failed to initialize MCP client manager: {str(e)}")
    
    async def _create_connection_pools(self):
        """创建MCP服务器连接池"""
        # 配置文件系统MCP服务器
        filesystem_pool = asyncio.Queue(maxsize=5)
        for _ in range(3):  # 预创建3个连接
            try:
                client = await self._create_filesystem_client()
                await filesystem_pool.put(client)
            except Exception as e:
                logger.warning(f"创建文件系统MCP客户端失败: {str(e)}")
        
        self.connection_pools["filesystem"] = filesystem_pool
        
        # 配置数据库MCP服务器  
        database_pool = asyncio.Queue(maxsize=5)
        for _ in range(2):  # 预创建2个连接
            try:
                client = await self._create_database_client()
                await database_pool.put(client)
            except Exception as e:
                logger.warning(f"创建数据库MCP客户端失败: {str(e)}")
        
        self.connection_pools["database"] = database_pool
        
        # 配置系统命令MCP服务器
        system_pool = asyncio.Queue(maxsize=3)
        for _ in range(2):  # 预创建2个连接
            try:
                client = await self._create_system_client()
                await system_pool.put(client)
            except Exception as e:
                logger.warning(f"创建系统MCP客户端失败: {str(e)}")
        
        self.connection_pools["system"] = system_pool
    
    async def _create_filesystem_client(self) -> ClientSession:
        """创建文件系统MCP客户端"""
        from .tools.filesystem import call_filesystem_tool
        
        class FilesystemMCPClient:
            async def list_tools(self):
                return [
                    {"name": "read_file", "description": "读取文件内容"},
                    {"name": "write_file", "description": "写入文件内容"},
                    {"name": "list_directory", "description": "列出目录内容"},
                    {"name": "file_info", "description": "获取文件信息"}
                ]
                
            async def call_tool(self, name: str, arguments: Dict[str, Any]):
                return await call_filesystem_tool(name, arguments)
                
            async def close(self):
                return None
        
        return FilesystemMCPClient()
    
    async def _create_database_client(self) -> ClientSession:
        """创建数据库MCP客户端"""
        from .tools.database import call_database_tool
        
        class DatabaseMCPClient:
            async def list_tools(self):
                return [
                    {"name": "execute_query", "description": "执行SQL查询"},
                    {"name": "describe_tables", "description": "描述数据库表结构"},
                    {"name": "execute_transaction", "description": "执行事务"}
                ]
                
            async def call_tool(self, name: str, arguments: Dict[str, Any]):
                return await call_database_tool(name, arguments)
                
            async def close(self):
                return None
        
        return DatabaseMCPClient()
    
    async def _create_system_client(self) -> ClientSession:
        """创建系统命令MCP客户端"""
        from .tools.system import call_system_tool
        
        class SystemMCPClient:
            async def list_tools(self):
                return [
                    {"name": "run_command", "description": "执行系统命令"},
                    {"name": "check_process", "description": "检查进程状态"},
                    {"name": "get_env", "description": "获取环境变量"},
                    {"name": "get_system_info", "description": "获取系统信息"}
                ]
                
            async def call_tool(self, name: str, arguments: Dict[str, Any]):
                return await call_system_tool(name, arguments)
                
            async def close(self):
                return None
        
        return SystemMCPClient()
    
    @asynccontextmanager
    async def get_client(self, server_type: str) -> AsyncGenerator[ClientSession, None]:
        """获取MCP客户端连接"""
        if not self._initialized:
            await self.initialize()
        
        if server_type not in self.connection_pools:
            raise MCPConnectionError(f"Unknown server type: {server_type}")
        
        pool = self.connection_pools[server_type]
        client = None
        
        try:
            # 从连接池获取客户端
            client = await asyncio.wait_for(pool.get(), timeout=5.0)
            yield client
        except asyncio.TimeoutError:
            # 连接池为空，创建新连接
            if server_type == "filesystem":
                client = await self._create_filesystem_client()
            elif server_type == "database":
                client = await self._create_database_client()
            elif server_type == "system":
                client = await self._create_system_client()
            else:
                raise MCPConnectionError(f"Cannot create client for type: {server_type}")
            
            yield client
        finally:
            # 将客户端返回连接池
            if client is not None:
                try:
                    pool.put_nowait(client)
                except asyncio.QueueFull:
                    # 连接池已满，关闭连接
                    await client.close()
    
    async def list_available_tools(self, server_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """列出可用的MCP工具"""
        tools = {}
        
        if server_type:
            server_types = [server_type]
        else:
            server_types = list(self.connection_pools.keys())
        
        for stype in server_types:
            try:
                async with self.get_client(stype) as client:
                    tools[stype] = await client.list_tools()
            except Exception as e:
                logger.error(f"获取{stype}工具列表失败: {str(e)}")
                tools[stype] = []
        
        return tools
    
    @handle_mcp_exception
    async def call_tool(self, server_type: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP工具"""
        async with MonitoringContextManager(
            operation_name="mcp_tool_call",
            server_type=server_type,
            tool_name=tool_name
        ):
            try:
                # 使用重试机制调用工具
                result = await self.retry_manager.execute_with_retry(
                    self._execute_tool_call,
                    f"tool_call_{server_type}_{tool_name}",
                    server_type,
                    tool_name,
                    arguments
                )
                
                logger.info(
                    f"MCP工具调用成功: {server_type}.{tool_name}",
                    extra={
                        "server_type": server_type,
                        "tool_name": tool_name,
                        "arguments_keys": list(arguments.keys()) if arguments else []
                    }
                )
                return result
                
            except Exception as e:
                logger.error(
                    f"MCP工具调用失败: {server_type}.{tool_name} - {str(e)}",
                    extra={
                        "server_type": server_type,
                        "tool_name": tool_name,
                        "error_type": type(e).__name__
                    }
                )
                
                # 重新抛出MCP异常，或包装为MCP异常
                if isinstance(e, (MCPConnectionError, MCPToolError, MCPTimeoutError)):
                    raise
                else:
                    raise MCPToolError(
                        f"Tool execution failed: {str(e)}",
                        tool_name=tool_name,
                        server_type=server_type,
                        arguments=arguments
                    )
    
    async def _execute_tool_call(self, server_type: str, tool_name: str, arguments: Dict[str, Any]):
        """执行工具调用的内部方法"""
        async with self.get_client(server_type) as client:
            try:
                result = await client.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                # 包装为MCP工具异常
                raise MCPToolError(
                    f"Tool {tool_name} execution failed: {str(e)}",
                    tool_name=tool_name,
                    server_type=server_type,
                    arguments=arguments
                )
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "initialized": self._initialized,
            "servers": {},
            "overall_healthy": True
        }
        
        for server_type in self.connection_pools.keys():
            try:
                async with asyncio.timeout(5):  # 5秒超时
                    async with self.get_client(server_type) as client:
                        # 尝试列出工具作为健康检查
                        await client.list_tools()
                        health_status["servers"][server_type] = {
                            "healthy": True,
                            "error": None
                        }
            except Exception as e:
                health_status["servers"][server_type] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["overall_healthy"] = False
        
        return health_status
    
    async def get_retry_stats(self) -> Dict[str, Any]:
        """获取重试统计信息"""
        return self.retry_manager.get_retry_stats()
    
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        return self.monitor.get_metrics_summary()

    async def get_available_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有可用工具列表，按服务器类型分组"""
        all_tools = {}
        
        for server_type in self.connection_pools.keys():
            try:
                async with self.get_client(server_type) as client:
                    tools = await client.list_tools()
                    all_tools[server_type] = tools
            except Exception as e:
                logger.warning(f"获取{server_type}服务器工具列表失败: {str(e)}")
                all_tools[server_type] = []
        
        return all_tools

    async def close_all(self):
        """关闭所有连接"""
        async with MonitoringContextManager(operation_name="close_manager"):
            for server_type, pool in self.connection_pools.items():
                while not pool.empty():
                    try:
                        client = pool.get_nowait()
                        await client.close()
                    except asyncio.QueueEmpty:
                        break
                    except Exception as e:
                        logger.error(f"关闭{server_type}客户端失败: {str(e)}")
            
            self.connection_pools.clear()
            self._initialized = False
            logger.info("所有MCP连接已关闭")

# 全局MCP客户端管理器实例 - 延迟初始化
_mcp_client_manager: Optional[MCPClientManager] = None

async def get_mcp_client_manager() -> MCPClientManager:
    """获取MCP客户端管理器依赖注入"""
    global _mcp_client_manager
    if _mcp_client_manager is None:
        _mcp_client_manager = MCPClientManager()
    if not _mcp_client_manager._initialized:
        await _mcp_client_manager.initialize()
    return _mcp_client_manager
from src.core.logging import get_logger
