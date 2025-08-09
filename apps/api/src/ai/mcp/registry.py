"""MCP工具注册表和发现机制"""

import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass

from .client import MCPClientManager, get_mcp_client_manager

logger = logging.getLogger(__name__)


@dataclass
class MCPToolDefinition:
    """MCP工具定义"""
    name: str
    server_type: str
    description: str
    parameters: Dict[str, Any]
    security_level: str = "medium"  # low, medium, high
    timeout_seconds: int = 30


class MCPToolRegistry:
    """MCP工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, MCPToolDefinition] = {}
        self.tool_validators: Dict[str, Callable[[Dict[str, Any]], Awaitable[bool]]] = {}
        self._initialized = False
    
    async def initialize(self):
        """初始化工具注册表"""
        if self._initialized:
            return
            
        try:
            await self._register_default_tools()
            self._initialized = True
            logger.info("MCP工具注册表初始化成功")
            
        except Exception as e:
            logger.error(f"MCP工具注册表初始化失败: {str(e)}")
            raise
    
    async def _register_default_tools(self):
        """注册默认工具"""
        # 文件系统工具
        self.register_tool(MCPToolDefinition(
            name="read_file",
            server_type="filesystem", 
            description="读取文件内容",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "encoding": {"type": "string", "default": "utf-8", "description": "文件编码"}
                },
                "required": ["path"]
            },
            security_level="medium"
        ))
        
        self.register_tool(MCPToolDefinition(
            name="write_file",
            server_type="filesystem",
            description="写入文件内容", 
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"},
                    "encoding": {"type": "string", "default": "utf-8", "description": "文件编码"}
                },
                "required": ["path", "content"]
            },
            security_level="high"
        ))
        
        self.register_tool(MCPToolDefinition(
            name="list_directory",
            server_type="filesystem",
            description="列出目录内容",
            parameters={
                "type": "object", 
                "properties": {
                    "path": {"type": "string", "description": "目录路径"},
                    "include_hidden": {"type": "boolean", "default": False, "description": "是否包含隐藏文件"}
                },
                "required": ["path"]
            },
            security_level="low"
        ))
        
        self.register_tool(MCPToolDefinition(
            name="file_info",
            server_type="filesystem",
            description="获取文件信息",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"] 
            },
            security_level="low"
        ))
        
        # 数据库工具
        self.register_tool(MCPToolDefinition(
            name="execute_query",
            server_type="database",
            description="执行SQL查询",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL查询语句"},
                    "parameters": {"type": "array", "description": "查询参数"},
                    "read_only": {"type": "boolean", "default": True, "description": "是否只读查询"}
                },
                "required": ["query"]
            },
            security_level="high"
        ))
        
        self.register_tool(MCPToolDefinition(
            name="describe_tables",
            server_type="database", 
            description="描述数据库表结构",
            parameters={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "表名（可选）"},
                    "schema": {"type": "string", "description": "模式名（可选）"}
                }
            },
            security_level="low"
        ))
        
        # 系统命令工具
        self.register_tool(MCPToolDefinition(
            name="run_command",
            server_type="system",
            description="执行系统命令",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "命令字符串"},
                    "working_dir": {"type": "string", "description": "工作目录"},
                    "timeout": {"type": "integer", "default": 30, "description": "超时时间（秒）"}
                },
                "required": ["command"]
            },
            security_level="high"
        ))
        
        self.register_tool(MCPToolDefinition(
            name="check_process", 
            server_type="system",
            description="检查进程状态",
            parameters={
                "type": "object",
                "properties": {
                    "process_name": {"type": "string", "description": "进程名称"},
                    "pid": {"type": "integer", "description": "进程ID"}
                }
            },
            security_level="medium"
        ))
        
        self.register_tool(MCPToolDefinition(
            name="get_env",
            server_type="system", 
            description="获取环境变量",
            parameters={
                "type": "object",
                "properties": {
                    "var_name": {"type": "string", "description": "环境变量名称"}
                },
                "required": ["var_name"]
            },
            security_level="low"
        ))
        
        self.register_tool(MCPToolDefinition(
            name="get_system_info",
            server_type="system",
            description="获取系统信息",
            parameters={
                "type": "object",
                "properties": {}
            },
            security_level="low"
        ))
    
    def register_tool(self, tool_def: MCPToolDefinition):
        """注册MCP工具"""
        self.tools[tool_def.name] = tool_def
        logger.debug(f"注册MCP工具: {tool_def.name}")
    
    def register_validator(self, tool_name: str, validator: Callable[[Dict[str, Any]], Awaitable[bool]]):
        """注册工具参数验证器"""
        self.tool_validators[tool_name] = validator
        logger.debug(f"注册工具验证器: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[MCPToolDefinition]:
        """获取工具定义"""
        return self.tools.get(tool_name)
    
    def list_tools(self, server_type: Optional[str] = None) -> List[MCPToolDefinition]:
        """列出工具"""
        if server_type:
            return [tool for tool in self.tools.values() if tool.server_type == server_type]
        return list(self.tools.values())
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具参数架构"""
        tool = self.get_tool(tool_name)
        if not tool:
            return None
            
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "server_type": tool.server_type,
            "security_level": tool.security_level
        }
    
    async def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[bool, str]:
        """验证工具调用参数"""
        tool = self.get_tool(tool_name)
        if not tool:
            return False, f"Unknown tool: {tool_name}"
        
        # 基础参数验证
        if not await self._validate_parameters(tool, arguments):
            return False, "Invalid parameters"
        
        # 自定义验证器
        if tool_name in self.tool_validators:
            try:
                if not await self.tool_validators[tool_name](arguments):
                    return False, "Custom validation failed"
            except Exception as e:
                return False, f"Validation error: {str(e)}"
        
        return True, "Valid"
    
    async def _validate_parameters(self, tool: MCPToolDefinition, arguments: Dict[str, Any]) -> bool:
        """验证参数格式"""
        try:
            # 简化的参数验证
            required = tool.parameters.get("required", [])
            for param in required:
                if param not in arguments:
                    return False
            return True
        except Exception:
            return False
    
    async def discover_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """从MCP服务器发现工具"""
        if not self._initialized:
            await self.initialize()
            
        try:
            client_manager = await get_mcp_client_manager()
            return await client_manager.list_available_tools()
        except Exception as e:
            logger.error(f"工具发现失败: {str(e)}")
            return {}


# 全局工具注册表实例
mcp_tool_registry = MCPToolRegistry()


async def get_mcp_tool_registry() -> MCPToolRegistry:
    """获取MCP工具注册表依赖注入"""
    if not mcp_tool_registry._initialized:
        await mcp_tool_registry.initialize()
    return mcp_tool_registry