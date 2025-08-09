"""MCP API路由"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...ai.mcp.client import get_mcp_client_manager, MCPClientManager
from ...ai.mcp.exceptions import MCPError, create_mcp_error_response
from ...ai.mcp.monitoring import get_monitor_dependency, MCPMonitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP"])


# Pydantic模型定义
class ToolCallRequest(BaseModel):
    """工具调用请求"""
    server_type: str = Field(..., description="MCP服务器类型 (filesystem, database, system)")
    tool_name: str = Field(..., description="工具名称")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="工具参数")


class ToolCallResponse(BaseModel):
    """工具调用响应"""
    success: bool = Field(..., description="调用是否成功")
    result: Optional[Any] = Field(None, description="调用结果")
    error: Optional[str] = Field(None, description="错误信息")
    error_type: Optional[str] = Field(None, description="错误类型")
    tool_name: str = Field(..., description="工具名称")
    server_type: str = Field(..., description="服务器类型")


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    initialized: bool = Field(..., description="是否初始化")
    overall_healthy: bool = Field(..., description="整体健康状态")
    servers: Dict[str, Dict[str, Any]] = Field(..., description="各服务器状态")


class AvailableToolsResponse(BaseModel):
    """可用工具响应"""
    tools: Dict[str, List[Dict[str, Any]]] = Field(..., description="按服务器类型分组的工具列表")


class MetricsResponse(BaseModel):
    """指标响应"""
    monitoring_stats: Dict[str, Any] = Field(..., description="监控统计")
    retry_stats: Dict[str, Any] = Field(..., description="重试统计")


@router.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(
    request: ToolCallRequest,
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager)
) -> ToolCallResponse:
    """调用MCP工具
    
    支持的服务器类型:
    - filesystem: 文件系统操作 (read_file, write_file, list_directory, file_info)
    - database: 数据库操作 (execute_query, describe_tables, execute_transaction)
    - system: 系统操作 (run_command, check_process, get_env, get_system_info)
    """
    try:
        logger.info(
            f"调用MCP工具: {request.server_type}.{request.tool_name}",
            extra={
                "server_type": request.server_type,
                "tool_name": request.tool_name,
                "arguments_keys": list(request.arguments.keys())
            }
        )
        
        result = await mcp_manager.call_tool(
            server_type=request.server_type,
            tool_name=request.tool_name,
            arguments=request.arguments
        )
        
        return ToolCallResponse(
            success=True,
            result=result,
            tool_name=request.tool_name,
            server_type=request.server_type
        )
        
    except MCPError as e:
        logger.error(f"MCP工具调用失败: {str(e)}")
        error_response = create_mcp_error_response(e)
        
        return ToolCallResponse(
            success=False,
            error=error_response["error"],
            error_type=error_response["error_type"],
            tool_name=request.tool_name,
            server_type=request.server_type
        )
        
    except Exception as e:
        logger.error(f"工具调用发生未知错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/tools", response_model=AvailableToolsResponse)
async def list_available_tools(
    server_type: Optional[str] = None,
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager)
) -> AvailableToolsResponse:
    """列出可用的MCP工具
    
    Args:
        server_type: 可选的服务器类型过滤器
    """
    try:
        tools = await mcp_manager.list_available_tools(server_type)
        
        logger.info(
            f"列出可用工具成功",
            extra={
                "server_type_filter": server_type,
                "tools_count": {k: len(v) for k, v in tools.items()}
            }
        )
        
        return AvailableToolsResponse(tools=tools)
        
    except Exception as e:
        logger.error(f"列出工具失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tools: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager)
) -> HealthCheckResponse:
    """MCP系统健康检查"""
    try:
        health_status = await mcp_manager.health_check()
        
        return HealthCheckResponse(
            initialized=health_status["initialized"],
            overall_healthy=health_status["overall_healthy"],
            servers=health_status["servers"]
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager),
    monitor: MCPMonitor = Depends(get_monitor_dependency)
) -> MetricsResponse:
    """获取MCP系统指标"""
    try:
        monitoring_stats = await mcp_manager.get_monitoring_stats()
        retry_stats = await mcp_manager.get_retry_stats()
        
        return MetricsResponse(
            monitoring_stats=monitoring_stats,
            retry_stats=retry_stats
        )
        
    except Exception as e:
        logger.error(f"获取指标失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.post("/tools/filesystem/read")
async def read_file(
    path: str,
    encoding: str = "utf-8",
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager)
) -> Dict[str, Any]:
    """读取文件 - 文件系统工具的便捷接口"""
    request = ToolCallRequest(
        server_type="filesystem",
        tool_name="read_file",
        arguments={"path": path, "encoding": encoding}
    )
    response = await call_tool(request, mcp_manager)
    
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.error
        )
    
    return response.result


@router.post("/tools/filesystem/write")
async def write_file(
    path: str,
    content: str,
    encoding: str = "utf-8",
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager)
) -> Dict[str, Any]:
    """写入文件 - 文件系统工具的便捷接口"""
    request = ToolCallRequest(
        server_type="filesystem",
        tool_name="write_file",
        arguments={"path": path, "content": content, "encoding": encoding}
    )
    response = await call_tool(request, mcp_manager)
    
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.error
        )
    
    return response.result


@router.get("/tools/filesystem/list")
async def list_directory(
    path: str,
    include_hidden: bool = False,
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager)
) -> Dict[str, Any]:
    """列出目录 - 文件系统工具的便捷接口"""
    request = ToolCallRequest(
        server_type="filesystem",
        tool_name="list_directory",
        arguments={"path": path, "include_hidden": include_hidden}
    )
    response = await call_tool(request, mcp_manager)
    
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.error
        )
    
    return response.result


@router.post("/tools/database/query")
async def execute_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager)
) -> Dict[str, Any]:
    """执行数据库查询 - 数据库工具的便捷接口"""
    arguments = {"query": query}
    if parameters:
        arguments["parameters"] = parameters
    
    request = ToolCallRequest(
        server_type="database",
        tool_name="execute_query",
        arguments=arguments
    )
    response = await call_tool(request, mcp_manager)
    
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.error
        )
    
    # 为便捷接口提供更用户友好的响应格式
    result = response.result.copy()
    if "data" in result:
        result["rows"] = result["data"]
    
    return result


@router.post("/tools/system/command")
async def run_command(
    command: str,
    timeout: Optional[int] = None,
    mcp_manager: MCPClientManager = Depends(get_mcp_client_manager)
) -> Dict[str, Any]:
    """执行系统命令 - 系统工具的便捷接口"""
    arguments = {"command": command}
    if timeout:
        arguments["timeout"] = timeout
    
    request = ToolCallRequest(
        server_type="system",
        tool_name="run_command",
        arguments=arguments
    )
    response = await call_tool(request, mcp_manager)
    
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.error
        )
    
    # 为便捷接口提供更用户友好的响应格式
    result = response.result.copy()
    if "stdout" in result:
        result["output"] = result["stdout"]
    
    return result