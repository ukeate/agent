"""MCP异常处理类层次结构"""

from typing import Any, Dict, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now

logger = get_logger(__name__)

# 移除了对ApiError的依赖，MCPError现在直接继承自Exception

class MCPError(Exception):
    """MCP基础异常类"""
    
    def __init__(
        self,
        error_msg: str,  # 改名避免与日志系统冲突
        error_type: str = "MCPError",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        super().__init__(error_msg)
        self.error_message = error_msg
        self.error_type = error_type
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = utc_now().isoformat()
        
        # 记录异常（现在参数名问题已解决）
        logger.error(
            f"MCP Error: {error_type}",
            extra={
                "error_type": error_type,
                "error_message": error_msg,
                "details": details,
                "timestamp": self.timestamp
            }
        )
    
    def __str__(self) -> str:
        """返回异常的字符串表示"""
        return self.error_message

class MCPConnectionError(MCPError):
    """MCP连接异常"""
    
    def __init__(self, error_msg: str, server_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        enhanced_details = {"server_type": server_type} if server_type else {}
        if details:
            enhanced_details.update(details)
        
        super().__init__(
            error_msg=error_msg,
            error_type="MCPConnectionError",
            details=enhanced_details,
            status_code=503
        )

class MCPToolError(MCPError):
    """MCP工具执行异常"""
    
    def __init__(
        self, 
        error_msg: str, 
        tool_name: Optional[str] = None,
        server_type: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        enhanced_details = {
            "tool_name": tool_name,
            "server_type": server_type,
            "arguments": arguments
        }
        if details:
            enhanced_details.update(details)
        
        super().__init__(
            error_msg=error_msg,
            error_type="MCPToolError",
            details=enhanced_details,
            status_code=422
        )

class MCPSecurityError(MCPError):
    """MCP安全异常"""
    
    def __init__(
        self,
        error_msg: str,
        violation_type: str,
        attempted_action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        enhanced_details = {
            "violation_type": violation_type,
            "attempted_action": attempted_action
        }
        if details:
            enhanced_details.update(details)
        
        super().__init__(
            error_msg=error_msg,
            error_type="MCPSecurityError", 
            details=enhanced_details,
            status_code=403
        )
        
        # 安全违规需要特殊日志记录
        logger.warning(
            f"Security violation: {violation_type}",
            extra={
                "security_event": True,
                "violation_type": violation_type,
                "attempted_action": attempted_action,
                "error_message": error_msg,
                "details": enhanced_details,
                "timestamp": self.timestamp
            }
        )

class MCPValidationError(MCPError):
    """MCP参数验证异常"""
    
    def __init__(
        self,
        error_msg: str,
        tool_name: Optional[str] = None,
        invalid_parameters: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        enhanced_details = {
            "tool_name": tool_name,
            "invalid_parameters": invalid_parameters
        }
        if details:
            enhanced_details.update(details)
        
        super().__init__(
            error_msg=error_msg,
            error_type="MCPValidationError",
            details=enhanced_details,
            status_code=400
        )

class MCPTimeoutError(MCPError):
    """MCP操作超时异常"""
    
    def __init__(
        self,
        error_msg: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        enhanced_details = {
            "operation": operation,
            "timeout_seconds": timeout_seconds
        }
        if details:
            enhanced_details.update(details)
        
        super().__init__(
            error_msg=error_msg,
            error_type="MCPTimeoutError",
            details=enhanced_details,
            status_code=408
        )

class MCPResourceError(MCPError):
    """MCP资源异常"""
    
    def __init__(
        self,
        error_msg: str,
        resource_type: Optional[str] = None,
        resource_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        enhanced_details = {
            "resource_type": resource_type,
            "resource_path": resource_path
        }
        if details:
            enhanced_details.update(details)
        
        super().__init__(
            error_msg=error_msg,
            error_type="MCPResourceError",
            details=enhanced_details,
            status_code=404
        )

def handle_mcp_exception(func):
    """MCP异常处理装饰器"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except MCPError:
            # MCP异常直接重新抛出
            raise
        except Exception as e:
            # 将其他异常包装为MCP异常
            logger.error(f"Unexpected error in MCP operation: {str(e)}")
            raise MCPError(
                message=f"Unexpected error: {str(e)}",
                error_type="UnexpectedMCPError",
                details={"original_exception": type(e).__name__}
            )
    
    return wrapper

def create_mcp_error_response(error: Exception) -> Dict[str, Any]:
    """创建标准化的MCP错误响应"""
    if isinstance(error, MCPError):
        return {
            "success": False,
            "error": error.error_message,
            "error_type": error.error_type,
            "details": error.details,
            "timestamp": error.timestamp
        }
    else:
        # 处理非MCP异常
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "details": None,
            "timestamp": utc_now().isoformat()
        }
from src.core.logging import get_logger
