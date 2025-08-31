"""
工作流错误处理和重试策略
"""
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from enum import Enum
import asyncio
import traceback
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    RetryError
)

from .state import MessagesState


class ErrorType(Enum):
    """错误类型枚举"""
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WorkflowError:
    """工作流错误信息"""
    id: str = field(default_factory=lambda: str(utc_now().timestamp()))
    error_type: ErrorType = ErrorType.UNKNOWN_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    message: str = ""
    details: str = ""
    node_name: Optional[str] = None
    workflow_id: Optional[str] = None
    timestamp: datetime = field(default_factory=utc_factory)
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    is_recoverable: bool = True


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_handlers: Dict[ErrorType, Callable] = {}
        self.global_error_callbacks: List[Callable] = []
    
    def register_handler(self, error_type: ErrorType, handler: Callable):
        """注册特定错误类型的处理器"""
        self.error_handlers[error_type] = handler
    
    def add_global_callback(self, callback: Callable):
        """添加全局错误回调"""
        self.global_error_callbacks.append(callback)
    
    async def handle_error(self, error: Exception, state: MessagesState, context: Optional[Dict[str, Any]] = None) -> WorkflowError:
        """处理错误"""
        workflow_error = self._classify_error(error, state, context)
        
        # 记录错误到状态
        if "errors" not in state["context"]:
            state["context"]["errors"] = []
        state["context"]["errors"].append({
            "id": workflow_error.id,
            "type": workflow_error.error_type.value,
            "severity": workflow_error.severity.value,
            "message": workflow_error.message,
            "timestamp": workflow_error.timestamp.isoformat(),
            "node_name": workflow_error.node_name,
            "retry_count": workflow_error.retry_count
        })
        
        # 更新状态为错误
        state["metadata"]["status"] = "error"
        state["metadata"]["last_error"] = workflow_error.message
        state["metadata"]["error_timestamp"] = workflow_error.timestamp.isoformat()
        
        # 执行特定错误处理器
        if workflow_error.error_type in self.error_handlers:
            try:
                await self.error_handlers[workflow_error.error_type](workflow_error, state)
            except Exception as handler_error:
                print(f"错误处理器执行失败: {handler_error}")
        
        # 执行全局回调
        for callback in self.global_error_callbacks:
            try:
                await callback(workflow_error, state)
            except Exception as callback_error:
                print(f"全局错误回调执行失败: {callback_error}")
        
        return workflow_error
    
    def _classify_error(self, error: Exception, state: MessagesState, context: Optional[Dict[str, Any]] = None) -> WorkflowError:
        """分类错误"""
        error_type = ErrorType.UNKNOWN_ERROR
        severity = ErrorSeverity.MEDIUM
        is_recoverable = True
        
        # 根据异常类型分类
        if isinstance(error, asyncio.TimeoutError):
            error_type = ErrorType.TIMEOUT_ERROR
            severity = ErrorSeverity.HIGH
        elif isinstance(error, ValueError) or isinstance(error, TypeError):
            error_type = ErrorType.VALIDATION_ERROR
            severity = ErrorSeverity.LOW
            is_recoverable = False
        elif isinstance(error, ConnectionError) or "network" in str(error).lower():
            error_type = ErrorType.NETWORK_ERROR
            severity = ErrorSeverity.MEDIUM
        elif isinstance(error, SystemError) or isinstance(error, RuntimeError):
            error_type = ErrorType.SYSTEM_ERROR
            severity = ErrorSeverity.HIGH
        
        # 根据错误消息进一步分类
        error_msg = str(error).lower()
        if "business" in error_msg or "logic" in error_msg:
            error_type = ErrorType.BUSINESS_LOGIC_ERROR
            severity = ErrorSeverity.MEDIUM
        
        return WorkflowError(
            error_type=error_type,
            severity=severity,
            message=str(error),
            details=traceback.format_exc(),
            node_name=context.get("node_name") if context else None,
            workflow_id=state.get("workflow_id"),
            stack_trace=traceback.format_exc(),
            context=context or {},
            is_recoverable=is_recoverable
        )


class RetryStrategy:
    """重试策略"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def should_retry(self, error: WorkflowError) -> bool:
        """判断是否应该重试"""
        if not error.is_recoverable:
            return False
        
        if error.retry_count >= error.max_retries:
            return False
        
        # 特定错误类型的重试策略
        if error.error_type == ErrorType.VALIDATION_ERROR:
            return False
        
        if error.error_type == ErrorType.BUSINESS_LOGIC_ERROR:
            return False
        
        if error.severity == ErrorSeverity.CRITICAL:
            return error.retry_count < 1  # 严重错误只重试一次
        
        return True
    
    def get_retry_delay(self, retry_count: int) -> float:
        """获取重试延迟时间（指数退避）"""
        delay = self.base_delay * (2 ** retry_count)
        return min(delay, self.max_delay)


class WorkflowErrorRecovery:
    """工作流错误恢复"""
    
    def __init__(self, error_handler: ErrorHandler = None, retry_strategy: RetryStrategy = None):
        self.error_handler = error_handler or ErrorHandler()
        self.retry_strategy = retry_strategy or RetryStrategy()
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """设置默认错误处理器"""
        
        async def handle_network_error(error: WorkflowError, state: MessagesState):
            """处理网络错误"""
            print(f"处理网络错误: {error.message}")
            state["context"]["recovery_action"] = "network_retry"
        
        async def handle_timeout_error(error: WorkflowError, state: MessagesState):
            """处理超时错误"""
            print(f"处理超时错误: {error.message}")
            state["context"]["recovery_action"] = "timeout_recovery"
            # 可以选择降级处理或增加超时时间
        
        async def handle_validation_error(error: WorkflowError, state: MessagesState):
            """处理验证错误"""
            print(f"处理验证错误: {error.message}")
            state["context"]["recovery_action"] = "validation_failed"
            state["metadata"]["status"] = "failed"  # 验证错误通常不可恢复
        
        self.error_handler.register_handler(ErrorType.NETWORK_ERROR, handle_network_error)
        self.error_handler.register_handler(ErrorType.TIMEOUT_ERROR, handle_timeout_error)
        self.error_handler.register_handler(ErrorType.VALIDATION_ERROR, handle_validation_error)
    
    async def execute_with_recovery(self, func: Callable, state: MessagesState, context: Optional[Dict[str, Any]] = None) -> Any:
        """带错误恢复的执行函数"""
        last_error = None
        
        for attempt in range(self.retry_strategy.max_attempts):
            try:
                # 更新重试次数
                if "retry_info" not in state["context"]:
                    state["context"]["retry_info"] = {}
                state["context"]["retry_info"]["current_attempt"] = attempt + 1
                state["context"]["retry_info"]["max_attempts"] = self.retry_strategy.max_attempts
                
                # 执行函数
                result = await func(state) if asyncio.iscoroutinefunction(func) else func(state)
                
                # 成功执行，清除错误信息
                if "retry_info" in state["context"]:
                    state["context"]["retry_info"]["success"] = True
                
                return result
                
            except Exception as e:
                last_error = await self.error_handler.handle_error(e, state, context)
                last_error.retry_count = attempt
                
                # 判断是否应该重试
                if not self.retry_strategy.should_retry(last_error):
                    break
                
                # 如果不是最后一次尝试，等待重试延迟
                if attempt < self.retry_strategy.max_attempts - 1:
                    delay = self.retry_strategy.get_retry_delay(attempt)
                    print(f"第{attempt + 1}次尝试失败，{delay}秒后重试: {last_error.message}")
                    await asyncio.sleep(delay)
        
        # 所有重试都失败了
        state["metadata"]["status"] = "failed"
        state["context"]["final_error"] = {
            "message": last_error.message if last_error else "未知错误",
            "type": last_error.error_type.value if last_error else "unknown",
            "total_attempts": self.retry_strategy.max_attempts
        }
        
        if last_error:
            raise RuntimeError(f"工作流执行失败，已重试{self.retry_strategy.max_attempts}次: {last_error.message}")
        else:
            raise RuntimeError("工作流执行失败，原因未知")


# 装饰器形式的重试机制
def workflow_retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """工作流重试装饰器"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=delay, max=60),
        retry=retry_if_exception_type(exceptions),
        reraise=True
    )


# 全局错误恢复实例
error_recovery = WorkflowErrorRecovery()