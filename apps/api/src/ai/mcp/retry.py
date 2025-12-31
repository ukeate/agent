"""MCP工具调用重试机制"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
import random
from .exceptions import MCPConnectionError, MCPTimeoutError, MCPToolError

logger = get_logger(__name__)

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0  # 基础延迟时间（秒）
    max_delay: float = 60.0  # 最大延迟时间（秒）
    exponential_base: float = 2.0  # 指数退避基数
    jitter: bool = True  # 是否添加随机抖动
    retryable_exceptions: List[type] = None
    
    def __post_init__(self):
        if self.retryable_exceptions is None:
            self.retryable_exceptions = [
                MCPConnectionError,
                MCPTimeoutError,
                asyncio.TimeoutError,
                ConnectionError
            ]

class RetryManager:
    """重试管理器"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.retry_stats: Dict[str, Dict[str, Any]] = {}
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算延迟时间"""
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # 添加±25%的随机抖动
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(delay, 0.1)  # 最小延迟0.1秒
    
    def _is_retryable(self, exception: Exception) -> bool:
        """判断异常是否可重试"""
        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)
    
    def _record_retry_stats(self, operation_key: str, attempt: int, success: bool, error: Optional[Exception] = None):
        """记录重试统计信息"""
        if operation_key not in self.retry_stats:
            self.retry_stats[operation_key] = {
                "total_attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": 0,
                "retry_attempts": 0,
                "last_error": None,
                "last_success": None
            }
        
        stats = self.retry_stats[operation_key]
        stats["total_attempts"] += 1
        
        if success:
            stats["successful_attempts"] += 1
            stats["last_success"] = utc_now().isoformat()
            if attempt > 1:
                stats["retry_attempts"] += 1
        else:
            stats["failed_attempts"] += 1
            stats["last_error"] = str(error) if error else "Unknown error"
    
    async def execute_with_retry(
        self,
        operation: Callable,
        operation_key: str,
        *args,
        **kwargs
    ) -> Any:
        """执行带重试的操作
        
        Args:
            operation: 要执行的异步操作
            operation_key: 操作唯一标识符（用于统计）
            *args, **kwargs: 传递给操作的参数
            
        Returns:
            操作执行结果
            
        Raises:
            最后一次尝试的异常
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                start_time = utc_now()
                result = await operation(*args, **kwargs)
                
                # 记录成功统计
                execution_time = (utc_now() - start_time).total_seconds()
                self._record_retry_stats(operation_key, attempt, True)
                
                if attempt > 1:
                    logger.info(
                        f"Operation {operation_key} succeeded after {attempt} attempts",
                        extra={
                            "operation_key": operation_key,
                            "attempt": attempt,
                            "execution_time": execution_time,
                            "retry_success": True
                        }
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # 记录失败统计
                self._record_retry_stats(operation_key, attempt, False, e)
                
                # 检查是否可重试
                if not self._is_retryable(e):
                    logger.error(
                        f"Operation {operation_key} failed with non-retryable error: {str(e)}",
                        extra={
                            "operation_key": operation_key,
                            "attempt": attempt,
                            "error_type": type(e).__name__,
                            "retryable": False
                        }
                    )
                    raise e
                
                # 检查是否已达到最大重试次数
                if attempt >= self.config.max_attempts:
                    logger.error(
                        f"Operation {operation_key} failed after {attempt} attempts",
                        extra={
                            "operation_key": operation_key,
                            "max_attempts_reached": True,
                            "final_error": str(e)
                        }
                    )
                    break
                
                # 计算延迟时间并等待
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Operation {operation_key} failed (attempt {attempt}/{self.config.max_attempts}), retrying in {delay:.2f}s: {str(e)}",
                    extra={
                        "operation_key": operation_key,
                        "attempt": attempt,
                        "max_attempts": self.config.max_attempts,
                        "delay": delay,
                        "error_type": type(e).__name__
                    }
                )
                
                await asyncio.sleep(delay)
        
        # 所有重试都失败了，抛出最后一个异常
        raise last_exception
    
    def get_retry_stats(self, operation_key: Optional[str] = None) -> Dict[str, Any]:
        """获取重试统计信息"""
        if operation_key:
            return self.retry_stats.get(operation_key, {})
        return self.retry_stats.copy()
    
    def reset_stats(self, operation_key: Optional[str] = None):
        """重置统计信息"""
        if operation_key:
            self.retry_stats.pop(operation_key, None)
        else:
            self.retry_stats.clear()

# 全局重试管理器实例
default_retry_manager = RetryManager()

def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    operation_key: Optional[str] = None
):
    """重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间
        max_delay: 最大延迟时间
        exponential_base: 指数退避基数
        jitter: 是否添加随机抖动
        operation_key: 操作标识符
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter
            )
            
            retry_manager = RetryManager(config)
            key = operation_key or f"{func.__module__}.{func.__name__}"
            
            return await retry_manager.execute_with_retry(func, key, *args, **kwargs)
        
        return wrapper
    return decorator

async def get_retry_manager() -> RetryManager:
    """获取重试管理器依赖注入"""
    return default_retry_manager
from src.core.logging import get_logger
