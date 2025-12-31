"""
工作流超时控制和取消机制
"""

from typing import Any, Dict, Optional, Callable
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from dataclasses import dataclass, field
from enum import Enum
from .state import MessagesState
from .checkpoints import checkpoint_manager

from src.core.logging import get_logger
logger = get_logger(__name__)

class TimeoutType(Enum):
    """超时类型"""
    NODE_TIMEOUT = "node_timeout"      # 节点级超时
    WORKFLOW_TIMEOUT = "workflow_timeout"  # 工作流级超时
    IDLE_TIMEOUT = "idle_timeout"      # 空闲超时

@dataclass
class TimeoutConfig:
    """超时配置"""
    node_timeout: float = 300.0        # 单个节点超时时间（秒）
    workflow_timeout: float = 3600.0   # 整个工作流超时时间（秒）
    idle_timeout: float = 1800.0       # 空闲超时时间（秒）
    enable_timeout: bool = True        # 是否启用超时控制

class CancellationToken:
    """取消令牌"""
    
    def __init__(self):
        self._cancelled = False
        self._cancellation_reason: Optional[str] = None
        self._cancelled_at: Optional[datetime] = None
        self._callbacks: list = []
    
    @property
    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self._cancelled
    
    @property
    def cancellation_reason(self) -> Optional[str]:
        """获取取消原因"""
        return self._cancellation_reason
    
    @property
    def cancelled_at(self) -> Optional[datetime]:
        """获取取消时间"""
        return self._cancelled_at
    
    def cancel(self, reason: str = "用户主动取消"):
        """取消操作"""
        if not self._cancelled:
            self._cancelled = True
            self._cancellation_reason = reason
            self._cancelled_at = utc_now()
            
            # 执行取消回调
            for callback in self._callbacks:
                try:
                    callback(reason)
                except Exception as e:
                    logger.error("取消回调执行失败", error=str(e), exc_info=True)
    
    def add_cancellation_callback(self, callback: Callable[[str], None]):
        """添加取消回调"""
        self._callbacks.append(callback)
    
    def throw_if_cancelled(self):
        """如果已取消则抛出异常"""
        if self._cancelled:
            raise asyncio.CancelledError(f"操作已取消: {self._cancellation_reason}")

class TimeoutManager:
    """超时管理器"""
    
    def __init__(self, config: TimeoutConfig = None):
        self.config = config or TimeoutConfig()
        self.active_timeouts: Dict[str, asyncio.Task] = {}
        self.cancellation_tokens: Dict[str, CancellationToken] = {}
    
    def create_cancellation_token(self, workflow_id: str) -> CancellationToken:
        """创建取消令牌"""
        token = CancellationToken()
        self.cancellation_tokens[workflow_id] = token
        return token
    
    def get_cancellation_token(self, workflow_id: str) -> Optional[CancellationToken]:
        """获取取消令牌"""
        return self.cancellation_tokens.get(workflow_id)
    
    def cancel_workflow(self, workflow_id: str, reason: str = "用户主动取消") -> bool:
        """取消工作流"""
        token = self.cancellation_tokens.get(workflow_id)
        if token:
            token.cancel(reason)
            return True
        return False
    
    async def execute_with_timeout(
        self, 
        func: Callable,
        state: MessagesState,
        timeout_seconds: float,
        timeout_type: TimeoutType = TimeoutType.NODE_TIMEOUT
    ) -> Any:
        """带超时控制的执行"""
        if not self.config.enable_timeout:
            # 如果禁用超时，直接执行
            return await func(state) if asyncio.iscoroutinefunction(func) else func(state)
        
        workflow_id = state["workflow_id"]
        token = self.get_cancellation_token(workflow_id)
        
        try:
            # 创建超时任务
            async def timeout_handler():
                await asyncio.sleep(timeout_seconds)
                raise asyncio.TimeoutError(f"{timeout_type.value} 超时 ({timeout_seconds}秒)")
            
            # 创建执行任务
            async def execute_task():
                if token:
                    token.throw_if_cancelled()
                
                result = await func(state) if asyncio.iscoroutinefunction(func) else func(state)
                
                if token:
                    token.throw_if_cancelled()
                
                return result
            
            # 并发执行，哪个先完成就返回哪个结果
            timeout_task = create_task_with_logging(timeout_handler())
            execute_task_obj = create_task_with_logging(execute_task())
            
            try:
                done, pending = await asyncio.wait(
                    [timeout_task, execute_task_obj],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # 取消未完成的任务
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        raise
                
                # 检查结果
                completed_task = list(done)[0]
                if completed_task == timeout_task:
                    # 超时了
                    await self._handle_timeout(state, timeout_type, timeout_seconds)
                    raise asyncio.TimeoutError(f"{timeout_type.value} 执行超时")
                else:
                    # 正常完成
                    return completed_task.result()
                    
            except asyncio.CancelledError:
                # 被取消了
                await self._handle_cancellation(state, token.cancellation_reason if token else "未知原因")
                raise
            
        except asyncio.TimeoutError:
            await self._handle_timeout(state, timeout_type, timeout_seconds)
            raise
        except asyncio.CancelledError:
            await self._handle_cancellation(state, token.cancellation_reason if token else "未知原因")
            raise
    
    async def _handle_timeout(self, state: MessagesState, timeout_type: TimeoutType, timeout_seconds: float):
        """处理超时"""
        timeout_info = {
            "type": timeout_type.value,
            "timeout_seconds": timeout_seconds,
            "timestamp": utc_now().isoformat(),
            "node": state["metadata"].get("current_node", "unknown")
        }
        
        # 更新状态
        state["metadata"]["status"] = "timeout"
        state["metadata"]["timeout_info"] = timeout_info
        
        # 记录超时日志
        if "timeouts" not in state["context"]:
            state["context"]["timeouts"] = []
        state["context"]["timeouts"].append(timeout_info)
        
        # 创建超时检查点
        try:
            await checkpoint_manager.create_checkpoint(
                workflow_id=state["workflow_id"],
                state=state,
                metadata={"type": "timeout_checkpoint", "timeout_type": timeout_type.value}
            )
        except Exception as e:
            logger.error("创建超时检查点失败", error=str(e), exc_info=True)
    
    async def _handle_cancellation(self, state: MessagesState, reason: str):
        """处理取消"""
        cancellation_info = {
            "reason": reason,
            "timestamp": utc_now().isoformat(),
            "node": state["metadata"].get("current_node", "unknown")
        }
        
        # 更新状态
        state["metadata"]["status"] = "cancelled"
        state["metadata"]["cancellation_info"] = cancellation_info
        
        # 记录取消日志
        if "cancellations" not in state["context"]:
            state["context"]["cancellations"] = []
        state["context"]["cancellations"].append(cancellation_info)
        
        # 创建取消检查点
        try:
            await checkpoint_manager.create_checkpoint(
                workflow_id=state["workflow_id"],
                state=state,
                metadata={"type": "cancellation_checkpoint", "reason": reason}
            )
        except Exception as e:
            logger.error("创建取消检查点失败", error=str(e), exc_info=True)
    
    async def start_workflow_timeout(self, workflow_id: str):
        """启动工作流级超时监控"""
        if workflow_id in self.active_timeouts:
            return  # 已经在监控中
        
        async def workflow_timeout_monitor():
            try:
                await asyncio.sleep(self.config.workflow_timeout)
                # 超时，取消工作流
                self.cancel_workflow(workflow_id, f"工作流超时 ({self.config.workflow_timeout}秒)")
            except asyncio.CancelledError:
                raise  # 正常取消监控
        
        self.active_timeouts[workflow_id] = create_task_with_logging(workflow_timeout_monitor())
    
    async def stop_workflow_timeout(self, workflow_id: str):
        """停止工作流级超时监控"""
        timeout_task = self.active_timeouts.get(workflow_id)
        if timeout_task:
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                raise
            del self.active_timeouts[workflow_id]
    
    async def start_idle_timeout(self, workflow_id: str, reset_callback: Optional[Callable] = None):
        """启动空闲超时监控"""
        idle_key = f"{workflow_id}_idle"
        
        # 取消现有的空闲监控
        if idle_key in self.active_timeouts:
            self.active_timeouts[idle_key].cancel()
        
        async def idle_timeout_monitor():
            try:
                await asyncio.sleep(self.config.idle_timeout)
                # 空闲超时，暂停工作流
                token = self.get_cancellation_token(workflow_id)
                if token and not token.is_cancelled:
                    # 可以选择暂停而不是取消
                    logger.warning("工作流空闲超时，建议暂停", workflow_id=workflow_id)
                    if reset_callback:
                        await reset_callback()
            except asyncio.CancelledError:
                raise
        
        self.active_timeouts[idle_key] = create_task_with_logging(idle_timeout_monitor())
    
    def reset_idle_timeout(self, workflow_id: str):
        """重置空闲超时"""
        idle_key = f"{workflow_id}_idle"
        if idle_key in self.active_timeouts:
            self.active_timeouts[idle_key].cancel()
            # 重新启动空闲监控
            create_task_with_logging(self.start_idle_timeout(workflow_id))
    
    def cleanup(self, workflow_id: str):
        """清理超时监控"""
        # 清理工作流超时
        if workflow_id in self.active_timeouts:
            self.active_timeouts[workflow_id].cancel()
            del self.active_timeouts[workflow_id]
        
        # 清理空闲超时
        idle_key = f"{workflow_id}_idle"
        if idle_key in self.active_timeouts:
            self.active_timeouts[idle_key].cancel()
            del self.active_timeouts[idle_key]
        
        # 清理取消令牌
        if workflow_id in self.cancellation_tokens:
            del self.cancellation_tokens[workflow_id]

# 全局超时管理器实例
timeout_manager = TimeoutManager()
