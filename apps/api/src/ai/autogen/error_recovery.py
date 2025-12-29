"""
错误恢复和补偿机制
实现事件处理的重试、死信队列和补偿事务
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from .events import Event, EventType, EventPriority
from .event_processors import EventProcessor, EventContext, ProcessingResult
from .event_store import EventStore

from src.core.logging import get_logger
logger = get_logger(__name__)

class RetryStrategy(str, Enum):
    """重试策略"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    CUSTOM = "custom"

class CompensationAction(str, Enum):
    """补偿动作"""
    ROLLBACK = "rollback"
    COMPENSATE = "compensate"
    RETRY = "retry"
    SKIP = "skip"
    MANUAL = "manual"

@dataclass
class RetryPolicy:
    """重试策略"""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay_ms: int = 1000
    max_delay_ms: int = 60000
    multiplier: float = 2.0
    jitter: bool = True
    retryable_errors: List[type] = field(default_factory=list)
    non_retryable_errors: List[type] = field(default_factory=list)
    
    def calculate_delay(self, attempt: int) -> int:
        """计算重试延迟"""
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(
                self.initial_delay_ms * (self.multiplier ** attempt),
                self.max_delay_ms
            )
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(
                self.initial_delay_ms * attempt,
                self.max_delay_ms
            )
        else:  # FIXED_DELAY
            delay = self.initial_delay_ms
        
        # 添加抖动
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return int(delay)
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_retries:
            return False
        
        # 检查不可重试错误
        if self.non_retryable_errors:
            for error_type in self.non_retryable_errors:
                if isinstance(error, error_type):
                    return False
        
        # 检查可重试错误
        if self.retryable_errors:
            for error_type in self.retryable_errors:
                if isinstance(error, error_type):
                    return True
            return False  # 如果指定了可重试错误列表，其他错误不重试
        
        return True  # 默认所有错误都可重试

@dataclass
class CircuitBreakerState:
    """断路器状态"""
    CLOSED = "closed"  # 正常状态
    OPEN = "open"      # 断开状态
    HALF_OPEN = "half_open"  # 半开状态

@dataclass
class CircuitBreaker:
    """断路器"""
    failure_threshold: int = 5
    recovery_timeout: int = 60000  # 毫秒
    success_threshold: int = 3
    
    state: str = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=lambda: utc_now())
    
    def record_success(self) -> None:
        """记录成功"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.close()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = utc_now()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.open()
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.open()
    
    def open(self) -> None:
        """打开断路器"""
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = utc_now()
        self.success_count = 0
        logger.warning("断路器打开", failure_count=self.failure_count)
    
    def close(self) -> None:
        """关闭断路器"""
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = utc_now()
        self.failure_count = 0
        self.success_count = 0
        logger.info("断路器关闭")
    
    def half_open(self) -> None:
        """半开断路器"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_state_change = utc_now()
        self.success_count = 0
        logger.info("断路器半开")
    
    def is_open(self) -> bool:
        """检查断路器是否打开"""
        if self.state == CircuitBreakerState.OPEN:
            # 检查是否到了恢复时间
            if self.last_failure_time:
                time_since_failure = (utc_now() - self.last_failure_time).total_seconds() * 1000
                if time_since_failure >= self.recovery_timeout:
                    self.half_open()
                    return False
            return True
        return False
    
    def allow_request(self) -> bool:
        """是否允许请求通过"""
        return not self.is_open()

class RetryableEventProcessor(EventProcessor):
    """可重试的事件处理器"""
    
    def __init__(
        self,
        wrapped_processor: EventProcessor,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        super().__init__(name=f"Retryable_{wrapped_processor.name}")
        self.wrapped_processor = wrapped_processor
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker
        
        self.retry_stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0
        }
    
    async def process(self, event: Event, context: EventContext) -> ProcessingResult:
        """处理事件（带重试）"""
        # 检查断路器
        if self.circuit_breaker and not self.circuit_breaker.allow_request():
            return ProcessingResult(
                success=False,
                error="Circuit breaker is open",
                should_retry=False
            )
        
        attempt = 0
        last_error = None
        
        while attempt <= self.retry_policy.max_retries:
            try:
                # 如果不是第一次尝试，等待
                if attempt > 0:
                    delay_ms = self.retry_policy.calculate_delay(attempt - 1)
                    await asyncio.sleep(delay_ms / 1000)
                    self.retry_stats["total_retries"] += 1
                    
                    logger.info(
                        "重试事件处理",
                        attempt=attempt,
                        delay_ms=delay_ms,
                        event_type=event.type
                    )
                
                # 执行处理
                result = await self.wrapped_processor.process(event, context)
                
                if result.success:
                    if self.circuit_breaker:
                        self.circuit_breaker.record_success()
                    if attempt > 0:
                        self.retry_stats["successful_retries"] += 1
                    return result
                else:
                    # 处理失败但没有异常
                    if not result.should_retry:
                        return result
                    
                    last_error = Exception(result.error or "Processing failed")
                    
            except Exception as e:
                last_error = e
                logger.error(
                    "事件处理异常",
                    attempt=attempt,
                    error=str(e),
                    event_type=event.type
                )
            
            # 检查是否应该重试
            if last_error and not self.retry_policy.should_retry(last_error, attempt):
                break
            
            attempt += 1
        
        # 所有重试都失败
        if self.circuit_breaker:
            self.circuit_breaker.record_failure()
        
        if attempt > 0:
            self.retry_stats["failed_retries"] += 1
        
        return ProcessingResult(
            success=False,
            error=str(last_error) if last_error else "Max retries exceeded",
            retry_count=attempt,
            should_retry=False
        )
    
    def can_handle(self, event: Event) -> bool:
        """判断是否能处理该事件"""
        return self.wrapped_processor.can_handle(event)
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """获取重试统计"""
        return {
            **self.retry_stats,
            "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else None
        }

class DeadLetterQueue:
    """死信队列"""
    
    def __init__(self, event_store: Optional[EventStore] = None):
        self.event_store = event_store
        self.dead_letters: List[Dict[str, Any]] = []
        self.max_dead_letters = 10000
        
        self.stats = {
            "total_dead_letters": 0,
            "reprocessed": 0,
            "discarded": 0
        }
    
    async def add_dead_letter(
        self,
        event: Event,
        error: str,
        retry_count: int = 0,
        processor_name: Optional[str] = None
    ) -> None:
        """添加死信"""
        dead_letter = {
            "id": str(event.id if hasattr(event, 'id') else None),
            "event": event.to_dict() if hasattr(event, 'to_dict') else str(event),
            "error": error,
            "retry_count": retry_count,
            "processor_name": processor_name,
            "timestamp": utc_now().isoformat()
        }
        
        # 存储到持久化存储
        if self.event_store:
            await self.event_store.move_to_dead_letter(event, error, retry_count)
        
        # 内存缓存
        self.dead_letters.append(dead_letter)
        if len(self.dead_letters) > self.max_dead_letters:
            self.dead_letters = self.dead_letters[-self.max_dead_letters:]
        
        self.stats["total_dead_letters"] += 1
        
        logger.warning(
            "事件移至死信队列",
            event_id=dead_letter["id"],
            error=error,
            retry_count=retry_count
        )
    
    async def get_dead_letters(
        self,
        limit: int = 100,
        processor_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取死信"""
        if self.event_store:
            # 从持久化存储获取
            return await self.event_store.get_dead_letter_events(limit)
        
        # 从内存获取
        dead_letters = self.dead_letters
        if processor_name:
            dead_letters = [
                dl for dl in dead_letters
                if dl.get("processor_name") == processor_name
            ]
        
        return dead_letters[-limit:]
    
    async def reprocess_dead_letter(
        self,
        dead_letter_id: str,
        processor: EventProcessor,
        context: Optional[EventContext] = None
    ) -> ProcessingResult:
        """重新处理死信"""
        # 查找死信
        dead_letter = None
        for dl in self.dead_letters:
            if dl["id"] == dead_letter_id:
                dead_letter = dl
                break
        
        if not dead_letter:
            return ProcessingResult(
                success=False,
                error=f"Dead letter not found: {dead_letter_id}"
            )
        
        try:
            # 重建事件
            event_data = dead_letter["event"]
            if isinstance(event_data, dict):
                event = Event(**event_data)
            else:
                return ProcessingResult(
                    success=False,
                    error="Cannot reconstruct event from dead letter"
                )
            
            # 创建上下文
            if not context:
                context = EventContext(
                    correlation_id=str(event.correlation_id if hasattr(event, 'correlation_id') else ""),
                    metadata={"is_reprocess": True, "dead_letter_id": dead_letter_id}
                )
            
            # 重新处理
            result = await processor.process(event, context)
            
            if result.success:
                self.stats["reprocessed"] += 1
                # 从死信队列移除
                self.dead_letters.remove(dead_letter)
            
            return result
            
        except Exception as e:
            logger.error(f"重新处理死信失败", dead_letter_id=dead_letter_id, error=str(e))
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    def discard_dead_letter(self, dead_letter_id: str) -> bool:
        """丢弃死信"""
        for i, dl in enumerate(self.dead_letters):
            if dl["id"] == dead_letter_id:
                del self.dead_letters[i]
                self.stats["discarded"] += 1
                logger.info(f"丢弃死信", dead_letter_id=dead_letter_id)
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "current_dead_letters": len(self.dead_letters)
        }

class CompensationManager:
    """补偿管理器"""
    
    def __init__(self):
        self.compensation_handlers: Dict[str, Callable] = {}
        self.saga_transactions: Dict[str, List[Dict[str, Any]]] = {}
        
        self.stats = {
            "total_compensations": 0,
            "successful_compensations": 0,
            "failed_compensations": 0
        }
    
    def register_compensation(
        self,
        operation_type: str,
        compensation_handler: Callable
    ) -> None:
        """注册补偿处理器"""
        self.compensation_handlers[operation_type] = compensation_handler
        logger.info(f"注册补偿处理器", operation_type=operation_type)
    
    async def start_saga(self, saga_id: str) -> None:
        """开始Saga事务"""
        self.saga_transactions[saga_id] = []
        logger.info(f"开始Saga事务", saga_id=saga_id)
    
    async def record_operation(
        self,
        saga_id: str,
        operation_type: str,
        operation_data: Dict[str, Any]
    ) -> None:
        """记录操作（用于后续补偿）"""
        if saga_id not in self.saga_transactions:
            await self.start_saga(saga_id)
        
        self.saga_transactions[saga_id].append({
            "operation_type": operation_type,
            "operation_data": operation_data,
            "timestamp": utc_now().isoformat()
        })
    
    async def compensate_saga(
        self,
        saga_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """补偿Saga事务"""
        if saga_id not in self.saga_transactions:
            return {
                "success": False,
                "error": f"Saga not found: {saga_id}"
            }
        
        operations = self.saga_transactions[saga_id]
        compensated = []
        failed = []
        
        self.stats["total_compensations"] += 1
        
        # 反向执行补偿
        for operation in reversed(operations):
            operation_type = operation["operation_type"]
            operation_data = operation["operation_data"]
            
            if operation_type in self.compensation_handlers:
                try:
                    handler = self.compensation_handlers[operation_type]
                    await handler(operation_data)
                    compensated.append(operation_type)
                    
                    logger.info(
                        "补偿操作成功",
                        saga_id=saga_id,
                        operation_type=operation_type
                    )
                    
                except Exception as e:
                    failed.append({
                        "operation_type": operation_type,
                        "error": str(e)
                    })
                    
                    logger.error(
                        "补偿操作失败",
                        saga_id=saga_id,
                        operation_type=operation_type,
                        error=str(e)
                    )
            else:
                logger.warning(
                    "未找到补偿处理器",
                    saga_id=saga_id,
                    operation_type=operation_type
                )
        
        # 清理Saga记录
        del self.saga_transactions[saga_id]
        
        if failed:
            self.stats["failed_compensations"] += 1
            return {
                "success": False,
                "compensated": compensated,
                "failed": failed,
                "reason": reason
            }
        else:
            self.stats["successful_compensations"] += 1
            return {
                "success": True,
                "compensated": compensated,
                "reason": reason
            }
    
    def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """获取Saga状态"""
        if saga_id not in self.saga_transactions:
            return None
        
        operations = self.saga_transactions[saga_id]
        return {
            "saga_id": saga_id,
            "operation_count": len(operations),
            "operations": [op["operation_type"] for op in operations],
            "start_time": operations[0]["timestamp"] if operations else None,
            "last_operation_time": operations[-1]["timestamp"] if operations else None
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "active_sagas": len(self.saga_transactions),
            "registered_handlers": len(self.compensation_handlers)
        }

class ErrorRecoveryService:
    """错误恢复服务"""
    
    def __init__(
        self,
        dead_letter_queue: DeadLetterQueue,
        compensation_manager: CompensationManager,
        event_store: Optional[EventStore] = None
    ):
        self.dead_letter_queue = dead_letter_queue
        self.compensation_manager = compensation_manager
        self.event_store = event_store
        
        # 恢复策略
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # 统计
        self.stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0
        }
    
    def register_recovery_strategy(
        self,
        error_type: str,
        strategy: Callable
    ) -> None:
        """注册恢复策略"""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"注册恢复策略", error_type=error_type)
    
    async def handle_error(
        self,
        event: Event,
        error: Exception,
        context: EventContext,
        processor_name: Optional[str] = None
    ) -> CompensationAction:
        """处理错误"""
        error_type = type(error).__name__
        
        # 查找恢复策略
        if error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error_type]
                action = await strategy(event, error, context)
                
                self.stats["total_recoveries"] += 1
                
                if action in [CompensationAction.RETRY, CompensationAction.COMPENSATE]:
                    self.stats["successful_recoveries"] += 1
                
                return action
                
            except Exception as e:
                logger.error(f"恢复策略执行失败", error_type=error_type, error=str(e))
                self.stats["failed_recoveries"] += 1
        
        # 默认策略：移至死信队列
        await self.dead_letter_queue.add_dead_letter(
            event,
            str(error),
            retry_count=context.metadata.get("retry_count", 0) if context else 0,
            processor_name=processor_name
        )
        
        return CompensationAction.SKIP
    
    async def perform_compensation(
        self,
        saga_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """执行补偿"""
        return await self.compensation_manager.compensate_saga(saga_id, reason)
    
    async def analyze_failures(
        self,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """分析失败模式"""
        analysis = {
            "time_window_minutes": time_window_minutes,
            "failure_patterns": [],
            "recommendations": []
        }
        
        # 获取死信队列中的失败事件
        dead_letters = await self.dead_letter_queue.get_dead_letters(limit=1000)
        
        # 分析错误类型分布
        error_types = {}
        for dl in dead_letters:
            error = dl.get("error", "unknown")
            error_type = error.split(":")[0] if ":" in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # 找出最常见的错误
        if error_types:
            sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
            top_errors = sorted_errors[:5]
            
            analysis["failure_patterns"] = [
                {
                    "error_type": error_type,
                    "count": count,
                    "percentage": count / len(dead_letters) * 100
                }
                for error_type, count in top_errors
            ]
            
            # 生成建议
            for error_type, count in top_errors:
                if count > 10:
                    analysis["recommendations"].append(
                        f"错误类型 '{error_type}' 频繁出现，建议添加特定的恢复策略"
                    )
        
        # 分析重试模式
        high_retry_events = [
            dl for dl in dead_letters
            if dl.get("retry_count", 0) >= 3
        ]
        
        if len(high_retry_events) > len(dead_letters) * 0.2:
            analysis["recommendations"].append(
                "大量事件经过多次重试仍然失败，建议检查重试策略配置"
            )
        
        return analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "dead_letter_stats": self.dead_letter_queue.get_stats(),
            "compensation_stats": self.compensation_manager.get_stats(),
            "registered_strategies": len(self.recovery_strategies)
        }
