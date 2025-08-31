"""
异步事件处理框架扩展
实现事件处理器、上下文和处理引擎
"""
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, Union
import structlog

from .events import Event, EventType, EventPriority, EventBus

logger = structlog.get_logger(__name__)


@dataclass
class EventContext:
    """事件处理上下文"""
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: utc_now())
    
    def add_metadata(self, key: str, value: Any) -> None:
        """添加元数据"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)


@dataclass
class ProcessingResult:
    """事件处理结果"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    retry_count: int = 0
    should_retry: bool = False
    retry_after: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "processing_time": self.processing_time,
            "retry_count": self.retry_count,
            "should_retry": self.should_retry,
            "retry_after": self.retry_after.total_seconds() if self.retry_after else None
        }


class EventProcessor(ABC):
    """事件处理器基类"""
    
    def __init__(self, name: str = None, priority: int = 10):
        self.name = name or self.__class__.__name__
        self._priority = priority
        self.metrics = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "total_processing_time": 0.0
        }
    
    @abstractmethod
    async def process(self, event: Event, context: EventContext) -> ProcessingResult:
        """处理事件"""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """判断是否能处理该事件"""
        pass
    
    @property
    def priority(self) -> int:
        """处理器优先级（数字越小优先级越高）"""
        return self._priority
    
    async def pre_process(self, event: Event, context: EventContext) -> None:
        """预处理钩子"""
        pass
    
    async def post_process(self, event: Event, context: EventContext, result: ProcessingResult) -> None:
        """后处理钩子"""
        self.metrics["processed"] += 1
        if result.success:
            self.metrics["succeeded"] += 1
        else:
            self.metrics["failed"] += 1
        self.metrics["total_processing_time"] += result.processing_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取处理器指标"""
        return {
            **self.metrics,
            "average_processing_time": (
                self.metrics["total_processing_time"] / self.metrics["processed"]
                if self.metrics["processed"] > 0 else 0
            ),
            "success_rate": (
                self.metrics["succeeded"] / self.metrics["processed"]
                if self.metrics["processed"] > 0 else 0
            )
        }


class AsyncEventProcessingEngine:
    """异步事件处理引擎"""
    
    def __init__(self, max_workers: int = 10, batch_size: int = 100):
        self.processors: List[EventProcessor] = []
        self.priority_queues: Dict[EventPriority, asyncio.PriorityQueue] = {}
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.middleware: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        # 批处理缓冲区
        self.batch_buffers: Dict[EventPriority, List[tuple]] = {
            priority: [] for priority in EventPriority
        }
        self.batch_locks: Dict[EventPriority, asyncio.Lock] = {
            priority: asyncio.Lock() for priority in EventPriority
        }
        
        # 统计信息
        self.stats = {
            "events_submitted": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_retried": 0,
            "batches_processed": 0
        }
        
        logger.info(
            "事件处理引擎初始化",
            max_workers=max_workers,
            batch_size=batch_size
        )
    
    async def start(self) -> None:
        """启动事件处理引擎"""
        if self.running:
            logger.warning("事件处理引擎已在运行")
            return
        
        self.running = True
        
        # 初始化优先级队列
        for priority in EventPriority:
            self.priority_queues[priority] = asyncio.PriorityQueue()
        
        # 为每个优先级启动工作者
        for priority in EventPriority:
            # 根据优先级分配不同数量的工作者
            worker_count = self._get_worker_count_for_priority(priority)
            for i in range(worker_count):
                worker = asyncio.create_task(
                    self._process_events_worker(priority, f"worker-{priority.value}-{i}")
                )
                self.workers.append(worker)
        
        # 启动批处理刷新任务
        self.workers.append(
            asyncio.create_task(self._batch_flush_worker())
        )
        
        logger.info("事件处理引擎启动完成", worker_count=len(self.workers))
    
    async def stop(self) -> None:
        """停止事件处理引擎"""
        if not self.running:
            return
        
        self.running = False
        
        # 处理剩余的批处理缓冲
        await self._flush_all_batches()
        
        # 取消所有工作者
        for worker in self.workers:
            worker.cancel()
        
        # 等待所有工作者完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info(
            "事件处理引擎停止",
            stats=self.stats
        )
    
    def register_processor(self, processor: EventProcessor) -> None:
        """注册事件处理器"""
        self.processors.append(processor)
        # 按优先级排序
        self.processors.sort(key=lambda p: p.priority)
        logger.info(
            "注册事件处理器",
            processor=processor.name,
            priority=processor.priority
        )
    
    def unregister_processor(self, processor: EventProcessor) -> None:
        """注销事件处理器"""
        if processor in self.processors:
            self.processors.remove(processor)
            logger.info("注销事件处理器", processor=processor.name)
    
    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件"""
        self.middleware.append(middleware)
    
    def add_error_handler(self, handler: Callable) -> None:
        """添加错误处理器"""
        self.error_handlers.append(handler)
    
    async def submit_event(
        self, 
        event: Event, 
        priority: EventPriority = None,
        batch: bool = False
    ) -> None:
        """提交事件到处理队列"""
        if priority is None:
            priority = event.priority if hasattr(event, 'priority') else EventPriority.NORMAL
        
        self.stats["events_submitted"] += 1
        
        if batch:
            # 添加到批处理缓冲
            async with self.batch_locks[priority]:
                self.batch_buffers[priority].append((utc_now().timestamp(), event))
                
                # 如果缓冲区满了，立即刷新
                if len(self.batch_buffers[priority]) >= self.batch_size:
                    await self._flush_batch(priority)
        else:
            # 直接添加到队列
            await self.priority_queues[priority].put((utc_now().timestamp(), event))
    
    async def submit_batch(self, events: List[Event], priority: EventPriority = EventPriority.NORMAL) -> None:
        """批量提交事件"""
        for event in events:
            await self.submit_event(event, priority, batch=True)
    
    def _get_worker_count_for_priority(self, priority: EventPriority) -> int:
        """根据优先级获取工作者数量"""
        priority_weights = {
            EventPriority.CRITICAL: 0.4,  # 40%的工作者
            EventPriority.HIGH: 0.3,      # 30%的工作者
            EventPriority.NORMAL: 0.2,    # 20%的工作者
            EventPriority.LOW: 0.1        # 10%的工作者
        }
        
        if priority.value not in priority_weights:
            return 1
        
        count = max(1, int(self.max_workers * priority_weights[priority]))
        return count
    
    async def _process_events_worker(self, priority: EventPriority, worker_id: str) -> None:
        """事件处理工作者"""
        logger.info("事件处理工作者启动", worker=worker_id, priority=priority.value)
        queue = self.priority_queues[priority]
        
        while self.running:
            try:
                # 获取事件（带超时）
                timestamp, event = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # 创建处理上下文
                context = EventContext(
                    correlation_id=event.correlation_id if hasattr(event, 'correlation_id') else str(uuid.uuid4()),
                    session_id=event.session_id if hasattr(event, 'session_id') else None
                )
                
                # 执行中间件
                for middleware in self.middleware:
                    await middleware(event, context)
                
                # 处理事件
                await self._process_single_event(event, context, worker_id)
                
                queue.task_done()
                self.stats["events_processed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"事件处理工作者异常", worker=worker_id, error=str(e))
                self.stats["events_failed"] += 1
        
        logger.info("事件处理工作者停止", worker=worker_id)
    
    async def _process_single_event(self, event: Event, context: EventContext, worker_id: str) -> None:
        """处理单个事件"""
        start_time = asyncio.get_event_loop().time()
        
        # 找到所有能处理该事件的处理器
        processors = [p for p in self.processors if p.can_handle(event)]
        
        if not processors:
            logger.debug("没有找到事件处理器", event_type=event.type, worker=worker_id)
            return
        
        # 按优先级顺序处理
        for processor in processors:
            try:
                # 预处理
                await processor.pre_process(event, context)
                
                # 处理事件
                result = await processor.process(event, context)
                result.processing_time = asyncio.get_event_loop().time() - start_time
                
                # 后处理
                await processor.post_process(event, context, result)
                
                # 如果需要重试
                if not result.success and result.should_retry:
                    await self._schedule_retry(event, result, context)
                
                logger.debug(
                    "事件处理完成",
                    event_type=event.type,
                    processor=processor.name,
                    success=result.success,
                    processing_time_ms=result.processing_time * 1000,
                    worker=worker_id
                )
                
            except Exception as e:
                logger.error(
                    "事件处理器异常",
                    processor=processor.name,
                    event_type=event.type,
                    error=str(e),
                    worker=worker_id
                )
                
                # 执行错误处理器
                for error_handler in self.error_handlers:
                    try:
                        await error_handler(event, context, e)
                    except Exception as handler_error:
                        logger.error("错误处理器异常", error=str(handler_error))
    
    async def _schedule_retry(self, event: Event, result: ProcessingResult, context: EventContext) -> None:
        """调度事件重试"""
        self.stats["events_retried"] += 1
        
        # 计算重试延迟
        if result.retry_after:
            delay = result.retry_after.total_seconds()
        else:
            # 指数退避
            delay = min(300, 2 ** result.retry_count)  # 最多等待5分钟
        
        logger.info(
            "调度事件重试",
            event_type=event.type,
            retry_count=result.retry_count,
            delay_seconds=delay
        )
        
        # 延迟后重新提交事件
        await asyncio.sleep(delay)
        
        # 更新重试计数
        if hasattr(event, 'data') and isinstance(event.data, dict):
            event.data['retry_count'] = result.retry_count + 1
        
        await self.submit_event(event, EventPriority.HIGH)
    
    async def _batch_flush_worker(self) -> None:
        """批处理刷新工作者"""
        logger.info("批处理刷新工作者启动")
        
        while self.running:
            try:
                # 定期刷新所有批处理缓冲
                await asyncio.sleep(1.0)  # 每秒检查一次
                
                for priority in EventPriority:
                    async with self.batch_locks[priority]:
                        if self.batch_buffers[priority]:
                            await self._flush_batch(priority)
                
            except Exception as e:
                logger.error("批处理刷新异常", error=str(e))
        
        logger.info("批处理刷新工作者停止")
    
    async def _flush_batch(self, priority: EventPriority) -> None:
        """刷新批处理缓冲"""
        if not self.batch_buffers[priority]:
            return
        
        batch = self.batch_buffers[priority].copy()
        self.batch_buffers[priority].clear()
        
        # 将批处理事件添加到队列
        for item in batch:
            await self.priority_queues[priority].put(item)
        
        self.stats["batches_processed"] += 1
        
        logger.debug(
            "批处理缓冲刷新",
            priority=priority.value,
            batch_size=len(batch)
        )
    
    async def _flush_all_batches(self) -> None:
        """刷新所有批处理缓冲"""
        for priority in EventPriority:
            async with self.batch_locks[priority]:
                await self._flush_batch(priority)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        queue_sizes = {
            priority.value: self.priority_queues[priority].qsize()
            for priority in EventPriority
            if priority in self.priority_queues
        }
        
        buffer_sizes = {
            priority.value: len(self.batch_buffers[priority])
            for priority in EventPriority
        }
        
        processor_metrics = {
            processor.name: processor.get_metrics()
            for processor in self.processors
        }
        
        return {
            **self.stats,
            "queue_sizes": queue_sizes,
            "buffer_sizes": buffer_sizes,
            "processor_metrics": processor_metrics,
            "running": self.running,
            "worker_count": len(self.workers),
            "processor_count": len(self.processors)
        }


# 示例：智能体消息处理器
class AgentMessageProcessor(EventProcessor):
    """智能体消息处理器"""
    
    def __init__(self, agent_manager=None):
        super().__init__(name="AgentMessageProcessor", priority=1)
        self.agent_manager = agent_manager
    
    async def process(self, event: Event, context: EventContext) -> ProcessingResult:
        """处理智能体消息事件"""
        try:
            if event.type == EventType.MESSAGE_SENT:
                # 处理消息发送
                await self._handle_message_sent(event, context)
            elif event.type == EventType.MESSAGE_RECEIVED:
                # 处理消息接收
                await self._handle_message_received(event, context)
            
            return ProcessingResult(success=True)
            
        except Exception as e:
            logger.error("处理智能体消息失败", error=str(e))
            return ProcessingResult(
                success=False,
                error=str(e),
                should_retry=True,
                retry_after=timedelta(seconds=5)
            )
    
    def can_handle(self, event: Event) -> bool:
        """检查是否能处理该事件"""
        return event.type in [EventType.MESSAGE_SENT, EventType.MESSAGE_RECEIVED]
    
    async def _handle_message_sent(self, event: Event, context: EventContext) -> None:
        """处理消息发送事件"""
        message_data = event.data if hasattr(event, 'data') else {}
        
        # 记录消息发送
        logger.info(
            "处理消息发送事件",
            source=event.source if hasattr(event, 'source') else None,
            target=event.target if hasattr(event, 'target') else None,
            message_id=message_data.get('message_id')
        )
        
        # 更新智能体状态
        if self.agent_manager and hasattr(event, 'source'):
            # 这里可以更新智能体的状态
            pass
    
    async def _handle_message_received(self, event: Event, context: EventContext) -> None:
        """处理消息接收事件"""
        message_data = event.data if hasattr(event, 'data') else {}
        
        # 记录消息接收
        logger.info(
            "处理消息接收事件",
            source=event.source if hasattr(event, 'source') else None,
            target=event.target if hasattr(event, 'target') else None,
            message_id=message_data.get('message_id')
        )
        
        # 触发智能体响应
        if self.agent_manager and hasattr(event, 'target'):
            # 这里可以触发智能体的响应逻辑
            pass


# 任务处理器
class TaskProcessor(EventProcessor):
    """任务事件处理器"""
    
    def __init__(self):
        super().__init__(name="TaskProcessor", priority=5)
    
    async def process(self, event: Event, context: EventContext) -> ProcessingResult:
        """处理任务事件"""
        try:
            if event.type == EventType.TASK_ASSIGNED:
                await self._handle_task_assigned(event, context)
            elif event.type == EventType.TASK_STARTED:
                await self._handle_task_started(event, context)
            elif event.type == EventType.TASK_COMPLETED:
                await self._handle_task_completed(event, context)
            elif event.type == EventType.TASK_FAILED:
                await self._handle_task_failed(event, context)
            
            return ProcessingResult(success=True)
            
        except Exception as e:
            return ProcessingResult(success=False, error=str(e))
    
    def can_handle(self, event: Event) -> bool:
        """检查是否能处理该事件"""
        return event.type in [
            EventType.TASK_ASSIGNED,
            EventType.TASK_STARTED,
            EventType.TASK_COMPLETED,
            EventType.TASK_FAILED
        ]
    
    async def _handle_task_assigned(self, event: Event, context: EventContext) -> None:
        """处理任务分配事件"""
        logger.info("任务已分配", task_id=event.data.get('task_id') if hasattr(event, 'data') else None)
    
    async def _handle_task_started(self, event: Event, context: EventContext) -> None:
        """处理任务启动事件"""
        logger.info("任务已启动", task_id=event.data.get('task_id') if hasattr(event, 'data') else None)
    
    async def _handle_task_completed(self, event: Event, context: EventContext) -> None:
        """处理任务完成事件"""
        logger.info("任务已完成", task_id=event.data.get('task_id') if hasattr(event, 'data') else None)
    
    async def _handle_task_failed(self, event: Event, context: EventContext) -> None:
        """处理任务失败事件"""
        logger.error("任务失败", task_id=event.data.get('task_id') if hasattr(event, 'data') else None)


# 系统监控处理器
class SystemMonitorProcessor(EventProcessor):
    """系统监控事件处理器"""
    
    def __init__(self):
        super().__init__(name="SystemMonitorProcessor", priority=10)
        self.error_count = 0
        self.last_error_time = None
    
    async def process(self, event: Event, context: EventContext) -> ProcessingResult:
        """处理系统监控事件"""
        try:
            if event.type == EventType.ERROR_OCCURRED:
                await self._handle_error(event, context)
            elif event.type == EventType.SYSTEM_STATUS_CHANGED:
                await self._handle_status_change(event, context)
            
            return ProcessingResult(success=True)
            
        except Exception as e:
            return ProcessingResult(success=False, error=str(e))
    
    def can_handle(self, event: Event) -> bool:
        """检查是否能处理该事件"""
        return event.type in [EventType.ERROR_OCCURRED, EventType.SYSTEM_STATUS_CHANGED]
    
    async def _handle_error(self, event: Event, context: EventContext) -> None:
        """处理错误事件"""
        self.error_count += 1
        self.last_error_time = utc_now()
        
        logger.error(
            "系统错误",
            error_count=self.error_count,
            error_data=event.data if hasattr(event, 'data') else {}
        )
        
        # 如果错误频率过高，可以触发告警
        if self.error_count > 10:
            logger.critical("错误频率过高", error_count=self.error_count)
    
    async def _handle_status_change(self, event: Event, context: EventContext) -> None:
        """处理状态变化事件"""
        logger.info(
            "系统状态变化",
            new_status=event.data.get('status') if hasattr(event, 'data') else None
        )