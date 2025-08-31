"""
事件队列服务 - 基于Redis的高性能事件队列处理
"""
import asyncio
import json
import pickle
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError

from core.logging import get_logger
from models.schemas.event_tracking import CreateEventRequest, EventStatus

logger = get_logger(__name__)


class QueuePriority(str, Enum):
    """队列优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class QueueStatus(str, Enum):
    """队列状态"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class QueuedEvent:
    """队列中的事件"""
    event_id: str
    event_data: Dict[str, Any]
    priority: QueuePriority
    queue_name: str
    queued_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None
    processing_timeout: int = 300  # 5 minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 序列化datetime对象
        data['queued_at'] = self.queued_at.isoformat()
        if self.next_retry_at:
            data['next_retry_at'] = self.next_retry_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueuedEvent':
        """从字典创建"""
        # 反序列化datetime对象
        data['queued_at'] = datetime.fromisoformat(data['queued_at'])
        if data.get('next_retry_at'):
            data['next_retry_at'] = datetime.fromisoformat(data['next_retry_at'])
        return cls(**data)


@dataclass
class QueueMetrics:
    """队列指标"""
    queue_name: str
    priority: QueuePriority
    pending_count: int = 0
    processing_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    retry_count: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_per_minute: float = 0.0
    last_processed_at: Optional[datetime] = None


class EventQueueService:
    """事件队列服务"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_prefix: str = "event_queue",
        max_connections: int = 10
    ):
        self.redis_url = redis_url
        self.queue_prefix = queue_prefix
        self.max_connections = max_connections
        
        # Redis连接池
        self.redis_pool: Optional[aioredis.ConnectionPool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
        # 队列配置
        self.queues: Dict[str, QueueMetrics] = {}
        self.queue_status: Dict[str, QueueStatus] = {}
        self.processors: Dict[str, List[asyncio.Task]] = {}
        
        # 事件处理器
        self.event_handlers: Dict[str, Callable] = {}
        
        # 监控任务
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        self.is_running = False
        
        # 配置参数
        self.batch_size = 100
        self.processing_timeout = 300
        self.retry_delay_seconds = 60
        self.cleanup_interval = 3600  # 1 hour
    
    async def initialize(self):
        """初始化队列服务"""
        if self.is_running:
            return
        
        try:
            # 创建Redis连接池
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=False  # 保持二进制模式以支持pickle
            )
            
            self.redis_client = aioredis.Redis(connection_pool=self.redis_pool)
            
            # 测试连接
            await self.redis_client.ping()
            
            # 初始化默认队列
            await self._initialize_default_queues()
            
            # 启动后台任务
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("Event Queue Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Event Queue Service: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """关闭队列服务"""
        if not self.is_running:
            return
        
        logger.info("Shutting down Event Queue Service")
        self.is_running = False
        
        # 停止所有处理器
        for queue_name, tasks in self.processors.items():
            logger.info(f"Stopping {len(tasks)} processors for queue {queue_name}")
            for task in tasks:
                task.cancel()
        
        # 等待任务完成
        all_tasks = [task for tasks in self.processors.values() for task in tasks]
        if self.monitoring_task:
            all_tasks.append(self.monitoring_task)
        if self.cleanup_task:
            all_tasks.append(self.cleanup_task)
        
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # 关闭Redis连接
        if self.redis_client:
            await self.redis_client.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()
        
        self.processors.clear()
        logger.info("Event Queue Service shutdown completed")
    
    async def _initialize_default_queues(self):
        """初始化默认队列"""
        default_queues = [
            ("events_critical", QueuePriority.CRITICAL),
            ("events_high", QueuePriority.HIGH),
            ("events_normal", QueuePriority.NORMAL),
            ("events_low", QueuePriority.LOW)
        ]
        
        for queue_name, priority in default_queues:
            await self.create_queue(queue_name, priority)
    
    async def create_queue(
        self,
        queue_name: str,
        priority: QueuePriority,
        worker_count: int = 2
    ) -> bool:
        """创建队列"""
        try:
            full_queue_name = f"{self.queue_prefix}:{queue_name}"
            
            # 初始化队列指标
            self.queues[full_queue_name] = QueueMetrics(
                queue_name=full_queue_name,
                priority=priority
            )
            
            # 设置队列状态
            self.queue_status[full_queue_name] = QueueStatus.ACTIVE
            
            # 启动工作器
            if self.is_running:
                await self._start_queue_processors(full_queue_name, worker_count)
            
            logger.info(f"Created queue {full_queue_name} with {worker_count} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create queue {queue_name}: {e}")
            return False
    
    async def enqueue_event(
        self,
        event: CreateEventRequest,
        queue_name: str = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        delay_seconds: int = 0
    ) -> str:
        """将事件加入队列"""
        try:
            # 确定队列名称
            if queue_name is None:
                queue_name = self._get_default_queue_name(priority)
            
            full_queue_name = f"{self.queue_prefix}:{queue_name}"
            
            # 创建队列事件
            queued_event = QueuedEvent(
                event_id=event.event_id,
                event_data=event.dict(),
                priority=priority,
                queue_name=full_queue_name,
                queued_at=utc_now()
            )
            
            # 序列化事件数据
            event_data = pickle.dumps(queued_event.to_dict())
            
            if delay_seconds > 0:
                # 延迟队列
                score = (utc_now() + timedelta(seconds=delay_seconds)).timestamp()
                await self.redis_client.zadd(f"{full_queue_name}:delayed", {event_data: score})
            else:
                # 立即队列
                if priority == QueuePriority.CRITICAL:
                    # 关键事件插入队列头部
                    await self.redis_client.lpush(f"{full_queue_name}:pending", event_data)
                else:
                    # 其他事件插入队列尾部
                    await self.redis_client.rpush(f"{full_queue_name}:pending", event_data)
            
            # 更新指标
            if full_queue_name in self.queues:
                self.queues[full_queue_name].pending_count += 1
            
            logger.debug(f"Enqueued event {event.event_id} to queue {full_queue_name}")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue event {event.event_id}: {e}")
            raise
    
    async def dequeue_events(
        self,
        queue_name: str,
        batch_size: int = None
    ) -> List[QueuedEvent]:
        """从队列中取出事件"""
        try:
            full_queue_name = f"{self.queue_prefix}:{queue_name}"
            batch_size = batch_size or self.batch_size
            
            events = []
            
            # 批量取出事件
            for _ in range(batch_size):
                # 从pending队列中取出事件
                event_data = await self.redis_client.blpop(f"{full_queue_name}:pending", timeout=1)
                
                if not event_data:
                    break
                
                try:
                    # 反序列化事件数据
                    queued_event_dict = pickle.loads(event_data[1])
                    queued_event = QueuedEvent.from_dict(queued_event_dict)
                    
                    # 移动到处理中队列
                    processing_data = pickle.dumps(queued_event.to_dict())
                    await self.redis_client.hset(
                        f"{full_queue_name}:processing",
                        queued_event.event_id,
                        processing_data
                    )
                    
                    events.append(queued_event)
                    
                except Exception as e:
                    logger.error(f"Failed to deserialize queued event: {e}")
                    continue
            
            # 更新指标
            if full_queue_name in self.queues:
                metrics = self.queues[full_queue_name]
                metrics.pending_count = max(0, metrics.pending_count - len(events))
                metrics.processing_count += len(events)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to dequeue events from {queue_name}: {e}")
            return []
    
    async def complete_event(self, event_id: str, queue_name: str, success: bool = True):
        """标记事件完成"""
        try:
            full_queue_name = f"{self.queue_prefix}:{queue_name}"
            
            # 从处理中队列移除
            await self.redis_client.hdel(f"{full_queue_name}:processing", event_id)
            
            # 更新指标
            if full_queue_name in self.queues:
                metrics = self.queues[full_queue_name]
                metrics.processing_count = max(0, metrics.processing_count - 1)
                
                if success:
                    metrics.completed_count += 1
                else:
                    metrics.failed_count += 1
                
                metrics.last_processed_at = utc_now()
            
            logger.debug(f"Completed event {event_id} in queue {full_queue_name}, success: {success}")
            
        except Exception as e:
            logger.error(f"Failed to complete event {event_id}: {e}")
    
    async def retry_event(self, event_id: str, queue_name: str, delay_seconds: int = None):
        """重试事件"""
        try:
            full_queue_name = f"{self.queue_prefix}:{queue_name}"
            delay_seconds = delay_seconds or self.retry_delay_seconds
            
            # 从处理中队列获取事件
            event_data = await self.redis_client.hget(f"{full_queue_name}:processing", event_id)
            
            if not event_data:
                logger.warning(f"Event {event_id} not found in processing queue")
                return
            
            # 反序列化事件
            queued_event_dict = pickle.loads(event_data)
            queued_event = QueuedEvent.from_dict(queued_event_dict)
            
            # 增加重试次数
            queued_event.retry_count += 1
            queued_event.next_retry_at = utc_now() + timedelta(seconds=delay_seconds)
            
            if queued_event.retry_count >= queued_event.max_retries:
                # 超过最大重试次数，移到失败队列
                failed_data = pickle.dumps(queued_event.to_dict())
                await self.redis_client.hset(f"{full_queue_name}:failed", event_id, failed_data)
                await self.redis_client.hdel(f"{full_queue_name}:processing", event_id)
                
                # 更新指标
                if full_queue_name in self.queues:
                    metrics = self.queues[full_queue_name]
                    metrics.processing_count = max(0, metrics.processing_count - 1)
                    metrics.failed_count += 1
                
                logger.warning(f"Event {event_id} failed after {queued_event.retry_count} retries")
            else:
                # 重新入队，延迟处理
                retry_data = pickle.dumps(queued_event.to_dict())
                score = queued_event.next_retry_at.timestamp()
                await self.redis_client.zadd(f"{full_queue_name}:delayed", {retry_data: score})
                await self.redis_client.hdel(f"{full_queue_name}:processing", event_id)
                
                # 更新指标
                if full_queue_name in self.queues:
                    metrics = self.queues[full_queue_name]
                    metrics.processing_count = max(0, metrics.processing_count - 1)
                    metrics.retry_count += 1
                
                logger.info(f"Event {event_id} scheduled for retry #{queued_event.retry_count} after {delay_seconds}s")
            
        except Exception as e:
            logger.error(f"Failed to retry event {event_id}: {e}")
    
    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.event_handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    async def _start_background_tasks(self):
        """启动后台任务"""
        # 延迟队列监控任务
        self.monitoring_task = asyncio.create_task(self._delayed_queue_monitor())
        
        # 清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        # 为每个队列启动处理器
        for queue_name in self.queues:
            await self._start_queue_processors(queue_name, 2)
    
    async def _start_queue_processors(self, queue_name: str, worker_count: int):
        """启动队列处理器"""
        if queue_name not in self.processors:
            self.processors[queue_name] = []
        
        for i in range(worker_count):
            processor_task = asyncio.create_task(
                self._queue_processor_worker(queue_name, f"worker-{i}")
            )
            self.processors[queue_name].append(processor_task)
        
        logger.info(f"Started {worker_count} processors for queue {queue_name}")
    
    async def _queue_processor_worker(self, queue_name: str, worker_id: str):
        """队列处理器工作线程"""
        logger.info(f"Queue processor {worker_id} started for {queue_name}")
        
        while self.is_running and self.queue_status.get(queue_name) == QueueStatus.ACTIVE:
            try:
                # 获取事件批次
                events = await self.dequeue_events(queue_name.split(':')[-1], 10)
                
                if not events:
                    await asyncio.sleep(1)
                    continue
                
                # 处理事件
                for event in events:
                    try:
                        await self._process_single_event(event)
                        await self.complete_event(event.event_id, queue_name, True)
                    except Exception as e:
                        logger.error(f"Error processing event {event.event_id}: {e}")
                        await self.retry_event(event.event_id, queue_name)
                
            except Exception as e:
                logger.error(f"Error in queue processor {worker_id} for {queue_name}: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Queue processor {worker_id} stopped for {queue_name}")
    
    async def _process_single_event(self, queued_event: QueuedEvent):
        """处理单个事件"""
        # 重建事件对象
        event_dict = queued_event.event_data
        
        # 根据事件类型找到处理器
        event_type = event_dict.get('event_type')
        
        if event_type in self.event_handlers:
            handler = self.event_handlers[event_type]
            await handler(event_dict)
        else:
            # 默认处理逻辑
            logger.debug(f"Processing event {queued_event.event_id} of type {event_type}")
            # 模拟处理时间
            await asyncio.sleep(0.1)
    
    async def _delayed_queue_monitor(self):
        """延迟队列监控器"""
        while self.is_running:
            try:
                current_time = utc_now().timestamp()
                
                for queue_name in self.queues:
                    # 检查延迟队列中到期的事件
                    delayed_key = f"{queue_name}:delayed"
                    
                    # 获取到期的事件
                    expired_events = await self.redis_client.zrangebyscore(
                        delayed_key, 0, current_time, withscores=True
                    )
                    
                    if expired_events:
                        # 移动到pending队列
                        for event_data, _ in expired_events:
                            await self.redis_client.rpush(f"{queue_name}:pending", event_data)
                        
                        # 从延迟队列中移除
                        await self.redis_client.zremrangebyscore(delayed_key, 0, current_time)
                        
                        logger.debug(f"Moved {len(expired_events)} events from delayed to pending in {queue_name}")
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                logger.error(f"Error in delayed queue monitor: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_worker(self):
        """清理工作器"""
        while self.is_running:
            try:
                # 清理超时的处理中事件
                current_time = utc_now()
                
                for queue_name in self.queues:
                    processing_key = f"{queue_name}:processing"
                    
                    # 获取所有处理中的事件
                    processing_events = await self.redis_client.hgetall(processing_key)
                    
                    for event_id, event_data in processing_events.items():
                        try:
                            queued_event_dict = pickle.loads(event_data)
                            queued_event = QueuedEvent.from_dict(queued_event_dict)
                            
                            # 检查是否超时
                            processing_duration = (current_time - queued_event.queued_at).total_seconds()
                            
                            if processing_duration > self.processing_timeout:
                                logger.warning(f"Event {event_id} timeout, moving to retry")
                                await self.retry_event(event_id.decode(), queue_name)
                        
                        except Exception as e:
                            logger.error(f"Error processing cleanup for event {event_id}: {e}")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(300)
    
    def _get_default_queue_name(self, priority: QueuePriority) -> str:
        """获取默认队列名称"""
        queue_map = {
            QueuePriority.CRITICAL: "events_critical",
            QueuePriority.HIGH: "events_high",
            QueuePriority.NORMAL: "events_normal",
            QueuePriority.LOW: "events_low"
        }
        return queue_map.get(priority, "events_normal")
    
    async def get_queue_metrics(self, queue_name: str = None) -> Union[QueueMetrics, Dict[str, QueueMetrics]]:
        """获取队列指标"""
        if queue_name:
            full_queue_name = f"{self.queue_prefix}:{queue_name}"
            return self.queues.get(full_queue_name)
        else:
            return {name: metrics for name, metrics in self.queues.items()}
    
    async def pause_queue(self, queue_name: str):
        """暂停队列"""
        full_queue_name = f"{self.queue_prefix}:{queue_name}"
        self.queue_status[full_queue_name] = QueueStatus.PAUSED
        logger.info(f"Queue {full_queue_name} paused")
    
    async def resume_queue(self, queue_name: str):
        """恢复队列"""
        full_queue_name = f"{self.queue_prefix}:{queue_name}"
        self.queue_status[full_queue_name] = QueueStatus.ACTIVE
        logger.info(f"Queue {full_queue_name} resumed")
    
    async def clear_queue(self, queue_name: str, include_processing: bool = False):
        """清空队列"""
        try:
            full_queue_name = f"{self.queue_prefix}:{queue_name}"
            
            # 清空pending队列
            await self.redis_client.delete(f"{full_queue_name}:pending")
            
            # 清空延迟队列
            await self.redis_client.delete(f"{full_queue_name}:delayed")
            
            if include_processing:
                # 清空处理中队列
                await self.redis_client.delete(f"{full_queue_name}:processing")
            
            # 重置指标
            if full_queue_name in self.queues:
                metrics = self.queues[full_queue_name]
                metrics.pending_count = 0
                if include_processing:
                    metrics.processing_count = 0
            
            logger.info(f"Cleared queue {full_queue_name}")
            
        except Exception as e:
            logger.error(f"Failed to clear queue {queue_name}: {e}")
            raise