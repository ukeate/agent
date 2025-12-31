"""
事件缓冲服务 - 高性能事件批量处理和缓冲机制
"""

import asyncio
import json
import time
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
from src.models.schemas.event_tracking import CreateEventRequest, EventStatus, DataQuality
from src.repositories.event_tracking_repository import EventStreamRepository
from src.services.event_processing_service import EventProcessingService

from src.core.logging import get_logger
logger = get_logger(__name__)

class BufferStrategy(str, Enum):
    """缓冲策略"""
    SIZE_BASED = "size_based"         # 基于大小的缓冲
    TIME_BASED = "time_based"         # 基于时间的缓冲
    HYBRID = "hybrid"                 # 混合策略
    PRIORITY_BASED = "priority_based" # 基于优先级的缓冲

class BufferPriority(str, Enum):
    """缓冲优先级"""
    CRITICAL = "critical"    # 关键事件，立即处理
    HIGH = "high"           # 高优先级，快速处理
    NORMAL = "normal"       # 正常优先级
    LOW = "low"             # 低优先级，可延迟处理

@dataclass
class BufferConfig:
    """缓冲配置"""
    strategy: BufferStrategy = BufferStrategy.HYBRID
    max_buffer_size: int = 1000          # 最大缓冲区大小
    flush_interval_seconds: int = 30      # 刷新间隔（秒）
    max_batch_size: int = 100            # 单次批处理最大大小
    max_retry_attempts: int = 3          # 最大重试次数
    retry_delay_seconds: int = 5         # 重试延迟（秒）
    
    # 优先级配置
    critical_flush_threshold: int = 1    # 关键事件立即刷新阈值
    high_priority_flush_interval: int = 5   # 高优先级刷新间隔
    
    # 内存管理
    max_memory_usage_mb: int = 512       # 最大内存使用量（MB）
    memory_check_interval: int = 60      # 内存检查间隔（秒）
    
    # 持久化配置
    enable_persistence: bool = True       # 启用持久化
    persistence_interval: int = 300       # 持久化间隔（秒）

@dataclass
class BufferedEvent:
    """缓冲事件"""
    event: CreateEventRequest
    priority: BufferPriority = BufferPriority.NORMAL
    buffered_at: datetime = field(default_factory=lambda: utc_now())
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None
    batch_id: Optional[str] = None
    partition_key: Optional[str] = None  # 用于分区处理

@dataclass
class BufferMetrics:
    """缓冲指标"""
    total_buffered: int = 0
    total_flushed: int = 0
    total_failed: int = 0
    total_retries: int = 0
    buffer_size_current: int = 0
    avg_buffer_time_ms: float = 0.0
    avg_batch_size: float = 0.0
    memory_usage_mb: float = 0.0
    last_flush_at: Optional[datetime] = None
    
    # 按优先级统计
    priority_stats: Dict[BufferPriority, int] = field(default_factory=dict)

class EventBuffer:
    """事件缓冲区"""
    
    def __init__(self, config: BufferConfig, partition_key: str = "default"):
        self.config = config
        self.partition_key = partition_key
        self.events: deque[BufferedEvent] = deque()
        self.priority_queues: Dict[BufferPriority, deque[BufferedEvent]] = {
            priority: deque() for priority in BufferPriority
        }
        self.lock = threading.RLock()
        self.last_flush_time = utc_now()
        self.metrics = BufferMetrics()
    
    def add_event(self, event: CreateEventRequest, priority: BufferPriority = BufferPriority.NORMAL) -> bool:
        """添加事件到缓冲区"""
        with self.lock:
            # 检查缓冲区大小限制
            if len(self.events) >= self.config.max_buffer_size:
                logger.warning(f"Buffer {self.partition_key} is full, dropping event")
                return False
            
            buffered_event = BufferedEvent(
                event=event,
                priority=priority,
                partition_key=self.partition_key
            )
            
            # 添加到主队列和优先级队列
            self.events.append(buffered_event)
            self.priority_queues[priority].append(buffered_event)
            
            # 更新指标
            self.metrics.total_buffered += 1
            self.metrics.buffer_size_current = len(self.events)
            self.metrics.priority_stats[priority] = self.metrics.priority_stats.get(priority, 0) + 1
            
            return True
    
    def should_flush(self) -> bool:
        """检查是否应该刷新缓冲区"""
        with self.lock:
            now = utc_now()
            
            # 关键事件立即刷新
            if len(self.priority_queues[BufferPriority.CRITICAL]) >= self.config.critical_flush_threshold:
                return True
            
            # 高优先级事件快速刷新
            if (len(self.priority_queues[BufferPriority.HIGH]) > 0 and 
                (now - self.last_flush_time).total_seconds() >= self.config.high_priority_flush_interval):
                return True
            
            # 基于策略的刷新判断
            if self.config.strategy == BufferStrategy.SIZE_BASED:
                return len(self.events) >= self.config.max_batch_size
            
            elif self.config.strategy == BufferStrategy.TIME_BASED:
                return (now - self.last_flush_time).total_seconds() >= self.config.flush_interval_seconds
            
            elif self.config.strategy == BufferStrategy.HYBRID:
                return (len(self.events) >= self.config.max_batch_size or 
                       (now - self.last_flush_time).total_seconds() >= self.config.flush_interval_seconds)
            
            elif self.config.strategy == BufferStrategy.PRIORITY_BASED:
                # 优先级策略：根据优先级和时间决定
                total_high_priority = (len(self.priority_queues[BufferPriority.CRITICAL]) + 
                                     len(self.priority_queues[BufferPriority.HIGH]))
                if total_high_priority > 0:
                    return True
                return (now - self.last_flush_time).total_seconds() >= self.config.flush_interval_seconds
            
            return False
    
    def get_flush_batch(self) -> List[BufferedEvent]:
        """获取要刷新的事件批次"""
        with self.lock:
            if not self.events:
                return []
            
            batch_size = min(len(self.events), self.config.max_batch_size)
            batch = []
            
            # 优先处理关键和高优先级事件
            for priority in [BufferPriority.CRITICAL, BufferPriority.HIGH, BufferPriority.NORMAL, BufferPriority.LOW]:
                while len(batch) < batch_size and self.priority_queues[priority]:
                    event = self.priority_queues[priority].popleft()
                    batch.append(event)
                    self.events.remove(event)  # 从主队列中移除
            
            # 更新指标
            self.metrics.buffer_size_current = len(self.events)
            self.metrics.last_flush_at = utc_now()
            self.last_flush_time = utc_now()
            
            if batch:
                avg_buffer_time = sum(
                    (utc_now() - event.buffered_at).total_seconds() * 1000
                    for event in batch
                ) / len(batch)
                self.metrics.avg_buffer_time_ms = avg_buffer_time
                self.metrics.avg_batch_size = (self.metrics.avg_batch_size + len(batch)) / 2
            
            return batch
    
    def return_failed_events(self, failed_events: List[BufferedEvent]):
        """将失败的事件重新加入缓冲区"""
        with self.lock:
            for event in failed_events:
                event.retry_count += 1
                event.last_retry_at = utc_now()
                
                if event.retry_count <= self.config.max_retry_attempts:
                    # 重新加入缓冲区（降低优先级）
                    new_priority = BufferPriority.LOW if event.priority == BufferPriority.NORMAL else event.priority
                    self.events.append(event)
                    self.priority_queues[new_priority].append(event)
                    self.metrics.total_retries += 1
                else:
                    # 超过重试次数，记录失败
                    self.metrics.total_failed += 1
                    logger.error(f"Event {event.event.event_id} failed after {event.retry_count} retries")

class EventBufferService:
    """事件缓冲服务"""
    
    def __init__(
        self,
        config: BufferConfig,
        processing_service: EventProcessingService,
        persistence_storage: Optional[Any] = None
    ):
        self.config = config
        self.processing_service = processing_service
        self.persistence_storage = persistence_storage
        
        # 分区缓冲区
        self.buffers: Dict[str, EventBuffer] = {}
        self.buffer_lock = threading.RLock()
        
        # 后台任务
        self.flush_task: Optional[asyncio.Task] = None
        self.memory_monitor_task: Optional[asyncio.Task] = None
        self.persistence_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # 线程池用于处理
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="EventBuffer")
        
        # 全局指标
        self.global_metrics = BufferMetrics()
        
        # 回调函数
        self.flush_callbacks: List[Callable[[List[BufferedEvent]], None]] = []
    
    def start(self):
        """启动缓冲服务"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting Event Buffer Service")
        
        # 启动后台任务
        loop = asyncio.get_running_loop()
        
        self.flush_task = loop.create_task(self._flush_worker())
        self.memory_monitor_task = loop.create_task(self._memory_monitor())
        
        if self.config.enable_persistence:
            self.persistence_task = loop.create_task(self._persistence_worker())
    
    def stop(self):
        """停止缓冲服务"""
        if not self.is_running:
            return
        
        logger.info("Stopping Event Buffer Service")
        self.is_running = False
        
        # 取消后台任务
        if self.flush_task:
            self.flush_task.cancel()
        if self.memory_monitor_task:
            self.memory_monitor_task.cancel()
        if self.persistence_task:
            self.persistence_task.cancel()
        
        # 强制刷新所有缓冲区
        self._force_flush_all_buffers()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
    
    def buffer_event(self, event: CreateEventRequest, partition_key: str = None, 
                    priority: BufferPriority = BufferPriority.NORMAL) -> bool:
        """缓冲事件"""
        if not self.is_running:
            logger.warning("Buffer service is not running")
            return False
        
        # 确定分区键
        if partition_key is None:
            partition_key = self._determine_partition_key(event)
        
        # 获取或创建缓冲区
        buffer = self._get_or_create_buffer(partition_key)
        
        # 添加事件到缓冲区
        success = buffer.add_event(event, priority)
        
        # 更新全局指标
        if success:
            self.global_metrics.total_buffered += 1
            self.global_metrics.buffer_size_current = sum(
                buf.metrics.buffer_size_current for buf in self.buffers.values()
            )
        
        # 如果是关键事件，触发立即刷新
        if priority == BufferPriority.CRITICAL:
            create_task_with_logging(self._flush_buffer(buffer))
        
        return success
    
    def _determine_partition_key(self, event: CreateEventRequest) -> str:
        """确定分区键"""
        # 基于实验ID和事件类型分区，提高处理效率
        return f"{event.experiment_id}_{event.event_type.value}"
    
    def _get_or_create_buffer(self, partition_key: str) -> EventBuffer:
        """获取或创建缓冲区"""
        with self.buffer_lock:
            if partition_key not in self.buffers:
                self.buffers[partition_key] = EventBuffer(self.config, partition_key)
                logger.debug(f"Created new buffer for partition: {partition_key}")
            return self.buffers[partition_key]
    
    async def _flush_worker(self):
        """刷新工作器"""
        while self.is_running:
            try:
                await self._check_and_flush_buffers()
                await asyncio.sleep(1)  # 每秒检查一次
            except Exception as e:
                logger.error(f"Error in flush worker: {e}", exc_info=True)
                await asyncio.sleep(5)  # 错误时延迟5秒
    
    async def _check_and_flush_buffers(self):
        """检查和刷新缓冲区"""
        buffers_to_flush = []
        
        with self.buffer_lock:
            for partition_key, buffer in self.buffers.items():
                if buffer.should_flush():
                    buffers_to_flush.append(buffer)
        
        # 并发刷新缓冲区
        if buffers_to_flush:
            flush_tasks = [self._flush_buffer(buffer) for buffer in buffers_to_flush]
            await asyncio.gather(*flush_tasks, return_exceptions=True)
    
    async def _flush_buffer(self, buffer: EventBuffer):
        """刷新单个缓冲区"""
        try:
            batch = buffer.get_flush_batch()
            if not batch:
                return
            
            logger.debug(f"Flushing {len(batch)} events from buffer {buffer.partition_key}")
            
            # 异步处理批次
            await asyncio.get_running_loop().run_in_executor(
                self.executor, self._process_batch, batch
            )
            
            # 更新全局指标
            self.global_metrics.total_flushed += len(batch)
            self.global_metrics.buffer_size_current = sum(
                buf.metrics.buffer_size_current for buf in self.buffers.values()
            )
            
            # 调用回调函数
            for callback in self.flush_callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    logger.error(f"Error in flush callback: {e}")
            
        except Exception as e:
            logger.error(f"Error flushing buffer {buffer.partition_key}: {e}", exc_info=True)
    
    def _process_batch(self, batch: List[BufferedEvent]):
        """处理事件批次（同步方法）"""
        failed_events = []
        
        for buffered_event in batch:
            try:
                # 这里可以调用处理服务的同步版本或使用其他处理逻辑
                # 由于是在线程池中执行，可以使用同步调用
                success = self._process_single_event(buffered_event.event)
                
                if not success:
                    failed_events.append(buffered_event)
            
            except Exception as e:
                logger.error(f"Error processing event {buffered_event.event.event_id}: {e}")
                failed_events.append(buffered_event)
        
        # 处理失败的事件
        if failed_events:
            # 找到对应的缓冲区并重新加入失败事件
            buffer_groups = defaultdict(list)
            for event in failed_events:
                buffer_groups[event.partition_key].append(event)
            
            with self.buffer_lock:
                for partition_key, events in buffer_groups.items():
                    if partition_key in self.buffers:
                        self.buffers[partition_key].return_failed_events(events)
    
    def _process_single_event(self, event: CreateEventRequest) -> bool:
        """处理单个事件（同步版本）"""
        try:
            # 这里可以实现简化的同步处理逻辑
            # 或者使用 asyncio.run() 调用异步处理服务
            # 为简化实现，这里假设处理总是成功
            return True
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            return False
    
    async def _memory_monitor(self):
        """内存监控器"""
        import psutil
        
        while self.is_running:
            try:
                # 获取当前进程内存使用情况
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 更新指标
                self.global_metrics.memory_usage_mb = memory_mb
                
                # 如果内存使用超过限制，强制刷新缓冲区
                if memory_mb > self.config.max_memory_usage_mb:
                    logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.config.max_memory_usage_mb}MB, forcing flush")
                    await self._force_flush_oldest_buffers()
                
                await asyncio.sleep(self.config.memory_check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
                await asyncio.sleep(60)  # 错误时延迟1分钟
    
    async def _force_flush_oldest_buffers(self):
        """强制刷新最旧的缓冲区"""
        with self.buffer_lock:
            if not self.buffers:
                return
            
            # 按最后刷新时间排序，刷新最旧的缓冲区
            sorted_buffers = sorted(
                self.buffers.values(),
                key=lambda b: b.last_flush_time
            )
            
            # 刷新最旧的一半缓冲区
            buffers_to_flush = sorted_buffers[:len(sorted_buffers) // 2 + 1]
            
        # 异步刷新
        flush_tasks = [self._flush_buffer(buffer) for buffer in buffers_to_flush]
        await asyncio.gather(*flush_tasks, return_exceptions=True)
    
    def _force_flush_all_buffers(self):
        """强制刷新所有缓冲区"""
        logger.info("Force flushing all buffers")
        
        with self.buffer_lock:
            for buffer in self.buffers.values():
                batch = buffer.get_flush_batch()
                if batch:
                    self._process_batch(batch)
    
    async def _persistence_worker(self):
        """持久化工作器"""
        while self.is_running:
            try:
                if self.persistence_storage:
                    await self._persist_buffer_state()
                await asyncio.sleep(self.config.persistence_interval)
            except Exception as e:
                logger.error(f"Error in persistence worker: {e}")
                await asyncio.sleep(60)
    
    async def _persist_buffer_state(self):
        """持久化缓冲区状态"""
        try:
            if not self.persistence_storage:
                return

            with self.buffer_lock:
                buffers_snapshot = {}
                for partition_key, buffer in self.buffers.items():
                    with buffer.lock:
                        events = [
                            {
                                "event": be.event.model_dump(),
                                "priority": be.priority.value,
                                "buffered_at": be.buffered_at.isoformat(),
                                "retry_count": be.retry_count,
                                "last_retry_at": be.last_retry_at.isoformat() if be.last_retry_at else None,
                                "batch_id": be.batch_id,
                                "partition_key": be.partition_key,
                            }
                            for be in list(buffer.events)
                        ]

                    buffers_snapshot[partition_key] = {
                        "partition_key": partition_key,
                        "last_flush_time": buffer.last_flush_time.isoformat(),
                        "metrics": {
                            "total_buffered": buffer.metrics.total_buffered,
                            "total_flushed": buffer.metrics.total_flushed,
                            "total_failed": buffer.metrics.total_failed,
                            "total_retries": buffer.metrics.total_retries,
                            "buffer_size_current": buffer.metrics.buffer_size_current,
                            "avg_buffer_time_ms": buffer.metrics.avg_buffer_time_ms,
                            "avg_batch_size": buffer.metrics.avg_batch_size,
                            "memory_usage_mb": buffer.metrics.memory_usage_mb,
                            "last_flush_at": buffer.metrics.last_flush_at.isoformat() if buffer.metrics.last_flush_at else None,
                            "priority_stats": {k.value: v for k, v in buffer.metrics.priority_stats.items()},
                        },
                        "events": events,
                    }

                state = {
                    "persisted_at": utc_now().isoformat(),
                    "config": {
                        "strategy": self.config.strategy.value,
                        "max_buffer_size": self.config.max_buffer_size,
                        "flush_interval_seconds": self.config.flush_interval_seconds,
                        "max_batch_size": self.config.max_batch_size,
                        "max_retry_attempts": self.config.max_retry_attempts,
                        "retry_delay_seconds": self.config.retry_delay_seconds,
                    },
                    "global_metrics": {
                        "total_buffered": self.global_metrics.total_buffered,
                        "total_flushed": self.global_metrics.total_flushed,
                        "total_failed": self.global_metrics.total_failed,
                        "total_retries": self.global_metrics.total_retries,
                        "buffer_size_current": self.global_metrics.buffer_size_current,
                        "avg_buffer_time_ms": self.global_metrics.avg_buffer_time_ms,
                        "avg_batch_size": self.global_metrics.avg_batch_size,
                        "memory_usage_mb": self.global_metrics.memory_usage_mb,
                        "last_flush_at": self.global_metrics.last_flush_at.isoformat() if self.global_metrics.last_flush_at else None,
                        "priority_stats": {k.value: v for k, v in self.global_metrics.priority_stats.items()},
                    },
                    "buffers": buffers_snapshot,
                }

            payload = json.dumps(state, ensure_ascii=False)
            ttl = max(self.config.persistence_interval * 3, 3600)
            await self.persistence_storage.setex("event_buffer:state", ttl, payload)
        except Exception as e:
            logger.error(f"持久化缓冲区状态失败: {e}")
    
    def add_flush_callback(self, callback: Callable[[List[BufferedEvent]], None]):
        """添加刷新回调"""
        self.flush_callbacks.append(callback)
    
    def remove_flush_callback(self, callback: Callable[[List[BufferedEvent]], None]):
        """移除刷新回调"""
        if callback in self.flush_callbacks:
            self.flush_callbacks.remove(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取缓冲服务指标"""
        buffer_metrics = {}
        
        with self.buffer_lock:
            for partition_key, buffer in self.buffers.items():
                buffer_metrics[partition_key] = {
                    'buffer_size': buffer.metrics.buffer_size_current,
                    'total_buffered': buffer.metrics.total_buffered,
                    'total_flushed': buffer.metrics.total_flushed,
                    'total_failed': buffer.metrics.total_failed,
                    'avg_buffer_time_ms': buffer.metrics.avg_buffer_time_ms,
                    'last_flush_at': buffer.metrics.last_flush_at.isoformat() if buffer.metrics.last_flush_at else None,
                    'priority_stats': {k.value: v for k, v in buffer.metrics.priority_stats.items()}
                }
        
        return {
            'global_metrics': {
                'total_buffered': self.global_metrics.total_buffered,
                'total_flushed': self.global_metrics.total_flushed,
                'total_failed': self.global_metrics.total_failed,
                'buffer_size_current': self.global_metrics.buffer_size_current,
                'memory_usage_mb': self.global_metrics.memory_usage_mb,
                'active_buffers': len(self.buffers)
            },
            'buffer_metrics': buffer_metrics,
            'config': {
                'strategy': self.config.strategy.value,
                'max_buffer_size': self.config.max_buffer_size,
                'flush_interval_seconds': self.config.flush_interval_seconds,
                'max_batch_size': self.config.max_batch_size
            }
        }
