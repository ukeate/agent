"""
事件批处理管理器 - 协调缓冲、队列和处理流程
"""

import asyncio
import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import uuid
from src.core.database import get_db_session
from src.models.schemas.event_tracking import CreateEventRequest, BatchEventsRequest, EventStatus
from src.repositories.event_tracking_repository import EventStreamRepository, EventDeduplicationRepository, EventSchemaRepository, EventErrorRepository
from src.services.event_processing_service import EventProcessingService
from src.services.event_buffer_service import EventBufferService, BufferConfig, BufferPriority, BufferedEvent

from src.core.logging import get_logger
logger = get_logger(__name__)

class ProcessingMode(str, Enum):
    """处理模式"""
    IMMEDIATE = "immediate"    # 立即处理
    BUFFERED = "buffered"      # 缓冲处理
    QUEUED = "queued"          # 队列处理
    ADAPTIVE = "adaptive"      # 自适应处理

@dataclass
class BatchJobConfig:
    """批处理任务配置"""
    job_id: str
    experiment_ids: List[str]
    processing_mode: ProcessingMode = ProcessingMode.BUFFERED
    priority: BufferPriority = BufferPriority.NORMAL
    timeout_seconds: int = 300
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()

@dataclass
class BatchJobStatus:
    """批处理任务状态"""
    job_id: str
    status: EventStatus
    total_events: int = 0
    processed_events: int = 0
    failed_events: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0

class EventBatchManager:
    """事件批处理管理器"""
    
    def __init__(self):
        self.processing_service: Optional[EventProcessingService] = None
        self.buffer_service: Optional[EventBufferService] = None
        self.active_jobs: Dict[str, BatchJobStatus] = {}
        self.job_callbacks: Dict[str, List[Callable]] = {}
        self.is_initialized = False
        
        # 性能配置
        self.max_concurrent_jobs = 10
        self.job_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        # 统计信息
        self.stats = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'total_events_processed': 0,
            'avg_processing_time_ms': 0.0
        }
    
    async def initialize(self):
        """初始化批处理管理器"""
        if self.is_initialized:
            return
        
        try:
            # 获取数据库会话
            async with get_db_session() as db:
                # 初始化repositories
                event_repo = EventStreamRepository(db)
                dedup_repo = EventDeduplicationRepository(db)
                schema_repo = EventSchemaRepository(db)
                error_repo = EventErrorRepository(db)
                
                # 初始化处理服务
                self.processing_service = EventProcessingService(
                    event_repo, dedup_repo, schema_repo, error_repo
                )
                
                # 初始化缓冲服务
                buffer_config = BufferConfig(
                    max_buffer_size=5000,
                    flush_interval_seconds=10,
                    max_batch_size=500,
                    max_retry_attempts=3
                )
                
                self.buffer_service = EventBufferService(
                    config=buffer_config,
                    processing_service=self.processing_service
                )
                
                # 启动缓冲服务
                self.buffer_service.start()
                
                # 添加回调函数
                self.buffer_service.add_flush_callback(self._on_batch_flushed)
                
                self.is_initialized = True
                logger.info("Event Batch Manager initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize Event Batch Manager: {e}", exc_info=True)
            raise
    
    async def shutdown(self):
        """关闭批处理管理器"""
        if not self.is_initialized:
            return
        
        logger.info("Shutting down Event Batch Manager")
        
        # 等待所有活跃任务完成
        if self.active_jobs:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete")
            await self._wait_for_active_jobs()
        
        # 停止缓冲服务
        if self.buffer_service:
            self.buffer_service.stop()
        
        self.is_initialized = False
    
    async def submit_event(
        self,
        event: CreateEventRequest,
        processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE,
        priority: BufferPriority = BufferPriority.NORMAL
    ) -> str:
        """提交单个事件处理"""
        if not self.is_initialized:
            await self.initialize()
        
        # 根据处理模式决定处理方式
        if processing_mode == ProcessingMode.IMMEDIATE:
            return await self._process_immediate(event)
        
        elif processing_mode == ProcessingMode.BUFFERED:
            return await self._process_buffered(event, priority)
        
        elif processing_mode == ProcessingMode.ADAPTIVE:
            return await self._process_adaptive(event, priority)
        
        else:
            raise ValueError(f"Unsupported processing mode: {processing_mode}")
    
    async def submit_batch(
        self,
        batch: BatchEventsRequest,
        config: BatchJobConfig = None
    ) -> str:
        """提交批量事件处理"""
        if not self.is_initialized:
            await self.initialize()
        
        job_id = config.job_id if config else str(uuid.uuid4())
        
        # 创建任务状态
        job_status = BatchJobStatus(
            job_id=job_id,
            status=EventStatus.PENDING,
            total_events=len(batch.events),
            started_at=utc_now()
        )
        
        self.active_jobs[job_id] = job_status
        self.stats['total_jobs'] += 1
        
        # 异步处理批次
        create_task_with_logging(self._process_batch_job(batch, config or BatchJobConfig(job_id=job_id)))
        
        return job_id
    
    async def _process_immediate(self, event: CreateEventRequest) -> str:
        """立即处理事件"""
        try:
            result = await self.processing_service.process_event(event)
            return result.event_id
        except Exception as e:
            logger.error(f"Immediate processing failed for event {event.event_id}: {e}")
            raise
    
    async def _process_buffered(self, event: CreateEventRequest, priority: BufferPriority) -> str:
        """缓冲处理事件"""
        success = self.buffer_service.buffer_event(event, priority=priority)
        if not success:
            raise RuntimeError("Failed to buffer event - buffer may be full")
        return event.event_id
    
    async def _process_adaptive(self, event: CreateEventRequest, priority: BufferPriority) -> str:
        """自适应处理事件"""
        # 获取系统负载信息
        buffer_metrics = self.buffer_service.get_metrics()
        current_buffer_size = buffer_metrics['global_metrics']['buffer_size_current']
        
        # 自适应决策逻辑
        if priority == BufferPriority.CRITICAL:
            # 关键事件始终立即处理
            return await self._process_immediate(event)
        
        elif current_buffer_size < 100:  # 低负载
            # 缓冲处理以提高批处理效率
            return await self._process_buffered(event, priority)
        
        elif current_buffer_size > 1000:  # 高负载
            # 立即处理以避免延迟
            return await self._process_immediate(event)
        
        else:  # 中等负载
            # 根据优先级决定
            if priority == BufferPriority.HIGH:
                return await self._process_immediate(event)
            else:
                return await self._process_buffered(event, priority)
    
    async def _process_batch_job(self, batch: BatchEventsRequest, config: BatchJobConfig):
        """处理批量任务"""
        async with self.job_semaphore:
            job_status = self.active_jobs[config.job_id]
            
            try:
                job_status.status = EventStatus.PROCESSED
                start_time = utc_now()
                
                # 根据配置选择处理模式
                if config.processing_mode == ProcessingMode.IMMEDIATE:
                    await self._process_batch_immediate(batch, job_status)
                else:
                    await self._process_batch_buffered(batch, job_status, config.priority)
                
                # 完成任务
                job_status.status = EventStatus.PROCESSED
                job_status.completed_at = utc_now()
                job_status.progress_percentage = 100.0
                
                # 更新统计信息
                processing_time = (job_status.completed_at - start_time).total_seconds() * 1000
                self.stats['successful_jobs'] += 1
                self.stats['total_events_processed'] += job_status.processed_events
                self._update_avg_processing_time(processing_time)
                
                logger.info(f"Batch job {config.job_id} completed successfully: {job_status.processed_events}/{job_status.total_events} events processed")
                
                # 调用回调函数
                await self._call_job_callbacks(config.job_id, job_status)
                
            except Exception as e:
                job_status.status = EventStatus.FAILED
                job_status.error_message = str(e)
                job_status.completed_at = utc_now()
                self.stats['failed_jobs'] += 1
                
                logger.error(f"Batch job {config.job_id} failed: {e}", exc_info=True)
                
                # 调用回调函数
                await self._call_job_callbacks(config.job_id, job_status)
            
            finally:
                # 清理活跃任务（延迟清理以便查询状态）
                create_task_with_logging(self._cleanup_job(config.job_id, delay=300))
    
    async def _process_batch_immediate(self, batch: BatchEventsRequest, job_status: BatchJobStatus):
        """立即批量处理"""
        results = await self.processing_service.process_events_batch(batch.events)
        
        for result in results:
            if result.success:
                job_status.processed_events += 1
            else:
                job_status.failed_events += 1
            
            # 更新进度
            job_status.progress_percentage = (job_status.processed_events + job_status.failed_events) / job_status.total_events * 100
    
    async def _process_batch_buffered(self, batch: BatchEventsRequest, job_status: BatchJobStatus, priority: BufferPriority):
        """缓冲批量处理"""
        for event in batch.events:
            success = self.buffer_service.buffer_event(event, priority=priority)
            if success:
                job_status.processed_events += 1
            else:
                job_status.failed_events += 1
            
            # 更新进度
            job_status.progress_percentage = (job_status.processed_events + job_status.failed_events) / job_status.total_events * 100
    
    def _on_batch_flushed(self, flushed_events: List[BufferedEvent]):
        """缓冲区刷新回调"""
        logger.debug(f"Batch flushed: {len(flushed_events)} events")
        # 这里可以添加额外的后处理逻辑
    
    async def _call_job_callbacks(self, job_id: str, job_status: BatchJobStatus):
        """调用任务回调函数"""
        if job_id in self.job_callbacks:
            for callback in self.job_callbacks[job_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(job_status)
                    else:
                        callback(job_status)
                except Exception as e:
                    logger.error(f"Error in job callback for {job_id}: {e}")
    
    async def _cleanup_job(self, job_id: str, delay: int = 0):
        """清理任务（延迟清理）"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        if job_id in self.job_callbacks:
            del self.job_callbacks[job_id]
    
    async def _wait_for_active_jobs(self, timeout: int = 300):
        """等待所有活跃任务完成"""
        start_time = utc_now()
        
        while self.active_jobs and (utc_now() - start_time).total_seconds() < timeout:
            await asyncio.sleep(1)
        
        if self.active_jobs:
            logger.warning(f"Timeout waiting for {len(self.active_jobs)} jobs to complete")
    
    def _update_avg_processing_time(self, processing_time_ms: float):
        """更新平均处理时间"""
        if self.stats['successful_jobs'] == 1:
            self.stats['avg_processing_time_ms'] = processing_time_ms
        else:
            # 指数移动平均
            alpha = 0.1
            self.stats['avg_processing_time_ms'] = (
                alpha * processing_time_ms + 
                (1 - alpha) * self.stats['avg_processing_time_ms']
            )
    
    # 公共API方法
    
    def get_job_status(self, job_id: str) -> Optional[BatchJobStatus]:
        """获取任务状态"""
        return self.active_jobs.get(job_id)
    
    def list_active_jobs(self) -> List[BatchJobStatus]:
        """列出所有活跃任务"""
        return list(self.active_jobs.values())
    
    def add_job_callback(self, job_id: str, callback: Callable):
        """添加任务完成回调"""
        if job_id not in self.job_callbacks:
            self.job_callbacks[job_id] = []
        self.job_callbacks[job_id].append(callback)
    
    def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        if job_id in self.active_jobs:
            job_status = self.active_jobs[job_id]
            if job_status.status == EventStatus.PENDING:
                job_status.status = EventStatus.FAILED
                job_status.error_message = "Job cancelled by user"
                job_status.completed_at = utc_now()
                return True
        return False
    
    def get_buffer_metrics(self) -> Dict[str, Any]:
        """获取缓冲服务指标"""
        if self.buffer_service:
            return self.buffer_service.get_metrics()
        return {}
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        return {
            'stats': self.stats.copy(),
            'active_jobs_count': len(self.active_jobs),
            'active_jobs': [
                {
                    'job_id': status.job_id,
                    'status': status.status.value,
                    'progress_percentage': status.progress_percentage,
                    'total_events': status.total_events,
                    'processed_events': status.processed_events,
                    'failed_events': status.failed_events
                }
                for status in self.active_jobs.values()
            ]
        }
    
    def set_max_concurrent_jobs(self, max_jobs: int):
        """设置最大并发任务数"""
        self.max_concurrent_jobs = max_jobs
        self.job_semaphore = asyncio.Semaphore(max_jobs)

# 全局批处理管理器实例
_batch_manager_instance: Optional[EventBatchManager] = None

def get_batch_manager() -> EventBatchManager:
    """获取全局批处理管理器实例"""
    global _batch_manager_instance
    if _batch_manager_instance is None:
        _batch_manager_instance = EventBatchManager()
    return _batch_manager_instance

async def initialize_batch_manager():
    """初始化全局批处理管理器"""
    manager = get_batch_manager()
    await manager.initialize()

async def shutdown_batch_manager():
    """关闭全局批处理管理器"""
    global _batch_manager_instance
    if _batch_manager_instance:
        await _batch_manager_instance.shutdown()
        _batch_manager_instance = None
