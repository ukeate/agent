"""
事件流处理管道 - 整合验证、去重、质量检查、缓冲和存储的完整pipeline
"""

import asyncio
import time
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Optional, Any, Callable, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import uuid
from contextlib import asynccontextmanager
from src.core.database import get_db
from src.models.schemas.event_tracking import CreateEventRequest, EventStatus, DataQuality
from src.repositories.event_tracking_repository import (

    EventStreamRepository, EventDeduplicationRepository, 
    EventSchemaRepository, EventErrorRepository
)
from src.services.event_processing_service import EventProcessingService, EventValidationResult
from src.services.event_buffer_service import EventBufferService, BufferConfig, BufferPriority
from src.services.event_queue_service import EventQueueService, QueuePriority
from src.services.data_quality_service import DataQualityService, DeduplicationResult, QualityCheckResult

from src.core.logging import get_logger
logger = get_logger(__name__)

class PipelineStage(str, Enum):
    """管道阶段"""
    INGESTION = "ingestion"        # 数据摄取
    VALIDATION = "validation"      # 数据验证
    DEDUPLICATION = "deduplication"# 去重检查
    QUALITY_CHECK = "quality_check"# 质量检查
    ENRICHMENT = "enrichment"      # 数据增强
    ROUTING = "routing"           # 路由分发
    BUFFERING = "buffering"       # 缓冲存储
    PERSISTENCE = "persistence"   # 持久化存储
    AGGREGATION = "aggregation"   # 聚合处理
    NOTIFICATION = "notification" # 通知下游

class PipelineMode(str, Enum):
    """管道模式"""
    STREAMING = "streaming"       # 流式处理
    BATCH = "batch"              # 批处理
    HYBRID = "hybrid"            # 混合模式

@dataclass
class PipelineEvent:
    """管道事件"""
    original_event: CreateEventRequest
    event_id: str
    stage: PipelineStage
    
    # 处理结果
    validation_result: Optional[EventValidationResult] = None
    dedup_result: Optional[DeduplicationResult] = None
    quality_results: Optional[List[QualityCheckResult]] = None
    quality_score: float = 0.0
    data_quality: DataQuality = DataQuality.HIGH
    
    # 处理状态
    status: EventStatus = EventStatus.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # 时间追踪
    created_at: datetime = field(default_factory=lambda: utc_now())
    updated_at: datetime = field(default_factory=lambda: utc_now())
    stage_timestamps: Dict[PipelineStage, datetime] = field(default_factory=dict)
    
    # 元数据
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    routing_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineMetrics:
    """管道指标"""
    total_events: int = 0
    events_by_stage: Dict[PipelineStage, int] = field(default_factory=dict)
    events_by_status: Dict[EventStatus, int] = field(default_factory=dict)
    avg_processing_time_ms: Dict[PipelineStage, float] = field(default_factory=dict)
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    duplicate_rate: float = 0.0
    quality_distribution: Dict[DataQuality, int] = field(default_factory=dict)
    
    # 性能指标
    peak_throughput: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_updated: datetime = field(default_factory=lambda: utc_now())

class EventStageProcessor:
    """事件阶段处理器基类"""
    
    def __init__(self, stage: PipelineStage):
        self.stage = stage
        self.processing_time_samples = []
        self.max_samples = 1000
    
    async def process(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """处理事件"""
        start_time = time.time()
        pipeline_event.stage = self.stage
        pipeline_event.stage_timestamps[self.stage] = utc_now()
        
        try:
            result = await self._process_internal(pipeline_event)
            result.updated_at = utc_now()
            
            # 记录处理时间
            processing_time = (time.time() - start_time) * 1000
            self._record_processing_time(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in stage {self.stage}: {e}", exc_info=True)
            pipeline_event.status = EventStatus.FAILED
            pipeline_event.error_message = str(e)
            pipeline_event.updated_at = utc_now()
            return pipeline_event
    
    async def _process_internal(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """内部处理逻辑，由子类实现"""
        raise NotImplementedError
    
    def _record_processing_time(self, processing_time_ms: float):
        """记录处理时间"""
        self.processing_time_samples.append(processing_time_ms)
        if len(self.processing_time_samples) > self.max_samples:
            self.processing_time_samples.pop(0)
    
    def get_avg_processing_time(self) -> float:
        """获取平均处理时间"""
        if not self.processing_time_samples:
            return 0.0
        return sum(self.processing_time_samples) / len(self.processing_time_samples)

class IngestionProcessor(EventStageProcessor):
    """数据摄取处理器"""
    
    def __init__(self):
        super().__init__(PipelineStage.INGESTION)
    
    async def _process_internal(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """数据摄取处理"""
        # 基本的数据完整性检查
        event = pipeline_event.original_event
        
        if not event.event_id:
            event.event_id = str(uuid.uuid4())
        
        if not event.event_timestamp:
            event.event_timestamp = utc_now()
        
        # 添加摄取元数据
        pipeline_event.processing_metadata['ingested_at'] = utc_now().isoformat()
        pipeline_event.processing_metadata['pipeline_id'] = str(uuid.uuid4())
        
        logger.debug(f"Ingested event {event.event_id}")
        return pipeline_event

class ValidationProcessor(EventStageProcessor):
    """数据验证处理器"""
    
    def __init__(self, processing_service: EventProcessingService):
        super().__init__(PipelineStage.VALIDATION)
        self.processing_service = processing_service
    
    async def _process_internal(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """数据验证处理"""
        validation_result = await self.processing_service.validation_service.validate_event(
            pipeline_event.original_event
        )
        
        pipeline_event.validation_result = validation_result
        
        if not validation_result.is_valid:
            pipeline_event.status = EventStatus.FAILED
            pipeline_event.error_message = "Validation failed"
        
        logger.debug(f"Validated event {pipeline_event.event_id}, valid: {validation_result.is_valid}")
        return pipeline_event

class DeduplicationProcessor(EventStageProcessor):
    """去重处理器"""
    
    def __init__(self, quality_service: DataQualityService):
        super().__init__(PipelineStage.DEDUPLICATION)
        self.quality_service = quality_service
    
    async def _process_internal(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """去重处理"""
        dedup_result = await self.quality_service.dedup_engine.check_duplicates(
            pipeline_event.original_event
        )
        
        pipeline_event.dedup_result = dedup_result
        
        if dedup_result.is_duplicate:
            pipeline_event.status = EventStatus.DUPLICATE
            pipeline_event.processing_metadata['duplicate_info'] = {
                'original_event_id': dedup_result.original_event_id,
                'similarity_score': dedup_result.similarity_score,
                'duplicate_type': dedup_result.duplicate_type
            }
        
        logger.debug(f"Deduplication check for event {pipeline_event.event_id}, duplicate: {dedup_result.is_duplicate}")
        return pipeline_event

class QualityCheckProcessor(EventStageProcessor):
    """质量检查处理器"""
    
    def __init__(self, quality_service: DataQualityService):
        super().__init__(PipelineStage.QUALITY_CHECK)
        self.quality_service = quality_service
    
    async def _process_internal(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """质量检查处理"""
        quality_results = await self.quality_service.quality_checker.perform_quality_checks(
            pipeline_event.original_event
        )
        
        pipeline_event.quality_results = quality_results
        
        # 计算总体质量分数
        overall_score = self.quality_service._calculate_overall_quality_score(quality_results)
        quality_level = self.quality_service._determine_quality_level(overall_score)
        
        pipeline_event.quality_score = overall_score
        pipeline_event.data_quality = quality_level
        
        # 如果质量太低，标记为失败
        if quality_level == DataQuality.INVALID:
            pipeline_event.status = EventStatus.FAILED
            pipeline_event.error_message = "Data quality too low"
        
        logger.debug(f"Quality check for event {pipeline_event.event_id}, score: {overall_score:.2f}, level: {quality_level}")
        return pipeline_event

class EnrichmentProcessor(EventStageProcessor):
    """数据增强处理器"""
    
    def __init__(self):
        super().__init__(PipelineStage.ENRICHMENT)
    
    async def _process_internal(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """数据增强处理"""
        event = pipeline_event.original_event
        
        # 添加处理元数据到事件属性
        if not event.experiment_context:
            event.experiment_context = {}
        
        event.experiment_context.update({
            'pipeline_processed': True,
            'data_quality_score': pipeline_event.quality_score,
            'data_quality_level': pipeline_event.data_quality.value,
            'processed_at': utc_now().isoformat()
        })
        
        # 如果有质量问题，添加警告信息
        if pipeline_event.quality_results:
            warnings = []
            for result in pipeline_event.quality_results:
                warnings.extend(result.suggestions)
            
            if warnings:
                event.experiment_context['quality_warnings'] = warnings[:5]  # 限制数量
        
        logger.debug(f"Enriched event {pipeline_event.event_id}")
        return pipeline_event

class RoutingProcessor(EventStageProcessor):
    """路由分发处理器"""
    
    def __init__(self):
        super().__init__(PipelineStage.ROUTING)
    
    async def _process_internal(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """路由分发处理"""
        event = pipeline_event.original_event
        
        # 基于事件特征确定路由策略
        routing_info = {
            'priority': self._determine_priority(pipeline_event),
            'target_queue': self._determine_target_queue(pipeline_event),
            'processing_mode': self._determine_processing_mode(pipeline_event)
        }
        
        pipeline_event.routing_info = routing_info
        
        logger.debug(f"Routed event {pipeline_event.event_id} to {routing_info['target_queue']} with priority {routing_info['priority']}")
        return pipeline_event
    
    def _determine_priority(self, pipeline_event: PipelineEvent) -> str:
        """确定处理优先级"""
        event = pipeline_event.original_event
        
        # 转化事件高优先级
        if event.event_type.value == 'conversion':
            return 'high'
        
        # 关键实验高优先级
        if 'critical' in event.experiment_id.lower():
            return 'high'
        
        # 数据质量差的事件低优先级
        if pipeline_event.data_quality in [DataQuality.LOW, DataQuality.INVALID]:
            return 'low'
        
        return 'normal'
    
    def _determine_target_queue(self, pipeline_event: PipelineEvent) -> str:
        """确定目标队列"""
        priority = pipeline_event.routing_info.get('priority', 'normal')
        event_type = pipeline_event.original_event.event_type.value
        
        return f"events_{event_type}_{priority}"
    
    def _determine_processing_mode(self, pipeline_event: PipelineEvent) -> str:
        """确定处理模式"""
        # 重复事件直接丢弃
        if pipeline_event.status == EventStatus.DUPLICATE:
            return 'discard'
        
        # 失败事件进入错误处理流程
        if pipeline_event.status == EventStatus.FAILED:
            return 'error_handling'
        
        # 高质量事件快速处理
        if pipeline_event.data_quality == DataQuality.HIGH:
            return 'fast_track'
        
        return 'standard'

class PersistenceProcessor(EventStageProcessor):
    """持久化处理器"""
    
    def __init__(self, event_repo: EventStreamRepository):
        super().__init__(PipelineStage.PERSISTENCE)
        self.event_repo = event_repo
    
    async def _process_internal(self, pipeline_event: PipelineEvent) -> PipelineEvent:
        """持久化处理"""
        # 跳过重复和失败的事件
        if pipeline_event.status in [EventStatus.DUPLICATE, EventStatus.FAILED]:
            return pipeline_event
        
        try:
            # 存储事件到数据库
            db_event = await self.event_repo.create_event(
                pipeline_event.original_event,
                pipeline_event.data_quality
            )
            
            pipeline_event.processing_metadata['db_event_id'] = db_event.id
            pipeline_event.status = EventStatus.PROCESSED
            
            logger.debug(f"Persisted event {pipeline_event.event_id} to database")
            
        except Exception as e:
            pipeline_event.status = EventStatus.FAILED
            pipeline_event.error_message = f"Persistence failed: {str(e)}"
            logger.error(f"Failed to persist event {pipeline_event.event_id}: {e}")
        
        return pipeline_event

class EventStreamPipeline:
    """事件流处理管道"""
    
    def __init__(
        self,
        mode: PipelineMode = PipelineMode.STREAMING,
        max_concurrent_events: int = 100
    ):
        self.mode = mode
        self.max_concurrent_events = max_concurrent_events
        self.semaphore = asyncio.Semaphore(max_concurrent_events)
        
        # 处理器链
        self.processors: List[EventStageProcessor] = []
        
        # 指标收集
        self.metrics = PipelineMetrics()
        self.metrics_lock = asyncio.Lock()
        
        # 服务组件
        self.processing_service: Optional[EventProcessingService] = None
        self.quality_service: Optional[DataQualityService] = None
        self.buffer_service: Optional[EventBufferService] = None
        self.queue_service: Optional[EventQueueService] = None
        
        # 运行状态
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        
        # 回调函数
        self.stage_callbacks: Dict[PipelineStage, List[Callable]] = {}
        self.completion_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
    
    async def initialize(self):
        """初始化管道"""
        if self.is_running:
            return
        
        logger.info("Initializing Event Stream Pipeline")
        
        try:
            # 初始化数据库连接和仓库
            async with asynccontextmanager(get_async_db)() as db:
                event_repo = EventStreamRepository(db)
                dedup_repo = EventDeduplicationRepository(db)
                schema_repo = EventSchemaRepository(db)
                error_repo = EventErrorRepository(db)
                
                # 初始化服务
                self.processing_service = EventProcessingService(
                    event_repo, dedup_repo, schema_repo, error_repo
                )
                
                self.quality_service = DataQualityService(dedup_repo, schema_repo)
                
                # 初始化缓冲服务
                buffer_config = BufferConfig(
                    max_buffer_size=1000,
                    flush_interval_seconds=30,
                    max_batch_size=100
                )
                self.buffer_service = EventBufferService(buffer_config, self.processing_service)
                
                # 初始化队列服务
                self.queue_service = EventQueueService()
                await self.queue_service.initialize()
                
                # 构建处理器链
                await self._build_processor_chain(event_repo)
                
                self.is_running = True
                logger.info("Event Stream Pipeline initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def _build_processor_chain(self, event_repo: EventStreamRepository):
        """构建处理器链"""
        self.processors = [
            IngestionProcessor(),
            ValidationProcessor(self.processing_service),
            DeduplicationProcessor(self.quality_service),
            QualityCheckProcessor(self.quality_service),
            EnrichmentProcessor(),
            RoutingProcessor(),
            PersistenceProcessor(event_repo)
        ]
        
        logger.info(f"Built processor chain with {len(self.processors)} stages")
    
    async def process_event(self, event: CreateEventRequest) -> PipelineEvent:
        """处理单个事件"""
        if not self.is_running:
            await self.initialize()
        
        async with self.semaphore:
            pipeline_event = PipelineEvent(
                original_event=event,
                event_id=event.event_id or str(uuid.uuid4())
            )
            
            start_time = time.time()
            
            try:
                # 依次通过所有处理器
                for processor in self.processors:
                    pipeline_event = await processor.process(pipeline_event)
                    
                    # 调用阶段回调
                    await self._call_stage_callbacks(processor.stage, pipeline_event)
                    
                    # 如果事件失败，提前结束处理
                    if pipeline_event.status == EventStatus.FAILED:
                        break
                
                # 更新指标
                processing_time = (time.time() - start_time) * 1000
                await self._update_metrics(pipeline_event, processing_time)
                
                # 调用完成回调
                await self._call_completion_callbacks(pipeline_event)
                
                logger.debug(f"Completed processing event {pipeline_event.event_id}, status: {pipeline_event.status}")
                
                return pipeline_event
            
            except Exception as e:
                pipeline_event.status = EventStatus.FAILED
                pipeline_event.error_message = str(e)
                
                # 调用错误回调
                await self._call_error_callbacks(pipeline_event, e)
                
                logger.error(f"Pipeline processing failed for event {pipeline_event.event_id}: {e}", exc_info=True)
                return pipeline_event
    
    async def process_events_batch(self, events: List[CreateEventRequest]) -> List[PipelineEvent]:
        """批量处理事件"""
        if self.mode == PipelineMode.STREAMING:
            # 流式模式：并发处理
            tasks = [self.process_event(event) for event in events]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_event = PipelineEvent(
                        original_event=events[i],
                        event_id=events[i].event_id or str(uuid.uuid4()),
                        stage=PipelineStage.INGESTION,
                        status=EventStatus.FAILED,
                        error_message=str(result)
                    )
                    processed_results.append(error_event)
                else:
                    processed_results.append(result)
            
            return processed_results
        
        else:
            # 批处理模式：顺序处理
            results = []
            for event in events:
                result = await self.process_event(event)
                results.append(result)
            
            return results
    
    async def _update_metrics(self, pipeline_event: PipelineEvent, processing_time_ms: float):
        """更新管道指标"""
        async with self.metrics_lock:
            self.metrics.total_events += 1
            
            # 按阶段统计
            for stage in pipeline_event.stage_timestamps:
                self.metrics.events_by_stage[stage] = self.metrics.events_by_stage.get(stage, 0) + 1
            
            # 按状态统计
            self.metrics.events_by_status[pipeline_event.status] = \
                self.metrics.events_by_status.get(pipeline_event.status, 0) + 1
            
            # 按质量统计
            self.metrics.quality_distribution[pipeline_event.data_quality] = \
                self.metrics.quality_distribution.get(pipeline_event.data_quality, 0) + 1
            
            # 更新延迟指标
            alpha = 0.1  # 指数移动平均
            if self.metrics.avg_latency_ms == 0:
                self.metrics.avg_latency_ms = processing_time_ms
            else:
                self.metrics.avg_latency_ms = (
                    alpha * processing_time_ms + 
                    (1 - alpha) * self.metrics.avg_latency_ms
                )
            
            # 计算错误率和重复率
            total = self.metrics.total_events
            failed = self.metrics.events_by_status.get(EventStatus.FAILED, 0)
            duplicates = self.metrics.events_by_status.get(EventStatus.DUPLICATE, 0)
            
            self.metrics.error_rate = (failed / total) * 100 if total > 0 else 0
            self.metrics.duplicate_rate = (duplicates / total) * 100 if total > 0 else 0
            
            self.metrics.last_updated = utc_now()
    
    async def _call_stage_callbacks(self, stage: PipelineStage, pipeline_event: PipelineEvent):
        """调用阶段回调"""
        if stage in self.stage_callbacks:
            for callback in self.stage_callbacks[stage]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(pipeline_event)
                    else:
                        callback(pipeline_event)
                except Exception as e:
                    logger.error(f"Error in stage callback for {stage}: {e}")
    
    async def _call_completion_callbacks(self, pipeline_event: PipelineEvent):
        """调用完成回调"""
        for callback in self.completion_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(pipeline_event)
                else:
                    callback(pipeline_event)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")
    
    async def _call_error_callbacks(self, pipeline_event: PipelineEvent, error: Exception):
        """调用错误回调"""
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(pipeline_event, error)
                else:
                    callback(pipeline_event, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def add_stage_callback(self, stage: PipelineStage, callback: Callable):
        """添加阶段回调"""
        if stage not in self.stage_callbacks:
            self.stage_callbacks[stage] = []
        self.stage_callbacks[stage].append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """添加完成回调"""
        self.completion_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def get_metrics(self) -> PipelineMetrics:
        """获取管道指标"""
        return self.metrics
    
    def get_stage_metrics(self) -> Dict[PipelineStage, Dict[str, Any]]:
        """获取各阶段指标"""
        stage_metrics = {}
        
        for processor in self.processors:
            stage_metrics[processor.stage] = {
                'avg_processing_time_ms': processor.get_avg_processing_time(),
                'total_processed': self.metrics.events_by_stage.get(processor.stage, 0)
            }
        
        return stage_metrics
    
    async def shutdown(self):
        """关闭管道"""
        if not self.is_running:
            return
        
        logger.info("Shutting down Event Stream Pipeline")
        self.is_running = False
        
        # 取消所有处理任务
        for task in self.processing_tasks:
            task.cancel()
        
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # 关闭服务组件
        if self.buffer_service:
            self.buffer_service.stop()
        
        if self.queue_service:
            await self.queue_service.shutdown()
        
        logger.info("Event Stream Pipeline shutdown completed")

# 全局管道实例
_pipeline_instance: Optional[EventStreamPipeline] = None

def get_event_pipeline() -> EventStreamPipeline:
    """获取全局管道实例"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EventStreamPipeline()
    return _pipeline_instance

async def initialize_pipeline():
    """初始化全局管道"""
    pipeline = get_event_pipeline()
    await pipeline.initialize()

async def shutdown_pipeline():
    """关闭全局管道"""
    global _pipeline_instance
    if _pipeline_instance:
        await _pipeline_instance.shutdown()
        _pipeline_instance = None
