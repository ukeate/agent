"""
事件批处理API端点 - 管理事件批处理任务和缓冲
"""

from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import Field
from src.models.schemas.event_tracking import CreateEventRequest, BatchEventsRequest, EventStatus
from src.services.event_batch_manager import (
    get_batch_manager, EventBatchManager, BatchJobConfig, BatchJobStatus,
    ProcessingMode
)
from src.api.base_model import ApiBaseModel
from src.services.event_buffer_service import BufferPriority

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/event-batch", tags=["事件批处理"])

# 请求和响应模型
class SubmitEventRequest(ApiBaseModel):
    """提交单个事件请求"""
    event: CreateEventRequest
    processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE
    priority: BufferPriority = BufferPriority.NORMAL

class SubmitBatchRequest(ApiBaseModel):
    """提交批量事件请求"""
    batch: BatchEventsRequest
    job_config: Optional[BatchJobConfig] = None

class EventSubmissionResponse(ApiBaseModel):
    """事件提交响应"""
    event_id: str
    job_id: Optional[str] = None
    status: str
    message: str
    submitted_at: datetime = Field(default_factory=lambda: utc_now())

class BatchSubmissionResponse(ApiBaseModel):
    """批量提交响应"""
    job_id: str
    total_events: int
    status: str
    message: str
    estimated_completion_time: Optional[datetime] = None
    submitted_at: datetime = Field(default_factory=lambda: utc_now())

class JobStatusResponse(ApiBaseModel):
    """任务状态响应"""
    job_id: str
    status: EventStatus
    total_events: int
    processed_events: int
    failed_events: int
    progress_percentage: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    estimated_remaining_time: Optional[int] = None  # seconds

class BufferMetricsResponse(ApiBaseModel):
    """缓冲指标响应"""
    global_metrics: Dict[str, Any]
    buffer_metrics: Dict[str, Any]
    config: Dict[str, Any]

class ManagerStatsResponse(ApiBaseModel):
    """管理器统计响应"""
    stats: Dict[str, Any]
    active_jobs_count: int
    active_jobs: List[Dict[str, Any]]

# 依赖注入
async def get_manager() -> EventBatchManager:
    manager = get_batch_manager()
    if not manager.is_initialized:
        await manager.initialize()
    return manager

@router.post("/events/submit", response_model=EventSubmissionResponse)
async def submit_single_event(
    request: SubmitEventRequest,
    background_tasks: BackgroundTasks,
    manager: EventBatchManager = Depends(get_manager)
):
    """提交单个事件处理"""
    try:
        event_id = await manager.submit_event(
            event=request.event,
            processing_mode=request.processing_mode,
            priority=request.priority
        )
        
        return EventSubmissionResponse(
            event_id=event_id,
            status="submitted",
            message=f"Event submitted for {request.processing_mode.value} processing"
        )
        
    except Exception as e:
        logger.error(f"Failed to submit event {request.event.event_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Event submission failed: {str(e)}")

@router.post("/batches/submit", response_model=BatchSubmissionResponse)
async def submit_batch(
    request: SubmitBatchRequest,
    background_tasks: BackgroundTasks,
    manager: EventBatchManager = Depends(get_manager)
):
    """提交批量事件处理"""
    try:
        job_id = await manager.submit_batch(
            batch=request.batch,
            config=request.job_config
        )
        
        # 估算完成时间（简单估算）
        estimated_completion = None
        if request.job_config and request.job_config.processing_mode == ProcessingMode.IMMEDIATE:
            # 立即处理模式：基于事件数量估算
            estimated_seconds = len(request.batch.events) * 0.1  # 假设每个事件0.1秒
            estimated_completion = utc_now().replace(microsecond=0) + \
                                 timedelta(seconds=int(estimated_seconds))
        
        return BatchSubmissionResponse(
            job_id=job_id,
            total_events=len(request.batch.events),
            status="submitted",
            message=f"Batch job submitted with {len(request.batch.events)} events",
            estimated_completion_time=estimated_completion
        )
        
    except Exception as e:
        logger.error(f"Failed to submit batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {str(e)}")

@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    manager: EventBatchManager = Depends(get_manager)
):
    """获取任务状态"""
    try:
        job_status = manager.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # 估算剩余时间
        estimated_remaining = None
        if job_status.status == EventStatus.PROCESSED and job_status.progress_percentage < 100:
            if job_status.started_at and job_status.processed_events > 0:
                elapsed_seconds = (utc_now() - job_status.started_at).total_seconds()
                processing_rate = job_status.processed_events / elapsed_seconds
                remaining_events = job_status.total_events - job_status.processed_events
                if processing_rate > 0:
                    estimated_remaining = int(remaining_events / processing_rate)
        
        return JobStatusResponse(
            job_id=job_status.job_id,
            status=job_status.status,
            total_events=job_status.total_events,
            processed_events=job_status.processed_events,
            failed_events=job_status.failed_events,
            progress_percentage=job_status.progress_percentage,
            started_at=job_status.started_at,
            completed_at=job_status.completed_at,
            error_message=job_status.error_message,
            estimated_remaining_time=estimated_remaining
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/jobs", response_model=List[JobStatusResponse])
async def list_active_jobs(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of jobs to return"),
    manager: EventBatchManager = Depends(get_manager)
):
    """列出活跃的批处理任务"""
    try:
        active_jobs = manager.list_active_jobs()
        
        # 限制返回数量并转换格式
        limited_jobs = active_jobs[:limit]
        
        response_jobs = []
        for job_status in limited_jobs:
            # 估算剩余时间
            estimated_remaining = None
            if job_status.status == EventStatus.PROCESSED and job_status.progress_percentage < 100:
                if job_status.started_at and job_status.processed_events > 0:
                    elapsed_seconds = (utc_now() - job_status.started_at).total_seconds()
                    processing_rate = job_status.processed_events / elapsed_seconds if elapsed_seconds > 0 else 0
                    remaining_events = job_status.total_events - job_status.processed_events
                    if processing_rate > 0:
                        estimated_remaining = int(remaining_events / processing_rate)
            
            response_jobs.append(JobStatusResponse(
                job_id=job_status.job_id,
                status=job_status.status,
                total_events=job_status.total_events,
                processed_events=job_status.processed_events,
                failed_events=job_status.failed_events,
                progress_percentage=job_status.progress_percentage,
                started_at=job_status.started_at,
                completed_at=job_status.completed_at,
                error_message=job_status.error_message,
                estimated_remaining_time=estimated_remaining
            ))
        
        return response_jobs
        
    except Exception as e:
        logger.error(f"Failed to list active jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    manager: EventBatchManager = Depends(get_manager)
):
    """取消批处理任务"""
    try:
        success = manager.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")
        
        return {"message": f"Job {job_id} cancelled successfully", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.get("/buffer/metrics", response_model=BufferMetricsResponse)
async def get_buffer_metrics(
    manager: EventBatchManager = Depends(get_manager)
):
    """获取缓冲服务指标"""
    try:
        metrics = manager.get_buffer_metrics()
        
        return BufferMetricsResponse(
            global_metrics=metrics.get('global_metrics', {}),
            buffer_metrics=metrics.get('buffer_metrics', {}),
            config=metrics.get('config', {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get buffer metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get buffer metrics: {str(e)}")

@router.get("/stats", response_model=ManagerStatsResponse)
async def get_manager_stats(
    manager: EventBatchManager = Depends(get_manager)
):
    """获取批处理管理器统计信息"""
    try:
        stats = manager.get_manager_stats()
        
        return ManagerStatsResponse(
            stats=stats['stats'],
            active_jobs_count=stats['active_jobs_count'],
            active_jobs=stats['active_jobs']
        )
        
    except Exception as e:
        logger.error(f"Failed to get manager stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get manager stats: {str(e)}")

@router.post("/buffer/flush")
async def force_flush_buffers(
    partition_keys: Optional[List[str]] = None,
    manager: EventBatchManager = Depends(get_manager)
):
    """强制刷新缓冲区"""
    try:
        # 获取缓冲服务并强制刷新
        if manager.buffer_service:
            # 这里可以添加强制刷新特定分区的逻辑
            # 目前实现为刷新所有缓冲区
            await manager.buffer_service._force_flush_oldest_buffers()
            
            return {
                "message": "Buffer flush initiated",
                "partition_keys": partition_keys or "all"
            }
        else:
            raise HTTPException(status_code=503, detail="Buffer service not available")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to flush buffers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to flush buffers: {str(e)}")

@router.post("/config/concurrent-jobs")
async def set_max_concurrent_jobs(
    max_jobs: int = Query(..., ge=1, le=100, description="Maximum concurrent jobs"),
    manager: EventBatchManager = Depends(get_manager)
):
    """设置最大并发任务数"""
    try:
        manager.set_max_concurrent_jobs(max_jobs)
        
        return {
            "message": f"Maximum concurrent jobs set to {max_jobs}",
            "max_concurrent_jobs": max_jobs
        }
        
    except Exception as e:
        logger.error(f"Failed to set max concurrent jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set max concurrent jobs: {str(e)}")

@router.get("/health")
async def health_check(
    manager: EventBatchManager = Depends(get_manager)
):
    """批处理服务健康检查"""
    try:
        health_info = {
            "status": "healthy" if manager.is_initialized else "initializing",
            "initialized": manager.is_initialized,
            "buffer_service_running": manager.buffer_service is not None and manager.buffer_service.is_running,
            "active_jobs": len(manager.active_jobs),
            "timestamp": utc_now().isoformat()
        }
        
        # 检查缓冲服务健康状态
        if manager.buffer_service:
            buffer_metrics = manager.buffer_service.get_metrics()
            health_info["buffer_health"] = {
                "total_buffers": len(manager.buffer_service.buffers),
                "memory_usage_mb": buffer_metrics['global_metrics'].get('memory_usage_mb', 0),
                "total_buffered": buffer_metrics['global_metrics'].get('total_buffered', 0)
            }
        
        return health_info
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": utc_now().isoformat()
        }
