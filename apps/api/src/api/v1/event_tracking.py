"""
事件追踪API端点 - 提供事件收集、查询和管理功能
"""
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import time
import hashlib
import json

from core.database import get_db
from core.logging import get_logger
from models.schemas.event_tracking import (
    CreateEventRequest, BatchEventsRequest, EventResponse, BatchEventsResponse,
    EventStreamResponse, EventQueryRequest, EventQueryResponse, 
    EventProcessingStats, EventValidationResult, EventDeduplicationInfo,
    EventStatus, DataQuality, EventType
)
from repositories.event_tracking_repository import (
    EventStreamRepository, EventDeduplicationRepository, 
    EventSchemaRepository, EventErrorRepository
)
from services.event_processing_service import EventProcessingService

logger = get_logger(__name__)
router = APIRouter(prefix="/event-tracking", tags=["事件追踪"])

# 依赖注入
async def get_event_stream_repo(db: AsyncSession = Depends(get_db)) -> EventStreamRepository:
    return EventStreamRepository(db)

async def get_dedup_repo(db: AsyncSession = Depends(get_db)) -> EventDeduplicationRepository:
    return EventDeduplicationRepository(db)

async def get_schema_repo(db: AsyncSession = Depends(get_db)) -> EventSchemaRepository:
    return EventSchemaRepository(db)

async def get_error_repo(db: AsyncSession = Depends(get_db)) -> EventErrorRepository:
    return EventErrorRepository(db)

async def get_event_processing_service(
    event_repo: EventStreamRepository = Depends(get_event_stream_repo),
    dedup_repo: EventDeduplicationRepository = Depends(get_dedup_repo),
    schema_repo: EventSchemaRepository = Depends(get_schema_repo),
    error_repo: EventErrorRepository = Depends(get_error_repo)
) -> EventProcessingService:
    return EventProcessingService(event_repo, dedup_repo, schema_repo, error_repo)

def get_client_ip(request: Request) -> str:
    """获取客户端IP地址"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


@router.post("/events", response_model=EventResponse)
async def create_event(
    event: CreateEventRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    processing_service: EventProcessingService = Depends(get_event_processing_service)
):
    """创建单个事件"""
    try:
        client_ip = get_client_ip(request)
        
        # 使用事件处理服务处理事件
        result = await processing_service.process_event(event, client_ip)
        
        # 如果处理成功且状态为PENDING，添加后台处理任务
        if result.success and result.status == EventStatus.PENDING:
            background_tasks.add_task(
                _process_event_async_v2, 
                event.event_id,
                processing_service.event_repo
            )
        
        return EventResponse(
            event_id=result.event_id,
            status=result.status,
            message=result.message,
            validation_errors=result.errors,
            processed_at=utc_now()
        )
        
    except Exception as e:
        logger.error(f"创建事件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建事件失败: {str(e)}")


@router.post("/events/batch", response_model=BatchEventsResponse)
async def create_events_batch(
    batch_request: BatchEventsRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    processing_service: EventProcessingService = Depends(get_event_processing_service)
):
    """批量创建事件"""
    start_time = time.time()
    
    try:
        client_ip = get_client_ip(request)
        
        # 使用处理服务批量处理事件
        results = await processing_service.process_events_batch(batch_request.events, client_ip)
        
        # 统计结果
        successful_count = sum(1 for r in results if r.success and r.status == EventStatus.PENDING)
        failed_count = sum(1 for r in results if not r.success)
        duplicate_count = sum(1 for r in results if r.success and r.status == EventStatus.DUPLICATE)
        
        # 转换为响应格式
        event_responses = []
        pending_event_ids = []
        
        for result in results:
            event_responses.append(EventResponse(
                event_id=result.event_id,
                status=result.status,
                message=result.message,
                validation_errors=result.errors
            ))
            
            if result.success and result.status == EventStatus.PENDING:
                pending_event_ids.append(result.event_id)
        
        # 添加后台批量处理任务
        if pending_event_ids:
            background_tasks.add_task(
                _process_batch_async_v2,
                batch_request.batch_id,
                pending_event_ids,
                processing_service.event_repo
            )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return BatchEventsResponse(
            batch_id=batch_request.batch_id,
            total_events=len(batch_request.events),
            successful_events=successful_count,
            failed_events=failed_count,
            duplicate_events=duplicate_count,
            processing_time_ms=processing_time,
            events=event_responses
        )
        
    except Exception as e:
        logger.error(f"批量创建事件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


@router.get("/events", response_model=EventQueryResponse)
async def query_events(
    query: EventQueryRequest = Depends(),
    event_repo: EventStreamRepository = Depends(get_event_stream_repo)
):
    """查询事件"""
    try:
        start_time = time.time()
        
        # 构建查询条件
        filters = {}
        if query.experiment_ids:
            filters['experiment_id'] = query.experiment_ids
        if query.user_ids:
            filters['user_id'] = query.user_ids
        if query.variant_ids:
            filters['variant_id'] = query.variant_ids
        if query.event_types:
            filters['event_type'] = [et.value for et in query.event_types]
        if query.event_names:
            filters['event_name'] = query.event_names
        if query.start_time:
            filters['start_time'] = query.start_time
        if query.end_time:
            filters['end_time'] = query.end_time
        if query.status:
            filters['status'] = [s.value for s in query.status]
        if query.data_quality:
            filters['data_quality'] = [dq.value for dq in query.data_quality]
        
        # 执行查询
        events, total_count = await event_repo.query_events(
            filters=filters,
            page=query.page,
            page_size=query.page_size,
            sort_by=query.sort_by,
            sort_order=query.sort_order
        )
        
        # 转换为响应模型
        event_responses = []
        for event in events:
            event_responses.append(EventStreamResponse(
                id=event.id,
                event_id=event.event_id,
                experiment_id=event.experiment_id,
                variant_id=event.variant_id,
                user_id=event.user_id,
                session_id=event.session_id,
                event_type=EventType(event.event_type),
                event_name=event.event_name,
                event_category=event.event_category,
                event_timestamp=event.event_timestamp,
                server_timestamp=event.server_timestamp,
                processed_at=event.processed_at,
                properties=event.properties,
                user_properties=event.user_properties,
                experiment_context=event.experiment_context,
                client_info=event.client_info,
                device_info=event.device_info,
                geo_info=event.geo_info,
                status=EventStatus(event.status),
                data_quality=DataQuality(event.data_quality),
                validation_errors=event.validation_errors,
                created_at=event.created_at,
                updated_at=event.updated_at
            ))
        
        query_time = int((time.time() - start_time) * 1000)
        total_pages = (total_count + query.page_size - 1) // query.page_size
        
        return EventQueryResponse(
            total_count=total_count,
            page=query.page,
            page_size=query.page_size,
            total_pages=total_pages,
            events=event_responses,
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"查询事件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.get("/events/{event_id}", response_model=EventStreamResponse)
async def get_event(
    event_id: str,
    event_repo: EventStreamRepository = Depends(get_event_stream_repo)
):
    """获取单个事件详情"""
    try:
        event = await event_repo.get_event_by_id(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="事件不存在")
        
        return EventStreamResponse(
            id=event.id,
            event_id=event.event_id,
            experiment_id=event.experiment_id,
            variant_id=event.variant_id,
            user_id=event.user_id,
            session_id=event.session_id,
            event_type=EventType(event.event_type),
            event_name=event.event_name,
            event_category=event.event_category,
            event_timestamp=event.event_timestamp,
            server_timestamp=event.server_timestamp,
            processed_at=event.processed_at,
            properties=event.properties,
            user_properties=event.user_properties,
            experiment_context=event.experiment_context,
            client_info=event.client_info,
            device_info=event.device_info,
            geo_info=event.geo_info,
            status=EventStatus(event.status),
            data_quality=DataQuality(event.data_quality),
            validation_errors=event.validation_errors,
            created_at=event.created_at,
            updated_at=event.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取事件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取事件失败: {str(e)}")


@router.get("/stats", response_model=EventProcessingStats)
async def get_processing_stats(
    experiment_id: Optional[str] = Query(None, description="实验ID过滤"),
    hours: int = Query(24, ge=1, le=168, description="统计时间范围（小时）"),
    event_repo: EventStreamRepository = Depends(get_event_stream_repo)
):
    """获取事件处理统计"""
    try:
        start_time = utc_now() - timedelta(hours=hours)
        stats = await event_repo.get_processing_stats(
            start_time=start_time,
            experiment_id=experiment_id
        )
        return stats
        
    except Exception as e:
        logger.error(f"获取处理统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.post("/events/{event_id}/reprocess")
async def reprocess_event(
    event_id: str,
    background_tasks: BackgroundTasks,
    event_repo: EventStreamRepository = Depends(get_event_stream_repo)
):
    """重新处理事件"""
    try:
        event = await event_repo.get_event_by_id(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="事件不存在")
        
        if event.status != EventStatus.FAILED.value:
            raise HTTPException(status_code=400, detail="只能重新处理失败的事件")
        
        # 重置事件状态
        await event_repo.update_event_status(event_id, EventStatus.PENDING)
        
        # 添加后台处理任务
        background_tasks.add_task(_process_event_async, event.id, event_repo)
        
        return {"message": "事件已加入重新处理队列", "event_id": event_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重新处理事件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重新处理失败: {str(e)}")


# 后台任务函数
async def _process_event_async_v2(event_id: str, event_repo: EventStreamRepository):
    """异步处理单个事件（新版本）"""
    try:
        # 模拟事件后处理逻辑：聚合、分析等
        await asyncio.sleep(0.1)  # 模拟处理时间
        
        # 更新事件状态为已处理
        await event_repo.update_event_status(event_id, EventStatus.PROCESSED)
        await event_repo.update_processed_at(event_id, utc_now())
        
        logger.info(f"事件 {event_id} 后处理完成")
        
    except Exception as e:
        logger.error(f"异步处理事件 {event_id} 失败: {e}")
        try:
            await event_repo.update_event_status(event_id, EventStatus.FAILED)
        except Exception:
            pass


async def _process_batch_async_v2(batch_id: str, event_ids: List[str], event_repo: EventStreamRepository):
    """异步处理事件批次（新版本）"""
    try:
        logger.info(f"开始后处理批次 {batch_id}，包含 {len(event_ids)} 个事件")
        
        # 并发处理所有事件
        tasks = [_process_event_async_v2(event_id, event_repo) for event_id in event_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"批次 {batch_id} 后处理完成")
        
    except Exception as e:
        logger.error(f"批量后处理失败 {batch_id}: {e}")


# 保留原有函数以兼容性
async def _process_event_async(event_id: str, event_repo: EventStreamRepository):
    """异步处理单个事件（兼容版本）"""
    return await _process_event_async_v2(event_id, event_repo)


async def _process_batch_async(batch_id: str, event_ids: List[str], event_repo: EventStreamRepository):
    """异步处理事件批次（兼容版本）"""
    return await _process_batch_async_v2(batch_id, event_ids, event_repo)