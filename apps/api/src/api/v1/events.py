"""
事件处理系统API端点
提供事件查询、监控和管理接口
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from pydantic import Field
import asyncio
import json
import uuid
from src.ai.autogen.events import Event, EventType, EventPriority
from src.ai.autogen.event_store import EventStore, EventReplayService
from src.ai.autogen.distributed_events import DistributedEventCoordinator
from src.ai.autogen.event_processors import AsyncEventProcessingEngine
from src.ai.autogen.monitoring import EventProcessingMonitor
from src.api.base_model import ApiBaseModel

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/events", tags=["Events"])

# 全局实例（实际应该通过依赖注入）
event_store: Optional[EventStore] = None
event_coordinator: Optional[DistributedEventCoordinator] = None
processing_engine: Optional[AsyncEventProcessingEngine] = None
event_monitor: Optional[EventProcessingMonitor] = None
replay_service: Optional[EventReplayService] = None

class EventQuery(ApiBaseModel):
    """事件查询参数"""
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    event_types: Optional[List[str]] = Field(None, description="事件类型列表")
    source: Optional[str] = Field(None, description="事件源")
    target: Optional[str] = Field(None, description="事件目标")
    severity: Optional[str] = Field(None, description="严重程度")
    limit: int = Field(100, description="返回数量限制")
    offset: int = Field(0, description="偏移量")

class EventResponse(ApiBaseModel):
    """事件响应"""
    id: str
    timestamp: datetime
    type: str
    source: Optional[str] = None
    target: Optional[str] = None
    title: str
    message: str
    agent: Optional[str] = None
    severity: str
    data: Dict[str, Any] = Field(default_factory=dict)

class EventStats(ApiBaseModel):
    """事件统计"""
    total: int
    info: int
    warning: int
    error: int
    success: int
    critical: int
    by_source: Dict[str, int]
    by_type: Dict[str, int]

class ReplayRequest(ApiBaseModel):
    """事件重播请求"""
    agent_id: Optional[str] = Field(None, description="智能体ID")
    conversation_id: Optional[str] = Field(None, description="会话ID")
    start_time: datetime = Field(..., description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")

class ClusterStatus(ApiBaseModel):
    """集群状态"""
    node_id: str
    role: str
    status: str
    load: float
    active_nodes: int
    nodes: Dict[str, Dict[str, Any]]
    stats: Dict[str, Any]

def init_services(
    store: EventStore,
    coordinator: DistributedEventCoordinator,
    engine: AsyncEventProcessingEngine,
    monitor: EventProcessingMonitor
):
    """初始化服务实例"""
    global event_store, event_coordinator, processing_engine, event_monitor, replay_service
    event_store = store
    event_coordinator = coordinator
    processing_engine = engine
    event_monitor = monitor
    if store and engine:
        replay_service = EventReplayService(store, engine)

def convert_event_to_response(event: Event) -> EventResponse:
    """转换事件为响应格式"""
    # 根据事件类型映射标题
    title_map = {
        EventType.AGENT_CREATED: "智能体创建",
        EventType.AGENT_DESTROYED: "智能体销毁",
        EventType.AGENT_STATUS_CHANGED: "智能体状态变更",
        EventType.MESSAGE_SENT: "消息发送",
        EventType.MESSAGE_RECEIVED: "消息接收",
        EventType.TASK_ASSIGNED: "任务分配",
        EventType.TASK_STARTED: "任务开始",
        EventType.TASK_COMPLETED: "任务完成",
        EventType.TASK_FAILED: "任务失败",
        EventType.ERROR_OCCURRED: "错误发生",
        EventType.SYSTEM_STATUS_CHANGED: "系统状态变更",
    }
    
    # 根据事件类型映射UI类型
    type_map = {
        EventType.ERROR_OCCURRED: "error",
        EventType.TASK_COMPLETED: "success",
        EventType.TASK_FAILED: "error",
        EventType.MESSAGE_FAILED: "error",
        EventType.SYSTEM_STATUS_CHANGED: "warning",
        EventType.AGENT_STATUS_CHANGED: "warning",
    }
    
    # 根据优先级映射严重程度
    severity_map = {
        EventPriority.CRITICAL: "critical",
        EventPriority.HIGH: "high",
        EventPriority.NORMAL: "medium",
        EventPriority.LOW: "low"
    }
    
    return EventResponse(
        id=getattr(event, 'id', str(uuid.uuid4())),
        timestamp=getattr(event, 'timestamp', utc_now()),
        type=type_map.get(event.type, "info"),
        source=getattr(event, 'source', 'System'),
        target=getattr(event, 'target', None),
        title=title_map.get(event.type, event.type.value if hasattr(event.type, 'value') else str(event.type)),
        message=event.data.get('message', '') if hasattr(event, 'data') else '',
        agent=getattr(event, 'source', None),
        severity=severity_map.get(getattr(event, 'priority', EventPriority.NORMAL), "low"),
        data=getattr(event, 'data', {})
    )

@router.get("/list", response_model=List[EventResponse])
async def get_events(query: EventQuery = Depends()) -> List[EventResponse]:
    """获取事件列表"""
    if not event_store:
        raise HTTPException(status_code=503, detail="Event store not initialized")
    
    try:
        # 设置默认时间范围
        start_time = query.start_time or utc_now() - timedelta(hours=24)
        end_time = query.end_time or utc_now()
        
        # 构建过滤条件
        filters = {}
        if query.source:
            filters['source'] = query.source
        if query.target:
            filters['target'] = query.target
        
        # 查询事件
        events = await event_store.replay_events(
            start_time=start_time,
            end_time=end_time,
            event_types=[EventType(t) for t in query.event_types] if query.event_types else None,
            filters=filters
        )
        
        # 转换为响应格式
        responses = [convert_event_to_response(event) for event in events]
        
        # 应用分页
        start_idx = query.offset
        end_idx = query.offset + query.limit
        
        return responses[start_idx:end_idx]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=EventStats)
async def get_event_stats(
    hours: int = Query(24, description="统计时间范围（小时）")
) -> EventStats:
    """获取事件统计信息"""
    if not event_store:
        raise HTTPException(status_code=503, detail="Event store not initialized")
    
    try:
        # 查询时间范围内的事件
        start_time = utc_now() - timedelta(hours=hours)
        end_time = utc_now()
        
        events = await event_store.replay_events(
            start_time=start_time,
            end_time=end_time
        )
        
        # 统计事件
        stats = {
            'total': len(events),
            'info': 0,
            'warning': 0,
            'error': 0,
            'success': 0,
            'critical': 0,
            'by_source': {},
            'by_type': {}
        }
        
        for event in events:
            # 转换为响应格式以获取类型和严重程度
            response = convert_event_to_response(event)
            
            # 统计类型
            stats[response.type] = stats.get(response.type, 0) + 1
            
            # 统计严重程度
            if response.severity == 'critical':
                stats['critical'] += 1
            
            # 统计来源
            source = response.source or 'Unknown'
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            # 统计事件类型
            event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
            stats['by_type'][event_type] = stats['by_type'].get(event_type, 0) + 1
        
        return EventStats(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/replay")
async def replay_events(request: ReplayRequest) -> Dict[str, Any]:
    """重播历史事件"""
    if not replay_service:
        raise HTTPException(status_code=503, detail="重播服务未初始化")
    
    try:
        if request.agent_id:
            result = await replay_service.replay_for_agent(
                agent_id=request.agent_id,
                from_time=request.start_time,
                to_time=request.end_time
            )
        elif request.conversation_id:
            result = await replay_service.replay_conversation(
                conversation_id=request.conversation_id,
                from_time=request.start_time
            )
        else:
            raise HTTPException(status_code=400, detail="必须指定agent_id或conversation_id")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cluster/status", response_model=ClusterStatus)
async def get_cluster_status() -> ClusterStatus:
    """获取集群状态"""
    if not event_coordinator:
        raise HTTPException(status_code=503, detail="事件协调器未初始化")
    try:
        status = await event_coordinator.get_cluster_status()
        return ClusterStatus(**status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/metrics")
async def get_monitoring_metrics() -> Dict[str, Any]:
    """获取监控指标"""
    if not event_monitor:
        raise HTTPException(status_code=503, detail="监控系统未初始化")
    try:
        return await event_monitor.get_processing_stats()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/stream")
async def event_stream(websocket: WebSocket):
    """WebSocket事件流"""
    await websocket.accept()

    if not event_store or not getattr(event_store, "redis", None):
        await websocket.send_json({"type": "error", "message": "事件系统未初始化"})
        await websocket.close()
        return

    channel = f"{event_store.stream_prefix}pubsub"
    pubsub = event_store.redis.pubsub()
    await pubsub.subscribe(channel)

    async def forward_events() -> None:
        async for message in pubsub.listen():
            if not message or message.get("type") != "message":
                continue
            raw = message.get("data")
            if raw in (1, b"1"):
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")

            try:
                payload = json.loads(raw)
                event = Event.from_dict(payload)
            except Exception:
                continue

            response = convert_event_to_response(event)
            await websocket.send_json({"type": "event", "data": response.model_dump(mode="json")})

    forward_task = asyncio.create_task(forward_events())

    try:
        await websocket.send_json({"type": "connection", "message": "已连接到事件流"})

        while True:
            data = await websocket.receive_text()
            if data.lower() == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("WebSocket连接断开")
    except Exception as e:
        logger.error("事件流WebSocket异常", error=str(e))
    finally:
        forward_task.cancel()
        await asyncio.gather(forward_task, return_exceptions=True)
        try:
            await pubsub.unsubscribe(channel)
        except Exception:
            logger.exception("事件流取消订阅失败", exc_info=True)
        try:
            await pubsub.close()
        except Exception:
            logger.exception("事件流关闭pubsub失败", exc_info=True)

@router.get("/dead-letter")
async def get_dead_letter_events(limit: int = 100) -> List[Dict[str, Any]]:
    """获取死信队列事件"""
    if not event_store:
        return []
    
    try:
        return await event_store.get_dead_letter_events(limit)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit")
async def submit_event(
    event_type: str = Query(..., description="事件类型"),
    source: str = Query(..., description="事件源"),
    message: str = Query("", description="事件消息"),
    priority: str = Query("normal", description="优先级")
) -> Dict[str, str]:
    """手动提交事件（用于测试）"""
    if not processing_engine or not event_store:
        raise HTTPException(status_code=503, detail="事件系统未初始化")
    
    try:
        # 创建事件
        event = Event(
            id=str(uuid.uuid4()),
            type=EventType[event_type.upper()] if event_type.upper() in EventType.__members__ else EventType.MESSAGE_SENT,
            source=source,
            data={"message": message},
            timestamp=utc_now(),
            priority=EventPriority[priority.upper()] if priority.upper() in EventPriority.__members__ else EventPriority.NORMAL
        )
        
        if event_coordinator:
            await event_coordinator.distribute_event(event)
        else:
            await processing_engine.submit_event(event, event.priority)
            await event_store.append_event(event)
        
        return {"status": "success", "event_id": event.id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
