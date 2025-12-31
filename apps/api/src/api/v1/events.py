"""
事件处理系统API端点
提供事件查询、监控和管理接口
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends
from starlette.requests import HTTPConnection
from typing import List, Optional, Dict, Any
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from src.core.utils.async_utils import create_task_with_logging
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

def _get_state_service(connection: HTTPConnection, key: str, error_message: str):
    service = getattr(connection.app.state, key, None)
    if not service:
        raise HTTPException(status_code=503, detail=error_message)
    return service

def get_event_store(connection: HTTPConnection) -> EventStore:
    return _get_state_service(connection, "autogen_event_store", "Event store not initialized")

def get_event_coordinator(connection: HTTPConnection) -> DistributedEventCoordinator:
    return _get_state_service(connection, "autogen_event_coordinator", "Event coordinator not initialized")

def get_optional_event_coordinator(connection: HTTPConnection) -> Optional[DistributedEventCoordinator]:
    return getattr(connection.app.state, "autogen_event_coordinator", None)

def get_processing_engine(connection: HTTPConnection) -> AsyncEventProcessingEngine:
    return _get_state_service(connection, "autogen_processing_engine", "Processing engine not initialized")

def get_event_monitor(connection: HTTPConnection) -> EventProcessingMonitor:
    return _get_state_service(connection, "autogen_event_monitor", "Event monitor not initialized")

def get_replay_service(
    connection: HTTPConnection,
    store: EventStore = Depends(get_event_store),
    engine: AsyncEventProcessingEngine = Depends(get_processing_engine),
) -> EventReplayService:
    replay_service = getattr(connection.app.state, "autogen_event_replay_service", None)
    if replay_service:
        return replay_service
    replay_service = EventReplayService(store, engine)
    connection.app.state.autogen_event_replay_service = replay_service
    return replay_service

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
async def get_events(
    query: EventQuery = Depends(),
    store: EventStore = Depends(get_event_store),
) -> List[EventResponse]:
    """获取事件列表"""
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
        events = await store.replay_events(
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
    hours: int = Query(24, description="统计时间范围（小时）"),
    store: EventStore = Depends(get_event_store),
) -> EventStats:
    """获取事件统计信息"""
    try:
        # 查询时间范围内的事件
        start_time = utc_now() - timedelta(hours=hours)
        end_time = utc_now()
        
        events = await store.replay_events(
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
async def replay_events(
    request: ReplayRequest,
    service: EventReplayService = Depends(get_replay_service),
) -> Dict[str, Any]:
    """重播历史事件"""
    try:
        if request.agent_id:
            result = await service.replay_for_agent(
                agent_id=request.agent_id,
                from_time=request.start_time,
                to_time=request.end_time
            )
        elif request.conversation_id:
            result = await service.replay_conversation(
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
async def get_cluster_status(
    coordinator: DistributedEventCoordinator = Depends(get_event_coordinator),
) -> ClusterStatus:
    """获取集群状态"""
    try:
        status = await coordinator.get_cluster_status()
        return ClusterStatus(**status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/metrics")
async def get_monitoring_metrics(
    monitor: EventProcessingMonitor = Depends(get_event_monitor),
) -> Dict[str, Any]:
    """获取监控指标"""
    try:
        return await monitor.get_processing_stats()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/stream")
async def event_stream(
    websocket: WebSocket,
    store: EventStore = Depends(get_event_store),
):
    """WebSocket事件流"""
    await websocket.accept()

    if not getattr(store, "redis", None):
        await websocket.send_json({"type": "error", "message": "事件系统未初始化"})
        await websocket.close()
        return

    channel = f"{store.stream_prefix}pubsub"
    pubsub = store.redis.pubsub()
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

    forward_task = create_task_with_logging(forward_events(), logger=logger, keep_reference=False)

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
async def get_dead_letter_events(
    limit: int = 100,
    store: EventStore = Depends(get_event_store),
) -> List[Dict[str, Any]]:
    """获取死信队列事件"""
    try:
        return await store.get_dead_letter_events(limit)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit")
async def submit_event(
    event_type: str = Query(..., description="事件类型"),
    source: str = Query(..., description="事件源"),
    message: str = Query("", description="事件消息"),
    priority: str = Query("normal", description="优先级"),
    engine: AsyncEventProcessingEngine = Depends(get_processing_engine),
    store: EventStore = Depends(get_event_store),
    coordinator: Optional[DistributedEventCoordinator] = Depends(get_optional_event_coordinator),
) -> Dict[str, str]:
    """手动提交事件（用于测试）"""
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
        
        if coordinator:
            await coordinator.distribute_event(event)
        else:
            await engine.submit_event(event, event.priority)
            await store.append_event(event)
        
        return {"status": "success", "event_id": event.id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
