"""
用户反馈系统API端点实现

提供隐式和显式反馈收集、处理、分析的REST API接口
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Query
from pydantic import Field, field_validator, ValidationInfo
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from src.core.utils.timezone_utils import utc_now
import json
import uuid
from src.core.database import get_db, get_db_session
from src.core.utils.async_utils import create_task_with_logging
from src.models.schemas.feedback import FeedbackType, FeedbackEvent as FeedbackEventModel
from src.services.reward_generator import FeedbackNormalizer
from sqlalchemy import select, func, desc, cast, Float
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.base_model import ApiBaseModel
from src.repositories.feedback_repository import FeedbackRepository

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback"])

async def get_feedback_repo(db: AsyncSession = Depends(get_db)) -> FeedbackRepository:
    return FeedbackRepository(db)

# Pydantic模型定义
class FeedbackContext(ApiBaseModel):
    """反馈上下文信息"""
    url: str
    page_title: str = Field(default="", description="页面标题")
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    viewport: Optional[Dict[str, int]] = None
    timestamp: int
    user_agent: str

class FeedbackEvent(ApiBaseModel):
    """反馈事件模型"""
    event_id: str
    user_id: str
    session_id: str
    item_id: Optional[str] = None
    feedback_type: FeedbackType
    value: Union[int, float, str, bool]
    raw_value: Optional[Any] = None
    context: FeedbackContext
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('value')
    def validate_feedback_value(cls, v, info: ValidationInfo):
        """验证反馈值的合理性"""
        feedback_type = info.data.get('feedback_type')
        
        if feedback_type == FeedbackType.RATING:
            if not isinstance(v, (int, float)) or not (1 <= v <= 5):
                raise ValueError('评分必须在1-5之间')
        elif feedback_type in [FeedbackType.LIKE, FeedbackType.DISLIKE, FeedbackType.BOOKMARK]:
            if not isinstance(v, (int, bool)):
                raise ValueError('点赞/收藏反馈必须是布尔值或0/1')
        elif feedback_type == FeedbackType.SCROLL_DEPTH:
            if not isinstance(v, (int, float)) or not (0 <= v <= 100):
                raise ValueError('滚动深度必须在0-100%之间')
        elif feedback_type == FeedbackType.DWELL_TIME:
            if not isinstance(v, (int, float)) or v < 0:
                raise ValueError('停留时间必须为非负数')
        
        return v

class FeedbackBatch(ApiBaseModel):
    """反馈事件批次"""
    batch_id: str
    user_id: str
    session_id: str
    events: List[FeedbackEvent]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processed_at: Optional[datetime] = None

class ExplicitFeedbackRequest(ApiBaseModel):
    """显式反馈提交请求"""
    user_id: str
    session_id: str
    item_id: Optional[str] = None
    feedback_type: FeedbackType
    value: Union[int, float, str, bool]
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class FeedbackHistoryQuery(ApiBaseModel):
    """反馈历史查询参数"""
    user_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    feedback_types: Optional[List[FeedbackType]] = None
    item_id: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)

class FeedbackAnalyticsResponse(ApiBaseModel):
    """用户反馈分析响应"""
    user_id: str
    total_feedbacks: int
    feedback_distribution: Dict[FeedbackType, int]
    average_rating: Optional[float] = None
    engagement_score: float
    last_activity: Optional[datetime] = None
    preference_vector: List[float] = Field(default_factory=list)
    trust_score: float = Field(default=1.0)

class ItemFeedbackAnalyticsResponse(ApiBaseModel):
    """推荐项反馈分析响应"""
    item_id: str
    total_feedbacks: int
    average_rating: Optional[float] = None
    like_ratio: float
    engagement_metrics: Dict[str, float]
    feedback_distribution: Dict[FeedbackType, int]

class FeedbackQualityScore(ApiBaseModel):
    """反馈质量评分"""
    feedback_id: str
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_factors: Dict[str, float]
    is_valid: bool
    reasons: Optional[List[str]] = None

class ApiResponse(ApiBaseModel):
    """API统一响应格式"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None

# WebSocket连接管理器
class FeedbackWebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接已建立，当前连接数: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接已断开，当前连接数: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                self.disconnect(connection)

manager = FeedbackWebSocketManager()

def _serialize_feedback_event(event: FeedbackEventModel) -> Dict[str, Any]:
    batch_id = None
    if getattr(event, "batch", None) and getattr(event.batch, "batch_id", None):
        batch_id = event.batch.batch_id
    elif event.batch_id:
        batch_id = str(event.batch_id)
    return {
        "event_id": event.event_id,
        "batch_id": batch_id,
        "user_id": event.user_id,
        "session_id": event.session_id,
        "item_id": event.item_id,
        "feedback_type": event.feedback_type,
        "value": event.value,
        "raw_value": event.raw_value,
        "context": event.context or {},
        "metadata": event.event_metadata or {},
        "timestamp": event.timestamp.isoformat() if event.timestamp else None,
    }

def _build_event_payload(
    *,
    event_id: str,
    user_id: str,
    session_id: str,
    item_id: Optional[str],
    feedback_type: FeedbackType,
    value: Any,
    raw_value: Any = None,
    context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "user_id": user_id,
        "session_id": session_id,
        "item_id": item_id or None,
        "feedback_type": feedback_type.value,
        "value": value,
        "raw_value": raw_value,
        "context": context or {},
        "event_metadata": metadata or {},
        "timestamp": timestamp or utc_now(),
    }

def _parse_feedback_type(value: Any) -> FeedbackType:
    if isinstance(value, FeedbackType):
        return value
    try:
        return FeedbackType(str(value))
    except Exception:
        return FeedbackType.VIEW

# API端点实现
@router.post("/implicit", response_model=ApiResponse)
async def submit_implicit_feedback(
    batch: FeedbackBatch,
    repo: FeedbackRepository = Depends(get_feedback_repo),
):
    """
    提交隐式反馈批次
    """
    try:
        logger.info(f"收到隐式反馈批次: {batch.batch_id}, 事件数: {len(batch.events)}")
        
        # 验证批次数据
        if not batch.events:
            raise HTTPException(status_code=400, detail="反馈批次不能为空")
        
        if len(batch.events) > 100:
            raise HTTPException(status_code=400, detail="单次批次事件数不能超过100")

        for ev in batch.events:
            if ev.user_id != batch.user_id or ev.session_id != batch.session_id:
                raise HTTPException(status_code=400, detail="批次内事件的用户或会话不一致")

        event_ids = [ev.event_id for ev in batch.events]
        if len(event_ids) != len(set(event_ids)):
            raise HTTPException(status_code=400, detail="批次内事件ID重复")

        existing_batch = await repo.get_feedback_batch_by_batch_id(batch.batch_id)
        if existing_batch:
            return ApiResponse(
                success=True,
                message="批次已存在，忽略重复提交",
                data={
                    "batch_id": existing_batch.batch_id,
                    "stored_events": existing_batch.event_count,
                },
            )

        events_data = [
            _build_event_payload(
                event_id=ev.event_id,
                user_id=ev.user_id,
                session_id=ev.session_id,
                item_id=ev.item_id,
                feedback_type=ev.feedback_type,
                value=ev.value,
                raw_value=ev.raw_value,
                context=ev.context.model_dump(),
                metadata=ev.metadata,
            )
            for ev in batch.events
        ]

        existing_events = await repo.get_feedback_events_by_ids(event_ids)
        existing_ids = {e.event_id for e in existing_events}
        events_data = [e for e in events_data if e["event_id"] not in existing_ids]

        if not events_data:
            return ApiResponse(
                success=True,
                message="批次事件已存在，忽略重复提交",
                data={"batch_id": batch.batch_id, "stored_events": 0},
            )

        batch_data = {
            "batch_id": batch.batch_id,
            "user_id": batch.user_id,
            "session_id": batch.session_id,
            "start_time": batch.start_time,
            "end_time": batch.end_time,
            "processed_at": batch.processed_at,
        }
        created_batch = await repo.create_feedback_batch(events_data, batch_data)
        return ApiResponse(
            success=True,
            message=f"成功接收{len(events_data)}个隐式反馈事件",
            data={
                "batch_id": created_batch.batch_id,
                "stored_events": created_batch.event_count,
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交隐式反馈失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.post("/explicit", response_model=ApiResponse)
async def submit_explicit_feedback(
    feedback: ExplicitFeedbackRequest,
    repo: FeedbackRepository = Depends(get_feedback_repo),
):
    """
    提交显式反馈
    """
    try:
        logger.info(f"收到显式反馈: 用户{feedback.user_id}, 类型{feedback.feedback_type}")
        
        context_data = feedback.context or {}
        context = FeedbackContext(
            url=context_data.get('url', ''),
            page_title=context_data.get('page_title', ''),
            element_id=context_data.get('element_id'),
            element_type=context_data.get('element_type'),
            coordinates=context_data.get('coordinates'),
            viewport=context_data.get('viewport'),
            timestamp=int(context_data.get('timestamp') or (utc_now().timestamp() * 1000)),
            user_agent=context_data.get('user_agent', '')
        )
        event_id = f"explicit-{uuid.uuid4().hex}"
        created_event = await repo.create_feedback_event(
            _build_event_payload(
                event_id=event_id,
                user_id=feedback.user_id,
                session_id=feedback.session_id,
                item_id=feedback.item_id,
                feedback_type=feedback.feedback_type,
                value=feedback.value,
                raw_value=None,
                context=context.model_dump(),
                metadata=feedback.metadata,
            )
        )
        return ApiResponse(
            success=True,
            message="显式反馈提交成功",
            data={"event_id": created_event.event_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交显式反馈失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.get("/user/{user_id}", response_model=ApiResponse)
async def get_user_feedback_history(
    user_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    feedback_types: Optional[str] = None,
    item_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    repo: FeedbackRepository = Depends(get_feedback_repo),
):
    """
    获取用户反馈历史
    """
    try:
        # 解析反馈类型
        parsed_types = None
        if feedback_types:
            try:
                parsed_types = [FeedbackType(t.strip()) for t in feedback_types.split(',')]
            except Exception:
                raise HTTPException(status_code=400, detail="无效的反馈类型")
        
        query = FeedbackHistoryQuery(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            feedback_types=parsed_types,
            item_id=item_id,
            limit=limit,
            offset=offset
        )
        type_values = [t.value for t in parsed_types] if parsed_types else None
        events = await repo.get_feedback_events(
            user_id=user_id,
            item_id=item_id,
            feedback_types=type_values,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        total = await repo.count_feedback_events(
            user_id=user_id,
            item_id=item_id,
            feedback_types=type_values,
            start_date=start_date,
            end_date=end_date,
        )
        items = [_serialize_feedback_event(ev) for ev in events]
        
        return ApiResponse(
            success=True,
            data={
                "total": total,
                "items": items,
                "query": query.model_dump()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户反馈历史失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/analytics/user/{user_id}", response_model=ApiResponse)
async def get_user_feedback_analytics(
    user_id: str,
    repo: FeedbackRepository = Depends(get_feedback_repo),
):
    """
    获取用户反馈分析
    """
    try:
        analytics = await get_user_analytics(user_id, repo)
        
        return ApiResponse(
            success=True,
            data=analytics.model_dump()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户反馈分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@router.get("/analytics/item/{item_id}", response_model=ApiResponse)
async def get_item_feedback_analytics(
    item_id: str,
    repo: FeedbackRepository = Depends(get_feedback_repo),
):
    """
    获取推荐项反馈分析
    """
    try:
        analytics = await get_item_analytics(item_id, repo)
        
        return ApiResponse(
            success=True,
            data=analytics.model_dump()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取推荐项反馈分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@router.post("/quality/score", response_model=ApiResponse)
async def get_feedback_quality_score(
    feedback_ids: List[str],
    repo: FeedbackRepository = Depends(get_feedback_repo),
):
    """
    获取反馈质量评分
    """
    try:
        if not feedback_ids:
            raise HTTPException(status_code=400, detail="反馈ID列表不能为空")
        
        if len(feedback_ids) > 50:
            raise HTTPException(status_code=400, detail="单次查询ID数不能超过50")
        
        scores = await calculate_quality_scores(feedback_ids, repo)
        
        return ApiResponse(
            success=True,
            data=scores
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取反馈质量评分失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"计算失败: {str(e)}")

@router.post("/process/batch", response_model=ApiResponse)
async def process_feedback_batch_endpoint(
    batch_id: str,
    background_tasks: BackgroundTasks
):
    """
    手动触发反馈批次处理
    """
    try:
        background_tasks.add_task(reprocess_feedback_batch, batch_id)
        
        return ApiResponse(
            success=True,
            message=f"批次{batch_id}处理任务已启动"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理反馈批次失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.post("/reward/calculate", response_model=ApiResponse)
async def calculate_reward_signal(
    user_id: str,
    item_id: str,
    time_window: int = 3600,  # 默认1小时
    repo: FeedbackRepository = Depends(get_feedback_repo),
):
    """
    计算奖励信号
    """
    try:
        reward = await compute_reward_signal(user_id, item_id, time_window, repo)
        
        return ApiResponse(
            success=True,
            data={
                "user_id": user_id,
                "item_id": item_id,
                "reward_signal": reward,
                "time_window": time_window,
                "calculated_at": utc_now().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算奖励信号失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"计算失败: {str(e)}")

@router.get("/overview", response_model=ApiResponse)
async def get_feedback_overview(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    获取反馈系统概览
    """
    try:
        overview = await get_system_overview(start_date, end_date, db)
        
        return ApiResponse(
            success=True,
            data=overview
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取反馈系统概览失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/metrics/realtime", response_model=ApiResponse)
async def get_realtime_feedback_metrics(
    db: AsyncSession = Depends(get_db),
):
    """
    获取实时反馈指标
    """
    try:
        metrics = await get_realtime_metrics(db)
        
        return ApiResponse(
            success=True,
            data=metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实时反馈指标失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    反馈系统WebSocket端点
    用于实时传输反馈事件
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({'type': 'error', 'message': '消息格式无效'}),
                    websocket
                )
                continue
            if not isinstance(message, dict):
                await manager.send_personal_message(
                    json.dumps({'type': 'error', 'message': '消息体必须是对象'}),
                    websocket
                )
                continue
            
            if message.get('type') == 'feedback_batch':
                # 处理实时反馈批次
                events = message.get('events', [])
                logger.info(f"WebSocket收到{len(events)}个反馈事件")
                
                # 异步处理事件
                create_task_with_logging(process_websocket_feedback(events), logger=logger)
                
                # 确认收到
                await manager.send_personal_message(
                    json.dumps({
                        'type': 'ack',
                        'message': f'已收到{len(events)}个事件'
                    }),
                    websocket
                )
            elif message.get('type') == 'ping':
                await manager.send_personal_message(
                    json.dumps({'type': 'pong'}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket)

# 辅助函数实现
async def process_feedback_batch(batch: FeedbackBatch):
    """处理反馈批次"""
    try:
        events = batch.events
        logger.info(f"开始处理反馈批次: {batch.batch_id}, 数量: {len(events)}")

        if not events:
            logger.info(f"反馈批次为空: {batch.batch_id}")
            return

        async with get_db_session() as db:
            repo = FeedbackRepository(db)
            existing_batch = await repo.get_feedback_batch_by_batch_id(batch.batch_id)
            if existing_batch:
                logger.info(f"反馈批次已存在: {batch.batch_id}")
                return

            events_data = [
                _build_event_payload(
                    event_id=ev.event_id,
                    user_id=ev.user_id,
                    session_id=ev.session_id,
                    item_id=ev.item_id,
                    feedback_type=ev.feedback_type,
                    value=ev.value,
                    raw_value=ev.raw_value,
                    context=ev.context.model_dump(),
                    metadata=ev.metadata,
                )
                for ev in events
            ]
            batch_data = {
                "batch_id": batch.batch_id,
                "user_id": batch.user_id,
                "session_id": batch.session_id,
                "start_time": batch.start_time,
                "end_time": batch.end_time,
                "processed_at": batch.processed_at,
            }
            await repo.create_feedback_batch(events_data, batch_data)

        logger.info(f"反馈批次处理完成: {batch.batch_id}")
    except Exception as e:
        logger.error(f"处理反馈批次失败: {batch.batch_id}, 错误: {e}", exc_info=True)

async def process_single_feedback(event: FeedbackEvent):
    """处理单个反馈事件"""
    try:
        logger.info(f"开始处理反馈事件: {event.event_id}")
        async with get_db_session() as db:
            repo = FeedbackRepository(db)
            existing = await repo.get_feedback_events_by_ids([event.event_id])
            if existing:
                logger.info(f"反馈事件已存在: {event.event_id}")
                return
            await repo.create_feedback_event(
                _build_event_payload(
                    event_id=event.event_id,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    item_id=event.item_id,
                    feedback_type=event.feedback_type,
                    value=event.value,
                    raw_value=event.raw_value,
                    context=event.context.model_dump(),
                    metadata=event.metadata,
                )
            )
        logger.info(f"反馈事件处理完成: {event.event_id}")
    except Exception as e:
        logger.error(f"处理反馈事件失败: {event.event_id}, 错误: {e}", exc_info=True)

async def process_websocket_feedback(events: List[Dict]):
    """处理WebSocket反馈事件"""
    try:
        logger.info(f"处理WebSocket反馈事件: {len(events)}")

        payloads: List[Dict[str, Any]] = []
        for event_data in events:
            try:
                feedback_type = _parse_feedback_type(event_data.get("feedback_type"))
                payloads.append(
                    _build_event_payload(
                        event_id=event_data.get("event_id") or f"ws-{uuid.uuid4().hex}",
                        user_id=event_data.get("user_id", ""),
                        session_id=event_data.get("session_id", ""),
                        item_id=event_data.get("item_id"),
                        feedback_type=feedback_type,
                        value=event_data.get("value", 0),
                        raw_value=event_data.get("raw_value"),
                        context=event_data.get("context") or {},
                        metadata=event_data.get("metadata"),
                    )
                )
            except Exception as e:
                logger.error(f"解析WebSocket事件失败: {e}", exc_info=True)

        if payloads:
            async with get_db_session() as db:
                repo = FeedbackRepository(db)
                await repo.create_feedback_events(payloads)
    except Exception as e:
        logger.error(f"处理WebSocket反馈失败: {e}", exc_info=True)

async def get_user_analytics(user_id: str, repo: FeedbackRepository) -> FeedbackAnalyticsResponse:
    """获取用户分析数据"""
    stats = await repo.get_user_feedback_stats(user_id)
    total = int(stats.get("total_feedbacks") or 0)
    distribution_raw = stats.get("type_distribution") or {}
    distribution: Dict[FeedbackType, int] = {}
    for key, value in distribution_raw.items():
        try:
            distribution[FeedbackType(key)] = int(value)
        except Exception:
            continue

    engagement_score = min(1.0, total / 200) if total else 0.0
    total_safe = max(total, 1)
    preference_vector = [distribution.get(ft, 0) / total_safe for ft in FeedbackType]
    return FeedbackAnalyticsResponse(
        user_id=user_id,
        total_feedbacks=total,
        feedback_distribution={k: v for k, v in distribution.items()},
        average_rating=stats.get("average_rating"),
        engagement_score=engagement_score,
        last_activity=stats.get("last_feedback_time"),
        preference_vector=preference_vector,
        trust_score=float(stats.get("trust_score") or 0.0),
    )

async def get_item_analytics(item_id: str, repo: FeedbackRepository) -> ItemFeedbackAnalyticsResponse:
    """获取推荐项分析数据"""
    stats = await repo.get_item_feedback_stats(item_id)
    total = int(stats.get("total_feedbacks") or 0)
    distribution_raw = stats.get("type_distribution") or {}
    likes = distribution_raw.get(FeedbackType.LIKE.value, 0)
    dislikes = distribution_raw.get(FeedbackType.DISLIKE.value, 0)
    like_ratio = stats.get("like_ratio")
    if like_ratio is None:
        like_ratio = likes / max(likes + dislikes, 1)
    click_count = distribution_raw.get(FeedbackType.CLICK.value, 0)
    view_count = distribution_raw.get(FeedbackType.VIEW.value, 0)
    interaction_rate = (click_count + likes + dislikes) / max(total, 1)
    engagement_metrics = {
        "click_through_rate": click_count / max(view_count, 1),
        "dwell_time_avg": float(stats.get("dwell_time_avg") or 0.0),
        "scroll_depth_avg": float(stats.get("scroll_depth_avg") or 0.0),
        "interaction_rate": interaction_rate,
        "completion_rate": view_count / max(total, 1),
        "bounce_rate": 1 - interaction_rate if total else 0.0,
    }
    distribution: Dict[FeedbackType, int] = {}
    for key, value in distribution_raw.items():
        try:
            distribution[FeedbackType(key)] = int(value)
        except Exception:
            continue
    return ItemFeedbackAnalyticsResponse(
        item_id=item_id,
        total_feedbacks=total,
        average_rating=stats.get("average_rating"),
        like_ratio=like_ratio,
        engagement_metrics=engagement_metrics,
        feedback_distribution=distribution,
    )

async def calculate_quality_scores(
    feedback_ids: List[str],
    repo: FeedbackRepository,
) -> List[FeedbackQualityScore]:
    """计算反馈质量评分"""
    if not feedback_ids:
        return []
    events = await repo.get_feedback_events_by_ids(feedback_ids)
    if not events:
        return []

    user_ids = {e.user_id for e in events if e.user_id}
    user_stats = await repo.get_user_basic_stats(list(user_ids))

    quality_scores = []
    now = utc_now()
    total_type_count = len(list(FeedbackType)) or 1

    for ev in events:
        feedback_id = ev.event_id
        if not feedback_id:
            continue

        ts: datetime = ev.timestamp or now
        hours = (now - ts).total_seconds() / 3600
        timing = 1.0 if hours <= 24 else max(0.2, 1 - hours / 240)

        context = ev.context or {}
        context_relevance = 1.0 if context.get("url") else 0.6

        feedback_type = _parse_feedback_type(ev.feedback_type)
        normalized_val = FeedbackNormalizer.normalize_feedback_value(
            feedback_type,
            ev.value,
        )

        authenticity = 1.0 if ev.session_id else 0.7

        stats = user_stats.get(ev.user_id or "", {})
        frequency = min(1.0, (stats.get("total", 0) / 50)) if stats else 0.0
        diversity = min(1.0, (stats.get("types", 0) / total_type_count)) if stats else 0.0

        quality_factors = {
            "consistency": float(normalized_val),
            "timing": float(timing),
            "context_relevance": float(context_relevance),
            "frequency": float(frequency),
            "diversity": float(diversity),
            "authenticity": float(authenticity),
        }
        overall = sum(quality_factors.values()) / len(quality_factors)
        reasons = []
        if timing < 0.5:
            reasons.append("反馈过旧")
        if context_relevance < 0.7:
            reasons.append("缺少上下文信息")
        if normalized_val < 0.3:
            reasons.append("反馈信号较弱")
        if not reasons:
            reasons.append("高质量反馈")

        quality_scores.append(
            FeedbackQualityScore(
                feedback_id=feedback_id,
                quality_score=round(overall, 3),
                quality_factors=quality_factors,
                is_valid=overall >= 0.5,
                reasons=reasons,
            )
        )

    return quality_scores

async def reprocess_feedback_batch(batch_id: str):
    """重新处理反馈批次"""
    async with get_db_session() as db:
        repo = FeedbackRepository(db)
        events = await repo.get_feedback_events_by_batch_id(batch_id)

    if not events:
        logger.info(f"批次无事件: {batch_id}")
        return

    from src.services.feedback_processor import feedback_processor

    payloads = []
    for ev in events:
        payloads.append({
            "feedback_id": ev.event_id,
            "user_id": ev.user_id,
            "item_id": ev.item_id or "unknown",
            "feedback_type": ev.feedback_type,
            "value": ev.value,
            "timestamp": ev.timestamp or utc_now(),
            "context": ev.context or {},
            "metadata": ev.event_metadata or {},
        })

    await feedback_processor.process_feedback_batch(payloads)

async def compute_reward_signal(
    user_id: str,
    item_id: str,
    time_window: int,
    repo: FeedbackRepository,
) -> float:
    """计算奖励信号"""
    cutoff = utc_now() - timedelta(seconds=time_window)
    events = await repo.get_feedback_events(
        user_id=user_id,
        item_id=item_id,
        start_date=cutoff,
        limit=None,
    )
    if not events:
        return 0.0
    weights = {
        FeedbackType.RATING: 0.8,
        FeedbackType.LIKE: 0.6,
        FeedbackType.DISLIKE: -0.4,
        FeedbackType.BOOKMARK: 0.7,
        FeedbackType.SHARE: 0.9,
        FeedbackType.COMMENT: 0.5,
        FeedbackType.CLICK: 0.2,
        FeedbackType.VIEW: 0.1,
        FeedbackType.DWELL_TIME: 0.3,
        FeedbackType.SCROLL_DEPTH: 0.2
    }
    total = 0.0
    weight_sum = 0.0
    for ev in events:
        ftype = _parse_feedback_type(ev.feedback_type)
        weight = weights.get(ftype, 0.1)
        normalized = FeedbackNormalizer.normalize_feedback_value(ftype, ev.value)
        total += weight * normalized
        weight_sum += abs(weight)
    if weight_sum == 0:
        return 0.0
    reward = total / weight_sum
    return max(-1.0, min(1.0, reward))

async def get_system_overview(
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    db: AsyncSession,
) -> Dict:
    """获取系统概览"""
    filters = [FeedbackEventModel.valid.is_(True)]
    if start_date:
        filters.append(FeedbackEventModel.timestamp >= start_date)
    if end_date:
        filters.append(FeedbackEventModel.timestamp <= end_date)

    total_result = await db.execute(
        select(func.count(FeedbackEventModel.id)).where(*filters)
    )
    total = int(total_result.scalar() or 0)

    user_result = await db.execute(
        select(func.count(func.distinct(FeedbackEventModel.user_id))).where(*filters)
    )
    unique_users = int(user_result.scalar() or 0)

    type_result = await db.execute(
        select(
            FeedbackEventModel.feedback_type,
            func.count(FeedbackEventModel.id).label("count"),
        )
        .where(*filters)
        .group_by(FeedbackEventModel.feedback_type)
    )
    counts = {r.feedback_type: int(r.count) for r in type_result.all()}

    rating_result = await db.execute(
        select(
            func.avg(cast(FeedbackEventModel.value, Float)).label("avg_rating"),
            func.count(FeedbackEventModel.id).label("rating_count"),
        ).where(
            *filters,
            FeedbackEventModel.feedback_type == FeedbackType.RATING.value
        )
    )
    rating_stats = rating_result.one()
    average_rating = (
        float(rating_stats.avg_rating) if rating_stats.avg_rating is not None else None
    )

    like_result = await db.execute(
        select(func.count(FeedbackEventModel.id)).where(
            *filters,
            FeedbackEventModel.feedback_type == FeedbackType.LIKE.value
        )
    )
    dislike_result = await db.execute(
        select(func.count(FeedbackEventModel.id)).where(
            *filters,
            FeedbackEventModel.feedback_type == FeedbackType.DISLIKE.value
        )
    )
    like_count = int(like_result.scalar() or 0)
    dislike_count = int(dislike_result.scalar() or 0)
    like_total = like_count + dislike_count
    positive_ratio = like_count / like_total if like_total > 0 else None

    item_filters = filters + [
        FeedbackEventModel.item_id.is_not(None),
        FeedbackEventModel.item_id != "",
    ]
    item_count = func.count(FeedbackEventModel.id).label("count")
    item_result = await db.execute(
        select(FeedbackEventModel.item_id, item_count)
        .where(*item_filters)
        .group_by(FeedbackEventModel.item_id)
        .order_by(desc(item_count))
        .limit(5)
    )
    top_items = [
        {"item_id": r.item_id, "feedback_count": int(r.count)}
        for r in item_result.all()
    ]

    return {
        "total_feedbacks": total,
        "feedback_types": counts,
        "unique_users": unique_users,
        "average_quality_score": None,
        "average_rating": average_rating,
        "positive_ratio": positive_ratio,
        "top_items": top_items
    }

async def get_realtime_metrics(db: AsyncSession) -> Dict:
    """获取实时指标"""
    now = utc_now()
    window_start = now - timedelta(minutes=5)

    recent_stmt = select(
        func.count(FeedbackEventModel.id).label("recent_count"),
        func.count(func.distinct(FeedbackEventModel.session_id)).label("active_sessions"),
    ).where(
        FeedbackEventModel.timestamp >= window_start,
        FeedbackEventModel.timestamp <= now,
        FeedbackEventModel.valid.is_(True),
    )
    recent_row = (await db.execute(recent_stmt)).one()
    recent_count = int(recent_row.recent_count or 0)
    active_sessions = int(recent_row.active_sessions or 0)

    total_result = await db.execute(
        select(func.count(FeedbackEventModel.id)).where(FeedbackEventModel.valid.is_(True))
    )
    total_processed = int(total_result.scalar() or 0)

    events_per_minute = recent_count / 5 if recent_count else 0
    return {
        "active_sessions": active_sessions,
        "events_per_minute": events_per_minute,
        "buffer_status": {
            "pending_events": 0,
            "processed_events": total_processed,
            "failed_events": 0,
            "buffer_utilization": 0.0
        },
        "processing_latency": 0.0,
        "quality_score": 0.0,
        "anomaly_count": 0,
        "last_updated": now.isoformat()
    }
