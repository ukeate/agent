"""
用户反馈系统API端点实现

提供隐式和显式反馈收集、处理、分析的REST API接口
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer
from pydantic import Field, field_validator, ValidationInfo
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import json
import asyncio
from src.core.database import get_db_session
from src.models.schemas.feedback import FeedbackType
from src.services.reward_generator import FeedbackNormalizer
from sqlalchemy import text, bindparam
from src.api.base_model import ApiBaseModel

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback"])
security = HTTPBearer()

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
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")

manager = FeedbackWebSocketManager()
_feedback_tables_initialized = False
_feedback_tables_lock = asyncio.Lock()

async def _ensure_feedback_tables(db):
    global _feedback_tables_initialized
    if _feedback_tables_initialized:
        return
    async with _feedback_tables_lock:
        if _feedback_tables_initialized:
            return

        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback_events (
                id BIGSERIAL PRIMARY KEY,
                event_id TEXT UNIQUE NOT NULL,
                batch_id TEXT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                item_id TEXT,
                feedback_type TEXT NOT NULL,
                value JSONB NOT NULL,
                raw_value JSONB,
                context JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        await db.execute(text("CREATE INDEX IF NOT EXISTS idx_feedback_events_user_time ON feedback_events (user_id, timestamp DESC)"))
        await db.execute(text("CREATE INDEX IF NOT EXISTS idx_feedback_events_item_time ON feedback_events (item_id, timestamp DESC)"))
        await db.execute(text("CREATE INDEX IF NOT EXISTS idx_feedback_events_type_time ON feedback_events (feedback_type, timestamp DESC)"))
        await db.execute(text("CREATE INDEX IF NOT EXISTS idx_feedback_events_batch ON feedback_events (batch_id)"))
        await db.commit()

        _feedback_tables_initialized = True

async def _insert_events(db, events: List[Dict[str, Any]]):
    if not events:
        return
    await _ensure_feedback_tables(db)
    stmt = text("""
        INSERT INTO feedback_events (
            event_id, batch_id, user_id, session_id, item_id, feedback_type,
            value, raw_value, context, metadata, timestamp
        )
        VALUES (
            :event_id, :batch_id, :user_id, :session_id, :item_id, :feedback_type,
            CAST(:value AS jsonb),
            CAST(:raw_value AS jsonb),
            CAST(:context AS jsonb),
            CAST(:metadata AS jsonb),
            :timestamp
        )
        ON CONFLICT (event_id) DO NOTHING
    """)
    await db.execute(stmt, events)

async def _store_event(event: Dict[str, Any]):
    async with get_db_session() as db:
        await _insert_events(db, [{
            "event_id": event.get("event_id"),
            "batch_id": event.get("batch_id"),
            "user_id": event.get("user_id"),
            "session_id": event.get("session_id"),
            "item_id": event.get("item_id") or None,
            "feedback_type": event.get("feedback_type"),
            "value": json.dumps(event.get("value")),
            "raw_value": json.dumps(event.get("raw_value")) if event.get("raw_value") is not None else None,
            "context": json.dumps(event.get("context") or {}),
            "metadata": json.dumps(event.get("metadata") or {}),
            "timestamp": event.get("timestamp") or utc_now(),
        }])
        await db.commit()

async def _fetch_events(
    user_id: Optional[str] = None,
    item_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    types: Optional[List[FeedbackType]] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    where = []
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if user_id:
        where.append("user_id = :user_id")
        params["user_id"] = user_id
    if item_id:
        where.append("item_id = :item_id")
        params["item_id"] = item_id
    if start:
        where.append("timestamp >= :start")
        params["start"] = start
    if end:
        where.append("timestamp <= :end")
        params["end"] = end

    stmt_types = None
    if types:
        where.append("feedback_type IN :types")
        params["types"] = [t.value for t in types]
        stmt_types = bindparam("types", expanding=True)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    count_sql = f"SELECT COUNT(*) AS total FROM feedback_events {where_sql}"
    data_sql = f"""
        SELECT event_id, batch_id, user_id, session_id, item_id, feedback_type,
               value, raw_value, context, metadata, timestamp
        FROM feedback_events
        {where_sql}
        ORDER BY timestamp DESC
        OFFSET :offset
        LIMIT :limit
    """

    count_stmt = text(count_sql)
    data_stmt = text(data_sql)
    if stmt_types is not None:
        count_stmt = count_stmt.bindparams(stmt_types)
        data_stmt = data_stmt.bindparams(stmt_types)

    async with get_db_session() as db:
        await _ensure_feedback_tables(db)
        total_result = await db.execute(count_stmt, params)
        total = int(total_result.scalar() or 0)
        rows = await db.execute(data_stmt, params)
        items = [dict(r) for r in rows.mappings().all()]
        return {"total": total, "items": items}

# API端点实现
@router.post("/implicit", response_model=ApiResponse)
async def submit_implicit_feedback(
    batch: FeedbackBatch,
    background_tasks: BackgroundTasks
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
        
        events_data = []
        for ev in batch.events:
            events_data.append({
                "event_id": ev.event_id,
                "batch_id": batch.batch_id,
                "user_id": ev.user_id,
                "session_id": ev.session_id,
                "item_id": ev.item_id or "",
                "feedback_type": ev.feedback_type.value,
                "value": ev.value,
                "raw_value": ev.raw_value,
                "context": ev.context.model_dump(),
                "metadata": ev.metadata or {},
                "timestamp": utc_now(),
            })
        await asyncio.gather(*[_store_event(e) for e in events_data])
        return ApiResponse(
            success=True,
            message=f"成功接收{len(batch.events)}个隐式反馈事件",
            data={"batch_id": batch.batch_id}
        )
        
    except Exception as e:
        logger.error(f"提交隐式反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.post("/explicit", response_model=ApiResponse)
async def submit_explicit_feedback(
    feedback: ExplicitFeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    提交显式反馈
    """
    try:
        logger.info(f"收到显式反馈: 用户{feedback.user_id}, 类型{feedback.feedback_type}")
        
        # 创建反馈事件
        event = FeedbackEvent(
            event_id=f"explicit-{int(utc_now().timestamp() * 1000)}",
            user_id=feedback.user_id,
            session_id=feedback.session_id,
            item_id=feedback.item_id,
            feedback_type=feedback.feedback_type,
            value=feedback.value,
            context=FeedbackContext(
                url=feedback.context.get('url', '') if feedback.context else '',
                page_title=feedback.context.get('page_title', '') if feedback.context else '',
                timestamp=int(utc_now().timestamp() * 1000),
                user_agent=feedback.context.get('user_agent', '') if feedback.context else ''
            ),
            metadata=feedback.metadata
        )
        
        await _store_event({
            "event_id": event.event_id,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "item_id": event.item_id or "",
            "feedback_type": event.feedback_type.value,
            "value": event.value,
            "raw_value": event.raw_value,
            "context": event.context.model_dump(),
            "metadata": event.metadata or {},
            "timestamp": utc_now(),
        })
        return ApiResponse(
            success=True,
            message="显式反馈提交成功",
            data={"event_id": event.event_id}
        )
        
    except Exception as e:
        logger.error(f"提交显式反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.get("/user/{user_id}", response_model=ApiResponse)
async def get_user_feedback_history(
    user_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    feedback_types: Optional[str] = None,
    item_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    获取用户反馈历史
    """
    try:
        # 解析反馈类型
        parsed_types = None
        if feedback_types:
            parsed_types = [FeedbackType(t.strip()) for t in feedback_types.split(',')]
        
        query = FeedbackHistoryQuery(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            feedback_types=parsed_types,
            item_id=item_id,
            limit=limit,
            offset=offset
        )
        result = await _fetch_events(
            user_id=user_id,
            item_id=item_id,
            start=start_date,
            end=end_date,
            types=parsed_types,
            limit=limit,
            offset=offset,
        )
        
        return ApiResponse(
            success=True,
            data={
                "total": result["total"],
                "items": result["items"],
                "query": query.model_dump()
            }
        )
        
    except Exception as e:
        logger.error(f"获取用户反馈历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/analytics/user/{user_id}", response_model=ApiResponse)
async def get_user_feedback_analytics(user_id: str):
    """
    获取用户反馈分析
    """
    try:
        analytics = await get_user_analytics(user_id)
        
        return ApiResponse(
            success=True,
            data=analytics.model_dump()
        )
        
    except Exception as e:
        logger.error(f"获取用户反馈分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@router.get("/analytics/item/{item_id}", response_model=ApiResponse)
async def get_item_feedback_analytics(item_id: str):
    """
    获取推荐项反馈分析
    """
    try:
        analytics = await get_item_analytics(item_id)
        
        return ApiResponse(
            success=True,
            data=analytics.model_dump()
        )
        
    except Exception as e:
        logger.error(f"获取推荐项反馈分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@router.post("/quality/score", response_model=ApiResponse)
async def get_feedback_quality_score(feedback_ids: List[str]):
    """
    获取反馈质量评分
    """
    try:
        if not feedback_ids:
            raise HTTPException(status_code=400, detail="反馈ID列表不能为空")
        
        if len(feedback_ids) > 50:
            raise HTTPException(status_code=400, detail="单次查询ID数不能超过50")
        
        scores = await calculate_quality_scores(feedback_ids)
        
        return ApiResponse(
            success=True,
            data=scores
        )
        
    except Exception as e:
        logger.error(f"获取反馈质量评分失败: {e}")
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
        
    except Exception as e:
        logger.error(f"处理反馈批次失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.post("/reward/calculate", response_model=ApiResponse)
async def calculate_reward_signal(
    user_id: str,
    item_id: str,
    time_window: int = 3600  # 默认1小时
):
    """
    计算奖励信号
    """
    try:
        reward = await compute_reward_signal(user_id, item_id, time_window)
        
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
        
    except Exception as e:
        logger.error(f"计算奖励信号失败: {e}")
        raise HTTPException(status_code=500, detail=f"计算失败: {str(e)}")

@router.get("/overview", response_model=ApiResponse)
async def get_feedback_overview(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """
    获取反馈系统概览
    """
    try:
        overview = await get_system_overview(start_date, end_date)
        
        return ApiResponse(
            success=True,
            data=overview
        )
        
    except Exception as e:
        logger.error(f"获取反馈系统概览失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/metrics/realtime", response_model=ApiResponse)
async def get_realtime_feedback_metrics():
    """
    获取实时反馈指标
    """
    try:
        metrics = await get_realtime_metrics()
        
        return ApiResponse(
            success=True,
            data=metrics
        )
        
    except Exception as e:
        logger.error(f"获取实时反馈指标失败: {e}")
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
            message = json.loads(data)
            
            if message.get('type') == 'feedback_batch':
                # 处理实时反馈批次
                events = message.get('events', [])
                logger.info(f"WebSocket收到{len(events)}个反馈事件")
                
                # 异步处理事件
                asyncio.create_task(process_websocket_feedback(events))
                
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
        await asyncio.gather(*[_store_event({
            "event_id": ev.event_id,
            "batch_id": batch.batch_id,
            "user_id": ev.user_id,
            "session_id": ev.session_id,
            "item_id": ev.item_id or "",
            "feedback_type": ev.feedback_type.value,
            "value": ev.value,
            "raw_value": ev.raw_value,
            "context": ev.context.model_dump(),
            "metadata": ev.metadata or {},
            "timestamp": utc_now(),
        }) for ev in events])
        logger.info(f"反馈批次处理完成: {batch.batch_id}")
    except Exception as e:
        logger.error(f"处理反馈批次失败: {batch.batch_id}, 错误: {e}")

async def process_single_feedback(event: FeedbackEvent):
    """处理单个反馈事件"""
    try:
        logger.info(f"开始处理反馈事件: {event.event_id}")
        await _store_event({
            "event_id": event.event_id,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "item_id": event.item_id or "",
            "feedback_type": event.feedback_type.value,
            "value": event.value,
            "raw_value": event.raw_value,
            "context": event.context.model_dump(),
            "metadata": event.metadata or {},
            "timestamp": utc_now(),
        })
        logger.info(f"反馈事件处理完成: {event.event_id}")
    except Exception as e:
        logger.error(f"处理反馈事件失败: {event.event_id}, 错误: {e}")

async def process_websocket_feedback(events: List[Dict]):
    """处理WebSocket反馈事件"""
    try:
        logger.info(f"处理WebSocket反馈事件: {len(events)}")
        
        store_jobs = []
        for event_data in events:
            try:
                store_jobs.append(_store_event({
                    "event_id": event_data.get("event_id") or f"ws-{utc_now().timestamp()}",
                    "user_id": event_data.get("user_id", ""),
                    "session_id": event_data.get("session_id", ""),
                    "item_id": event_data.get("item_id", ""),
                    "feedback_type": event_data.get("feedback_type", FeedbackType.VIEW.value),
                    "value": event_data.get("value", 0),
                    "raw_value": event_data.get("raw_value"),
                    "context": event_data.get("context", {}),
                    "metadata": event_data.get("metadata", {}),
                    "timestamp": utc_now(),
                }))
            except Exception as e:
                logger.error(f"解析WebSocket事件失败: {e}")
        if store_jobs:
            await asyncio.gather(*store_jobs)
        
    except Exception as e:
        logger.error(f"处理WebSocket反馈失败: {e}")

async def get_feedback_history(query: FeedbackHistoryQuery) -> List[Dict]:
    """获取反馈历史"""
    result = await _fetch_events(
        user_id=query.user_id,
        item_id=query.item_id,
        start=query.start_date,
        end=query.end_date,
        types=query.feedback_types,
        limit=query.limit,
        offset=query.offset,
    )
    return result["items"]

async def get_user_analytics(user_id: str) -> FeedbackAnalyticsResponse:
    """获取用户分析数据"""
    async with get_db_session() as db:
        await _ensure_feedback_tables(db)

        total_result = await db.execute(
            text("SELECT COUNT(*) FROM feedback_events WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        total = int(total_result.scalar() or 0)

        dist_result = await db.execute(
            text(
                """
                SELECT feedback_type, COUNT(*) AS count
                FROM feedback_events
                WHERE user_id = :user_id
                GROUP BY feedback_type
                """
            ),
            {"user_id": user_id},
        )
        distribution: Dict[FeedbackType, int] = {}
        for row in dist_result.mappings().all():
            try:
                distribution[FeedbackType(row["feedback_type"])] = int(row["count"])
            except Exception:
                continue

        rating_result = await db.execute(
            text(
                """
                SELECT AVG((value #>> '{}')::double precision) AS avg_rating
                FROM feedback_events
                WHERE user_id = :user_id AND feedback_type = :ftype
                """
            ),
            {"user_id": user_id, "ftype": FeedbackType.RATING.value},
        )
        average_rating = rating_result.scalar()

        last_result = await db.execute(
            text("SELECT MAX(timestamp) FROM feedback_events WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        last_ts = last_result.scalar()

        trust_result = await db.execute(
            text(
                """
                SELECT COUNT(DISTINCT session_id)::double precision / NULLIF(COUNT(*), 0) AS trust
                FROM feedback_events
                WHERE user_id = :user_id
                """
            ),
            {"user_id": user_id},
        )
        trust_score = float(trust_result.scalar() or 0.0)

    engagement_score = min(1.0, total / 200) if total else 0.0
    total_safe = max(total, 1)
    preference_vector = [distribution.get(ft, 0) / total_safe for ft in FeedbackType]
    return FeedbackAnalyticsResponse(
        user_id=user_id,
        total_feedbacks=total,
        feedback_distribution={k: v for k, v in distribution.items()},
        average_rating=float(average_rating) if average_rating is not None else None,
        engagement_score=engagement_score,
        last_activity=last_ts,
        preference_vector=preference_vector,
        trust_score=trust_score
    )

async def get_item_analytics(item_id: str) -> ItemFeedbackAnalyticsResponse:
    """获取推荐项分析数据"""
    async with get_db_session() as db:
        await _ensure_feedback_tables(db)

        total_result = await db.execute(
            text("SELECT COUNT(*) FROM feedback_events WHERE item_id = :item_id"),
            {"item_id": item_id},
        )
        total = int(total_result.scalar() or 0)

        dist_result = await db.execute(
            text(
                """
                SELECT feedback_type, COUNT(*) AS count
                FROM feedback_events
                WHERE item_id = :item_id
                GROUP BY feedback_type
                """
            ),
            {"item_id": item_id},
        )
        distribution: Dict[str, int] = {row["feedback_type"]: int(row["count"]) for row in dist_result.mappings().all()}

        rating_result = await db.execute(
            text(
                """
                SELECT AVG((value #>> '{}')::double precision) AS avg_rating
                FROM feedback_events
                WHERE item_id = :item_id AND feedback_type = :ftype
                """
            ),
            {"item_id": item_id, "ftype": FeedbackType.RATING.value},
        )
        average_rating = rating_result.scalar()

        dwell_result = await db.execute(
            text(
                """
                SELECT AVG((value #>> '{}')::double precision) AS avg_dwell
                FROM feedback_events
                WHERE item_id = :item_id AND feedback_type = :ftype
                """
            ),
            {"item_id": item_id, "ftype": FeedbackType.DWELL_TIME.value},
        )
        dwell_time_avg = float(dwell_result.scalar() or 0.0)

        scroll_result = await db.execute(
            text(
                """
                SELECT AVG((value #>> '{}')::double precision) AS avg_scroll
                FROM feedback_events
                WHERE item_id = :item_id AND feedback_type = :ftype
                """
            ),
            {"item_id": item_id, "ftype": FeedbackType.SCROLL_DEPTH.value},
        )
        scroll_depth_avg = float(scroll_result.scalar() or 0.0)

    likes = distribution.get(FeedbackType.LIKE.value, 0)
    dislikes = distribution.get(FeedbackType.DISLIKE.value, 0)
    like_ratio = likes / max(likes + dislikes, 1)
    click_count = distribution.get(FeedbackType.CLICK.value, 0)
    view_count = distribution.get(FeedbackType.VIEW.value, 0)
    interaction_rate = (click_count + likes + dislikes) / max(total, 1)
    engagement_metrics = {
        "click_through_rate": click_count / max(view_count, 1),
        "dwell_time_avg": dwell_time_avg,
        "scroll_depth_avg": scroll_depth_avg,
        "interaction_rate": interaction_rate,
        "completion_rate": view_count / max(total, 1),
        "bounce_rate": 1 - interaction_rate if total else 0.0,
    }
    return ItemFeedbackAnalyticsResponse(
        item_id=item_id,
        total_feedbacks=total,
        average_rating=float(average_rating) if average_rating is not None else None,
        like_ratio=like_ratio,
        engagement_metrics=engagement_metrics,
        feedback_distribution=distribution
    )

async def calculate_quality_scores(feedback_ids: List[str]) -> List[FeedbackQualityScore]:
    """计算反馈质量评分"""
    if not feedback_ids:
        return []

    async with get_db_session() as db:
        await _ensure_feedback_tables(db)

        stmt = text(
            """
            SELECT event_id, user_id, session_id, feedback_type, value, context, timestamp
            FROM feedback_events
            WHERE event_id IN :event_ids
            """
        ).bindparams(bindparam("event_ids", expanding=True))
        rows = await db.execute(stmt, {"event_ids": feedback_ids})
        events = list(rows.mappings().all())

        user_ids = {e["user_id"] for e in events if e.get("user_id")}
        user_stats: Dict[str, Dict[str, int]] = {}
        if user_ids:
            stats_stmt = text(
                """
                SELECT user_id, COUNT(*) AS total, COUNT(DISTINCT feedback_type) AS types
                FROM feedback_events
                WHERE user_id IN :user_ids
                GROUP BY user_id
                """
            ).bindparams(bindparam("user_ids", expanding=True))
            stats_rows = await db.execute(stats_stmt, {"user_ids": list(user_ids)})
            user_stats = {
                r["user_id"]: {"total": int(r["total"]), "types": int(r["types"])}
                for r in stats_rows.mappings().all()
            }

    quality_scores = []
    now = utc_now()
    total_type_count = len(list(FeedbackType)) or 1

    for ev in events:
        feedback_id = ev.get("event_id")
        if not feedback_id:
            continue

        ts: datetime = ev.get("timestamp") or now
        hours = (now - ts).total_seconds() / 3600
        timing = 1.0 if hours <= 24 else max(0.2, 1 - hours / 240)

        context = ev.get("context") or {}
        context_relevance = 1.0 if context.get("url") else 0.6

        try:
            normalized_val = FeedbackNormalizer.normalize_feedback_value(
                FeedbackType(ev["feedback_type"]),
                ev.get("value"),
            )
        except Exception:
            normalized_val = 0.0

        authenticity = 1.0 if ev.get("session_id") else 0.7

        stats = user_stats.get(ev.get("user_id") or "", {})
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
        await _ensure_feedback_tables(db)
        rows = await db.execute(
            text(
                """
                SELECT event_id, user_id, item_id, feedback_type, value, timestamp, context, metadata
                FROM feedback_events
                WHERE batch_id = :batch_id
                ORDER BY timestamp ASC
                """
            ),
            {"batch_id": batch_id},
        )
        events = list(rows.mappings().all())

    if not events:
        logger.info(f"批次无事件: {batch_id}")
        return

    from src.services.feedback_processor import feedback_processor

    payloads = []
    for ev in events:
        payloads.append({
            "feedback_id": ev["event_id"],
            "user_id": ev["user_id"],
            "item_id": ev.get("item_id") or "unknown",
            "feedback_type": ev["feedback_type"],
            "value": ev.get("value"),
            "timestamp": ev.get("timestamp") or utc_now(),
            "context": ev.get("context") or {},
            "metadata": ev.get("metadata") or {},
        })

    await feedback_processor.process_feedback_batch(payloads)

async def compute_reward_signal(user_id: str, item_id: str, time_window: int) -> float:
    """计算奖励信号"""
    cutoff = utc_now() - timedelta(seconds=time_window)
    async with get_db_session() as db:
        await _ensure_feedback_tables(db)
        rows = await db.execute(
            text(
                """
                SELECT feedback_type, value
                FROM feedback_events
                WHERE user_id = :user_id AND item_id = :item_id AND timestamp >= :cutoff
                ORDER BY timestamp DESC
                """
            ),
            {"user_id": user_id, "item_id": item_id, "cutoff": cutoff},
        )
        events = list(rows.mappings().all())
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
        ftype = FeedbackType(ev["feedback_type"])
        weight = weights.get(ftype, 0.1)
        normalized = FeedbackNormalizer.normalize_feedback_value(ftype, ev.get("value"))
        total += weight * normalized
        weight_sum += abs(weight)
    if weight_sum == 0:
        return 0.0
    reward = total / weight_sum
    return max(-1.0, min(1.0, reward))

async def get_system_overview(start_date: Optional[datetime], end_date: Optional[datetime]) -> Dict:
    """获取系统概览"""
    where = []
    params: Dict[str, Any] = {}
    if start_date:
        where.append("timestamp >= :start")
        params["start"] = start_date
    if end_date:
        where.append("timestamp <= :end")
        params["end"] = end_date
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    async with get_db_session() as db:
        await _ensure_feedback_tables(db)

        total_sql = "SELECT COUNT(*) FROM feedback_events " + where_sql
        total_result = await db.execute(text(total_sql), params)
        total = int(total_result.scalar() or 0)

        user_result = await db.execute(
            text("SELECT COUNT(DISTINCT user_id) FROM feedback_events " + where_sql),
            params,
        )
        unique_users = int(user_result.scalar() or 0)

        type_sql = (
            "SELECT feedback_type, COUNT(*) AS count\n"
            "FROM feedback_events\n"
            + where_sql
            + "\nGROUP BY feedback_type"
        )
        type_rows = await db.execute(text(type_sql), params)
        counts = {r["feedback_type"]: int(r["count"]) for r in type_rows.mappings().all()}

        item_where = (
            where_sql + (" AND " if where_sql else "WHERE ") + "item_id IS NOT NULL AND item_id <> ''"
        )
        item_sql = (
            "SELECT item_id, COUNT(*) AS count\n"
            "FROM feedback_events\n"
            + item_where
            + "\nGROUP BY item_id\n"
            "ORDER BY count DESC\n"
            "LIMIT 5"
        )
        item_rows = await db.execute(text(item_sql), params)
        top_items = [
            {"item_id": r["item_id"], "feedback_count": int(r["count"])}
            for r in item_rows.mappings().all()
        ]

    return {
        "total_feedbacks": total,
        "feedback_types": counts,
        "unique_users": unique_users,
        "average_quality_score": None,
        "top_items": top_items
    }

async def get_realtime_metrics() -> Dict:
    """获取实时指标"""
    now = utc_now()
    window_start = now - timedelta(minutes=5)
    async with get_db_session() as db:
        await _ensure_feedback_tables(db)
        recent_result = await db.execute(
            text(
                """
                SELECT session_id
                FROM feedback_events
                WHERE timestamp >= :start AND timestamp <= :end
                """
            ),
            {"start": window_start, "end": now},
        )
        recent_sessions = [r["session_id"] for r in recent_result.mappings().all()]
        active_sessions = len({s for s in recent_sessions if s})
        recent_count = len(recent_sessions)

        total_result = await db.execute(text("SELECT COUNT(*) FROM feedback_events"))
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
